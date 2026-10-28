"""大模型生成模块。

## 它在流程里的位置

    用户问题 → 向量检索（双路召回）→ rerank 精排 → top_k
             → [本模块：把候选知识交给大模型] → 自然语言回答

检索负责「找到相关材料」，本模块负责「用这些材料把话说成人话」。
这是它和「检索器」的分界线：之前 /get_answer 返回的是知识原文堆砌，
现在才真正是「问答助手」。

## 为什么必须约束模型只用给定材料

RAG 最常见的翻车方式是**模型自由发挥**：材料里没有的内容它也会编，
而且编得比原文更自然、更可信。对校内问答这类场景，编错的代价很大。

所以提示词里做了三层约束：
1. 只依据材料回答，材料里没有就说没有
2. 不要编造通知、电话、时间等具体细节
3. 允许它组织语言、合并多条材料，但不得引入材料之外的事实

## 关于深度思考

deepseek-v4.1-flash 默认开启思考模式（reasoning），会先输出一大段思维链。
RAG 问答的材料和问题都摆在明面上，不需要多步推理，实测：

    默认思考      约 2.2s，产生 63 个 reasoning token
    enable_thinking=false  约 1.6s，0 个 reasoning token

所以默认关闭（LLM_DISABLE_THINKING=true）。思维链按输出 token 计费，
关掉既快又省钱。
"""
import logging

import requests

from .config import Settings

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60

# 材料为空或与问题无关时的统一回复，避免模型硬编一个答案
NO_CONTEXT_ANSWER = "抱歉，知识库中没有找到与这个问题相关的内容。"

SYSTEM_PROMPT = """你是一个校内知识问答助手，负责根据提供的参考材料回答用户的问题。

请严格遵守以下规则：
1. 只依据【参考材料】中的内容回答，不要使用你自身的知识补充材料之外的事实。
2. 如果参考材料不足以回答问题，直接说明「根据现有资料无法回答这个问题」，不要猜测。
3. 不要编造材料中没有的具体信息，例如电话、地址、时间、金额、人名。
4. 可以把多条材料组织整合成通顺的回答，但不得改变原意或添加材料没有的结论。
5. 用简洁自然的中文回答，直接说结论，不要说「根据参考材料」这类套话。
6. 回答长度控制在 200 字以内，信息多时用分点列出。"""

# 查询改写提示词。
#
# 这一步是让检索能工作的前提：检索层拿到的必须是一个**自包含**的问题，
# 而追问里的「它」「那」「还有」离开上文就没有意义。
#
# 关键约束：
#   1. 只改表述、补全省略，不得回答问题本身，也不得引入上文没有的新信息
#   2. 已经是完整问题时原样返回（否则模型会把「学校在哪」改成别的东西，
#      白白增加一次改写出错的机会）
#   3. 让模型只输出改写后的那一句，不要解释——否则会把解释文本当成查询词
REWRITE_PROMPT = """你是一个查询改写助手。你的任务是结合对话历史，把用户当前的问题改写成一句能独立检索的完整问题。

规则：
1. 如果当前问题已经完整、不依赖上文，**原样返回**原问题，不要做任何修改。
2. 如果当前问题含指代（如「它」「这个」「那」）或省略了主语，结合历史补全。
   例：历史在聊「宿舍条件」，当前问「它有空调吗」-> 改写为「宿舍有空调吗」
3. 只改写表述，不要回答问题，不要引入历史中没有的新信息。
4. 只输出改写后的那一句问题本身，不要加引号、不要解释、不要前缀。"""


def build_history_block(history: list) -> str:
    """把历史对话拼成提示词里的上文块。"""
    if not history:
        return ""
    lines = []
    for i, h in enumerate(history, start=1):
        q = (h.get("question") or "").strip()
        a = (h.get("answer") or "").strip()
        lines.append(f"[第{i}轮]\n用户：{q}\n助手：{a}")
    return "\n\n".join(lines)


class LLM:
    """大模型接口抽象。换模型/换供应商只需实现 generate 与 rewrite_query。"""

    def generate(self, question: str, contexts: list, history: list = None) -> str:
        """根据问题与候选材料生成回答。

        参数：
            question : 用户原始问题
            contexts : 候选材料，每项至少含 question / answer 字段
            history  : 最近几轮对话 [{question, answer}, ...]，可为空
        返回：
            自然语言回答；材料为空时返回 NO_CONTEXT_ANSWER。
        """
        raise NotImplementedError

    def rewrite_query(self, question: str, history: list) -> str:
        """把带指代的追问改写成能独立检索的问题。

        例：
            历史「宿舍条件怎么样」+ 当前「它有空调吗」
            -> 「宿舍有空调吗」

        没有历史时直接返回原问题，不浪费一次调用。
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, settings: Settings = None) -> "LLM":
        """按配置装配。未启用时返回 NoopLLM，调用方无需判空。"""
        cfg = settings or Settings.from_env()
        if not cfg.llm_enabled:
            return NoopLLM()
        if not cfg.api_key:
            log.warning("LLM_ENABLED=true 但没有 API Key，回答生成将退化为不过滤")
            return NoopLLM()
        return ChatLLM(cfg)


class NoopLLM(LLM):
    """空实现：不调用模型，把材料原文拼成回答返回。

    默认用它有两个好处：
    1. 没配 Key / 没开开关时整条链路照样跑通，前端不会白屏
    2. 上线生成前后只需改一个配置，代码结构不动
    """

    def generate(self, question: str, contexts: list, history: list = None) -> str:
        if not contexts:
            return NO_CONTEXT_ANSWER
        return "\n".join(f"· {c.get('answer', '')}" for c in contexts)

    def rewrite_query(self, question: str, history: list) -> str:
        # 不改写：没开模型时多轮指代无法消解，这里明确返回原问题，
        # 由上层体现为「检索不到」，而不是假装解决了
        return question


def build_context_block(contexts: list) -> str:
    """把候选材料拼成提示词里的材料块，带编号便于模型引用。"""
    parts = []
    for i, c in enumerate(contexts, start=1):
        q = (c.get("question") or "").strip()
        a = (c.get("answer") or "").strip()
        parts.append(f"[材料{i}]\n问题：{q}\n内容：{a}")
    return "\n\n".join(parts)


class ChatLLM(LLM):
    """OpenAI 兼容的 chat/completions 实现（本项目用 deepseek-v4.1-flash）。

    请求体要点：
      - `enable_thinking=false` 关闭深度思考（非 OpenAI 标准参数，
        但本项目直连 HTTP 接口，放在顶层即可）
      - temperature 调低（默认 0.3）：RAG 需要稳定复述材料，
        而不是发挥想象力
    """

    def __init__(self, settings: Settings, timeout: int = None):
        self.settings = settings
        self.timeout = timeout or settings.llm_timeout

    def _chat(self, messages: list, *, max_tokens: int = None,
              temperature: float = None) -> str:
        """发一次 chat/completions，返回助手回复文本。

        生成和查询改写都走这里，避免重复维护请求/解析逻辑。
        """
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": (
                self.settings.llm_temperature if temperature is None else temperature
            ),
        }
        if self.settings.llm_disable_thinking:
            payload["enable_thinking"] = False

        try:
            resp = requests.post(
                self.settings.llm_url,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"大模型接口请求失败: {e}") from e

        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise RuntimeError(f"大模型接口返回格式异常: {data!r}") from e

        content = content.strip()
        if not content:
            # 思考模式下偶尔会把内容全放进 reasoning_content，content 为空
            raise RuntimeError("大模型返回了空回答（可能是思考模式未关闭）")
        return content

    # ------------------------------------------------------------------
    def generate(self, question: str, contexts: list, history: list = None) -> str:
        if not contexts:
            # 没材料就别浪费一次调用，也避免模型凭空作答
            return NO_CONTEXT_ANSWER

        # 多轮时把上文作为独立消息传进去，而不是拼进同一个字符串：
        # 这样模型能分清哪些是「已经聊过的」、哪些是「这一轮要回答的」。
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in (history or []):
            q = (h.get("question") or "").strip()
            a = (h.get("answer") or "").strip()
            if q:
                messages.append({"role": "user", "content": q})
            if a:
                messages.append({"role": "assistant", "content": a})

        messages.append({
            "role": "user",
            "content": (
                f"【参考材料】\n{build_context_block(contexts)}\n\n"
                f"【用户问题】\n{question}"
            ),
        })
        return self._chat(messages)

    def rewrite_query(self, question: str, history: list) -> str:
        if not history:
            # 第一轮没有上文可结合，省一次调用
            return question

        messages = [
            {"role": "system", "content": REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【对话历史】\n{build_history_block(history)}\n\n"
                    f"【当前问题】\n{question}\n\n"
                    f"请输出改写后的那一句问题。"
                ),
            },
        ]
        # 改写只要一句话，token 给小、温度给 0 保证稳定
        try:
            text = self._chat(messages, max_tokens=120, temperature=0.0)
        except RuntimeError as e:
            # 改写失败不该让整个问答失败，退回原问题（检索可能失败，但流程不断）
            log.warning("查询改写失败，使用原问题：%s", e)
            return question

        # 模型偶尔会加引号或「改写后：」这类前缀，这里清理掉
        text = text.strip().strip('"').strip("'").strip()
        for prefix in ("改写后：", "改写后:", "改写为：", "改写为:", "问题："):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        text = text.splitlines()[0].strip() if text else ""

        # 模型有时会把整段回答倒出来，长度异常就放弃改写结果
        if not text or len(text) > 120:
            log.warning("查询改写结果异常，使用原问题：%r", text[:80])
            return question
        return text


def build_llm(settings: Settings = None) -> LLM:
    """对外工厂函数，等价于 LLM.from_config()。"""
    return LLM.from_config(settings)
