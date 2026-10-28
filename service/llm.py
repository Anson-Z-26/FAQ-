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


class LLM:
    """大模型接口抽象。换模型/换供应商只需实现 generate。"""

    def generate(self, question: str, contexts: list) -> str:
        """根据问题与候选材料生成回答。

        参数：
            question : 用户原始问题
            contexts : 候选材料，每项至少含 question / answer 字段
        返回：
            自然语言回答；材料为空时返回 NO_CONTEXT_ANSWER。
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

    def generate(self, question: str, contexts: list) -> str:
        if not contexts:
            return NO_CONTEXT_ANSWER
        return "\n".join(f"· {c.get('answer', '')}" for c in contexts)


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

    def generate(self, question: str, contexts: list) -> str:
        if not contexts:
            # 没材料就别浪费一次调用，也避免模型凭空作答
            return NO_CONTEXT_ANSWER

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【参考材料】\n{build_context_block(contexts)}\n\n"
                    f"【用户问题】\n{question}"
                ),
            },
        ]

        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": self.settings.llm_max_tokens,
            "temperature": self.settings.llm_temperature,
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


def build_llm(settings: Settings = None) -> LLM:
    """对外工厂函数，等价于 LLM.from_config()。"""
    return LLM.from_config(settings)
