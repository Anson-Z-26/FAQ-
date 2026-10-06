"""重排序（rerank）模块。

## 为什么需要 rerank

向量检索是「双塔」结构：问题和文档各自独立编码成向量，再算余弦相似度。
两边编码时互相看不见，会丢掉细粒度的交互信息——这就是为什么会出现
「正确答案明明在候选里，却排不到第一」。

rerank 是 cross-encoder：把 query 和每篇候选文档**拼在一起**送进模型打分，
能捕捉交互信息，因此排序比向量点积准得多。标准做法是：

    向量检索粗召回 N 条  →  rerank 精排  →  取 top_k  →  交给 LLM 生成

## 本模块在流程中的位置

    用户问题 → 向量 → 查问题索引 + 查答案索引 → 合并去重
             → [rerank 精排] → top_k → LLM 生成答案

## 接口约定

Reranker 只吃「文本」，不吃向量。这和「问题向量」无关——两个索引的向量
只用于各自的召回路径；进入 rerank 的是用户原始问题文本 + 候选文档文本。

    rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]

换任何 rerank 服务（qwen3-rerank / bge-rerank / 自建）都实现这个签名即可，
RAGService 不用改。

## 当前状态

- NoopReranker：默认实现，原样返回，不改变顺序。
- DashScopeReranker：已按官方接口写好，但默认不启用（RERANK_ENABLED=false）。
  启用前请阅读该类 docstring 里的注意事项。
"""
import logging

import requests

from .config import Settings

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class Reranker:
    """rerank 接口的抽象基类。"""

    def rerank(self, query: str, documents: list, top_n: int) -> list:
        """对候选文档重新排序。

        参数：
            query     : 用户原始问题文本
            documents : 候选文档文本列表
            top_n     : 返回前 N 条

        返回：
            [(原始下标, 相关度分数), ...]，按相关度降序。
            下标指向入参 documents 的位置，便于调用方映射回知识条目。
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, settings: Settings = None) -> "Reranker":
        """按配置装配。未启用时返回 NoopReranker，调用方无需判断 None。"""
        cfg = settings or Settings.from_env()
        if not cfg.rerank_enabled:
            return NoopReranker()
        if not cfg.api_key:
            log.warning("RERANK_ENABLED=true 但没有 API Key，rerank 退化为不过滤")
            return NoopReranker()
        return DashScopeReranker(cfg)


class NoopReranker(Reranker):
    """空实现：保持原顺序，只做截断。

    默认用它，好处是整条链路在没有 rerank 时也能完整跑通，
    接入 rerank 前后只需改一个配置，不用改代码结构。
    """

    def rerank(self, query: str, documents: list, top_n: int) -> list:
        if top_n is None or top_n <= 0 or top_n >= len(documents):
            return [(i, 0.0) for i in range(len(documents))]
        return [(i, 0.0) for i in range(top_n)]


class DashScopeReranker(Reranker):
    """千问AI平台 qwen3-rerank 实现（OpenAI 兼容接口）。

    接口：POST https://maas.qianwenaiapi.com/compatible-api/v1/reranks

    请求体是**扁平结构**（query / documents / top_n 都在顶层），
    与 embedding 接口不同，注意别混淆。

    官方限制：
        最大文档数 500 / 单文档最大 4000 token / 请求最大 120000 token
        （请求上限算法：query_token * 文档数 + 所有文档 token 总数）

    注意事项（启用前请确认）：
        1. 本实现尚未在真实接口上验证过。首次启用请先用小批量试通。
        2. 返回的 relevance_score 是「本次请求内的相对分数」，官方明确说明
           不可跨请求比较。所以不要拿它和 score_threshold 直接比大小。
        3. 文档数超过 500 时接口会拒绝，这里只截断并告警，不抛异常。
    """

    MAX_DOCUMENTS = 500

    def __init__(self, settings: Settings, timeout: int = DEFAULT_TIMEOUT):
        self.settings = settings
        self.timeout = timeout

    def rerank(self, query: str, documents: list, top_n: int) -> list:
        if not documents:
            return []

        docs = list(documents)
        if len(docs) > self.MAX_DOCUMENTS:
            log.warning("候选 %d 条超过接口上限 %d，只取前 %d 条",
                        len(docs), self.MAX_DOCUMENTS, self.MAX_DOCUMENTS)
            docs = docs[: self.MAX_DOCUMENTS]

        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": docs,
            # 官方默认行为即问答检索，显式写出便于日后按场景调整
            "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
        }
        if top_n and top_n > 0:
            payload["top_n"] = int(top_n)

        try:
            resp = requests.post(
                self.settings.rerank_url,
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
            raise RuntimeError(f"Rerank接口请求失败: {e}") from e

        try:
            results = data["results"]
            return [(int(r["index"]), float(r["relevance_score"])) for r in results]
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"Rerank接口返回格式异常: {data!r}") from e


def build_reranker(settings: Settings = None) -> Reranker:
    """对外工厂函数，等价于 Reranker.from_config()。"""
    return Reranker.from_config(settings)
