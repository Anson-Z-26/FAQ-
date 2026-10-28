"""检索流程封装。

## 为什么问题和答案要分开建索引

旧做法是把 `question + answer` 拼成一条文本去向量化，只存一个向量。这有两个问题：

1. **语义稀释**：答案文字会冲淡问题向量的语义。本知识库的条目全都以
   「西安电子科技大学」开头，区分度本来就低，再混入答案就更挤在一起。
2. **答案向量被浪费**：知识和用户问法经常对不上（用户问「宿舍条件」，
   知识库里只有「后勤保障」这种问法），但**答案正文**里可能写着住宿信息。
   把答案单独建索引，这条就能被捞回来。

所以改成双索引双路召回：

    建索引：doc['question'] → question.index
            doc['answer']   → answer.index

    检索：  用户问题 → 向量
              ├─ 查问题索引 → 候选 A（问法匹配）
              └─ 查答案索引 → 候选 B（内容匹配）
            合并去重 → [rerank 精排] → top_k → 交给 LLM 生成

两路的相似度分布不同（答案与问句的字面重合度天然更低），所以**各自有独立阈值**。

## 关于 rerank 的一个常见误解

rerank 吃的是**文本**，不是向量。两个索引的向量只用于各自的召回路径；
进入 rerank 的是「用户原始问题文本 + 候选知识条目的文本」。
"""
import json
import logging
from pathlib import Path

from .config import Settings
from .embedding import get_embedding
from .llm import LLM, NoopLLM
from .rerank import NoopReranker, Reranker
from .vector_index import VectorIndex

log = logging.getLogger(__name__)


def load_knowledge(path) -> list:
    """读取知识库 json，并校验每条记录都带 question/answer。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"知识库 {path} 顶层需要是数组")
    for i, doc in enumerate(data):
        if not isinstance(doc, dict) or "question" not in doc or "answer" not in doc:
            raise RuntimeError(f"知识库 {path} 第 {i} 条缺少 question/answer 字段: {doc!r}")
    return data


class RAGService:
    """面向调用方的检索服务。

    持有双索引（问题/答案）+ 知识库 + reranker，对外暴露 candidates/search/retrieve。
    """

    def __init__(self, knowledge_list, question_index: VectorIndex,
                 answer_index: VectorIndex = None, *, settings: Settings = None,
                 embed_fn=get_embedding, reranker: Reranker = None, llm: LLM = None,
                 top_k: int = None, score_threshold: float = None,
                 answer_score_threshold: float = None, recall_k: int = None,
                 dedupe_keep: str = None, rerank_score_threshold: float = None):
        """
        embed_fn 契约：embed_fn(text, *, settings=...) -> np.ndarray，与 get_embedding 一致。
        answer_index 为 None 时退化为「只用问题索引」。
        reranker 为 None 时退化为「不过滤」，行为等价于 NoopReranker。
        llm 为 None 时退化为「不外调模型」，由 NoopLLM 拼装材料原文。
        dedupe_keep 见 dedupe_by_answer()：best（默认）或 longest。
        """
        self.settings = settings or Settings.from_env()
        self.knowledge_list = knowledge_list
        self.question_index = question_index
        self.answer_index = answer_index
        self.embed_fn = embed_fn
        self.reranker = reranker
        self.llm = llm
        self.top_k = self.settings.top_k if top_k is None else top_k
        self.dedupe_keep = dedupe_keep or getattr(self.settings, "dedupe_keep", "best")
        self.rerank_score_threshold = (
            self.settings.rerank_score_threshold
            if rerank_score_threshold is None else rerank_score_threshold
        )
        self.score_threshold = (
            self.settings.score_threshold if score_threshold is None else score_threshold
        )
        self.answer_score_threshold = (
            self.settings.answer_score_threshold
            if answer_score_threshold is None else answer_score_threshold
        )
        self.recall_k = self.settings.recall_k if recall_k is None else recall_k

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, settings: Settings = None, *, embed_fn=get_embedding,
                    reranker: Reranker = None, llm: LLM = None) -> "RAGService":
        """按 .env 配置装配好知识库、双索引、rerank 与生成模型。整个进程调用一次即可。"""
        cfg = settings or Settings.from_env()
        q_index = VectorIndex.load(cfg.question_index_path, dim=cfg.embedding_dim)

        a_index = None
        if cfg.answer_index_enabled:
            a_index = VectorIndex.load(cfg.answer_index_path, dim=cfg.embedding_dim)

        # 索引为空是「忘了建索引」的典型症状：它不抛异常，只让检索永远返回空。
        # 这里明确告警，避免这种安静失败。
        if len(q_index) == 0:
            log.warning("问题索引为空或不存在（%s），检索将永远返回空结果。"
                        "请先运行 build_index.py", cfg.question_index_path)
        if a_index is not None and len(a_index) == 0:
            log.warning("答案索引为空或不存在（%s），双路召回退化为单路。"
                        "请先运行 build_index.py", cfg.answer_index_path)
        if cfg.llm_enabled:
            log.info("回答生成已启用：%s", cfg.llm_model)
        else:
            log.info("回答生成未启用（LLM_ENABLED=false），将直接返回知识原文")

        return cls(
            knowledge_list=load_knowledge(cfg.knowledge_path),
            question_index=q_index,
            answer_index=a_index,
            settings=cfg,
            embed_fn=embed_fn,
            reranker=reranker if reranker is not None else Reranker.from_config(cfg),
            llm=llm if llm is not None else LLM.from_config(cfg),
        )

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------
    def _recall_path(self, index, query_vector, *, top_k: int, threshold: float, source: str):
        """单索引召回。source 标注候选来自哪一路，便于排查。"""
        if index is None or len(index) == 0:
            return []
        scores, ids = index.search(query_vector, top_k=top_k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            # idx < 0 是 FAISS 在候选不足时补的占位
            # 写成 not (score >= ...) 而不是 score < ...，这样 NaN 也会被过滤掉
            if idx < 0 or not score >= threshold:
                continue
            out.append({"doc_idx": int(idx), "score": float(score), "source": source})
        return out

    def candidates(self, query: str) -> list:
        """双路召回 + 合并去重，返回候选列表（**尚未**经过 rerank）。

        每条候选：
            {
              "doc_idx": int,        # 在 knowledge_list 里的下标
              "question": str,
              "answer": str,
              "score": float,        # 两路中的最高分
              "source": str,         # question / answer / question+answer
              "path_scores": {...},  # 各路原始分数，便于排查
              "rerank_score": float, # 精排分数（NoopReranker 时为 0.0）
            }
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        query_vector = self.embed_fn(query, settings=self.settings)
        recall_k = max(1, int(self.recall_k))

        merged = {}

        def merge(hits):
            for h in hits:
                idx = h["doc_idx"]
                if idx in merged:
                    prev = merged[idx]
                    sources = set(prev["source"].split("+")) | {h["source"]}
                    prev["source"] = "+".join(sorted(sources))
                    prev["path_scores"][h["source"]] = h["score"]
                    prev["score"] = max(prev["score"], h["score"])
                else:
                    doc = self.knowledge_list[idx]
                    merged[idx] = {
                        "doc_idx": idx,
                        "question": doc["question"],
                        "answer": doc["answer"],
                        "score": h["score"],
                        "source": h["source"],
                        "path_scores": {h["source"]: h["score"]},
                        "rerank_score": 0.0,
                    }

        # 第 1 路：问题索引（问法匹配）
        merge(self._recall_path(self.question_index, query_vector, top_k=recall_k,
                                threshold=self.score_threshold, source="question"))

        # 第 2 路：答案索引（内容匹配，阈值更低）
        if self.answer_index is not None:
            merge(self._recall_path(self.answer_index, query_vector, top_k=recall_k,
                                    threshold=self.answer_score_threshold, source="answer"))

        # 合并后按分数降序；同分时两路都命中的优先（可信度更高）
        return sorted(
            merged.values(),
            key=lambda c: (c["score"], len(c["path_scores"])),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------
    @staticmethod
    def dedupe_by_answer(candidates: list, keep: str = "best") -> list:
        """把「同一条知识的多个问法变体」合并成一条。

        为什么需要：给知识补问法变体时必须拆成多行（每个变体一个尖锐的向量），
        但多条变体常常同时命中，会一起挤占 top_k，并把同一段答案重复拼进
        上下文浪费 token、干扰大模型。去重后每个答案只占一个位子。

        保留策略：
            keep="best"   —— 保留向量分数最高的那个变体问法
            keep="longest"—— 从命中变体里拼一个信息最全的问法（用 " / " 连接），
                             交给 rerank 时能看到更多问法，精排更准

        注意：只合并**同一个答案**。答案相同但确实是两条不同知识的情况不存在，
        因为合并的前提就是它们要给用户同一个答案。
        """
        if not candidates:
            return []

        by_answer = {}
        order = []
        for c in candidates:
            key = (c.get("answer") or "").strip()
            if key not in by_answer:
                by_answer[key] = []
                order.append(key)
            by_answer[key].append(c)

        merged = []
        for key in order:
            group = by_answer[key]
            if len(group) == 1:
                c = dict(group[0])
                c["variant_count"] = 1
                c["variants"] = [c["question"]]
                merged.append(c)
                continue

            if keep == "longest":
                # 按分数降序，把靠前的变体问法拼起来
                group_sorted = sorted(group, key=lambda x: x["score"], reverse=True)
                question = " / ".join(
                    dict.fromkeys(c["question"] for c in group_sorted)
                )
            else:
                group_sorted = sorted(group, key=lambda x: x["score"], reverse=True)
                question = group_sorted[0]["question"]

            best = dict(group_sorted[0])
            best["question"] = question
            best["variant_count"] = len(group)
            best["variants"] = [c["question"] for c in group_sorted]
            # 各路分数取「最大」而不是后写覆盖：多个变体可能来自同一路，
            # 用 update() 会让低分覆盖高分，排查时会看到误导性的数值
            path_scores = {}
            for c in group_sorted:
                for path, val in c.get("path_scores", {}).items():
                    path_scores[path] = max(path_scores.get(path, float("-inf")), val)
            best["path_scores"] = path_scores
            # score 同样取最大值，保证去重后不会比原来更低
            best["score"] = max(c["score"] for c in group_sorted)
            sources = set()
            for c in group_sorted:
                sources |= set(str(c.get("source", "")).split("+"))
            best["source"] = "+".join(sorted(s for s in sources if s))
            merged.append(best)

        return merged

    # ------------------------------------------------------------------
    # 精排
    # ------------------------------------------------------------------
    def search(self, query: str) -> list:
        """双路召回 → 按答案去重 → rerank 精排 → 返回 top_k 条。

        **顺序很重要**：先去重（让每个答案只占一个位子）、再 rerank 排序，
        取出的 top_k 才交给 LLM 生成答案。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        cands = self.candidates(query)
        if not cands:
            return []

        # 去重必须在 rerank 之前：否则同一条知识的多个变体会先占满位子，
        # rerank 即使想挑别的也挑不到。
        raw_n = len(cands)
        cands = self.dedupe_by_answer(cands, keep=self.dedupe_keep)
        if len(cands) < raw_n:
            log.debug("按答案去重：%d 条候选 -> %d 条不同答案", raw_n, len(cands))

        top_k = max(1, int(self.top_k))

        # rerank 吃文本：query 用用户原始问题，documents 用候选条目文本。
        # 问题和答案都给 reranker 看，信息量最大。
        documents = [f"{c['question']}\n{c['answer']}" for c in cands]

        reranker = self.reranker or Reranker.from_config(self.settings)

        # rerank 分数与向量相似度不同源，只在「确实拿到了 rerank 分数」时才用它做门槛。
        # NoopReranker 和降级路径返回的都是占位的 0.0，如果拿它过滤会把结果全滤光。
        use_rerank_gate = not isinstance(reranker, NoopReranker)
        try:
            # 关键：把**全部候选**交给 rerank，而不是只给 top_k 条。
            # 只给 top_k 条的话，模型只能在已选中的这几条里换顺序，
            # 没有机会说「这些都不相关」，也就无法纠正误报。
            # 排序后再用 rerank_score_threshold 卡门槛，最后才截 top_k。
            ranked = reranker.rerank(query, documents, top_n=len(cands))
        except RuntimeError as e:
            # rerank 是增强环节，挂了不该让整个检索失败：退回向量排序并告警。
            # 注意此时必须关掉门槛，否则 0.0 的占位分数会让结果全部被滤掉。
            log.warning("rerank 失败，退回向量检索排序（本次不做相关性门槛）：%s", e)
            ranked = [(i, 0.0) for i in range(len(cands))]
            use_rerank_gate = False

        hits = []
        for pos, rscore in ranked:
            if len(hits) >= top_k:
                break
            if not (0 <= pos < len(cands)):
                continue
            if use_rerank_gate and not rscore >= self.rerank_score_threshold:
                continue
            c = cands[pos]
            hits.append({
                "question": c["question"],
                "answer": c["answer"],
                "score": c["score"],            # 向量相似度，跨请求可比
                "rerank_score": float(rscore),  # 相对分数，仅本次请求内有意义
                "source": c["source"],
                # 这条答案由几个问法变体命中（去重前），便于观察变体补充的效果
                "variant_count": c.get("variant_count", 1),
                "variants": c.get("variants", [c["question"]]),
            })
        return hits

    def retrieve(self, query: str) -> str:
        """检索 → 去重 → rerank → 拼装成可直接喂给大模型的上下文文本。

        去重后每个答案只出现一次，不会再因为问法变体重复而浪费上下文 token。
        """
        return "\n\n".join(
            f"【参考知识 {i}】\n问题：{hit['question']}\n回答：{hit['answer']}"
            for i, hit in enumerate(self.search(query), start=1)
        )

    def answer(self, query: str) -> dict:
        """完整问答：检索 → 精排 → 交给大模型生成自然语言回答。

        这是对外的主入口。返回结构里同时给出：
          - answer  : 大模型生成的回答（未启用生成时是材料原文拼装）
          - hits    : 命中的知识条目，供前端展示「依据」
          - matched : 是否检索到相关内容，前端据此区分「没找到」和「回答为空」

        为什么把 hits 一起返回：用户需要有办法核对回答的依据。
        只给一段生成文本，出了问题无法追溯是检索错了还是模型编了。
        """
        hits = self.search(query)

        llm = self.llm or LLM.from_config(self.settings)
        try:
            text = llm.generate(query, hits)
        except RuntimeError as e:
            # 生成失败不该让整个问答失败：退回材料原文，并记下原因
            log.warning("大模型生成失败，退回材料原文：%s", e)
            text = "\n".join(f"· {h['answer']}" for h in hits) if hits else ""
            return {
                "answer": text,
                "hits": hits,
                "matched": bool(hits),
                "generated": False,
                "generation_error": str(e),
            }

        return {
            "answer": text,
            "hits": hits,
            "matched": bool(hits),
            "generated": not isinstance(llm, NoopLLM),
            "generation_error": "",
        }
