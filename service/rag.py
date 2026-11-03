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
from .manifest import IndexManifest
from .rerank import NoopReranker, Reranker
from .vector_index import VectorIndex

log = logging.getLogger(__name__)


def load_knowledge(path) -> list:
    """读取知识库，支持两种格式，并校验必需字段。

    格式自动识别：
      - 以 `[` 开头 → JSON 数组（旧格式 data.json）
      - 否则        → JSONL，每行一个对象（新格式，便于逐行 diff 与追加）

    必需字段：question、answer。
    可选字段：id（强烈建议）、aliases（问法变体列表）、category/source/url/updated。

    关于 id：有问法变体后，一个问题索引会对应**多行**向量，
    行号不再等于知识条目下标，必须靠 id 建立映射（见 service/manifest.py）。
    所以这里对缺失或重复的 id 直接报错，而不是悄悄用下标兜底 ——
    用下标兜底会得到「看起来正常但答案是错的」结果，比启动失败更糟。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在: {path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"知识库为空: {path}")

    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise RuntimeError(f"知识库 {path} 顶层需要是数组")
    else:
        data = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"知识库 {path} 第 {lineno} 行不是合法 JSON: {e}") from e

    # ---- 字段校验 ----
    seen_ids = {}
    for i, doc in enumerate(data):
        if not isinstance(doc, dict):
            raise RuntimeError(f"知识库 {path} 第 {i} 条不是对象: {doc!r}")
        for field in ("question", "answer"):
            if field not in doc:
                raise RuntimeError(
                    f"知识库 {path} 第 {i} 条缺少 {field} 字段: {doc!r}"
                )
        doc_id = doc.get("id")
        if not doc_id:
            raise RuntimeError(
                f"知识库 {path} 第 {i} 条缺少 id 字段。"
                f"有问法变体后行号与条目会错位，必须用 id 建立映射"
            )
        if doc_id in seen_ids:
            raise RuntimeError(
                f"知识库 {path} 中 id {doc_id!r} 重复"
                f"（第 {seen_ids[doc_id]} 条和第 {i} 条）"
            )
        seen_ids[doc_id] = i

        aliases = doc.get("aliases")
        if aliases is not None and not isinstance(aliases, list):
            raise RuntimeError(
                f"知识库 {path} 第 {i} 条 aliases 需要是数组: {aliases!r}"
            )

    return data


def knowledge_aliases(doc: dict) -> list:
    """取一条知识的问法变体，去掉空值、去重、并排除与 question 完全相同的。"""
    question = (doc.get("question") or "").strip()
    out, seen = [], {question}
    for a in doc.get("aliases") or []:
        a = str(a).strip()
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def index_by_id(knowledge_list: list) -> dict:
    """建立 id -> 知识条目 的映射。"""
    return {d["id"]: d for d in knowledge_list if d.get("id")}


class RAGService:
    """面向调用方的检索服务。

    持有双索引（问题/答案）+ 知识库 + reranker，对外暴露 candidates/search/retrieve。
    """

    def __init__(self, knowledge_list, question_index: VectorIndex,
                 answer_index: VectorIndex = None, *, settings: Settings = None,
                 embed_fn=get_embedding, reranker: Reranker = None, llm: LLM = None,
                 top_k: int = None, score_threshold: float = None,
                 answer_score_threshold: float = None, recall_k: int = None,
                 rerank_score_threshold: float = None,
                 question_row_ids: list = None, answer_row_ids: list = None,
                 recall_overfetch: int = None):
        """
        embed_fn 契约：embed_fn(text, *, settings=...) -> np.ndarray，与 get_embedding 一致。
        answer_index 为 None 时退化为「只用问题索引」。
        reranker 为 None 时退化为「不过滤」，行为等价于 NoopReranker。
        llm 为 None 时退化为「不外调模型」，由 NoopLLM 拼装材料原文。

        question_row_ids / answer_row_ids：
            「FAISS 第 i 行 → 知识 id」的映射，长度必须等于对应索引的行数。
            这是有问法变体后**必须**提供的：问题索引行数（条目+变体）
            远大于知识条数，靠行号当条目下标会静默错位。
            为空时退化为「行号即条目下标」，仅用于无变体的旧数据与单元测试。
        """
        self.settings = settings or Settings.from_env()
        self.knowledge_list = knowledge_list
        self.knowledge_by_id = index_by_id(knowledge_list)
        self.question_index = question_index
        self.answer_index = answer_index
        self.embed_fn = embed_fn
        self.reranker = reranker
        self.llm = llm
        self.top_k = self.settings.top_k if top_k is None else top_k
        self.recall_overfetch = (
            getattr(self.settings, "recall_overfetch", 3)
            if recall_overfetch is None else recall_overfetch
        )
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

        # 行号 → 知识 id 的映射。未提供时退化为恒等映射（行 i == 条目 i）
        self.question_row_ids = (
            list(question_row_ids) if question_row_ids is not None
            else [d.get("id") for d in knowledge_list]
        )
        self.answer_row_ids = (
            list(answer_row_ids) if answer_row_ids is not None
            else [d.get("id") for d in knowledge_list]
        )
        self._check_row_map("问题", self.question_row_ids, question_index)
        if answer_index is not None:
            self._check_row_map("答案", self.answer_row_ids, answer_index)

    @staticmethod
    def _check_row_map(label: str, row_ids: list, index: VectorIndex) -> None:
        """行映射长度必须与索引行数一致，否则一定会错位，直接报错。"""
        if index is None:
            return
        if len(row_ids) != len(index):
            raise RuntimeError(
                f"{label}索引的「行号→知识id」映射长度为 {len(row_ids)}，"
                f"但索引实际有 {len(index)} 行。"
                f"两者不一致会导致返回错误的答案，请重建索引（build_index.py）"
            )

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

        knowledge_list = load_knowledge(cfg.knowledge_path)
        q_row_ids, a_row_ids = cls._load_and_validate_index(
            cfg, knowledge_list, q_index, a_index
        )

        return cls(
            knowledge_list=knowledge_list,
            question_index=q_index,
            answer_index=a_index,
            settings=cfg,
            embed_fn=embed_fn,
            reranker=reranker if reranker is not None else Reranker.from_config(cfg),
            llm=llm if llm is not None else LLM.from_config(cfg),
            question_row_ids=q_row_ids,
            answer_row_ids=a_row_ids,
        )

    @staticmethod
    def _load_and_validate_index(cfg, knowledge_list, q_index, a_index):
        """读取索引清单并校验一致性，返回 (问题行映射, 答案行映射)。

        **为什么必须校验**：有问法变体后，问题索引行数（条目 + 变体）远大于知识
        条数。行号与条目错位的后果不是报错，而是**静默返回错误答案** ——
        所以这里宁可起不来，也不带着错位的索引对外服务。
        """
        manifest_path = cfg.manifest_path
        strict = getattr(cfg, "strict_index_check", True)

        # 索引为空（还没建过）时不需要清单，交给上层告警
        if len(q_index) == 0:
            log.warning("问题索引为空，跳过清单校验；请先运行 build_index.py")
            return None, None

        if not manifest_path.exists():
            msg = (
                f"找不到索引清单 {manifest_path}，但索引文件存在。"
                f"没有清单就无法确认「索引行号 → 知识条目」的对应关系"
                f"（有问法变体时行号不等于条目下标），"
                f"可能返回错误答案。请运行 build_index.py 重建索引"
            )
            if strict:
                raise RuntimeError(msg)
            log.error(msg + "（STRICT_INDEX_CHECK=false，继续启动但结果不可信）")
            return None, None

        manifest = IndexManifest.load(manifest_path)
        problems = manifest.validate(
            embedding_model=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
            knowledge_list=knowledge_list,
            question_ntotal=len(q_index),
            answer_ntotal=len(a_index) if a_index is not None else None,
        )
        if problems:
            detail = "\n  - " + "\n  - ".join(problems)
            msg = f"索引与当前配置/知识库不一致：{detail}\n请运行 build_index.py 重建索引"
            if strict:
                raise RuntimeError(msg)
            log.error(msg + "\n（STRICT_INDEX_CHECK=false，继续启动但检索结果不可信）")
            return None, None

        log.info(
            "索引校验通过：模型 %s，知识 %d 条，问题索引 %d 行，答案索引 %s 行",
            manifest.embedding_model, manifest.knowledge_count,
            len(manifest.question_rows), len(manifest.answer_rows),
        )
        return manifest.question_rows, manifest.answer_rows

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------
    def _recall_path(self, index, query_vector, *, top_k: int, threshold: float,
                     source: str, row_ids: list):
        """单索引召回。返回的是**知识 id** 而不是行号。

        有问法变体后，一行向量不再对应一条知识，所以必须在这里就把行号
        翻译成 id，后续所有逻辑都按 id 处理，避免行号泄漏到业务层。
        """
        if index is None or len(index) == 0:
            return []
        scores, ids = index.search(query_vector, top_k=top_k)
        out = []
        for score, row in zip(scores[0], ids[0]):
            row = int(row)
            # row < 0 是 FAISS 在候选不足时补的占位
            # 写成 not (score >= ...) 而不是 score < ...，这样 NaN 也会被过滤掉
            if row < 0 or not score >= threshold:
                continue
            if row >= len(row_ids):
                log.error("行号 %d 超出映射长度 %d，跳过该候选", row, len(row_ids))
                continue
            entry_id = row_ids[row]
            if entry_id is None:
                continue
            out.append({
                "entry_id": entry_id,
                "row": row,
                "score": float(score),
                "source": source,
            })
        return out

    def candidates(self, query: str) -> list:
        """双路召回 + **按知识条目**合并，返回候选列表（**尚未**经过 rerank）。

        每条候选：
            {
              "entry_id": str,       # 知识 id，去重与映射的唯一依据
              "question": str,
              "answer": str,
              "score": float,        # 两路中的最高分
              "source": str,         # question / answer / question+answer
              "path_scores": {...},  # 各路原始分数（取最大值），便于排查
              "hit_rows": int,       # 共命中几个向量（条目本身 + 若干问法变体）
              "variants": [...],     # 命中的问法文本，便于观察是哪个说法命中的
              "rerank_score": float, # 精排分数（NoopReranker 时为 0.0）
            }

        为什么要「超取」：一条知识有多个问法变体时，同一知识会占据多个索引行。
        如果只取 recall_k 行，去重后可能只剩 recall_k 的一半甚至更少。
        所以按 recall_overfetch 倍取行，合并去重后再截到 recall_k 个条目。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        query_vector = self.embed_fn(query, settings=self.settings)
        recall_k = max(1, int(self.recall_k))
        fetch_rows = max(recall_k, recall_k * max(1, int(self.recall_overfetch)))

        merged = {}

        def merge(hits):
            for h in hits:
                entry_id = h["entry_id"]
                doc = self.knowledge_by_id.get(entry_id)
                if doc is None:
                    log.error("索引引用了知识库中不存在的 id: %r，跳过", entry_id)
                    continue
                # 命中的是哪一条问法（用于展示与实际命中文本）
                hit_text = doc["question"]
                if h["source"] == "question" and self.question_row_ids is not None:
                    hit_text = self._question_text_for_row(h["row"]) or hit_text

                if entry_id in merged:
                    prev = merged[entry_id]
                    sources = set(prev["source"].split("+")) | {h["source"]}
                    prev["source"] = "+".join(sorted(s for s in sources if s))
                    # 同一路可能被多个变体命中，这里取最大值而不是后写覆盖，
                    # 否则低分会盖掉高分，排查时看到误导性数值
                    prev["path_scores"][h["source"]] = max(
                        prev["path_scores"].get(h["source"], float("-inf")), h["score"]
                    )
                    prev["score"] = max(prev["score"], h["score"])
                    prev["hit_rows"] += 1
                    if hit_text not in prev["variants"]:
                        prev["variants"].append(hit_text)
                else:
                    merged[entry_id] = {
                        "entry_id": entry_id,
                        "question": doc["question"],
                        "answer": doc["answer"],
                        "score": h["score"],
                        "source": h["source"],
                        "path_scores": {h["source"]: h["score"]},
                        "hit_rows": 1,
                        "variants": [hit_text],
                        "rerank_score": 0.0,
                    }

        # 第 1 路：问题索引（问法匹配，含所有问法变体）
        merge(self._recall_path(
            self.question_index, query_vector, top_k=fetch_rows,
            threshold=self.score_threshold, source="question",
            row_ids=self.question_row_ids,
        ))

        # 第 2 路：答案索引（内容匹配，阈值更低）
        if self.answer_index is not None:
            merge(self._recall_path(
                self.answer_index, query_vector, top_k=recall_k,
                threshold=self.answer_score_threshold, source="answer",
                row_ids=self.answer_row_ids,
            ))

        # 合并后按分数降序；同分时命中行数多的优先（多个说法都命中，可信度更高）
        ranked = sorted(
            merged.values(),
            key=lambda c: (c["score"], c["hit_rows"], len(c["path_scores"])),
            reverse=True,
        )
        return ranked[:recall_k]

    def _question_text_for_row(self, row: int) -> str | None:
        """把问题索引的行号还原成它实际被向量化的文本（question 或某个 alias）。

        仅用于展示与排查。索引行与文本的对应关系在构建时确定：
        每条知识先写 question，再依次写它的 aliases。
        """
        if not hasattr(self, "_question_row_texts") or self._question_row_texts is None:
            texts, seen = [], set()
            for doc in self.knowledge_list:
                entry_id = doc.get("id")
                if entry_id is None:
                    continue
                texts.append((entry_id, doc["question"]))
                for a in knowledge_aliases(doc):
                    texts.append((entry_id, a))
            self._question_row_texts = texts
        if 0 <= row < len(self._question_row_texts):
            return self._question_row_texts[row][1]
        return None

    # ------------------------------------------------------------------
    # 精排
    # ------------------------------------------------------------------
    def search(self, query: str) -> list:
        """双路召回 → 按答案去重 → rerank 精排 → 返回 top_k 条。

        **顺序很重要**：`candidates()` 已经按知识条目合并去重（同一知识的多个
        问法变体只占一个位子），再交给 rerank 排序，取出的 top_k 才给 LLM。

        注意这里**不再**按答案文本去重。旧版用 `dedupe_by_answer` 把「答案相同」
        的不同条目合并，实测会把真正不同的知识误合（曾导致评测从 96% 掉到 65%）。
        现在每条知识有唯一 id，按 id 合并是正确做法，答案文本不再参与去重。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        cands = self.candidates(query)
        if not cands:
            return []

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
                "id": c["entry_id"],
                "question": c["question"],
                "answer": c["answer"],
                "score": c["score"],            # 向量相似度，跨请求可比
                "rerank_score": float(rscore),  # 相对分数，仅本次请求内有意义
                "source": c["source"],
                # 这条知识共命中几个向量（条目本身 + 若干问法变体）
                "hit_rows": c.get("hit_rows", 1),
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

    def answer(self, query: str, history: list = None) -> dict:
        """完整问答：查询改写 → 检索 → 精排 → 交给大模型生成自然语言回答。

        这是对外的主入口。返回结构里同时给出：
          - answer          : 大模型生成的回答（未启用生成时是材料原文拼装）
          - hits            : 命中的知识条目，供前端展示「依据」
          - matched         : 是否检索到相关内容，前端据此区分「没找到」和「回答为空」
          - search_query    : 实际用于检索的查询（多轮时可能是改写后的）
          - rewritten       : 是否发生了改写，前端可据此提示用户

        为什么把 hits 一起返回：用户需要有办法核对回答的依据。
        只给一段生成文本，出了问题无法追溯是检索错了还是模型编了。

        多轮对话的关键在第一步「查询改写」：
        追问（如「它有空调吗」）离开上文无法检索，必须先补全成
        「宿舍有空调吗」再走向量匹配。详见 llm.REWRITE_PROMPT。
        """
        history = history or []
        llm = self.llm or LLM.from_config(self.settings)

        # 第 1 步：需要时把追问改写成能独立检索的问题
        search_query = query
        rewritten = False
        rewrite_error = ""
        if history:
            try:
                candidate = llm.rewrite_query(query, history)
                if candidate and candidate != query:
                    search_query = candidate
                    rewritten = True
            except RuntimeError as e:
                # 改写是增强环节，失败就用原问题继续，不让整个问答挂掉
                log.warning("查询改写失败，使用原问题：%s", e)
                rewrite_error = str(e)

        # 第 2 步：检索（用可能已改写过的查询）
        hits = self.search(search_query)

        # 第 3 步：生成回答（带上文，让模型理解「它」指什么）
        try:
            text = llm.generate(query, hits, history)
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
                "search_query": search_query,
                "rewritten": rewritten,
                "rewrite_error": rewrite_error,
            }

        return {
            "answer": text,
            "hits": hits,
            "matched": bool(hits),
            "generated": not isinstance(llm, NoopLLM),
            "generation_error": "",
            "search_query": search_query,
            "rewritten": rewritten,
            "rewrite_error": rewrite_error,
        }
