"""把知识库向量化并写入 FAISS 索引 + 索引清单。

用法：
    python build_index.py

## 建几个索引

    index/question.index  ← 每条知识的 question + **每个别名各一个向量**
    index/answer.index    ← 每条知识的 answer，一条一个
    index/manifest.json   ← 行号 → 知识 id 的映射（缺了它必然静默错答案）

## 为什么问题索引的「行数」不等于知识条数

一条知识可能有 3~4 个问法变体（aliases）。**每个变体必须单独成为一个向量**：

    question "宿舍条件怎么样"      → 1 个向量
    alias    "宿舍有空调吗"        → 1 个向量
    alias    "住宿环境好不好"      → 1 个向量

不能把三个变体拼成一个字符串去向量化 —— 那会把语义平均成一个模糊的点，
跟任何一个具体问法都不像。实测：5 个语义分散的变体拼接后 Top-1 为 0/5。

代价是行数放大（40 条知识 × 平均 4.1 ≈ 164 行），所以**行号不再等于知识下标**，
必须靠 manifest.json 建立映射。这就是本文件要产出清单的原因。

## 关于内容指纹

清单里为每条知识存了内容指纹（question+aliases+answer 的哈希），
启动时如果发现指纹不一致，会明确报告「索引已过期」，而不是继续用旧向量。
只改 source/url/updated 不会影响指纹（它们不参与向量化）。
"""
import logging
import sys

from service import Settings, VectorIndex, get_embeddings, knowledge_aliases, load_knowledge
from service.manifest import build_manifest

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_index")


def check_duplicates(docs: list) -> int:
    """数据质量检查：不同 id 之间是否出现完全相同的 question 或 answer。

    不直接报错，因为业务上可能存在「两条知识答案相同但来源不同」的合法情况。
    但这类重复会让检索结果出现重复内容，值得提醒。
    """
    warnings = 0
    seen_q, seen_a = {}, {}
    for d in docs:
        q = (d.get("question") or "").strip()
        a = (d.get("answer") or "").strip()
        if q in seen_q:
            log.warning("数据质量：%s 与 %s 的 question 完全相同", seen_q[q], d.get("id"))
            warnings += 1
        else:
            seen_q[q] = d.get("id")
        if a in seen_a:
            log.warning("数据质量：%s 与 %s 的 answer 完全相同", seen_a[a], d.get("id"))
            warnings += 1
        else:
            seen_a[a] = d.get("id")
    return warnings


def main() -> int:
    cfg = Settings.from_env()
    docs = load_knowledge(cfg.knowledge_path)
    if not docs:
        log.error("知识库为空: %s", cfg.knowledge_path)
        return 1

    total_variants = sum(len(knowledge_aliases(d)) for d in docs)
    log.info("读取知识库 %s：%d 条知识，%d 个别名",
             cfg.knowledge_path, len(docs), total_variants)
    log.info("embedding 模型 %s（维度 %d）", cfg.embedding_model, cfg.embedding_dim)

    check_duplicates(docs)

    # ---------------------------------------------------------------
    # 问题索引：question + 每个 alias 各一个向量
    # 顺序必须与 manifest 的 question_rows 严格一致
    # ---------------------------------------------------------------
    q_texts, q_row_ids = [], []
    for doc in docs:
        doc_id = doc["id"]
        q_texts.append(doc["question"])
        q_row_ids.append(doc_id)
        for alias in knowledge_aliases(doc):
            q_texts.append(alias)
            q_row_ids.append(doc_id)

    log.info("向量化问题索引 %d 条文本（%d 条知识 + %d 个别名）...",
             len(q_texts), len(docs), len(q_texts) - len(docs))
    q_vectors = get_embeddings(q_texts, settings=cfg)

    q_index = VectorIndex(q_vectors.shape[1])
    q_index.add(q_vectors)
    q_index.save(cfg.question_index_path)
    log.info("问题索引已写入 %s：%d 行向量", cfg.question_index_path.name, q_index.ntotal)

    # ---------------------------------------------------------------
    # 答案索引：每条知识一个向量（别名不参与，答案只有一份）
    # ---------------------------------------------------------------
    a_texts = [doc["answer"] for doc in docs]
    a_row_ids = [doc["id"] for doc in docs]

    log.info("向量化答案索引 %d 条 ...", len(a_texts))
    a_vectors = get_embeddings(a_texts, settings=cfg)

    a_index = VectorIndex(a_vectors.shape[1])
    a_index.add(a_vectors)
    a_index.save(cfg.answer_index_path)
    log.info("答案索引已写入 %s：%d 行向量", cfg.answer_index_path.name, a_index.ntotal)

    # ---------------------------------------------------------------
    # 索引清单：行号 → 知识 id
    # ---------------------------------------------------------------
    manifest = build_manifest(
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        knowledge_list=docs,
        question_rows=q_row_ids,
        answer_rows=a_row_ids,
    )
    manifest.save(cfg.manifest_path)
    log.info("索引清单已写入 %s", cfg.manifest_path.name)

    # ---------------------------------------------------------------
    # 自检：清单、索引、知识库三者必须一致
    # ---------------------------------------------------------------
    problems = manifest.validate(
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        knowledge_list=docs,
        question_ntotal=q_index.ntotal,
        answer_ntotal=a_index.ntotal,
    )
    if problems:
        log.error("建索引后自检失败：")
        for p in problems:
            log.error("  - %s", p)
        return 1

    log.info("自检通过：%d 条知识，问题索引 %d 行，答案索引 %d 行",
             len(docs), q_index.ntotal, a_index.ntotal)
    log.info("完成。接下来可运行 tests/eval_retrieval.py 查看检索效果")
    return 0


if __name__ == "__main__":
    sys.exit(main())
