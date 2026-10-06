"""把知识库向量化并写入 FAISS 索引。

用法：
    python build_index.py

建的是**两个**索引：
    index/question.index  ← 每条知识的 question 向量，用于「问法匹配」召回
    index/answer.index    ← 每条知识的 answer   向量，用于「内容匹配」召回

为什么要分开：把 question+answer 拼成一条去向量化，答案文字会稀释问题向量的语义
（本知识库条目全都以「西安电子科技大学」开头，区分度本来就低），而且答案向量
单独建索引后，能捞回「用户问法与知识库问法对不上、但答案内容相关」的条目。

两个索引的行号必须与 db/data.json 的数组下标严格一致，检索结果才能映射回条目。
"""
import logging
import sys

from service import Settings, VectorIndex, get_embeddings, load_knowledge

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_index")


def main() -> int:
    cfg = Settings.from_env()
    docs = load_knowledge(cfg.knowledge_path)
    if not docs:
        log.error("知识库为空: %s", cfg.knowledge_path)
        return 1
    log.info("读取知识库 %s，共 %d 条", cfg.knowledge_path, len(docs))
    log.info("embedding 模型 %s（维度 %d）", cfg.embedding_model, cfg.embedding_dim)

    # 问题、答案分别取文本，顺序与 docs 一致 —— 行号即条目下标，这是映射的关键
    questions = [doc["question"] for doc in docs]
    answers = [doc["answer"] for doc in docs]

    log.info("向量化 %d 条问题 ...", len(questions))
    q_vectors = get_embeddings(questions, settings=cfg)

    log.info("向量化 %d 条答案 ...", len(answers))
    a_vectors = get_embeddings(answers, settings=cfg)

    q_index = VectorIndex(q_vectors.shape[1])
    q_index.add(q_vectors)
    q_index.save(cfg.question_index_path)
    log.info("问题索引已写入 %s，共 %d 条向量", cfg.question_index_path.name, q_index.ntotal)

    a_index = VectorIndex(a_vectors.shape[1])
    a_index.add(a_vectors)
    a_index.save(cfg.answer_index_path)
    log.info("答案索引已写入 %s，共 %d 条向量", cfg.answer_index_path.name, a_index.ntotal)

    # 两个索引条数不一致会导致按行号映射时错位，这里做个自检
    if q_index.ntotal != len(docs) or a_index.ntotal != len(docs):
        log.error("索引条数与知识库不一致（知识 %d / 问题 %d / 答案 %d），请检查",
                  len(docs), q_index.ntotal, a_index.ntotal)
        return 1

    log.info("完成。接下来可运行 eval_retrieval.py 查看检索效果")
    return 0


if __name__ == "__main__":
    sys.exit(main())
