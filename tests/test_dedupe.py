"""验证「按知识条目去重」：一条知识有多个问法变体时，只能占一个候选位。

## 背景

有问法变体后，问题索引是「一条知识 = 多个向量」：

    KB001 "宿舍条件怎么样"  -> 1 个向量
    KB001 "宿舍有空调吗"    -> 1 个向量
    KB001 "住宿环境好不好"  -> 1 个向量

一次查询可能同时命中同一个条目的多个向量。如果不去重，`top_k=5` 会被一两条
知识占满，其他相关知识点根本进不来，而且同一段答案会在上下文里重复出现。

## 本脚本验证三件事

1. 同一 id 在候选列表里只出现一次
2. `hit_rows` 正确记录「这条知识命中了几个向量」
3. `variants` 记录实际命中的问法文本（便于排查为什么召回）

## 备注：为什么不再按答案文本去重

旧实现用 `dedupe_by_answer` 把「答案文本相同」的不同条目合并成一条。
实测这会把真正不同的知识误合（评测从 96% 掉到 65%），因为业务上
两条不同知识完全可能给出相同的答案文本。现在每条知识有唯一 id，
按 id 去重既安全又准确。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import NoopReranker, RAGService, Settings, VectorIndex

# 每条知识带 id 与 aliases。
# 注意 DOC4 与 DOC2 的答案文本**故意相同**，用来验证「不再按答案文本合并」。
DOCS = [
    {
        "id": "DOC1",
        "question": "宿舍条件怎么样",
        "aliases": ["宿舍有空调吗", "住宿环境好不好"],
        "answer": "宿舍条件良好，配备了基本生活设施。",
    },
    {
        "id": "DOC2",
        "question": "食堂怎么样",
        "aliases": ["食堂好吃吗"],
        "answer": "食堂环境整洁，菜品丰富。",
    },
    {
        "id": "DOC3",
        "question": "图书馆几点开门",
        "aliases": [],
        "answer": "图书馆每天 8:00 开放。",
    },
    {
        "id": "DOC4",
        "question": "宿舍几点熄灯",
        "aliases": [],
        # 与 DOC2 答案完全相同，但这是另一条知识，不该被合并掉
        "answer": "食堂环境整洁，菜品丰富。",
    },
]

DIM = 32


def fake_embed(text, *, settings=None, **kw):
    """确定性假向量：按字做哈希桶，保证不同文本落在不同位置。"""
    v = np.zeros(DIM, dtype=np.float32)
    for ch in str(text):
        v[ord(ch) % DIM] += 1.0
    if not v.any():
        v[0] = 1.0
    return v


def build_service(cfg):
    q_texts, q_row_ids = [], []
    for d in DOCS:
        q_texts.append(d["question"]); q_row_ids.append(d["id"])
        for a in d["aliases"]:
            q_texts.append(a); q_row_ids.append(d["id"])

    q_index = VectorIndex(DIM)
    q_index.add(np.stack([fake_embed(t) for t in q_texts]))

    return RAGService(
        knowledge_list=DOCS,
        question_index=q_index,
        answer_index=None,
        settings=cfg,
        embed_fn=fake_embed,
        reranker=NoopReranker(),
        top_k=2,
        score_threshold=0.0,
        recall_k=10,
        question_row_ids=q_row_ids,
    ), len(q_texts)


def main() -> int:
    cfg = Settings.from_env()
    svc, n_rows = build_service(cfg)

    print(f"语料：{len(DOCS)} 条知识，问题索引 {n_rows} 行"
          f"（{len(DOCS)} 个 question + "
          f"{sum(len(d['aliases']) for d in DOCS)} 个别名）\n")

    # ---------- 1. 同一 id 只出现一次 ----------
    query = "宿舍"
    cands = svc.candidates(query)
    ids = [c["entry_id"] for c in cands]
    print(f"[1] 查询 {query!r} -> {len(cands)} 条候选")
    for c in cands:
        print(f"    {c['entry_id']} 命中 {c['hit_rows']} 行  命中说法 {c['variants']}")
    assert len(ids) == len(set(ids)), f"同一 id 出现了多次: {ids}"
    print("    ✓ 无重复 id，同一知识只占一个候选位")

    # ---------- 2. 多变体应合并成一条，并记录命中行数 ----------
    doc1 = next(c for c in cands if c["entry_id"] == "DOC1")
    print(f"\n[2] DOC1 命中 {doc1['hit_rows']} 个向量，说法 {doc1['variants']}")
    assert doc1["hit_rows"] > 1, "DOC1 应该被多个向量命中（question + 2 个别名）"
    assert doc1["hit_rows"] == len(doc1["variants"]), "hit_rows 应与记录的说法数一致"
    print("    ✓ 多变体已合并为一条，并记录了命中行数与具体说法")

    # ---------- 3. 不再按答案文本合并：DOC2 与 DOC4 答案相同但必须都存在 ----------
    doc2 = next(c for c in cands if c["entry_id"] == "DOC2")
    doc4 = next((c for c in cands if c["entry_id"] == "DOC4"), None)
    merged_by_answer = doc4 is None
    print(f"\n[3] DOC2 与 DOC4 答案文本相同")
    print(f"    DOC2 存在: {doc2 is not None}   DOC4 存在: {doc4 is not None}")
    assert not merged_by_answer, (
        "DOC4 被按答案文本合并掉了！这正是旧 dedupe_by_answer 的缺陷 —— "
        "两条不同知识可以给出相同答案文本，按文本合并会丢失知识"
    )
    print("    ✓ 两条不同知识都被保留（不按答案文本合并）")

    # ---------- 4. top_k 不被单条知识占满 ----------
    hits = svc.search(query)
    top_ids = [h["id"] for h in hits]
    print(f"\n[4] top_k=2 的返回：{top_ids}")
    assert len(set(top_ids)) == len(top_ids), "返回结果里有重复知识"
    print("    ✓ 返回的每条知识互不相同")

    print("\n全部检查通过：按知识条目去重生效，且不再按答案文本误合并。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
