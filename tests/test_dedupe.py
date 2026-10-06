"""验证「按答案去重」：构造同答案多变体的语料，检查去重行为。

去重解决的问题：给一条知识补多个问法变体时必须拆成多行（每个变体一个尖锐向量），
但多条变体常同时命中，会一起挤占 top_k 并把同一段答案重复拼进上下文。
"""
import sys
from pathlib import Path

# 本脚本在 tests/ 下，项目根目录是其上一级；加入 sys.path 才能 import service
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import NoopReranker, RAGService, Settings

# 模拟：宿舍这条知识有 3 个问法变体，答案完全相同
DOCS = [
    {"question": "你是谁", "answer": "我是智能问答助手。"},
    {"question": "宿舍条件怎么样", "answer": "宿舍条件良好，配备了基本生活设施。"},
    {"question": "宿舍有空调吗", "answer": "宿舍条件良好，配备了基本生活设施。"},
    {"question": "宿舍能洗澡吗", "answer": "宿舍条件良好，配备了基本生活设施。"},
    {"question": "食堂怎么样", "answer": "食堂环境整洁，菜品丰富。"},
]


def main() -> int:
    cfg = Settings.from_env()
    print("语料：5 行，其中 3 行是「宿舍」的同一答案变体\n")

    # ---------- 1. dedupe_by_answer 纯函数测试 ----------
    cands = [
        {"question": "宿舍有空调吗", "answer": "A", "score": 0.9,
         "source": "question", "path_scores": {"question": 0.9}},
        {"question": "宿舍条件怎么样", "answer": "A", "score": 0.8,
         "source": "answer", "path_scores": {"answer": 0.8}},
        {"question": "宿舍能洗澡吗", "answer": "A", "score": 0.7,
         "source": "question", "path_scores": {"question": 0.7}},
        {"question": "食堂怎么样", "answer": "B", "score": 0.6,
         "source": "question", "path_scores": {"question": 0.6}},
    ]

    best = RAGService.dedupe_by_answer(cands, keep="best")
    print("[1] keep=best：4 条候选（答案 A 占 3 条）->", len(best), "条")
    for c in best:
        print(f"    question={c['question']!r} score={c['score']} "
              f"variant_count={c['variant_count']} source={c['source']}")
    assert len(best) == 2, "应合并为 A、B 两条"
    a = next(c for c in best if c["answer"] == "A")
    assert a["variant_count"] == 3, "应记录被合并的变体数"
    assert a["question"] == "宿舍有空调吗", "keep=best 应保留分数最高的变体"
    # 两路分数应取并集
    assert a["path_scores"] == {"question": 0.9, "answer": 0.8}, a["path_scores"]
    assert a["source"] == "answer+question", a["source"]
    print("    ✓ 合并正确：保留最高分变体、变体数=3、两路分数与来源已合并")

    longest = RAGService.dedupe_by_answer(cands, keep="longest")
    a2 = next(c for c in longest if c["answer"] == "A")
    print(f"\n[2] keep=longest：A 的 question 拼接为 {a2['question']!r}")
    assert a2["question"].count("/") == 2, "应把 3 个变体都拼进来"
    print("    ✓ 三个变体问法都拼进来了，rerank 能看到更多问法")

    # ---------- 2. 端到端：去重是否让 top_k 留出位子 ----------
    def fake_embed(text, *, settings=None, **kw):
        import numpy as np
        v = np.zeros(16, dtype=np.float32)
        for ch in str(text):
            v[ord(ch) % 16] += 1.0
        if not v.any():
            v[0] = 1.0
        return v

    from service import VectorIndex
    import numpy as np

    qs = [d["question"] for d in DOCS]
    vecs = np.stack([fake_embed(q) for q in qs])
    idx = VectorIndex(vecs.shape[1])
    idx.add(vecs)

    svc = RAGService(knowledge_list=DOCS, question_index=idx, answer_index=None,
                     settings=cfg, embed_fn=fake_embed, reranker=NoopReranker(),
                     top_k=3, score_threshold=0.0, recall_k=10)
    cands2 = svc.candidates("宿舍有空调吗")
    n_dup = sum(1 for c in cands2 if c["answer"].startswith("宿舍条件良好"))
    print(f"\n[3] 去重前候选 {len(cands2)} 条，其中宿舍变体 {n_dup} 条")

    hits = svc.search("宿舍有空调吗")
    answers = [h["answer"] for h in hits]
    print(f"    去重后 top_k 返回 {len(hits)} 条：")
    for h in hits:
        print(f"      {h['question'][:24]!r}  variant_count={h['variant_count']}")
    assert len(answers) == len(set(answers)), f"返回结果里仍有重复答案: {answers}"
    print("    ✓ 返回结果内无重复答案，同一答案只占一个位子")

    # 对比：不去重会怎样
    raw = cands2[:3]
    print(f"\n[4] 若不去重，top_k=3 会被前 3 条占满：")
    for c in raw:
        print(f"      {c['question'][:24]!r}")
    dup_ratio = len({c['answer'] for c in raw}) / len(raw)
    print(f"    → 3 个位子里只有 {len({c['answer'] for c in raw})} 个不同答案"
          f"（有效率 {dup_ratio:.0%}）")

    print("\n全部检查通过：按答案去重生效，且发生在 rerank 之前。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
