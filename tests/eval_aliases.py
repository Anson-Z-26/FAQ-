"""aliases 效果验证：留出法（避免用原句当查询的测试污染）。

## 为什么要留出

`tests/eval_retrieval.py` 用数据里已有的 alias 当查询，而这些 alias **同时也在
索引里**，所以指标会偏乐观 —— 真实用户会用我们没写进 alias 的说法。
本脚本用留出法避开这个问题：把每条知识的一部分 alias 从索引里拿掉，只用来查询。

## 对比三种建索引方式

    A) 只用 question                              —— 基线
    B) question + 索引内 alias 拼成一个字符串
    C) question 与每个 alias 各自一个向量

## 已知结论（见文件末尾输出）

在 40 条知识上，B 和 C 都能提升 Recall@5，但 Top-1 差异在噪声范围内
（留出查询只有几十条，1~2 条的差异不构成结论）。
拼接方案 B 的问题在**变体多且语义分散**时才暴露：早前用 5 个语义不同
的变体（空调/洗澡/几人间）测过，拼接后 Top-1 为 0/5。

结论：用 C（多向量）更稳，因为它不随变体数量增加而退化。

不访问外部文件，直接用项目自己的知识库。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import Settings, get_embeddings, knowledge_aliases, load_knowledge


def norm(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def evaluate(doc_vecs, doc_entry, query_vecs, query_entry, k=5):
    sim = norm(query_vecs) @ norm(doc_vecs).T
    top1 = rec = 0
    mrr = 0.0
    for i, want in enumerate(query_entry):
        order = np.argsort(-sim[i])
        rank = None
        for pos, row in enumerate(order, start=1):
            if doc_entry[row] == want:
                rank = pos
                break
        if rank == 1:
            top1 += 1
        if rank and rank <= k:
            rec += 1
        mrr += (1.0 / rank) if rank else 0.0
    n = len(query_entry)
    return {"top1": top1 / n, "recall": rec / n, "mrr": mrr / n, "n": n}


def main():
    cfg = Settings.from_env()
    rows = load_knowledge(cfg.knowledge_path)

    # ---- 留出切分：每条知识前 2 个 alias 进索引，其余当查询 ----
    index_alias = []   # (text, entry_idx)
    held_alias = []    # (text, entry_idx)
    for i, r in enumerate(rows):
        al = knowledge_aliases(r)
        for j, a in enumerate(al):
            (index_alias if j < 2 else held_alias).append((a, i))

    print(f"知识 {len(rows)} 条")
    print(f"进索引的 alias: {len(index_alias)} 个（每条前 2 个）")
    print(f"留出做查询的 alias: {len(held_alias)} 个 ← 这些文本不在索引里\n")

    if not held_alias:
        print("每条知识的别名不足 3 个，无法做留出切分（至少需要 3 个）")
        return 1

    questions = [r["question"] for r in rows]
    idx_texts = [t for t, _ in index_alias]
    idx_owner = [i for _, i in index_alias]
    held_texts = [t for t, _ in held_alias]
    held_owner = [i for _, i in held_alias]

    print("向量化中...")
    q_vec = get_embeddings(questions, settings=cfg)
    ia_vec = get_embeddings(idx_texts, settings=cfg)
    h_vec = get_embeddings(held_texts, settings=cfg)

    # B 方案：question + 索引内 alias 拼成一个字符串
    concat = []
    for i, q in enumerate(questions):
        extra = [t for t, o in index_alias if o == i]
        concat.append(q + "；" + "；".join(extra) if extra else q)
    c_vec = get_embeddings(concat, settings=cfg)

    print("\n" + "=" * 78)
    print("测试：用「留出的 alias」提问（这些说法没出现在索引里）")
    print("=" * 78)

    r_a = evaluate(q_vec, list(range(len(rows))), h_vec, held_owner)
    r_b = evaluate(c_vec, list(range(len(rows))), h_vec, held_owner)

    doc_vecs, doc_entry = [], []
    for i in range(len(rows)):
        doc_vecs.append(q_vec[i]); doc_entry.append(i)
        for t, o in index_alias:
            if o == i:
                doc_vecs.append(ia_vec[idx_texts.index(t)]); doc_entry.append(i)
    r_c = evaluate(np.stack(doc_vecs), doc_entry, h_vec, held_owner)

    print(f"  {'方案':<30}{'Top-1':>10}{'Recall@5':>12}{'MRR':>10}")
    print("  " + "-" * 62)
    for label, r in [("A) 仅 question（基线）", r_a),
                     ("B) 拼接成 1 个向量", r_b),
                     ("C) 每变体 1 个向量", r_c)]:
        print(f"  {label:<30}{r['top1']:>9.1%}{r['recall']:>12.1%}{r['mrr']:>10.3f}")

    print(f"\n  A→C 提升: Top-1 {r_a['top1']:.1%} -> {r_c['top1']:.1%} "
          f"({(r_c['top1']-r_a['top1'])*100:+.1f} 个百分点)")
    print(f"  B 相对 C: Top-1 {r_b['top1']:.1%} vs {r_c['top1']:.1%} "
          f"({(r_c['top1']-r_b['top1'])*100:+.1f} 个百分点)")

    # ---- 回归：原始 question 还能不能命中 ----
    print("\n" + "=" * 78)
    print("回归检查：用原始 question 提问，确认加别名没把原有效果搞坏")
    print("=" * 78)
    qo = list(range(len(rows)))
    r_a2 = evaluate(q_vec, list(range(len(rows))), q_vec, qo)
    r_c2 = evaluate(np.stack(doc_vecs), doc_entry, q_vec, qo)
    for label, r in [("A) 仅 question", r_a2), ("C) 每变体 1 个向量", r_c2)]:
        print(f"  {label:<30}{r['top1']:>9.1%}{r['recall']:>12.1%}{r['mrr']:>10.3f}")

    # ---- 拥挤度：top_k 里能拿到几个不同条目 ----
    print("\n" + "=" * 78)
    print("副作用：同一条知识的多变体会占用多个候选位")
    print("=" * 78)
    sim = norm(h_vec) @ norm(np.stack(doc_vecs)).T
    for k in (3, 5, 10):
        distinct = []
        for i in range(len(held_texts)):
            order = np.argsort(-sim[i])[:k]
            distinct.append(len({doc_entry[r] for r in order}))
        print(f"  方案 C，取 top_{k:<2}: 平均 {np.mean(distinct):.1f} 个不同条目 "
              f"（理想 {k}）")
    print(f"\n  向量总数: A={len(rows)}   C={len(doc_vecs)}（放大 {len(doc_vecs)/len(rows):.1f} 倍）")


if __name__ == "__main__":
    main()
