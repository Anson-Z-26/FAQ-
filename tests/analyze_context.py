"""量化：rerank 之后实际有多少条进入上下文、消耗多少 token、多给几条有没有用。

回答三个问题：
    1. 送进大模型的知识条数是「固定 5 条」还是随查询变化？
    2. 多给几条（5 vs 3 vs 1）到底能不能让正确答案进来？
    3. 上下文实际占多少 token？
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import RAGService, Settings, knowledge_aliases, load_knowledge

SAMPLE = 50   # 抽样查询数（每条一次 rerank 调用）


def est_tokens(text: str) -> int:
    """粗估 token：中文约 1 字 1 token，英文约 4 字符 1 token。只用于量级参考。"""
    cn = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return int(cn + (len(text) - cn) / 4)


def main() -> int:
    cfg = Settings.from_env()
    docs = load_knowledge(cfg.knowledge_path)
    svc = RAGService.from_config(cfg)

    # 构造查询：每条知识取一个别名（保证有正确答案），抽样
    queries = []
    for d in docs:
        al = knowledge_aliases(d)
        if al:
            queries.append((al[0], d["id"]))
    queries = queries[:SAMPLE]
    print(f"抽样 {len(queries)} 条查询（每条一项知识，用其第一个别名）\n")

    # 分两种 top_k 各测一遍：
    #   k=10 + 关闭动态截断：观察「按 rerank 顺序排下来」的完整情况（正确答案排第几名）
    #   k=5  + 使用配置的 ratio：线上实际行为，回答「到底会送几条」
    orig_top_k = svc.top_k
    orig_ratio = svc.rerank_relative_ratio
    results = {}

    for k in (10, 5):
        svc.top_k = k
        svc.rerank_relative_ratio = 0.0 if k == 10 else orig_ratio
        counts, ctx_tokens, ranks, scores_pairs = [], [], [], []
        for q, expect_id in queries:
            hits = svc.search(q)
            counts.append(len(hits))
            ctx_tokens.append(est_tokens(svc.retrieve(q)))
            ids = [h["id"] for h in hits]
            ranks.append(ids.index(expect_id) + 1 if expect_id in ids else None)
            sc = [h["rerank_score"] for h in hits]
            if len(sc) >= 5:
                scores_pairs.append((sc[0], sc[4]))
        results[k] = {
            "counts": counts, "ctx": ctx_tokens, "ranks": ranks, "scores": scores_pairs,
        }

    svc.top_k = orig_top_k
    svc.rerank_relative_ratio = orig_ratio
    counts = results[10]["counts"]
    ctx_tokens = results[5]["ctx"]
    rank_of_correct = results[10]["ranks"]
    score_rank = results[10]["scores"]

    print(f"动态截断 ratio = {orig_ratio}（0 表示关闭）\n")
    print("=" * 72)
    print("1. 实际进入上下文的条数分布")
    print("=" * 72)
    for k in (5, 10):
        c = results[k]["counts"]
        dist = {}
        for x in c:
            dist[x] = dist.get(x, 0) + 1
        hit_max = sum(1 for x in c if x == k)
        note = "（关闭动态截断，看完整排名）" if k == 10 else "（线上配置）"
        print(f"\n   top_k={k} {note}")
        print(f"     平均 {statistics.mean(c):.1f} 条，范围 {min(c)}~{max(c)} 条")
        print(f"     恰好 {k} 条（用满）: {hit_max}/{len(c)} "
              f"({100*hit_max/len(c):.0f}%)")
        print("     " + "  ".join(f"{x}条×{dist[x]}" for x in sorted(dist)))

    print("\n" + "=" * 72)
    print("2. 正确答案排在第几名（决定 top_k 取多少才有用）")
    print("=" * 72)
    rd = {}
    for r in rank_of_correct:
        key = r if r else "未召回"
        rd[key] = rd.get(key, 0) + 1
    for k in sorted(rd, key=lambda x: (isinstance(x, str), x)):
        bar = "█" * rd[k]
        print(f"   第 {k} 名: {rd[k]:>3} 次  {bar}")
    top1 = sum(1 for r in rank_of_correct if r == 1)
    top3 = sum(1 for r in rank_of_correct if r and r <= 3)
    top5 = sum(1 for r in rank_of_correct if r and r <= 5)
    n = len(rank_of_correct)
    print(f"\n   正确答案在 top-1: {top1}/{n} ({100*top1/n:.0f}%)")
    print(f"   正确答案在 top-3: {top3}/{n} ({100*top3/n:.0f}%)")
    print(f"   正确答案在 top-5: {top5}/{n} ({100*top5/n:.0f}%)")
    print(f"\n   如果把 top_k 从 5 降到 3，会多丢 {top5-top3} 条正确答案"
          f"（{100*(top5-top3)/n:.0f}%）")
    print(f"   如果降到 1，会多丢 {top5-top1} 条（{100*(top5-top1)/n:.0f}%）")

    print("\n" + "=" * 72)
    print("3. 上下文 token 成本")
    print("=" * 72)
    print(f"   平均 {statistics.mean(ctx_tokens):.0f} token，"
          f"最大 {max(ctx_tokens)} token（仅知识部分）")
    print(f"   系统提示词 + 用户问题约再加 200~300 token")
    print(f"   输出上限 LLM_MAX_TOKENS=800")
    if score_rank:
        first = statistics.mean(s for s, _ in score_rank)
        fifth = statistics.mean(s for _, s in score_rank)
        print(f"\n   第 1 名 rerank 分数均值 {first:.3f}，第 5 名均值 {fifth:.3f}")
        print(f"   门槛 0.40：第 5 名平均{'高于' if fifth >= 0.4 else '低于'}门槛")
    return 0


if __name__ == "__main__":
    sys.exit(main())
