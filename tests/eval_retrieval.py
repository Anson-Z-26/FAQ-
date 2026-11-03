"""检索效果评测（适配 40 条 FAQ + aliases 的新数据结构）。

## 为什么重写

旧版的 23 条用例是手写死的，引用的是旧知识库的问法。知识库换成新数据后，
这些问法全部失效，旧脚本会报「用例引用了不存在的问法」并且结果不可信。

新版**从数据自动生成用例**，不再硬编码具体问法，所以知识库怎么改都能直接跑：

    直问   ：用每条知识的 question 当查询，期望命中自己
    别名   ：用每个 alias 当查询，期望命中它所属的条目 ← aliases 要解决的场景
    无关   ：固定一批无关问题，期望拒答（命中 0 条算正确）

## 一个必须说明的评测局限

「别名」这一层用的是**数据里已有的 alias**，它们同时也是索引内容的一部分，
所以指标会偏乐观 —— 真实用户会用我们没写进 alias 的说法。
因此这一层只能说明「已覆盖的问法能否命中」，不能外推成真实召回率。
更严格的测法是留出法（把部分 alias 从索引里拿掉当查询），见 tests/eval_aliases.py。

## 用法

    python tests/eval_retrieval.py                 # 完整评测（含 rerank，较慢）
    python tests/eval_retrieval.py --no-rerank     # 不含 rerank，快很多
    python tests/eval_retrieval.py --limit 60      # 抽样，控制调用量
    python tests/eval_retrieval.py --json out.json # 存档，便于前后对比
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import RAGService, Settings, knowledge_aliases, load_knowledge

# 无关问题：预期拒答。分三级，难的那两级才真正能测出问题。
NOISE = [
    # L1 明显无关
    "今天天气怎么样",
    "帮我写一段 Python 代码",
    "北京有哪些大学",
    # L2 领域内但知识库没有
    "西电的保研率是多少",
    "研究生宿舍在哪",
    "校长是谁",
    "学校的 WiFi 密码是多少",
    "怎么申请出国交换",
    # L3 字面高度相似但语义不同
    "西安邮电大学什么时候选课",
    "清华大学的校训是什么",
]


def build_cases(docs):
    """从数据自动生成用例：(查询, 期望 id 或 None, 类型)。"""
    cases = []
    for d in docs:
        cases.append((d["question"], d["id"], "direct"))
    for d in docs:
        for a in knowledge_aliases(d):
            cases.append((a, d["id"], "alias"))
    for q in NOISE:
        cases.append((q, None, "noise"))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true", help="关闭 rerank（更快、更省）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条有效用例（0=全部）")
    ap.add_argument("--json", default=None, help="结果写入 json，便于前后对比")
    args = ap.parse_args()

    cfg = Settings.from_env()
    docs = load_knowledge(cfg.knowledge_path)
    print(f"知识库：{len(docs)} 条，别名 {sum(len(knowledge_aliases(d)) for d in docs)} 个")
    print(f"embedding 模型：{cfg.embedding_model}")

    svc = RAGService.from_config(cfg)
    if args.no_rerank:
        from service import NoopReranker
        svc.reranker = NoopReranker()
        print("rerank：已关闭")
    else:
        print(f"rerank：开启（模型 {cfg.rerank_model}，门槛 {cfg.rerank_score_threshold}）")
    print(f"索引：问题 {len(svc.question_index)} 行，答案 {len(svc.answer_index)} 行，"
          f"top_k={svc.top_k}，recall_k={svc.recall_k}，overfetch={svc.recall_overfetch}")

    cases = build_cases(docs)
    effective = [(q, e, s) for q, e, s in cases if s != "noise"]
    noise_cases = [(q, e, s) for q, e, s in cases if s == "noise"]
    if args.limit > 0:
        effective = effective[: args.limit]
    run_cases = effective + noise_cases

    print(f"\n用例：有效 {len(effective)} 条（直问 "
          f"{sum(1 for _,_,s in effective if s=='direct')} / 别名 "
          f"{sum(1 for _,_,s in effective if s=='alias')}），"
          f"无关 {len(noise_cases)} 条")
    print("开始评测...\n")

    stats = {"direct": [], "alias": []}
    noise_results = []
    latencies = []
    rows = []

    t_start = time.perf_counter()
    for i, (query, expect, stratum) in enumerate(run_cases, start=1):
        t0 = time.perf_counter()
        try:
            hits = svc.search(query)
            err = ""
        except Exception as e:
            hits, err = [], f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        got_ids = [h["id"] for h in hits]
        if stratum == "noise":
            rejected = len(hits) == 0
            noise_results.append(rejected)
            if not rejected:
                rows.append({"query": query, "stratum": stratum, "rejected": False,
                             "got": got_ids[:3], "ms": round(ms)})
            continue

        rank = got_ids.index(expect) + 1 if expect in got_ids else None
        stats[stratum].append({"rank": rank, "top1": rank == 1, "recall": rank is not None})
        if rank != 1:
            rows.append({"query": query, "stratum": stratum, "rank": rank,
                         "got": got_ids[:3], "expect": expect, "ms": round(ms), "error": err})

        if i % 40 == 0:
            print(f"  已完成 {i}/{len(run_cases)} ...")

    elapsed = time.perf_counter() - t_start

    # ---------------- 汇总 ----------------
    def pct(a, b):
        return f"{a}/{b} ({100*a/b:.1f}%)" if b else "n/a"

    print("\n" + "=" * 70)
    print("分层结果")
    print("=" * 70)
    print(f"  {'类型':<8}{'条数':>6}{'Top-1':>18}{'Recall@'+str(svc.top_k):>18}{'MRR':>10}")
    print("  " + "-" * 58)
    allr = []
    for stratum, label in [("direct", "直问"), ("alias", "别名")]:
        items = stats[stratum]
        if not items:
            continue
        allr += items
        t1 = sum(x["top1"] for x in items)
        rc = sum(x["recall"] for x in items)
        mrr = sum(1.0 / x["rank"] for x in items if x["rank"]) / len(items)
        print(f"  {label:<8}{len(items):>6}{pct(t1,len(items)):>18}"
              f"{pct(rc,len(items)):>18}{mrr:>10.3f}")
    if allr:
        t1 = sum(x["top1"] for x in allr)
        rc = sum(x["recall"] for x in allr)
        mrr = sum(1.0 / x["rank"] for x in allr if x["rank"]) / len(allr)
        print("  " + "-" * 58)
        print(f"  {'合计':<8}{len(allr):>6}{pct(t1,len(allr)):>18}"
              f"{pct(rc,len(allr)):>18}{mrr:>10.3f}")

    if noise_results:
        ok = sum(noise_results)
        print(f"\n  拒答正确率：{pct(ok, len(noise_results))}"
              f"  （{len(noise_results)-ok} 条误报）")

    print(f"\n  平均单次检索 {statistics.mean(latencies):.0f}ms，"
          f"P95 {sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms，"
          f"总耗时 {elapsed:.0f}s")

    # ---------------- 失败明细 ----------------
    if rows:
        print("\n" + "=" * 70)
        print(f"失败明细（{len(rows)} 条）")
        print("=" * 70)
        for r in rows[:20]:
            if r["stratum"] == "noise":
                print(f"  [误报] {r['query']!r} -> 命中 {r['got']}")
            else:
                print(f"  [{r['stratum']}] {r['query']!r}")
                print(f"         期望 {r['expect']}，实际排名 {r['rank'] or '未召回'}，"
                      f"前3={r['got']}")
        if len(rows) > 20:
            print(f"  ... 其余 {len(rows)-20} 条见 --json 输出")

    if args.json:
        out = {
            "knowledge_count": len(docs),
            "embedding_model": cfg.embedding_model,
            "rerank": not args.no_rerank,
            "top_k": svc.top_k,
            "recall_k": svc.recall_k,
            "by_stratum": {
                k: {
                    "n": len(v),
                    "top1": sum(x["top1"] for x in v),
                    "recall": sum(x["recall"] for x in v),
                } for k, v in stats.items() if v
            },
            "noise": {"n": len(noise_results), "rejected": sum(noise_results)},
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 1),
                "p95": round(sorted(latencies)[int(len(latencies)*0.95)], 1),
            },
            "failures": rows,
        }
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n结果已写入 {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
