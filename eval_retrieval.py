"""检索效果评测脚本。

目的：把「检索到底好不好」量化成一个数字，这样之后加 rerank、换 embedding 模型、
扩大资料库时，能看出是变好了还是变差了，而不是凭感觉调参。

用法：
    venv\\Scripts\\python.exe eval_retrieval.py
    venv\\Scripts\\python.exe eval_retrieval.py --model text-embedding-v3   # 换模型对比
    venv\\Scripts\\python.exe eval_retrieval.py --k 5 --json before.json    # 存档，用于前后对比

指标口径（刻意分开，便于定位病因）：
    Top-1      : 排第一的是不是正确答案        —— 反映「排序质量」
    召回@K     : 正确答案在不在前 K 名里        —— 反映「向量召回能力」
    生产命中   : 过 score_threshold + top_k 后是否还能命中 —— 反映「线上真实效果」
    误报率     : 无关问题被回答了（本应拒答）   —— 反映「拒答能力」

注意：
    评测走的是索引层的原始 KNN，绕过了 score_threshold 和 top_k 截断，
    因此 K 取多大就返回多少候选，不会因为线上 top_k=3 而看不全排序。
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from service import RAGService, Settings, VectorIndex, get_embeddings, load_knowledge

# ---------------------------------------------------------------------------
# 测试用例：question 是提问，expect 是 data.json 里正确答案的 question 文本。
# expect=None 表示「本应拒答」的无关问题。
# 用例类型覆盖：直问(direct) / 换词(paraphrase) / 口语(colloquial) / 无关(noise)
# ---------------------------------------------------------------------------
CASES = [
    # --- 直问：几乎照抄知识库标题，属于简单题 ---
    # 注意：expect 必须与 db/data.json 里的 question **逐字一致**，
    # 否则该用例会被标记为「失效」（脚本启动时会自动检查并报错）。
    # 改动知识库后请同步这里，或直接看脚本的失效提示。
    ("你是谁", "你是谁", "direct"),
    ("学校的地址是什么,快递地址应该怎么填写",
     "学校的地址是什么,快递地址应该怎么填写", "direct"),
    ("西安电子科技大学的校训是什么", "西安电子科技大学的校训是什么", "direct"),
    ("西安电子科技大学的校庆日是什么时候", "西安电子科技大学的校庆日是什么时候", "direct"),
    ("有多少个学院", "有多少个学院", "direct"),
    ("宿舍条件怎么样", "宿舍条件怎么样", "direct"),
    ("西安电子科技大学的校园环境怎么样，学校里包含了哪些设施？",
     "西安电子科技大学的校园环境怎么样，学校里包含了哪些设施？", "direct"),

    # --- 换词：同一个意思，换一种说法，考语义 ---
    ("西电的校训", "西安电子科技大学的校训是什么", "paraphrase"),
    ("西电的校歌叫什么", "西安电子科技大学的校歌是什么", "paraphrase"),
    ("学校的图书馆怎么样", "西安电子科技大学的图书馆怎么样", "paraphrase"),
    ("食堂的饭菜如何", "西安电子科技大学的食堂怎么样", "paraphrase"),
    ("你们学校是谁创办的", "西安电子科技大学的创始人是谁", "paraphrase"),
    ("学校校徽长什么样", "西安电子科技大学的校徽是什么", "paraphrase"),
    ("我能咨询哪些方面的问题", "我可以问你什么问题", "paraphrase"),
    # 复合问法：一条知识覆盖「地址」和「快递」两件事，只问其中一件
    ("学校地址在哪", "学校的地址是什么,快递地址应该怎么填写", "paraphrase"),
    ("快递地址怎么填", "学校的地址是什么,快递地址应该怎么填写", "paraphrase"),
    ("住宿条件好不好", "宿舍条件怎么样", "paraphrase"),
    ("学校风景美不美", "西安电子科技大学的校园环境怎么样，学校里包含了哪些设施？", "paraphrase"),

    # --- 口语：短句、省略、代词，最考验检索 ---
    ("学校在哪", "学校的地址是什么,快递地址应该怎么填写", "colloquial"),
    ("有几个学院", "有多少个学院", "colloquial"),
    ("你们是干嘛的", "你是谁", "colloquial"),
    ("校训", "西安电子科技大学的校训是什么", "colloquial"),
    ("宿舍", "宿舍条件怎么样", "colloquial"),

    # --- 无关：预期拒答，用来观察误报 ---
    ("今天天气怎么样", None, "noise"),
    ("帮我写一段Python代码", None, "noise"),
    ("北京有哪些大学", None, "noise"),
]


def check_cases(docs) -> list:
    """检查用例里的 expect 是否都还在知识库里。

    用例硬编码了问法文本，一旦知识库里的问法被改写，用例就会静默失效，
    表现成「命中率莫名下降」，很容易误判成检索变差。这里主动报出来。
    """
    have = {d["question"] for d in docs}
    return [(q, exp, kind) for q, exp, kind in CASES
            if exp is not None and exp not in have]


def raw_search(index: VectorIndex, query_vec, k: int):
    """绕过阈值与 top_k 的原始 KNN，返回 [(score, idx), ...] 按分数降序。"""
    scores, ids = index.search(query_vec, top_k=k)
    return [(float(s), int(i)) for s, i in zip(scores[0], ids[0]) if i >= 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="覆盖 embedding 模型，用于模型对比")
    ap.add_argument("--k", type=int, default=len(CASES), help="评估用召回深度")
    ap.add_argument("--json", default=None, help="把结果写入 json，便于前后对比")
    ap.add_argument("--rebuild", action="store_true",
                    help="按当前知识库现场重建向量（不读写 rag_index.index）")
    args = ap.parse_args()

    cfg = Settings.from_env()
    if args.model:
        cfg = replace(cfg, embedding_model=args.model)
    docs = load_knowledge(cfg.knowledge_path)

    # 先查用例是否失效：知识库问法被改写后，硬编码的 expect 会静默失配，
    # 表现成「命中率莫名下降」。这里直接报出来，避免误判。
    stale = check_cases(docs)
    if stale:
        print(f"!! 有 {len(stale)} 条用例引用了知识库里不存在的问法，结果不可信：")
        for q, exp, kind in stale:
            print(f"   [{kind}] 提问 {q!r}")
            print(f"          期望 {exp!r}")
        print("   请同步 eval_retrieval.py 的 CASES，或先把知识库问法改回。\n")

    # 知识库「标准问法 → 下标」的映射，用来判断正确答案应落在哪一行
    doc_idx_by_question = {d["question"]: i for i, d in enumerate(docs)}

    # 现场分别向量化「问题」和「答案」，与 build_index.py 的口径一致
    print(f"向量化 {len(docs)} 条知识的问题 ... (model={cfg.embedding_model})")
    q_vecs = get_embeddings([d["question"] for d in docs], settings=cfg)
    q_vec_index = VectorIndex(q_vecs.shape[1])
    q_vec_index.add(q_vecs)

    print(f"向量化 {len(docs)} 条知识的答案 ...")
    a_vecs = get_embeddings([d["answer"] for d in docs], settings=cfg)
    a_vec_index = VectorIndex(a_vecs.shape[1])
    a_vec_index.add(a_vecs)

    print(f"向量化 {len(CASES)} 条提问 ...\n")
    query_vecs = get_embeddings([q for q, _, _ in CASES], settings=cfg)
    qvec_by_query = {q: v for (q, _, _), v in zip(CASES, query_vecs)}

    print(f"评测配置: 模型={cfg.embedding_model}")
    print(f"          问题索引阈值={cfg.score_threshold}  答案索引阈值={cfg.answer_score_threshold}")
    print(f"          每路召回={cfg.recall_k}  线上top_k={cfg.top_k}  评估深度K={args.k}")
    print("注意: 下表「KNN耗时」只含本地检索，不含 embedding 网络请求，")
    print("      因此数值很小属正常；端到端延迟请以服务实测为准。")
    print("=" * 96)
    print(f"{'提问':<22}{'类型':<12}{'问题索引排名':<14}{'答案索引排名':<14}"
          f"{'合并top1':<10}{'生产命中':<10}{'KNN'}")
    print("-" * 96)

    stats = {"direct": [], "paraphrase": [], "colloquial": [], "noise": []}
    latencies = []
    rows = []

    # 用预计算向量驱动真实的 RAGService，这样「合并 + 阈值 + rerank」走的都是线上同一套逻辑
    def embed_lookup(text, *, settings=None, **kw):
        return qvec_by_query[str(text)]

    svc = RAGService(
        knowledge_list=docs,
        question_index=q_vec_index,
        answer_index=a_vec_index,
        settings=cfg,
        embed_fn=embed_lookup,
    )

    for (query, expect, kind), qv in zip(CASES, query_vecs):
        t0 = time.perf_counter()

        # 两路各自的排名（绕过阈值，看纯排序质量）
        ranked_q = raw_search(q_vec_index, qv, args.k)
        ranked_a = raw_search(a_vec_index, qv, args.k)

        # 线上真实链路：双路召回 → 阈值过滤 → 合并 → rerank → top_k
        try:
            hits = svc.search(query)
        except Exception as e:
            hits = []
            print(f"  [检索异常] {query!r}: {type(e).__name__} {e}")
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        want = doc_idx_by_question.get(expect) if expect is not None else None

        def rank_of(ranked):
            if want is None:
                return None
            for pos, (_, idx) in enumerate(ranked, start=1):
                if idx == want:
                    return pos
            return None

        rank_q = rank_of(ranked_q)
        rank_a = rank_of(ranked_a)

        # 生产命中：线上链路返回的 top_k 里有没有正确答案
        prod_ok = False
        merged_top1 = False
        if expect is not None:
            got = [doc_idx_by_question.get(h.get("question"), -1) for h in hits]
            prod_ok = want in got
            merged_top1 = bool(got) and got[0] == want

        if expect is None:
            false_pos = len(hits) > 0
            stats["noise"].append(false_pos)
            rq_txt = ra_txt = "不适用"
            mt1_txt = "—"
            prod_txt = "误报" if false_pos else "正确拒答"
        else:
            stats[kind].append({"top1": merged_top1, "recall": prod_ok})
            rq_txt = str(rank_q) if rank_q else "未召回"
            ra_txt = str(rank_a) if rank_a else "未召回"
            mt1_txt = "✓" if merged_top1 else "✗"
            prod_txt = "✓" if prod_ok else "✗"

        print(f"{query:<22}{kind:<12}{rq_txt:<14}{ra_txt:<14}"
              f"{mt1_txt:<10}{prod_txt:<10}{elapsed*1000:.0f}ms")
        rows.append({"query": query, "kind": kind, "expect": expect,
                     "rank_question": rank_q, "rank_answer": rank_a,
                     "merged_top1": merged_top1, "prod_hit": prod_ok,
                     "ms": round(elapsed * 1000, 1)})

    # ---------------- 汇总 ----------------
    print("=" * 96)

    def pct(part, whole):
        return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"

    print("\n分类型表现（合并链路 = 双路召回 + 阈值 + 合并 + rerank + top_k）：")
    for kind, label in [("direct", "直问  "), ("paraphrase", "换词  "), ("colloquial", "口语  ")]:
        items = stats[kind]
        if not items:
            continue
        t1 = sum(x["top1"] for x in items)
        rc = sum(x["recall"] for x in items)
        print(f"  {label}  合并Top-1 {pct(t1, len(items)):<14} 线上命中 {pct(rc, len(items))}")

    all_ans = stats["direct"] + stats["paraphrase"] + stats["colloquial"]
    if all_ans:
        t1 = sum(x["top1"] for x in all_ans)
        rc = sum(x["recall"] for x in all_ans)
        print(f"  {'合计  '}  合并Top-1 {pct(t1, len(all_ans)):<14} 线上命中 {pct(rc, len(all_ans))}")

    noise = stats["noise"]
    if noise:
        fp = sum(noise)
        print(f"  拒答    误报 {pct(fp, len(noise))}（越低越好）")

    # 两路各自「排第一」的用例数，用来判断答案索引到底有没有贡献
    q_top1 = sum(1 for r in rows if r["rank_question"] == 1)
    a_top1 = sum(1 for r in rows if r["rank_answer"] == 1)
    print(f"\n单路 Top-1 对比（正确答案排第 1 的用例数，共 {len(all_ans)} 条）：")
    print(f"  只查问题索引: {pct(q_top1, len(all_ans))}")
    print(f"  只查答案索引: {pct(a_top1, len(all_ans))}")

    # 答案索引的独特贡献：问题索引没召回、但答案索引召回了
    answer_saves = [r for r in rows
                    if r["expect"] and r["rank_question"] is None and r["rank_answer"] is not None]
    if answer_saves:
        print(f"\n答案索引的独特贡献（问题索引未召回、答案索引救回）: {len(answer_saves)} 条")
        for r in answer_saves:
            print(f"  {r['query']!r} -> 答案索引排名 {r['rank_answer']}")

    prod_ok_n = sum(1 for r in rows if r["expect"] and r["prod_hit"])
    ans_n = sum(1 for r in rows if r["expect"])
    print(f"\n线上真实命中率（含阈值过滤）: {pct(prod_ok_n, ans_n)}")
    print(f"平均单次检索耗时（含合并/rerank，不含embedding网络）: "
          f"{statistics.mean(latencies) * 1000:.1f}ms")

    # ---------------- 诊断提示 ----------------
    hard = [r for r in rows if r["expect"] and not r["merged_top1"]]
    if hard:
        print("\n合并后仍不正确的用例（正确答案不在第一）：")
        for r in hard:
            print(f"  [{r['kind']}] {r['query']!r} -> "
                  f"问题索引排名 {r['rank_question'] or '未召回'} / "
                  f"答案索引排名 {r['rank_answer'] or '未召回'}")

    if args.json:
        out = {"model": cfg.embedding_model, "k": args.k,
               "score_threshold": cfg.score_threshold,
               "answer_score_threshold": cfg.answer_score_threshold,
               "recall_k": cfg.recall_k, "rerank_enabled": cfg.rerank_enabled,
               "rows": rows,
               "merged_top1": sum(x["top1"] for x in all_ans) if all_ans else 0,
               "total": len(all_ans)}
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n结果已写入 {args.json}（可与其它配置的结果对比）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
