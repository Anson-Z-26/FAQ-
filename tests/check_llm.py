"""验证 /ask 的大模型生成链路（端到端，含真实延迟）。"""
import json
import time

import requests

BASE = "http://127.0.0.1:8000"

CASES = [
    ("你是谁", None),
    ("学校在哪", None),
    ("宿舍条件怎么样", None),
    ("校训是什么", None),
    ("今天天气怎么样", None),
    ("学校的电话是多少", None),
    # 显式只检索，不生成
    ("校训是什么", False),
]


def main():
    h = requests.get(f"{BASE}/health", timeout=30).json()
    print("== /health ==")
    for k in ("ready", "knowledge_count", "question_index_count",
              "rerank_enabled", "llm_enabled", "llm_model"):
        print(f"   {k}: {h.get(k)}")

    c = requests.get(f"{BASE}/ui-config", timeout=30).json()
    print(f"\n== /ui-config ==  llm_enabled={c.get('llm_enabled')}  "
          f"map_enabled={c.get('map_enabled')}")

    print("\n== /ask ==")
    for q, gen in CASES:
        body = {"query": q}
        if gen is not None:
            body["generate"] = gen
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/ask", json=body, timeout=180)
        ms = (time.perf_counter() - t0) * 1000

        if r.status_code != 200:
            print(f"  [{r.status_code}] {q!r} {r.text[:120]}")
            continue

        d = r.json()
        tag = "生成" if d["generated"] else "仅检索"
        extra = f" generate={gen}" if gen is not None else ""
        print(f"\n  {q!r}{extra}  ({ms:.0f}ms, {tag}, 命中{d['hit_count']}条)")
        if d["generation_error"]:
            print(f"    [生成异常] {d['generation_error']}")
        ans = d["answer"]
        print(f"    回答: {ans if ans else '（空，仅返回检索结果）'}")
        if d["hits"]:
            print(f"    依据: {[x['question'][:20] for x in d['hits'][:3]]}")

    print("\n== 参数校验 ==")
    for q in ["", "   "]:
        r = requests.post(f"{BASE}/ask", json={"query": q}, timeout=30)
        print(f"  query={q!r:6s} -> HTTP {r.status_code} {r.json().get('detail','')}")


if __name__ == "__main__":
    main()
