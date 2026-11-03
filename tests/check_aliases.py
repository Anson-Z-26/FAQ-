"""端到端联调：验证 aliases 新结构下 API 返回正确（含 id / hit_rows）。"""
import json
import os

import requests

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")


def main():
    print("=== /health ===")
    h = requests.get(f"{BASE}/health", timeout=30).json()
    print("  " + json.dumps(h, ensure_ascii=False))

    print("\n=== /ask（用别名提问，验证 id 与 hit_rows）===")
    for q in ["什么时候选课", "预选是什么意思", "国奖评选要求", "今天天气怎么样"]:
        d = requests.post(f"{BASE}/ask", json={"query": q}, timeout=120).json()
        print(f"\n  问: {q!r}  命中 {d['hit_count']} 条  matched={d['matched']}")
        if d["hits"]:
            for h2 in d["hits"][:2]:
                print(f"    [{h2['id']}] 命中 {h2['hit_rows']} 行 "
                      f"精排 {h2['rerank_score']:.3f}")
                print(f"        {h2['question'][:34]}")
        print(f"    答: {d['answer'][:70]}")

    print("\n=== /ask generate=false（只检索）===")
    d = requests.post(f"{BASE}/ask", json={"query": "什么时候选课", "generate": False},
                      timeout=60).json()
    print(f"  generated={d['generated']} 命中{d['hit_count']}条 answer={d['answer']!r}")

    print("\n=== 参数校验 ===")
    r = requests.post(f"{BASE}/ask", json={"query": "   "}, timeout=30)
    print(f"  空 query -> HTTP {r.status_code} {r.json().get('detail')}")


if __name__ == "__main__":
    main()
