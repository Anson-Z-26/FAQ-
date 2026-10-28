"""多轮对话验证：指代消解、会话隔离、重置。

用法（需先启动服务）：
    venv\\Scripts\\python.exe tests\\check_multiturn.py
    $env:BASE_URL="http://127.0.0.1:8001"; venv\\Scripts\\python.exe tests\\check_multiturn.py

这个脚本验证的是多轮特有的三件事，单轮测试覆盖不到：
  1. 追问里的指代（「它」「那」）能否被正确消解并检索到
  2. 不同 session 的历史是否互不干扰
  3. reset 是否真的清空了历史
"""
import os
import time

import requests

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")


def ask(query, session_id=None, reset=False, generate=None, keep_history=True):
    body = {"query": query, "keep_history": keep_history}
    if session_id:
        body["session_id"] = session_id
    if reset:
        body["reset"] = True
    if generate is not None:
        body["generate"] = generate
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/ask", json=body, timeout=180)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    d["_ms"] = ms
    return d


def show(tag, d):
    rw = f"  改写->{d['search_query']!r}" if d["rewritten"] else ""
    print(f"  {tag}")
    print(f"    命中{d['hit_count']}条  {d['_ms']:.0f}ms{rw}")
    if d["generation_error"]:
        print(f"    [生成异常] {d['generation_error']}")
    if d["rewrite_error"]:
        print(f"    [改写异常] {d['rewrite_error']}")
    print(f"    答: {d['answer'][:100]}")


def main():
    print("=" * 72)
    print("场景 1：同一会话内的追问（指代消解）")
    print("=" * 72)
    d1 = ask("宿舍条件怎么样")
    sid = d1["session_id"]
    show(f"第1轮 宿舍条件怎么样  session={sid[:8] or '(空)'}...", d1)
    if not sid:
        print("  !! 首轮没有返回 session_id，多轮无法进行")
        return

    for q in ["它有空调吗", "那食堂呢", "它在哪里"]:
        show(f"追问 {q}", ask(q, session_id=sid))

    print("\n" + "=" * 72)
    print("场景 2：会话隔离（新会话问「它有空调吗」，无上文应检索不到）")
    print("=" * 72)
    d_new = ask("它有空调吗")
    show(f"新会话 {d_new['session_id'][:8]}... 它有空调吗", d_new)
    print(f"    预期命中 0 条，实际 {d_new['hit_count']} 条 -> "
          f"{'✓ 正确隔离' if d_new['hit_count'] == 0 else '✗ 历史串了'}")

    print("\n" + "=" * 72)
    print("场景 3：reset 清空历史")
    print("=" * 72)
    d_reset = ask("它有空调吗", session_id=sid, reset=True)
    show("同一会话 + reset=true 后问「它有空调吗」", d_reset)
    print(f"    预期不再有上文可消解，命中 {d_reset['hit_count']} 条 -> "
          f"{'✓ 历史已清空' if d_reset['hit_count'] == 0 else '⚠ 仍有命中（可能靠答案索引蒙到）'}")

    print("\n" + "=" * 72)
    print("场景 4：会话状态查询")
    print("=" * 72)
    info = requests.get(f"{BASE}/session/{sid}", timeout=30).json()
    print(f"  会话 {sid[:8]}... 当前 {info['turn_count']} 轮，"
          f"服务端共 {info['known_sessions']} 个会话")
    for i, h in enumerate(info["history"], 1):
        print(f"    [{i}] 问:{h['question'][:26]!r}  答:{h['answer'][:30]!r}")

    print("\n" + "=" * 72)
    print("场景 5：单轮模式（keep_history=false，不读写历史）")
    print("=" * 72)
    d_single = ask("校训是什么", keep_history=False)
    print(f"  keep_history=false -> session_id={d_single['session_id']!r}（空表示未建会话）")
    print(f"  命中 {d_single['hit_count']} 条，答: {d_single['answer'][:60]}")


if __name__ == "__main__":
    main()
