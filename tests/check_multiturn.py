"""多轮对话验证：指代消解、会话隔离、重置。

用法（需先启动服务）：
    venv\\Scripts\\python.exe tests\\check_multiturn.py
    $env:BASE_URL="http://127.0.0.1:8001"; venv\\Scripts\\python.exe tests\\check_multiturn.py

## 为什么改成数据驱动

旧版把「宿舍条件怎么样」「校训是什么」这类问法写死了。知识库换成新的 40 条
FAQ 后这些条目不存在，测试会在「本来就查不到」的情况下通过，等于什么也没验证。

新版从知识库里取一条真实知识当第一轮提问，断言的是**行为**而不是内容：

    1. 同一会话追问时 `rewritten` 必须为 True   ← 证明历史被用上了
    2. 新会话同一追问时 `rewritten` 必须为 False ← 证明历史没串
    3. reset 之后 `rewritten` 回到 False         ← 证明历史被清空

这样知识库怎么改，测试都仍然有意义。
"""
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service import load_knowledge  # noqa: E402

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
KB = Path(os.environ.get("KNOWLEDGE_PATH", ROOT / "db" / "data.json"))

# 刻意用代词，离开上文无法检索。改写后应能补全成完整问题。
PRONOUN_FOLLOWUP = "它还有什么要注意的"


def ask(query, session_id=None, reset=False, generate=True):
    body = {"query": query, "keep_history": True, "generate": generate}
    if session_id:
        body["session_id"] = session_id
    if reset:
        body["reset"] = True
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/ask", json=body, timeout=180)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    d["_ms"] = ms
    return d


def show(tag, d):
    rw = f"  改写->{d['search_query']!r}" if d["rewritten"] else "  （未改写）"
    print(f"  {tag}")
    print(f"    命中{d['hit_count']}条  {d['_ms']:.0f}ms{rw}")
    if d["rewrite_error"]:
        print(f"    [改写异常] {d['rewrite_error']}")
    if d["answer"]:
        print(f"    答: {d['answer'][:90]}")


def main():
    docs = load_knowledge(KB)
    # 取一条问题足够具体、答案足够长的知识当第一轮
    target = next((d for d in docs
                   if len(d.get("question") or "") >= 8
                   and len(d.get("answer") or "") >= 40), docs[0])
    first_q = target["question"]

    print("=" * 74)
    print("场景 1：同一会话内的追问（指代消解）")
    print("=" * 74)
    print(f"  第一轮用知识库里的真实问法：{first_q!r}")
    d1 = ask(first_q)
    sid = d1["session_id"]
    show(f"第1轮  session={sid[:8] or '(空)'}...", d1)
    if not sid:
        print("  !! 首轮没有返回 session_id，多轮无法进行")
        return 1

    d2 = ask(PRONOUN_FOLLOWUP, session_id=sid)
    show(f"追问 {PRONOUN_FOLLOWUP!r}", d2)
    ok1 = d2["rewritten"]
    print(f"    预期改写=True，实际 {d2['rewritten']} -> "
          f"{'✓ 历史被用于消解指代' if ok1 else '✗ 历史未生效'}")

    print("\n" + "=" * 74)
    print("场景 2：会话隔离（新会话无上文，不应改写）")
    print("=" * 74)
    d_new = ask(PRONOUN_FOLLOWUP)
    show(f"新会话 {d_new['session_id'][:8]}...", d_new)
    ok2 = not d_new["rewritten"]
    print(f"    预期改写=False，实际 {d_new['rewritten']} -> "
          f"{'✓ 历史未串' if ok2 else '✗ 历史串了'}")

    print("\n" + "=" * 74)
    print("场景 3：reset 清空历史")
    print("=" * 74)
    d_reset = ask(PRONOUN_FOLLOWUP, session_id=sid, reset=True)
    show("同一会话 + reset=true", d_reset)
    ok3 = not d_reset["rewritten"]
    print(f"    预期改写=False，实际 {d_reset['rewritten']} -> "
          f"{'✓ 历史已清空' if ok3 else '✗ 历史未清空'}")

    print("\n" + "=" * 74)
    print("场景 4：会话状态查询")
    print("=" * 74)
    info = requests.get(f"{BASE}/session/{sid}", timeout=30).json()
    print(f"  会话 {sid[:8]}... 当前 {info['turn_count']} 轮，"
          f"服务端共 {info['known_sessions']} 个会话")
    for i, h in enumerate(info["history"], 1):
        print(f"    [{i}] 问:{h['question'][:30]!r}  答:{h['answer'][:30]!r}")

    print("\n" + "=" * 74)
    print("场景 5：单轮模式（keep_history=false，不建会话）")
    print("=" * 74)
    body = {"query": first_q, "keep_history": False}
    d_single = requests.post(f"{BASE}/ask", json=body, timeout=180).json()
    print(f"  keep_history=false -> session_id={d_single['session_id']!r}"
          f"（空表示未建会话）")
    print(f"  命中 {d_single['hit_count']} 条")

    print("\n" + "=" * 74)
    ok = ok1 and ok2 and ok3
    print("全部通过：多轮指代消解、会话隔离、reset 均生效" if ok
          else "存在失败项，见上方 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
