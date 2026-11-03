"""前端与接口联调验证。

用法（需先启动服务）：
    venv\\Scripts\\python.exe tests\\check_ui.py
    $env:BASE_URL="http://127.0.0.1:8001"; venv\\Scripts\\python.exe tests\\check_ui.py

验证静态资源可访问、页面结构完整、/ui-config 与 /health 正常、
/ask 返回结构化字段（含 id / hit_rows）。
"""
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service import load_knowledge  # noqa: E402

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")


def main():
    print("== 1. 静态资源可访问 ==")
    for path in ["/", "/ui/", "/ui/style.css", "/ui/app.js"]:
        r = requests.get(BASE + path, timeout=30)
        ctype = r.headers.get("content-type", "")
        print(f"  {path:<16} HTTP {r.status_code}  {ctype[:40]:<40} {len(r.content)}B")

    print("\n== 2. 页面内容检查 ==")
    html = requests.get(BASE + "/", timeout=30).text
    for token, label in [
        ('charset="UTF-8"', "UTF-8 声明"),
        ('id="map-btn"', "查询地图按钮"),
        ('id="ask-form"', "提问表单"),
        ('./style.css', "样式引用（相对路径）"),
        ('./app.js', "脚本引用（相对路径）"),
    ]:
        print(f"  {'✓' if token in html else '✗'} {label}")
    print(f"  标题: {html.split('<title>')[1].split('</title>')[0]}")

    print("\n== 3. /ui-config ==")
    print("  " + json.dumps(requests.get(BASE + "/ui-config", timeout=30).json(),
                              ensure_ascii=False))

    print("\n== 4. /health ==")
    print("  " + json.dumps(requests.get(BASE + "/health", timeout=30).json(),
                              ensure_ascii=False))

    print("\n== 5. /ask 结构化返回 ==")
    # 查询取自知识库本身（第一条知识的原问法），避免写死问法后因知识库更换而失效
    queries = []
    try:
        cfg_health = requests.get(BASE + "/health", timeout=30).json()
        if cfg_health.get("knowledge_count"):
            kb = ROOT / "db" / "data.json"
            docs = load_knowledge(kb)
            queries.append(docs[0]["question"])
    except Exception as e:
        print(f"  （读取知识库失败，使用默认查询: {e}）")
    queries += ["今天天气怎么样"]   # 无关问题，预期拒答
    for q in queries:
        r = requests.post(BASE + "/ask", json={"query": q}, timeout=120)
        d = r.json()
        print(f"  {q!r:20s} HTTP {r.status_code} matched={d['matched']} "
              f"hits={d['hit_count']}")
        for h in d["hits"][:2]:
            print(f"      [{h.get('id','?')}] 精排{h['rerank_score']:.3f} "
                  f"命中{h.get('hit_rows',1)}行 {h['question'][:30]}")
            print(f"                  {h['answer'][:40]}")

    print("\n== 6. 参数校验 ==")
    r = requests.post(BASE + "/ask", json={"query": "   "}, timeout=30)
    print(f"  空 query -> HTTP {r.status_code} {r.json().get('detail')}")


if __name__ == "__main__":
    main()
