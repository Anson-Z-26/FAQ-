"""前端与接口联调验证。"""
import json

import requests

BASE = "http://127.0.0.1:8000"


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
    for q in ["你是谁", "学校在哪", "今天天气怎么样"]:
        r = requests.post(BASE + "/ask", json={"query": q}, timeout=120)
        d = r.json()
        print(f"  {q!r:14s} HTTP {r.status_code} matched={d['matched']} hits={d['hit_count']}")
        for h in d["hits"][:2]:
            rs = h["rerank_score"]
            print(f"      [{rs:.3f}] {h['question'][:34]}")
            print(f"              {h['answer'][:44]}")

    print("\n== 6. 参数校验 ==")
    r = requests.post(BASE + "/ask", json={"query": "   "}, timeout=30)
    print(f"  空 query -> HTTP {r.status_code} {r.json().get('detail')}")


if __name__ == "__main__":
    main()
