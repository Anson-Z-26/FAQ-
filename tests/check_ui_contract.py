"""前端契约检查：确认 HTML / JS / CSS 之间的引用都对得上。

无头浏览器在当前沙箱下起不来（Edge 无法访问自己的 profile 目录），
所以改用静态检查三件最容易出错的事：
  1. JS 里 getElementById 的 id 在 HTML 里都存在
  2. HTML 引用的 css/js 文件真实存在
  3. JS 里用到的 CSS class 在样式表里有定义（漏了只会"不好看"，但值得知道）
"""
import re
import sys
from pathlib import Path

# 本脚本在 tests/ 下，前端源码在项目根目录的 frontend/
ROOT = Path(__file__).resolve().parent.parent
FE = ROOT / "frontend"


def read(name):
    return (FE / name).read_text(encoding="utf-8")


def main() -> int:
    html, js, css = read("index.html"), read("app.js"), read("style.css")
    ok = True

    # --- 1. HTML 里的 id ---
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    # JS 里 $('xxx') 和 getElementById('xxx')
    js_ids = set(re.findall(r"\$\('([^']+)'\)", js))
    js_ids |= set(re.findall(r"getElementById\('([^']+)'\)", js))
    # JS 运行时自己创建的 id（例如打字指示器），这类不算缺失
    js_created = set(re.findall(r"\.id\s*=\s*'([^']+)'", js))

    missing = sorted(js_ids - html_ids - js_created)
    print(f"[1] JS 引用的 id {len(js_ids)} 个（其中运行时自建 {len(js_created)} 个）")
    if missing:
        ok = False
        print(f"    ✗ HTML 里缺少这些 id: {missing}")
    else:
        print("    ✓ 全部可解析（HTML 里存在，或由 JS 运行时创建）")

    # 反向：HTML 有但 JS 没用的 id（只是提示，不算错）
    unused = sorted(html_ids - js_ids)
    print(f"    · HTML 中未被 JS 直接引用的 id: {unused or '无'}")

    # --- 2. 静态资源引用 ---
    print("\n[2] HTML 引用的静态资源")
    for m in re.findall(r'(?:href|src)="\./([^"]+)"', html):
        exists = (FE / m).exists()
        ok &= exists
        print(f"    {'✓' if exists else '✗'} {m}")

    # --- 3. JS 里用到的 class 是否有样式 ---
    print("\n[3] JS 动态生成的 class 是否在 CSS 中有定义")
    js_classes = set()
    for m in re.findall(r"className\s*=\s*[`'\"]([^`'\"]+)[`'\"]", js):
        js_classes |= set(m.split())
    for m in re.findall(r"classList\.(?:add|remove)\('([^']+)'\)", js):
        js_classes.add(m)
    # 拼在模板字符串里的 class="xxx"
    for m in re.findall(r'class="([^"]+)"', js):
        js_classes |= set(m.split())

    css_classes = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))
    # 过滤掉模板字符串插值和运算符误匹配（如 ${kind}、||）
    real = sorted(c for c in js_classes
                  if c.isidentifier() and c not in ("disabled", "hidden"))
    no_style = sorted(c for c in real if c not in css_classes)
    print(f"    JS 用到的 class: {real}")
    if no_style:
        print(f"    · 无对应样式（可能是有意的，仅提示）: {no_style}")
    else:
        print("    ✓ 全部有对应样式")

    # --- 4. 后端接口是否都被前端正确调用 ---
    # 这里是后端实际暴露的路由清单。新增接口时要同步，否则检查会误报「不存在」。
    print("\n[4] 前端调用的后端接口")
    called = sorted(set(re.findall(r"fetch\('([^']+)'", js)))
    available = {
        "/ask",            # 问答主入口（多轮）
        "/health",         # 服务与索引状态
        "/ui-config",      # 前端配置
        "/session/reset",  # 清空会话历史
        "/get_answer",     # 只检索（旧接口，保留兼容）
    }
    for c in called:
        hit = c in available
        ok &= hit
        print(f"    {'✓' if hit else '✗'} {c}")

    print("\n" + ("全部检查通过。" if ok else "存在失败项，见上方 ✗。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
