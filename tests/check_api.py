"""一行命令验证 API Key 是否可用（并可顺带建索引）。

用法（在项目根目录执行）：
    venv\\Scripts\\python.exe tests\\check_api.py          # 只验证连通性
    venv\\Scripts\\python.exe tests\\check_api.py --build  # 验证通过后直接建索引

它只发一次「你好」的请求，几乎不消耗额度。
"""
import sys
from pathlib import Path

# 本脚本在 tests/ 下，项目根目录是其上一级。
# 加入 sys.path 才能 import service（包）以及 build_index（根目录的模块）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import Settings, get_embedding


def main() -> int:
    build = "--build" in sys.argv
    cfg = Settings.from_env()

    print("接口地址:", cfg.embedding_url)
    print("模型    :", cfg.embedding_model)
    print("期望维度:", cfg.embedding_dim)
    print("Key     :", ("已读取 len=%d" % len(cfg.api_key)) if cfg.api_key else "!! 空 !!")
    print("-" * 56)

    if not cfg.api_key:
        print("失败：.env 里没有读到 API Key。")
        print("请在 .env 中填写 DASHSCOPE_API_KEY=你的新Key")
        return 1

    try:
        vec = get_embedding("你好", settings=cfg)
    except RuntimeError as e:
        print("失败：", e)
        print()
        print("如果报 401 / invalid_api_key，说明 Key 本身无效：")
        print("  1. 打开 https://platform.qianwenai.com/home 控制台")
        print("  2. 创建一个新的 API Key 并复制（注意不要多复制空格或换行）")
        print("  3. 粘贴到 .env 的 DASHSCOPE_API_KEY=")
        return 1

    print("成功：接口连通，返回维度 =", vec.shape[0])
    if vec.shape[0] != cfg.embedding_dim:
        print(f"警告：维度与 EMBEDDING_DIM({cfg.embedding_dim}) 不一致，请修改 .env")
        return 1

    if build:
        print("-" * 56)
        print("开始建立索引 ...")
        import build_index
        return build_index.main()

    print()
    print("下一步：venv\\Scripts\\python.exe build_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
