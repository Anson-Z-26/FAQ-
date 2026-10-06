"""服务启动入口。

原来 main.py 和 api/rag_service.py 各写了一份 FastAPI app（且 main.py 引用了一个
不存在的 load_index），两边逻辑还不一致。现在 app 只在 api/rag_service.py 定义一次，
这里只负责把它跑起来。
"""
from api.rag_service import app  # noqa: F401  uvicorn 的 "main:app" 需要这个名字可见

__all__ = ["app"]


def main():
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
