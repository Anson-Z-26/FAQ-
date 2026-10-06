"""HTTP 层。

只做参数校验和调用 RAGService，不再自己读知识库、拼索引路径。
索引/知识库的加载放在 lifespan 里，进程启动时执行一次，而不是模块导入时
（旧的 api/rag_service.py 在 import 阶段就读 db/data.json，测试和热重载都会重复触发）。

同时托管 frontend/ 目录下的静态页面，前端不需要单独起服务、也不需要构建步骤。
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载索引和知识库是同步的磁盘 IO，放进线程池，别卡住事件循环
    app.state.rag_service = await asyncio.to_thread(RAGService.from_config)
    yield
    app.state.rag_service = None


app = FastAPI(title="RAG服务", lifespan=lifespan)


class AnswerReq(BaseModel):
    query: str  # 客户端传递的用户问题文本


@app.get("/health")
async def health(req: Request):
    """健康检查。

    特意把索引条数暴露出来：索引没建好时服务照样能启动、接口也返回 200，
    但检索结果永远是空的。这种「安静失败」最难排查，所以让 /health 能看出来。
    """
    svc = req.app.state.rag_service
    doc_count = len(svc.knowledge_list)
    q_count = len(svc.question_index)
    a_count = len(svc.answer_index) if svc.answer_index is not None else None

    ready = q_count == doc_count and doc_count > 0
    if a_count is not None and a_count != doc_count:
        ready = False

    return {
        "code": 0,
        "ready": ready,
        "knowledge_count": doc_count,
        "question_index_count": q_count,
        "answer_index_count": a_count,
        "rerank_enabled": svc.settings.rerank_enabled,
        "embedding_model": svc.settings.embedding_model,
        "hint": "" if ready else "索引与知识库条数不一致，请先运行 build_index.py",
    }


@app.post("/get_answer")
async def get_answer(request: AnswerReq, req: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query不能为空")
    try:
        context = await asyncio.to_thread(req.app.state.rag_service.retrieve, request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # embedding 接口挂了/配置错了，是上游问题，不该报成 500
        raise HTTPException(status_code=502, detail=str(e))
    return {"code": 0, "answer": context}


@app.post("/ask")
async def ask(request: AnswerReq, req: Request):
    """结构化检索结果，供前端渲染。

    与 /get_answer 的区别：/get_answer 返回拼好的上下文字符串（给大模型用），
    这里返回结构化列表（给界面用），前端可以逐条展示、显示来源与匹配度。
    保留 /get_answer 是为了不破坏已有调用方。
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query不能为空")
    try:
        hits = await asyncio.to_thread(req.app.state.rag_service.search, request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "code": 0,
        "query": request.query,
        "hit_count": len(hits),
        # 命中为空时前端要显示「没找到」而不是「空回答」，所以单独给个标记
        "matched": len(hits) > 0,
        "hits": [
            {
                "question": h["question"],
                "answer": h["answer"],
                "score": round(float(h.get("score", 0.0)), 4),
                "rerank_score": round(float(h.get("rerank_score", 0.0)), 4),
                "source": h.get("source", ""),
                "variant_count": h.get("variant_count", 1),
            }
            for h in hits
        ],
    }


@app.get("/ui-config")
async def ui_config(req: Request):
    """前端启动时读取的配置。

    把「查询地图」地址放在后端配置里，前端不写死链接，
    以后换链接只改 .env 的 QUERY_MAP_URL 即可，不用动前端代码。
    """
    cfg = req.app.state.rag_service.settings
    return {
        "code": 0,
        "title": cfg.ui_title,
        "map_url": cfg.query_map_url,
        "map_enabled": bool(cfg.query_map_url),
        "rerank_enabled": cfg.rerank_enabled,
        "top_k": cfg.top_k,
    }


# ---------------------------------------------------------------------------
# 静态前端
#
# 挂载在 /ui 下，不影响已有的 /get_answer、/ask 等接口。
# 用 html=True 让 /ui/ 自动返回 index.html。
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def index():
        """根路径重定向到 /ui/。

        这里必须用重定向而不是直接返回 index.html：页面里的 css/js 用的是
        相对路径（./style.css），如果直接在 / 返回 HTML，浏览器会去请求
        /style.css，而静态资源挂在 /ui 下，结果全是 404——页面会变成
        「能用但没样式」的破样子。重定向后基路径是 /ui/，相对路径才成立。
        """
        return RedirectResponse(url="/ui/", status_code=307)
