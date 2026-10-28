"""HTTP 层。

只做参数校验和调用 RAGService，不再自己读知识库、拼索引路径。
索引/知识库的加载放在 lifespan 里，进程启动时执行一次，而不是模块导入时
（旧的 api/rag_service.py 在 import 阶段就读 db/data.json，测试和热重载都会重复触发）。

多轮对话的会话状态也在这里管理（SessionStore），因此本层同时承担「会话边界」的职责：
service 层保持无状态、可测试，会话的读写都在 HTTP 层完成。

同时托管 frontend/ 目录下的静态页面，前端不需要单独起服务、也不需要构建步骤。
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from service import RAGService, SessionStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载索引和知识库是同步的磁盘 IO，放进线程池，别卡住事件循环
    app.state.rag_service = await asyncio.to_thread(RAGService.from_config)
    cfg = app.state.rag_service.settings
    app.state.sessions = SessionStore(
        max_turns=cfg.max_turns,
        ttl_seconds=cfg.session_ttl_seconds,
    )
    yield
    app.state.rag_service = None
    app.state.sessions = None


app = FastAPI(title="RAG服务", lifespan=lifespan)


class AnswerReq(BaseModel):
    query: str  # 客户端传递的用户问题文本


class AskReq(BaseModel):
    query: str
    # 是否让大模型生成自然语言回答。默认跟随 .env 的 LLM_ENABLED；
    # 显式传 false 可以只做检索（用于调试检索效果、或前端只想看依据）。
    generate: bool | None = None
    # 是否保留多轮历史。默认 false = 单轮问答，不读也不写历史。
    #
    # 为什么默认 false 而不是 true：/ask 是通用接口，批量调用/脚本测试
    # 不该意外地写满服务端内存。前端会显式传 true 并带上 session_id。
    keep_history: bool = False
    # 会话 id。配合 keep_history=true 使用：
    #   - 传已知 id：接着这个会话聊
    #   - 不传或传空：新建会话，返回的 session_id 供后续请求使用
    session_id: str | None = None
    # 传 true 表示「开始新对话」：清空该会话历史后再回答本问。
    reset: bool = False


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
        "llm_enabled": svc.settings.llm_enabled,
        "llm_model": svc.settings.llm_model if svc.settings.llm_enabled else "",
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
async def ask(request: AskReq, req: Request):
    """问答主入口：查询改写 → 检索 → 精排 → （可选）大模型生成回答。

    `generate` 不传时跟随 .env 的 LLM_ENABLED：
      - 开启：返回模型生成的回答，并附带命中依据
      - 关闭：直接返回命中的知识原文（不调模型，不会产生生成费用）

    多轮对话（需 keep_history=true）：
      - keep_history=false（默认）：单轮处理，不读也不写历史
      - keep_history=true 且不传 session_id：新建会话，响应里返回新的 session_id
      - keep_history=true 且传了 session_id：读该会话历史用于消解追问里的指代，
        并把本轮追加进去
      - reset=true：先清空该会话历史再回答

    会话 id 的处理策略：客户端传了不存在的 id 时**新建会话并返回新 id**，
    而不是沿用传入的 id —— 避免客户端随便传字符串就占住服务端内存。

    与 /get_answer 的区别：/get_answer 只做检索、返回拼好的上下文字符串，
    且不支持多轮。保留它是为了不破坏已有调用方。
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query不能为空")

    svc = req.app.state.rag_service
    sessions = req.app.state.sessions

    # 未显式指定时，沿用配置里的开关
    want_generate = svc.settings.llm_enabled if request.generate is None else request.generate

    # ---- 会话准备（查询改写依赖历史，所以必须在检索之前拿到） ----
    #
    # 这里必须能被「第一次提问」触发：前端首次提问时还没有 session_id，
    # 只有 keep_history=true。所以判断条件是 keep_history or reset，
    # 而不是「有没有 session_id」——否则第一轮永远建不出会话，
    # 第二轮也就拿不到历史，多轮会静默失效。
    history = []
    session_id = ""
    if request.keep_history or request.reset:
        session = sessions.get_or_create(request.session_id or "")
        session_id = session.session_id
        if request.reset:
            sessions.reset(session_id)
        history = sessions.history(session_id)

    try:
        if want_generate:
            result = await asyncio.to_thread(svc.answer, request.query, history)
        else:
            # 只检索，不调模型。这里也做改写，否则多轮追问会检索不到
            search_query = request.query
            if history:
                search_query = await asyncio.to_thread(
                    svc.llm.rewrite_query, request.query, history
                )
            hits = await asyncio.to_thread(svc.search, search_query)
            result = {"answer": "", "hits": hits, "matched": bool(hits),
                      "generated": False, "generation_error": "",
                      "search_query": search_query,
                      "rewritten": search_query != request.query,
                      "rewrite_error": ""}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ---- 记录本轮（只记有意义的问答：命中为空时也记，用户可能接着追问） ----
    if session_id:
        answer_text = result["answer"]
        if not answer_text and result["hits"]:
            # 只检索模式下没有生成回答，用首条知识代替，保证历史可用
            answer_text = result["hits"][0]["answer"]
        sessions.add_turn(session_id, request.query, answer_text)

    hits = result["hits"]
    return {
        "code": 0,
        "query": request.query,
        # 生成后的回答。未启用生成或只做检索时为空字符串
        "answer": result["answer"],
        "generated": result["generated"],
        "generation_error": result["generation_error"],
        # 多轮相关：实际用于检索的查询（可能被改写）以及是否改写过
        "search_query": result.get("search_query", request.query),
        "rewritten": result.get("rewritten", False),
        "rewrite_error": result.get("rewrite_error", ""),
        "session_id": session_id,
        "hit_count": len(hits),
        # 命中为空时前端要显示「没找到」而不是「空回答」，所以单独给个标记
        "matched": result["matched"],
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


class SessionReq(BaseModel):
    session_id: str


@app.post("/session/reset")
async def reset_session(request: SessionReq, req: Request):
    """清空指定会话的历史（保留会话本身，可继续使用同一 id）。

    也可以直接用 /ask 的 reset=true 参数，效果相同且少一次往返，
    这个独立路由是给「只清历史、不提问」的场景准备的。
    """
    sessions = req.app.state.sessions
    session_id = request.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    found = sessions.reset(session_id)
    return {"code": 0, "session_id": session_id, "reset": found}


@app.get("/session/{session_id}")
async def session_info(session_id: str, req: Request):
    """查看会话当前历史轮数，便于调试多轮是否生效。"""
    sessions = req.app.state.sessions
    history = sessions.history(session_id)
    return {
        "code": 0,
        "session_id": session_id,
        "turn_count": len(history),
        "history": history,
        "known_sessions": sessions.count(),
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
        "llm_enabled": cfg.llm_enabled,
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
