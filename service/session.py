"""多轮对话的会话存储。

## 为什么需要它

单轮问答时，问题和知识库一一对应就够了。多轮对话的核心难点在于
**追问里的指代**：

    用户：宿舍条件怎么样
    用户：它有空调吗        ← "它"指什么？检索层只看到"它有空调吗"

如果不处理，第二问的向量和任何知识都匹配不上（实测命中 0 条）。
所以多轮对话必须解决两件事：
  1. 记住上文（本模块负责）
  2. 把追问改写成能独立检索的问题（见 llm.rewrite_query）

## 为什么先放内存

单进程、量小的场景下内存字典最省事。但它有两个已知边界，写在这里免得误用：
  - **多 worker 会失效**：uvicorn --workers 2 时两个进程各存一份，
    同一会话的请求落到不同 worker 就丢历史。要多 worker 需换 Redis。
  - **重启即丢**：进程退出历史就没了。
所以本模块定位是「单进程开发/小规模部署可用」，不是生产级方案。

## 清理策略

内存 dict 不做清理会一直涨。这里用两条约束：
  - TTL：超过 session_ttl_seconds 未活动的会话视为过期
  - 轮数上限：只保留最近 max_turns 轮，避免上下文无限变长（既费 token 又会稀释重点）
每次访问顺手清理过期会话，不做后台线程。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Turn:
    """一轮对话。question 是用户原话，answer 是助手回答。"""

    question: str
    answer: str


@dataclass
class Session:
    """一个会话的历史记录。"""

    session_id: str
    turns: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
        self.updated_at = time.time()

    def history(self, max_turns: int) -> list:
        """取最近 max_turns 轮，按时间正序返回 [{question, answer}, ...]。"""
        if max_turns <= 0:
            return []
        recent = self.turns[-max_turns:]
        return [{"question": t.question, "answer": t.answer} for t in recent]


class SessionStore:
    """线程安全的会话存储。

    FastAPI 用线程池跑同步逻辑（asyncio.to_thread），
    多个请求可能并发改同一个会话，所以加锁。
    """

    def __init__(self, max_turns: int = 5, ttl_seconds: int = 1800):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._sessions = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _is_expired(self, session: Session, now: float) -> bool:
        return (now - session.updated_at) > self.ttl_seconds

    def _purge_expired(self, now: float) -> None:
        """顺手清理过期会话。调用前必须已持锁。"""
        dead = [sid for sid, s in self._sessions.items() if self._is_expired(s, now)]
        for sid in dead:
            self._sessions.pop(sid, None)

    # ------------------------------------------------------------------
    def create(self) -> Session:
        session = Session(session_id=uuid.uuid4().hex)
        with self._lock:
            now = time.time()
            self._purge_expired(now)
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """取会话；不存在或已过期返回 None。"""
        if not session_id:
            return None
        with self._lock:
            now = time.time()
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if self._is_expired(session, now):
                self._sessions.pop(session_id, None)
                return None
            return session

    def get_or_create(self, session_id: str = None) -> Session:
        """按 id 取会话，取不到就新建。

        注意：传入的 id 不存在时会**新建一个带新 id 的会话**，
        而不是沿用传入的 id——避免客户端随便传字符串就占用内存。
        """
        existing = self.get(session_id)
        return existing if existing is not None else self.create()

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.add_turn(question, answer)
            # 只保留最近若干轮，防止无限增长
            if len(session.turns) > self.max_turns:
                session.turns = session.turns[-self.max_turns:]
            session.updated_at = time.time()

    def reset(self, session_id: str) -> bool:
        """清空某个会话的历史（保留会话本身）。返回是否找到该会话。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.turns = []
            session.updated_at = time.time()
            return True

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def history(self, session_id: str) -> list:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return session.history(self.max_turns)
