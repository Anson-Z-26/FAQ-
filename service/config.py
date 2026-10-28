"""集中读取运行配置。

原来 embedding.py / faiss.py 各自在 import 时读 os.environ，导致改配置必须重启进程、
模块导入带副作用、测试也没法传不同参数。现在统一收在这里，Settings 本身只是一份快照。
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（service/ 的上一级），用于把 .env 里的相对路径解析成绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent


def _first_env(*names, default=None):
    """按顺序返回第一个非空环境变量，用于兼容历史变量名。"""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _env_int(name, default):
    raw = _first_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"环境变量 {name} 需要是整数，当前为 {raw!r}")


def _env_float(name, default):
    raw = _first_env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"环境变量 {name} 需要是数字，当前为 {raw!r}")


def _env_bool(name, default):
    raw = _first_env(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _env_path(name, default):
    raw = _first_env(name, default=default)
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _normalize_embedding_url(raw: str) -> str:
    """把 base_url 形态的配置补全成完整的 embeddings 接口地址。

    只配到 base_url（如 .../compatible-mode/v1）是最常见的踩坑点，
    这里补上 /embeddings，让两种写法都能用。
    """
    url = (raw or "").strip().rstrip("/")
    if not url or url.endswith("/embeddings"):
        return url
    if url.endswith("/compatible-mode/v1") or url.endswith("/compatible-mode/v1/"):
        return f"{url}/embeddings"
    if url.endswith("/v1") or url.endswith("/api/v1"):
        return f"{url}/embeddings"
    return url


def _normalize_rerank_url(raw: str) -> str:
    """补全 rerank 接口地址。

    注意 rerank 的路径与 embedding 不同：是 /compatible-api/v1/reranks
    （compatible-api，不是 compatible-mode），别配错成同一套。
    """
    url = (raw or "").strip().rstrip("/")
    if not url or url.endswith("/reranks"):
        return url
    if url.endswith("/compatible-api/v1"):
        return f"{url}/reranks"
    return url


def _normalize_chat_url(raw: str) -> str:
    """补全 chat completions 接口地址。

    三种写法都接受：
        .../compatible-mode/v1                      -> 补 /chat/completions
        .../compatible-mode/v1/                     -> 同上
        .../compatible-mode/v1/chat/completions     -> 原样返回
    注意 chat 和 embedding 共用同一个 base_url，但路径后缀不同。
    """
    url = (raw or "").strip().rstrip("/")
    if not url or url.endswith("/chat/completions"):
        return url
    if url.endswith("/compatible-mode/v1") or url.endswith("/v1"):
        return f"{url}/chat/completions"
    return url


@dataclass(frozen=True)
class Settings:
    """一次配置快照。构造后不可变，可安全地在多个对象间共享。"""

    api_key: str
    embedding_url: str
    embedding_model: str
    embedding_dim: int
    top_k: int

    # --- 双路召回 ---
    # 问题和答案分成两个索引：问题索引负责「问法匹配」，答案索引负责「内容匹配」。
    # 两条路的相似度分布不同，所以各自有独立阈值。
    score_threshold: float          # 问题索引阈值
    answer_score_threshold: float   # 答案索引阈值
    recall_k: int                   # 每路各取多少候选，合并后交给 rerank
    answer_index_enabled: bool      # 关掉则退化为「只用问题索引」
    dedupe_keep: str                # 同答案多变体去重时保留哪个问法：best / longest

    # --- rerank（默认关闭）---
    rerank_enabled: bool
    rerank_url: str
    rerank_model: str
    # 开启 rerank 后用它做「要不要回答」的门槛。
    # 注意：这个分数与向量相似度 score 不同源，不能混用同一个阈值。
    rerank_score_threshold: float

    # --- 大模型生成（检索之后才能用）---
    # 关闭时 /ask 只返回检索到的知识原文；开启后才会生成自然语言回答。
    llm_enabled: bool
    llm_url: str
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    llm_timeout: int
    # 是否关闭深度思考。RAG 问答不需要思维链，
    # 实测关闭后不产生 reasoning token，延迟从约 2.2s 降到约 1.6s。
    llm_disable_thinking: bool

    max_batch_size: int
    index_dir: Path
    knowledge_path: Path

    # --- 前端 ---
    # 「查询地图」跳转按钮的地址。留空时前端把按钮置灰并提示，
    # 不硬编码任何具体地图链接，等确定后再填。
    query_map_url: str
    ui_title: str
    ui_static_dir: Path

    # --- 派生路径 ---
    @property
    def question_index_path(self) -> Path:
        return self.index_dir / "question.index"

    @property
    def answer_index_path(self) -> Path:
        return self.index_dir / "answer.index"

    @classmethod
    def from_env(cls) -> "Settings":
        # 只在这里读 .env，模块导入本身没有任何副作用
        load_dotenv()
        return cls(
            # 按供应商常见命名依次尝试：DashScope/通义千问 → 旧 DeepSeek 变量名
            api_key=_first_env("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "API_KEY", default=""),
            # 兼容历史变量名与 base_url 写法
            embedding_url=_normalize_embedding_url(
                _first_env("EMBEDDING_URL", "DASHSCOPE_EMBEDDING_URL",
                           "DEEPSEEK_EMBEDDING_URL", default="")
            ),
            embedding_model=_first_env("EMBEDDING_MODEL", default="text-embedding-v3"),
            embedding_dim=_env_int("EMBEDDING_DIM", 1024),
            top_k=_env_int("TOP_K", 3),
            score_threshold=_env_float("SCORE_THRESHOLD", 0.45),
            # 答案与用户问句的字面重合度天然低于「标准问法」，所以阈值要更低
            answer_score_threshold=_env_float("ANSWER_SCORE_THRESHOLD", 0.35),
            recall_k=_env_int("RECALL_K", 10),
            answer_index_enabled=_env_bool("ANSWER_INDEX_ENABLED", True),
            dedupe_keep=_first_env("DEDUPE_KEEP", default="best"),
            rerank_enabled=_env_bool("RERANK_ENABLED", False),
            rerank_url=_normalize_rerank_url(
                _first_env("RERANK_URL",
                           default="https://maas.qianwenaiapi.com/compatible-api/v1/reranks")
            ),
            rerank_model=_first_env("RERANK_MODEL", default="qwen3-rerank"),
            # 实测：有效查询 rerank 最高分 0.49~0.94，噪声 0.26~0.29，0.40 卡在中间
            rerank_score_threshold=_env_float("RERANK_SCORE_THRESHOLD", 0.40),
            # 默认关闭：没有配好模型时不该让 /ask 直接报错
            llm_enabled=_env_bool("LLM_ENABLED", False),
            llm_url=_normalize_chat_url(
                _first_env("LLM_URL", default="https://maas.qianwenaiapi.com/compatible-mode/v1")
            ),
            llm_model=_first_env("LLM_MODEL", default="deepseek-v4.1-flash"),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 800),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.3),
            llm_timeout=_env_int("LLM_TIMEOUT", 60),
            llm_disable_thinking=_env_bool("LLM_DISABLE_THINKING", True),
            # 接口单次 input 条数上限，本平台为 10
            max_batch_size=_env_int("MAX_BATCH_SIZE", 10),
            index_dir=_env_path("INDEX_DIR", "index"),
            knowledge_path=_env_path("KNOWLEDGE_PATH", "db/data.json"),
            # 留空即未配置，前端会禁用按钮
            query_map_url=_first_env("QUERY_MAP_URL", default=""),
            ui_title=_first_env("UI_TITLE", default="智能问答助手"),
            ui_static_dir=_env_path("UI_STATIC_DIR", "frontend"),
        )
