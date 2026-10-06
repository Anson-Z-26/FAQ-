"""Embedding 客户端。

这里保持函数式而不是包成类：没有任何需要跨调用保留的状态，
参数全部来自 Settings，测试时传一个 Settings(...) 就能替换掉真实接口。
"""
import numpy as np
import requests

from .config import Settings

DEFAULT_TIMEOUT = 30


def _embed_batch(texts, cfg, timeout):
    """把一批文本（不超过单次上限）发给接口，返回 shape=(len(texts), dim)。"""
    try:
        resp = requests.post(
            cfg.embedding_url,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.embedding_model,
                "input": texts,
                # 显式指定维度，避免与 EMBEDDING_DIM 不一致导致索引写入时报错
                "dimensions": cfg.embedding_dim,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Embedding接口请求失败: {e}") from e

    try:
        # 接口返回的 data 不保证与入参顺序一致，按 index 排回去
        items = sorted(data["data"], key=lambda item: item.get("index", 0))
        vectors = np.array([item["embedding"] for item in items], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"Embedding接口返回格式异常: {data!r}") from e

    expected = (len(texts), cfg.embedding_dim)
    if vectors.shape != expected:
        raise RuntimeError(
            f"Embedding维度不符：期望 {expected}，实际 {vectors.shape}。"
            f"请检查 .env 中的 EMBEDDING_DIM 是否与模型输出一致"
        )
    return vectors


def get_embedding(text: str, *, settings: Settings = None, timeout: int = DEFAULT_TIMEOUT) -> np.ndarray:
    """单条文本向量化，返回 shape=(dim,) 的 float32 向量。"""
    return get_embeddings([text], settings=settings, timeout=timeout)[0]


def get_embeddings(texts, *, settings: Settings = None, timeout: int = DEFAULT_TIMEOUT) -> np.ndarray:
    """批量向量化，返回 shape=(len(texts), dim) 的 float32 数组。

    接口对单次 input 条数有上限（本平台为 10 条，超过报 400
    "batch size is invalid, it should not be larger than 10"），
    所以这里按 cfg.max_batch_size 自动切片、分批请求，最后按原顺序拼接。
    """
    cfg = settings or Settings.from_env()
    if not cfg.embedding_url:
        raise RuntimeError("未配置 embedding 接口地址，请检查 .env 中的 EMBEDDING_URL")
    if not cfg.api_key:
        raise RuntimeError("未配置 API Key，请检查 .env 中的 DASHSCOPE_API_KEY")

    texts = list(texts)
    if not texts:
        return np.zeros((0, cfg.embedding_dim), dtype=np.float32)

    batch_size = max(1, int(getattr(cfg, "max_batch_size", 10)))

    # 未超过上限时走原路径，保持一次请求
    if len(texts) <= batch_size:
        return _embed_batch(texts, cfg, timeout)

    chunks = [
        _embed_batch(texts[i:i + batch_size], cfg, timeout)
        for i in range(0, len(texts), batch_size)
    ]
    return np.concatenate(chunks, axis=0)
