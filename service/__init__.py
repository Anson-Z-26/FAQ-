"""service 包对外出口。调用方只需要 `from service import RAGService`。"""
from .config import Settings
from .embedding import get_embedding, get_embeddings
from .llm import ChatLLM, LLM, NoopLLM, build_llm
from .rag import RAGService, load_knowledge
from .rerank import DashScopeReranker, NoopReranker, Reranker, build_reranker
from .vector_index import VectorIndex

__all__ = [
    "Settings",
    "RAGService",
    "VectorIndex",
    "load_knowledge",
    "get_embedding",
    "get_embeddings",
    "Reranker",
    "NoopReranker",
    "DashScopeReranker",
    "build_reranker",
    "LLM",
    "NoopLLM",
    "ChatLLM",
    "build_llm",
]
