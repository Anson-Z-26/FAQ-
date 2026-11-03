"""service 包对外出口。调用方只需要 `from service import RAGService`。"""
from .config import Settings
from .embedding import get_embedding, get_embeddings
from .llm import ChatLLM, LLM, NoopLLM, build_llm
from .manifest import IndexManifest, build_manifest, entry_fingerprint
from .rag import RAGService, index_by_id, knowledge_aliases, load_knowledge
from .rerank import DashScopeReranker, NoopReranker, Reranker, build_reranker
from .session import Session, SessionStore, Turn
from .vector_index import VectorIndex

__all__ = [
    "Settings",
    "RAGService",
    "VectorIndex",
    "load_knowledge",
    "knowledge_aliases",
    "index_by_id",
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
    "Session",
    "SessionStore",
    "Turn",
    "IndexManifest",
    "build_manifest",
    "entry_fingerprint",
]
