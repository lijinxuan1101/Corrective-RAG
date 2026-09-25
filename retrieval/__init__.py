"""混合检索：Qwen3-Embedding + BM25 → RRF → Qwen3-Reranker。"""
from retrieval.search import HybridRetriever

__all__ = ["HybridRetriever"]
