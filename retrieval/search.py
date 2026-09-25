"""
混合检索：稠密 + BM25 → RRF 融合 → 归并到父块 → Qwen3-Reranker 重排。

稠密向量擅长语义近似但对精确术语和数字不敏感，BM25 恰好相反——
年报查询同时包含"应收账款坏账准备"这种术语和"127,398,915,484.11"这种数字，
两路缺一不可。
"""
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from langchain_core.documents import Document

from retrieval import index as idx

RRF_K = 60          # RRF 平滑常数，抑制头部排名的过度主导
CANDIDATES = 60     # 每路召回的单元数
RERANK_POOL = 30    # 送进重排器的父块数
TOP_K = 5

Reranker = Callable[[str, Sequence[str]], List[float]]


def default_reranker() -> Reranker:
    """
    重排走百炼 API：20 篇真实父块 1.28s，本地 Qwen3-Reranker 要 3-9s，
    而且省下 1.2GB 常驻内存。

    缺 DASHSCOPE_API_KEY 时**直接报错，不退回本地**——两个模型对无关查询的
    打分差一个量级，静默切换会让 `rag/nodes.py` 的 IRRELEVANT 阈值失去意义。
    这里预先探一次 key，让它在构造检索器时就失败，而不是等到 grade 阶段才炸。
    需要本地重排请显式传入 `HybridRetriever(reranker=...)`。
    """
    from providers import config
    from providers.rerank import score

    config.dashscope_key()   # 缺 key 立即抛出可操作的报错
    return score


def _rrf(*rankings: List[int]) -> dict[int, float]:
    """Reciprocal Rank Fusion：只用名次，避免两路分数量纲不可比。"""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (RRF_K + rank)
    return fused


class HybridRetriever:
    def __init__(self, index_dir: Path = idx.INDEX_DIR,
                 parents: List[Document] | None = None,
                 reranker: Reranker | None = None):
        self.vectors, self.bm25, self.units = idx.load(index_dir)
        self.encoder = idx.load_encoder()
        self.reranker = reranker or default_reranker()
        # 命中的是行级单元，但交给 LLM 的必须是完整父块
        if parents is None:
            from doc_parsing import load_all
            parents = load_all()
        self.parents = {p.metadata["chunk_id"]: p for p in parents}

    def _mask(self, filters: dict | None) -> np.ndarray | None:
        """按元数据筛出可参与检索的单元下标。"""
        if not filters:
            return None
        keep = [i for i, u in enumerate(self.units)
                if all(str(u.metadata.get(k)) == str(v) for k, v in filters.items())]
        return np.array(keep, dtype=np.int64)

    def _dense(self, query: str, k: int, allowed: np.ndarray | None = None) -> List[int]:
        vector = self.encoder.encode(
            [query], prompt=f"Instruct: {idx.QUERY_INSTRUCT}\nQuery: ",
            normalize_embeddings=True, convert_to_numpy=True,
        )[0].astype(np.float32)
        scores = self.vectors @ vector
        if allowed is None:
            return np.argsort(-scores)[:k].tolist()
        return allowed[np.argsort(-scores[allowed])[:k]].tolist()

    def _sparse(self, query: str, k: int, allowed: np.ndarray | None = None) -> List[int]:
        scores = self.bm25.get_scores(idx.tokenize_zh(query))
        pool = np.arange(len(scores)) if allowed is None else allowed
        return [int(i) for i in pool[np.argsort(-scores[pool])[:k]] if scores[i] > 0]

    def search(self, query: str, top_k: int = TOP_K, use_reranker: bool = True,
               filters: dict | None = None) -> List[Document]:
        """filters 按元数据精确匹配，例如 {"company": "五粮液", "year": "2024"}。"""
        allowed = self._mask(filters)
        fused = _rrf(self._dense(query, CANDIDATES, allowed),
                     self._sparse(query, CANDIDATES, allowed))

        # 同一父块可能被多个行级单元命中，取其最高融合分
        best: dict[str, float] = {}
        for unit_id, score in fused.items():
            parent_id = self.units[unit_id].metadata["parent_id"]
            best[parent_id] = max(best.get(parent_id, 0.0), score)

        ranked = sorted(best, key=best.get, reverse=True)
        if not use_reranker:
            return [self._tag(pid) for pid in ranked[:top_k] if pid in self.parents]

        pool = [pid for pid in ranked[:RERANK_POOL] if pid in self.parents]
        scores = self.reranker(query, [self.parents[pid].page_content for pid in pool])
        order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        return [self._tag(pool[i], scores[i]) for i in order[:top_k]]

    def _tag(self, parent_id: str, score: float | None = None) -> Document:
        """带上重排分数返回副本——直接改 self.parents 会污染后续查询。"""
        parent = self.parents[parent_id]
        return Document(page_content=parent.page_content,
                        metadata={**parent.metadata, "rerank_score": score})
