"""
阿里云百炼（DashScope）文本嵌入。

接口本身一次最多 25 条、且偶发限流，这里把分批、重试、并发都包掉，
调用方只管传一个列表进来。
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Sequence

import dashscope
import numpy as np

from providers import config

MODEL = "qwen3.7-text-embedding-flash"
DIMENSION = 1024
BATCH_SIZE = 25      # 接口硬上限，26 条即报 InvalidParameter
WORKERS = 6
MAX_RETRIES = 4

TextType = Literal["query", "document"]

dashscope.api_key = config.dashscope_key()


def _call(texts: Sequence[str], text_type: TextType, model: str,
          dimension: int) -> np.ndarray:
    for attempt in range(MAX_RETRIES):
        response = dashscope.TextEmbedding.call(
            model=model, input=list(texts),
            text_type=text_type, dimension=dimension,
        )
        if response.status_code == 200:
            # 接口不保证返回顺序，必须按 index 还原
            rows = sorted(response.output["embeddings"], key=lambda e: e["index"])
            return np.array([r["embedding"] for r in rows], dtype=np.float32)

        # 4xx 是请求本身的问题，重试没有意义
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise RuntimeError(f"DashScope 嵌入失败 [{response.code}]: {response.message}")
        time.sleep(2 ** attempt)

    raise RuntimeError(f"DashScope 嵌入重试 {MAX_RETRIES} 次仍失败：{response.message}")


def embed(texts: Sequence[str], text_type: TextType = "document",
          model: str = MODEL, dimension: int = DIMENSION,
          normalize: bool = True, workers: int = WORKERS) -> np.ndarray:
    """
    把文本批量转成向量，返回形状 (len(texts), dimension) 的矩阵。

    text_type 必须区分 query 和 document——模型是非对称训练的，
    查询侧和文档侧用同一个 type 会损失召回质量。
    """
    if not texts:
        return np.empty((0, dimension), dtype=np.float32)

    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(lambda b: _call(b, text_type, model, dimension), batches))

    vectors = np.vstack(parts)
    if normalize:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def embed_query(text: str, **kwargs) -> np.ndarray:
    """单条查询的便捷入口，返回一维向量。"""
    return embed([text], text_type="query", **kwargs)[0]
