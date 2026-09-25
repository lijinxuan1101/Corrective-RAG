"""
阿里云百炼文本重排。

和 retrieval.rerank（本地 Qwen3-Reranker）保持同样的 score() 签名，两者可以直接互换。
"""
import time
from typing import List, Sequence

import requests

from providers import config

URL = "https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank"
MODEL = "qwen3.7-text-rerank"
BATCH_SIZE = 50      # 实测 100 篇也能过，取一半留余量
TIMEOUT = 120
MAX_RETRIES = 4


def _call(query: str, documents: Sequence[str], model: str) -> List[float]:
    payload = {
        "model": model,
        "input": {"query": query, "documents": list(documents)},
        "parameters": {"top_n": len(documents), "return_documents": False},
    }
    headers = {"Authorization": f"Bearer {config.dashscope_key()}",
               "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        response = requests.post(URL, headers=headers, json=payload, timeout=TIMEOUT)
        if response.status_code == 200:
            # 返回结果按分数降序，必须按 index 还原成输入顺序
            scores = [0.0] * len(documents)
            for item in response.json()["output"]["results"]:
                scores[item["index"]] = item["relevance_score"]
            return scores

        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise RuntimeError(f"重排失败 [{response.status_code}]: {response.text[:200]}")
        time.sleep(2 ** attempt)

    raise RuntimeError(f"重排重试 {MAX_RETRIES} 次仍失败: {response.text[:200]}")


def score(query: str, documents: Sequence[str], model: str = MODEL) -> List[float]:
    """
    返回每篇文档与查询的相关性分数，顺序与输入一致。

    交叉编码器对每个 (query, document) 对独立打分，
    所以超过单次上限时分批不会影响分数的可比性。
    """
    if not documents:
        return []

    scores: List[float] = []
    for start in range(0, len(documents), BATCH_SIZE):
        scores.extend(_call(query, documents[start:start + BATCH_SIZE], model))
    return scores
