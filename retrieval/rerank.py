"""
Qwen3-Reranker 交叉编码重排。

它不是常规的 cross-encoder，而是让因果语言模型回答 yes/no，
取这两个 token 的 logits 做 softmax 得到相关性分数。
"""
from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
MAX_LENGTH = 4096
DEFAULT_TASK = "判断该年报片段能否回答用户的问题"

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements "
          "based on the Query and the Instruct provided. Note that the answer can "
          'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@lru_cache(maxsize=1)
def _load(model_name: str = RERANK_MODEL):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device).eval()
    yes = tokenizer.convert_tokens_to_ids("yes")
    no = tokenizer.convert_tokens_to_ids("no")
    return tokenizer, model, device, yes, no


def score(query: str, documents: List[str], task: str = DEFAULT_TASK,
          batch_size: int = 8, model_name: str = RERANK_MODEL) -> List[float]:
    """返回每个文档与查询的相关性分数（0~1）。"""
    if not documents:
        return []
    tokenizer, model, device, yes_id, no_id = _load(model_name)

    prompts = [
        f"{PREFIX}<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}{SUFFIX}"
        for doc in documents
    ]

    # 按长度排序再分批：一批要 padding 到最长的那条，长短混批会浪费大量算力
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

    scores = [0.0] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        batch = tokenizer(
            [prompts[i] for i in chunk],
            padding=True, truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :]
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        for i, value in zip(chunk, torch.softmax(pair, dim=1)[:, 1].tolist()):
            scores[i] = value
    return scores
