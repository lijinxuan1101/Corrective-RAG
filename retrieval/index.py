"""
构建并持久化混合索引：Qwen3-Embedding 稠密向量 + BM25 稀疏索引。

稠密向量直接存 numpy——1 万量级的向量点积只要几毫秒，引入向量库反而是负担。
"""
import json
import pickle
from pathlib import Path
from typing import List

import bm25s
import jieba
import numpy as np
from langchain_core.documents import Document

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
INDEX_DIR = Path("index")
BATCH_SIZE = 16
MAX_TOKENS = 4096   # 模型支持 32k，但最长的块只有 3212 token，截短可省显存

# Qwen3-Embedding 对查询侧需要任务指令，文档侧不加
QUERY_INSTRUCT = "Given a question about a Chinese company annual report, retrieve the passage that answers it"


def _device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(model_name: str = EMBED_MODEL):
    import torch
    from sentence_transformers import SentenceTransformer

    device = _device()
    # MPS 上 fp32 比 fp16 慢约 20 倍，而向量归一化后精度差异可以忽略
    encoder = SentenceTransformer(
        model_name, device=device,
        model_kwargs={"dtype": torch.float16 if device == "mps" else torch.float32},
    )
    encoder.max_seq_length = MAX_TOKENS
    return encoder


def tokenize_zh(text: str) -> List[str]:
    """BM25 需要分词；中文没有空格，用 jieba 切。"""
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


def build(units: List[Document], index_dir: Path = INDEX_DIR,
          model_name: str = EMBED_MODEL) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [u.page_content for u in units]

    print(f"🧮 正在编码 {len(texts)} 个单元（{model_name}，{_device()}）...")
    encoder = load_encoder(model_name)
    vectors = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,     # 归一化后点积即余弦相似度
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(index_dir / "dense.npy", vectors)

    print("🔤 正在构建 BM25 索引...")
    retriever = bm25s.BM25()
    retriever.index([tokenize_zh(text) for text in texts])
    with open(index_dir / "bm25.pkl", "wb") as fh:
        pickle.dump(retriever, fh)

    with open(index_dir / "units.jsonl", "w") as fh:
        for unit in units:
            fh.write(json.dumps(
                {"text": unit.page_content, "metadata": unit.metadata},
                ensure_ascii=False) + "\n")

    print(f"✅ 索引已写入 {index_dir}/  "
          f"(稠密 {vectors.shape[0]}×{vectors.shape[1]}, "
          f"{vectors.nbytes / 1024 / 1024:.0f} MB)")


def load_metadata(index_dir: Path = INDEX_DIR) -> List[dict]:
    """只读元数据，不碰向量和模型——判断查询范围之类的轻量场景用这个。"""
    with open(index_dir / "units.jsonl") as fh:
        return [json.loads(line)["metadata"] for line in fh]


def load(index_dir: Path = INDEX_DIR):
    """返回 (稠密矩阵, BM25 检索器, 单元列表)。"""
    vectors = np.load(index_dir / "dense.npy")
    with open(index_dir / "bm25.pkl", "rb") as fh:
        bm25 = pickle.load(fh)
    units = []
    with open(index_dir / "units.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            units.append(Document(page_content=row["text"], metadata=row["metadata"]))
    return vectors, bm25, units
