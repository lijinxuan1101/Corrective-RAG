#!/usr/bin/env python
"""
构建混合索引。

用法:
    python retrieval/build_index.py
    python retrieval/build_index.py --model Qwen/Qwen3-Embedding-4B
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_parsing import build_embedding_units, load_all  # noqa: E402
from retrieval import index  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=index.EMBED_MODEL)
    parser.add_argument("--markdown-dir", default="data/markdown")
    parser.add_argument("--index-dir", default=str(index.INDEX_DIR))
    args = parser.parse_args()

    parents = load_all(args.markdown_dir)
    if not parents:
        print(f"❌ 在 '{args.markdown_dir}' 中没有加载到内容。")
        print("   💡 请先运行 ./doc_parsing/pdf_to_markdown/convert.sh")
        raise SystemExit(1)

    units = build_embedding_units(parents)
    print(f"📦 父块 {len(parents)} → 嵌入单元 {len(units)}")
    index.build(units, Path(args.index_dir), args.model)


if __name__ == "__main__":
    main()
