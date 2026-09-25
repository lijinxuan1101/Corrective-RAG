#!/usr/bin/env python
"""
检视切块质量：打印统计指标，并按关键词抽样查看块内容与元数据。

用法:
    python doc_parsing/chunking/inspect_chunks.py            # 只看统计
    python doc_parsing/chunking/inspect_chunks.py 货币资金     # 搜关键词
    python doc_parsing/chunking/inspect_chunks.py 研发投入 -n 3
"""
import argparse
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from doc_parsing.chunking import clean_markdown, load_all  # noqa: E402


def report(documents):
    lengths = sorted(len(d.page_content) for d in documents)
    print(f"📦 总块数 {len(documents)}")
    print(f"   长度 中位数={statistics.median(lengths):.0f} "
          f"均值={statistics.mean(lengths):.0f} 最大={max(lengths)}")

    for threshold in (30, 100, 200):
        n = sum(1 for length in lengths if length < threshold)
        print(f"   <{threshold:>4} 字符: {n:>5} 块 ({n / len(lengths) * 100:.1f}%)")

    print("   按文档:")
    for source, count in sorted(Counter(d.metadata["source"] for d in documents).items()):
        print(f"     {source:<24} {count:>5} 块")


PUA = re.compile(r"[\ue000-\uf8ff]")
MAX_EMBED = 8000        # 超过此长度有顶爆 embedding 上限的风险
FLAT_RATIO = 0.9        # 单一深度占比超过此值，说明标题层级没被还原


def _residual_pua(markdown_dir: str) -> Counter:
    """
    统计清洗后仍残留的私有区字符。必须在切分前查：MarkdownHeaderTextSplitter
    会静默吞掉这些字符，未收录的勾选框会让"☑适用 □不适用"变成"适用 不适用"，
    选中项彻底丢失且不留痕迹。
    """
    found = Counter()
    for path in sorted(Path(markdown_dir).glob("*.md")):
        cleaned, _ = clean_markdown(path.read_text())
        found.update(PUA.findall(cleaned))
    return found


def health(documents, markdown_dir):
    """
    把踩过的坑固化成体检项。这几类问题都会静默发生——不报错，只是悄悄
    产出更差的切块，换数据源时尤其容易中招。
    """
    total = len(documents)
    pua = _residual_pua(markdown_dir)
    depths = Counter(
        d.metadata["section"].count(" > ") + 1 if d.metadata["section"] else 0
        for d in documents
    )
    _, top = depths.most_common(1)[0]

    checks = [
        ("缺页码元数据", "块",
         sum(1 for d in documents if d.metadata["page_start"] is None),
         "页码标记被清洗掉了，答案无法标注出处"),
        ("表格缺表头", "块",
         sum(1 for d in documents
             if d.page_content.lstrip().startswith("|") and "| --- |" not in d.page_content),
         "表格续块只剩数据行，无从判断列含义"),
        ("元数据解析失败", "块",
         sum(1 for d in documents if not d.metadata["year"].isdigit()),
         "文件名不匹配 `公司+年份+年报`，跨年份过滤会失效"),
        ("私有区字符残留", "处",
         sum(pua.values()),
         f"{'/'.join(f'U+{ord(c):04X}' for c in pua)} 未映射，"
         "切分时会被静默删除，勾选项语义丢失（见 PUA_REPLACEMENTS）"),
        ("超长块", "块",
         sum(1 for d in documents if len(d.page_content) > MAX_EMBED),
         f"超过 {MAX_EMBED} 字符，可能顶爆 embedding 上限"),
        ("标题层级扁平", "块",
         total if top / total > FLAT_RATIO else 0,
         "编号体例未匹配，章节路径无法区分同名小节（见 HEADING_LEVELS）"),
    ]

    print("\n🩺 体检")
    for label, unit, count, hint in checks:
        if count:
            print(f"   ⚠️  {label}: {count} {unit} — {hint}")
        else:
            print(f"   ✅ {label}")
    print(f"   章节路径深度分布: {dict(sorted(depths.items()))}")


def show(documents, keyword, limit):
    hits = [d for d in documents if keyword in d.page_content]
    print(f"\n🔍 '{keyword}' 命中 {len(hits)} 块，显示前 {min(limit, len(hits))} 个：")
    for doc in hits[:limit]:
        meta = doc.metadata
        print(f"\n{'─' * 70}")
        print(f"[{meta['company']} {meta['year']}] p{meta['page_start']}-{meta['page_end']}")
        print(f"章节: {meta['section']}")
        print(f"{'─' * 70}")
        print(doc.page_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyword", nargs="?", help="按关键词抽样查看")
    parser.add_argument("-n", type=int, default=2, help="显示条数")
    parser.add_argument("--dir", default="data/markdown")
    args = parser.parse_args()

    documents = load_all(args.dir)
    if not documents:
        print(f"❌ 在 '{args.dir}' 中没有加载到任何内容。")
        print("   💡 请先运行 ./doc_parsing/pdf_to_markdown/convert.sh 生成 Markdown。")
        raise SystemExit(1)

    report(documents)
    health(documents, args.dir)
    if args.keyword:
        show(documents, args.keyword, args.n)


if __name__ == "__main__":
    main()
