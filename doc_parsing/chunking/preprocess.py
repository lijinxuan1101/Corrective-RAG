"""
清洗 MinerU 输出的 Markdown：去页眉页脚、还原勾选框、合并跨页表格。
"""
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

PAGE_MARKER = re.compile(r"<!-- page (\d+) of (\d+) -->")
IMAGE_BLOCK = re.compile(r"!\[Image block\]\([^)]*\)")
TABLE_SEPARATOR = re.compile(r"^\|(\s*-{3,}\s*\|)+\s*$")

HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
# MinerU 把所有标题一律输出成 ##，层级全丢。年报遵循中文公文编号体例，
# 据此还原深度，否则"（2）按坏账计提方法分类披露"这类标题会在同一篇里
# 重复十几次，分属应收票据/应收账款/其他应收款却无法区分。
HEADING_LEVELS = (
    (re.compile(r"^第[一二三四五六七八九十百]+节"), 2),
    (re.compile(r"^[一二三四五六七八九十]+[、．.]"), 3),
    (re.compile(r"^[（(]\s*[一二三四五六七八九十]+\s*[)）]"), 4),
    (re.compile(r"^\d+[、．]"), 5),
    (re.compile(r"^[（(]\s*\d+\s*[)）]"), 6),
    (re.compile(r"^\d+\s*[)）]"), 6),
    (re.compile(r"^[①-⑳]"), 6),
)
# "□适用 ☑不适用"是正文里的勾选项，被 MinerU 误判成了标题
FALSE_HEADING = re.compile(r"^[□☑☐]")

# MinerU 从 Wingdings 字体透传出来的私有区字符，直接喂给 LLM 是乱码。
# 年报里大量的"□适用 ☑不适用"靠它区分选中项，必须还原。
PUA_REPLACEMENTS = {
    "\uf052": "☑",
    "\uf0a3": "☐",
    "\uf06f": "☐",
}


@dataclass
class Page:
    number: int
    lines: List[str]


def _split_pages(text: str) -> List[Page]:
    parts = PAGE_MARKER.split(text)
    # split 后形如 [前言, 页码, 总页数, 正文, 页码, 总页数, 正文, ...]
    pages = []
    for i in range(1, len(parts) - 2, 3):
        pages.append(Page(number=int(parts[i]), lines=parts[i + 2].splitlines()))
    return pages


def _detect_header(pages: List[Page]) -> str:
    """页眉是绝大多数页面重复的首行，各家公司文案不同，因此按频次推断。"""
    firsts = Counter()
    for page in pages:
        for line in page.lines:
            if line.strip():
                firsts[line.strip()] += 1
                break
    if not firsts:
        return ""
    candidate, count = firsts.most_common(1)[0]
    return candidate if count >= len(pages) * 0.5 else ""


def _strip_noise(page: Page, header: str) -> None:
    lines = [line.rstrip() for line in page.lines]

    while lines and not lines[0].strip():
        lines.pop(0)
    if header and lines and lines[0].strip() == header:
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()
    # 页脚是孤立的页码数字
    if lines and re.fullmatch(r"\d+", lines[-1].strip()):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    page.lines = lines


def _merge_cross_page_tables(pages: List[Page]) -> int:
    """
    表格跨页时 MinerU 会把续表当成新表：首行是数据行，紧跟着一条多余的
    分隔行，导致该块丢失原表头。这里删掉分隔行，让续表并回上一页的表尾。
    """
    merged = 0
    for prev, cur in zip(pages, pages[1:]):
        if not prev.lines or not cur.lines:
            continue
        if not prev.lines[-1].lstrip().startswith("|"):
            continue
        if not cur.lines[0].lstrip().startswith("|"):
            continue
        if len(cur.lines) > 1 and TABLE_SEPARATOR.match(cur.lines[1].strip()):
            del cur.lines[1]
            merged += 1
    return merged


def _normalize_headings(pages: List[Page]) -> int:
    """按中文编号体例重写 # 的数量，让标题深度反映真实层级。"""
    rewritten = 0
    numbered_depth = 1  # 最近一个带编号标题的深度，用于安置无编号标题
    for page in pages:
        result = []
        for line in page.lines:
            match = HEADING.match(line)
            if not match or len(match.group(1)) == 1:  # 文档标题保持 h1
                result.append(line)
                continue
            title = match.group(2)
            if FALSE_HEADING.match(title):
                result.append(title)  # 勾选项不是标题，降级为正文
                rewritten += 1
                continue
            depth = next((d for pattern, d in HEADING_LEVELS if pattern.match(title)), None)
            if depth is None:
                # 无编号标题挂在最近的编号标题之下；同级兄弟因此获得相同深度
                depth = min(numbered_depth + 1, 6)
            else:
                numbered_depth = depth
            result.append(f"{'#' * depth} {title}")
            rewritten += depth != len(match.group(1))
        page.lines = result
    return rewritten


def clean_markdown(text: str) -> tuple[str, dict]:
    """返回清洗后的 Markdown 与统计信息。页码标记保留，供切分阶段提取元数据。"""
    for pua, replacement in PUA_REPLACEMENTS.items():
        text = text.replace(pua, replacement)
    text, n_images = IMAGE_BLOCK.subn("", text)

    pages = _split_pages(text)
    if not pages:
        return text, {"pages": 0}

    header = _detect_header(pages)
    for page in pages:
        _strip_noise(page, header)
    n_merged = _merge_cross_page_tables(pages)

    n_headings = _normalize_headings(pages)

    total = PAGE_MARKER.search(text).group(2)
    chunks = []
    for page in pages:
        body = "\n".join(page.lines).strip()
        if body:
            chunks.append(f"<!-- page {page.number} of {total} -->\n{body}")

    stats = {
        "pages": len(pages),
        "header": header,
        "merged_tables": n_merged,
        "images_removed": n_images,
        "headings_releveled": n_headings,
    }
    return "\n\n".join(chunks) + "\n", stats
