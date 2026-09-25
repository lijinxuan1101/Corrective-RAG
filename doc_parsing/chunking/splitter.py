"""
按 Markdown 标题层级切分年报，并为每个块附上公司、年份、页码和章节路径。
"""
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from doc_parsing.chunking.preprocess import PAGE_MARKER, TABLE_SEPARATOR, clean_markdown

HEADERS = [("#" * n, f"h{n}") for n in range(1, 7)]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
ORPHAN_TEXT = 300  # 短于此长度的正文片段并入相邻表格，避免产生无检索价值的碎块

FILENAME = re.compile(r"^(?P<company>.+?)(?P<year>\d{4})年报$")


def _parse_filename(path: Path) -> dict:
    match = FILENAME.match(path.stem)
    if not match:
        return {"company": path.stem, "year": ""}
    return {"company": match.group("company"), "year": match.group("year")}


def _extract_pages(text: str) -> tuple[str, int | None, int | None]:
    """取出块内所有页码标记，返回去掉标记的正文与页码范围。"""
    pages = [int(m.group(1)) for m in PAGE_MARKER.finditer(text)]
    text = PAGE_MARKER.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 标记原先夹在跨页表格中间，移除后留下的空行会把表体与表头割裂
    text = re.sub(r"(\|[^\n]*\|)[ \t]*\n\s*\n(?=[ \t]*\|)", r"\1\n", text).strip()
    if not pages:
        return text, None, None
    return text, min(pages), max(pages)


def _coalesce(sections: List[Document]) -> List[Document]:
    """
    年报里大量小节只有一行标题或一句"□适用 ☑不适用"，逐节成块会产生海量碎片。
    这里把相邻小节合并到接近 CHUNK_SIZE，元数据沿用其中第一节的章节路径。
    """
    merged: List[Document] = []
    for section in sections:
        if merged and len(merged[-1].page_content) + len(section.page_content) < CHUNK_SIZE:
            merged[-1].page_content += "\n\n" + section.page_content
        else:
            merged.append(Document(page_content=section.page_content,
                                   metadata=dict(section.metadata)))
    return merged


HTML_TABLE = re.compile(r"<table>.*?</table>", re.S)


def _segment(text: str) -> List[tuple[bool, str]]:
    """
    把小节拆成「HTML 表格」与「普通正文」两类片段。
    带合并单元格的表格是 HTML，按字符切会从标签中间断开；最大的表也只有约
    4.8k 字符，远低于 embedding 上限，因此整块保留、单独成块。
    """
    segments: List[tuple[bool, str]] = []
    cursor = 0
    for match in HTML_TABLE.finditer(text):
        if body := text[cursor:match.start()].strip():
            segments.append((False, body))
        segments.append((True, match.group(0)))
        cursor = match.end()
    if body := text[cursor:].strip():
        segments.append((False, body))

    # 表格前后常是"单位:元""□适用 ☑不适用"这类短句，单独成块毫无检索价值，
    # 并入相邻表格既消除碎片，又保留了表格的量纲等说明。
    merged: List[tuple[bool, str]] = []
    for is_table, body in segments:
        if (not is_table and len(body) < ORPHAN_TEXT and merged and merged[-1][0]):
            merged[-1] = (True, f"{merged[-1][1]}\n\n{body}")
        elif (is_table and merged and not merged[-1][0]
              and len(merged[-1][1]) < ORPHAN_TEXT):
            merged[-1] = (True, f"{merged[-1][1]}\n\n{body}")
        else:
            merged.append((is_table, body))
    return merged


def _find_table_header(lines: List[str]) -> str | None:
    for i in range(len(lines) - 1):
        if lines[i].lstrip().startswith("|") and TABLE_SEPARATOR.match(lines[i + 1].strip()):
            return f"{lines[i]}\n{lines[i + 1]}"
    return None


def _carry_table_header(piece: str, header_rows: str | None) -> tuple[str, str | None]:
    """
    长表格被二次切分后，续块只剩数据行。补回表头，否则检索命中时无法判断列含义。
    """
    lines = piece.splitlines()
    found = _find_table_header(lines)
    if found:
        return piece, found
    first = next((l for l in lines if l.strip()), "")
    if header_rows and first.lstrip().startswith("|"):
        return f"{header_rows}\n{piece}", header_rows
    return piece, header_rows


def context_prefix(meta: dict) -> str:
    """公司 + 年份 + 章节路径，用于让块脱离原文后仍能自我说明。"""
    prefix = f"{meta['company']} {meta['year']}年报"
    if meta["section"]:
        prefix += f" | {meta['section']}"
    return prefix


def _contextualize(content: str, meta: dict) -> str:
    """
    向量库只嵌入 page_content，元数据字段不参与检索。年报里指代密度极高
    （"本公司""报告期内""期末余额"），块脱离上下文后无从判断归属。
    """
    return f"{context_prefix(meta)}\n\n{content}"


def split_document(path: Path, contextualize: bool = True) -> List[Document]:
    cleaned, _ = clean_markdown(path.read_text())
    base = _parse_filename(path) | {"source": path.name}

    # 先按标题切，保证每块落在单一章节内；strip_headers=False 让标题留在正文里，
    # 检索命中后 LLM 能直接看到"这段属于哪张报表"。
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    sections = _coalesce(header_splitter.split_text(cleaned))

    # 财务报表章节远超 embedding 上限，需要二次切分。
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    documents: List[Document] = []
    for section in sections:
        # h1 是文档标题，与 company/year 重复，不计入章节路径
        breadcrumb = " > ".join(
            section.metadata[key] for _, key in HEADERS[1:] if key in section.metadata
        )
        header_rows = None
        pieces: List[str] = []
        for is_table, segment in _segment(section.page_content):
            if is_table:
                pieces.append(segment)
            else:
                pieces.extend(body_splitter.split_text(segment))

        for piece in pieces:
            piece, header_rows = _carry_table_header(piece, header_rows)
            content, first, last = _extract_pages(piece)
            if not content:
                continue
            # 页码标记只出现在跨页处，块内无标记时沿用上一块的页码
            if first is None and documents:
                first = last = documents[-1].metadata.get("page_end")
            meta = {**base, "section": breadcrumb,
                    "page_start": first, "page_end": last,
                    "chunk_id": f"{path.stem}#{len(documents)}"}
            documents.append(
                Document(
                    page_content=_contextualize(content, meta) if contextualize else content,
                    metadata=meta,
                )
            )
    return documents


def load_all(markdown_dir: str = "data/markdown",
             contextualize: bool = True) -> List[Document]:
    documents = []
    for path in sorted(Path(markdown_dir).glob("*.md")):
        documents.extend(split_document(path, contextualize))
    return documents
