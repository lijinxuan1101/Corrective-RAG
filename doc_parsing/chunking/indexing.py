"""
生成向量库的嵌入单元，把「用于嵌入的文本」与「返回给 LLM 的文本」解耦。

一张 20 行的表压成单个向量后，查询"货币资金期末余额"要和其余 19 行抢表达
能力，结果被平均稀释。这里为表格额外拆出行级单元单独嵌入，命中后通过
parent_id 取回整块交给 LLM（即 LangChain 的 MultiVectorRetriever 模式）。
"""
import re
from html.parser import HTMLParser
from typing import List

from langchain_core.documents import Document

from doc_parsing.chunking.splitter import context_prefix

HTML_TABLE = re.compile(r"<table>.*?</table>", re.S)
PIPE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
PIPE_SEPARATOR = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")
MIN_ROW_LEN = 8  # 过短的行（如空行、单个符号）没有检索价值
NUMERIC = re.compile(r"^[\d,.\-+%（）()]*\d[\d,.\-+%（）()]*$")


class _TableParser(HTMLParser):
    """解析 <table>，展开 rowspan/colspan 得到规整网格。"""

    def __init__(self):
        super().__init__()
        self.rows: List[List[tuple[str, int, int]]] = []
        self._row = None
        self._cell = None
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            attr = dict(attrs)
            self._cell = []
            self._span = (int(attr.get("colspan", 1)), int(attr.get("rowspan", 1)))

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(("".join(self._cell).strip(), *self._span))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _expand(parsed: List[List[tuple[str, int, int]]]) -> List[List[str]]:
    """把带 span 的单元格铺成矩形网格，合并单元格的值向右/向下复制。"""
    grid: List[List[str]] = []
    carry: dict[int, tuple[str, int]] = {}  # 列号 -> (值, 剩余行数)
    for raw in parsed:
        row: List[str] = []
        column = 0

        def drain():
            nonlocal column
            while column in carry:
                value, left = carry[column]
                row.append(value)
                if left > 1:
                    carry[column] = (value, left - 1)
                else:
                    del carry[column]
                column += 1

        drain()
        for value, colspan, rowspan in raw:
            for _ in range(colspan):
                row.append(value)
                if rowspan > 1:
                    carry[column] = (value, rowspan - 1)
                column += 1
                drain()
        grid.append(row)

    if not grid:
        return []
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


def _header_depth(parsed: List[List[tuple[str, int, int]]]) -> int:
    """首行若含 rowspan，说明表头跨多行（如"项目"竖跨两行、"2024年末"横跨两列）。"""
    if not parsed:
        return 0
    return max((rowspan for _, _, rowspan in parsed[0]), default=1)


def _has_header(row: List[str]) -> bool:
    """
    约 10% 的表首行其实是数据：键值表（"股票简称|五粮液|股票代码|000858"）或
    表头本就缺失的续表。强行当表头会产出"张庆=男"这种垃圾，因此靠数字识别。
    """
    return not any(NUMERIC.match(cell) for cell in row if cell)


def _merge_header_rows(grid: List[List[str]], depth: int) -> List[str]:
    headers = []
    for column in zip(*grid[:depth]):
        parts = list(dict.fromkeys(part for part in column if part))  # 去重保序
        headers.append(" ".join(parts))
    return headers


def parse_table(html: str) -> tuple[List[str], List[List[str]]]:
    """返回（合并后的列名, 数据行）。列名为空表示该表没有可用表头。"""
    parser = _TableParser()
    parser.feed(html)
    grid = _expand(parser.rows)
    if not grid:
        return [], []
    if not _has_header(grid[0]):
        return [], grid

    depth = max(1, min(_header_depth(parser.rows), len(grid)))
    return _merge_header_rows(grid, depth), grid[depth:]


def parse_pipe_table(text: str) -> tuple[List[str], List[List[str]]]:
    """解析 Markdown 管道表。"""
    lines = [l for l in text.splitlines() if PIPE_ROW.match(l)]
    rows = [[c.strip() for c in PIPE_ROW.match(l).group(1).split("|")]
            for l in lines if not PIPE_SEPARATOR.match(l)]
    if not rows:
        return [], []
    if not _has_header(rows[0]):
        return [], rows
    return rows[0], rows[1:]


def _linearize(headers: List[str], row: List[str]) -> str:
    """行级文本：每个单元格都带上列名，脱离表格后依然可读。"""
    if not headers:
        return " | ".join(dict.fromkeys(cell for cell in row if cell))
    pairs = [f"{head}={value}" if head else value
             for head, value in zip(headers, row) if value]
    return " | ".join(dict.fromkeys(pairs))


def _row_units(content: str) -> List[str]:
    tables = [parse_table(m.group(0)) for m in HTML_TABLE.finditer(content)]
    if not tables:
        tables = [parse_pipe_table(content)]
    units = []
    for headers, rows in tables:
        for row in rows:
            line = _linearize(headers, row)
            if len(line) >= MIN_ROW_LEN:
                units.append(line)
    return units


def build_embedding_units(documents: List[Document]) -> List[Document]:
    """
    为每个块生成嵌入单元。表格块额外产出行级单元；所有单元都带 parent_id，
    检索命中后据此取回完整的父块交给 LLM。
    """
    units: List[Document] = []
    for parent in documents:
        meta = parent.metadata
        base = {**meta, "parent_id": meta["chunk_id"]}
        units.append(Document(page_content=parent.page_content,
                              metadata={**base, "unit_type": "chunk"}))

        prefix = context_prefix(meta)
        for line in _row_units(parent.page_content):
            units.append(Document(page_content=f"{prefix} | {line}",
                                  metadata={**base, "unit_type": "row"}))
    return units
