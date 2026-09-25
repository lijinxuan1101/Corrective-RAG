"""
图的各个节点。

节点只负责编排，检索交给 retrieval.HybridRetriever，模型调用交给 chains。
"""
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from rag.chains import document_grader, format_context, generator, query_analyzer
from rag.state import GraphState

TOP_K = 5

# 重排分数低于此值时，文档与问题毫无关系，不必再花 LLM 调用去评分。
# 按百炼重排 API 实测标定：语料内可答 0.936~0.999，完全无关 0.0038~0.0488。
# 换重排模型必须重新标定——本地 Qwen3-Reranker 的无关区间是 0.0003~0.0025，低一个量级。
IRRELEVANT = 0.1


@lru_cache(maxsize=1)
def get_retriever():
    from retrieval import HybridRetriever
    return HybridRetriever()


@lru_cache(maxsize=1)
def known_pairs() -> frozenset:
    """
    索引里实际存在的 (公司, 年份) 组合。

    存组合而不是两个独立集合：语料不齐时（比如只有五粮液 2024 和宁德时代 2025），
    分开校验会让"五粮液 2025"这种不存在的组合蒙混过关。

    只读元数据，不走 get_retriever()——越界的提问不该为了查范围
    先把嵌入模型和整个索引加载一遍。
    """
    from retrieval.index import load_metadata
    return frozenset((m.get("company"), str(m.get("year"))) for m in load_metadata())


def corpus_summary() -> str:
    """语料覆盖范围，用于越界时给出有用的答复而不是干巴巴的拒绝。"""
    by_company: dict[str, list[str]] = {}
    for company, year in known_pairs():
        by_company.setdefault(company, []).append(year)
    return "、".join(f"{c} {'/'.join(sorted(ys))}年报"
                     for c, ys in sorted(by_company.items()))


def check_scope(company: str | None, year: str | None) -> str | None:
    """越界时返回说明文字，在范围内返回 None。"""
    pairs = known_pairs()
    companies = {c for c, _ in pairs}
    years = {y for _, y in pairs}

    if company and company not in companies:
        return f"语料中没有{company}的年报"
    if year and year not in years:
        return f"语料中没有 {year} 年的数据"
    if company and year and (company, year) not in pairs:
        return f"语料中没有{company} {year} 年报"
    return None


# ---------------------------------------------------------------
def analyze(state: GraphState) -> dict:
    """补全指代，抽出公司和年份作为检索过滤条件。"""
    print("---ANALYZE: 分析查询---")
    question = state["question"]
    history = "\n".join(state.get("chat_history", [])) or "（无）"

    try:
        result = query_analyzer.invoke({"question": question, "chat_history": history})
    except Exception as exc:
        # 查询分析出问题不该让整轮问答失败，退回用原始提问直接检索
        print(f"   ⚠️ 分析失败，改用原始提问: {type(exc).__name__}")
        return {"query": question, "filters": {}, "scope": "in_corpus", "documents": []}

    company = result.company
    year = str(result.year) if result.year else None
    query = result.query or question

    if query != question:
        print(f"   ✨ 重写为: {query}")

    reason = check_scope(company, year)
    if reason:
        print(f"   🚫 超出语料范围：{reason}")
        return {"query": query, "filters": {}, "documents": [],
                "scope": "out_of_corpus", "out_of_scope_reason": reason}

    filters = {k: v for k, v in (("company", company), ("year", year)) if v}
    if filters:
        print(f"   🔖 过滤条件: {filters}")

    return {"query": query, "filters": filters, "scope": "in_corpus", "documents": []}


def retrieve(state: GraphState) -> dict:
    """混合检索 + 重排，返回带元数据的父块。"""
    print("---RETRIEVE: 检索本地知识库---")
    docs = get_retriever().search(
        state["query"], top_k=TOP_K, filters=state.get("filters") or None)

    confidence = max((d.metadata.get("rerank_score") or 0.0 for d in docs), default=0.0)
    print(f"   📄 召回 {len(docs)} 篇，最高重排分 {confidence:.4f}")
    return {"documents": docs, "confidence": confidence}


def grade(state: GraphState) -> dict:
    """
    CRAG 的检索评估器，决定走 correct / ambiguous / incorrect 三条路径之一。

    重排分数只能判断话题相关性，判断不了事实可答性——
    "茅台的营业收入"配上五粮液的收入表能拿到 0.916 的高分。
    所以低分靠分数直接拦掉，高分还要过一遍 LLM 做公司和年份的核对。
    """
    print("---GRADE: 评估检索质量---")
    # 用重写后的查询而非原始提问：追问"那2024年的呢？"本身不含主语，
    # 拿它去评分等于让模型盲判
    documents, question = state["documents"], state["query"]

    if not documents or state["confidence"] < IRRELEVANT:
        print(f"   ❌ 无相关文档（最高分 {state.get('confidence', 0):.4f}）")
        return {"documents": [], "action": "incorrect"}

    def check(doc: Document) -> bool:
        meta = doc.metadata
        source = f"{meta.get('company')} {meta.get('year')}年报 · {meta.get('section')}"
        try:
            return document_grader.invoke(
                {"question": question, "document": doc.page_content, "source": source}
            ).relevant
        except Exception as exc:
            # 评不出来就当不相关：宁可少给上下文，也不能放没核验过的数据进生成
            print(f"   ⚠️ 评分失败，按不相关处理: {type(exc).__name__}")
            return False

    with ThreadPoolExecutor(max_workers=len(documents)) as pool:
        verdicts = list(pool.map(check, documents))

    kept = [d for d, ok in zip(documents, verdicts) if ok]
    print(f"   ✅ {len(kept)}/{len(documents)} 篇通过")

    if not kept:
        action = "incorrect"
    elif len(kept) < len(documents):
        action = "ambiguous"     # 部分可用，补一次联网搜索
    else:
        action = "correct"
    return {"documents": kept, "action": action}


def web_search(state: GraphState) -> dict:
    """本地知识库不足时补充联网结果。缺 key 时降级而不是崩溃。"""
    print("---WEB SEARCH: 联网补充---")
    documents = list(state["documents"])

    if not os.getenv("TAVILY_API_KEY"):
        print("   ⚠️ 未配置 TAVILY_API_KEY，跳过联网搜索")
        return {"documents": documents}

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        results = TavilySearchResults(k=3).invoke({"query": state["query"]})
        for item in results:
            documents.append(Document(
                page_content=item["content"],
                metadata={"source_label": f"网络检索 · {item.get('url', '')}"},
            ))
        print(f"   🌐 补充 {len(results)} 条网络结果")
    except Exception as exc:
        print(f"   ⚠️ 联网搜索失败，继续用现有文档: {exc}")

    return {"documents": documents}


def generate(state: GraphState) -> dict:
    """基于上下文生成带引用的答案。"""
    print("---GENERATE: 生成答案---")
    documents = state["documents"]
    if not documents:
        reason = state.get("out_of_scope_reason") or "现有年报资料中没有能回答这个问题的依据"
        return {"generation": f"抱歉，{reason}。\n目前收录的是：{corpus_summary()}。"}

    answer = generator.invoke({
        "context": format_context(documents),
        "question": state.get("query") or state["question"],
    })
    return {"generation": answer}


def cite(documents: List[Document]) -> List[str]:
    """把文档列表整理成人类可读的出处清单。"""
    lines = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        if meta.get("source_label"):
            lines.append(f"[{i}] {meta['source_label']}")
        else:
            lines.append(f"[{i}] {meta.get('company')} {meta.get('year')}年报 "
                         f"第{meta.get('page_start')}页 · {meta.get('section')}")
    return lines
