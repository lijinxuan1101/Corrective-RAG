"""
CRAG 状态机。

    analyze ─┬─ in_corpus ────→ retrieve → grade ─┬─ correct ─→ generate
             │                                     └─ 其余 ────┐
             └─ out_of_corpus ──────────────────────────────→ search_web → generate
"""
from langgraph.graph import END, StateGraph

from rag.nodes import analyze, generate, grade, retrieve, web_search
from rag.state import GraphState


def route_scope(state: GraphState) -> str:
    """
    公司或年份不在语料内时跳过本地检索。

    这一跳省掉一次检索加五次评分调用——问"茅台"时答案必然是"没有依据"，
    在 analyze 阶段就已经确定，没必要再让重排器和评分器走一遍流程。
    """
    if state["scope"] == "out_of_corpus":
        print("---ROUTE: 跳过本地检索---")
        return "search_web"
    return "retrieve"


def route_action(state: GraphState) -> str:
    """CRAG 的纠正动作：检索够用就直接生成，否则先补充外部知识。"""
    action = state["action"]
    print(f"---ROUTE: {action}---")
    return "generate" if action == "correct" else "search_web"


workflow = StateGraph(GraphState)

# 节点名和 state 键不能重名，否则 LangGraph 分不清是节点输出还是状态字段
workflow.add_node("analyze", analyze)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("search_web", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", route_scope,
                               {"retrieve": "retrieve", "search_web": "search_web"})
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route_action,
                               {"generate": "generate", "search_web": "search_web"})
workflow.add_edge("search_web", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
