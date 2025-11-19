from langgraph.graph import END, StateGraph
from src.state import GraphState
from src.nodes import retrieve, grade_documents, web_search, generate

def decide_to_generate(state):
    """条件判断：下一步是去搜索还是直接生成？"""
    print("---DECIDE: 决策下一步---")
    if state["web_search"] == "Yes":
        return "web_search"
    else:
        return "generate"

# 构建图
workflow = StateGraph(GraphState)

# 1. 添加节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# 2. 设置入口点
workflow.set_entry_point("retrieve")

# 3. 添加边
workflow.add_edge("retrieve", "grade_documents")

# 4. 添加条件边 (CRAG 的核心逻辑)
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# 编译图
app = workflow.compile()