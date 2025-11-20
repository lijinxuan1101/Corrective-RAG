from langgraph.graph import END, StateGraph
from src.state import GraphState
# ⚠️ 引入新的 rewrite_query 节点
from src.nodes import rewrite_query, retrieve, grade_documents, web_search, generate

def decide_to_generate(state):
    """条件判断：下一步是去搜索还是直接生成？"""
    print("---DECIDE: 决策下一步---")
    if state["web_search"] == "Yes":
        return "web_search"
    else:
        return "generate"

# 构建图
workflow = StateGraph(GraphState)

# 1. 添加节点 (新增 rewrite_query)
workflow.add_node("rewrite_query", rewrite_query) # ⚠️ 新增
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# 2. 设置入口点
workflow.set_entry_point("rewrite_query") # ⚠️ 修改入口点

# 3. 添加边
workflow.add_edge("rewrite_query", "retrieve") # ⚠️ 新增边
workflow.add_edge("retrieve", "grade_documents")
# ... (后续边保持不变)
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