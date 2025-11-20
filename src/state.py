from typing import List, TypedDict

class GraphState(TypedDict):
    """
    定义图的状态 (State)。
    """
    question: str           # 用户当前的问题 (经过重写可能已改变)
    generation: str         # LLM 生成的最终答案
    web_search: str         # 是否需要联网搜索 ("Yes" or "No")
    documents: List[str]    # 检索到的文档列表
    chat_history: List[str] # 存储历史对话记录，用于查询重写