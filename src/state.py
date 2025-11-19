from typing import List, TypedDict

class GraphState(TypedDict):
    """
    定义图的状态 (State)。
    各个节点会读取或修改这里的属性。
    """
    question: str           # 用户的问题
    generation: str         # LLM 生成的最终答案
    web_search: str         # 是否需要联网搜索 ("Yes" or "No")
    documents: List[str]    # 检索到的文档列表