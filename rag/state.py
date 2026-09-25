"""图的状态定义。"""
from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str            # 用户原始提问，始终保留用于最终生成
    query: str               # 重写后用于检索的独立查询
    filters: dict            # 从问题里抽出的公司/年份，收窄检索范围
    scope: str               # in_corpus / out_of_corpus，决定是否值得走本地检索
    out_of_scope_reason: str  # 越界时的说明，直接用于答复
    documents: List[Document]  # 保留 Document 而非字符串，元数据要一路带到引用标注
    confidence: float        # 重排器给出的最高分
    action: str              # correct / ambiguous / incorrect
    generation: str
    chat_history: List[str]
