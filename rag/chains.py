"""
三条 LLM 链：查询分析、文档评分、答案生成。

评分是高频小任务（每次查询若干篇），用 flash；生成只调一次，用 plus。
"""
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from providers import chat_model

GRADER_MODEL = "qwen-flash"
GENERATOR_MODEL = "qwen-plus"

# 结构化输出走 json_schema 而不是 LangChain 默认的 function_calling：
# 后者在 Qwen 上会间歇性陷入生成循环，实测有一次吐满 32768 token 才报错退出。
# json_mode 则被 DashScope 直接拒绝（400）。
STRUCTURED = {"method": "json_schema"}
MAX_TOKENS = 512    # 结构化输出都很短，封顶可以兜住万一的失控生成

# ---------------------------------------------------------------
# 1. 查询分析：补全指代 + 抽取可过滤的元数据
# ---------------------------------------------------------------
class QueryAnalysis(BaseModel):
    """把口语化的提问变成可检索的查询，并抽出结构化条件。"""
    query: str = Field(description="补全指代后的完整检索查询")
    company: Optional[str] = Field(None, description="问题涉及的公司简称，没提到则为 null")
    year: Optional[str] = Field(None, description="问题涉及的年份，四位数字，没提到则为 null")


ANALYZE_SYSTEM = """你在为一个企业年报问答系统预处理查询。

任务：
1. 结合历史对话，把依赖上下文的提问（如"它去年的呢？"）补全成可独立检索的查询。
2. 抽出问题涉及的公司简称和年份。

要求：
- query 必须是完整的一句话，把公司名和年份都写进去。即使原问题已经完整，
  也要原样保留其中的公司和年份，不要精简成"营收"这样的残句。
- company 和 year 只在问题或历史对话中明确出现时才填，不要猜测或推断。
- year 填四位数字字符串，如 "2024"。"去年""同期"这类相对表述要结合上下文换算成具体年份。
- 只做改写和抽取，不要回答问题本身。"""

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", ANALYZE_SYSTEM),
    ("human", "历史对话：\n{chat_history}\n\n当前问题：{question}"),
])

query_analyzer = analyze_prompt | chat_model(
    GRADER_MODEL, max_tokens=MAX_TOKENS).with_structured_output(QueryAnalysis, **STRUCTURED)


# ---------------------------------------------------------------
# 2. 文档评分：CRAG 的检索评估器
# ---------------------------------------------------------------
class GradeDocument(BaseModel):
    relevant: bool = Field(description="该片段能否为回答问题提供依据")


GRADE_SYSTEM = """你在评估一个年报片段能否用来回答用户的问题。

判定为不相关的情形：
- 片段属于**其他公司**。问的是 A 公司，片段来自 B 公司，即使内容类型完全对应，也必须判不相关。
- 片段属于**其他年份**。问的是某一年的数据，片段是另一年的，判不相关。
- 片段只是话题接近，并不包含能支撑回答的具体信息。

判定为相关：片段包含可以直接或间接支撑回答的事实、数据或表述。

宁可判错为不相关，也不要让错误公司或错误年份的数据通过——
下游会拿它当事实依据生成答案。"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", GRADE_SYSTEM),
    ("human", "片段来源：{source}\n\n片段内容：\n{document}\n\n用户问题：{question}"),
])

document_grader = grade_prompt | chat_model(
    GRADER_MODEL, max_tokens=MAX_TOKENS).with_structured_output(GradeDocument, **STRUCTURED)


# ---------------------------------------------------------------
# 3. 答案生成
# ---------------------------------------------------------------
GENERATE_SYSTEM = """你是企业年报分析助手。依据给定的上下文回答问题。

要求：
- 所有数据必须来自上下文，绝对不可编造或估算。
- 引用数据时用 [1] [2] 标注来源编号。
- 上下文不足以回答时，明确说明缺少什么信息，不要用相近的数据搪塞。
- 涉及金额时带上单位和所属年度。
- 回答用中文，简洁直接。"""

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATE_SYSTEM),
    ("human", "上下文：\n{context}\n\n问题：{question}"),
])

generator = generate_prompt | chat_model(GENERATOR_MODEL) | StrOutputParser()


def format_context(documents: List) -> str:
    """给每篇文档编号并附上出处，让模型能够引用。"""
    blocks = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        origin = meta.get("source_label") or (
            f"{meta.get('company', '')} {meta.get('year', '')}年报"
            f" · {meta.get('section', '')} · 第{meta.get('page_start', '?')}页"
        )
        blocks.append(f"[{i}] {origin}\n{doc.page_content}")
    return "\n\n".join(blocks)
