from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# --- 配置 LLM ---
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# --- 1. 文档评分器 (Retrieval Grader) ---
class GradeDocuments(BaseModel):
    """对检索文档的相关性进行打分"""
    binary_score: str = Field(description="文档是否与问题相关，'yes' 或 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """你是一个评分员，负责评估检索到的文档与用户问题的相关性。
如果文档包含与问题相关的关键词或语义，请评级为 'yes'，否则评级为 'no'。"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "检索文档: \n\n {document} \n\n 用户问题: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# --- 2. 最终生成器 (Generator) ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的 AI 助手。请根据提供的上下文回答用户的问题。如果你不知道答案，就说不知道。"),
        ("human", "上下文: {context} \n\n 问题: {question}"),
    ]
)

rag_chain = prompt | llm | StrOutputParser()

# --- 3. 查询重写器 (Query Rewriter) ---

rewrite_system_prompt = """
你是一个查询重写器。你的任务是分析当前的问题和历史对话上下文，
将用户依赖上下文的模糊问题（例如：‘它去年的呢？’）重写为一个可以独立用于检索的、完整、清晰的查询语句。
如果当前问题已经是完整的，则直接返回原问题。
只返回重写后的查询语句，不要包含任何额外解释或标点。
"""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewrite_system_prompt),
        # 我们使用一个简单的字符串格式来传递历史记录和问题
        ("human", "历史对话上下文: {chat_history} \n\n 当前问题: {question}"),
    ]
)

# 重写 Chain 不使用结构化输出，直接返回字符串
query_rewriter = rewrite_prompt | llm | StrOutputParser()