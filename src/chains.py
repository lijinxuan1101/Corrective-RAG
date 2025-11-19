from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- 配置 LLM ---
# 建议使用 gpt-4o-mini，速度快且便宜
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# --- 1. 文档评分器 (Retrieval Grader) ---
# 使用 Pydantic 定义结构化输出，强制 LLM 只返回 yes/no
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
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的 AI 助手。请根据提供的上下文回答用户的问题。如果你不知道答案，就说不知道。"),
        ("human", "上下文: {context} \n\n 问题: {question}"),
    ]
)

rag_chain = prompt | llm | StrOutputParser()