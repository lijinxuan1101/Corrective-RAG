from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from runtime import ensure_environment, ensure_vector_store

# 先初始化环境（必须在导入 graph 前完成）
ensure_environment()

from src.graph import app as graph_app  # noqa: E402

# 初始化或加载向量库
ensure_vector_store()


class QuestionRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    chat_history: List[str] = Field(default_factory=list, description="可选的历史对话")


class AnswerResponse(BaseModel):
    answer: str


app = FastAPI(
    title="Corrective RAG API",
    description="基于 LangGraph 的自适应问答服务",
    version="1.0.0",
)


@app.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse, tags=["rag"])
async def ask_question(payload: QuestionRequest):
    try:
        inputs = {"question": payload.question, "chat_history": payload.chat_history}
        result = graph_app.invoke(inputs)
        answer = result.get("generation", "未生成答案")
        return AnswerResponse(answer=answer)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc

