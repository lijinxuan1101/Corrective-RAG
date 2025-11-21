# 🤖 Adaptive CRAG Agent: 基于 LangGraph 的自适应 RAG 问答系统

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/AI-LangGraph-orange)
![RAG](https://img.shields.io/badge/RAG-Corrective-green)

## 📖 项目简介 (Introduction)

本项目实现了一个基于 **Corrective RAG (CRAG)** 架构的智能问答助手。与传统 RAG 不同，该系统引入了 **"自我评估 (Self-Evaluation)"** 机制：它不仅能检索本地知识库，还能主动评估检索到的文档是否相关。

如果本地文档质量不足，系统会自动触发 **Tavily Web Search** 进行联网搜索，从而显著减少模型幻觉 (Hallucination)，解决 "不知道强行回答" 的问题。

## 🏗️ 系统架构 (Architecture)

本项目采用 **LangGraph** 构建 State Machine (状态机)，流程如下：

```mermaid
graph TD
    A[User Query] --> B[Retrieve Documents]
    B --> C[Grade Documents]
    C --> D{Is Relevant?}
    D -- Yes --> E[Generate Answer]
    D -- No/Ambiguous --> F[Web Search (Tavily)]
    F --> E
    E --> G[Final Output]

## 🚀 运行方式 (Usage)

1. **命令行模式**
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

2. **Streamlit Web UI**
   ```bash
   streamlit run app.py
   ```

3. **FastAPI 服务**
   ```bash
   uvicorn api_server:app --reload
   ```
   启动后访问 `http://localhost:8000/docs` 体验在线接口文档。

> 环境变量 `FORCE_REBUILD_VECTOR_STORE=true` 可用于强制重建向量库；默认情况下系统会复用 `./chroma_db` 中已持久化的向量。