from typing import Any, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document 
from src.state import GraphState
from src.utils import get_vector_store
# 一次性导入 src.chains 中的所有相关对象
from src.chains import retrieval_grader, rag_chain, query_rewriter 


# ----------------------------------------------------
# 1. 查询重写节点
# ----------------------------------------------------
def rewrite_query(state: GraphState):
    """节点：根据历史对话，将当前问题重写为完整的、可检索的查询"""
    print("---REWRITE: 正在重写查询语句---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # 格式化历史记录，用于 Prompt
    formatted_history = "\n".join(chat_history)
    
    # 调用重写 LLM Chain
    rewritten_query = query_rewriter.invoke({
        "question": question,
        "chat_history": formatted_history
    })
    
    # 打印重写结果
    if rewritten_query.strip() != question.strip():
        print(f"   💡 原问题: '{question}'")
        print(f"   ✨ 重写结果: '{rewritten_query}'")
    else:
        print(f"   ✅ 问题无需重写: '{question}'")

    # 更新状态中的 question，并确保传递 chat_history
    # state.get("documents", []) 确保在入口节点时 documents 为空
    return {"question": rewritten_query, "documents": state.get("documents", []), "chat_history": chat_history}


# ----------------------------------------------------
# 2. 检索节点
# ----------------------------------------------------
def retrieve(state: GraphState):
    """节点：从向量数据库检索文档 (使用重写后的问题)"""
    print("---RETRIEVE: 正在检索本地知识库---")
    
    question = state["question"]
    chat_history = state.get("chat_history", []) # 确保传递 chat_history
    
    vector_store = get_vector_store()
    if not vector_store:
        raise Exception("向量库未初始化，请检查 main.py")

    # 使用 ChromaDB 的 Retriever 接口 (k=5 表示检索 top 5 块)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 
    
    # 检索并提取 content
    docs = retriever.invoke(question)
    documents = [d.page_content for d in docs]

    # 确保返回了 chat_history
    return {"documents": documents, "question": question, "chat_history": chat_history}


# ----------------------------------------------------
# 3. 文档评分节点
# ----------------------------------------------------
def grade_documents(state: GraphState):
    """节点：评估检索到的文档质量 (CRAG 核心)"""
    print("---CHECK RELEVANCE: 正在评估文档质量---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", []) # 确保传递 chat_history
    
    filtered_docs = []
    web_search = "No"
    
    # 逐一评分
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d})
        grade = score.binary_score
        
        if grade == "yes":
            print(f"  - 文档相关 (保留)")
            filtered_docs.append(d)
        else:
            print(f"  - 文档不相关 (丢弃)")
            continue
            
    # 如果没有相关文档，标记需要联网搜索
    if not filtered_docs:
        print("  ! 警告: 没有找到相关文档，准备联网搜索")
        web_search = "Yes"
        
    # 确保返回了 chat_history
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "chat_history": chat_history}


# ----------------------------------------------------
# 4. 联网搜索节点
# ----------------------------------------------------
def web_search(state: GraphState):
    """节点：联网搜索补充信息"""
    print("---WEB SEARCH: 正在调用 Tavily 搜索---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", []) # 确保传递 chat_history
    
    # 使用 langchain_community 的 TavilySearchResults
    tool = TavilySearchResults(k=3)
    docs = tool.invoke({"query": question})
    
    # 将搜索结果整理成字符串加入文档列表
    web_results = "\n".join([d["content"] for d in docs])
    documents.append(web_results)
    
    # 确保返回了 chat_history
    return {"documents": documents, "question": question, "chat_history": chat_history}


# ----------------------------------------------------
# 5. 生成节点
# ----------------------------------------------------
def generate(state: GraphState):
    """节点：生成最终答案"""
    print("---GENERATE: 正在生成答案---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", []) # 确保传递 chat_history
    
    # LLM 使用文档作为上下文生成答案
    generation = rag_chain.invoke({"context": documents, "question": question})
    
    # 确保返回了 chat_history
    return {"documents": documents, "question": question, "generation": generation, "chat_history": chat_history}