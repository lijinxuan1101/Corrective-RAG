from typing import Any, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from src.chains import retrieval_grader, rag_chain
from src.state import GraphState

# --- 模拟检索工具 (实际项目中替换为 VectorDB) ---
def mock_retriever(query: str):
    # 这里为了演示，硬编码了两个文档
    # 如果你问“黑神话”，它能查到；问别的，查不到
    return [
        "文档1: 《黑神话：悟空》是一款由游戏科学开发的动作角色扮演游戏。",
        "文档2: 今天天气真不错，适合出去散步。"
    ]

def retrieve(state: GraphState):
    """节点：从向量数据库检索文档"""
    print("---RETRIEVE: 正在检索本地知识库---")
    question = state["question"]
    
    # 实际项目中：documents = vectorstore.as_retriever().invoke(question)
    documents = mock_retriever(question)
    
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """节点：评估检索到的文档质量 (CRAG 核心)"""
    print("---CHECK RELEVANCE: 正在评估文档质量---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = "No"
    
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
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state: GraphState):
    """节点：联网搜索补充信息"""
    print("---WEB SEARCH: 正在调用 Tavily 搜索---")
    question = state["question"]
    documents = state["documents"]
    
    tool = TavilySearchResults(k=3)
    docs = tool.invoke({"query": question})
    
    # 将搜索结果整理成字符串加入文档列表
    web_results = "\n".join([d["content"] for d in docs])
    documents.append(web_results)
    
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """节点：生成最终答案"""
    print("---GENERATE: 正在生成答案---")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}