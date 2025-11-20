import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ⚠️ 确保这里的配置与 src/utils.py 中的配置完全一致！
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-ada-002" 

def check_vector_store():
    print(f"尝试加载向量库: {PERSIST_DIR}")
    
    # 确保 API Key 存在，否则 Embedding Function 会报错
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：请确保在 .env 文件中设置了 OPENAI_API_KEY。")
        return

    try:
        # 1. 初始化 Embedding Model
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # 2. 加载持久化的 Chroma 实例
        # 注意：这里不需要传入 chunk_size，因为它只在创建时用于API批处理。
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        
        # 3. 获取 Collection 的状态信息
        collection = db.get()
        print(f"\n✅ 向量库加载成功。")
        print(f"总共存储的文档切块 (Chunks): {len(collection['ids'])}")
        
        # 4. 执行一个简单的相似度搜索
        query = "公司在海外市场的布局情况"
        print(f"\n🔍 尝试搜索关键词: '{query}'...")
        
        # 使用 retriever 进行相似度检索 (k=2)
        retriever = db.as_retriever(search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke(query)
        
        # 5. 打印结果
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- 检索结果 {i+1} ---")
            print(f"📝 原始文本: \n{doc.page_content[:200]}...") # 只打印前200字符
            # 打印文档的元数据，元数据中通常包含原始文件名等信息
            print(f"🏷️ 元数据 (Metadata): {doc.metadata}")
            
    except Exception as e:
        print(f"\n❌ 检查向量库时发生错误: {e}")
        print("   💡 请确认 chroma_db 文件夹存在，且没有被其他程序占用。")


if __name__ == "__main__":
    check_vector_store()