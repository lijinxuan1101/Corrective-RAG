import os
# ⚠️ 关键修改：引入 DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 定义全局变量来存储向量库实例
VECTOR_STORE = None
PERSIST_DIR = "./chroma_db"

# 核心修改：函数接收一个目录路径 data_dir，默认值为 'data'
def initialize_vector_store(data_dir: str = "data"):
    """
    加载指定目录及其子目录下的所有 PDF 文件，切割，嵌入，并初始化 ChromaDB 向量库。
    """
    global VECTOR_STORE
    
    # --- 1. 检查并加载已存在的向量库 ---
    if os.path.exists(PERSIST_DIR) and VECTOR_STORE is None:
        print("✅ 正在加载已存在的向量库...")
        try:
            VECTOR_STORE = Chroma(persist_directory=PERSIST_DIR, 
                                  embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=100))
            print("✅ 向量库加载成功。")
            return VECTOR_STORE
        except Exception as e:
            print(f"加载向量库失败: {e}，将尝试重新创建。")
            
    if VECTOR_STORE is not None:
        return VECTOR_STORE

    # --- 2. 加载文档并创建新的向量库 ---
    try:
        # 使用 DirectoryLoader 递归加载所有子文件夹下的 .pdf 文件
        print(f"📚 正在加载目录 '{data_dir}' 下的所有 PDF 文档...")
        
        # glob="**/*.pdf": 递归查找子目录 (**) 下的所有 .pdf 文件
        loader = DirectoryLoader(
            path=data_dir,
            glob="**/*.pdf",  
            loader_cls=PyPDFLoader, # 指定使用 PyPDFLoader 处理每个文件
            recursive=True # 允许递归查找子目录
        )
        documents = loader.load()
        
        if not documents:
            print(f"❌ 错误：在指定目录 '{data_dir}' 或其子目录中没有找到任何 PDF 文件。请检查路径和文件后缀。")
            return None

        # 3. 文档切分 (Chunking)
        print(f"✂️ 找到了 {len(documents)} 个文档页面，正在进行文档切块...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(documents)

        # 4. 嵌入模型 (Embedding)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", 
                                      chunk_size=100)

        # 5. 存储到 ChromaDB
        print("💾 正在创建/持久化向量库 (可能需要一些时间)...")
        VECTOR_STORE = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        # ⚠️ 确保数据被持久化到磁盘
        VECTOR_STORE.persist() 
        print(f"🎉 ChromaDB 向量库创建成功，存储于 {PERSIST_DIR}。")
        
        return VECTOR_STORE

    except Exception as e:
        print(f"❌ 初始化向量库过程中发生错误: {e}")
        # 如果是文件相关的错误，给出更明确的提示
        if "No such file or directory" in str(e) and not os.path.exists(data_dir):
             print(f"   💡 请确认项目根目录下存在名为 '{data_dir}' 的文件夹。")
        return None

def get_vector_store():
    """获取向量库实例"""
    return VECTOR_STORE