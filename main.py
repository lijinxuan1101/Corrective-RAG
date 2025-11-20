import os
import warnings
from dotenv import load_dotenv
from src.graph import app
from src.utils import initialize_vector_store # 引入初始化函数

# 忽略所有 DeprecationWarning，保持控制台清洁
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# 加载环境变量
load_dotenv()

# 检查 API KEY 是否存在
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    print("❌ 错误：请在 .env 文件中配置 OPENAI_API_KEY 和 TAVILY_API_KEY")
    exit()

# 启动前初始化向量库
if initialize_vector_store(data_dir="data") is None: 
    print("程序因向量库初始化失败而退出。请根据提示修复。")
    exit()

if __name__ == "__main__":
    print("🤖 Corrective RAG Agent 已启动...")
    
    # 测试问题：问文档中没有的内容，触发 Web Search
    query = "五粮液24年经营状况如何？"
    
    # 测试问题：问文档中的内容
    # query = "请总结一下文档中的核心观点。"

    inputs = {"question": query}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            # 这里可以打印中间过程，保持静默，只打印最终结果
            pass
            
    # 确保打印的是最后一个节点的输出
    final_result = value["generation"] if "generation" in value else "未找到答案"
    print("\n================ FINAL ANSWER ================")
    print(final_result)