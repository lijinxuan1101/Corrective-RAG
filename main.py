import os
from dotenv import load_dotenv
from src.graph import app
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 加载环境变量
load_dotenv()

# 检查 API KEY 是否存在
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    print("请在 .env 文件中配置 OPENAI_API_KEY 和 TAVILY_API_KEY")
    exit()

if __name__ == "__main__":
    print("🤖 Corrective RAG Agent 已启动...")
    
    # 测试问题 1：知识库里有的（应该直接回答）
    # query = "介绍一下黑神话悟空"
    
    # 测试问题 2：知识库里没有的（应该触发联网搜索）
    query = "今天上海的天气怎么样？"
    
    inputs = {"question": query}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            # 这里可以打印中间过程
            pass
            
    print("\n================ FINAL ANSWER ================")
    print(value["generation"])