from runtime import ensure_environment, ensure_vector_store

try:
    ensure_environment()
except EnvironmentError as e:
    print(f"❌ 环境初始化失败: {e}")
    exit()

# 在环境变量加载后再导入需要 API key 的模块
from src.graph import app

try:
    ensure_vector_store()
except RuntimeError as e:
    print(f"❌ {e}")
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