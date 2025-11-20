import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from src.graph import app as graph_app
from src.utils import initialize_vector_store

# 忽略所有 DeprecationWarning，保持控制台清洁
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# 加载环境变量
load_dotenv()

# --- 1. 页面初始化和配置 ---

st.set_page_config(page_title="CRAG 智能助手", page_icon="🤖")

st.title("🤖 Corrective RAG (CRAG) 智能助手")
st.caption("🚀 基于 LangGraph 的自适应检索增强生成系统 (已支持多轮对话)")

# 检查 API KEY 是否存在
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error("❌ 错误：请在 .env 文件中配置 OPENAI_API_KEY 和 TAVILY_API_KEY！")
    st.stop()


# 初始化向量库（显示在状态栏中）
if 'vector_store_initialized' not in st.session_state:
    with st.spinner('正在初始化/加载向量库 (请耐心等待，只需首次运行)...'):
        # ⚠️ 这里使用 data_dir="data"，函数会递归加载所有 PDF
        if initialize_vector_store(data_dir="data") is None:
            st.error("向量库初始化失败，请检查终端日志和 .env 配置！")
        else:
            st.session_state['vector_store_initialized'] = True
            st.success("向量库加载成功！")


# 初始化会话状态 (用于存储聊天记录)
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# --- 2. 聊天记录显示 ---

# 显示历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# --- 3. 处理用户输入和 Agent 执行 ---

if prompt := st.chat_input("向你的知识库提问..."):
    # 1. 保存和显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. 准备历史记录 (用于 Query Rewriting)
    chat_history = []
    # 从 st.session_state.messages 中提取历史对话，格式为 ["Human: Q", "AI: A"]
    for msg in st.session_state.messages[:-1]: # 排除当前用户输入
        if msg["role"] == "user":
            # 确保历史记录中只包含完整的问答对
            chat_history.append(f"Human: {msg['content']}")
        elif msg["role"] == "assistant":
            chat_history.append(f"AI: {msg['content']}")


    # 3. 处理 AI 回复
    with st.chat_message("assistant"):
        status_container = st.status("🤖 Agent 正在思考...", expanded=True)
        
        try:
            # 关键修改：传入 question 和 chat_history
            inputs = {"question": prompt, "chat_history": chat_history} 
            final_answer = ""
            
            # 流式获取图的运行结果
            for output in graph_app.stream(inputs):
                for key, value in output.items():
                    # 显示中间步骤
                    if key == "rewrite_query":
                        status_container.write("🧠 正在基于上下文重写查询...")
                    elif key == "retrieve":
                        status_container.write("📚 正在检索本地知识库...")
                    elif key == "grade_documents":
                        # 检查 web_search 标志
                        if value.get("web_search") == "Yes":
                            status_container.write("⚠️ 本地文档质量不足，准备联网搜索...")
                        else:
                            status_container.write("✅ 本地文档质量达标。")
                    elif key == "web_search":
                        status_container.write("🌐 正在调用 Tavily 进行联网搜索...")
                    elif key == "generate":
                        final_answer = value["generation"]
                        status_container.write("💡 正在生成最终答案...")
            
            status_container.update(label="✅ 回答完毕", state="complete", expanded=False)
            
            # 显示最终答案
            st.write(final_answer)
            
            # 4. 保存到历史记录
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            status_container.update(label="❌ 发生错误", state="error")
            st.error(f"运行出错: {e}")
            st.warning("请检查终端中的详细错误日志。")