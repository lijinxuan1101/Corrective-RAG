import streamlit as st
import os
from dotenv import load_dotenv
from src.graph import app as graph_app

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(page_title="CRAG 智能助手", page_icon="🤖")

st.title("🤖 Corrective RAG (CRAG) 智能助手")
st.caption("🚀 基于 LangGraph 的自适应检索增强生成系统")

# 初始化会话状态 (用于存储聊天记录)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 在侧边栏显示说明
with st.sidebar:
    st.header("关于项目")
    st.markdown("""
    这是一个 **Self-Corrective RAG** 系统，具备以下能力：
    
    - 📚 **混合检索**：本地知识库 + 联网搜索
    - 🕵️ **自我评估**：自动判断检索文档的相关性
    - 🌐 **智能路由**：当本地文档不足时，自动调用 Tavily 搜索
    """)
    
    # 添加一个清空对话的按钮
    if st.button("清空对话"):
        st.session_state["messages"] = []
        st.rerun()

# 显示历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理用户输入
if prompt := st.chat_input():
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. 处理 AI 回复
    with st.chat_message("assistant"):
        status_container = st.status("🤖 Agent 正在思考...", expanded=True)
        
        try:
            inputs = {"question": prompt}
            final_answer = ""
            
            # 流式获取图的运行结果
            for output in graph_app.stream(inputs):
                for key, value in output.items():
                    # 显示中间步骤
                    if key == "retrieve":
                        status_container.write("📚 正在检索本地知识库...")
                    elif key == "grade_documents":
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
            
            # 保存到历史记录
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            status_container.update(label="❌ 发生错误", state="error")
            st.error(f"运行出错: {e}")