"""
阿里云百炼（DashScope）对话模型。

百炼提供 OpenAI 兼容端点，所以直接复用 ChatOpenAI，
只是把 base_url 指过去——上层的 LangChain 链路一行都不用改。
"""
from langchain_openai import ChatOpenAI

from providers import config

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def chat_model(model: str = DEFAULT_MODEL, temperature: float = 0,
               **kwargs) -> ChatOpenAI:
    """
    可选型号：qwen-flash（最快最便宜）、qwen-plus（均衡）、
    qwen-max / qwen3-max（最强，贵且慢）。

    评分这类高频小任务适合 qwen-flash，最终生成适合 qwen-plus 以上。
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=config.dashscope_key(),
        base_url=BASE_URL,
        **kwargs,
    )
