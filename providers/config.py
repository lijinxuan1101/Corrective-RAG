"""
集中读取 .env 里的凭据。

所有 provider 都从这里取 key，不要在别处直接读 os.environ——
这样换 key 或加新供应商时只有一个地方要改。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def require(name: str) -> str:
    """取必需的环境变量，缺失时给出可操作的报错。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}。\n"
            f"   💡 请在 {ROOT / '.env'} 中添加一行：{name}=你的密钥"
        )
    return value


def dashscope_key() -> str:
    return require("DASHSCOPE_API_KEY")
