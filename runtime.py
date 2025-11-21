import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from src.utils import initialize_vector_store

# 统一忽略 DeprecationWarning，防止多入口重复配置
warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = BASE_DIR / ".env"

_ENV_READY = False
_VECTOR_STORE_READY = False


def ensure_environment() -> None:
    """
    加载 .env 并校验关键配置。
    该函数可安全重复调用。
    """
    global _ENV_READY
    if _ENV_READY:
        return

    env_file = os.getenv("ENV_FILE_PATH", str(DEFAULT_ENV_PATH))
    env_path = Path(env_file)

    if not env_path.exists():
        raise EnvironmentError(f".env 文件不存在，请在 {env_path} 创建或通过 ENV_FILE_PATH 指定路径。")

    load_dotenv(dotenv_path=env_path)

    missing = []
    for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        if not os.getenv(key):
            missing.append(key)

    if missing:
        raise EnvironmentError(f"缺少必要的环境变量: {', '.join(missing)}")

    _ENV_READY = True


def ensure_vector_store() -> None:
    """
    初始化或加载向量库，可通过环境变量控制是否强制重建。
    该函数可安全重复调用。
    """
    global _VECTOR_STORE_READY
    if _VECTOR_STORE_READY:
        return

    force_rebuild_env = os.getenv("FORCE_REBUILD_VECTOR_STORE", "false").lower()
    force_rebuild = force_rebuild_env in {"1", "true", "yes"}
    data_dir = os.getenv("DATA_DIR", "data")

    vector_store = initialize_vector_store(data_dir=data_dir, force_rebuild=force_rebuild)
    if vector_store is None:
        raise RuntimeError("向量库初始化失败，请检查日志提示。")

    _VECTOR_STORE_READY = True

