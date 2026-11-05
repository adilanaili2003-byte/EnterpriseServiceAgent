import os
from pathlib import Path

from dotenv import load_dotenv

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 加载根目录下的 .env 文件
load_dotenv(BASE_DIR / ".env")


# ==============================
# DeepSeek API 基础配置
# ==============================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# ==============================
# 模型配置
# ==============================

# 默认模型：日常任务
DEEPSEEK_MODEL = "deepseek-flash"

# 高性能模型：复杂推理任务
DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"


def validate_config():
    """检查必要的环境配置是否存在。"""

    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "未检测到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件。"
        )