from app.core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_PRO_MODEL,
    validate_config,
)

validate_config()

print("DeepSeek 配置读取成功！")
print("API Key：已读取（不显示具体内容）")
print("Base URL：", DEEPSEEK_BASE_URL)
print("默认模型：", DEEPSEEK_MODEL)
print("高性能模型：", DEEPSEEK_PRO_MODEL)