from openai import AsyncOpenAI

from app.core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_PRO_MODEL,
    validate_config,
)


# 检查配置
validate_config()


# 创建异步 DeepSeek 客户端
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=60.0,
    max_retries=2,
)


async def chat_completion(
    messages: list[dict],
    model: str = DEEPSEEK_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    调用 DeepSeek 模型并返回文本回答。

    Args:
        messages: 对话消息列表
        model: 使用的模型，默认 deepseek-chat
        temperature: 控制回答的随机性
        max_tokens: 最大输出 Token 数

    Returns:
        模型生成的文本
    """

    if not messages:
        raise ValueError("messages 不能为空")

    if model not in (DEEPSEEK_MODEL, DEEPSEEK_PRO_MODEL):
        raise ValueError(f"不支持的模型：{model}")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("模型返回了空内容")

        return content.strip()

    except Exception:
        # 不记录 API Key 或完整请求内容
        raise


async def close_llm_client():
    """关闭异步客户端连接。"""
    await client.close()