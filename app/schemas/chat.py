from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """用户聊天请求模型"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户发送的消息"
    )

    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="会话 ID，不传则自动创建新会话"
    )


class ChatResponse(BaseModel):
    """AI 聊天响应模型"""

    reply: str = Field(
        ...,
        description="AI 返回的回答"
    )

    session_id: str = Field(
        ...,
        description="当前会话 ID"
    )