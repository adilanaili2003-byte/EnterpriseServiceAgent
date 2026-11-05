import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import chat_completion


# 创建日志记录器
logger = logging.getLogger(__name__)

# 创建 API 路由
router = APIRouter(
    prefix="/api/v1",
    tags=["AI 对话"]
)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="AI 智能客服对话"
)
async def chat(request: ChatRequest):
    """
    接收用户消息，调用 DeepSeek 模型并返回回答。
    """

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名专业、友好、耐心的企业售后客服助手。"
                    "请使用中文回答用户问题，表达清晰、简洁。"
                    "如果用户询问具体订单或退款状态，"
                    "请说明需要查询订单信息，不要编造订单或退款结果。"
                )
            },
            {
                "role": "user",
                "content": request.message
            }
        ]

        reply = await chat_completion(messages)

        return ChatResponse(reply=reply)

    except Exception:
        logger.exception("AI 聊天接口调用失败")

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI 服务暂时不可用，请稍后重试。"
        )