import logging
from uuid import uuid4
from fastapi import APIRouter, HTTPException, status

from app.agents.workflow import agent_workflow
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["AI 对话"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="企业智能客服 Agent 对话"
)
async def chat(request: ChatRequest):
    """
    调用 LangGraph 工作流处理用户请求。
    """

    # 优先使用客户端传入的 session_id，否则创建新会话
    session_id = request.session_id or str(uuid4())

    try:
        result = await agent_workflow.ainvoke(
            {
                "user_message": request.message
            },
            config={
                "configurable": {
                    "thread_id": session_id
                }
            }
        )

        answer = result.get("answer")

        if not answer:
            raise RuntimeError("Agent 工作流未生成有效回答")

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=answer
            )

        return ChatResponse(
            reply=answer,
            session_id=session_id
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Agent 工作流执行失败")

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="智能客服服务暂时不可用，请稍后重试。"
        )