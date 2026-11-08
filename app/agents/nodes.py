import logging
import re
from datetime import datetime

from app.agents.state import AgentState
from app.rag.answer_service import answer_with_rag
from app.services.order_service import get_order_by_id
from app.services.after_sales_policy import (
    check_after_sales_eligibility,
)


logger = logging.getLogger(__name__)


async def general_response(state: AgentState) -> dict:
    """
    普通咨询响应节点。

    调用 RAG 知识库检索企业售后政策，
    结合历史对话生成回答。

    只保留最近三轮对话。
    """

    user_message = state.get("user_message", "")

    if not user_message or not user_message.strip():
        return {
            "answer": "请输入您想咨询的问题。",
            "status": "success",
            "error": "",
        }

    # 第一步：读取历史对话
    chat_history = state.get("chat_history", [])

    # 只保留最近三轮历史对话
    conversation_history = chat_history[-6:]

    try:
        # 第二步：调用 RAG
        answer = await answer_with_rag(
            question=user_message,
            top_k=5,
            conversation_history=conversation_history,
        )

        # 第三步：保存本轮对话
        updated_history = conversation_history + [
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        # 第四步：限制历史对话长度
        updated_history = updated_history[-6:]

        return {
            "chat_history": updated_history,
            "answer": answer,
            "status": "success",
            "error": "",
        }

    except Exception:
        logger.exception("RAG 普通咨询响应失败")

        return {
            "answer": "抱歉，当前暂时无法查询售后政策，请稍后重试。",
            "status": "error",
            "error": "RAG 普通咨询响应失败",
        }


def extract_order_id(message: str) -> str | None:
    """
    从用户消息中提取订单编号。

    当前项目订单编号格式：
    ORD + 9 位数字。
    """

    match = re.search(
        r"\bORD\d{9}\b",
        message,
        re.IGNORECASE,
    )

    if match:
        return match.group(0).upper()

    return None


def extract_request_type(message: str) -> str | None:
    """
    提取售后申请类型。
    """

    if "退款" in message or "退钱" in message:
        return "refund"

    if "退货" in message or "退回" in message:
        return "return"

    return None


async def after_sales_response(state: AgentState) -> dict:
    """
    售后业务处理节点。

    执行步骤：
    1. 提取订单编号。
    2. 识别售后申请类型。
    3. 查询订单。
    4. 调用售后规则引擎。
    5. 根据判断结果生成回复。
    6. 结合 RAG 检索结果补充政策解释。
    """

    user_message = state.get("user_message", "")

    # 第一步：提取订单编号和申请类型
    order_id = (
        extract_order_id(user_message)
        or state.get("order_id")
    )

    request_type = (
        extract_request_type(user_message)
        or state.get("request_type")
    )

    # 第二步：检查订单编号
    if not order_id:
        return {
            "request_type": request_type,
            "answer": (
                "为了查询您的售后申请，请提供订单编号。"
                "例如：ORD202611001。"
            ),
            "status": "success",
            "error": "",
        }

    # 第三步：检查申请类型
    if not request_type:
        return {
            "order_id": order_id,
            "answer": (
                f"已获取订单编号 {order_id}。"
                "请问您希望申请退款还是退货？"
            ),
            "status": "success",
            "error": "",
        }

    # 第四步：查询订单
    try:
        order = get_order_by_id(order_id)

        if order is None:
            eligibility_result = check_after_sales_eligibility(
                order=None,
                request_type=request_type,
            )

        else:
            # 第五步：执行售后规则判断
            eligibility_result = check_after_sales_eligibility(
                order=order,
                request_type=request_type,
                current_time=datetime.now(),
            )

        # 第六步：根据业务判断生成回复
        if not eligibility_result["eligible"]:
            answer = (
                f"订单编号：{order_id}\n"
                f"售后申请类型：{request_type}\n"
                f"处理结果：暂不符合基础申请条件\n"
                f"原因：{eligibility_result['reason']}"
            )

        else:
            answer = (
                f"订单编号：{order.order_id}\n"
                f"商品名称：{order.product_name}\n"
                f"订单金额：{order.amount} 元\n"
                f"订单状态：{order.status}\n"
                f"处理结果：符合基础售后申请条件。\n"
                f"下一步：可以进入人工审核流程。\n"
                f"注意：这不代表退款或退货已经批准。"
            )

        # 第七步：结合售后政策生成回答
        business_context = (
            f"订单编号：{order_id}\n"
            f"申请类型：{request_type}\n"
            f"订单查询结果：{'订单存在' if order is not None else '订单不存在'}\n"
            f"售后资格判断："
            f"{'符合基础申请条件' if eligibility_result['eligible'] else '不符合基础申请条件'}\n"
            f"规则引擎判断原因：{eligibility_result.get('reason', '未提供')}\n"
            "注意：符合基础申请条件不代表最终审批通过。"
        )

        # RAG 失败时保留原有规则引擎回答
        try:
            rag_answer = await answer_with_rag(
                question=user_message,
                top_k=3,
                business_context=business_context,
            )

            if rag_answer:
                answer = rag_answer

        except Exception:
            logger.exception(
                "售后政策 RAG 回答失败，保留规则引擎回答"
            )

        return {
            "order_id": order_id,
            "request_type": request_type,
            "order_info": (
                order.model_dump(mode="json")
                if order is not None
                else {}
            ),
            "eligibility_result": eligibility_result,
            "answer": answer,
            "status": "success",
            "error": "",
        }

    except Exception:
        logger.exception("售后业务处理失败")

        return {
            "answer": "售后业务处理暂时出现异常，请稍后重试。",
            "status": "error",
            "error": "售后业务处理失败",
        }


async def error_response(state: AgentState) -> dict:
    """
    工作流错误处理节点。
    """

    error_message = state.get("error", "")

    logger.error(
        "Agent 工作流执行失败：%s",
        error_message,
    )

    return {
        "answer": "抱歉，系统暂时无法处理您的请求，请稍后重试。",
        "status": "error",
    }