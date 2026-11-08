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
    1. 提取订单编号和申请类型。
    2. 查询订单。
    3. 如果是普通订单查询，直接返回订单信息。
    4. 如果是售后办理，调用售后规则引擎。
    5. 结合 RAG 检索结果补充政策解释。
    6. 保存订单上下文和历史对话。
    """

    user_message = state.get("user_message", "").strip()

    # 第一步：读取历史对话
    chat_history = state.get("chat_history", [])
    conversation_history = chat_history[-6:]

    # 第二步：提取订单编号和申请类型
    order_id = (
        extract_order_id(user_message)
        or state.get("order_id")
    )

    request_type = (
        extract_request_type(user_message)
        or state.get("request_type")
    )

    # 第三步：检查订单编号
    if not order_id:
        answer = (
            "为了查询您的订单或售后申请，请提供订单编号。"
            "例如：ORD202611001。"
        )

        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]

        return {
            "chat_history": updated_history[-6:],
            "request_type": request_type,
            "answer": answer,
            "status": "success",
            "error": "",
        }

    try:
        # 第四步：查询订单
        order = get_order_by_id(order_id)

        if order is None:
            eligibility_result = check_after_sales_eligibility(
                order=None,
                request_type=request_type,
            )

            answer = (
                f"暂未查询到订单 {order_id}，"
                "请检查订单编号是否正确。"
            )

            updated_history = conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer},
            ]

            return {
                "order_id": order_id,
                "order_info": {},
                "request_type": request_type,
                "eligibility_result": eligibility_result,
                "chat_history": updated_history[-6:],
                "answer": answer,
                "status": "success",
                "error": "",
            }

        # 第五步：整理订单信息
        order_info = order.model_dump(mode="json")

        product_name = order.product_name
        amount = order.amount
        order_status = order.status

        # 第六步：区分普通订单查询和售后办理
        if not request_type:
            # 判断用户是否明确在查询订单信息
            order_query_keywords = [
                "价格",
                "多少钱",
                "金额",
                "商品名称",
                "商品是什么",
                "是什么商品",
                "商品叫什么",
                "订单状态",
                "物流状态",
                "发货了吗",
                "签收了吗",
                "订单到哪了",
            ]

            is_order_query = any(
                keyword in user_message
                for keyword in order_query_keywords
            )

            answer = (
                f"订单编号：{order_id}\n"
                f"商品名称：{product_name}\n"
                f"订单金额：{amount} 元\n"
                f"订单状态：{order_status}"
            )

            # 如果用户不是单纯查询订单信息，
            # 则进一步询问售后申请类型
            if not is_order_query:
                answer += "\n\n请问您希望申请退款还是退货？"

            updated_history = conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer},
            ]

            return {
                "order_id": order_id,
                "order_info": order_info,
                "chat_history": updated_history[-6:],
                "answer": answer,
                "status": "success",
                "error": "",
            }

        # 第七步：执行售后规则判断
        eligibility_result = check_after_sales_eligibility(
            order=order,
            request_type=request_type,
            current_time=datetime.now(),
        )

        if not eligibility_result["eligible"]:
            answer = (
                f"订单编号：{order_id}\n"
                f"售后申请类型：{request_type}\n"
                f"处理结果：暂不符合基础申请条件\n"
                f"原因：{eligibility_result['reason']}"
            )

        else:
            answer = (
                f"订单编号：{order_id}\n"
                f"商品名称：{product_name}\n"
                f"订单金额：{amount} 元\n"
                f"订单状态：{order_status}\n"
                f"处理结果：符合基础售后申请条件。\n"
                f"下一步：可以进入人工审核流程。\n"
                f"注意：这不代表退款或退货已经批准。"
            )

        # 第八步：构建业务上下文
        business_context = (
            f"订单编号：{order_id}\n"
            f"商品名称：{product_name}\n"
            f"订单金额：{amount} 元\n"
            f"订单状态：{order_status}\n"
            f"申请类型：{request_type}\n"
            "订单查询结果：订单存在\n"
            f"售后资格判断："
            f"{'符合基础申请条件' if eligibility_result['eligible'] else '不符合基础申请条件'}\n"
            f"规则引擎判断原因：{eligibility_result.get('reason', '未提供')}\n"
            "注意：符合基础申请条件不代表最终审批通过。"
        )

        # 第九步：结合 RAG 补充售后政策
        try:
            rag_answer = await answer_with_rag(
                question=user_message,
                top_k=3,
                business_context=business_context,
                conversation_history=conversation_history,
            )

            if rag_answer:
                # RAG 负责补充政策解释，不覆盖订单事实和规则判断
                answer = (
                    f"{answer}\n\n"
                    f"相关政策说明：\n{rag_answer}"
                )

        except Exception:
            logger.exception(
                "售后政策 RAG 回答失败，保留原有规则引擎回答"
            )

        # 第十步：保存本轮对话
        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]

        return {
            "order_id": order_id,
            "request_type": request_type,
            "order_info": order_info,
            "eligibility_result": eligibility_result,
            "chat_history": updated_history[-6:],
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