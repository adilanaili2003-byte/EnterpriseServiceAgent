import logging
import re

from app.agents.state import AgentState
from app.core.config import DEEPSEEK_MODEL
from app.services.llm_service import chat_completion


logger = logging.getLogger(__name__)


# 当前项目使用的订单编号格式：ORD + 9 位数字
ORDER_ID_PATTERN = re.compile(r"\bORD\d{9}\b", re.IGNORECASE)


def classify_intent_by_rule(user_message: str) -> str | None:
    """
    基于规则优先识别明显意图。

    优先级：
    1. 具体订单编号
    2. 明确的政策、规则和流程咨询
    3. 明确的售后业务办理请求
    4. 无法确定时交给 LLM
    """

    message = user_message.strip()

    if not message:
        return None

    # 第一优先级：存在具体订单编号。
    # 用户已经指向具体订单，进入售后业务处理。
    if ORDER_ID_PATTERN.search(message):
        return "after_sales"

    # 第二优先级：识别政策、规则和流程咨询。
    # 即使包含"申请退款""申请退货"，只要是在询问规则，
    # 就应该进入 general，由 RAG 检索政策知识库。
    consultation_patterns = [
        r"(几天|多久|什么时候).*(退款|退货|换货|到账)",
        r"(退款|退货|换货).*(政策|规则|条件|流程|期限|时间|多久|几天)",
        r"(支持|是否支持|能不能|可不可以|可以不可以).*(退款|退货|换货)",
        r"(无理由退货|七天无理由|七日无理由)",
        r"(超过|超出).*(七天|7天).*(退货|退款)",
        r"(如何|怎么|怎样).*(申请|办理|操作).*(退款|退货|换货)",
        r"(退款|退货|换货).*(怎么办|怎么处理|如何处理)",
        r"(申请退款|申请退货|申请换货).*(流程|条件|规则|需要什么)",
        r"(不知道|不清楚).*(退款|退货|换货).*(流程|规则|条件)",
        r"(商品|产品).*(质量问题|有问题).*(怎么办|怎么处理)",
    ]

    for pattern in consultation_patterns:
        if re.search(pattern, message):
            return "general"

    # 第三优先级：明确要求办理售后业务。
    business_patterns = [
        r"(帮我|请帮我|我要|我想|我需要).*(申请|办理|提交).*(退款|退货|换货)",
        r"^(申请|办理|提交)(退款|退货|换货)$",
        r"(查询|查一下|帮我查|查看).*(订单|物流|快递)",
        r"(取消|撤销).*(订单|购买)",
    ]

    for pattern in business_patterns:
        if re.search(pattern, message):
            return "after_sales"

    return None


async def classify_intent(state: AgentState) -> dict:
    """
    意图识别节点。

    首先通过规则识别明显意图，
    无法确定时再调用 DeepSeek 进行分类。

    意图：
    - general：普通咨询、政策咨询、流程咨询。
    - after_sales：具体订单查询、售后业务办理。
    """

    user_message = state.get("user_message", "").strip()

    if not user_message:
        return {
            "intent": "general",
            "status": "error",
            "error": "用户消息不能为空",
        }

    try:
        # 第一步：规则优先识别。
        rule_result = classify_intent_by_rule(user_message)

        if rule_result is not None:
            return {
                "intent": rule_result,
                "status": "success",
                "error": "",
            }

        # 第二步：规则无法确定时，调用 LLM。
        messages = [
            {
                "role": "system",
                "content": """
你是一个企业智能售后客服系统的意图分类器。

你只能从以下两个类别中选择一个：

1. general：
   普通聊天、问候、产品介绍、政策咨询、
   退款规则咨询、退货期限咨询、售后流程咨询、
   商品质量问题处理方法咨询。

2. after_sales：
   用户要求查询具体订单、查询具体物流状态、
   办理退款、提交退货申请、申请换货、
   处理具体订单异常或投诉。

重要判断原则：

- 询问退款政策，不等于申请退款。
- 询问退货期限，不等于申请退货。
- 询问售后流程，不等于要求系统办理售后。
- 用户明确要求办理具体业务时，才归类为 after_sales。
- 如果用户只是询问如何处理问题，归类为 general。

示例：

支持七天无理由退货吗？ -> general
退款审核后多久到账？ -> general
商品质量问题怎么办？ -> general
我想申请退款，但是不知道流程 -> general
帮我申请退款 -> after_sales
我要退掉 ORD202611001 -> after_sales
查询 ORD202611001 的订单状态 -> after_sales
我的订单为什么还没发货？ -> after_sales

只输出 general 或 after_sales。
不要解释，不要输出其他内容。
"""
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        result = await chat_completion(
            messages=messages,
            model=DEEPSEEK_MODEL,
            temperature=0,
            max_tokens=20,
        )

        intent = result.strip().lower()

        if intent not in ("general", "after_sales"):
            logger.warning("模型返回了无效意图")
            return {
                "intent": "",
                "status": "error",
                "error": "意图识别结果无效",
            }

        return {
            "intent": intent,
            "status": "success",
            "error": "",
        }

    except Exception:
        logger.exception("意图识别失败")

        return {
            "intent": "",
            "status": "error",
            "error": "意图识别服务暂时不可用",
        }