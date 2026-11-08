from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_message: str

    # 意图识别
    intent: str

    # 普通咨询对话历史
    chat_history: list[dict]

    # 售后请求信息
    order_id: str
    request_type: str

    # 订单及售后数据
    policy_context: str
    order_info: dict
    eligibility_result: dict

    # 工作流返回信息
    answer: str
    status: str
    error: str