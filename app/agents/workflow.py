from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.state import AgentState
from app.agents.intent_classifier import classify_intent
from app.agents.nodes import (
    general_response,
    after_sales_response,
    error_response,
)


def route_intent(state: AgentState) -> str:
    """
    根据意图识别结果决定下一步执行哪个节点。
    """

    if state.get("status") == "error":
        return "error"

    intent = state.get("intent")

    if intent == "general":
        return "general"

    if intent == "after_sales":
        return "after_sales"

    return "error"


def create_workflow(checkpointer=None):
    """
    创建企业智能售后 Agent 工作流。

    checkpointer 用于保存工作流执行状态，
    为后续 HITL 暂停和恢复提供支持。
    """

    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("general_response", general_response)
    graph.add_node("after_sales_response", after_sales_response)
    graph.add_node("error_response", error_response)

    # 设置起点
    graph.add_edge(START, "classify_intent")

    # 根据意图进行条件路由
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "general": "general_response",
            "after_sales": "after_sales_response",
            "error": "error_response",
        },
    )

    # 设置各分支的终点
    graph.add_edge("general_response", END)
    graph.add_edge("after_sales_response", END)
    graph.add_edge("error_response", END)

    # 编译工作流
    return graph.compile(checkpointer=checkpointer)


# 开发阶段使用内存检查点
memory_saver = InMemorySaver()

# 创建可供其他模块调用的工作流实例
agent_workflow = create_workflow(checkpointer=memory_saver)