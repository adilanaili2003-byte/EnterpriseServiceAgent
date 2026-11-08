import asyncio

from app.agents.workflow import agent_workflow
from app.agents import intent_classifier


from uuid import uuid4


def run_agent(message):
    return asyncio.run(
        agent_workflow.ainvoke(
            {
                "user_message": message
            },
            config={
                "configurable": {
                    "thread_id": f"test-{uuid4()}"
                }
            }
        )
    )


def mock_after_sales_intent(monkeypatch):
    """模拟模型将用户请求识别为售后业务。"""

    async def mock_chat_completion(**kwargs):
        return "after_sales"

    monkeypatch.setattr(
        intent_classifier,
        "chat_completion",
        mock_chat_completion
    )


def test_agent_valid_refund_request(monkeypatch):
    """测试完整的退款申请流程。"""

    mock_after_sales_intent(monkeypatch)

    result = run_agent(
        "我的订单 ORD202611001 有质量问题，我想申请退款。"
    )

    assert result["intent"] == "after_sales"
    assert result["order_id"] == "ORD202611001"
    assert result["request_type"] == "refund"
    assert result["eligibility_result"]["eligible"] is True
    assert result["status"] == "success"


def test_agent_order_not_found(monkeypatch):
    """测试不存在订单的完整处理流程。"""

    mock_after_sales_intent(monkeypatch)

    result = run_agent(
        "我的订单 ORD999999999 有问题，我想申请退款。"
    )

    assert result["intent"] == "after_sales"
    assert result["eligibility_result"]["code"] == "ORDER_NOT_FOUND"
    assert result["status"] == "success"


def test_agent_missing_order_id(monkeypatch):
    """测试缺少订单编号的情况。"""

    mock_after_sales_intent(monkeypatch)

    result = run_agent(
        "我的商品有质量问题，我想申请退款。"
    )

    assert result["intent"] == "after_sales"
    assert "订单编号" in result["answer"]
    assert result["status"] == "success"


def test_agent_missing_request_type(monkeypatch):
    """测试没有明确退款或退货类型。"""

    mock_after_sales_intent(monkeypatch)

    result = run_agent(
        "我的订单 ORD202611001 有质量问题。"
    )

    assert result["intent"] == "after_sales"
    assert "退款还是退货" in result["answer"]
    assert result["status"] == "success"