import asyncio
from unittest.mock import AsyncMock, patch

from app.agents.nodes import after_sales_response, general_response


def run_node(message: str):
    with patch(
        "app.agents.nodes.answer_with_rag",
        new_callable=AsyncMock,
        return_value=(
            "符合基础申请条件，可以进入人工审核流程，"
            "但尚未获得最终批准。"
        ),
    ):
        return asyncio.run(
            after_sales_response({
                "user_message": message
            })
        )


def test_after_sales_with_valid_order():
    """测试有效订单的退款申请。"""

    result = run_node(
        "我的订单 ORD202611001 有质量问题，我想申请退款。"
    )

    assert result["status"] == "success"
    assert result["order_id"] == "ORD202611001"
    assert result["request_type"] == "refund"
    assert result["eligibility_result"]["eligible"] is True
    assert "人工审核" in result["answer"]


def test_after_sales_without_order_id():
    """测试用户没有提供订单编号。"""

    result = run_node(
        "我的商品有质量问题，我想申请退款。"
    )

    assert result["status"] == "success"
    assert "订单编号" in result["answer"]


def test_after_sales_order_not_found():
    """测试不存在的订单。"""

    result = run_node(
        "我的订单 ORD999999999 有问题，我想申请退款。"
    )

    assert result["status"] == "success"
    assert result["eligibility_result"]["eligible"] is False
    assert result["eligibility_result"]["code"] == "ORDER_NOT_FOUND"


def test_after_sales_missing_request_type():
    """测试用户提供订单编号但没有明确申请类型。"""

    result = run_node(
        "我的订单 ORD202611001 有质量问题。"
    )

    assert result["status"] == "success"
    assert "退款还是退货" in result["answer"]


def test_after_sales_rag_failure():
    """测试 RAG 失败时仍然返回规则引擎结果。"""

    with patch(
        "app.agents.nodes.answer_with_rag",
        new_callable=AsyncMock,
        side_effect=RuntimeError("模拟 RAG 服务异常"),
    ):
        result = asyncio.run(
            after_sales_response({
                "user_message": (
                    "我的订单 ORD202611001 有质量问题，"
                    "我想申请退款。"
                )
            })
        )

    assert result["status"] == "success"
    assert result["eligibility_result"]["eligible"] is True
    assert "人工审核" in result["answer"]


def test_general_response_uses_top_5():
    """测试普通咨询节点使用 Top 5 检索知识库。"""

    mock_answer = "普通退货期限为签收后7天。"

    with patch(
        "app.agents.nodes.answer_with_rag",
        new_callable=AsyncMock,
        return_value=mock_answer,
    ) as mock_rag:

        result = asyncio.run(
            general_response({
                "user_message": "商品签收超过7天还能申请退货吗？"
            })
        )

    assert result["status"] == "success"
    assert result["answer"] == mock_answer

    mock_rag.assert_awaited_once_with(
        question="商品签收超过7天还能申请退货吗？",
        top_k=5,
        conversation_history=[],
    )

def test_after_sales_rule_engine_failure():
    """测试售后规则引擎异常时是否返回友好错误提示。"""

    with patch(
        "app.agents.nodes.check_after_sales_eligibility",
        side_effect=RuntimeError("模拟规则引擎异常"),
    ):
        result = asyncio.run(
            after_sales_response({
                "user_message": (
                    "我的订单 ORD202611001 有质量问题，"
                    "我想申请退款。"
                )
            })
        )

    assert result["status"] == "error"
    assert "售后业务处理暂时出现异常" in result["answer"]
