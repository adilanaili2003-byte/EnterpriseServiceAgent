import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.rag.answer_service import (
    answer_with_rag,
    format_retrieved_context,
)


def test_format_retrieved_context():
    results = [
        {
            "content": "用户签收商品后七天内，可以申请退货。",
            "source": "after_sales_policy.md",
            "score": 0.95,
        },
        {
            "content": "退款申请通过审核后，按照原支付渠道退款。",
            "source": "after_sales_policy.md",
            "score": 0.88,
        },
    ]

    context = format_retrieved_context(results)

    assert "七天内" in context
    assert "原支付渠道" in context
    assert "after_sales_policy.md" in context
    assert "0.9500" in context


def test_format_empty_context():
    context = format_retrieved_context([])

    assert "没有检索到相关政策内容" in context


def test_answer_with_rag():
    fake_results = [
        {
            "content": "用户签收商品后七天内，可以申请退货。",
            "source": "after_sales_policy.md",
            "score": 0.95,
        }
    ]

    fake_answer = "根据售后政策，用户签收商品后七天内可以申请退货。"

    with (
        patch(
            "app.rag.answer_service.retrieve_knowledge",
            return_value=fake_results,
        ) as mock_retrieve,
        patch(
            "app.rag.answer_service.chat_completion",
            new_callable=AsyncMock,
        ) as mock_chat,
    ):
        mock_chat.return_value = fake_answer

        answer = asyncio.run(
            answer_with_rag("签收后几天可以退货？")
        )

        assert answer == fake_answer

        mock_retrieve.assert_called_once_with(
            query="签收后几天可以退货？",
            top_k=3,
        )

        mock_chat.assert_awaited_once()

        messages = mock_chat.await_args.args[0]

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "七天内" in messages[1]["content"]
        assert "after_sales_policy.md" in messages[1]["content"]


def test_answer_with_empty_question():
    with pytest.raises(
        ValueError,
        match="用户问题不能为空",
    ):
        asyncio.run(answer_with_rag(""))


def test_answer_with_rag_when_llm_returns_empty():
    fake_results = [
        {
            "content": "用户签收商品后七天内，可以申请退货。",
            "source": "after_sales_policy.md",
            "score": 0.95,
        }
    ]

    with (
        patch(
            "app.rag.answer_service.retrieve_knowledge",
            return_value=fake_results,
        ),
        patch(
            "app.rag.answer_service.chat_completion",
            new_callable=AsyncMock,
            return_value="",
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="没有生成有效回答",
        ):
            asyncio.run(
                answer_with_rag("签收后几天可以退货？")
            )

def test_answer_with_business_context():
    business_context = "订单符合基础退款申请条件，但尚未经过人工审核。"

    with (
        patch(
            "app.rag.answer_service.retrieve_knowledge",
            return_value=[],
        ),
        patch(
            "app.rag.answer_service.chat_completion",
            new_callable=AsyncMock,
            return_value="订单符合基础退款申请条件，但仍需人工审核。",
        ) as mock_chat,
    ):
        answer = asyncio.run(
            answer_with_rag(
                question="我的订单可以退款吗？",
                business_context=business_context,
            )
        )

        messages = mock_chat.await_args.args[0]

        assert answer
        assert business_context in messages[1]["content"]
        assert "不得擅自改变后端业务判断结果" in messages[1]["content"]