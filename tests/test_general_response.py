import asyncio
from unittest.mock import AsyncMock

from app.agents.nodes import general_response


def test_general_response_calls_rag(monkeypatch):
    mock_answer = AsyncMock(
        return_value="根据售后政策，已签收商品7天内可以申请普通退货。"
    )

    monkeypatch.setattr(
        "app.agents.nodes.answer_with_rag",
        mock_answer,
    )

    result = asyncio.run(
        general_response({
            "user_message": "商品签收后几天内可以退货？"
        })
    )

    mock_answer.assert_awaited_once_with(
        question="商品签收后几天内可以退货？",
        top_k=5,
        conversation_history=[],
    )

    assert result["status"] == "success"
    assert "7天内" in result["answer"]


def test_general_response_handles_rag_error(monkeypatch):
    mock_answer = AsyncMock(
        side_effect=RuntimeError("RAG 服务异常")
    )

    monkeypatch.setattr(
        "app.agents.nodes.answer_with_rag",
        mock_answer,
    )

    result = asyncio.run(
        general_response({
            "user_message": "如何申请退货？"
        })
    )

    assert result["status"] == "error"
    assert "暂时无法查询售后政策" in result["answer"]


def test_general_response_handles_empty_message():
    result = asyncio.run(
        general_response({
            "user_message": ""
        })
    )

    assert result["status"] == "success"
    assert "请输入" in result["answer"]