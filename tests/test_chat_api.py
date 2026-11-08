import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import chat as chat_module

client = TestClient(app)


def test_chat_success(monkeypatch):
    """测试 LangGraph 工作流正常返回结果。"""

    async def mock_ainvoke(state,config=None):
        return {
            "user_message": state["user_message"],
            "intent": "general",
            "answer": "您好，我是企业智能售后客服助手。",
            "status": "success",
            "error": ""
        }

    monkeypatch.setattr(
        chat_module.agent_workflow,
        "ainvoke",
        mock_ainvoke
    )

    response = client.post(
        "/api/v1/chat",
        json={"message": "你好"}
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "您好，我是企业智能售后客服助手。"


def test_chat_empty_message():
    """测试空消息校验。"""

    response = client.post(
        "/api/v1/chat",
        json={"message": ""}
    )

    assert response.status_code == 422


def test_chat_workflow_error(monkeypatch):
    """测试 LangGraph 工作流执行失败。"""

    async def mock_ainvoke(state):
        return {
            "answer": "抱歉，系统暂时无法处理您的请求，请稍后重试。",
            "status": "error"
        }

    monkeypatch.setattr(
        chat_module.agent_workflow,
        "ainvoke",
        mock_ainvoke
    )

    response = client.post(
        "/api/v1/chat",
        json={"message": "查询我的订单"}
    )

    assert response.status_code == 502

def test_chat_api_with_real_workflow(monkeypatch):
    """测试 API 到真实 LangGraph 售后业务工作流。"""

    from uuid import uuid4
    from app.agents import nodes

    captured = {}

    async def mock_answer_with_rag(**kwargs):
        captured.update(kwargs)
        return (
            "订单符合基础售后申请条件，"
            "可以进入人工审核流程。"
        )

    # 只隔离外部 RAG 回答生成，不 Mock 售后业务逻辑。
    monkeypatch.setattr(
        nodes,
        "answer_with_rag",
        mock_answer_with_rag
    )

    session_id = f"e2e-{uuid4()}"

    response = client.post(
        "/api/v1/chat",
        json={
            "message": (
                "我的订单 ORD202611001 "
                "有质量问题，我想申请退款。"
            ),
            "session_id": session_id
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["session_id"] == session_id
    assert data["reply"]

    # 验证真实售后业务处理结果。
    assert "订单查询结果：订单存在" in captured["business_context"]
    assert "符合基础申请条件" in captured["business_context"]
    assert captured["question"] == "我的订单 ORD202611001 有质量问题，我想申请退款。"
    assert captured["top_k"] == 3