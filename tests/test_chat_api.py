from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import chat as chat_module


client = TestClient(app)


def test_chat_success(monkeypatch):
    """测试正常聊天请求"""

    async def mock_chat_completion(messages):
        return "您好，我是企业智能售后客服助手。"

    monkeypatch.setattr(
        chat_module,
        "chat_completion",
        mock_chat_completion
    )

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "你好"
        }
    )

    assert response.status_code == 200
    assert response.json()["reply"] == (
        "您好，我是企业智能售后客服助手。"
    )


def test_chat_empty_message():
    """测试空消息验证"""

    response = client.post(
        "/api/v1/chat",
        json={
            "message": ""
        }
    )

    assert response.status_code == 422


def test_chat_llm_error(monkeypatch):
    """测试大模型调用失败"""

    async def mock_chat_completion(messages):
        raise RuntimeError("模拟模型调用失败")

    monkeypatch.setattr(
        chat_module,
        "chat_completion",
        mock_chat_completion
    )

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "我的订单出现了问题"
        }
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "AI 服务暂时不可用，请稍后重试。"
    )