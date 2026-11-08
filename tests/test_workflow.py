import asyncio
from uuid import uuid4

from app.agents.workflow import agent_workflow
from app.agents import intent_classifier, nodes


def run_workflow(message):
    """运行一次独立的工作流测试。"""

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


def test_general_intent(monkeypatch):
    """测试普通咨询是否进入普通回答分支。"""

    async def mock_chat_completion(**kwargs):
        return "general"

    async def mock_answer_with_rag(**kwargs):
        return "模拟普通咨询回答。"

    # Mock 意图识别模型调用
    monkeypatch.setattr(
        intent_classifier,
        "chat_completion",
        mock_chat_completion
    )

    # Mock RAG 回答，避免请求真实 DeepSeek API
    monkeypatch.setattr(
        nodes,
        "answer_with_rag",
        mock_answer_with_rag
    )

    result = run_workflow(
        "你好，请介绍一下你们的售后服务。"
    )

    assert result["intent"] == "general"
    assert result["status"] == "success"
    assert result["answer"]


def test_after_sales_intent(monkeypatch):
    """测试售后业务是否进入售后响应分支。"""

    async def mock_chat_completion(**kwargs):
        return "after_sales"

    monkeypatch.setattr(
        intent_classifier,
        "chat_completion",
        mock_chat_completion
    )

    result = run_workflow(
        "我的商品出现质量问题，我想申请退款。"
    )

    assert result["intent"] == "after_sales"
    assert result["status"] == "success"
    assert "售后" in result["answer"]


def test_intent_classification_error(monkeypatch):
    """测试意图识别失败时是否进入错误处理分支。"""

    async def mock_chat_completion(**kwargs):
        raise RuntimeError("模拟模型服务异常")

    monkeypatch.setattr(
        intent_classifier,
        "chat_completion",
        mock_chat_completion
    )

    result = run_workflow(
        "帮我看看这个情况应该怎么处理。"
    )

    assert result["status"] == "error"
    assert "无法处理" in result["answer"]


def test_multiturn_after_sales_memory(monkeypatch):
    """测试同一会话能否记住上一轮的售后申请类型。"""

    async def mock_chat_completion(**kwargs):
        return "after_sales"

    async def mock_answer_with_rag(**kwargs):
        return "模拟售后政策回答。"

    # 模拟意图识别
    monkeypatch.setattr(
        intent_classifier,
        "chat_completion",
        mock_chat_completion
    )

    # 记录实际查询过的订单编号
    queried_order_ids = []

    def mock_get_order_by_id(order_id):
        queried_order_ids.append(order_id)
        return None

    monkeypatch.setattr(
        nodes,
        "get_order_by_id",
        mock_get_order_by_id
    )

    # 记录规则引擎收到的申请类型
    eligibility_request_types = []

    def mock_check_eligibility(**kwargs):
        eligibility_request_types.append(
            kwargs["request_type"]
        )

        return {
            "eligible": False,
            "reason": "模拟订单不存在"
        }

    monkeypatch.setattr(
        nodes,
        "check_after_sales_eligibility",
        mock_check_eligibility
    )

    # 模拟 RAG 回答
    monkeypatch.setattr(
        nodes,
        "answer_with_rag",
        mock_answer_with_rag
    )

    async def run_multiturn():
        # 两轮对话使用相同的 thread_id
        thread_id = f"test-{uuid4()}"

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # 第一轮：用户提出退款申请，但没有提供订单号
        first_result = await agent_workflow.ainvoke(
            {
                "user_message": "我的商品有质量问题，我想申请退款。"
            },
            config=config
        )

        # 第二轮：用户只提供订单号
        second_result = await agent_workflow.ainvoke(
            {
                "user_message": "ORD999999999"
            },
            config=config
        )

        return first_result, second_result

    first_result, second_result = asyncio.run(
        run_multiturn()
    )

    # 验证第一轮保存了退款申请类型
    assert first_result["request_type"] == "refund"
    assert "订单编号" in first_result["answer"]

    # 验证第二轮读取了上一轮保存的退款申请类型
    assert second_result["request_type"] == "refund"

    # 验证第二轮没有再次询问申请类型
    assert "退款还是退货" not in second_result["answer"]

    # 验证第二轮确实使用订单号查询订单
    assert queried_order_ids == ["ORD999999999"]

    # 验证规则引擎收到的是退款申请类型
    assert eligibility_request_types == ["refund"]

def test_multiturn_general_conversation_memory(monkeypatch):
    """测试普通咨询能否记住上一轮对话。"""

    async def mock_chat_completion(**kwargs):
        return "general"

    monkeypatch.setattr(
        intent_classifier,
        "chat_completion",
        mock_chat_completion,
    )

    rag_calls = []

    async def mock_answer_with_rag(**kwargs):
        rag_calls.append(kwargs)

        if kwargs["question"] == "商品签收后几天可以退货？":
            return "已签收商品自实际签收时间起7天内可以提交普通退货申请。"

        return "请结合具体退货政策准备相关材料。"

    monkeypatch.setattr(
        nodes,
        "answer_with_rag",
        mock_answer_with_rag,
    )

    async def run_multiturn():
        thread_id = f"test-{uuid4()}"

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # 第一轮：询问退货期限
        first_result = await agent_workflow.ainvoke(
            {
                "user_message": "商品签收后几天可以退货？"
            },
            config=config,
        )

        # 第二轮：追问需要什么材料
        second_result = await agent_workflow.ainvoke(
            {
                "user_message": "那需要提供什么材料？"
            },
            config=config,
        )

        return first_result, second_result

    first_result, second_result = asyncio.run(
        run_multiturn()
    )

    # 验证第一轮正常回答
    assert first_result["status"] == "success"

    # 验证第二轮正常回答
    assert second_result["status"] == "success"

    # 验证第二轮确实读取了第一轮历史
    second_history = rag_calls[1]["conversation_history"]

    assert second_history[0]["role"] == "user"
    assert second_history[0]["content"] == "商品签收后几天可以退货？"

    assert second_history[1]["role"] == "assistant"
    assert second_history[1]["content"] == first_result["answer"]

    # 验证第二轮问题正确传递
    assert rag_calls[1]["question"] == "那需要提供什么材料？"

    # 验证历史记录包含两轮对话
    assert len(second_result["chat_history"]) == 4