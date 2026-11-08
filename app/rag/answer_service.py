from app.rag.retrieval_service import retrieve_knowledge
from app.services.llm_service import chat_completion


SYSTEM_PROMPT = """
你是一名专业、严谨的企业智能售后客服。

你的任务是根据提供的企业售后政策知识回答用户问题。

请严格遵守以下规则：

1. 优先依据提供的政策知识回答问题。
2. 不允许编造政策中没有提及的内容。
3. 如果知识库没有提供足够的信息，明确告诉用户目前无法根据现有政策确认。
4. 如果不同政策内容存在冲突，不要自行选择或编造结论，应提示人工客服核实。
5. 回答应当清晰、简洁、礼貌。
6. 不要把政策中的模拟规则描述为普遍适用的法律规定。
7. 回答涉及具体订单的退款资格、订单状态或退款操作时，
   不得仅凭政策文本作出业务审批，应交由后端业务规则和人工审核处理。
8. 如果提供了后端业务判断结果，必须以该结果作为具体订单业务判断的依据，不得擅自修改。
9. 如果提供了业务判断结果，应结合政策知识解释原因，但不能将基础申请资格描述为最终审批结果。
10. 如果检索到相关政策，在回答末尾标注参考来源。
11. 可以结合历史对话理解用户当前问题中的指代和省略信息。
12. 当前用户问题是回答的重点，历史对话仅用于辅助理解。
13. 历史对话不能替代知识库中的政策依据。

参考来源格式：
参考政策：after_sales_policy.md
"""


def format_retrieved_context(results: list[dict]) -> str:
    """
    将检索结果整理成 LLM 可理解的上下文。
    """

    if not results:
        return "知识库中没有检索到相关政策内容。"

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"""
政策片段 {index}
来源：{result.get("source", "未知")}
相似度：{result.get("score", 0):.4f}
内容：
{result.get("content", "")}
"""
        )

    return "\n".join(context_parts)


def format_conversation_history(
    conversation_history: list[dict] | None,
) -> str:
    """
    将历史对话整理成大模型可以理解的格式。
    """

    if not conversation_history:
        return "暂无历史对话。"

    history_parts = []

    for item in conversation_history:
        role = item.get("role", "")
        content = item.get("content", "")

        if role == "user":
            role_name = "用户"
        elif role == "assistant":
            role_name = "客服"
        else:
            continue

        if content:
            history_parts.append(
                f"{role_name}：{content}"
            )

    if not history_parts:
        return "暂无历史对话。"

    return "\n".join(history_parts)


def build_retrieval_query(
    question: str,
    conversation_history: list[dict] | None,
) -> str:
    """
    结合最近的用户问题构建检索文本。

    主要用于解决多轮对话中的指代问题。
    """

    previous_questions = []

    if conversation_history:
        for item in conversation_history:
            if item.get("role") == "user":
                content = item.get("content", "").strip()

                if content:
                    previous_questions.append(content)

    # 只取最近三轮用户问题
    previous_questions = previous_questions[-3:]

    query_parts = previous_questions + [question]

    return "\n".join(query_parts)


async def answer_with_rag(
    question: str,
    top_k: int = 3,
    business_context: str | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    使用 RAG 检索结果辅助大模型回答问题。

    business_context:
        可选的后端业务判断结果。

    conversation_history:
        可选的历史对话记录。
    """

    if not question or not question.strip():
        raise ValueError("用户问题不能为空")

    # 第一步：结合历史对话构建检索文本
    retrieval_query = build_retrieval_query(
        question=question,
        conversation_history=conversation_history,
    )

    # 第二步：检索知识库
    results = retrieve_knowledge(
        query=retrieval_query,
        top_k=top_k,
    )

    context = format_retrieved_context(results)

    # 第三步：整理历史对话
    history_context = format_conversation_history(
        conversation_history
    )

    business_info = (
        business_context.strip()
        if business_context and business_context.strip()
        else "未提供后端业务判断结果。"
    )

    # 第四步：组织大模型输入
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"""
历史对话：
{history_context}

当前用户问题：
{question}

检索到的企业政策知识：
{context}

后端业务判断结果：
{business_info}

请结合历史对话、政策知识和后端业务判断结果回答用户。

注意：
1. 优先回答当前用户问题。
2. 可以利用历史对话理解当前问题的上下文。
3. 回答必须有政策依据。
4. 如果没有足够的政策依据，请明确说明，不要编造。
5. 不得擅自改变后端业务判断结果。
""",
        },
    ]

    # 第五步：调用大模型
    answer = await chat_completion(messages)

    if not answer:
        raise RuntimeError("RAG 问答服务没有生成有效回答")

    return answer