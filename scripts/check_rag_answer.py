import asyncio

from app.rag.answer_service import answer_with_rag


async def main():
    questions = [
        "商品签收后几天内可以申请退货？",
        "退款审核通过后，钱会退到哪里？",
        "如果收到的商品存在质量问题，应该怎么办？",
        "如果商品已经超过七天，还能申请无理由退货吗？",
    ]

    for index, question in enumerate(questions, start=1):
        print("\n" + "=" * 60)
        print(f"测试问题 {index}")
        print("=" * 60)

        print(f"\n用户问题：{question}")

        try:
            answer = await answer_with_rag(question)

            print(f"\n智能客服回答：\n{answer}")

        except Exception as error:
            print(f"\n测试失败：{type(error).__name__}: {error}")


if __name__ == "__main__":
    asyncio.run(main())