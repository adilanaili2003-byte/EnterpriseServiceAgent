import asyncio

from app.agents.intent_classifier import classify_intent


async def main():

    test_messages = [
        "你好，请介绍一下你自己",
        "我的商品收到后出现质量问题",
        "我想申请退款",
        "你们支持哪些支付方式？",
        "我的订单什么时候发货？"
    ]

    for message in test_messages:

        state = {
            "user_message": message
        }

        result = await classify_intent(state)

        print("-" * 50)
        print("用户输入：", message)
        print("识别意图：", result["intent"])


if __name__ == "__main__":
    asyncio.run(main())