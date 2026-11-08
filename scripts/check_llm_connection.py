import asyncio

from app.services.llm_service import chat_completion


async def main():
    messages = [
        {
            "role": "system",
            "content": "浣犳槸涓€涓笓涓氱殑浼佷笟鍞悗瀹㈡湇鍔╂墜銆?
        },
        {
            "role": "user",
            "content": "浣犲ソ锛岃绠€鍗曚粙缁嶄竴涓嬩綘鑷繁銆?
        }
    ]

    print("姝ｅ湪璋冪敤 DeepSeek 妯″瀷...")

    reply = await chat_completion(messages)

    print("\n妯″瀷鍥炵瓟锛?)
    print(reply)


if __name__ == "__main__":
    asyncio.run(main())
