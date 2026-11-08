from app.rag.retrieval_service import (
    build_knowledge_base,
    retrieve_knowledge,
)


def main():
    print("=" * 50)
    print("第一步：构建售后政策知识库")
    print("=" * 50)

    count = build_knowledge_base()

    print(f"知识库构建完成，共写入 {count} 个文本块。")

    print("\n" + "=" * 50)
    print("第二步：测试语义检索")
    print("=" * 50)

    questions = [
        "商品签收后几天内可以申请退货？",
        "退款通过审核后，钱会退到哪里？",
        "如果收到的商品存在质量问题怎么办？",
    ]

    for question in questions:
        print(f"\n用户问题：{question}")

        results = retrieve_knowledge(
            query=question,
            top_k=3,
        )

        if not results:
            print("没有检索到相关政策。")
            continue

        for index, result in enumerate(results, start=1):
            print(f"\n检索结果 {index}")
            print(f"相似度：{result['score']:.4f}")
            print(f"来源：{result['source']}")
            print(f"内容：{result['content']}")


if __name__ == "__main__":
    main()