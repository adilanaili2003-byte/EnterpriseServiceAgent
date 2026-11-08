from app.rag.retrieval_service import retrieve_knowledge


# 检索评估数据集
# expected_chunks 表示与问题相关的预期 Chunk 编号
EVALUATION_DATASET = [
    {
        "id": "Q1",
        "query": "商品签收后几天可以退货？",
        "expected_chunks": [2],
    },
    {
        "id": "Q2",
        "query": "商品已经签收超过7天，还能申请普通退货吗？",
        "expected_chunks": [2],
    },
    {
        "id": "Q3",
        "query": "申请退货需要满足什么条件？",
        "expected_chunks": [3],
    },
    {
        "id": "Q4",
        "query": "申请退货需要提供哪些材料？",
        "expected_chunks": [1, 2, 3],
    },
    {
        "id": "Q5",
        "query": "退货申请提交之后就算成功了吗？",
        "expected_chunks": [2, 10],
    },
    {
        "id": "Q6",
        "query": "退货具体要经过哪些步骤？",
        "expected_chunks": [4],
    },
    {
        "id": "Q7",
        "query": "退款申请提交成功后，什么时候能到账？",
        "expected_chunks": [7, 6, 10],
    },
    {
        "id": "Q8",
        "query": "商品有质量问题，需要提供什么证据？",
        "expected_chunks": [8],
    },
    {
        "id": "Q9",
        "query": "什么情况下可以申请换货？",
        "expected_chunks": [9],
    },
    {
        "id": "Q10",
        "query": "哪些售后情况需要人工审核？",
        "expected_chunks": [10],
    },
]


def evaluate_single_query(
    item: dict,
    top_k: int = 5,
) -> dict:
    """
    评估单条查询的检索效果。
    """

    query = item["query"]
    expected_chunks = set(item["expected_chunks"])

    results = retrieve_knowledge(
        query=query,
        top_k=top_k,
    )

    retrieved_chunks = [
        result["chunk_index"]
        for result in results
    ]

    # 找到第一个相关 Chunk 的排名
    first_relevant_rank = None

    for rank, chunk_index in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if chunk_index in expected_chunks:
            first_relevant_rank = rank
            break

    # Hit@3
    hit_at_3 = int(
        any(
            chunk_index in expected_chunks
            for chunk_index in retrieved_chunks[:3]
        )
    )

    # Hit@5
    hit_at_5 = int(
        any(
            chunk_index in expected_chunks
            for chunk_index in retrieved_chunks[:5]
        )
    )

    # MRR
    mrr = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        else 0
    )

    return {
        "id": item["id"],
        "query": query,
        "expected_chunks": sorted(expected_chunks),
        "results": results,
        "retrieved_chunks": retrieved_chunks,
        "first_relevant_rank": first_relevant_rank,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "mrr": mrr,
    }


def main():
    """
    执行完整检索评估。
    """

    evaluation_results = []

    print("\n" + "=" * 80)
    print("RAG 真实检索质量评估")
    print("=" * 80)

    for item in EVALUATION_DATASET:
        result = evaluate_single_query(
            item=item,
            top_k=5,
        )

        evaluation_results.append(result)

        print(f"\n[{result['id']}] {result['query']}")

        print(
            f"预期 Chunk：{result['expected_chunks']}"
        )

        print(
            f"实际 Chunk：{result['retrieved_chunks']}"
        )

        for rank, chunk in enumerate(
            result["results"],
            start=1,
        ):
            print(
                f"  Top{rank} | "
                f"Chunk={chunk['chunk_index']} | "
                f"Score={chunk['score']:.4f}"
            )

        print(
            f"Hit@3={result['hit_at_3']} | "
            f"Hit@5={result['hit_at_5']} | "
            f"MRR={result['mrr']:.4f}"
        )

    # 计算整体指标
    total = len(evaluation_results)

    avg_hit_at_3 = (
        sum(r["hit_at_3"] for r in evaluation_results)
        / total
    )

    avg_hit_at_5 = (
        sum(r["hit_at_5"] for r in evaluation_results)
        / total
    )

    avg_mrr = (
        sum(r["mrr"] for r in evaluation_results)
        / total
    )

    print("\n" + "=" * 80)
    print("整体评估结果")
    print("=" * 80)

    print(f"测试问题数量：{total}")
    print(f"Hit@3：{avg_hit_at_3:.4f}")
    print(f"Hit@5：{avg_hit_at_5:.4f}")
    print(f"MRR：{avg_mrr:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()