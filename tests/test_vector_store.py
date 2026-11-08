import pytest
from qdrant_client import QdrantClient

from app.rag.vector_store import (
    COLLECTION_NAME,
    create_collection,
    recreate_collection,
    upsert_chunks,
    search_similar_chunks,
)


@pytest.fixture
def qdrant_client(tmp_path):
    """
    创建临时 Qdrant 数据库。

    每次测试使用独立目录，避免污染正式知识库。
    """
    client = QdrantClient(path=str(tmp_path))
    yield client
    client.close()


@pytest.fixture
def sample_chunks():
    return [
        {
            "content": "用户签收商品后七天内，可以申请退货。",
            "source": "after_sales_policy.md",
            "file_name": "after_sales_policy.md",
            "chunk_index": 0,
        },
        {
            "content": "退款申请通过审核后，平台会按照原支付渠道处理退款。",
            "source": "after_sales_policy.md",
            "file_name": "after_sales_policy.md",
            "chunk_index": 1,
        },
        {
            "content": "商品存在质量问题时，用户可以联系平台客服。",
            "source": "after_sales_policy.md",
            "file_name": "after_sales_policy.md",
            "chunk_index": 2,
        },
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_create_collection(qdrant_client):
    create_collection(qdrant_client, vector_size=3)

    assert qdrant_client.collection_exists(COLLECTION_NAME)


def test_upsert_chunks(
    qdrant_client,
    sample_chunks,
    sample_embeddings,
):
    count = upsert_chunks(
        qdrant_client,
        sample_chunks,
        sample_embeddings,
    )

    assert count == 3

    collection_info = qdrant_client.get_collection(COLLECTION_NAME)

    assert collection_info.points_count == 3


def test_search_similar_chunks(
    qdrant_client,
    sample_chunks,
    sample_embeddings,
):
    upsert_chunks(
        qdrant_client,
        sample_chunks,
        sample_embeddings,
    )

    results = search_similar_chunks(
        qdrant_client,
        query_embedding=[1.0, 0.0, 0.0],
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["content"] == sample_chunks[0]["content"]
    assert results[0]["source"] == "after_sales_policy.md"
    assert results[0]["score"] == pytest.approx(1.0)


def test_upsert_empty_chunks(qdrant_client):
    count = upsert_chunks(
        qdrant_client,
        chunks=[],
        embeddings=[],
    )

    assert count == 0


def test_upsert_dimension_mismatch(
    qdrant_client,
    sample_chunks,
):
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]

    with pytest.raises(ValueError, match="维度必须一致"):
        upsert_chunks(
            qdrant_client,
            sample_chunks,
            embeddings,
        )


def test_search_empty_query(qdrant_client):
    with pytest.raises(ValueError, match="查询向量不能为空"):
        search_similar_chunks(
            qdrant_client,
            query_embedding=[],
        )

def test_search_query_dimension_mismatch(qdrant_client):
    """验证查询向量维度与 Qdrant 集合维度不一致时抛出异常。"""

    # 创建一个维度为 3 的 Qdrant 集合。
    create_collection(
        qdrant_client,
        vector_size=3,
    )

    # 使用维度为 2 的查询向量进行检索。
    with pytest.raises(
        ValueError,
        match="查询向量维度为 2",
    ):
        search_similar_chunks(
            client=qdrant_client,
            query_embedding=[1.0, 0.0],
            limit=3,
        )

def test_search_nonexistent_collection(qdrant_client):
    results = search_similar_chunks(
        qdrant_client,
        query_embedding=[1.0, 0.0, 0.0],
    )

    assert results == []

def test_recreate_collection(qdrant_client):
    """验证旧集合可以被删除并重新创建。"""

    create_collection(qdrant_client, vector_size=3)

    assert qdrant_client.collection_exists(COLLECTION_NAME)

    recreate_collection(qdrant_client, vector_size=3)

    assert qdrant_client.collection_exists(COLLECTION_NAME)

    collection_info = qdrant_client.get_collection(COLLECTION_NAME)

    assert collection_info.points_count == 0


def test_recreate_collection_with_invalid_dimension(qdrant_client):
    """验证无效向量维度会被拦截。"""

    with pytest.raises(ValueError, match="vector_size 必须大于 0"):
        recreate_collection(qdrant_client, vector_size=0)


def test_recreate_collection_clears_old_data(
    qdrant_client,
    sample_chunks,
    sample_embeddings,
):
    """验证重建集合后，旧向量数据会被清除。"""

    upsert_chunks(
        qdrant_client,
        sample_chunks,
        sample_embeddings,
    )

    assert (
        qdrant_client.get_collection(COLLECTION_NAME).points_count
        == 3
    )

    recreate_collection(qdrant_client, vector_size=3)

    print("\n========== Qdrant 重建诊断 ==========")

    print("集合列表：")
    print(qdrant_client.get_collections())

    print("get_collection 返回的 points_count：")
    print(qdrant_client.get_collection(COLLECTION_NAME).points_count)

    print("精确统计数量：")
    print(
        qdrant_client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count
    )

    print("实际查询到的向量数据：")
    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    print(points)

    print("====================================\n")

    assert (
        qdrant_client.get_collection(COLLECTION_NAME).points_count
        == 0
    )