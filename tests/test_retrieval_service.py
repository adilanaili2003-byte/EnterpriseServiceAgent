from unittest.mock import patch

import pytest

from app.rag.retrieval_service import (
    build_knowledge_base,
    retrieve_knowledge,
)


@pytest.fixture
def sample_documents():
    return [
        {
            "content": "用户签收商品后七天内，可以申请退货。商品需要保持完好。",
            "source": "after_sales_policy.md",
            "file_name": "after_sales_policy.md",
        },
        {
            "content": "退款申请通过审核后，平台会按照原支付渠道处理退款。",
            "source": "after_sales_policy.md",
            "file_name": "after_sales_policy.md",
        },
    ]


@pytest.fixture
def sample_chunks():
    return [
        {
            "content": "用户签收商品后七天内，可以申请退货。商品需要保持完好。",
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
    ]


def test_build_knowledge_base(
    sample_documents,
    sample_chunks,
):
    fake_embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    with (
        patch(
            "app.rag.retrieval_service.load_markdown_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.retrieval_service.split_documents",
            return_value=sample_chunks,
        ),
        patch(
            "app.rag.retrieval_service.embed_texts",
            return_value=fake_embeddings,
        ),
        patch(
            "app.rag.retrieval_service.get_qdrant_client",
        ) as mock_get_client,
        patch(
            "app.rag.retrieval_service.upsert_chunks",
            return_value=2,
        ) as mock_upsert,
    ):
        mock_client = mock_get_client.return_value
        mock_client.collection_exists.return_value = False

        count = build_knowledge_base()

        assert count == 2
        mock_upsert.assert_called_once()
        mock_client.close.assert_called_once()


def test_build_knowledge_base_without_documents():
    with patch(
        "app.rag.retrieval_service.load_markdown_documents",
        return_value=[],
    ):
        with pytest.raises(
            ValueError,
            match="没有可用的 Markdown 文档",
        ):
            build_knowledge_base()

def test_build_knowledge_base_without_chunks(sample_documents):
    """验证文档切分后没有生成文本块时，能够正确抛出异常。"""

    with (
        patch(
            "app.rag.retrieval_service.load_markdown_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.retrieval_service.split_documents",
            return_value=[],
        ),
    ):
        with pytest.raises(
            ValueError,
            match="文档切分后没有生成有效文本块",
        ):
            build_knowledge_base()

def test_build_knowledge_base_with_empty_embeddings(
    sample_documents,
    sample_chunks,
):
    """验证 Embedding 返回空列表时，能够正确抛出异常。"""

    with (
        patch(
            "app.rag.retrieval_service.load_markdown_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.retrieval_service.split_documents",
            return_value=sample_chunks,
        ),
        patch(
            "app.rag.retrieval_service.embed_texts",
            return_value=[],
        ),
        patch(
            "app.rag.retrieval_service.get_qdrant_client",
        ) as mock_get_client,
    ):
        with pytest.raises(
            ValueError,
            match="Embedding 模型未生成有效向量",
        ):
            build_knowledge_base()

        # Embedding 失败后，不应该创建 Qdrant 客户端。
        mock_get_client.assert_not_called()

def test_build_knowledge_base_closes_client_on_error(
    sample_documents,
    sample_chunks,
):
    """验证知识库写入失败时，Qdrant 客户端仍然会关闭。"""

    fake_embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    with (
        patch(
            "app.rag.retrieval_service.load_markdown_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.retrieval_service.split_documents",
            return_value=sample_chunks,
        ),
        patch(
            "app.rag.retrieval_service.embed_texts",
            return_value=fake_embeddings,
        ),
        patch(
            "app.rag.retrieval_service.get_qdrant_client",
        ) as mock_get_client,
        patch(
            "app.rag.retrieval_service.recreate_collection",
        ),
        patch(
            "app.rag.retrieval_service.upsert_chunks",
            side_effect=RuntimeError("模拟 Qdrant 写入失败"),
        ),
    ):
        mock_client = mock_get_client.return_value

        with pytest.raises(
            RuntimeError,
            match="模拟 Qdrant 写入失败",
        ):
            build_knowledge_base()

        # 即使写入失败，也必须关闭 Qdrant 客户端。
        mock_client.close.assert_called_once()

def test_retrieve_knowledge():
    fake_results = [
        {
            "content": "用户签收商品后七天内，可以申请退货。",
            "source": "after_sales_policy.md",
            "file_name": "after_sales_policy.md",
            "chunk_index": 0,
            "score": 0.95,
        }
    ]

    with (
        patch(
            "app.rag.retrieval_service.embed_query",
            return_value=[1.0, 0.0, 0.0],
        ),
        patch(
            "app.rag.retrieval_service.get_qdrant_client",
        ) as mock_get_client,
        patch(
            "app.rag.retrieval_service.search_similar_chunks",
            return_value=fake_results,
        ) as mock_search,
    ):
        mock_client = mock_get_client.return_value

        results = retrieve_knowledge(
            query="签收后几天可以退货？",
            top_k=3,
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert "七天内" in results[0]["content"]

        mock_search.assert_called_once()
        mock_client.close.assert_called_once()

def test_retrieve_knowledge_without_results():
    """验证没有检索到相关知识时，返回空列表。"""

    with (
        patch(
            "app.rag.retrieval_service.embed_query",
            return_value=[1.0, 0.0, 0.0],
        ),
        patch(
            "app.rag.retrieval_service.get_qdrant_client",
        ) as mock_get_client,
        patch(
            "app.rag.retrieval_service.search_similar_chunks",
            return_value=[],
        ) as mock_search,
    ):
        mock_client = mock_get_client.return_value

        results = retrieve_knowledge(
            query="如何申请商品换货？",
            top_k=3,
        )

        # 没有匹配的知识时，应该返回空列表。
        assert results == []

        # 确保执行了向量检索。
        mock_search.assert_called_once()

        # 确保 Qdrant 客户端正常关闭。
        mock_client.close.assert_called_once()

def test_retrieve_knowledge_with_empty_query():
    with pytest.raises(
        ValueError,
        match="用户查询不能为空",
    ):
        retrieve_knowledge("")


def test_retrieve_knowledge_with_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="top_k 必须大于 0",
    ):
        retrieve_knowledge(
            query="如何申请退款？",
            top_k=0,
        )


def test_retrieve_knowledge_closes_client_on_error():
    with (
        patch(
            "app.rag.retrieval_service.embed_query",
            return_value=[1.0, 0.0, 0.0],
        ),
        patch(
            "app.rag.retrieval_service.get_qdrant_client",
        ) as mock_get_client,
        patch(
            "app.rag.retrieval_service.search_similar_chunks",
            side_effect=RuntimeError("模拟检索失败"),
        ),
    ):
        mock_client = mock_get_client.return_value

        with pytest.raises(RuntimeError, match="模拟检索失败"):
            retrieve_knowledge("如何申请退款？")

        mock_client.close.assert_called_once()