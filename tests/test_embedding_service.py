import pytest

from app.rag.embedding_service import embed_query, embed_texts


def test_embed_texts():
    """测试批量文本向量化。"""
    texts = [
        "商品存在质量问题，可以申请退款。",
        "用户可以申请退货。",
    ]

    embeddings = embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == len(embeddings[1])


def test_embed_query():
    """测试用户查询向量化。"""
    embedding = embed_query("商品坏了怎么办？")

    assert isinstance(embedding, list)
    assert len(embedding) > 0


def test_embed_empty_list():
    """测试空文本列表。"""
    embeddings = embed_texts([])

    assert embeddings == []


def test_embed_empty_text():
    """测试空文本。"""
    with pytest.raises(ValueError):
        embed_texts(["商品质量问题", "   "])


def test_embed_empty_query():
    """测试空查询。"""
    with pytest.raises(ValueError):
        embed_query("   ")