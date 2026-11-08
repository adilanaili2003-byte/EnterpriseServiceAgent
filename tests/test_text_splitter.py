import pytest

from app.rag.text_splitter import split_documents


def create_document(content: str):
    """创建测试用文档。"""
    return {
        "content": content,
        "source": "test_policy.md",
        "file_name": "test_policy.md",
    }


def test_split_long_document():
    """测试长文档能否正确切分。"""
    document = create_document("A" * 1000)

    chunks = split_documents(
        [document],
        chunk_size=500,
        chunk_overlap=100,
    )

    assert len(chunks) == 3
    assert len(chunks[0]["content"]) == 500
    assert len(chunks[1]["content"]) == 500
    assert len(chunks[2]["content"]) == 200


def test_chunk_overlap():
    """测试相邻文本块是否存在重叠。"""
    document = create_document("A" * 1000)

    chunks = split_documents(
        [document],
        chunk_size=500,
        chunk_overlap=100,
    )

    assert chunks[0]["content"][-100:] == chunks[1]["content"][:100]


def test_preserve_document_metadata():
    """测试文本块是否保留原始文档信息。"""
    document = create_document("企业售后服务政策")

    chunks = split_documents([document])

    assert chunks[0]["source"] == "test_policy.md"
    assert chunks[0]["file_name"] == "test_policy.md"
    assert chunks[0]["chunk_index"] == 0


def test_skip_empty_document():
    """测试空文档是否被跳过。"""
    document = create_document("   ")

    chunks = split_documents([document])

    assert chunks == []


def test_invalid_chunk_size():
    """测试非法切分长度。"""
    document = create_document("测试内容")

    with pytest.raises(ValueError):
        split_documents([document], chunk_size=0)


def test_invalid_chunk_overlap():
    """测试非法重叠长度。"""
    document = create_document("测试内容")

    with pytest.raises(ValueError):
        split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=100,
        )

def test_preserve_parent_heading():
    """测试切分 Markdown 子章节时是否保留父标题。"""
    document = create_document(
        """## 2. 退货政策

#### 退货期限说明

已签收商品自实际签收时间起7天内，可以提交普通退货申请。

超过签收后7天的订单，不符合普通退货规则。"""
    )

    chunks = split_documents(
        [document],
        chunk_size=500,
        chunk_overlap=100,
    )

    assert len(chunks) == 1

    content = chunks[0]["content"]

    assert "## 2. 退货政策" in content
    assert "#### 退货期限说明" in content
    assert "已签收商品自实际签收时间起7天内" in content