from pathlib import Path

import pytest

from app.rag.document_loader import load_markdown_documents


def test_load_existing_policy_document():
    """测试能否成功读取售后政策文档。"""
    documents = load_markdown_documents()

    assert len(documents) >= 1

    document = documents[0]

    assert document["content"]
    assert document["file_name"] == "after_sales_policy.md"
    assert "售后服务政策" in document["content"]
    assert document["source"].endswith("after_sales_policy.md")


def test_load_multiple_markdown_documents(tmp_path: Path):
    """测试能否读取多个 Markdown 文件。"""
    (tmp_path / "policy_a.md").write_text(
        "# 政策 A",
        encoding="utf-8"
    )

    (tmp_path / "policy_b.md").write_text(
        "# 政策 B",
        encoding="utf-8"
    )

    documents = load_markdown_documents(tmp_path)

    assert len(documents) == 2
    assert documents[0]["file_name"] == "policy_a.md"
    assert documents[1]["file_name"] == "policy_b.md"


def test_skip_empty_markdown_document(tmp_path: Path):
    """测试跳过空 Markdown 文件。"""
    (tmp_path / "empty.md").write_text(
        "   ",
        encoding="utf-8"
    )

    documents = load_markdown_documents(tmp_path)

    assert documents == []


def test_missing_directory(tmp_path: Path):
    """测试知识库目录不存在时是否抛出异常。"""
    missing_dir = tmp_path / "nonexistent_knowledge_base"

    with pytest.raises(FileNotFoundError):
        load_markdown_documents(missing_dir)