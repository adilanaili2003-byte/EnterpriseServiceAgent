import re
from typing import TypedDict

from app.rag.document_loader import Document


class DocumentChunk(TypedDict):
    content: str
    source: str
    file_name: str
    chunk_index: int


def split_markdown_sections(content: str) -> list[str]:
    """
    按 Markdown 标题层级切分文档。

    子标题切分时保留所有父标题，避免章节上下文丢失。
    """
    sections: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    current_section: list[str] = []
    has_content = False

    for line in content.splitlines():
        match = re.match(r"^(#{1,6})\s+", line)

        if match:
            if current_section and has_content:
                section = "\n".join(current_section).strip()
                if section:
                    sections.append(section)

            level = len(match.group(1))

            heading_stack = [
                item for item in heading_stack
                if item[0] < level
            ]

            heading_stack.append((level, line))

            current_section = [
                heading for _, heading in heading_stack
            ]
            has_content = False

        else:
            current_section.append(line)

            if line.strip():
                has_content = True

    if current_section and has_content:
        section = "\n".join(current_section).strip()
        if section:
            sections.append(section)

    return sections


def split_long_section(
    section: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    对过长章节进行递归式切分。

    优先保留段落，其次保留句子，最后才按字符切分。
    """
    if len(section) <= chunk_size:
        return [section]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", section)
        if paragraph.strip()
    ]

    if len(paragraphs) > 1:
        return merge_text_units(paragraphs, chunk_size, chunk_overlap)

    sentences = re.split(r"(?<=[。！？；.!?;])\s*|\n", section)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if len(sentences) > 1:
        return merge_text_units(sentences, chunk_size, chunk_overlap)

    return split_by_characters(section, chunk_size, chunk_overlap)


def merge_text_units(
    units: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    合并段落或句子，在尽量保持完整性的同时控制文本块长度。
    """
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(unit) > chunk_size:
            sub_chunks = split_long_section(
                unit,
                chunk_size,
                chunk_overlap,
            )
            chunks.extend(sub_chunks[:-1])
            current = sub_chunks[-1] if sub_chunks else ""
        else:
            current = unit

    if current:
        chunks.append(current.strip())

    return chunks


def split_by_characters(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    最后的兜底策略：按照字符窗口切分。
    """
    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """
    根据 Markdown 结构切分文档。

    优先按标题和段落切分，过长内容再按句子或字符切分。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: list[DocumentChunk] = []

    for document in documents:
        content = document["content"].strip()

        if not content:
            continue

        sections = split_markdown_sections(content)

        document_chunks: list[str] = []

        for section in sections:
            section_chunks = split_long_section(
                section,
                chunk_size,
                chunk_overlap,
            )
            document_chunks.extend(section_chunks)

        for chunk_index, chunk_content in enumerate(document_chunks):
            chunks.append({
                "content": chunk_content,
                "source": document["source"],
                "file_name": document["file_name"],
                "chunk_index": chunk_index,
            })

    return chunks