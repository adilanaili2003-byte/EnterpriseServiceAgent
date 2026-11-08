from app.rag.document_loader import load_markdown_documents
from app.rag.text_splitter import split_documents
from app.rag.embedding_service import embed_texts, embed_query
from app.rag.vector_store import (
    get_qdrant_client,
    upsert_chunks,
    search_similar_chunks,
    recreate_collection,
)


def build_knowledge_base() -> int:
    """
    构建售后政策知识库。

    执行流程：
    1. 加载 Markdown 文档。
    2. 将文档切分成文本块。
    3. 使用 Embedding 模型生成向量。
    4. 将文本块和向量保存到 Qdrant。

    Returns:
        成功写入的文本块数量。
    """
    documents = load_markdown_documents()

    if not documents:
        raise ValueError("知识库中没有可用的 Markdown 文档")

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    if not chunks:
        raise ValueError("文档切分后没有生成有效文本块")

    texts = [chunk["content"] for chunk in chunks]

    embeddings = embed_texts(texts)

    # 检查 Embedding 是否成功生成向量
    if not embeddings:
        raise ValueError("Embedding 模型未生成有效向量")

    client = get_qdrant_client()

    try:
        vector_size = len(embeddings[0])

        recreate_collection(
            client=client,
            vector_size=vector_size,
        )

        count = upsert_chunks(
            client=client,
            chunks=chunks,
            embeddings=embeddings,
        )

        return count

    finally:
        client.close()


def retrieve_knowledge(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    根据用户问题检索相关售后政策。

    Args:
        query: 用户提出的问题。
        top_k: 返回最相关的文本块数量。

    Returns:
        检索结果列表。
    """
    if not query or not query.strip():
        raise ValueError("用户查询不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    query_embedding = embed_query(query)

    client = get_qdrant_client()

    try:
        results = search_similar_chunks(
            client=client,
            query_embedding=query_embedding,
            limit=top_k,
        )

        return results
    finally:
        client.close()