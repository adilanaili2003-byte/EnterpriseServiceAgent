from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

COLLECTION_NAME = "after_sales_policies"


def get_qdrant_client() -> QdrantClient:
    """
    获取本地 Qdrant 客户端。

    使用本地持久化模式，无需启动 Docker。
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    return QdrantClient(path=str(VECTOR_STORE_DIR))


def create_collection(
    client: QdrantClient,
    vector_size: int,
) -> None:
    """
    创建向量集合。

    如果集合已经存在，则不重复创建。
    """
    if vector_size <= 0:
        raise ValueError("vector_size 必须大于 0")

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    将文本块及其向量保存到 Qdrant。

    Args:
        client: Qdrant 客户端。
        chunks: 文本块列表。
        embeddings: 对应的向量列表。

    Returns:
        成功写入的文本块数量。
    """
    if len(chunks) != len(embeddings):
        raise ValueError("文本块数量与向量数量不一致")

    if not chunks:
        return 0

    vector_size = len(embeddings[0])

    if vector_size == 0:
        raise ValueError("Embedding 向量不能为空")

    for embedding in embeddings:
        if len(embedding) != vector_size:
            raise ValueError("所有 Embedding 向量的维度必须一致")

    create_collection(client, vector_size)

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "file_name": chunk["file_name"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)

def recreate_collection(
    client: QdrantClient,
    vector_size: int,
) -> None:
    """
    清空旧向量数据，确保集合存在。

    用于知识库全量重建，避免重复写入旧数据。
    """
    if vector_size <= 0:
        raise ValueError("vector_size 必须大于 0")

    if not client.collection_exists(COLLECTION_NAME):
        create_collection(client, vector_size)
        return

    collection_info = client.get_collection(COLLECTION_NAME)
    existing_vector_size = collection_info.config.params.vectors.size

    if existing_vector_size != vector_size:
        raise ValueError(
            f"现有集合向量维度为 {existing_vector_size}，"
            f"当前 Embedding 维度为 {vector_size}，不一致。"
        )

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[])
        ),
        wait=True,
    )


def search_similar_chunks(
    client: QdrantClient,
    query_embedding: list[float],
    limit: int = 3,
) -> list[dict]:
    """
    根据用户问题的向量检索相似文本块。

    Args:
        client: Qdrant 客户端。
        query_embedding: 用户问题对应的向量。
        limit: 返回结果数量。

    Returns:
        包含文本内容、来源和相似度分数的列表。
    """
    if not query_embedding:
        raise ValueError("查询向量不能为空")

    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    if not client.collection_exists(COLLECTION_NAME):
        return []

    # 获取集合信息，检查查询向量维度是否匹配。
    collection_info = client.get_collection(COLLECTION_NAME)

    existing_vector_size = (
        collection_info.config.params.vectors.size
    )

    if len(query_embedding) != existing_vector_size:
        raise ValueError(
            f"查询向量维度为 {len(query_embedding)}，"
            f"但 Qdrant 集合向量维度为 {existing_vector_size}。"
        )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    )

    search_results = []

    for point in result.points:
        payload = point.payload or {}

        search_results.append({
            "content": payload.get("content", ""),
            "source": payload.get("source", ""),
            "file_name": payload.get("file_name", ""),
            "chunk_index": payload.get("chunk_index", -1),
            "score": point.score,
        })

    return search_results