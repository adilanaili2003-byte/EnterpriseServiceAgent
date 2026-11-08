from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient, models


temp_dir = TemporaryDirectory()
client = QdrantClient(path=temp_dir.name)

collection_name = "clear_test"

try:
    print("1. 创建集合")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=3,
            distance=models.Distance.COSINE,
        ),
    )

    print("\n2. 写入三条测试数据")

    points = [
        models.PointStruct(
            id=1,
            vector=[1.0, 0.0, 0.0],
            payload={"text": "测试数据1"},
        ),
        models.PointStruct(
            id=2,
            vector=[0.0, 1.0, 0.0],
            payload={"text": "测试数据2"},
        ),
        models.PointStruct(
            id=3,
            vector=[0.0, 0.0, 1.0],
            payload={"text": "测试数据3"},
        ),
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

    print(
        "写入后数量：",
        client.count(
            collection_name=collection_name,
            exact=True,
        ).count,
    )

    print("\n3. 清空集合中的所有向量")

    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[])
        ),
        wait=True,
    )

    print(
        "清空后数量：",
        client.count(
            collection_name=collection_name,
            exact=True,
        ).count,
    )

finally:
    client.close()
    temp_dir.cleanup()