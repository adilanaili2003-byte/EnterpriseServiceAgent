from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model = None


def get_embedding_model():
    """
    加载 Embedding 模型。

    使用单例方式，避免重复加载模型。
    """
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    将多条文本转换为向量。

    Args:
        texts: 待向量化的文本列表。

    Returns:
        文本对应的向量列表。
    """
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("文本列表中不能包含空字符串")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    将用户查询转换为向量。
    """
    if not query or not query.strip():
        raise ValueError("查询内容不能为空")

    model = get_embedding_model()

    embedding = model.encode(
        query.strip(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()