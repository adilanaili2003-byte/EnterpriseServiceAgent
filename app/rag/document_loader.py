from pathlib import Path
from typing import TypedDict


BASE_DIR = Path(__file__).resolve().parents[2]

POLICY_DIR = BASE_DIR / "data" / "knowledge_base" / "policies"


class Document(TypedDict):
    content: str
    source: str
    file_name: str


def load_markdown_documents(
    directory: Path = POLICY_DIR,
) -> list[Document]:
    """加载指定目录下的 Markdown 文档。"""

    if not directory.exists():
        raise FileNotFoundError(f"知识库目录不存在：{directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"指定路径不是目录：{directory}")

    documents: list[Document] = []

    for file_path in sorted(directory.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append({
            "content": content,
            "source": (
                str(file_path.relative_to(BASE_DIR))
                if file_path.is_relative_to(BASE_DIR)
                else file_path.name
            ),
            "file_name": file_path.name,
        })

    return documents