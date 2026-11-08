from app.rag.document_loader import load_markdown_documents
from app.rag.text_splitter import split_documents

documents = load_markdown_documents()
chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100,
)

print(f"文档切分完成，共生成 {len(chunks)} 个文本块\n")

for chunk in chunks:
    print("=" * 70)
    print(f"文本块编号：{chunk['chunk_index']}")
    print(f"来源文件：{chunk['file_name']}")
    print(f"字符长度：{len(chunk['content'])}")
    print("-" * 70)
    print(chunk["content"])
    print()