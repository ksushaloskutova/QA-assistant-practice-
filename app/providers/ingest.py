import hashlib
import os
import re
import uuid

import torch
from chonkie import RecursiveChunker, SentenceChunker, TokenChunker
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.objects.model_custom_embeddings import E5Embeddings

# === Константы ===
DATA_PATH = "app/docs"
QDRANT_PATH = "app/qdrant_db"
COLLECTION_NAME = "qa_documents"
global_unique_hashes = set()

embedding_model = E5Embeddings(
    model_name=os.getenv("EMBEDDINGS_MODEL_NAME", "intfloat/multilingual-e5-base"),
    device="cuda" if torch.cuda.is_available() else "cpu",  # Используем GPU при наличии
)


# === Предобработка текста ===
def preprocess_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.strip().lower().startswith(("читать", "подробнее", "#"))
    ]
    return "\n".join(cleaned_lines)


# === Превращение текста в документы ===


def text_to_documents(text: str, source: str) -> list[Document]:
    blocks = text.split("\n\n")
    base_name = os.path.basename(source)
    dir_name = os.path.basename(os.path.dirname(source))

    # Попробуем извлечь дату из текста (если она есть)
    date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
    date = date_match.group(1) if date_match else None

    # Генерируем метадату
    metadata_base = {
        "source": source,
        "file": base_name,
        "category": dir_name,
    }
    if date:
        metadata_base["date"] = date

    return [
        Document(page_content=block.strip(), metadata=metadata_base.copy())
        for block in blocks
        if block.strip()
    ]


# === Загрузка и предварительная обработка документов ===
def walk_through_files(path, file_extension='.txt'):
    for dir_path, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents():
    documents = []
    for f_name in walk_through_files(DATA_PATH):
        loader = TextLoader(f_name, encoding="utf-8")
        raw = loader.load()[0].page_content
        cleaned = preprocess_text(raw)
        processed_docs = text_to_documents(cleaned, source=f_name)
        documents.extend(processed_docs)
        print(f"[LOADED] {f_name} → {len(processed_docs)} segments")
    return documents


# === Хэш текста ===
def hash_text(text: str):
    return hashlib.sha256(text.encode()).hexdigest()


# === Выбор чанкера ===
def get_chunker(method: str):
    if method == "token":
        return TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=64)

    elif method == "sentence":
        return SentenceChunker(chunk_size=10, chunk_overlap=2)

    elif method == "recursive":
        try:
            return RecursiveChunker(chunk_size=500, chunk_overlap=100)
        except TypeError:
            return RecursiveChunker()

    elif method == "recursive_char":
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    else:
        raise ValueError(f"Unknown chunking method: {method}")


def _run_splitter(splitter, text: str):
    """Вернуть список строк-чанков, независимо от API чонкера."""
    if hasattr(splitter, "chunk"):  # chonkie.* обычно
        chunks = splitter.chunk(text)
        # могут вернуть объекты с .text или чистые строки
        return [c.text if hasattr(c, "text") else str(c) for c in chunks]

    if hasattr(splitter, "split"):  # некоторые правила
        chunks = splitter.split(text)
        return [c.text if hasattr(c, "text") else str(c) for c in chunks]

    if hasattr(splitter, "split_text"):  # интерфейс типа langchain
        chunks = splitter.split_text(text)
        return [str(c) for c in chunks]

    if hasattr(splitter, "split_documents"):  # как у RecursiveCharacterTextSplitter
        docs = splitter.split_documents([Document(page_content=text)])
        return [d.page_content for d in docs]

    raise TypeError(f"Unsupported splitter API: {type(splitter).__name__}")


# === Разбиение на чанки ===
def split_text(documents: list[Document], method: str):
    splitter = get_chunker(method)
    chunks: list[Document] = []

    if method == "recursive_char":
        # уже отдаёт Documents
        chunks = splitter.split_documents(documents)
    else:
        for doc in documents:
            for chunk_text in _run_splitter(splitter, f"passage: {doc.page_content}"):
                chunk_hash = hash_text(chunk_text)
                if chunk_hash not in global_unique_hashes:
                    chunks.append(
                        Document(page_content=chunk_text, metadata=doc.metadata)
                    )
                    global_unique_hashes.add(chunk_hash)

    print(
        f"Split {len(documents)} documents → {len(chunks)} unique chunks using '{method}' splitter."
    )
    return chunks


# === Сохранение в Qdrant ===
def save_to_qdrant(chunks: list[Document]):
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_model.dim, distance=Distance.COSINE
            ),
        )

    points = []
    for doc in chunks:
        vector = embedding_model.embed_query(doc.page_content)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=doc.metadata | {"text": doc.page_content},
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(
        f"Saved {len(points)} chunks to Qdrant at {QDRANT_PATH} (collection: {COLLECTION_NAME})"
    )


# === Финальный пайплайн ===
def generate_data_store(chunk_method="recursive_char"):
    print("Загрузка документов...")
    documents = load_documents()

    print("Нарезка на чанки...")
    chunks = split_text(documents, method=chunk_method)

    print("Сохранение в Qdrant...")
    save_to_qdrant(chunks)


if __name__ == "__main__":
    generate_data_store(chunk_method="recursive_char")
