import hashlib
import os

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_qdrant import Qdrant
from contextlib import closing

from app.objects.model_custom_embeddings import E5Embeddings

from chonkie import (
    TokenChunker, SentenceChunker, RecursiveChunker,
    RecursiveRules, SemanticChunker, SDPMChunker, LateChunker
)

# === Константы ===
DATA_PATH = "app/docs"
QDRANT_PATH = "app/qdrant_db"
COLLECTION_NAME = "qa_documents"
global_unique_hashes = set()

embedding_model = E5Embeddings(
    model_name="intfloat/multilingual-e5-large",
    device="cpu"
)
from qdrant_client import QdrantClient



# === Чанкеры ===
def get_chunker(method: str):
    if method == "token":
        return TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=64)
    elif method == "sentence":
        return SentenceChunker(chunk_size=10, chunk_overlap=2)
    elif method == "recursive":
        return RecursiveChunker(chunk_size=500, chunk_overlap=100)
    elif method == "rules":
        return RecursiveRules(chunk_size=500, chunk_overlap=100)
    else:
        raise ValueError(f"Unknown chunking method: {method}")

# === Загрузка документов ===
def walk_through_files(path, file_extension='.txt'):
    for dir_path, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)

def load_documents():
    documents = []
    for f_name in walk_through_files(DATA_PATH):
        loader = TextLoader(f_name, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            print(f"[LOADED] {f_name} - {len(doc.page_content)} chars")
        documents.extend(loaded)
    return documents

# === Хэш текста ===
def hash_text(text: str):
    return hashlib.sha256(text.encode()).hexdigest()

# === Разбиение на чанки ===
def split_text(documents: list[Document], method: str):
    splitter = get_chunker(method)
    chunks = []
    for doc in documents:
        raw_chunks = splitter.chunk(f"passage: {doc.page_content}")
        for chunk_text in raw_chunks:
            chunk_hash = hash_text(chunk_text.text)
            if chunk_hash not in global_unique_hashes:
                chunks.append(Document(page_content=chunk_text.text, metadata=doc.metadata))
                global_unique_hashes.add(chunk_hash)
    print(f"Split {len(documents)} documents → {len(chunks)} unique chunks using '{method}' splitter.")
    return chunks

# === Сохранение в Qdrant ===
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

def save_to_qdrant(chunks: list[Document]):
    # Настройка клиента в локальном режиме
    client = QdrantClient(path=QDRANT_PATH)

    # Создание коллекции (если нет)
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_model.dim,  # размерность вектора
                distance=Distance.COSINE
            )
        )

    # Подготовка данных
    points = []
    for doc in chunks:
        vector = embedding_model.embed_query(doc.page_content)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=doc.metadata | {"text": doc.page_content}
        )
        points.append(point)

    # Запись в коллекцию
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Saved {len(points)} chunks to Qdrant at {QDRANT_PATH} (collection: {COLLECTION_NAME})")


def generate_data_store(chunk_method="token"):
    print("Загрузка документов...")
    documents = load_documents()

    print("Нарезка на чанки...")
    chunks = split_text(documents, method=chunk_method)

    print("Сохранение в Qdrant...")
    save_to_qdrant(chunks)

if __name__ == "__main__":
    generate_data_store(chunk_method="token")
