from __future__ import annotations

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .config import COLLECTION_NAME, QDRANT_PATH


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        path=QDRANT_PATH,
        prefer_grpc=True,
        timeout=10,
    )


def get_vectorstore(client: QdrantClient, embedding) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding,
    )
