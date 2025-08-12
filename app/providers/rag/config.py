from __future__ import annotations

import os

# === Константы (как в исходнике) ===
MODEL_DIR = os.getenv("MODEL_DIR", "/models/my_model")
QDRANT_PATH = os.path.abspath("app/qdrant_db")  # локальная папка Qdrant
COLLECTION_NAME = "qa_documents"  # коллекция, созданная ingest

PER_DOC_LIMIT = 400  # лимит токенов на 1 документ до склейки
MAX_INPUT_TOKENS = 800  # суммарный лимит токенов контекста
MAX_NEW_TOKENS = 160  # длина ответа (для скорости)


def _rag_params() -> dict:
    """Чтение RAG-параметров из ENV (как было)."""
    return {
        "TOP_K_DOCS": int(os.getenv("RAG_TOP_K_DOCS", "2")),
        "FETCH_K": int(os.getenv("RAG_FETCH_K", "8")),
        "SCORE_THRESHOLD": float(os.getenv("RAG_SCORE_THRESHOLD", "0.25")),
        "MMR_LAMBDA": float(os.getenv("RAG_MMR_LAMBDA", "0.5")),
        "HNSW_EF": int(os.getenv("RAG_HNSW_EF", "4")),
    }
