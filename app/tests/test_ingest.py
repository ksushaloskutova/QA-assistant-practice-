# tests/test_ingest.py
import os

import pytest
from langchain.schema import Document

# путь подстрой под свой проект:
from app.providers import ingest


def test_load_documents():
    docs = ingest.load_documents()
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert "source" in docs[0].metadata
    assert "file" in docs[0].metadata
    assert "category" in docs[0].metadata


@pytest.mark.parametrize("method", ["token", "sentence", "recursive", "recursive_char"])
def test_split_text(method):
    docs = ingest.load_documents()
    chunks = ingest.split_text(docs, method=method)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)


def test_split_text_empty_input():
    chunks = ingest.split_text([], method="token")
    assert chunks == []


def test_invalid_chunking_method():
    with pytest.raises(ValueError) as exc_info:
        ingest.split_text([Document(page_content="тест")], method="not_existing")
    assert "Unknown chunking method" in str(exc_info.value)


def test_save_to_qdrant(qdrant_cleanup):
    docs = ingest.load_documents()
    chunks = ingest.split_text(docs, method="token")
    ingest.save_to_qdrant(chunks)

    assert os.path.exists(ingest.QDRANT_PATH), "Qdrant path не существует"
    assert any(os.scandir(ingest.QDRANT_PATH)), "Qdrant база пуста"
