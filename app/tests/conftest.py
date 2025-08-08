# tests/conftest.py
import os
import shutil

import pytest

# путь подстрой под свой проект:
from app.providers import ingest


@pytest.fixture(autouse=True)
def _reset_state():
    """Сбрасываем глобальный кэш между тестами, чтобы не получать 0 чанков."""
    ingest.global_unique_hashes.clear()
    yield
    ingest.global_unique_hashes.clear()


@pytest.fixture(scope="module", autouse=True)
def create_test_doc():
    """
    Создаём один тестовый файл на модуль.
    ВАЖНО: здесь не используем monkeypatch (scope mismatch).
    """
    test_dir = os.path.join("app", "docs")
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "test_doc.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Это тестовый текст.\n" * 10)
    yield
    try:
        os.remove(test_file)
    except FileNotFoundError:
        pass


class FakeEmbeddings:
    """Лёгкая заглушка, чтобы тесты не тянули HF-модель и сеть."""

    dim = 32

    def embed_query(self, text: str):
        h = abs(hash(text))
        return [(h % 1000 + i) / 1000.0 for i in range(self.dim)]


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch):
    """
    Function-scoped: безопасно использует monkeypatch.
    Подменяем настоящий эмбеддер на фейковый.
    """
    monkeypatch.setattr(ingest, "embedding_model", FakeEmbeddings())
    yield


@pytest.fixture
def qdrant_cleanup():
    """Чистим локальную БД Qdrant перед/после теста, где нужно."""
    path = ingest.QDRANT_PATH
    if os.path.exists(path):
        shutil.rmtree(path)
    yield
    if os.path.exists(path):
        shutil.rmtree(path)
