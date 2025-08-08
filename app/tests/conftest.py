# tests/conftest.py
import os
import shutil
import textwrap

import pytest
from langchain.schema import Document

from app.providers import ingest, scapper


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


class FakeAsyncHtmlLoader:
    """
    Полностью офлайн-замена AsyncHtmlLoader.
    Возвращает два HTML-документа с "мусорными" блоками и полезным контентом.
    """

    def __init__(self, links):
        self.links = links

    def load(self):
        html1 = textwrap.dedent(
            """
            <html><body>
              <div class="main-header">HEADER TO DROP</div>
              <div>Полезный текст №1</div>
              <div class="blog-article-menu">MENU TO DROP</div>
            </body></html>
            """
        )
        html2 = textwrap.dedent(
            """
            <html><body>
              <div class="breadcrumbs">BREADCRUMBS TO DROP</div>
              <p>Полезный текст №2 с <a href="http://example.com">ссылкой</a> и <img src="x.jpg"/></p>
              <div class="new-footer">FOOTER TO DROP</div>
            </body></html>
            """
        )
        return [Document(page_content=html1), Document(page_content=html2)]


@pytest.fixture(autouse=True)
def patch_constants_and_loader(tmp_path, monkeypatch):
    """
    - Подменяем пути FILE_TO_PARSE и DIR_TO_STORE на временные (tmp_path).
    - Подменяем AsyncHtmlLoader на FakeAsyncHtmlLoader.
    """
    links_file = tmp_path / "links.txt"
    out_dir = tmp_path / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(scapper, "FILE_TO_PARSE", str(links_file))
    monkeypatch.setattr(scapper, "DIR_TO_STORE", str(out_dir))
    monkeypatch.setattr(scapper, "AsyncHtmlLoader", FakeAsyncHtmlLoader)

    return {"links_file": links_file, "out_dir": out_dir}
