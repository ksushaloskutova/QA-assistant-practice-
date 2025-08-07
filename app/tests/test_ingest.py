import os                     # для работы с файловой системой (создание/удаление)
import shutil                 # для удаления тестовой папки после выполнения
import pytest                 # фреймворк для написания тестов

from langchain.schema import Document             # используем для проверки типа
from app import ingest                            # импортируем твой основной скрипт (ingest.py)

# --- Пути к тестовым файлам и БД ---
TEST_DOC_PATH = "app/docs/test_doc.txt"           # путь к тестовому .txt файлу
TEST_QDRANT_PATH = ingest.QDRANT_PATH
TEST_QDRANT_COLLECTION = ingest.COLLECTION_NAME

def test_load_documents():
    docs = ingest.load_documents()                # вызываем функцию загрузки
    assert isinstance(docs, list)                 # проверяем, что возвращается список
    assert len(docs) > 0                          # список не пуст
    assert isinstance(docs[0], Document)          # элемент списка — это Document

@pytest.mark.parametrize("method", ["token", "sentence", "recursive"])
def test_split_text(method):
    docs = ingest.load_documents()                # загружаем документы
    chunks = ingest.split_text(docs, method=method)  # разбиваем указанным методом
    assert isinstance(chunks, list)               # возвращается список
    assert len(chunks) > 0                        # список не пуст
    assert isinstance(chunks[0], Document)        # элементы списка — это Document

def test_save_to_qdrant():
    docs = ingest.load_documents()
    chunks = ingest.split_text(docs, method="token")

    # Сохраняем в Qdrant
    ingest.save_to_qdrant(chunks)

    # Проверяем, что база действительно создана
    assert os.path.exists(TEST_QDRANT_PATH), "Qdrant path не существует"
    assert any(os.scandir(TEST_QDRANT_PATH)), "Qdrant база пуста"

    # Удаляем после теста
    shutil.rmtree(TEST_QDRANT_PATH)



# --- Тест на пустой список документов ---
def test_split_text_empty_input():
    chunks = ingest.split_text([], method="token")
    assert chunks == []  # должен вернуть пустой список без ошибок

# --- Тест на неправильный метод чанкинга ---
def test_invalid_chunking_method():
    with pytest.raises(ValueError) as exc_info:
        ingest.split_text([Document(page_content="тест")], method="not_existing")
    assert "Unknown chunking method" in str(exc_info.value)