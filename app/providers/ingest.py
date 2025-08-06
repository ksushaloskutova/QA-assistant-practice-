# Langchain dependencies
import hashlib                         # стандартный модуль для подсчёта SHA-256 хэша
import os                              # работа с файловой системой (пути, обход)
import shutil                          # высокоуровневые операции с файлами/директориями

from langchain_community.document_loaders import TextLoader           # загрузка .txt → Document
from langchain.text_splitter import MarkdownTextSplitter              # (не используется здесь) — разрезание markdown по структуре
from langchain.schema import Document                                 # тип Document (содержит текст и метаданные)
from langchain_community.vectorstores import Chroma                   # векторная БД Chroma (локальная)
from langchain_huggingface import HuggingFaceEmbeddings               # эмбеддер на основе модели HuggingFace

from langchain_text_splitters import (                                # импорт различных вариантов чанкеров
    TokenChunker, SentenceChunker, RecursiveChunker,
    RecursiveRules, SemanticChunker, SDPMChunker, LateChunker
)

# === Модель эмбеддингов (E5) ===
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",                      # название модели из Hugging Face
    model_kwargs={"device": "cuda"},                                  # устройство (cuda/gpu или cpu)
    encode_kwargs={"prompt": "passage: "},                            # промпт для документов
    query_encode_kwargs={"prompt": "query: "}                         # промпт для запросов
)

# Path to the directory to save Chroma database
CHROMA_PATH = "app/db_metadata_v5"      # папка, куда будет сохранён Chroma-индекс
DATA_PATH = "app/docs"                  # папка с исходными очищенными .txt
global_unique_hashes = set()           # глобальный набор хэшей для дедупликации чанков

# === Варианты чанкеров ===
def get_chunker(method: str):
    if method == "token":                                                    # чанкер по токенам
        return TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=64)
    elif method == "sentence":                                               # чанкер по предложениям
        return SentenceChunker(chunk_size=10, chunk_overlap=2)
    elif method == "recursive":                                              # recursive по символам (заголовки, параграфы, строки)
        return RecursiveChunker(chunk_size=500, chunk_overlap=100)
    elif method == "rules":                                                 # правила на основе структуры текста (MD, заголовки и т.д.)
        return RecursiveRules(chunk_size=500, chunk_overlap=100)
    elif method == "semantic":                                              # семантическое разбиение на основе эмбеддингов
        return SemanticChunker(embedding_model)
    elif method == "sdpm":                                                  # семантический декомпозитор (структурная декомпозиция)
        return SDPMChunker(embedding_model)
    elif method == "late":                                                  # "поздний" семантический чанкер (после линейной нарезки)
        return LateChunker(embedding_model)
    else:
        raise ValueError(f"Unknown chunking method: {method}")              # если передан неизвестный метод — ошибка

# Функция обхода папок и поиска .txt файлов
def walk_through_files(path, file_extension='.txt'):
    for (dir_path, dir_names, filenames) in os.walk(path):                 # рекурсивно обходим каталог
        for filename in filenames:
            if filename.endswith(file_extension):                          # фильтрация по расширению
                yield os.path.join(dir_path, filename)                     # возвращаем полный путь к файлу

# Загрузка всех документов из указанной папки
def load_documents():
    """
    Загрузить все .txt из DATA_PATH и превратить в LangChain-Document.
    Возврат: список Document.
    """
    documents = []                                                         # итоговый список документов
    for f_name in walk_through_files(DATA_PATH):                           # перебираем все файлы
        document_loader = TextLoader(f_name, encoding="utf-8")             # создаём загрузчик
        documents.extend(document_loader.load())                           # загружаем документы и добавляем в список
    return documents                                                       # возвращаем все документы

# Получение хэша для проверки уникальности чанка
def hash_text(text):
    hash_object = hashlib.sha256(text.encode())                            # вычисляем SHA256
    return hash_object.hexdigest()                                         # возвращаем hex-строку

# Разделение документов на чанки и удаление дубликатов
def split_text(documents: list[Document], method: str):
    splitter = get_chunker(method)                                         # получаем нужный чанкер
    chunks = splitter.chunk_documents(documents)                           # применяем разбиение

    unique_chunks = []                                                     # список для уникальных чанков
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)                         # хэшируем содержимое
        if chunk_hash not in global_unique_hashes:                         # проверяем на дубликат
            unique_chunks.append(chunk)                                    # добавляем в список
            global_unique_hashes.add(chunk_hash)                           # сохраняем хэш

    print(f"Split {len(documents)} documents → {len(unique_chunks)} unique chunks using '{method}' splitter.")
    return unique_chunks                                                   # возвращаем уникальные чанки

# === Сохранение в Chroma ===
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):                                        # если папка с БД уже существует
        shutil.rmtree(CHROMA_PATH)                                         # удаляем её
    db = Chroma.from_documents(                                            # создаём Chroma БД
        documents=chunks,                                                  # список чанков
        embedding=embedding_model,                                         # эмбеддер
        persist_directory=CHROMA_PATH                                      # путь для хранения
    )
    db.persist()
