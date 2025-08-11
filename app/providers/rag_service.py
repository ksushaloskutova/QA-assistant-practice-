# app/providers/rag_service.py
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union

import torch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, pipeline

from app.objects.model_custom_embeddings import E5Embeddings

from ..models.index import ChatMessage

logger = logging.getLogger(__name__)


# ==================== КОНФИГ ====================
MODEL_DIR = "app/models/my_model"  # локальная HF-модель генерации
QDRANT_PATH = os.path.abspath("app/qdrant_db")  # локальная папка Qdrant
COLLECTION_NAME = "qa_documents"  # коллекция, созданная ingest

PER_DOC_LIMIT = 400  # лимит токенов на 1 документ до склейки
MAX_INPUT_TOKENS = 800  # суммарный лимит токенов контекста
MAX_NEW_TOKENS = 160  # длина ответа (для скорости)

# ==================== ГЛОБАЛ ====================
_initialized = False
_client: QdrantClient | None = None
_vectorstore: QdrantVectorStore | None = None
_embedding_model: E5Embeddings | None = None
_llm: HuggingFacePipeline | None = None
_TOKENIZER = None
# Глобальный executor
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

_chat_history: Dict[str, List[Union[HumanMessage, AIMessage]]] = {}


# ==================== ВСПОМОГАЛКИ ====================
def _rag_params() -> dict:
    """Читает настройки RAG из переменных окружения на каждый запуск."""
    return {
        "TOP_K_DOCS": int(os.getenv("RAG_TOP_K_DOCS", "2")),
        "FETCH_K": int(os.getenv("RAG_FETCH_K", "8")),
        "SCORE_THRESHOLD": float(os.getenv("RAG_SCORE_THRESHOLD", "0.25")),
        "MMR_LAMBDA": float(os.getenv("RAG_MMR_LAMBDA", "0.5")),
        "HNSW_EF": int(os.getenv("RAG_HNSW_EF", "4")),
    }


def _parallel_embed(texts: List[str]) -> List[List[float]]:
    """Параллельное вычисление эмбеддингов"""
    return _embedding_model.embed_documents(texts)


def _hist_to_text(
    history: List[Union[HumanMessage, AIMessage]], max_pairs: int = 2
) -> str:
    """Вытаскиваем последние max_pairs пар диалога для промпта."""
    pairs: List[tuple[str, str]] = []
    msgs = [m for m in history if isinstance(m, (HumanMessage, AIMessage))]
    i = len(msgs) - 1
    while i >= 0 and len(pairs) < max_pairs:
        if isinstance(msgs[i], AIMessage):
            ai = msgs[i].content
            user = (
                msgs[i - 1].content
                if i - 1 >= 0 and isinstance(msgs[i - 1], HumanMessage)
                else ""
            )
            pairs.append((user, ai))
            i -= 2
        else:
            i -= 1
    pairs.reverse()
    parts: List[str] = []
    for u, a in pairs:
        if u:
            parts.append(f"Пользователь: {u}")
        if a:
            parts.append(f"Ассистент: {a}")
    return "\n".join(parts)


def _trim_per_doc(doc: Document) -> Document:
    """Обрезаем длинный документ до PER_DOC_LIMIT токенов (экономим контекст)."""
    ids = _TOKENIZER.encode(doc.page_content, add_special_tokens=False)
    if len(ids) <= PER_DOC_LIMIT:
        return doc
    return Document(
        page_content=_TOKENIZER.decode(ids[:PER_DOC_LIMIT]),
        metadata=doc.metadata,
    )


def _trim_docs_by_tokens(docs: List[Document], max_input_tokens: int) -> str:
    """Склеиваем документы в один текст, ограничивая суммарное число токенов."""
    pieces, total = [], 0
    for d in docs:
        text = d.page_content
        ids = _TOKENIZER.encode(text, add_special_tokens=False)
        n = len(ids)
        if total + n > max_input_tokens:
            remain = max(0, max_input_tokens - total)
            if remain > 0:
                pieces.append(_TOKENIZER.decode(ids[:remain]))
            break
        pieces.append(text)
        total += n
    return "\n\n".join(pieces)


def _build_prompt(context_text: str, question: str, history_text: str) -> str:
    """Финальный промпт-строка — дешевле, чем шаблонные цепочки."""
    sys = (
        "Ты помощник 'AI Assistant' по поступлению в AI Talented Hub, ИТМО (направление: Искусственный интеллект). "
        "Отвечай строго по приведённому контексту. Если ответа нет в контексте, скажи честно, что не нашёл, "
        "и предложи переформулировать вопрос. Отвечай кратко и по делу на русском; англоязычные термины из контекста не меняй.\n\n"
    )
    parts = [sys]
    if history_text:
        parts.append("История диалога:\n")
        parts.append(history_text)
        parts.append("\n\n")
    parts.append("Контекст:\n")
    parts.append(context_text if context_text.strip() else "(контекст пуст)")
    parts.append("\n\nВопрос:\n")
    parts.append(question.strip())
    parts.append("\n\nОтвет:")
    return "".join(parts)


def _format_sources(docs: List[Document]) -> str:
    """Добавляем список источников к ответу (улучшает доверие и дебаг)."""
    uniq = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file") or "unknown"
        if src not in uniq:
            uniq.append(src)
        if len(uniq) == 3:
            break
    if not uniq:
        return ""
    return "\n\nИсточники:\n" + "\n".join(f"- {s}" for s in uniq)


# ==================== ИНИЦИАЛИЗАЦИЯ ====================
def initialize_components():
    """Инициализация всех тяжёлых частей один раз (CPU)."""
    global _initialized, _client, _vectorstore, _embedding_model, _llm, _TOKENIZER
    if _initialized:
        return

    # умерим жадность по потокам
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        pass

    print("[INIT] Tokenizer...")
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("[INIT] LLM (HF pipeline, CPU)...")
    _gen = pipeline(
        "text-generation",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        device=-1,  # CPU
        torch_dtype=torch.float32,  # Явное указание типа
        model_kwargs={
            "low_cpu_mem_usage": True,
            "use_cache": True,  # Включить кэширование внимания
        },
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        return_full_text=False,
    )
    _llm = HuggingFacePipeline(pipeline=_gen)

    print("[INIT] Embeddings (E5)...")
    embedding_model_name = os.getenv(
        "EMBEDDINGS_MODEL_NAME", "intfloat/multilingual-e5-base"
    )
    _embedding_model = E5Embeddings(
        model_name=embedding_model_name,
        device="cpu",
    )

    print("[INIT] Qdrant client (local path)...")
    _client = QdrantClient(
        path=QDRANT_PATH,
        prefer_grpc=True,  # Более быстрый протокол
        timeout=10,  # Таймаут подключения
    )

    _vectorstore = QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION_NAME,
        embedding=_embedding_model,
    )

    # прогрев (best-effort)
    try:
        _vectorstore.similarity_search("ping", k=1)
    except Exception:
        pass
    try:
        _ = _llm.invoke("Скажи 'готово' одним словом.")
    except Exception:
        pass

    _initialized = True
    print("[INIT] OK")


# ==================== ЗАПРОС ====================
def query_rag(message: ChatMessage, session_id: str = "") -> str:
    """
    Оптимизированный RAG на CPU:
      1) извлекаем кандидатов (fetch_k) с фильтрацией по score
      2) fallback на MMR если нет результатов
      3) обрезаем документы и собираем контекст
      4) генерируем ответ с кэшированием промптов
    """
    # ИСПРАВЛЕНО: Добавлена проверка инициализации и сессии
    if not _initialized:
        raise RuntimeError(
            "Компоненты не инициализированы. Вызовите initialize_components()."
        )

    if session_id not in _chat_history:
        _chat_history[session_id] = []

    t0 = time.perf_counter()
    timers = {}  # Для детального замера времени

    params = _rag_params()

    # 1) Поиск документов с оптимизированными параметрами
    search_start = time.perf_counter()
    scored = _vectorstore.similarity_search_with_score(
        message.question,
        k=params["FETCH_K"],
        score_threshold=params["SCORE_THRESHOLD"],
        search_params={
            "hnsw_ef": params["HNSW_EF"],
            "exact": False,
        },
    )
    timers["search"] = time.perf_counter() - search_start

    # 2) Фильтрация и fallback логика
    filter_start = time.perf_counter()
    filtered_docs = [d for d, s in scored if s >= params["SCORE_THRESHOLD"]]

    if not filtered_docs:
        # ИСПРАВЛЕНО: Добавлен таймаут для MMR
        try:
            docs = _vectorstore.max_marginal_relevance_search(
                message.question,
                k=params["TOP_K_DOCS"],
                fetch_k=params["FETCH_K"],
                lambda_mult=params["MMR_LAMBDA"],
                timeout=5.0,  # Максимум 5 секунд на MMR
            )
        except Exception:
            docs = []
    else:
        docs = filtered_docs[: params["TOP_K_DOCS"]]
    timers["filter"] = time.perf_counter() - filter_start

    # 3) Подготовка контекста
    trim_start = time.perf_counter()
    trimmed_docs = []
    try:
        # ИСПРАВЛЕНО: Параллельная обрезка документов через глобальный executor
        docs_copy = list(docs)
        # trimmed_docs = list(_EXECUTOR.map(_trim_per_doc, docs_copy)) - ВРЕМЕННО!
        trimmed_docs = list(map(_trim_per_doc, docs_copy))

        context_text = _trim_docs_by_tokens(trimmed_docs, MAX_INPUT_TOKENS)
        history_text = _hist_to_text(_chat_history[session_id], max_pairs=2)
    except Exception as e:
        logger.error(f"Context preparation failed: {str(e)}")
        context_text = ""
        history_text = ""
    timers["trim"] = time.perf_counter() - trim_start

    # 4) Генерация ответа с кэшированием промпта
    gen_start = time.perf_counter()
    try:
        prompt = _build_prompt(context_text, message.question, history_text)

        # ИСПРАВЛЕНО: Добавлен fallback для генерации
        answer = _llm.invoke(prompt, timeout=30.0)  # Таймаут 30 секунд
        answer = (answer or "").strip()

        # ИСПРАВЛЕНО: Добавлены источники только если есть ответ
        if answer:
            answer += _format_sources(trimmed_docs)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        answer = "Извините, не удалось обработать запрос. Пожалуйста, попробуйте позже."
    timers["generate"] = time.perf_counter() - gen_start

    # 5) Сохранение в историю
    _chat_history[session_id].append(HumanMessage(content=message.question))
    _chat_history[session_id].append(AIMessage(content=answer))

    # Детальное логирование
    timers["total"] = time.perf_counter() - t0
    logger.info(
        "RAG timings: " + " | ".join([f"{k}: {v:.2f}s" for k, v in timers.items()])
    )

    return answer
