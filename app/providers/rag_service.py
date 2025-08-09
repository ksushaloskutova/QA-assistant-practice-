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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.objects.model_custom_embeddings import E5Embeddings

from ..models.index import ChatMessage

logger = logging.getLogger(__name__)


# ==================== КОНФИГ ====================
# Имя/путь модели и токенизатора можно переопределить через env.
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "microsoft/phi-2")
TOKENIZER_NAME = os.getenv("LLM_TOKENIZER_NAME", MODEL_NAME)
QUANTIZATION = os.getenv("LLM_QUANTIZATION", "8bit")  # 4bit/8bit/none
QDRANT_PATH = os.path.abspath("app/qdrant_db")  # локальная папка Qdrant
COLLECTION_NAME = "qa_documents"  # коллекция, созданная ingest

TOP_K_DOCS = 2  # финальное кол-во чанков в контексте
FETCH_K = 8  # сколько кандидатов тянем для отбора
SCORE_THRESHOLD = 0.25  # порог отсечения нерелевантного (подбирается)
PER_DOC_LIMIT = 400  # лимит токенов на 1 документ до склейки
MAX_INPUT_TOKENS = 800  # суммарный лимит токенов контекста
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
MMR_LAMBDA = 0.5  # диверсификация MMR (0..1)

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
    _TOKENIZER = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True
    )

    print("[INIT] LLM (quantized)...")
    if MODEL_NAME.endswith(".gguf"):
        from langchain_community.llms import LlamaCpp

        _llm = LlamaCpp(
            model_path=MODEL_NAME,
            n_ctx=MAX_INPUT_TOKENS,
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
        )
    else:
        model_kwargs = {"trust_remote_code": True}
        if QUANTIZATION == "4bit":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                load_in_4bit=True,
                device_map="auto",
                **model_kwargs,
            )
        elif QUANTIZATION == "8bit":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                load_in_8bit=True,
                device_map="auto",
                **model_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                **model_kwargs,
            )

        _gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=_TOKENIZER,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=False,
        )
        _llm = HuggingFacePipeline(pipeline=_gen)

    print("[INIT] Embeddings (E5)...")
    _embedding_model = E5Embeddings(
        model_name="intfloat/multilingual-e5-large",
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
      1) извлекаем кандидатов (FETCH_K) с фильтрацией по score
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

    # 1) Поиск документов с оптимизированными параметрами
    search_start = time.perf_counter()
    scored = _vectorstore.similarity_search_with_score(
        message.question,
        k=FETCH_K,
        score_threshold=SCORE_THRESHOLD,
        search_params={
            "hnsw_ef": 32,  # ИСПРАВЛЕНО: Уменьшено для скорости
            "exact": False,
        },
    )
    timers["search"] = time.perf_counter() - search_start

    # 2) Фильтрация и fallback логика
    filter_start = time.perf_counter()
    filtered_docs = [d for d, s in scored if s >= SCORE_THRESHOLD]

    if not filtered_docs:
        # ИСПРАВЛЕНО: Добавлен таймаут для MMR
        try:
            docs = _vectorstore.max_marginal_relevance_search(
                message.question,
                k=TOP_K_DOCS,
                fetch_k=FETCH_K,
                lambda_mult=MMR_LAMBDA,
                timeout=5.0,  # Максимум 5 секунд на MMR
            )
        except Exception:
            docs = []
    else:
        docs = filtered_docs[:TOP_K_DOCS]
    timers["filter"] = time.perf_counter() - filter_start

    # 3) Подготовка контекста
    trim_start = time.perf_counter()
    try:
        # ИСПРАВЛЕНО: Параллельная обрезка документов
        with ThreadPoolExecutor(max_workers=4) as executor:
            docs = list(executor.map(_trim_per_doc, docs))

        context_text = _trim_docs_by_tokens(docs, MAX_INPUT_TOKENS)
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
            answer += _format_sources(docs)
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
