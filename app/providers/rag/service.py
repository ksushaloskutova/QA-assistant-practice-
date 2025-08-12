from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from models.index import ChatMessage
from providers.rag.config import (  # noqa: F401 (MAX_NEW_TOKENS используется в llm)
    MAX_NEW_TOKENS,
    _rag_params,
)

from .embeddings import load_embeddings
from .history import append_pair, get_session_history
from .llm import load_llm_pipeline
from .prompt import _build_prompt, _format_sources, _hist_to_text
from .retriever import get_qdrant_client, get_vectorstore
from .tokenizer import get_tokenizer
from .trimming import _trim_docs_by_tokens, _trim_per_doc

logger = logging.getLogger(__name__)

# === Глобалы сервиса (как было) ===
_initialized = False
_client = None  # QdrantClient
_vectorstore: QdrantVectorStore | None = None
_embedding_model = None
_llm: HuggingFacePipeline | None = None
_TOKENIZER = None
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def initialize_components():
    """Инициализация всех тяжёлых частей один раз (CPU или GPU). Логика сохранена."""
    global _initialized, _client, _vectorstore, _embedding_model, _llm, _TOKENIZER

    if _initialized:
        return

    # потоки CPU
    try:
        if not torch.cuda.is_available():
            torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        pass

    print("[INIT] Tokenizer...")
    _TOKENIZER = get_tokenizer()

    print(
        f"[INIT] LLM (HF pipeline, {'GPU' if torch.cuda.is_available() else 'CPU'})..."
    )
    _llm = load_llm_pipeline(_TOKENIZER)

    print("[INIT] Embeddings (E5)...")
    _embedding_model = load_embeddings()

    print("[INIT] Qdrant client (local path)...")
    _client = get_qdrant_client()

    _vectorstore = get_vectorstore(_client, _embedding_model)

    # ВАЖНО: проставляем токенайзер в модуль trimming (как в исходнике — глобальная переменная)
    import app.providers.rag.trimming as trimming

    trimming._TOKENIZER = _TOKENIZER  # noqa: SLF001 (осознанно)

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


def query_rag(message: ChatMessage, session_id: str = "") -> str:
    """
    Оптимизированный RAG (CPU, использует GPU при наличии):
      1) извлекаем кандидатов (fetch_k) с фильтрацией по score
      2) fallback на MMR если нет результатов
      3) обрезаем документы и собираем контекст
      4) генерируем ответ с кэшированием промптов
    """
    if not _initialized:
        raise RuntimeError(
            "Компоненты не инициализированы. Вызовите initialize_components()."
        )

    # история
    hist = get_session_history(session_id)

    t0 = time.perf_counter()
    timers = {}

    params = _rag_params()

    # 1) Поиск документов
    search_start = time.perf_counter()
    scored = _vectorstore.similarity_search_with_score(
        message.question,
        k=params["FETCH_K"],
        score_threshold=params["SCORE_THRESHOLD"],
        search_params={"hnsw_ef": params["HNSW_EF"], "exact": False},
    )
    timers["search"] = time.perf_counter() - search_start

    # 2) Фильтрация и fallback
    filter_start = time.perf_counter()
    filtered_docs = [
        d for d, s in scored if s >= params["SCORE_THRESHOLD"]
    ]  # как в исходнике
    if not filtered_docs:
        try:
            docs = _vectorstore.max_marginal_relevance_search(
                message.question,
                k=params["TOP_K_DOCS"],
                fetch_k=params["FETCH_K"],
                lambda_mult=params["MMR_LAMBDA"],
                timeout=5.0,
            )
        except Exception:
            docs = []
    else:
        docs = filtered_docs[: params["TOP_K_DOCS"]]
    timers["filter"] = time.perf_counter() - filter_start

    # 3) Подготовка контекста
    trim_start = time.perf_counter()
    try:
        docs_copy = list(docs)
        # ВРЕМЕННО: последовательная обрезка (как в твоём текущем файле)
        trimmed_docs = list(map(_trim_per_doc, docs_copy))
        from .config import MAX_INPUT_TOKENS

        context_text = _trim_docs_by_tokens(trimmed_docs, MAX_INPUT_TOKENS)
        history_text = _hist_to_text(hist, max_pairs=2)
    except Exception as e:
        logger.error(f"Context preparation failed: {str(e)}")
        trimmed_docs = []
        context_text = ""
        history_text = ""
    timers["trim"] = time.perf_counter() - trim_start

    # 4) Генерация ответа
    gen_start = time.perf_counter()
    try:
        prompt = _build_prompt(context_text, message.question, history_text)
        answer = _llm.invoke(prompt, timeout=30.0)  # как было
        answer = (answer or "").strip()
        if answer:
            answer += _format_sources(trimmed_docs)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        answer = "Извините, не удалось обработать запрос. Пожалуйста, попробуйте позже."
    timers["generate"] = time.perf_counter() - gen_start

    # 5) История и лог
    append_pair(session_id, message.question, answer)
    timers["total"] = time.perf_counter() - t0
    logger.info(
        "RAG timings: " + " | ".join([f"{k}: {v:.2f}s" for k, v in timers.items()])
    )

    return answer
