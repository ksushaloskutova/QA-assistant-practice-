# prof_bench.py
import logging
import os
import random
import statistics as st
import time

from line_profiler import LineProfiler

import app.providers.rag_service as rag

# важно: путь может отличаться у тебя в проекте
from app.models.index import ChatMessage
from app.providers.rag_service import (
    _trim_docs_by_tokens,
    _trim_per_doc,
    initialize_components,
    query_rag,
)

# -------- логирование таймеров из rag_service --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)  # потише


# --------- последовательный "executоr" для профилирования trim ----------
class _SeqExecutor:
    def map(self, func, iterable):
        return list(map(func, iterable))


def _warmup():
    # один раз инициализация (вне профилирования запроса!)
    initialize_components()
    # прогрев модели/кэшей, чтобы не считать холодный старт
    for _ in range(2):
        query_rag(ChatMessage(question="прогрев модели"), session_id="bench")


def profile_trim_once():
    """Один запрос, где тримминг идёт без потоков, чтобы line_profiler увидел время."""
    old_exec = rag._EXECUTOR
    try:
        rag._EXECUTOR = _SeqExecutor()  # временная подмена
        query_rag(
            ChatMessage(question="Что такое AI Talented Hub?"), session_id="profile"
        )
    finally:
        rag._EXECUTOR = old_exec  # вернуть как было


def batch_bench(n_runs: int = 20):
    qs = [
        "Какие документы нужны для поступления?",
        "Сроки подачи заявлений?",
        "Есть ли бюджетные места?",
        "Какие курсы по ИИ есть в программе?",
        "Что такое AI Talented Hub?",
    ]
    times = []
    for _ in range(n_runs):
        q = random.choice(qs)
        t0 = time.perf_counter()
        query_rag(ChatMessage(question=q), session_id="bench")
        times.append(time.perf_counter() - t0)

    p50 = st.median(times)
    p95 = sorted(times)[int(0.95 * (len(times) - 1))]
    print(
        f"runs={len(times)} | avg={sum(times)/len(times):.3f}s | p50={p50:.3f}s | p95={p95:.3f}s"
    )


def main():
    _warmup()

    # ---- line_profiler по функциям тримминга ----
    lp = LineProfiler()
    lp.add_function(_trim_per_doc)
    lp.add_function(_trim_docs_by_tokens)
    lp_wrapper = lp(profile_trim_once)
    lp_wrapper()
    lp.print_stats()

    # ---- мини-бенч с таймерами из rag_service ----
    batch_bench(n_runs=20)


if __name__ == "__main__":
    # (необязательно) чуть уменьшим длину ответа для скорости бенча
    os.environ.setdefault("RAG_TOP_K_DOCS", "2")
    os.environ.setdefault("RAG_FETCH_K", "8")
    # запусти
    main()
