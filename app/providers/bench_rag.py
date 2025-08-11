# bench_rag.py
import random
import statistics as st
import time

from app.models.index import ChatMessage  # если путь иной, поправь
from app.providers.rag_service import initialize_components, query_rag


def run_once(q: str) -> float:
    t0 = time.perf_counter()
    query_rag(ChatMessage(question=q), session_id="bench")
    return time.perf_counter() - t0


def main():
    initialize_components()
    # прогрев (важно!)
    for _ in range(3):
        query_rag(ChatMessage(question="привет, это прогрев"), session_id="bench")

    qs = [
        "Какие документы нужны для поступления?"
        # "Сроки подачи заявлений?",
        # "Есть ли бюджетные места?",
        # "Какие курсы по ИИ есть в программе?",
    ]
    times = []
    for _ in range(20):
        q = random.choice(qs)
        times.append(run_once(q))

    print(
        f"runs={len(times)}, avg={sum(times) / len(times):.3f}s, "
        f"p50={st.median(times):.3f}s, p95={sorted(times)[int(len(times) * 0.95) - 1]:.3f}s"
    )  # Исправлено: добавлена закрывающая кавычка


if __name__ == "__main__":
    main()
