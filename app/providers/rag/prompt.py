from __future__ import annotations

from typing import List, Union

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


def _hist_to_text(
    history: List[Union[HumanMessage, AIMessage]], max_pairs: int = 2
) -> str:
    pairs = []
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
    parts = []
    for u, a in pairs:
        if u:
            parts.append(f"Пользователь: {u}")
        if a:
            parts.append(f"Ассистент: {a}")
    return "\n".join(parts)


def _build_prompt(context_text: str, question: str, history_text: str) -> str:
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
