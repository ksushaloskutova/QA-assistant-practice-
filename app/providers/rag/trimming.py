from __future__ import annotations

from typing import List

from langchain_core.documents import Document

# ВАЖНО: как в исходнике — используем глобальный токенайзер, который проставит service.initialize_components()
_TOKENIZER = None


def _trim_per_doc(doc: Document) -> Document:
    """Обрезаем длинный документ до PER_DOC_LIMIT токенов (экономим контекст)."""
    ids = _TOKENIZER.encode(doc.page_content, add_special_tokens=False)
    from .config import (  # импорт внутри — чтобы избежать циклов при инициализации
        PER_DOC_LIMIT,
    )

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
