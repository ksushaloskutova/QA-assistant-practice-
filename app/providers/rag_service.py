# Перебрасываем старые импорты на новый сервис — чтобы ничего не ломать.
from .rag.service import initialize_components, query_rag
from .rag.trimming import _trim_docs_by_tokens, _trim_per_doc

__all__ = [
    "initialize_components",
    "query_rag",
    "_trim_per_doc",
    "_trim_docs_by_tokens",
]
