from __future__ import annotations

from transformers import AutoTokenizer

from .config import MODEL_DIR

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    return _TOKENIZER
