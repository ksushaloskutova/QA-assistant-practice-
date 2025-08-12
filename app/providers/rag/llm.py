from __future__ import annotations

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from .config import MAX_NEW_TOKENS


def load_llm_pipeline(tokenizer) -> HuggingFacePipeline:
    """Создание HF text-generation pipeline (GPU при наличии, иначе CPU). Логика повторяет исходник."""
    device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        torch.backends.cudnn.benchmark = (
            True  # как в исходнике (хотя на трансформерах почти без эффекта)
        )

    gen = pipeline(
        "text-generation",
        model=tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else None,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16 if device >= 0 else torch.float32,
        model_kwargs={
            "use_cache": True,
        },
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)
