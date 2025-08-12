from __future__ import annotations

import os

import torch
from objects.model_custom_embeddings import E5Embeddings


def load_embeddings():
    embedding_model_name = os.getenv(
        "EMBEDDINGS_MODEL_NAME", "intfloat/multilingual-e5-base"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return E5Embeddings(model_name=embedding_model_name, device=device)
