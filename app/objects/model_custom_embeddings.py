import os
from typing import List
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

DEFAULT_EMB_MODEL = os.getenv("EMBEDDINGS_MODEL_NAME", "/opt/embeddings/e5_base")

def _ensure_prefixed(text: str, prefix: str) -> str:
    # чтобы не получить "passage: passage: ..." если где-то уже добавили
    t = text.lstrip()
    return t if t.startswith(prefix) else (prefix + t)

class E5Embeddings(Embeddings):
    """
    E5-эмбеддер. ВАЖНО:
    - Документы кодируются с префиксом 'passage: '
    - Запросы кодируются с префиксом 'query: '
    - Вектор L2-нормируется (для cosine)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 64,
    ):
        self.model_name = model_name or DEFAULT_EMB_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

        self.model = SentenceTransformer(self.model_name, device=self.device)
        # Переносить на dtype у SentenceTransformer не всегда корректно — оставим устройство,
        # а dtype пусть выберет backend. Если очень надо, можно задать:
        # self.model.max_seq_length = 512  # на всякий случай (E5 обычно 512)
        try:
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback — обычно 768/1024. Лучше не падать.
            self.dim = getattr(self.model, "embedding_dimension", 768)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        prompts = [_ensure_prefixed(t, self.passage_prefix) for t in texts]
        return self.model.encode(
            prompts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine в Qdrant — нормируем
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        prompt = _ensure_prefixed(text, self.query_prefix)
        return self.model.encode(
            prompt,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
