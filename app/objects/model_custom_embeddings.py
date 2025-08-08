from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class E5Embeddings(Embeddings):
    """
    Кастомный эмбеддер под E5-модель (например, intfloat/multilingual-e5-large).
    Работает с prompt-инструкциями "query: " и "passage: ".
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "cpu",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.dim = self.model.get_sentence_embedding_dimension()  # ✅ вот это добавлено

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prompts = [self.passage_prefix + text for text in texts]
        return self.model.encode(prompts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        prompt = self.query_prefix + text
        return self.model.encode(prompt, convert_to_numpy=True).tolist()
