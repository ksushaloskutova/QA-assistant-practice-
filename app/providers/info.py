from qdrant_client import QdrantClient

from app.objects.model_custom_embeddings import E5Embeddings

QDRANT_PATH = "app/qdrant_db"
COLLECTION_NAME = "qa_documents"

# 1. Инициализация эмбеддера
embedding_model = E5Embeddings(
    model_name="intfloat/multilingual-e5-large", device="cpu"
)

# 2. Инициализация клиента
client = QdrantClient(path=QDRANT_PATH)

# 3. Векторизуем запрос
query = "Расскажи, как поступить в AI Talented Hub?"
query_vector = embedding_model.embed_query(query)

# 4. Поиск
hits = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5,  # количество результатов
    with_payload=True,  # вернуть содержимое
)


# 5. Вывод результатов
for i, hit in enumerate(hits, 1):
    print(f"\n--- Результат {i} ---")
    print("ID:", hit.id)
    print("Score:", hit.score)
    print("Text:", hit.payload.get("text"))
    print("Metadata:", hit.payload)
