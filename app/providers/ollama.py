from langchain.chains.combine_documents import (  # “stuff-RAG”: конкатенирует документы в prompt
    create_stuff_documents_chain,
)
from langchain_chroma import Chroma  # импорт векторной БД Chroma
from langchain_core.messages import (  # типы сообщений для сохранения чата
    AIMessage,
    HumanMessage,
)
from langchain_core.prompts import (  # генератор шаблонов ChatPrompt + плейс-холдер истории
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import (  # обёртки для Ollama-LLM и Ollama-Embeddings
    OllamaEmbeddings,
    OllamaLLM,
)

from ..models.index import ChatMessage  # pydantic-модель входного сообщения

CHROMA_PATH = "app/db_metadata_v5"  # папка, где сохранён Chroma-индекс

# -------------------- LLM -------------------------------------------------
MODEL_ID = (
    "electromagneticcyclone/t-lite-q:6_k"  # идентификатор локальной модели в Ollama
)
model = OllamaLLM(
    model=MODEL_ID, temperature=0.1
)  # инициализация LLM с небольшой температурой

# -------------------- Embeddings (тот же, что и при ingest) --------------
EMBED_ID = "mxbai-embed-large"  # модель эмбеддингов
embedding_function = OllamaEmbeddings(
    model=EMBED_ID
)  # обёртка над /v1/embeddings Ollama

# -------------------- Подключаем существующую базу -----------------------
db = Chroma(
    persist_directory=CHROMA_PATH,  # путь к дисковому индексу
    embedding_function=embedding_function,
)  # ⚠️ тот же эмбеддер обязателен
chat_history = {}  # словарь session_id → список [HumanMessage, AIMessage]

# -------------------- Шаблон промпта -------------------------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",  # системная инструкция
            """
                [INST]You are a sales manager with the name 'AI Assistant'. ...         # длинная “роль” бота
                Ready for an online meeting?[/INST]
                [INST]Answer the question based only on the following context:
                {context}[/INST]                                                        # сюда подставится контекст k документов
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # тут будет история диалога
        ("human", "{question}"),  # последняя реплика пользователя
    ]
)

# Собираем цепочку “контекст → LLM”
document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)


# -------------------- Функция-обёртка для приложения ----------------------
def query_rag(message: ChatMessage, session_id: str = "") -> str:
    """
    Выполнить RAG-запрос:
      message    – объект с полем .question
      session_id – ID диалога (для истории)
      return     – ответ LLM в формате str
    """

    if session_id not in chat_history:  # если сессия новая — заводим список
        chat_history[session_id] = []

    # --- формируем контекст top-3 документов и вызываем LLM ---------------
    response_text = document_chain.invoke(
        {
            "context": db.similarity_search(
                message.question, k=3
            ),  # поиск наиболее похожих 3 чанков
            "question": message.question,  # сам вопрос
            "chat_history": chat_history[session_id],  # предыдущие сообщения
        }
    )

    # сохраняем переписку для последующих запросов
    chat_history[session_id].append(HumanMessage(content=message.question))
    chat_history[session_id].append(AIMessage(content=response_text))

    return response_text  # отдаём ответ вызывающей стороне
