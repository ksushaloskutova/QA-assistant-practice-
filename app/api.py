import uvicorn
from fastapi.middleware.cors import CORSMiddleware          # CORS-middleware для разрешения запросов с
# браузера
from app.models.index import ChatMessage                        # pydantic-модель входного сообщения {question: str}
from app.providers.ollama import query_rag                      # функция, выполняющая RAG-поиск и LLM-ответ

from fastapi import FastAPI                                 # основной веб-фреймворк
from fastapi.staticfiles import StaticFiles                 # хэндлер для раздачи статических файлов (JS/HTML/CSS)


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    pass


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)

@app.get("/")                                               # health-check / приветственный endpoint
async def read_root():
    return {"Hello": "world"}                               # простой JSON-ответ

# POST /chat/{chat_id} — основной endpoint чата
@app.post("/chat/{chat_id}")
async def ask(chat_id: str, message: ChatMessage):          # chat_id берётся из URL, message из JSON-тела
    # вызываем RAG-функцию и оборачиваем результат словарём (FastAPI → JSON)
    return {"response": query_rag(message, chat_id)}
