import logging
import time

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.index import ChatMessage
from providers.rag_service import _EXECUTOR, initialize_components, query_rag

logger = logging.getLogger(__name__)

app = FastAPI()

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Инициализация при старте приложения"""
    initialize_components()


@app.on_event("shutdown")
async def shutdown_event():
    """Корректное завершение ресурсов при остановке приложения"""
    _EXECUTOR.shutdown(wait=True)


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Service is running"}


class ChatRequest(BaseModel):
    question: str


# @app.post("/chat/{chat_id}")
# async def chat_endpoint(chat_id: str, request: ChatRequest):
#     try:
#         response = query_rag(ChatMessage(question=request.question), chat_id)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/chat/{chat_id}",
    response_model=dict,
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"},
    },
)
async def chat_endpoint(chat_id: str, request: ChatRequest):
    """
    Обработчик RAG-запросов через прямое HTTP API

    Параметры:
    - chat_id: идентификатор сессии/диалога
    - question: текст вопроса пользователя

    Возвращает:
    - JSON с полями:
      - response: текст ответа
      - sources: список источников (если есть)
      - timings: время выполнения этапов (в мс)
    """
    start_time = time.time()
    timings = {}

    try:
        # Валидация входных данных
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty",
            )

        # Выполнение RAG-запроса
        rag_start = time.time()
        response = query_rag(ChatMessage(question=request.question), chat_id)
        timings["rag_processing"] = round((time.time() - rag_start) * 1000, 2)

        # Парсинг ответа (если нужно выделить источники)
        sources = []
        if "Источники:" in response:
            response, *source_lines = response.split("\n\nИсточники:")
            sources = (
                [s.strip() for s in source_lines[0].split("\n- ") if s.strip()]
                if source_lines
                else []
            )

        # Формирование ответа
        timings["total"] = round((time.time() - start_time) * 1000, 2)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "response": response.strip(),
                "sources": sources,
                "timings": timings,
                "status": "success",
            },
        )

    except HTTPException:
        raise  # Пробрасываем уже обработанные ошибки

    except Exception as e:
        logger.error(f"API Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": str(e),
                "timings": timings,
            },
        )


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        workers=1,  # Для моделей лучше 1 worker
    )
