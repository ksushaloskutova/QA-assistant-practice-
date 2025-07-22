### Быстрый старт (полностью офлайн)

```bash
# 1. Подготовка модели Llama с использованием Ollama.

Переходим на сайт проекта, https://ollama.com/ качаем клиент, тут все просто
Далее нам будут необходимы 2 нейросетки одна непосредственно LLM которая будет генерировать ответ пользователю, вторая нейросеть будет отвечать за создание embeddings

ollama pull nomic-embed-text-light  #адаптирована для русского языка
ollama pull mxbai-embed-large   





Ставим зависимости
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Скачиваем веса GGUF (пример: Phi-3-mini)
wget -O phi3.gguf https://huggingface.co/QuantFactory/Phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf

# 3. Запускаем OpenAI-совместимый сервер
python -m llama_cpp.server --model phi3.gguf --n_ctx 4096 --port 8000 &

# 4. Запускаем Qdrant
docker-compose up -d qdrant

# 5. Заполняем .env (LLAMA_URL, TELEGRAM_TOKEN и т.д.)

# 6. Стартуем приложение
python -m app.main
