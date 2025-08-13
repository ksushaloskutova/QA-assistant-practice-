FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

COPY models/my_model /opt/models/my_model
ENV MODEL_DIR=/opt/models/my_model \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# 4) Прогреваем ТОЛЬКО токенизатор (без загрузки весов!)
RUN python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/opt/models/my_model', trust_remote_code=True, local_files_only=True)
print('Tokenizer OK:', tok.__class__.__name__)
PY

COPY models/e5_base /opt/embeddings/e5_base
ENV EMBEDDINGS_MODEL_NAME=/opt/embeddings/e5_base \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1