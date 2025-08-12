FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY models/my_model /models/my_model
ENV MODEL_DIR=/app/models/my_model \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# КОД
COPY api.py /app/
COPY providers/ /app/providers/

EXPOSE 8080
# для прод-режима без reload:
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8080","--workers","1"]