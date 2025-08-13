FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

COPY models/my_model /opt/models/my_model
COPY models/e5_base  /opt/embeddings/e5_base

ENV MODEL_DIR=/opt/models/my_model \
    EMBEDDINGS_MODEL_NAME=/opt/embeddings/e5_base \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
