import os

import torch
from transformers import pipeline

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Пайплайн генерации
generation_pipeline = pipeline(
    "text-generation",
    model="RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4",
    device=device,
    torch_dtype=torch.float16,
)
