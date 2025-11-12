import os
import logging
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Directories
CHROMA_DIR = BASE_DIR / "chroma_data"
DOWNLOADS_FOLDER = BASE_DIR / "telegram_downloads"

# Telegram API
API_TOKEN = "8214055923:AAEOSM-npoCJewJHQ-17wmlh-q9IbTsKEp4"

# Hugging Face Model
HF_MODEL = "google/flan-t5-base"
HF_TOKEN_INLINE = os.getenv("HF_TOKEN", "")  # Read from env or use empty string

# Llama Model (optional)
LLAMA_MODEL_PATH = str(BASE_DIR / "models" / "ggml-model.bin")

# Logging
LOG_LEVEL = logging.INFO