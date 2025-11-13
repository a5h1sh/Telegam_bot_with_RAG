import os
import logging
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Telegram
API_TOKEN = os.getenv("API_TOKEN", "").strip()

# Env / tokens
HF_TOKEN_INLINE = os.getenv("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Directories
CHROMA_DIR = BASE_DIR / "chroma_data"
DOWNLOADS_FOLDER = BASE_DIR / "telegram_downloads"
DB_DIR = BASE_DIR / "db"

# Embeddings / chunking / retrieval defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000")) # Increased to 1000
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100")) # Increased overlap
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))

# LLM provider defaults
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama") # Changed to ollama
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b") # Using llama3 8b

# Keyword tuning
KEYWORD_COUNT = int(os.getenv("KEYWORD_COUNT", "6"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))

# Image captioning defaults
IMAGE_MODELS = ["blip2", "clip_interrogator", "llava"]
DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "blip2")
ENABLE_IMAGE_CAPTIONING = os.getenv("ENABLE_IMAGE_CAPTIONING", "true").lower() == "true"

# Pipeline enable flags (control which pipelines are active)
# Set via .env: ENABLE_TEXT_PIPELINE=true/false, ENABLE_IMAGE_PIPELINE=true/false
ENABLE_TEXT_PIPELINE = os.getenv("ENABLE_TEXT_PIPELINE", "true").lower() == "true"
ENABLE_IMAGE_PIPELINE = os.getenv("ENABLE_IMAGE_PIPELINE", "true").lower() == "true"

# Logging
LOG_LEVEL = logging.INFO

print(f"✅ Config loaded successfully")
print(f"   Embedding: {EMBEDDING_MODEL}")
print(f"   LLM: {LLM_PROVIDER} ({HF_MODEL if LLM_PROVIDER == 'huggingface' else ''})")

# Print which pipelines are active
if ENABLE_TEXT_PIPELINE and ENABLE_IMAGE_PIPELINE:
    print("   Pipelines enabled: text + image")
elif ENABLE_TEXT_PIPELINE:
    print("   Pipelines enabled: text only")
elif ENABLE_IMAGE_PIPELINE:
    print("   Pipelines enabled: image only")
else:
    print("   Pipelines enabled: none")

#print(f"   Image captioning enabled: {ENABLE_IMAGE_CAPTIONING}")