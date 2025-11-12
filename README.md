# RAG Bot — README

## Overview
Telegram Retrieval-Augmented-Generation bot with two pipelines:
- Text pipeline: upload documents → chunk → embed → index → search (/ask, /summarize).
- Image pipeline: photo captioning (BLIP2 / Clip-Interrogator / LLAVA) → short caption + 3 tags.

Config via `.env` or `config.py`.

## Quick start (local)
1. Create virtualenv and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Note: For PyTorch, pick the correct wheel for your CUDA or CPU:
   https://pytorch.org/get-started/locally/

2. Configure `.env` (example keys):
   - API_TOKEN (Telegram bot token)
   - EMBEDDING_MODEL (default: all-MiniLM-L6-v2)
   - CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SIMILARITY_THRESHOLD
   - LLM_PROVIDER, HF_MODEL, HF_TOKEN, OPENAI_API_KEY, ...
   - ENABLE_TEXT_PIPELINE=true/false
   - ENABLE_IMAGE_PIPELINE=true/false
   - ENABLE_IMAGE_CAPTIONING=true/false

3. Run locally:
   ```
   cd d:\DEMO
   python main.py
   ```

## Docker (docker-compose)
1. Place `.env` in project root.
2. Start:
   ```
   docker compose up --build
   ```

## Files / Commands (Telegram)
- /help — usage & active pipelines
- /ask <question> — RAG query (mention filename to restrict)
- /summarize [image|chat] — summarize last image or recent chat
- /last_doc [image|doc] — show last processed item
- /docs — list stored documents
- /clear <docname> — delete document

## Models (configurable)
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (default)
- LLM providers: Hugging Face (`google/flan-t5-base` default), Ollama (local), OpenAI
- Image captioning: BLIP2, Clip-Interrogator, LLAVA (configurable order)

## Reindexing
- If you change embedding model/chunk params, delete DB and reindex:
  ```
  del db\vectors.db
  del db\embeddings_cache.db
  python reindex.py --clear-db
  ```
- Or run `python reindex.py` to (re)process files in `telegram_downloads/`.

## System diagram (high-level)
```
[ Telegram ] <-> [ Bot Handlers ] -> [ FileExtractor -> Chunker ] -> [ EmbeddingsManager ] -> [ VectorStore (SQLite) ]
                                           ^                                                     |
                                           |                                                     v
                                     [ ImageCaptioner ]                                    [ Retriever & LLM ]
```

## Troubleshooting
- "Token must contain a colon": ensure API_TOKEN loads (check .env and config.py load_dotenv).
- No results: lower SIMILARITY_THRESHOLD, reindex after model change.
- ClipInterrogator warnings: run on CPU or apply monkeypatch included in image_captioner.
