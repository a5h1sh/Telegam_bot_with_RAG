import logging
import json
import sqlite3
import hashlib
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, DB_DIR

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manage local embeddings using SentenceTransformers with a basic cache."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

        # ensure DB_DIR exists and init cache DB
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_path = DB_DIR / "embeddings_cache.db"
        self._init_cache_db()

    def _init_cache_db(self):
        conn = sqlite3.connect(str(self._cache_path))
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                key TEXT PRIMARY KEY,
                text TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _make_key(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    def _get_cached(self, text: str):
        key = self._make_key(text)
        conn = sqlite3.connect(str(self._cache_path))
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM query_cache WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

    def _set_cache(self, text: str, embedding: List[float]):
        key = self._make_key(text)
        conn = sqlite3.connect(str(self._cache_path))
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT OR REPLACE INTO query_cache (key, text, embedding) VALUES (?, ?, ?)",
                (key, text, json.dumps(embedding))
            )
            conn.commit()
        except Exception as e:
            logger.debug(f"Failed to write embedding cache: {e}")
        finally:
            conn.close()

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for texts, use cache for exact matches."""
        results = []
        to_compute = []
        indices_to_compute = []

        # check cache per text
        for i, t in enumerate(texts):
            cached = self._get_cached(t)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                to_compute.append(t)
                indices_to_compute.append(i)

        if to_compute:
            try:
                embs = self.model.encode(to_compute, convert_to_numpy=True, show_progress_bar=False)
                if embs.ndim == 1:
                    embs = embs.reshape(1, -1)
                for idx, emb in enumerate(embs):
                    normed = self._normalize(emb).tolist()
                    results[indices_to_compute[idx]] = normed
                    # store each computed embedding in cache using original text
                    self._set_cache(to_compute[idx], normed)
                logger.debug(f"Embedded {len(to_compute)} texts, cached results.")
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise

        # final sanity: ensure all slots filled
        final = [r if r is not None else [0.0] for r in results]
        return final

    def embed_single(self, text: str) -> List[float]:
        cached = self._get_cached(text)
        if cached is not None:
            return cached
        emb = self.embed([text])[0]
        return emb