import logging
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from config import DB_DIR, SIMILARITY_THRESHOLD, KEYWORD_WEIGHT

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, db_path: str = None):
        if db_path is None:
            DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = DB_DIR / "vectors.db"
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT NOT NULL,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                embedding BLOB,
                keywords TEXT,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                upload_order INTEGER DEFAULT 0,
                content_hash TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_last (
                user_id TEXT PRIMARY KEY,
                last_doc TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # New table to store per-user interactions (history). Keep full timestamps.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        # run lightweight migrations for missing columns if needed (kept safe)
        cursor.execute("PRAGMA table_info(documents)")
        cols = [r[1] for r in cursor.fetchall()]
        if "keywords" not in cols:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN keywords TEXT")
            except Exception:
                pass
        if "user_id" not in cols:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN user_id TEXT")
            except Exception:
                pass
        conn.commit()
        conn.close()
        logger.info(f"Vector store initialized: {self.db_path}")

    def _get_content_hash(self, content: str) -> str:
        import hashlib
        return hashlib.md5(content.strip().lower().encode()).hexdigest()

    def add_documents(self, chunks: List[dict], embeddings: List[List[float]], user_id: Optional[str] = None, keywords: Optional[str] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(upload_order) FROM documents")
        row = cursor.fetchone()
        max_order = row[0] or 0
        upload_order = max_order + 1

        added = 0
        skipped = 0
        for chunk, emb in zip(chunks, embeddings):
            embedding_blob = json.dumps(np.asarray(emb).tolist())
            content_hash = self._get_content_hash(chunk["text"])
            try:
                cursor.execute("""
                    INSERT INTO documents (doc_name, chunk_index, content, embedding, keywords, user_id, upload_order, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk["doc_name"],
                    chunk.get("chunk_index", 0),
                    chunk["text"],
                    embedding_blob,
                    keywords,
                    user_id,
                    upload_order,
                    content_hash
                ))
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1
            except Exception as e:
                logger.debug(f"Insert fallback: {e}")
                try:
                    cursor.execute("""
                        INSERT INTO documents (doc_name, chunk_index, content, embedding, user_id, upload_order)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        chunk["doc_name"],
                        chunk.get("chunk_index", 0),
                        chunk["text"],
                        embedding_blob,
                        user_id,
                        upload_order
                    ))
                    added += 1
                except Exception as ex:
                    logger.error(f"Failed to insert chunk: {ex}")
        conn.commit()
        conn.close()
        logger.info(f"Stored {added} chunks, skipped {skipped} duplicates (order={upload_order})")

    def set_last_uploaded(self, user_id: str, doc_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_last (user_id, last_doc, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET last_doc=excluded.last_doc, updated_at=CURRENT_TIMESTAMP
        """, (user_id, doc_name))
        conn.commit()
        conn.close()

    def get_last_uploaded(self, user_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT last_doc FROM user_last WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def list_doc_names(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT doc_name FROM documents")
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_chunks_for_doc(self, doc_name: str) -> List[dict]:
        """Return all chunks for a document ordered by chunk_index."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_index, content, embedding, keywords, upload_order
            FROM documents
            WHERE LOWER(doc_name) = ?
            ORDER BY chunk_index ASC
        """, (doc_name.lower(),))
        rows = cur.fetchall()
        conn.close()
        out = []
        for idx, content, embedding_blob, keywords, upload_order in rows:
            out.append({
                "chunk_index": idx,
                "content": content,
                "embedding": json.loads(embedding_blob) if embedding_blob else None,
                "keywords": keywords,
                "upload_order": upload_order or 0
            })
        return out

    def search(self, query_embedding: List[float], top_k: int = 10,
               similarity_threshold: float = SIMILARITY_THRESHOLD,
               recency_weight: float = 0.2,
               doc_name_filter: Optional[str] = None,
               query_keywords: Optional[List[str]] = None,
               deduplicate_docs: bool = False
               ) -> List[dict]:
        """
        Retrieve top-k chunks (by default) using similarity + keyword overlap + recency.
        If deduplicate_docs=True returns one best chunk per document (useful for doc-level listing).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        sql = "SELECT doc_name, chunk_index, content, embedding, keywords, upload_order FROM documents"
        params = []
        if doc_name_filter:
            sql += " WHERE LOWER(doc_name) = ?"
            params.append(doc_name_filter.lower())
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        q = np.asarray(query_embedding, dtype=float)
        qnorm = np.linalg.norm(q)
        if qnorm == 0:
            return []
        q = q / qnorm

        scored = []
        max_upload = max((r[5] or 0 for r in rows), default=1)

        for doc_name, chunk_index, content, embedding_blob, keywords_blob, upload_order in rows:
            try:
                emb = np.asarray(json.loads(embedding_blob), dtype=float)
            except Exception:
                continue
            # cosine similarity (embeddings are expected normalized; safe-guard)
            denom = (np.linalg.norm(emb) * 1.0 + 1e-12)
            sim = float(np.dot(q, emb) / denom)

            if sim < similarity_threshold:
                continue

            # keyword overlap score
            kw_score = 0.0
            if query_keywords and keywords_blob:
                stored_kw = {k.strip().lower() for k in str(keywords_blob).split(',') if k.strip()}
                query_kw = {k.strip().lower() for k in query_keywords if k.strip()}
                if stored_kw and query_kw:
                    kw_score = len(stored_kw & query_kw) / max(len(query_kw), 1)

            recency = (upload_order or 0) / max(max_upload, 1)
            combined = sim * (1 - KEYWORD_WEIGHT - recency_weight) + kw_score * KEYWORD_WEIGHT + recency * recency_weight

            scored.append({
                "combined": combined,
                "similarity": sim,
                "keyword_score": kw_score,
                "doc_name": doc_name,
                "chunk_index": chunk_index,
                "content": content,
                "upload_order": upload_order or 0
            })

        if not scored:
            return []

        # sort by combined score desc
        scored.sort(key=lambda x: x["combined"], reverse=True)

        if deduplicate_docs:
            out = []
            seen = set()
            for item in scored:
                if item["doc_name"] in seen:
                    continue
                out.append({
                    "doc_name": item["doc_name"],
                    "content": item["content"],
                    "similarity_score": item["similarity"],
                    "keyword_score": item["keyword_score"],
                    "recency_order": item["upload_order"],
                    "chunk_index": item["chunk_index"]
                })
                seen.add(item["doc_name"])
                if len(out) >= top_k:
                    break
            logger.info(f"Search (dedup) returned {len(out)} docs")
            return out

        # return top-k chunks (may include multiple chunks from same doc)
        out = []
        for item in scored[:top_k]:
            out.append({
                "doc_name": item["doc_name"],
                "chunk_index": item["chunk_index"],
                "content": item["content"],
                "similarity_score": item["similarity"],
                "keyword_score": item["keyword_score"],
                "recency_order": item["upload_order"]
            })
        logger.info(f"Search returned {len(out)} chunks")
        return out

    def get_document_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        total_chunks = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT doc_name FROM documents")
        unique_docs = len(cursor.fetchall())

        cursor.execute("""
            SELECT doc_name, COUNT(*) as chunks
            FROM documents
            GROUP BY doc_name
            ORDER BY MAX(upload_order) DESC
        """)
        doc_stats = cursor.fetchall()
        conn.close()

        return {
            "total_chunks": total_chunks,
            "unique_documents": unique_docs,
            "documents": [{"name": d[0], "chunks": d[1]} for d in doc_stats]
        }

    def delete_document(self, doc_name: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE doc_name = ?", (doc_name,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Deleted: {doc_name} ({deleted} chunks)")
        return deleted > 0

    def add_user_interaction(self, user_id: str, message: str, source: Optional[str] = None, keep: int = 3):
        """Record a user interaction and trim history to the latest `keep` entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_history (user_id, message, source) VALUES (?, ?, ?)",
            (user_id, message, source)
        )
        conn.commit()
        # delete older entries keeping only the most recent `keep`
        try:
            cursor.execute("""
                DELETE FROM user_history
                WHERE user_id = ?
                  AND id NOT IN (
                      SELECT id FROM user_history
                      WHERE user_id = ?
                      ORDER BY datetime(created_at) DESC
                      LIMIT ?
                  )
            """, (user_id, user_id, keep))
            conn.commit()
        except Exception as e:
            logger.debug(f"Failed to trim user history: {e}")
        conn.close()

    def get_user_history(self, user_id: str, limit: int = 3) -> List[dict]:
        """Return the last `limit` interactions for a user, newest first."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message, source, created_at
            FROM user_history
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [{"message": r[0], "source": r[1], "created_at": r[2]} for r in rows]