import json
import sqlite3
import sys
import numpy as np
from pathlib import Path
from embeddings_manager import EmbeddingsManager
from config import DB_DIR

DB_PATH = Path(DB_DIR) / "vectors.db"

def load_rows(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, content, embedding, upload_order FROM documents")
    rows = cur.fetchall()
    conn.close()
    return rows

def parse_embedding(blob):
    try:
        return np.asarray(json.loads(blob), dtype=float)
    except Exception:
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_search.py \"your query\"")
        return
    query = sys.argv[1]
    if not DB_PATH.exists():
        print("DB not found:", DB_PATH)
        return

    em = EmbeddingsManager()  # will log model loaded
    q = np.asarray(em.embed_single(query), dtype=float)
    qnorm = np.linalg.norm(q)
    print(f"Query dim={q.shape}, norm={qnorm:.6f}")

    rows = load_rows(DB_PATH)
    print(f"Loaded {len(rows)} DB rows")

    scores = []
    for row in rows:
        row_id, doc_name, content, blob, upload_order = row
        emb = parse_embedding(blob)
        if emb is None:
            continue
        # ensure numeric
        emb = emb.astype(float)
        # compute cosine (assumes embeddings may not be normalized)
        denom = (np.linalg.norm(emb) * (qnorm if qnorm>0 else 1.0)) + 1e-12
        sim = float(np.dot(q, emb) / denom)
        scores.append((sim, doc_name, content[:200], upload_order))

    if not scores:
        print("No valid embeddings found in DB.")
        return

    scores.sort(reverse=True, key=lambda x: x[0])
    top_n = min(20, len(scores))
    print(f"Top {top_n} similarities:")
    for i, (sim, doc, snippet, order) in enumerate(scores[:top_n], 1):
        print(f"{i:02d}. sim={sim:.4f}  upload_order={order}  doc={doc}\n   snippet={snippet}\n")

if __name__ == "__main__":
    main()