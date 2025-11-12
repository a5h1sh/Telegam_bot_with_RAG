import logging
from typing import List, Optional
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, KEYWORD_COUNT
from config import DOWNLOADS_FOLDER
from file_extractors import FileExtractorFactory
from document_chunker import DocumentChunker
from embeddings_manager import EmbeddingsManager
from vector_store import VectorStore
import os

logger = logging.getLogger(__name__)

class FileProcessor:
    """Process files: extract → chunk → embed → store."""
    def __init__(self):
        self.chunker = DocumentChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        self.embedder = EmbeddingsManager()
        self.vector_store = VectorStore()

    def _extract_keywords(self, text: str, top_n: int = KEYWORD_COUNT) -> List[str]:
        """Very small keyword extractor (frequency-based, stopword removal)."""
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
        words = [w for w in text.split() if len(w) > 2]
        stop = {
            "the","and","for","that","with","this","from","have","were","which","when","your",
            "you","are","not","but","was","what","can","will","how","all","any","our"
        }
        freqs = {}
        for w in words:
            if w in stop or w.isdigit():
                continue
            freqs[w] = freqs.get(w, 0) + 1
        sorted_k = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_k[:top_n]]

    def process_and_store(self, file_path, file_name, file_type, user_id=None, llm_manager=None):
        """Full pipeline: extract → chunk → embed → store."""
        try:
            factory = FileExtractorFactory(llm_manager=llm_manager)
            content = factory.extract(file_path, file_type)

            # extract keywords from the full document
            keywords = self._extract_keywords(content)

            # Chunk
            chunks = self.chunker.chunk(content, file_name)
            if not chunks:
                logger.warning(f"No chunks generated from {file_name}")
                return False

            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed(texts)

            # Store: pass user_id and keywords as metadata (CSV)
            keywords_str = ",".join(keywords)
            self.vector_store.add_documents(chunks, embeddings, user_id=str(user_id) if user_id else None, keywords=keywords_str)
            # update last uploaded pointer for user
            if user_id:
                self.vector_store.set_last_uploaded(str(user_id), file_name)

            logger.info(f"Processed and stored {file_name}")
            return True
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return False

def init_directories():
    """Create required directories."""
    DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloads folder: {DOWNLOADS_FOLDER}")