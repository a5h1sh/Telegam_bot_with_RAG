import logging
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Split documents into meaningful chunks with semantic awareness."""
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, min_chunk_words: int = 10):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.min_chunk_words = int(min_chunk_words)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.replace("\r", " ").replace("\n", " ").strip()
        text = " ".join(text.split())
        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Rudimentary sentence splitter (no external deps)."""
        # keep punctuation as sentence boundary markers
        import re
        # split on period, exclamation, question or newline followed by space
        parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
        sentences = [p.strip() for p in parts if p and p.strip()]
        return sentences

    def _chunk_by_sentences(self, text: str, target_size: int) -> List[str]:
        """Group sentences into word-count-based chunks."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            # fallback split by words
            words = text.split()
            return [" ".join(words[i:i+target_size]) for i in range(0, len(words), target_size)]

        chunks = []
        current = []
        current_words = 0

        for s in sentences:
            s_words = len(s.split())
            # too long single sentence -> break into word slices
            if s_words >= target_size:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_words = 0
                words = s.split()
                for i in range(0, len(words), target_size):
                    chunks.append(" ".join(words[i:i+target_size]))
                continue

            if current_words + s_words <= target_size:
                current.append(s)
                current_words += s_words
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [s]
                current_words = s_words

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlapping context between chunks (by words)."""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        overlap_words = max(1, int(self.overlap / 5))  # heuristic -> words
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = " ".join(chunks[i-1].split()[-overlap_words:])
            current = (prev_tail + " " + chunks[i]).strip()
            overlapped.append(current)
        return overlapped

    def chunk(self, text: str, doc_name: str = "document") -> List[dict]:
        """Split text into semantic chunks with metadata."""
        if not text or not text.strip():
            logger.warning(f"Empty text for {doc_name}")
            return []

        text = self._clean_text(text)
        total_words = len(text.split())
        target = max(1, self.chunk_size)

        # If the whole document is short, return single chunk
        if total_words <= max(self.min_chunk_words, target):
            logger.info(f"Document '{doc_name}' short ({total_words} words) — returning single chunk")
            return [{
                "text": text,
                "doc_name": doc_name,
                "chunk_index": 0,
                "word_count": total_words
            }]

        chunks_list = self._chunk_by_sentences(text, target_size=target)
        chunks_list = self._add_overlap(chunks_list)

        # Filter out very tiny chunks, but ensure at least one chunk remains
        filtered = [c for c in chunks_list if len(c.split()) >= self.min_chunk_words]
        if not filtered:
            # fallback: keep the largest chunk or entire text
            if chunks_list:
                largest = max(chunks_list, key=lambda x: len(x.split()))
                filtered = [largest]
            else:
                filtered = [text]

        chunks = []
        for idx, c in enumerate(filtered):
            chunks.append({
                "text": c,
                "doc_name": doc_name,
                "chunk_index": idx,
                "word_count": len(c.split())
            })

        logger.info(f"Split '{doc_name}' into {len(chunks)} semantic chunks")
        return chunks