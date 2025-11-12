import logging
import chromadb
from config import CHROMA_DIR

logger = logging.getLogger(__name__)

def init_vector_db():
    """Initialize ChromaDB (persistent or ephemeral fallback)."""
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_or_create_collection(name="telegram_files")
        logger.info("ChromaDB initialized (persistent)")
        return collection
    except Exception as e:
        logger.warning(f"ChromaDB persistent failed: {e}")
        try:
            chroma_client = chromadb.EphemeralClient()
            collection = chroma_client.get_or_create_collection(name="telegram_files")
            logger.info("ChromaDB initialized (ephemeral fallback)")
            return collection
        except Exception as e2:
            logger.error(f"ChromaDB failed: {e2}")
            return None