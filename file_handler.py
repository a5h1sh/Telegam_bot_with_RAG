import logging
from config import DOWNLOADS_FOLDER
from file_extractors import FileExtractorFactory

logger = logging.getLogger(__name__)

def store_file_to_db(collection, file_info, file_name, file_path, user_id, file_type="document", llm_manager=None):
    """Store file content in vector DB."""
    if not collection:
        return False
    try:
        # Use factory to extract content
        factory = FileExtractorFactory(llm_manager=llm_manager)
        file_content = factory.extract(file_path, file_type)
        
        collection.add(
            ids=[file_info.file_id],
            metadatas=[{"filename": file_name, "user_id": str(user_id), "file_type": file_type}],
            documents=[file_content]
        )
        logger.info(f"File stored: {file_name} (type: {file_type})")
        return True
    except Exception as e:
        logger.error(f"Failed to store file: {e}")
        return False

def init_directories():
    """Create required directories."""
    DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloads folder: {DOWNLOADS_FOLDER}")