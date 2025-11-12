import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FileExtractor:
    """Base class for file extraction."""
    def extract(self, file_path):
        raise NotImplementedError

class TextFileExtractor(FileExtractor):
    """Extract text from .txt files."""
    def extract(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Extracted text from TXT: {file_path.name}")
            return text if text.strip() else f"TXT file: {file_path.name} (empty)"
        except Exception as e:
            logger.warning(f"Failed to extract TXT: {e}")
            return f"TXT file: {file_path.name} (extraction failed)"

class DocxFileExtractor(FileExtractor):
    """Extract text from .docx files."""
    def extract(self, file_path):
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            logger.info(f"Extracted text from DOCX: {file_path.name}")
            return text if text.strip() else f"DOCX file: {file_path.name} (empty or no text)"
        except Exception as e:
            logger.warning(f"Failed to extract DOCX: {e}")
            return f"DOCX file: {file_path.name} (extraction failed)"

class PdfFileExtractor(FileExtractor):
    """Extract text from .pdf files."""
    def extract(self, file_path):
        try:
            import PyPDF2
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text += f"--- Page {page_num + 1} ---\n"
                    text += page.extract_text() + "\n"
            logger.info(f"Extracted text from PDF: {file_path.name}")
            return text if text.strip() else f"PDF file: {file_path.name} (empty or no text)"
        except Exception as e:
            logger.warning(f"Failed to extract PDF: {e}")
            return f"PDF file: {file_path.name} (extraction failed)"

class ImageExtractor(FileExtractor):
    """Extract caption from image files using vision model."""
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager

    def extract(self, file_path):
        try:
            if self.llm_manager and hasattr(self.llm_manager, 'caption_image'):
                caption = self.llm_manager.caption_image(file_path)
                logger.info(f"Generated caption for image: {file_path.name}")
                return f"Image caption: {caption}"
            else:
                logger.warning("Image captioning not available")
                return f"Photo file: {file_path.name} (image captioning not available)"
        except Exception as e:
            logger.warning(f"Failed to extract image caption: {e}")
            return f"Image file: {file_path.name} (caption generation failed)"

class FileExtractorFactory:
    """Factory to create appropriate file extractor based on file type."""
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager

    def get_extractor(self, file_path, file_type):
        """Get appropriate extractor for file type."""
        if file_type == "photo":
            return ImageExtractor(self.llm_manager)
        elif file_path.suffix.lower() == ".pdf":
            return PdfFileExtractor()
        elif file_path.suffix.lower() == ".docx":
            return DocxFileExtractor()
        elif file_path.suffix.lower() == ".txt":
            return TextFileExtractor()
        else:
            # Try text extraction as fallback
            return TextFileExtractor()

    def extract(self, file_path, file_type):
        """Extract content using appropriate extractor."""
        try:
            extractor = self.get_extractor(file_path, file_type)
            return extractor.extract(file_path)
        except Exception as e:
            logger.error(f"Extraction failed for {file_path}: {e}")
            return f"File: {file_path.name} (extraction error)"