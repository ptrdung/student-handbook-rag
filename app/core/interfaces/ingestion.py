from abc import ABC, abstractmethod
from typing import List
from app.core.schemas.common import ProcessedChunk


class IVisionAnalyzer(ABC):
    @abstractmethod
    def summarize_image(self, image_bytes: bytes) -> str:
        """Process image bytes and return markdown analysis."""
        pass

class ITableParser(ABC):
    @abstractmethod
    def extract_table(self, file_path: str) -> str:
        """Extract table from a PDF file and return markdown."""
        pass

class IChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> List[ProcessedChunk]:
        """Split text into chunks with metadata."""
        pass

class IExtractor(ABC):
    @abstractmethod
    def process_pdf(self, file_path: str, output_dir: str = None) -> str:
        """Extract content from PDF and return enriched markdown."""
        pass
