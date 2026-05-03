from abc import ABC, abstractmethod
from typing import List, Any, Optional
from app.core.schemas.common import ProcessedChunk

class IEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert single text to a vector embedding."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Convert a batch of texts to a list of vector embeddings with specified batch size."""
        pass


class IVectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: List[ProcessedChunk], embeddings: List[List[float]]) -> None:
        """Store chunks and their corresponding embeddings."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], limit: int = 5) -> List[Any]:
        """Search for similar chunks based on a query vector."""
        pass


