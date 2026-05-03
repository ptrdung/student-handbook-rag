import os
from typing import List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.interfaces.base import IVectorStore
from app.core.schemas.common import ProcessedChunk

class QdrantVectorStore(IVectorStore):
    """
    Qdrant implementation of the VectorStore interface.
    Shared service used by both Ingestion and Retrieval modules.
    """
    def __init__(
        self, 
        host: str = None, 
        port: int = None, 
        collection_name: str = None,
        vector_size: int = 768
    ):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", 6333))
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "student_handbook")
        self.vector_size = vector_size
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def _ensure_collection(self):
        """Creates the collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, 
                    distance=models.Distance.COSINE
                ),
            )

    def upsert(self, chunks: List[ProcessedChunk], embeddings: List[List[float]]) -> None:
        """Store chunks and their corresponding embeddings."""
        if not chunks or not embeddings:
            return

        self._ensure_collection()
        
        points = []
        for i, chunk in enumerate(chunks):
            payload = chunk.metadata.model_dump()
            payload["content"] = chunk.content
            
            points.append(models.PointStruct(
                id=chunk.metadata.chunk_id,
                vector=embeddings[i],
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], limit: int = 5) -> List[Any]:
        """Search for similar chunks based on a query vector."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points
        
        return results
