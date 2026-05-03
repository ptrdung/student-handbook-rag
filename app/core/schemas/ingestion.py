from pydantic import BaseModel, Field
from app.core.schemas.common import ChunkMetadata, ProcessedChunk, ContentType

class ChunkingConfiguration(BaseModel):
    chunk_size: int = Field(default=1000, description="Target size of each chunk in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks")
