from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    LIST = "list"

class ChunkMetadata(BaseModel):
    file_name: str
    chunk_id: str
    header_1: Optional[str] = None
    header_2: Optional[str] = None
    content_type: ContentType
    is_table: bool = False
    parent_text: Optional[str] = None

class ProcessedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata
