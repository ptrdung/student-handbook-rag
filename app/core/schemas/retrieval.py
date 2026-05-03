from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.core.schemas.common import ChunkMetadata

class SearchRequest(BaseModel):
    query: str = Field(..., description="The user query string")
    limit_search: int = Field(default=20, description="Number of results to return from search", alias="top_k")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    use_rerank: bool = Field(default=True, description="Whether to apply reranking after retrieval")
    limit_rerank: int = Field(default=5, description="Number of final results to return after rerank")

    model_config = {
        "populate_by_name": True
    }

class SearchResult(BaseModel):
    content: str
    metadata: ChunkMetadata
    score: float = Field(..., description="Similarity score")
