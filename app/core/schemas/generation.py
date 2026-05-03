from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.core.schemas.retrieval import SearchResult

class GenerationRequest(BaseModel):
    """
    Request object for the LLM generation service.
    """
    query: str = Field(..., description="The user query")
    session_id: str = Field(default="default_session", description="Session ID for managing conversation history")
    context: List[SearchResult] = Field(default_factory=list, description="Retrieved and reranked context documents")
    stream: bool = Field(default=False, description="Whether to stream the response")

class GenerationResponse(BaseModel):
    """
    Response object from the LLM generation service.
    """
    answer: str = Field(..., description="The generated answer in Markdown format")
    session_id: str = Field(..., description="Session ID")
    
class GenerationChunk(BaseModel):
    """
    Represents a single chunk of streamed response.
    """
    text: str = Field(..., description="The chunk of text")
    is_last: bool = Field(default=False, description="Whether this is the last chunk")
