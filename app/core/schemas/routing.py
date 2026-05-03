from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class QueryType(str, Enum):
    """Enumeration of possible query types for routing."""
    ACADEMIC_POLICY = "ACADEMIC_POLICY"
    SMALL_TALK = "SMALL_TALK"
    GREETING_CLOSING = "GREETING_CLOSING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

class RoutingResult(BaseModel):
    """Schema for query routing results."""
    query_type: QueryType = Field(..., description="The classified type of the query")
    reasoning: Optional[str] = Field(None, description="The reasoning behind the classification")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of the classification")
