from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvaluationRecord(BaseModel):
    question: str
    contexts: List[str]
    answer: str

class MetricResult(BaseModel):
    name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

class EvaluationResult(BaseModel):
    metrics: List[MetricResult]
    summary: Dict[str, float]
