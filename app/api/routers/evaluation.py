from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.api.deps import get_evaluation_pipeline
from app.modules.pipeline.evaluation_pipeline import EvaluationPipeline
from app.core.schemas.evaluation import EvaluationResult

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

class EvaluationRequest(BaseModel):
    evaluator_name: str = "search"
    data_path: str = "data/evaluation_sample.json"
    use_rerank: bool = True
    limit_search: int = 20
    limit_rerank: int = 5

@router.post("/run", response_model=EvaluationResult)
async def run_evaluation(
    request: EvaluationRequest,
    pipeline: EvaluationPipeline = Depends(get_evaluation_pipeline)
):
    """
    Run the quantitative evaluation pipeline.
    """
    try:
        result = pipeline.run_evaluation(
            evaluator_name=request.evaluator_name,
            data_path=request.data_path,
            use_rerank=request.use_rerank,
            limit_search=request.limit_search,
            limit_rerank=request.limit_rerank
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
