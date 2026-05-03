import pytest
import tempfile
import json
import os
from unittest.mock import Mock
from app.modules.pipeline.evaluation_pipeline import EvaluationPipeline
from app.core.schemas.evaluation import EvaluationResult, MetricResult

def test_evaluation_pipeline():
    test_data = [
        {
            "question": "What is AI?",
            "contexts": ["AI is artificial intelligence."],
            "answer": "Artificial Intelligence"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        json.dump(test_data, tmp)
        tmp_path = tmp.name
        
    try:
        mock_evaluator = Mock()
        expected_result = EvaluationResult(
            metrics=[MetricResult(name="Hit@5", value=1.0)],
            summary={"Hit@5": 1.0}
        )
        mock_evaluator.evaluate.return_value = expected_result
        
        pipeline = EvaluationPipeline(evaluators={"search": mock_evaluator})
        result = pipeline.run_evaluation(evaluator_name="search", data_path=tmp_path, use_rerank=True, limit_search=10, limit_rerank=5)
        
        assert result == expected_result
        mock_evaluator.evaluate.assert_called_once()
        kwargs = mock_evaluator.evaluate.call_args[1]
        assert len(kwargs["dataset"]) == 1
        assert kwargs["use_rerank"] is True
        assert kwargs["limit_search"] == 10
        assert kwargs["limit_rerank"] == 5
        
    finally:
        os.remove(tmp_path)
