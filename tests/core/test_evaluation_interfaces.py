import pytest
from app.core.interfaces.evaluation import BaseMetric, BaseEvaluator, MetricTracker
from app.core.schemas.evaluation import EvaluationRecord, MetricResult, EvaluationResult

def test_cannot_instantiate_abstract_classes():
    with pytest.raises(TypeError):
        BaseMetric()
        
    with pytest.raises(TypeError):
        BaseEvaluator([])
        
    with pytest.raises(TypeError):
        MetricTracker()

def test_evaluation_schemas():
    record = EvaluationRecord(
        question="What is this?",
        contexts=["Context 1", "Context 2"],
        answer="This is a test."
    )
    assert record.question == "What is this?"
    assert len(record.contexts) == 2
    
    metric_result = MetricResult(
        name="test_metric",
        value=0.95
    )
    assert metric_result.name == "test_metric"
    assert metric_result.value == 0.95
    
    eval_result = EvaluationResult(
        metrics=[metric_result],
        summary={"test_metric": 0.95}
    )
    assert len(eval_result.metrics) == 1
    assert eval_result.summary["test_metric"] == 0.95

def test_concrete_implementation():
    class DummyMetric(BaseMetric):
        @property
        def name(self) -> str:
            return "dummy"
            
        def calculate(self, predictions, ground_truths, **kwargs) -> float:
            return 1.0

    metric = DummyMetric()
    assert metric.name == "dummy"
    assert metric.calculate([], []) == 1.0
