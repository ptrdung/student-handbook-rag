import pytest
from app.modules.evaluation.metrics.mrr import MRRMetric

def test_mrr_metric():
    metric = MRRMetric(k=3)
    assert metric.name == "MRR@3"
    
    predictions = [
        ["doc1", "doc2", "doc3", "doc4"],
        ["docA", "docB", "docC"],
        ["docX", "docY", "docZ"]
    ]
    ground_truths = [
        ["doc2"],   # Rank 2 -> RR = 1/2
        ["docA"],   # Rank 1 -> RR = 1/1
        ["docW"]    # Not found -> RR = 0
    ]
    
    score = metric.calculate(predictions, ground_truths)
    # (0.5 + 1.0 + 0) / 3 = 0.5
    assert score == 0.5

def test_mrr_metric_k():
    metric = MRRMetric(k=1)
    predictions = [
        ["doc1", "doc2"]
    ]
    ground_truths = [
        ["doc2"] # Hit at rank 2, but k=1 so RR=0
    ]
    score = metric.calculate(predictions, ground_truths)
    assert score == 0.0
