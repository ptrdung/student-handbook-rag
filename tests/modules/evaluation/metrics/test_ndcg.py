import pytest
import math
from app.modules.evaluation.metrics.ndcg import NDCGMetric

def test_ndcg_metric():
    metric = NDCGMetric(k=3)
    assert metric.name == "nDCG@3"
    
    predictions = [
        ["doc1", "doc2", "doc3"]
    ]
    ground_truths = [
        ["doc2"] # Hit at rank 2
    ]
    
    # DCG = 1/log2(1 + 2) = 1/log2(3) = 0.6309
    # IDCG = 1/log2(0 + 2) = 1/1 = 1.0
    # nDCG = 0.6309
    
    score = metric.calculate(predictions, ground_truths)
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0
    expected_ndcg = expected_dcg / expected_idcg
    
    assert abs(score - expected_ndcg) < 1e-4

def test_ndcg_metric_multiple_gt():
    metric = NDCGMetric(k=5)
    
    predictions = [
        ["doc1", "doc2", "doc3", "doc4"]
    ]
    ground_truths = [
        ["doc2", "doc4"]
    ]
    
    # Hit at rank 2 (index 1) and rank 4 (index 3)
    # DCG = 1/log2(1+2) + 1/log2(3+2) = 1/log2(3) + 1/log2(5)
    # IDCG = 1/log2(0+2) + 1/log2(1+2) = 1/1 + 1/log2(3)
    
    score = metric.calculate(predictions, ground_truths)
    expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    expected_ndcg = expected_dcg / expected_idcg
    
    assert abs(score - expected_ndcg) < 1e-4
