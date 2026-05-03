import pytest
import json
import tempfile
import os
from app.modules.evaluation.data_loader import JSONDataLoader
from app.modules.evaluation.evaluators.search_evaluator import SearchEvaluator
from app.modules.evaluation.metrics.hit_at_k import HitAtKMetric
from app.modules.evaluation.metrics.mrr import MRRMetric
from app.modules.evaluation.metrics.ndcg import NDCGMetric

from app.core.interfaces.retrieval import ISearchEngine
from app.core.schemas.retrieval import SearchRequest, SearchResult
from app.core.schemas.common import ChunkMetadata

class MockSearchEngine(ISearchEngine):
    """A mock search engine that returns predetermined results for the evaluation demo."""
    def search(self, request: SearchRequest) -> list[SearchResult]:
        if "đối tượng" in request.query:
            return [
                SearchResult(content="Quy chế này áp dụng cho sinh viên đại học hệ chính quy", metadata=ChunkMetadata(id="1", chunk_id="c1", doc_id="d1", file_name="f1", content_type="text", chunk_index=0), score=0.9),
                SearchResult(content="Sinh viên hệ cao đẳng không áp dụng", metadata=ChunkMetadata(id="2", chunk_id="c2", doc_id="d1", file_name="f1", content_type="text", chunk_index=1), score=0.5)
            ]
        elif "khóa luận" in request.query:
            # We put the relevant context at rank 2 (index 1) to test MRR and nDCG
            return [
                SearchResult(content="Điều kiện để được bảo vệ là gì?", metadata=ChunkMetadata(id="3", chunk_id="c3", doc_id="d2", file_name="f2", content_type="text", chunk_index=0), score=0.8),
                SearchResult(content="Sinh viên phải hoàn thành ít nhất 80% chương trình học", metadata=ChunkMetadata(id="4", chunk_id="c4", doc_id="d2", file_name="f2", content_type="text", chunk_index=1), score=0.7)
            ]
        return []

def test_search_evaluator_e2e():
    test_data = [
        {
            "question": "Quy chế đào tạo đại học áp dụng cho đối tượng nào?",
            "contexts": ["Quy chế này áp dụng cho sinh viên đại học hệ chính quy"],
            "answer": "Sinh viên đại học hệ chính quy"
        },
        {
            "question": "Điều kiện để làm khóa luận tốt nghiệp là gì?",
            "contexts": ["Sinh viên phải hoàn thành ít nhất 80% chương trình học"],
            "answer": "Sinh viên phải hoàn thành ít nhất 80% chương trình học"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        json.dump(test_data, tmp)
        tmp_path = tmp.name
        
    try:
        loader = JSONDataLoader(tmp_path)
        dataset = loader.load()
        
        metrics = [HitAtKMetric(k=5), MRRMetric(k=5), NDCGMetric(k=5)]
        search_engine = MockSearchEngine()
        
        evaluator = SearchEvaluator(search_engine, metrics)
        
        # Test Base Retrieval (no rerank)
        ret_result = evaluator.evaluate(dataset, use_rerank=False, limit_search=5)
        assert ret_result.summary["Hit@5"] == 1.0
        assert ret_result.summary["MRR@5"] == 0.75
        
        # Test ReRank (with rerank)
        rerank_result = evaluator.evaluate(dataset, use_rerank=True, limit_search=10, limit_rerank=5)
        assert rerank_result.summary["Hit@5"] == 1.0
        assert rerank_result.summary["MRR@5"] == 0.75
        
    finally:
        os.remove(tmp_path)
