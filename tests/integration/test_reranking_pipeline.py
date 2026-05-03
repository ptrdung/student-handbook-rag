import pytest
from unittest.mock import MagicMock
from app.modules.retrieval.search_pipeline import SearchPipeline
from app.retrieval.schemas.search import SearchRequest, SearchResult
from app.schemas.common import ChunkMetadata
from app.retrieval.schemas.reranker import RerankResult

def test_search_pipeline_reranking_integration():
    """
    Test that the search pipeline correctly calls the search engine and then
    reranks the candidates using the provided reranker.
    """
    # 1. Setup Search Engine Mock
    mock_search_engine = MagicMock()
    # Assume 3 results are returned originally
    mock_search_engine.search.return_value = [
        SearchResult(content="A", metadata=ChunkMetadata(file_name="doc1.pdf", chunk_id="1", content_type="text"), score=0.5),
        SearchResult(content="B", metadata=ChunkMetadata(file_name="doc2.pdf", chunk_id="2", content_type="text"), score=0.4),
        SearchResult(content="C", metadata=ChunkMetadata(file_name="doc3.pdf", chunk_id="3", content_type="text"), score=0.3),
    ]
    
    # 2. Setup Reranker Mock
    mock_reranker = MagicMock()
    # Reranker decides C is the best (0.9), then A (0.8), then B (0.2)
    mock_reranker.rerank.return_value = [
        RerankResult(text="C", score=0.9, metadata={"file_name": "doc3.pdf", "chunk_id": "3", "content_type": "text"}),
        RerankResult(text="A", score=0.8, metadata={"file_name": "doc1.pdf", "chunk_id": "1", "content_type": "text"}),
        RerankResult(text="B", score=0.2, metadata={"file_name": "doc2.pdf", "chunk_id": "2", "content_type": "text"}),
    ]
    
    pipeline = SearchPipeline(search_engine=mock_search_engine, reranker=mock_reranker)
    
    # Request with limit 2
    request = SearchRequest(query="test query", limit_search=3, use_rerank=True, limit_rerank=2)
    
    # 3. Action
    results = pipeline.search(request)
    
    # 4. Verify
    assert len(results) == 2  # Limited by request.limit_rerank
    assert results[0].content == "C"
    assert results[0].score == 0.9
    assert results[1].content == "A"
    assert results[1].score == 0.8
    
    # Ensure reranker was called once
    mock_reranker.rerank.assert_called_once()
    
    # Ensure metadata was correctly restored as ChunkMetadata objects
    assert isinstance(results[0].metadata, ChunkMetadata)
    assert results[0].metadata.file_name == "doc3.pdf"

def test_search_pipeline_bypass_rerank():
    """
    Test that reranking is bypassed when use_rerank is false.
    """
    # Setup
    mock_search_engine = MagicMock()
    mock_search_engine.search.return_value = [
        SearchResult(content="A", metadata=ChunkMetadata(file_name="doc1.pdf", chunk_id="1", content_type="text"), score=0.5),
    ]
    mock_reranker = MagicMock()
    
    pipeline = SearchPipeline(search_engine=mock_search_engine, reranker=mock_reranker)
    
    # Action: use_rerank=False
    request = SearchRequest(query="test query", use_rerank=False)
    results = pipeline.search(request)
    
    # Verify
    assert len(results) == 1
    assert results[0].content == "A"
    mock_reranker.rerank.assert_not_called()

def test_search_pipeline_no_reranker():
    """
    Test that the pipeline still returns results even if NO reranker is provided.
    """
    mock_search_engine = MagicMock()
    mock_search_engine.search.return_value = [
        SearchResult(content="A", metadata=ChunkMetadata(file_name="doc1.pdf", chunk_id="1", content_type="text"), score=0.5),
    ]
    
    # Pipeline initialized with reranker=None
    pipeline = SearchPipeline(search_engine=mock_search_engine, reranker=None)
    
    # Action
    request = SearchRequest(query="test query", use_rerank=True)
    results = pipeline.search(request)
    
    # Verify
    assert len(results) == 1
    assert results[0].content == "A"
