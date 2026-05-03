import pytest
from unittest.mock import MagicMock
from app.modules.retrieval.semantic import SemanticSearchEngine
from app.retrieval.schemas.search import SearchRequest, SearchResult
from app.schemas.common import ChunkMetadata, ContentType

@pytest.fixture
def mock_embedder():
    return MagicMock()

@pytest.fixture
def mock_vector_store():
    return MagicMock()

@pytest.fixture
def search_engine(mock_embedder, mock_vector_store):
    return SemanticSearchEngine(mock_embedder, mock_vector_store)

def test_search_success(search_engine, mock_embedder, mock_vector_store):
    # Setup
    mock_embedder.embed_text.return_value = [0.1, 0.2]
    
    # Mock Qdrant ScoredPoint (simulating what results.points returns)
    mock_point = MagicMock()
    mock_point.id = "1"
    mock_point.score = 0.95
    mock_point.payload = {
        "content": "Đây là nội dung thử nghiệm",
        "file_name": "test.pdf",
        "chunk_id": "1",
        "content_type": "text",
        "is_table": False
    }
    mock_vector_store.search.return_value = [mock_point]
    
    # Execute
    request = SearchRequest(query="thử nghiệm", top_k=1)
    results = search_engine.search(request)
    
    # Assert
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].content == "Đây là nội dung thử nghiệm"
    assert results[0].score == 0.95
    assert results[0].metadata.file_name == "test.pdf"
    
    mock_embedder.embed_text.assert_called_once_with("thử nghiệm")
    mock_vector_store.search.assert_called_once()

def test_search_empty_results(search_engine, mock_embedder, mock_vector_store):
    # Setup
    mock_embedder.embed_text.return_value = [0.1, 0.2]
    mock_vector_store.search.return_value = []
    
    # Execute
    request = SearchRequest(query="not found", top_k=5)
    results = search_engine.search(request)
    
    # Assert
    assert len(results) == 0

def test_search_error_handling(search_engine, mock_embedder):
    # Setup
    mock_embedder.embed_text.side_effect = Exception("Embedding failed")
    
    # Execute
    request = SearchRequest(query="error case")
    results = search_engine.search(request)
    
    # Assert
    assert results == []
