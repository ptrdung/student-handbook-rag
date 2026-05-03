import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.schemas.reranker import RerankCandidate, RerankResult
from app.modules.retrieval.reranker import ViRankerService

@pytest.fixture
def mock_cross_encoder():
    with patch("app.modules.retrieval.reranker.CrossEncoder") as mock:
        yield mock

def test_viranker_service_initialization(mock_cross_encoder):
    # Setup
    service = ViRankerService(model_name="namdp-ptit/ViRanker")
    
    # Verify
    assert service.model_name == "namdp-ptit/ViRanker"
    # Should be deferred loading, so CrossEncoder not called yet
    mock_cross_encoder.assert_not_called()

def test_viranker_rerank_logic(mock_cross_encoder):
    # Setup
    mock_model = MagicMock()
    # Mock predict to return scores
    mock_model.predict.return_value = [0.2, 0.9, 0.5]
    mock_cross_encoder.return_value = mock_model
    
    service = ViRankerService(model_name="namdp-ptit/ViRanker")
    
    query = "học phí ngành CNTT"
    candidates = [
        RerankCandidate(text="văn bản 1", metadata={"id": 1}),
        RerankCandidate(text="văn bản 2", metadata={"id": 2}),
        RerankCandidate(text="văn bản 3", metadata={"id": 3}),
    ]
    
    # Action
    results = service.rerank(query, candidates)
    
    # Verify
    assert len(results) == 3
    # Check sorting: highest score first
    assert results[0].score == 0.9
    assert results[0].text == "văn bản 2"
    assert results[1].score == 0.5
    assert results[1].text == "văn bản 3"
    assert results[2].score == 0.2
    assert results[2].text == "văn bản 1"
    
    # Verify mock calls
    mock_model.predict.assert_called_once()
    pairs = mock_model.predict.call_args[0][0]
    assert len(pairs) == 3
    assert pairs[0] == (query, "văn bản 1")
