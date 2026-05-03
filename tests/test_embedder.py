import pytest
from unittest.mock import MagicMock, patch
from app.core.embedder import BkaiVietnameseEmbedder

def test_embedder_initialization():
    """Test that the embedder initializes with correct model name and deferred loading."""
    embedder = BkaiVietnameseEmbedder(model_name="test-model")
    assert embedder.model_name == "test-model"
    assert embedder._model is None

@patch("app.core.embedder.SentenceTransformer")
def test_embed_text(mock_st_class):
    """Test single text embedding call."""
    # Setup mock
    mock_model = MagicMock()
    # Mock return value as a numpy array like sentence-transformers would
    import numpy as np
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_st_class.return_value = mock_model
    
    embedder = BkaiVietnameseEmbedder()
    result = embedder.embed_text("Xin chào")
    
    assert result == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once_with("Xin chào")

@patch("app.core.embedder.SentenceTransformer")
def test_embed_batch(mock_st_class):
    """Test batch text embedding call."""
    # Setup mock
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_st_class.return_value = mock_model
    
    embedder = BkaiVietnameseEmbedder()
    texts = ["Câu một", "Câu hai"]
    result = embedder.embed_batch(texts)
    
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.assert_called_once_with(texts)
