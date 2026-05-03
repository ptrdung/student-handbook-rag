import pytest
from unittest.mock import MagicMock, patch
from app.core.services.cache import RedisSemanticCache
from app.core.services.redis_service import RedisService

@pytest.fixture
def mock_redis_service():
    service = MagicMock(spec=RedisService)
    service.url = "redis://localhost:6379"
    return service

def test_cache_initialization(mock_redis_service):
    """Test cache initialization with default and custom values."""
    cache = RedisSemanticCache(mock_redis_service, threshold=0.9, ttl=500)
    assert cache.threshold == 0.9
    assert cache.ttl == 500
    assert cache.distance_threshold == pytest.approx(0.1)
    assert cache._cache is None

@patch("app.core.services.cache.SemanticCache")
def test_cache_lazy_loading(mock_semantic_cache_class, mock_redis_service):
    """Test that SemanticCache is initialized only when needed."""
    mock_client = MagicMock()
    mock_redis_service.get_client.return_value = mock_client
    
    cache = RedisSemanticCache(mock_redis_service)
    
    # Access cache property to trigger initialization
    _ = cache.cache
    
    mock_semantic_cache_class.assert_called_once()
    assert cache._cache is not None

@patch("app.core.services.cache.SemanticCache")
def test_cache_get_hit(mock_semantic_cache_class, mock_redis_service):
    """Test cache hit scenario."""
    mock_cache_instance = MagicMock()
    mock_semantic_cache_class.return_value = mock_cache_instance
    mock_redis_service.get_client.return_value = MagicMock()
    
    mock_cache_instance.check.return_value = [{"response": "cached answer", "dist": 0.05}]
    
    cache = RedisSemanticCache(mock_redis_service)
    result = cache.get([0.1] * 768)
    
    assert result == "cached answer"
    mock_cache_instance.check.assert_called_once()

@patch("app.core.services.cache.SemanticCache")
def test_cache_get_miss(mock_semantic_cache_class, mock_redis_service):
    """Test cache miss scenario."""
    mock_cache_instance = MagicMock()
    mock_semantic_cache_class.return_value = mock_cache_instance
    mock_redis_service.get_client.return_value = MagicMock()
    
    mock_cache_instance.check.return_value = []
    
    cache = RedisSemanticCache(mock_redis_service)
    result = cache.get([0.1] * 768)
    
    assert result is None

@patch("app.core.services.cache.SemanticCache")
def test_cache_set(mock_semantic_cache_class, mock_redis_service):
    """Test storing value in cache."""
    mock_cache_instance = MagicMock()
    mock_semantic_cache_class.return_value = mock_cache_instance
    mock_redis_service.get_client.return_value = MagicMock()
    
    cache = RedisSemanticCache(mock_redis_service)
    cache.set("hello", [0.1] * 768, "hi there")
    
    mock_cache_instance.store.assert_called_once_with(
        prompt="hello",
        response="hi there",
        vector=[0.1] * 768
    )

@patch("app.core.services.cache.SemanticCache")
def test_cache_fail_safe(mock_semantic_cache_class, mock_redis_service):
    """Test that cache handles Redis unavailability gracefully."""
    # Mock RedisService returning None (connection failure)
    mock_redis_service.get_client.return_value = None
    
    cache = RedisSemanticCache(mock_redis_service)
    
    # Should not raise exception and return None
    result = cache.get([0.1] * 768)
    assert result is None
    
    # Should not raise exception
    cache.set("hello", [0.1] * 768, "hi there")
    mock_semantic_cache_class.assert_not_called()
