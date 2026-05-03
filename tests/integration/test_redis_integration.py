import pytest
import time
import numpy as np
from app.core.services.redis_service import RedisService
from app.core.services.cache import RedisSemanticCache
from app.core.config import settings

def get_random_vector(dim=768):
    return np.random.rand(dim).tolist()

def test_redis_connection_live():
    """Test connection to live Redis instance."""
    service = RedisService()
    assert service.is_healthy() is True
    assert service.client.ping() is True

def test_semantic_cache_integration():
    """Test actual semantic search with RedisVL and live Redis."""
    service = RedisService()
    cache = RedisSemanticCache(service, threshold=0.9, ttl=60)
    
    # Clear cache before test
    if cache.cache:
        cache.cache.clear()
    
    query = "Học phí năm nhất là bao nhiêu?"
    # Tạo vector ngẫu nhiên cho tính đa dạng
    vector = get_random_vector()
    response = "Học phí năm nhất là 20 triệu đồng."
    
    # 1. Set
    cache.set(query, vector, response)
    
    # 2. Get exact match
    hit = cache.get(vector)
    assert hit == response
    
    # 3. Get similar match (thêm một chút nhiễu cực nhỏ)
    noise = np.random.normal(0, 0.0001, 768).tolist()
    similar_vector = (np.array(vector) + np.array(noise)).tolist()
    hit_similar = cache.get(similar_vector)
    assert hit_similar == response

def test_threshold_behavior():
    """Test that queries below threshold are not returned."""
    service = RedisService()
    # Ngưỡng rất cao để dễ test miss
    cache = RedisSemanticCache(service, threshold=0.99)
    
    if cache.cache:
        cache.cache.clear()
    
    query = "Thủ tục nhập học?"
    # Vector hướng X
    vector_a = [0.0] * 768
    vector_a[0] = 1.0
    
    response = "Vui lòng mang theo học bạ."
    
    cache.set(query, vector_a, response)
    
    # Vector hướng Y (vuông góc hoàn toàn với X -> similarity = 0)
    vector_b = [0.0] * 768
    vector_b[1] = 1.0
    
    miss = cache.get(vector_b)
    
    assert miss is None
