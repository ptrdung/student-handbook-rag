import functools
import logging
import time
import inspect
import os
from typing import List, Optional, Any, Callable
from redisvl.extensions.cache.llm import SemanticCache
from app.core.interfaces.generation import ISemanticCache
from app.modules.generation.redis_service import RedisService

logger = logging.getLogger(__name__)

class RedisSemanticCache(ISemanticCache):
    """
    Redis implementation of Semantic Cache using RedisVL.
    """
    def __init__(
        self, 
        redis_service: RedisService,
        threshold: float = None,
        ttl: int = None,
        vector_distance_threshold: float = None # redisvl uses distance, not similarity
    ):
        self.redis_service = redis_service
        self.threshold = float(threshold or os.getenv("CACHE_THRESHOLD", 0.85))
        self.ttl = int(ttl or os.getenv("CACHE_TTL", 3600))
        
        # RedisVL SemanticCache uses distance threshold. 
        # Typically distance = 1 - similarity.
        # If similarity threshold is 0.85, distance threshold is 0.15.
        self.distance_threshold = 1.0 - self.threshold
        
        self._cache = None

    @property
    def cache(self) -> Optional[SemanticCache]:
        """Lazy initialization of RedisVL SemanticCache."""
        if self._cache is None:
            client = self.redis_service.get_client()
            if client:
                try:
                    self._cache = SemanticCache(
                        redis_url=self.redis_service.url,
                        distance_threshold=self.distance_threshold,
                        ttl=self.ttl,
                        name="llm_semantic_cache"
                    )
                    logger.info("🚀 RedisVL SemanticCache initialized.")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize RedisVL SemanticCache: {str(e)}")
                    self._cache = None
        return self._cache

    def get(self, query_vector: List[float]) -> Optional[str]:
        """
        Retrieve cached response based on semantic similarity.
        """
        cache_instance = self.cache
        if not cache_instance:
            return None

        try:
            # RedisVL SemanticCache.check expects vector or text. 
            # We provide vector directly.
            hits = cache_instance.check(vector=query_vector)
            if hits:
                logger.info("🎯 Semantic Cache Hit!")
                return hits[0]["response"]
        except Exception as e:
            logger.warning(f"⚠️ Semantic Cache check failed: {str(e)}")
        
        return None

    def set(self, query_text: str, query_vector: List[float], response: str) -> None:
        """
        Store query and response in cache.
        """
        cache_instance = self.cache
        if not cache_instance:
            return

        try:
            cache_instance.store(
                prompt=query_text,
                response=response,
                vector=query_vector
            )
            logger.info("💾 Response stored in Semantic Cache.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to store in Semantic Cache: {str(e)}")
