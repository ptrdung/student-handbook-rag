import os
import logging
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

class RedisService:
    """
    Service to manage Redis connection and health checks.
    Provides a fail-safe mechanism for semantic caching.
    """
    def __init__(self, url: str = None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client = None
        self._is_available = True

    @property
    def client(self) -> Redis:
        """Lazy initialization of Redis client."""
        if self._client is None:
            try:
                self._client = Redis.from_url(
                    self.url, 
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self._client.ping()
                self._is_available = True
                logger.info(f"✅ Connected to Redis at {self.url}")
            except (ConnectionError, TimeoutError) as e:
                self._is_available = False
                self._client = None
                logger.error(f"❌ Failed to connect to Redis: {str(e)}")
        return self._client

    def is_healthy(self) -> bool:
        """Check if Redis is available and responding."""
        if self._client is None:
            # Try to reconnect if it was previously unavailable
            return self.client is not None
        
        try:
            self._client.ping()
            self._is_available = True
            return True
        except (ConnectionError, TimeoutError):
            self._is_available = False
            self._client = None # Reset client for reconnection attempt next time
            return False

    def get_client(self):
        """Returns the client if healthy, otherwise None."""
        if self.is_healthy():
            return self._client
        return None
