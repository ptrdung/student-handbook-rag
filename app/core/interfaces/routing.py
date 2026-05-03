from abc import ABC, abstractmethod
from app.core.schemas.routing import RoutingResult

class IQueryRouter(ABC):
    """Interface for query routing logic."""
    
    @abstractmethod
    async def route(self, query: str) -> RoutingResult:
        """
        Route a user query to the appropriate category.
        
        Args:
            query: The user's input query string.
            
        Returns:
            RoutingResult: The classification outcome.
        """
        pass
