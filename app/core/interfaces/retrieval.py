from abc import ABC, abstractmethod
from typing import List
from app.core.schemas.retrieval import SearchRequest, SearchResult

class ISearchEngine(ABC):
    @abstractmethod
    def search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Execute a search based on the provided request.
        
        Args:
            request: The search request containing the query and any filters.
            
        Returns:
            A list of search results.
        """
        pass

class IReranker(ABC):
    """
    Interface for document reranking services.
    """
    
    @abstractmethod
    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank a list of candidates based on their semantic relevance to a query.
        
        Args:
            query: The user query.
            candidates: List of RerankCandidate objects.
            
        Returns:
            A list of RerankResult objects sorted by score descending.
        """
        pass
