from typing import List, Optional
from app.core.interfaces.retrieval import ISearchEngine, IReranker
from app.core.schemas.retrieval import SearchRequest, SearchResult
from app.core.schemas.common import ChunkMetadata

class SearchPipeline(ISearchEngine):
    """
    Search Pipeline that orchestrates retrieval and reranking.
    Follows ISearchEngine interface for seamless integration.
    """
    
    def __init__(self, search_engine: ISearchEngine, reranker: Optional[IReranker] = None):
        """
        Initialize the pipeline.
        
        Args:
            search_engine: Implementation of ISearchEngine (e.g. SemanticSearchEngine)
            reranker: Implementation of IReranker (e.g. ViRankerService)
        """
        self.search_engine = search_engine
        self.reranker = reranker

    def search(self, request: SearchRequest) -> List[SearchResult]:
        """
        1. Base Retrieval
        2. Optional Reranking
        """
        # 1. Execute initial retrieval (bi-encoder/hybrid)
        results = self.search_engine.search(request)
        
        # 2. Skip reranking if disabled, missing reranker, or no results
        if not request.use_rerank or self.reranker is None or not results:
            return results
            
        # 3. Perform reranking
        reranked = self.reranker.rerank(request.query, results)
        
        # 4. Map back to SearchResults, restoring ChunkMetadata models
        # We take only up to request.limit (top_k) results after reranking
        limit = request.limit_rerank
        final_results = reranked[:limit]
            
        return final_results
