import logging
from typing import List
from app.core.interfaces.retrieval import ISearchEngine
from app.core.interfaces.base import IEmbedder, IVectorStore
from app.core.schemas.retrieval import SearchRequest, SearchResult
from app.core.schemas.common import ChunkMetadata

logger = logging.getLogger(__name__)

class SemanticSearchEngine(ISearchEngine):
    """
    Core search logic that combines embedding generation and vector database retrieval.
    """
    def __init__(self, embedder: IEmbedder, vector_store: IVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Execute semantic search by embedding query and retrieving from vector store.
        """
        try:
            # 1. Generate embedding for the query
            query_vector = self.embedder.embed_text(request.query)
            
            # 2. Perform search in vector store
            results = self.vector_store.search(query_vector, limit=request.limit_search)
            
            # 3. Map to SearchResult schemas
            search_results = []
            for point in results:
                payload = point.payload.copy()
                content = payload.pop("content", "")
                
                try:
                    metadata = ChunkMetadata(**payload)
                    search_results.append(SearchResult(
                        content=content,
                        metadata=metadata,
                        score=point.score
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse metadata for point {point.id}: {e}")
                    continue
                    
            return search_results
            
        except Exception as e:
            logger.error(f"Search engine failure: {str(e)}")
            return []
