import os
from typing import List, Optional
from sentence_transformers import CrossEncoder
from app.core.interfaces.retrieval import IReranker
from app.core.schemas.retrieval import SearchResult

class ViRankerService(IReranker):
    """
    Reranking service using namdp-ptit/ViRanker (Cross-Encoder).
    Standard OOP implementation following the retrieval module's architecture.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        """
        Initialize the reranker service.
        
        Args:
            model_name: The HuggingFace model ID for ViRanker.
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", "namdp-ptit/ViRanker")
        self.device = device
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """
        Returns the cross-encoder model, loading it only if it hasn't been loaded yet.
        (Deferred loading pattern)
        """
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank candidates based on semantic relevance to the query.
        
        Args:
            query: The user query string.
            candidates: List of document chunks retrieved by search.
            
        Returns:
            Sorted list of RerankResult objects (highest score first).
        """
        if not candidates:
            return []

        # Prepare (query, passage) pairs for Cross-Encoder
        pairs = [(query, cand.content) for cand in candidates]
        
        # Scoring using the cross-encoder
        # Note: Cross-encoder's predict returns relevance scores directly
        scores = self.model.predict(pairs)
        
        # Combine candidates with their new scores
        results = []
        for i, cand in enumerate(candidates):
            results.append(
                SearchResult(
                    content=cand.content,
                    score=float(scores[i]),
                    metadata=cand.metadata
                )
            )
            
        # Sorting descending by semantic relevance score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
