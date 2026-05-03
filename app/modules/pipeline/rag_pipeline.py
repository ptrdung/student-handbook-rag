from typing import AsyncGenerator, Optional, Dict, Any
from app.modules.retrieval import SearchPipeline
from app.modules.generation import GenerationPipeline
from app.core.schemas import SearchRequest, GenerationRequest, GenerationChunk, QueryType
from app.core.interfaces.base import IEmbedder
from app.core.interfaces.generation import ISemanticCache
from app.modules.routing import LLMQueryRouter

class RAGPipeline:
    """
    High-level orchestrator that connects Retrieval and Generation.
    Takes a query, finds relevant context, and generates an answer.
    """
    
    def __init__(
        self, 
        search_pipeline: SearchPipeline, 
        generation_pipeline: GenerationPipeline,
        router: LLMQueryRouter,
        cache: ISemanticCache = None,
        embedder: IEmbedder = None
    ):
        self.search_pipeline = search_pipeline
        self.generation_pipeline = generation_pipeline
        self.router = router
        self.cache = cache
        self.embedder = embedder

    async def answer(
        self, 
        query: str, 
        session_id: str,
        limit_search: int = 20,
        use_rerank: bool = True,
        limit_rerank: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[GenerationChunk, None]:
        """
        Full RAG flow: Cache -> Route -> (Search) -> Generate (Streaming)
        """
        import logging
        logger = logging.getLogger(__name__)

        # 0. Semantic Cache Check (Ultra-Fast Path)
        if self.cache and self.embedder:
            query_vector = self.embedder.embed_text(query)
            cached_answer = self.cache.get(query_vector)
            
            if cached_answer:
                logger.info(f"🚀 [PIPELINE CACHE HIT] Returning cached answer for: '{query[:50]}...'")
                yield GenerationChunk(text=cached_answer, is_last=False)
                yield GenerationChunk(text="", is_last=True)
                return

        # 1. Query Routing (Optimization)
        if self.router:
            routing_result = await self.router.route(query)
            logger.info(f"🚦 Query routed as: {routing_result.query_type} (Reason: {routing_result.reasoning})")
            
            if routing_result.query_type != QueryType.ACADEMIC_POLICY:
                # For non-academic queries, we bypass retrieval to save resources.
                # The LLM will respond based on chat history and its own knowledge.
                gen_request = GenerationRequest(
                    query=query,
                    context=[], # Empty context for non-RAG flow
                    session_id=session_id
                )
                async for chunk in self.generation_pipeline.process(gen_request):
                    yield chunk
                return

        # 1. Retrieval
        search_request = SearchRequest(
            query=query,
            limit_search=limit_search,
            use_rerank=use_rerank,
            limit_rerank=limit_rerank,
            filters=filters
        )
        retrieved_docs = self.search_pipeline.search(search_request)
        
        # 2. Generation
        gen_request = GenerationRequest(
            query=query,
            context=retrieved_docs,
            session_id=session_id
        )
        
        async for chunk in self.generation_pipeline.process(gen_request):
            yield chunk
