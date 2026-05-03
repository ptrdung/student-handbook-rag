import os
from functools import lru_cache
from app.core.services.embedder import BkaiVietnameseEmbedder
from app.core.services.vector_store import QdrantVectorStore
from app.modules.retrieval.reranker import ViRankerService
from app.modules.retrieval.semantic import SemanticSearchEngine
from app.modules.retrieval.search_pipeline import SearchPipeline

from app.modules.generation.llm import LLMService, GoogleLLMService
from app.modules.generation.memory import ConversationMemoryManager
from app.modules.generation.pipeline import GenerationPipeline
from app.modules.routing.router import LLMQueryRouter
from app.modules.pipeline.rag_pipeline import RAGPipeline

from app.modules.ingestion.vision_analyzer import NvidiaVisionAnalyzer,GoogleVisionAnalyzer
from app.modules.ingestion.table_parser import DoclingTableParser
from app.modules.ingestion.extractor import HybridExtractor
from app.modules.ingestion.chunker import RecursiveCharacterChunker
from app.modules.pipeline.ingestion_pipeline import IngestionPipeline
from app.core.schemas.ingestion import ChunkingConfiguration

from app.modules.generation.redis_service import RedisService
from app.modules.generation.cache import RedisSemanticCache

# --- Simple Service Registry (Singletons) ---

@lru_cache()
def get_embedder():
    return BkaiVietnameseEmbedder(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@lru_cache()
def get_redis_service():
    return RedisService()

@lru_cache()
def get_semantic_cache():
    return RedisSemanticCache(redis_service=get_redis_service())

@lru_cache()
def get_vector_store():
    return QdrantVectorStore()

@lru_cache()
def get_reranker():
    return ViRankerService()

@lru_cache()
def get_llm_service():
    provider = os.getenv("LLM_PROVIDER", "nvidia").lower()
    if provider == "google":
        return GoogleLLMService(
            semantic_cache=get_semantic_cache(),
            embedder=get_embedder()
        )
    return LLMService(
        semantic_cache=get_semantic_cache(),
        embedder=get_embedder()
    )

@lru_cache()
def get_memory_manager():
    return ConversationMemoryManager()

@lru_cache()
def get_query_router():
    return LLMQueryRouter(llm=get_llm_service())

# --- Pipeline Factories ---

def get_search_pipeline() -> SearchPipeline:
    semantic_search = SemanticSearchEngine(
        embedder=get_embedder(),
        vector_store=get_vector_store()
    )
    return SearchPipeline(
        search_engine=semantic_search,
        reranker=get_reranker()
    )

def get_generation_pipeline() -> GenerationPipeline:
    return GenerationPipeline(
        generator=get_llm_service(),
        memory_manager=get_memory_manager()
    )

def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        search_pipeline=get_search_pipeline(),
        generation_pipeline=get_generation_pipeline(),
        router=get_query_router(),
        cache=get_semantic_cache(),
        embedder=get_embedder()
    )

# --- Ingestion Factory ---

def get_ingestion_pipeline() -> IngestionPipeline:
    vision_analyzer = GoogleVisionAnalyzer()
    table_parser = DoclingTableParser()
    extractor = HybridExtractor(vision_analyzer, table_parser)
    
    # Default config
    config = ChunkingConfiguration(chunk_size=1000, chunk_overlap=200)
    chunker = RecursiveCharacterChunker(config)
    
    return IngestionPipeline(
        extractor=extractor,
        chunker=chunker,
        embedder=get_embedder(),
        vector_store=get_vector_store()
    )

# --- Evaluation Factory ---

def get_evaluation_pipeline() -> "EvaluationPipeline":
    from app.modules.pipeline.evaluation_pipeline import EvaluationPipeline
    from app.modules.evaluation.evaluators.search_evaluator import SearchEvaluator
    from app.modules.evaluation.metrics.hit_at_k import HitAtKMetric
    from app.modules.evaluation.metrics.mrr import MRRMetric
    from app.modules.evaluation.metrics.ndcg import NDCGMetric
    
    metrics = [HitAtKMetric(k=5), MRRMetric(k=5), NDCGMetric(k=5)]
    search_evaluator = SearchEvaluator(
        search_pipeline=get_search_pipeline(),
        metrics=metrics
    )
    
    return EvaluationPipeline(
        evaluators={
            "search": search_evaluator
        }
    )
