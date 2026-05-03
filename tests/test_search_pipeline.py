import pytest
import time
from unittest.mock import patch, MagicMock
from app.modules.retrieval.semantic import SemanticSearchEngine
from app.core.embedder import BkaiVietnameseEmbedder
from app.core.services.vector_store import QdrantVectorStore
from app.schemas.common import ProcessedChunk, ChunkMetadata, ContentType
from app.retrieval.schemas.search import SearchRequest

def test_search_pipeline_integration():
    """
    Test the SearchEngine end-to-end with a real Qdrant instance.
    Embedder is mocked to avoid downloading heavy models.
    """
    # 1. Setup Mock Embedder
    with patch("app.core.embedder.SentenceTransformer") as mock_st_class:
        mock_model = MagicMock()
        
        def mock_encode(texts):
            import numpy as np
            # Simulation logic for embeddings to test retrieval
            if isinstance(texts, str):
                if "học bổng" in texts.lower():
                    # High value in first dimension
                    return np.array([1.0] + [0.0] * 767)
                elif "đăng ký" in texts.lower():
                    # High value in second dimension
                    return np.array([0.0, 1.0] + [0.0] * 766)
                return np.array([0.1] * 768)
            
            # Batch encoding
            results = []
            for t in texts:
                if "học bổng" in t.lower():
                    results.append([1.0] + [0.0] * 767)
                elif "đăng ký" in t.lower():
                    results.append([0.0, 1.0] + [0.0] * 766)
                else:
                    results.append([0.1] * 768)
            return np.array(results)
            
        mock_model.encode.side_effect = mock_encode
        mock_st_class.return_value = mock_model
        
        embedder = BkaiVietnameseEmbedder()
        
        # 2. Setup Real Vector Store
        collection_name = "test_search_engine_collection"
        vector_store = QdrantVectorStore(collection_name=collection_name)
        
        # Ensure fresh start
        try:
            vector_store.client.delete_collection(collection_name)
        except Exception:
            pass
            
        # 3. Seed data
        chunks = [
            ProcessedChunk(
                content="Chính sách về học bổng sinh viên nghèo vượt khó",
                metadata=ChunkMetadata(
                    file_name="hb.pdf",
                    chunk_id="00000000-0000-0000-0000-000000000001",
                    content_type=ContentType.TEXT
                )
            ),
            ProcessedChunk(
                content="Hướng dẫn đăng ký thi lại cho học viên",
                metadata=ChunkMetadata(
                    file_name="reg.pdf",
                    chunk_id="00000000-0000-0000-0000-000000000002",
                    content_type=ContentType.TEXT
                )
            )
        ]
        
        # We use the embedder to get vectors (which uses our mock)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)
        
        vector_store.upsert(chunks, embeddings)
        
        # Wait for Qdrant consistency
        time.sleep(1)
        
        # 4. Initialize Search Engine
        search_engine = SemanticSearchEngine(embedder, vector_store)
        
        # 5. Execute Search for "học bổng"
        request_hb = SearchRequest(query="Tôi muốn tìm học bổng", top_k=5)
        results_hb = search_engine.search(request_hb)
        
        assert len(results_hb) > 0
        # The top result should be the scholarship one
        assert results_hb[0].metadata.chunk_id == "00000000-0000-0000-0000-000000000001"
        assert "học bổng" in results_hb[0].content
        
        # 6. Execute Search for "đăng ký"
        request_reg = SearchRequest(query="Cách đăng ký", top_k=1)
        results_reg = search_engine.search(request_reg)
        
        assert len(results_reg) == 1
        assert results_reg[0].metadata.chunk_id == "00000000-0000-0000-0000-000000000002"
        
        # Cleanup
        vector_store.client.delete_collection(collection_name)
