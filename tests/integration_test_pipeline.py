import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock
from app.core.embedder import BkaiVietnameseEmbedder
from app.core.services.vector_store import QdrantVectorStore
from app.modules.ingestion.ingestion_pipeline import IngestionPipeline
from app.schemas.common import ProcessedChunk, ChunkMetadata, ContentType

def test_pipeline_integration_with_qdrant_mocked_embedder():
    """
    Integration test that runs the full pipeline against a real Qdrant instance.
    The embedding model is mocked to avoid heavy network/disk usage during tests.
    """
    # 1. Setup Mock Extractor and Chunker
    mock_extractor = MagicMock()
    mock_extractor.process_pdf.return_value = "Sample content for testing."
    
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [
        ProcessedChunk(
            content="Thông tin về học bổng sinh viên",
            metadata=ChunkMetadata(
                file_name="test.pdf", 
                chunk_id="00000000-0000-0000-0000-000000000001", 
                content_type=ContentType.TEXT
            )
        ),
        ProcessedChunk(
            content="Quy định về đăng ký học phần bổ sung",
            metadata=ChunkMetadata(
                file_name="test.pdf", 
                chunk_id="00000000-0000-0000-0000-000000000002", 
                content_type=ContentType.TEXT
            )
        )
    ]

    # 2. Setup Mock Embedder
    with patch("app.core.embedder.SentenceTransformer") as mock_st_class:
        mock_model = MagicMock()
        def mock_encode(texts, *args, **kwargs):
            return np.random.rand(len(texts), 768).astype(np.float32)
            
        mock_model.encode.side_effect = mock_encode
        mock_st_class.return_value = mock_model
        
        embedder = BkaiVietnameseEmbedder()
        
        # 3. Setup Real Vector Store (pointing to local docker)
        collection_name = "test_integration_collection"
        vector_store = QdrantVectorStore(collection_name=collection_name)
        
        # Ensure cleanup
        try:
            vector_store.client.delete_collection(collection_name)
        except Exception:
            pass
            
        # 4. Setup Pipeline
        pipeline = IngestionPipeline(mock_extractor, mock_chunker, embedder, vector_store)
        
        # Create a dummy file for the test
        dummy_file = "test.pdf"
        with open(dummy_file, "w") as f:
            f.write("dummy content")

        # 5. Run Pipeline
        pipeline.run(dummy_file)
        
        # 6. Verify in Qdrant (Wait a bit for consistency)
        time.sleep(1)
        
        # Search for anything
        search_results = vector_store.search(query_vector=[0.0]*768, limit=10)
        
        assert len(search_results) == 2
        ids = [res.id for res in search_results]
        assert "00000000-0000-0000-0000-000000000001" in ids
        assert "00000000-0000-0000-0000-000000000002" in ids
        
        # Check payload
        for res in search_results:
            if res.id == "00000000-0000-0000-0000-000000000001":
                assert res.payload["file_name"] == "test.pdf"
                assert "học bổng" in res.payload["content"]
        
        # Cleanup
        import os
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        vector_store.client.delete_collection(collection_name)
