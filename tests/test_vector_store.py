import pytest
from unittest.mock import MagicMock, patch
from app.core.services.vector_store import QdrantVectorStore
from app.schemas.common import ProcessedChunk, ChunkMetadata, ContentType

def test_vector_store_initialization():
    """Test that the vector store initializes with correct parameters."""
    store = QdrantVectorStore(host="remote-host", port=9999, collection_name="user-collection")
    assert store.host == "remote-host"
    assert store.port == 9999
    assert store.collection_name == "user-collection"
    assert store._client is None

@patch("app.core.services.vector_store.QdrantClient")
def test_ensure_collection_creates_missing(mock_qdrant_class):
    """Test that the collection is created if it doesn't exist."""
    mock_client = MagicMock()
    mock_qdrant_class.return_value = mock_client
    
    # Mock no collections exist
    mock_client.get_collections.return_value = MagicMock(collections=[])
    
    store = QdrantVectorStore(collection_name="new-coll")
    store._ensure_collection()
    
    mock_client.create_collection.assert_called_once()
    kwargs = mock_client.create_collection.call_args[1]
    assert kwargs["collection_name"] == "new-coll"

@patch("app.core.services.vector_store.QdrantClient")
def test_upsert_logic(mock_qdrant_class):
    """Test the upsert mapping from ProcessedChunk to Qdrant PointStruct."""
    mock_client = MagicMock()
    mock_qdrant_class.return_value = mock_client
    
    # Mock collection already exists
    mock_client.get_collections.return_value = MagicMock(collections=[
        MagicMock(name="student_handbook")
    ])
    
    store = QdrantVectorStore()
    
    chunk = ProcessedChunk(
        content="Học phí năm 2024",
        metadata=ChunkMetadata(
            file_name="quydinh.pdf",
            chunk_id="unique-uuid-123",
            content_type=ContentType.TEXT
        )
    )
    
    embedding = [0.1] * 768
    store.upsert([chunk], [embedding])
    
    mock_client.upsert.assert_called_once()
    kwargs = mock_client.upsert.call_args[1]
    assert kwargs["collection_name"] == "student_handbook"
    assert len(kwargs["points"]) == 1
    
    point = kwargs["points"][0]
    assert point.id == "unique-uuid-123"
    assert point.vector == embedding
    assert point.payload["content"] == "Học phí năm 2024"
    assert point.payload["file_name"] == "quydinh.pdf"
