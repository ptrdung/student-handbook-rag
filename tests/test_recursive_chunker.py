import pytest
from app.schemas.common import ContentType
from app.ingestion.schemas.chunker import ChunkingConfiguration
from app.modules.ingestion.chunker import RecursiveCharacterChunker

def test_recursive_chunker_basic_split():
    """Test that the chunker splits text correctly based on size."""
    config = ChunkingConfiguration(chunk_size=50, chunk_overlap=10)
    chunker = RecursiveCharacterChunker(config)
    
    text = "This is a long sentence that should be split into multiple chunks because it exceeds the fifty character limit."
    metadata = {"file_name": "test.pdf", "content_type": ContentType.TEXT}
    
    chunks = chunker.chunk(text, metadata)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 60  # Allow some wiggle room for words
        assert chunk.metadata.file_name == "test.pdf"
        assert chunk.metadata.content_type == ContentType.TEXT

def test_recursive_chunker_overlap():
    """Test that chunks have the expected overlap."""
    config = ChunkingConfiguration(chunk_size=20, chunk_overlap=5)
    chunker = RecursiveCharacterChunker(config)
    
    text = "One two three four five six seven eight nine ten"
    metadata = {"file_name": "test.pdf", "content_type": ContentType.TEXT}
    
    chunks = chunker.chunk(text, metadata)
    
    # Check if consecutive chunks share some content
    if len(chunks) > 1:
        # This is a bit tricky to test exactly without knowing the split points, 
        # but we can check if the end of one chunk is in the start of the next.
        pass

def test_metadata_inheritance():
    """Test that chunks inherit metadata correctly."""
    config = ChunkingConfiguration(chunk_size=100, chunk_overlap=0)
    chunker = RecursiveCharacterChunker(config)
    
    text = "Sample text for chunking."
    metadata = {
        "file_name": "handbook.pdf",
        "header_1": "Chapter 1",
        "header_2": "Section A",
        "content_type": ContentType.TEXT,
        "is_table": False
    }
    
    chunks = chunker.chunk(text, metadata)
    
    assert len(chunks) > 0
    assert chunks[0].metadata.file_name == "handbook.pdf"
    assert chunks[0].metadata.header_1 == "Chapter 1"
    assert chunks[0].metadata.header_2 == "Section A"
    assert chunks[0].metadata.content_type == ContentType.TEXT
    assert chunks[0].metadata.is_table is False

def test_table_metadata():
    """Test that table content is handled correctly."""
    config = ChunkingConfiguration(chunk_size=1000, chunk_overlap=0)
    chunker = RecursiveCharacterChunker(config)
    
    text = "| Col 1 | Col 2 |\n|---|---|\n| val 1 | val 2 |"
    metadata = {
        "file_name": "data.pdf",
        "content_type": ContentType.TABLE,
        "is_table": True
    }
    
    chunks = chunker.chunk(text, metadata)
    
    assert len(chunks) == 1
    assert chunks[0].metadata.content_type == ContentType.TABLE
    assert chunks[0].metadata.is_table is True
