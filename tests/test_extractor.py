import pytest
from app.ingestion import HybridExtractor

def test_hybrid_extractor_initialization():
    """Test that the extractor initializes correctly."""
    extractor = HybridExtractor()
    assert extractor is not None

def test_hybrid_extractor_process_missing_file():
    """Test that processing a non-existent file raises FileNotFoundError."""
    extractor = HybridExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.process_pdf("non_existent_file.pdf")
