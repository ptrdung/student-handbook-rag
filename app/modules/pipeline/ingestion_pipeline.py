import logging
from pathlib import Path
from typing import List, Optional

from app.core.interfaces.base import IEmbedder, IVectorStore
from app.core.interfaces.ingestion import IExtractor, IChunker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the full RAG ingestion flow:
    Extract -> Chunking -> Embed -> Store
    """
    def __init__(
        self, 
        extractor: IExtractor, 
        chunker: IChunker, 
        embedder: IEmbedder, 
        vector_store: IVectorStore
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self, file_path: str, output_dir: Optional[str] = None, batch_size: int = 32) -> None:
        """
        Executes the full ingestion pipeline for a single PDF file.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return

        logger.info(f"🚀 Starting ingestion pipeline for: {path.name}")

        # 1. Extract content from PDF to Markdown
        logger.info("Step 1/4: Extracting content (PDF -> Markdown)...")
        markdown_text = self.extractor.process_pdf(file_path, output_dir=output_dir)
        
        # 2. Chunking the markdown content
        logger.info(f"Step 2/4: Chunking content...")
        # Basic metadata for chunking
        metadata = {"file_name": path.name}
        chunks = self.chunker.chunk(markdown_text, metadata)
        
        if not chunks:
            logger.warning(f"No chunks generated for {path.name}. Stopping.")
            return

        # 3. Generate embeddings
        logger.info(f"Step 3/4: Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts, batch_size=batch_size)

        # 4. Upsert to Vector Store
        logger.info("Step 4/4: Upserting to vector store...")
        self.vector_store.upsert(chunks, embeddings)

        logger.info(f"✅ Ingestion pipeline completed successfully for: {path.name}")
