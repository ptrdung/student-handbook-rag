import os
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.interfaces.base import IEmbedder

class BkaiVietnameseEmbedder(IEmbedder):
    """
    Vietnamese text embedder using bkai-foundation-models/vietnamese-bi-encoder.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # We wrap this in a property to defer loading the heavy model until actually used
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Convert single text string to vector embedding."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Convert list of text strings to vector embeddings using model's batching."""
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings.tolist()
