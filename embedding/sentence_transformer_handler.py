# chatRAG/embeddings/sentence_transformer_handler.py
from typing import List
import os
from .base_handler import BaseEmbeddingHandler


class SentenceTransformerHandler(BaseEmbeddingHandler):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Lazy loading to avoid importing if not used
        self._model = None
        self._dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-distilroberta-v1": 768,
            # Add more models and their dimensions as needed
        }

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using SentenceTransformers."""
        print(f"This is from the get_embedding func in st-mini")
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Error getting SentenceTransformer embedding: {str(e)}")

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model_name, 384)  # Default to 384 if model not in dict
