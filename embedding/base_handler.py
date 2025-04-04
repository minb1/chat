# chatRAG/embeddings/base_handler.py
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingHandler(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Convert text into embedding vector.

        Args:
            text: The text to encode

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vector."""
        pass

