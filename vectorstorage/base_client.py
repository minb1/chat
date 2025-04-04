# chatRAG/vectorstorage/base_client.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorClient(ABC):
    @abstractmethod
    def retrieve_vectors(self, vector: List[float], top_k: int = 50) -> List[str]:
        """
        Retrieve file paths based on vector similarity.

        Args:
            vector: The query embedding vector
            top_k: Number of results to return

        Returns:
            List of file paths of relevant documents
        """
        pass