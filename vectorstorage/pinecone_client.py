# chatRAG/vectorstorage/pinecone_client.py
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
from .base_client import BaseVectorClient

load_dotenv()


class PineconeClient(BaseVectorClient):
    def __init__(self, index_name="google-embed-004-768d", namespace="ns1"):
        self.index_name = index_name
        self.namespace = namespace
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found for Pinecone")

    def retrieve_vectors(self, vector: List[float], top_k: int = 50) -> List[str]:
        """Retrieve file paths from Pinecone based on vector similarity."""
        pc = Pinecone(api_key=self.api_key)
        index = pc.Index(self.index_name)

        response = index.query(
            namespace=self.namespace,
            vector=vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        # Retrieves all relative file paths
        chunk_paths = [match["metadata"].get("file_path") for match in response["matches"]]
        return chunk_paths