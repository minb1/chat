# chatRAG/vectorstorage/vector_factory.py
import os
from typing import Dict, List
from .base_client import BaseVectorClient
from .pinecone_client import PineconeClient
from .qdrant_client import QdrantClient

# Define knowledge bases with their configurations
KNOWLEDGE_BASES = {
    # Pinecone knowledge bases
    # Qdrant knowledge bases
    "qdrant-logius": {
        "type": "qdrant",
        "config": {
            "collection_name": "logius_standaarden",
            "host": "localhost",  # Service name in docker-compose
            "port": 6333
        }
    }
}


def get_available_knowledge_bases() -> Dict[str, str]:
    """
    Return a dictionary of available knowledge bases.

    Returns:
        Dictionary with KB IDs as keys and display names as values
    """
    available_kbs = {}

    # Check Pinecone availability
    if os.getenv("PINECONE_API_KEY"):
        available_kbs.update({
            "pinecone-legal": "Legal Documents (Pinecone)",
            "pinecone-technical": "Technical Documentation (Pinecone)"
        })

    # Qdrant is always available in docker setup
    available_kbs.update({
        "qdrant-logius": "Logius Standaarden (Qdrant)",
        "qdrant-legislation": "Legislation and Regulations (Qdrant)",
        "qdrant-research": "Research Papers (Qdrant)"
    })

    return available_kbs


def get_vector_client(kb_id):
    """
    Get the appropriate vector client based on the knowledge base ID.
    Modify this function to properly handle Docker container networking.
    """
    # Replace 'localhost' with the service name from docker-compose.yml
    if kb_id == 'qdrant-logius':
        # Use the service name from docker-compose.yml as the host
        return QdrantClient(collection_name='qdrant-logius', host='qdrant', port=6333)
    # Add other knowledge bases as needed
    else:
        raise ValueError(f"Unknown knowledge base ID: {kb_id}")


def get_embedding_model_for_kb(kb_id: str) -> str:
    """
    Get the appropriate embedding model ID for a knowledge base.

    Args:
        kb_id: Knowledge base identifier

    Returns:
        Embedding model identifier
    """
    # Map knowledge bases to appropriate embedding models
    kb_to_embedding = {
        "qdrant-logius": "snowflake-arctic-embed-l-v2.0",  # Product manuals use SentenceTransformer
    }

    return kb_to_embedding.get(kb_id, "snowflake-arctic-embed-l-v2.0")  # Default to SentenceTransformer