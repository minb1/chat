# chatRAG/embeddings/embedding_factory.py
import os
from typing import Dict, List
from .base_handler import BaseEmbeddingHandler
from .gemini_handler import GeminiEmbeddingHandler
from .sentence_transformer_handler import SentenceTransformerHandler


def get_available_embedding_models() -> Dict[str, str]:
    """Return a dictionary of available embedding models."""
    models = {}

    # Always add sentence-transformers as it doesn't require API keys
    models["st-minilm"] = "SentenceTransformer (all-MiniLM-L6-v2)"
    models["st-mpnet"] = "SentenceTransformer (all-mpnet-base-v2)"

    # Add API-dependent models if keys are available
    if os.getenv("GOOGLE_API_KEY"):
        models["gemini"] = "Gemini (text-embedding-004)"


    return models


def get_embedding_handler(model_name: str) -> BaseEmbeddingHandler:
    """Factory function to get the appropriate embedding model handler."""
    if model_name == "gemini":
        return GeminiEmbeddingHandler()
    elif model_name == "st-minilm":
        return SentenceTransformerHandler("all-MiniLM-L6-v2")
    elif model_name == "st-mpnet":
        return SentenceTransformerHandler("all-mpnet-base-v2")
    elif model_name == "snowflake-arctic-embed-l-v2.0":
        return SentenceTransformerHandler('Snowflake/snowflake-arctic-embed-l-v2.0')
    else:
        # Fallback to SentenceTransformer as it doesn't require API key
        return SentenceTransformerHandler()


def get_embedding_for_vector_store(vector_store: str) -> BaseEmbeddingHandler:
    """
    Return the appropriate embedding handler based on the vector store.

    Args:
        vector_store: The name/type of vector store (e.g., 'pinecone', 'qdrant')

    Returns:
        An embedding handler appropriate for the vector store
    """
    # Map vector stores to appropriate embedding models
    store_to_model = {
        "pinecone": "openai-small",  # Example: Pinecone uses OpenAI embeddings
        "qdrant": "Snowflake/snowflake-arctic-embed-l-v2.0",  # Example: Qdrant uses SentenceTransformers
        "chroma": "st-mpnet",  # Example: Chroma uses SentenceTransformer
    }

    model_name = store_to_model.get(vector_store, "Snowflake/snowflake-arctic-embed-l-v2.0")  # Default to SentenceTransformer
    return get_embedding_handler(model_name)