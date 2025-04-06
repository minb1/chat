import logging
from typing import List, Dict, Tuple, Any
from sentence_transformers import CrossEncoder

# Set up logger
logger = logging.getLogger(__name__)


class SentenceTransformerReranker:
    """
    Reranker using Sentence-Transformers CrossEncoder models.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a CrossEncoder model.

        Args:
            model_name: Name of the CrossEncoder model to use
        """
        logger.info(f"Initializing SentenceTransformerReranker with model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]],
               content_key: str = "content",
               top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: The user query
            documents: List of document dictionaries with content and metadata
            content_key: Key for accessing document content in the dictionary
            top_k: Number of top documents to return after reranking

        Returns:
            List of reranked documents (top_k)
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        try:
            # Prepare pairs for the cross-encoder
            pairs = [(query, doc[content_key]) for doc in documents]

            # Score all the pairs
            scores = self.model.predict(pairs)

            # Add scores to documents
            for i, score in enumerate(scores):
                documents[i]["rerank_score"] = float(score)

            # Sort documents by score in descending order
            reranked_documents = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

            # Return top_k documents
            return reranked_documents[:top_k]

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return documents[:top_k]  # Fallback to original order if reranking fails