# chatRAG/vectorstorage/qdrant_client.py
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
import logging
from .base_client import BaseVectorClient

# Configure logging
logger = logging.getLogger(__name__)

# --- Qdrant Imports ---
try:
    from qdrant_client import QdrantClient as Qdrant
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        UpdateStatus,
        CollectionInfo,
        VectorsConfig,
        PointIdsList, # Import PointIdsList
        Record       # Import Record for retrieve return type
    )

    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False
    # Type hints for IDE support
    Qdrant = None
    Distance = None
    VectorParams = None
    PointStruct = None
    UpdateStatus = None
    CollectionInfo = None
    VectorsConfig = None
    PointIdsList = None # Add dummy type hint
    Record = None       # Add dummy type hint
# --- End Qdrant Imports ---

load_dotenv()


class QdrantClient(BaseVectorClient):
    # ... (keep existing __init__, client property, ensure_collection_exists, _verify_collection_parameters, _extract_vector_params, upsert_vectors) ...

    def retrieve_vectors(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve full payloads and scores based on vector similarity.

        Args:
            vector: Query vector
            top_k: Number of similar vectors to retrieve

        Returns:
            List of dictionaries, where each dictionary contains:
                'id': The Qdrant point ID (string or int)
                'score': The similarity score
                'payload': The full payload dictionary stored in Qdrant

        Raises:
            Exception: If retrieval operation fails (implicitly via Qdrant client)
        """
        logger.info(f"Retrieving top {top_k} vectors with payloads from '{self.collection_name}'...")

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,  # Ensure payload is requested
                with_vectors=False  # Usually not needed for retrieval results
            )

            retrieved_data = []
            for match in search_result:
                if match.payload and isinstance(match.payload, dict):
                    file_path = match.payload.get("file_path", "MISSING_PATH")
                    doc_tag = match.payload.get("doc_tag", "MISSING_TAG")

                    logger.debug(
                        f"Retrieved item via search: ID={match.id}, Score={match.score:.4f}, "
                        f"Path='{file_path}', Tag='{doc_tag}'"
                    )

                    retrieved_data.append({
                        # Ensure ID is consistently treated, e.g., as string if UUIDs are used
                        "id": str(match.id) if isinstance(match.id, uuid.UUID) else match.id,
                        "score": match.score,
                        "payload": match.payload
                    })
                else:
                    logger.warning(
                        f"Vector match (ID: {match.id}, Score: {match.score}) has missing or invalid payload. "
                        f"Payload: {match.payload}"
                    )

            logger.info(f"Retrieved {len(retrieved_data)} results with payload data via search.")
            return retrieved_data

        except Exception as e:
            error_msg = f"Error retrieving vectors via search from Qdrant collection '{self.collection_name}': {str(e)}"
            logger.exception(error_msg)
            return []

    # --- NEW METHOD ---
    def retrieve_by_ids(self, ids: List[Union[str, int, uuid.UUID]]) -> List[Dict[str, Any]]:
        """Retrieve points by their specific IDs.

        Args:
            ids: A list of Qdrant point IDs to retrieve.

        Returns:
            List of dictionaries, where each dictionary contains:
                'id': The Qdrant point ID
                'payload': The full payload dictionary stored in Qdrant
                'score': None (as this is direct retrieval, not search)

        Raises:
            Exception: If retrieval operation fails.
        """
        if not ids:
            logger.info("No IDs provided for direct retrieval.")
            return []

        logger.info(f"Retrieving {len(ids)} points by ID from '{self.collection_name}'...")

        try:
            # Qdrant retrieve expects a list of IDs in the format PointIdsList
            point_ids = PointIdsList(ids=[str(point_id) if isinstance(point_id, uuid.UUID) else point_id for point_id in ids])

            # Retrieve points with payload, without vectors
            records: List[Record] = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids.ids, # Pass the list of ids directly
                with_payload=True,
                with_vectors=False
            )

            retrieved_data = []
            found_ids = set()
            for record in records:
                if record.payload and isinstance(record.payload, dict):
                    point_id = str(record.id) if isinstance(record.id, uuid.UUID) else record.id
                    found_ids.add(point_id)
                    file_path = record.payload.get("file_path", "MISSING_PATH")
                    doc_tag = record.payload.get("doc_tag", "MISSING_TAG")
                    logger.debug(
                         f"Retrieved item by ID: ID={point_id}, Path='{file_path}', Tag='{doc_tag}'"
                    )
                    retrieved_data.append({
                        "id": point_id,
                        "score": None, # No score for direct retrieval
                        "payload": record.payload
                    })
                else:
                     logger.warning(f"Point retrieved by ID ({record.id}) has missing or invalid payload.")

            # Log missing IDs
            missing_ids = set(map(str, ids)) - found_ids # Ensure comparison uses strings
            if missing_ids:
                logger.warning(f"Could not retrieve points for IDs: {list(missing_ids)}")

            logger.info(f"Retrieved {len(retrieved_data)} points by ID.")
            return retrieved_data

        except Exception as e:
            error_msg = f"Error retrieving points by ID from Qdrant collection '{self.collection_name}': {str(e)}"
            logger.exception(error_msg)
            return []
    # --- END NEW METHOD ---
