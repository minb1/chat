# chatRAG/vectorstorage/qdrant_client.py
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
import logging
from .base_client import BaseVectorClient

# Configure logging
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

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
        PointIdsList, # For retrieving by specific IDs
        Record,       # Return type for retrieve operation
        SearchParams, # Explicitly import SearchParams if needed later
        Filter,       # Explicitly import Filter if needed later
        FieldCondition, # For payload filtering
        Range          # For payload filtering
    )
    from qdrant_client.http import models as rest

    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant client package not found. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False
    # Define dummy types for type hinting if Qdrant is not installed
    Qdrant = object
    Distance = object
    VectorParams = object
    PointStruct = object
    UpdateStatus = object
    CollectionInfo = object
    VectorsConfig = object
    PointIdsList = object
    Record = object
    SearchParams = object
    Filter = object
    FieldCondition = object
    Range = object
    rest = object
# --- End Qdrant Imports ---

load_dotenv() # Load environment variables from .env file


class QdrantClient(BaseVectorClient):
    """
    Client for interacting with a Qdrant vector database instance.

    Handles connection management, collection creation/verification,
    vector upsertion, similarity search, and direct ID retrieval.
    """

    def __init__(self, collection_name: str, host: Optional[str] = None, port: Optional[int] = 6333, api_key: Optional[str] = None, url: Optional[str] = None):
        """
        Initialize the Qdrant client.

        Can connect via host/port, URL, or Cloud API key/URL.

        Args:
            collection_name: Name of the Qdrant collection to interact with.
            host: Qdrant server hostname or IP address (e.g., "localhost").
                  Takes precedence over URL if both are provided without api_key.
            port: Qdrant server port (e.g., 6333). Used with host.
            api_key: API key for Qdrant Cloud. If provided, URL is also required.
            url: Full URL for Qdrant instance (e.g., "http://localhost:6333" or Cloud URL).
                 Takes precedence over host/port if api_key is provided.

        Raises:
            ImportError: If the 'qdrant-client' package is not installed.
            ValueError: If connection parameters are insufficient or conflicting.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Please install qdrant-client: pip install qdrant-client")

        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self.url = url
        self._client: Optional[Qdrant] = None

        # Determine connection parameters log message
        if self.api_key and self.url:
            connection_info = f"Qdrant Cloud at {self.url} using API key"
        elif self.url:
            connection_info = f"Qdrant at {self.url}"
        elif self.host and self.port:
            connection_info = f"Qdrant at {self.host}:{self.port}"
        else:
            # Attempt to load from environment variables as a fallback
            self.host = os.getenv("QDRANT_HOST", "localhost")
            self.port = int(os.getenv("QDRANT_PORT", 6333))
            self.api_key = os.getenv("QDRANT_API_KEY")
            self.url = os.getenv("QDRANT_URL")
            if self.api_key and self.url:
                 connection_info = f"Qdrant Cloud at {self.url} using API key (loaded from env)"
            elif self.url:
                 connection_info = f"Qdrant at {self.url} (loaded from env)"
            else: # Default to host/port from env or defaults
                 connection_info = f"Qdrant at {self.host}:{self.port} (loaded from env/defaults)"
                 self.url = None # Ensure URL isn't used if defaulting to host/port

        logger.info(f"Initialized QdrantClient for collection '{collection_name}'. Connection target: {connection_info}")


    @property
    def client(self) -> Qdrant:
        """
        Get or initialize the Qdrant client connection lazily.

        Returns:
            A connected Qdrant client instance.

        Raises:
            ConnectionError: If the connection to the Qdrant server fails.
            ValueError: If connection parameters are invalid.
        """
        if self._client is None:
            logger.info("Attempting to connect to Qdrant...")
            connection_args = {}
            if self.api_key and self.url:
                # Cloud connection
                connection_args['url'] = self.url
                connection_args['api_key'] = self.api_key
                logger.info(f"Connecting using URL: {self.url} and API Key.")
            elif self.url:
                 # URL connection (might be local or remote without API key)
                 connection_args['url'] = self.url
                 logger.info(f"Connecting using URL: {self.url}")
            elif self.host and self.port:
                # Host/Port connection (typically local)
                connection_args['host'] = self.host
                connection_args['port'] = self.port
                logger.info(f"Connecting using Host: {self.host}, Port: {self.port}")
            else:
                 # This case should ideally be handled in __init__ by falling back to env vars/defaults
                 error_msg = "Insufficient Qdrant connection parameters (host/port or url/api_key)."
                 logger.error(error_msg)
                 raise ValueError(error_msg)

            try:
                self._client = Qdrant(**connection_args)
                # Perform a lightweight operation to test the connection
                self._client.get_collections()
                logger.info("Qdrant connection successful.")
            except Exception as e:
                error_msg = f"Could not connect to Qdrant with provided configuration. Error: {e}"
                logger.error(error_msg, exc_info=True) # Include stack trace
                self._client = None # Ensure client remains None on failure
                raise ConnectionError(error_msg) from e

        return self._client

    def ensure_collection_exists(
            self,
            vector_dim: int,
            distance_metric: Union[str, Distance] = Distance.COSINE
    ) -> None:
        """
        Checks if the specified collection exists with the correct vector parameters.
        If the collection does not exist, it creates it.

        Args:
            vector_dim: The dimension of the vectors that will be stored in the collection.
            distance_metric: The distance metric to use for similarity calculations
                             (e.g., Distance.COSINE, Distance.EUCLID, Distance.DOT).
                             Can be passed as a string ('COSINE', 'EUCLID', 'DOT').

        Raises:
            ValueError: If the collection exists but has incompatible vector parameters,
                        or if an invalid distance metric string is provided.
            ConnectionError: If the connection to Qdrant fails.
            Exception: For other Qdrant-related errors during collection management.
        """
        logger.info(f"Ensuring collection '{self.collection_name}' exists with dimension {vector_dim}...")

        # Convert string distance metric to Qdrant Distance enum if necessary
        if isinstance(distance_metric, str):
            metric_upper = distance_metric.upper()
            if hasattr(Distance, metric_upper):
                distance_metric = getattr(Distance, metric_upper)
            else:
                valid_options = [d.name for d in Distance]
                raise ValueError(f"Invalid distance metric string: '{distance_metric}'. Valid options: {valid_options}")

        try:
            collection_exists = self.client.collection_exists(collection_name=self.collection_name)

            if not collection_exists:
                logger.info(f"Collection '{self.collection_name}' does not exist. Creating...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_dim, distance=distance_metric)
                    # Add other configurations like hnsw_config, optimizers_config if needed
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            else:
                logger.info(f"Collection '{self.collection_name}' found. Verifying parameters...")
                self._verify_collection_parameters(vector_dim, distance_metric)

        except ConnectionError: # Re-raise connection errors immediately
             raise
        except ValueError: # Re-raise ValueError (from _verify or metric conversion)
             raise
        except Exception as e:
            error_msg = f"An error occurred while ensuring Qdrant collection '{self.collection_name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e # Wrap other exceptions

    def _verify_collection_parameters(self, expected_dim: int, expected_distance: Distance) -> None:
        """
        Internal helper to verify if an existing collection's parameters match expectations.

        Args:
            expected_dim: The expected vector dimension.
            expected_distance: The expected distance metric (as Qdrant Distance enum).

        Raises:
            ValueError: If the collection parameters do not match the expected values.
            Exception: If unable to retrieve or parse collection configuration.
        """
        try:
            collection_info: CollectionInfo = self.client.get_collection(collection_name=self.collection_name)

            # --- Extract actual parameters ---
            # Modern Qdrant clients use collection_info.vectors_config
            if hasattr(collection_info, 'vectors_config') and collection_info.vectors_config:
                 # Can be VectorParams (single unnamed vector space) or VectorsConfig (named vectors)
                 if isinstance(collection_info.vectors_config, VectorParams):
                     actual_params = collection_info.vectors_config
                     logger.debug("Found single vector space config (VectorParams).")
                 elif isinstance(collection_info.vectors_config, VectorsConfig):
                     # Assume default unnamed vector space if multiple named spaces exist
                     if '' in collection_info.vectors_config.params_map:
                          actual_params = collection_info.vectors_config.params_map['']
                          logger.debug("Found multiple vector spaces config (VectorsConfig), using default ('').")
                     else: # Fallback if only named vectors exist, maybe pick first? Error is safer.
                          raise ValueError(f"Collection '{self.collection_name}' uses named vectors, but no default ('') space found for verification.")
                 else:
                      raise ValueError(f"Unexpected type for collection_info.vectors_config: {type(collection_info.vectors_config)}")
            else:
                # Handle older client versions or potentially unexpected responses
                 raise ValueError(f"Could not reliably determine vector parameters from collection info for '{self.collection_name}'.")
            # --- End Extraction ---

            actual_dim = actual_params.size
            actual_distance = actual_params.distance

            logger.info(f"Found existing collection parameters: Dimension={actual_dim}, Distance={actual_distance}")

            if actual_dim != expected_dim or actual_distance != expected_distance:
                error_msg = (
                    f"Collection '{self.collection_name}' exists but has incompatible parameters. "
                    f"Expected: Dim={expected_dim}, Distance={expected_distance}. "
                    f"Found: Dim={actual_dim}, Distance={actual_distance}."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                logger.info(f"Collection '{self.collection_name}' parameters verified successfully.")

        except Exception as e:
             error_msg = f"Failed to verify parameters for collection '{self.collection_name}': {e}"
             logger.error(error_msg, exc_info=True)
             # Re-raise specific errors or a general one
             if isinstance(e, ValueError): raise
             raise Exception(error_msg) from e


    def upsert_vectors(self, points: List[Tuple[Union[str, int, uuid.UUID], List[float], Dict[str, Any]]]) -> None:
        """
        Upserts (inserts or updates) vectors into the collection.

        Args:
            points: A list of tuples, where each tuple contains:
                    - point_id: A unique identifier (string, integer, or UUID) for the vector.
                    - vector: The vector embedding (list of floats).
                    - payload: A dictionary containing metadata associated with the vector.

        Raises:
            ValueError: If the input `points` list is empty.
            ConnectionError: If the connection to Qdrant fails.
            Exception: For other Qdrant-related errors during the upsert operation.
        """
        if not points:
            logger.warning("No points provided for upsert operation. Skipping.")
            # raise ValueError("Cannot upsert an empty list of points.") # Or just return
            return

        num_points = len(points)
        logger.info(f"Upserting {num_points} points into collection '{self.collection_name}'...")

        try:
            # Convert input tuples to Qdrant PointStruct objects
            point_structs = [
                PointStruct(
                    id=str(point_id) if isinstance(point_id, uuid.UUID) else point_id, # Ensure UUIDs are strings if needed by db/consistency
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in points
            ]

            # Perform the upsert operation
            # Use batching for very large lists if necessary (client might handle internally)
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=point_structs,
                wait=True # Wait for the operation to complete (recommended for consistency)
            )

            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully upserted {num_points} points. Operation ID: {operation_info.operation_id}")
            else:
                # Log warning if status is not completed (e.g., acknowledged but not finished)
                logger.warning(f"Upsert operation for {num_points} points finished with status: {operation_info.status}. Operation ID: {operation_info.operation_id}")

        except ConnectionError:
            raise
        except Exception as e:
            error_msg = f"Error upserting {num_points} vectors into Qdrant collection '{self.collection_name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e

    def retrieve_vectors(self, vector: List[float], top_k: int = 10, filter_conditions: Optional[Filter] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar vectors to the query vector,
        optionally applying a filter. Returns full points including payload and score.

        Args:
            vector: The query vector (list of floats).
            top_k: The maximum number of similar vectors to retrieve.
            filter_conditions: An optional Qdrant Filter object to apply constraints
                               on the payload of the vectors being searched.

        Returns:
            A list of dictionaries. Each dictionary represents a retrieved point and contains:
                - 'id': The Qdrant point ID (string or int).
                - 'score': The similarity score (float).
                - 'payload': The full payload dictionary stored with the vector in Qdrant.

        Raises:
            ConnectionError: If the connection to Qdrant fails.
            Exception: For other Qdrant-related errors during the search operation.
        """
        logger.info(f"Retrieving top {top_k} vectors via similarity search from '{self.collection_name}'...")
        if filter_conditions:
            logger.info(f"Applying filter: {filter_conditions}")

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=filter_conditions, # Pass the filter object here
                limit=top_k,
                with_payload=True,  # Request the payload
                with_vectors=False  # Vectors usually not needed in the result set
                # Add search_params (e.g., hnsw_ef, exact=True) if needed
            )

            retrieved_data = []
            for match in search_result:
                # Basic validation: ensure payload exists and is a dictionary
                if match.payload and isinstance(match.payload, dict):
                    point_id = str(match.id) if isinstance(match.id, uuid.UUID) else match.id
                    file_path = match.payload.get("file_path", "MISSING_PATH") # Safely get common fields
                    doc_tag = match.payload.get("doc_tag", "MISSING_TAG")

                    # Log details (consider DEBUG level)
                    logger.debug(
                        f"Retrieved item via search: ID={point_id}, Score={match.score:.4f}, "
                        f"Path='{file_path}', Tag='{doc_tag}'"
                    )

                    # Append the structured result
                    retrieved_data.append({
                        "id": point_id,
                        "score": match.score,
                        "payload": match.payload  # Include the full payload
                    })
                else:
                    logger.warning(
                        f"Vector search match (ID: {match.id}, Score: {match.score:.4f}) "
                        f"has missing or invalid payload. Payload: {match.payload}"
                    )

            logger.info(f"Retrieved {len(retrieved_data)} results via similarity search.")
            return retrieved_data

        except ConnectionError:
             raise
        except Exception as e:
            error_msg = f"Error retrieving vectors via search from Qdrant collection '{self.collection_name}': {e}"
            logger.exception(error_msg) # Use exception to include stack trace
            return [] # Return empty list on error

    def retrieve_by_ids(self, ids: List[Union[str, int, uuid.UUID]]) -> List[Dict[str, Any]]:
        """
        Retrieves specific points from the collection based on their IDs.

        Args:
            ids: A list of Qdrant point IDs (string, int, or UUID) to retrieve.

        Returns:
            A list of dictionaries. Each dictionary represents a retrieved point and contains:
                - 'id': The Qdrant point ID.
                - 'payload': The full payload dictionary stored with the vector.
                - 'score': None (as score is not applicable for direct ID retrieval).

        Raises:
            ConnectionError: If the connection to Qdrant fails.
            Exception: For other Qdrant-related errors during the retrieval operation.
        """
        if not ids:
            logger.info("No IDs provided for direct retrieval by ID. Skipping.")
            return []

        num_ids = len(ids)
        logger.info(f"Retrieving {num_ids} points by ID from '{self.collection_name}'...")

        try:
            # Convert all provided IDs to the format expected by Qdrant (string or int)
            # UUIDs generally need to be strings unless Qdrant specifically handles them
            processed_ids = [str(point_id) if isinstance(point_id, uuid.UUID) else point_id for point_id in ids]

            # Retrieve points by ID using the client's retrieve method
            records: List[Record] = self.client.retrieve(
                collection_name=self.collection_name,
                ids=processed_ids,
                with_payload=True, # Request the payload
                with_vectors=False # Vectors typically not needed
            )

            retrieved_data = []
            found_ids = set()
            for record in records:
                 # Record ID might be int or string depending on how it was inserted/configured
                 point_id = str(record.id) if isinstance(record.id, uuid.UUID) else record.id
                 found_ids.add(str(point_id)) # Store as string for comparison

                 if record.payload and isinstance(record.payload, dict):
                     file_path = record.payload.get("file_path", "MISSING_PATH")
                     doc_tag = record.payload.get("doc_tag", "MISSING_TAG")
                     logger.debug(
                         f"Retrieved item by ID: ID={point_id}, Path='{file_path}', Tag='{doc_tag}'"
                     )
                     retrieved_data.append({
                         "id": point_id,
                         "score": None, # No similarity score for direct retrieval
                         "payload": record.payload
                     })
                 else:
                     logger.warning(f"Point retrieved by ID ({point_id}) has missing or invalid payload. Payload: {record.payload}")

            # Log which requested IDs were not found
            requested_ids_str = set(map(str, processed_ids)) # Ensure comparison set uses strings
            missing_ids = requested_ids_str - found_ids
            if missing_ids:
                logger.warning(f"Could not retrieve points for {len(missing_ids)} requested IDs: {list(missing_ids)}")

            logger.info(f"Successfully retrieved {len(retrieved_data)} out of {num_ids} requested points by ID.")
            return retrieved_data

        except ConnectionError:
            raise
        except Exception as e:
            error_msg = f"Error retrieving points by ID from Qdrant collection '{self.collection_name}': {e}"
            logger.exception(error_msg) # Log with stack trace
            return [] # Return empty list on error

    def delete_collection(self) -> None:
        """Deletes the entire collection. Use with caution!"""
        logger.warning(f"Attempting to delete collection '{self.collection_name}'...")
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted successfully.")
        except Exception as e:
            error_msg = f"Failed to delete collection '{self.collection_name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e

    # Add other methods as needed, e.g., delete_points, update_payload, count_points, etc.
