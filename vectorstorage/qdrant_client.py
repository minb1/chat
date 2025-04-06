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
        VectorsConfig
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
# --- End Qdrant Imports ---

load_dotenv()


class QdrantClient(BaseVectorClient):
    """Client for interacting with Qdrant vector database.

    This client handles connection, collection management, and vector operations
    for the Qdrant vector database.
    """

    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        """Initialize a Qdrant client.

        Args:
            collection_name: Name of the collection to use
            host: Qdrant server hostname or IP
            port: Qdrant server port

        Raises:
            ImportError: If qdrant-client package is not installed
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Please install qdrant-client: pip install qdrant-client")

        self.collection_name = collection_name
        self.host = host
        self.port = port
        self._client = None
        logger.info(f"Initialized QdrantClient for collection '{collection_name}' at {host}:{port}")

    @property
    def client(self) -> Qdrant:
        """Get or initialize the Qdrant client connection.

        Returns:
            Connected Qdrant client instance

        Raises:
            ConnectionError: If connection to Qdrant server fails
        """
        if self._client is None:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}...")
            try:
                self._client = Qdrant(host=self.host, port=self.port)
                # Test connection
                self._client.get_collections()
                logger.info("Qdrant connection successful")
            except Exception as e:
                error_msg = f"Could not connect to Qdrant at {self.host}:{self.port}. Error: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
        return self._client

    def ensure_collection_exists(
            self,
            vector_dim: int,
            distance_metric: Union[str, Distance] = Distance.COSINE
    ) -> None:
        """Checks if collection exists with correct parameters, creates it if not.

        Args:
            vector_dim: Dimension of vectors to be stored
            distance_metric: Distance metric for vector similarity

        Raises:
            ValueError: If collection exists with incompatible parameters
            ConnectionError: If connection to Qdrant fails
        """
        logger.info(f"Ensuring collection '{self.collection_name}' exists with dimension {vector_dim}...")

        # Convert string distance metric to enum if needed
        if isinstance(distance_metric, str):
            try:
                distance_metric = Distance[distance_metric.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid distance metric: {distance_metric}. Valid options: {[d.name for d in Distance]}")

        try:
            exists = self.client.collection_exists(collection_name=self.collection_name)

            if not exists:
                logger.info(f"Creating collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_dim, distance=distance_metric)
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
                return

            # Collection exists, verify parameters
            logger.info(f"Collection '{self.collection_name}' found. Verifying parameters...")
            self._verify_collection_parameters(vector_dim, distance_metric)

        except ConnectionError as e:
            # Re-raise connection errors
            raise e
        except Exception as e:
            error_msg = f"Error ensuring Qdrant collection '{self.collection_name}': {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _verify_collection_parameters(self, vector_dim: int, distance_metric: Distance) -> None:
        """Verify that the collection has the expected parameters.

        Args:
            vector_dim: Expected vector dimension
            distance_metric: Expected distance metric

        Raises:
            ValueError: If collection parameters don't match expectations
        """
        config: CollectionInfo = self.client.get_collection(collection_name=self.collection_name)

        # Extract vector parameters based on the configuration structure
        actual_params = self._extract_vector_params(config)

        if not actual_params:
            error_msg = (
                f"Collection '{self.collection_name}' exists, but could not verify parameters "
                f"for the default vector space. Expected: Dim={vector_dim}, Distance={distance_metric}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Verify parameters match
        current_dim = actual_params.size
        current_dist = actual_params.distance

        if current_dim != vector_dim or current_dist != distance_metric:
            error_msg = (
                f"Collection '{self.collection_name}' exists but has incompatible parameters. "
                f"Expected: Dim={vector_dim}, Distance={distance_metric}; "
                f"Found: Dim={current_dim}, Distance={current_dist}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Collection '{self.collection_name}' has correct parameters")

    def _extract_vector_params(self, config: CollectionInfo) -> Optional[VectorParams]:
        """Extract vector parameters from collection config.

        This handles different possible structures of the collection configuration.

        Args:
            config: Collection configuration

        Returns:
            Vector parameters or None if not found
        """
        # Case 1: Direct VectorParams (older style)
        if hasattr(config, 'vectors_config') and isinstance(config.vectors_config, VectorParams):
            logger.debug("Found default vector space configuration (VectorParams)")
            return config.vectors_config

        # Case 2: VectorsConfig with named vectors (newer style)
        if hasattr(config, 'vectors_config') and isinstance(config.vectors_config, VectorsConfig):
            logger.debug("Found VectorsConfig structure")
            if '' in config.vectors_config.params_map:
                return config.vectors_config.params_map['']

        # Case 3: Vectors dictionary (fallback)
        if hasattr(config, 'vectors') and isinstance(config.vectors, dict):
            logger.debug("Found 'vectors' dictionary structure")
            if '' in config.vectors:
                return config.vectors['']

        logger.warning("Could not extract vector parameters from collection config")
        return None

    def upsert_vectors(self, points: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Upsert vectors into the collection.

        Args:
            points: List of (id, vector, payload) tuples

        Raises:
            Exception: If upsert operation fails
        """
        if not points:
            logger.info("No points to upsert")
            return

        logger.info(f"Upserting {len(points)} points into collection '{self.collection_name}'...")

        try:
            # Convert to PointStruct objects
            point_structs = [
                PointStruct(id=point_id, vector=vector, payload=payload)
                for point_id, vector, payload in points
            ]

            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=point_structs,
                wait=True
            )

            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully upserted {len(points)} points")
            else:
                logger.warning(f"Upsert operation finished with status: {operation_info.status}")

        except Exception as e:
            error_msg = f"Error upserting vectors into Qdrant collection '{self.collection_name}': {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def retrieve_vectors(self, vector: List[float], top_k: int = 25) -> List[str]:
        """Retrieve file paths based on vector similarity.

        Args:
            vector: Query vector
            top_k: Number of similar vectors to retrieve

        Returns:
            List of file paths from the matched vector payloads

        Raises:
            Exception: If retrieval operation fails
        """
        logger.info(f"Retrieving top {top_k} vectors from '{self.collection_name}'...")

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True
            )

            chunk_paths = []
            for match in search_result:
                if match.payload and isinstance(match.payload, dict) and "file_path" in match.payload:
                    chunk_paths.append(match.payload["file_path"])
                else:
                    logger.warning(
                        f"Vector match (ID: {match.id}, Score: {match.score}) missing 'file_path'. "
                        f"Payload: {match.payload}"
                    )

            logger.info(f"Retrieved {len(chunk_paths)} file paths")
            return chunk_paths

        except Exception as e:
            error_msg = f"Error retrieving vectors from Qdrant: {str(e)}"
            logger.error(error_msg)
            return []