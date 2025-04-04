import os
import json
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models


def insert_into_qdrant(
        directory: Optional[str] = "textembeddings",
        file_path: Optional[str] = None,
        collection_name: str = "qdrant-logius",
        host: str = "qdrant",
        port: int = 6333,
        vector_size: int = 384
) -> int:
    """
    Insert embeddings into Qdrant from files or directory.

    Args:
        directory: Directory containing embedding files
        file_path: Path to a specific embedding file (alternative to directory)
        collection_name: Name of the Qdrant collection
        host: Qdrant server host
        port: Qdrant server port
        vector_size: Dimension of the vectors

    Returns:
        Number of embeddings inserted
    """
    # Connect to Qdrant
    client = QdrantClient(host=host, port=port)
    print(f"Connected to Qdrant at {host}:{port}")

    embeddings_data = []

    # Process a single file if specified
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
                if isinstance(data, dict) and "embedding" in data:
                    embeddings_data.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "embedding" in item:
                            embeddings_data.append(item)
            print(f"Loaded {len(embeddings_data)} embeddings from {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    elif directory:
        # Load embeddings from directory
        print(f"Loading embeddings from {directory}...")
        embeddings_data = load_json_files(directory)
    else:
        print("Error: Either directory or file_path must be provided")
        return 0

    # Create collection
    create_qdrant_collection(client, collection_name, vector_size)

    # Insert embeddings
    print(f"Inserting {len(embeddings_data)} embeddings into collection {collection_name}...")
    insert_embeddings(client, collection_name, embeddings_data)

    print("Import completed!")
    return len(embeddings_data)


def load_json_files(base_directory: str) -> List[Dict[str, Any]]:
    """
    Recursively load all JSON files from a directory structure.

    Args:
        base_directory: The root directory to start the search

    Returns:
        A list of dictionaries containing the parsed JSON data
    """
    embeddings_data = []
    processed_files = 0

    print(f"Starting to scan directory: {base_directory}")

    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.json') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                processed_files += 1

                if processed_files % 100 == 0:
                    print(f"Processed {processed_files} files so far...")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Try to parse as a single JSON object first
                        try:
                            data = json.loads(content)
                            if isinstance(data, dict):
                                if "embedding" in data and "file_path" in data:
                                    embeddings_data.append(data)
                                    print(f"Found embedding in {file_path}")
                                else:
                                    print(f"Warning: File {file_path} is missing required fields")
                            elif isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and "embedding" in item and "file_path" in item:
                                        embeddings_data.append(item)
                                print(f"Found {len(data)} embeddings in {file_path}")
                        except json.JSONDecodeError:
                            # Try line by line for files with one JSON object per line
                            line_count = 0
                            for line in content.splitlines():
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        if "embedding" in data and "file_path" in data:
                                            embeddings_data.append(data)
                                            line_count += 1
                                    except json.JSONDecodeError:
                                        continue
                            if line_count > 0:
                                print(f"Found {line_count} embeddings in {file_path} (line-by-line)")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    print(f"Total files processed: {processed_files}")
    print(f"Total embeddings found: {len(embeddings_data)}")

    return embeddings_data


def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 384):
    """
    Create a Qdrant collection if it doesn't exist.

    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to create
        vector_size: Dimension of the vectors (384 for st-all-minilm)
    """
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection {collection_name} already exists")


def insert_embeddings(client: QdrantClient, collection_name: str, embeddings_data: List[Dict[str, Any]]):
    """
    Insert embeddings into the Qdrant collection.

    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        embeddings_data: List of dictionaries containing embeddings and metadata
    """
    batch_size = 100
    total_count = len(embeddings_data)

    if total_count == 0:
        print("No embeddings to insert!")
        return

    for i in range(0, total_count, batch_size):
        batch = embeddings_data[i:i + batch_size]
        points = []

        for idx, item in enumerate(batch):
            point_id = i + idx

            # Extract embedding vector
            embedding = item.get("embedding")
            if not embedding or not isinstance(embedding, list):
                print(f"Warning: Invalid embedding format for item {point_id}")
                continue

            # Verify embedding dimensions
            if len(embedding) != 384:
                print(f"Warning: Expected 384 dimensions but got {len(embedding)} for item {point_id}")

            # Create metadata/payload
            payload = {
                "file_path": item.get("file_path", "unknown"),
                # Add other metadata fields if needed
            }

            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))

        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )

        print(f"Processed {min(i + batch_size, total_count)}/{total_count} embeddings")


# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import embeddings into Qdrant collection")
    parser.add_argument("--dir", type=str, help="Directory containing embedding files")
    parser.add_argument("--collection", type=str, default="qdrant-logius", help="Qdrant collection name")
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant server host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--file-path", type=str, help="Process a single file instead of a directory")

    args = parser.parse_args()

    # Call the function with command line arguments
    insert_into_qdrant(
        directory=args.dir,
        file_path=args.file_path,
        collection_name=args.collection,
        host=args.host,
        port=args.port
    )