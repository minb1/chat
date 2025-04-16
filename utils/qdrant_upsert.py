import os
import json
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

def insert_into_qdrant(
    directory: Optional[str] = "textembeddings",
    file_path: Optional[str] = None,
    collection_name: str = "qdrant-logius",
    host: str = "qdrant",
    port: int = 6333,
    vector_size: int = 1024 # 1024 for Arctic embed
) -> int:
    """
    Insert embeddings into Qdrant from files or directory.
    """
    client = QdrantClient(host=host, port=port)
    print(f"Connected to Qdrant at {host}:{port}")

    embeddings_data = []

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
        print(f"Loading embeddings from {directory}...")
        embeddings_data = load_json_files(directory)
    else:
        print("Error: Either directory or file_path must be provided")
        return 0

    create_qdrant_collection(client, collection_name, vector_size)

    print(f"Inserting {len(embeddings_data)} embeddings into collection {collection_name}...")
    insert_embeddings(client, collection_name, embeddings_data)

    print("Import completed!")
    return len(embeddings_data)

def load_json_files(base_directory: str) -> List[Dict[str, Any]]:
    """
    Recursively load all JSON files from a directory structure.
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

def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size):
    """
    Create a Qdrant collection if it doesn't exist and ensure payload index exists.
    """
    try:
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
            print(f"Collection {collection_name} already exists.")

        try:
            collection_info = client.get_collection(collection_name=collection_name)
            existing_indexes = collection_info.payload_schema if hasattr(collection_info, 'payload_schema') else {}
            if 'doc_tag' not in existing_indexes:
                print(f"Creating payload index for 'doc_tag' in collection '{collection_name}'...")
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="doc_tag",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                print(f"Payload index for 'doc_tag' created or already exists.")
            else:
                print(f"Payload index for 'doc_tag' seems to exist.")
        except Exception as e:
            print(f"Warning: Could not verify or create payload index for 'doc_tag': {e}")

    except UnexpectedResponse as e:
        print(f"Error communicating with Qdrant during collection creation/check: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during collection setup: {e}")
        raise

def insert_embeddings(client: QdrantClient, collection_name: str, embeddings_data: List[Dict[str, Any]]) -> int:
    """
    Upsert embeddings into the Qdrant collection with unique IDs based on file_path.
    """
    batch_size = 100
    total_count = len(embeddings_data)
    inserted_count = 0

    if total_count == 0:
        print("No embeddings to insert!")
        return 0

    for i in range(0, total_count, batch_size):
        batch = embeddings_data[i:i + batch_size]
        points = []

        for item in batch:
            import hashlib
            import uuid
            file_path = item.get("file_path")
            if not file_path:
                print(f"Warning: Skipping item due to missing 'file_path'.")
                continue

            hashed_path = hashlib.sha1(file_path.encode('utf-8')).digest()
            point_id = str(uuid.UUID(bytes=hashed_path[:16]))

            embedding = item.get("embedding")
            if not embedding or not isinstance(embedding, list):
                print(f"Warning: Invalid embedding format for item ID {point_id} ({file_path}). Skipping.")
                continue

            doc_tag = item.get("doc_tag")
            if not doc_tag:
                print(f"Warning: Missing 'doc_tag' for item ID {point_id} ({file_path}). Using 'unknown'.")
                doc_tag = "unknown"

            payload = {
                "file_path": file_path,
                "doc_tag": doc_tag,
                "original_url": item.get("original_url", ""),
                "chunk_url": item.get("chunk_url", "")
            }

            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))

        if points:
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=False
                )
                inserted_count += len(points)
            except Exception as e:
                print(f"Error during Qdrant upsert batch starting at index {i}: {e}")

        print(f"  Upserted batch {i//batch_size + 1}/{(total_count + batch_size - 1)//batch_size}. Total processed: {min(i + batch_size, total_count)}/{total_count}")

    return inserted_count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import embeddings into Qdrant collection")
    parser.add_argument("--dir", type=str, help="Directory containing embedding files")
    parser.add_argument("--collection", type=str, default="qdrant-logius", help="Qdrant collection name")
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant server host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--file-path", type=str, help="Process a single file instead of a directory")

    args = parser.parse_args()

    insert_into_qdrant(
        directory=args.dir,
        file_path=args.file_path,
        collection_name=args.collection,
        host=args.host,
        port=args.port
    )