import os
import argparse
import re
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml

import sys
sys.path.append('../')
try:
    from embedding.embedding_factory import get_embedding_handler
    from embedding.base_handler import BaseEmbeddingHandler
    from embedding.sentence_transformer_handler import SentenceTransformerHandler
except ImportError as e:
    print(f"Error importing chatRAG embedding modules: {e}")
    print("Please ensure the script is run from a location where 'chatRAG' is importable,")
    print("or adjust the Python path.")
    exit(1)

DEFAULT_INPUT_DIR = "chunks_optimized"
DEFAULT_OUTPUT_DIR = "textembeddings"
DEFAULT_EMBEDDING_MODEL = "snowflake-arctic-embed-l-v2.0"

def parse_chunk_file(filepath: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Reads a chunk file, separates YAML frontmatter from content, and parses the YAML.
    Returns:
        Tuple[str, Optional[Dict[str, Any]]]: (content_string, metadata_dict or None)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except Exception as e:
        print(f"  Error reading file {filepath}: {e}")
        return "", None

    content_lines = []
    yaml_lines = []
    in_yaml = False
    in_content = False
    sep_count = 0

    for line in raw_content.splitlines():
        stripped_line = line.strip()
        if stripped_line == '---':
            sep_count += 1
            if sep_count == 1:
                in_yaml = True
            elif sep_count == 2:
                in_yaml = False
                in_content = True
            continue

        if in_yaml:
            yaml_lines.append(line)
        elif in_content:
            content_lines.append(line)
        elif sep_count == 0:
            content_lines.append(line)

    content_string = "\n".join(content_lines).strip()
    metadata = None

    if yaml_lines:
        yaml_string = "\n".join(yaml_lines)
        try:
            metadata = yaml.safe_load(yaml_string)
            if not isinstance(metadata, dict):
                print(f"  Warning: YAML frontmatter in {filepath} did not parse into a dictionary. Metadata ignored.")
                metadata = None
        except yaml.YAMLError as e:
            print(f"  Error parsing YAML frontmatter in {filepath}: {e}. Metadata ignored.")
            metadata = None
        except Exception as e:
            print(f"  Unexpected error parsing YAML in {filepath}: {e}. Metadata ignored.")
            metadata = None
    elif sep_count < 2 and content_string:
        print(f"  Warning: No frontmatter found in {filepath}. Proceeding without metadata.")
    elif not content_string:
        print(f"  Warning: No content found after frontmatter (or no content at all) in {filepath}.")

    return content_string, metadata

def construct_urls(metadata: Dict[str, Any]) -> Tuple[str, str]:
    """
    Constructs original_url and chunk_url from metadata.
    """
    base_url = "https://gitdocumentatie.logius.nl/publicatie/"
    original_html_path = metadata.get("original_html_path", "")
    anchor = metadata.get("anchor", "")
    folder_path = os.path.dirname(original_html_path)
    original_url = base_url + folder_path
    chunk_url = original_url + "#" + anchor if anchor else original_url
    return original_url, chunk_url

def generate_and_save_embeddings(input_dir: str, output_dir: str, embedding_handler: BaseEmbeddingHandler):
    """Finds chunk files, generates embeddings, and saves them to JSON files."""
    print(f"\n--- Starting Embedding Generation ---")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Embedding Model:  {embedding_handler.model_name if hasattr(embedding_handler, 'model_name') else 'N/A'}")
    print(f"Embedding Dim:    {embedding_handler.dimension}")

    root_input_dir = Path(input_dir)
    root_output_dir = Path(output_dir)

    if not root_input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Delete existing output directory if it exists
    if root_output_dir.exists():
        print(f"Cleaning previous output directory: {root_output_dir}")
        try:
            shutil.rmtree(root_output_dir)
            print("Previous output directory cleaned.")
        except OSError as e:
            print(f"Error removing directory {root_output_dir}: {e}. Proceeding might lead to old data.")

    # Create fresh output directory
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {root_output_dir}")

    print("\nScanning for .txt chunk files...")
    chunk_files = list(root_input_dir.rglob("*/*/*.txt"))
    print(f"Found {len(chunk_files)} chunk files.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for filepath in chunk_files:
        relative_path_to_input_root = filepath.relative_to(root_input_dir)
        print(
            f"\nProcessing [{processed_count + skipped_count + error_count + 1}/{len(chunk_files)}]: {relative_path_to_input_root}")

        content, metadata = parse_chunk_file(filepath)

        if not content:
            print(f"  Skipping empty file or file with no content: {relative_path_to_input_root}")
            skipped_count += 1
            continue

        if not metadata:
            print(f"  Skipping file due to missing or invalid metadata: {relative_path_to_input_root}")
            skipped_count += 1
            continue

        file_path_from_meta = metadata.get("file_path")
        doc_tag_from_meta = metadata.get("doc_tag")

        if not file_path_from_meta or not doc_tag_from_meta:
            print(f"  Skipping file due to missing 'file_path' or 'doc_tag' in metadata: {relative_path_to_input_root}")
            skipped_count += 1
            continue

        try:
            print(f"  Generating embedding for {len(content)} characters...")
            embedding = embedding_handler.get_embedding(content)
            if not embedding or len(embedding) != embedding_handler.dimension:
                print(
                    f"  Error: Invalid embedding generated (Dim: {len(embedding) if embedding else 'None'} vs Expected: {embedding_handler.dimension}). Skipping.")
                error_count += 1
                continue
            print(f"  Embedding generated (dimension: {len(embedding)}).")

        except Exception as e:
            print(f"  Error generating embedding for {relative_path_to_input_root}: {e}")
            error_count += 1
            continue

        if metadata:
            original_url, chunk_url = construct_urls(metadata)
        else:
            original_url = ""
            chunk_url = ""

        output_subpath = Path(file_path_from_meta).with_suffix(".json")
        output_filepath = root_output_dir / output_subpath

        output_data = {
            "file_path": file_path_from_meta,
            "doc_tag": doc_tag_from_meta,
            "original_url": original_url,
            "chunk_url": chunk_url,
            "embedding": embedding
        }

        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, indent=None)
            print(f"  Saved embedding to: {output_filepath}")
            processed_count += 1
        except Exception as e:
            print(f"  Error saving embedding JSON to {output_filepath}: {e}")
            error_count += 1

    print("\n--- Embedding Generation Summary ---")
    print(f"Total files found:        {len(chunk_files)}")
    print(f"Successfully processed:   {processed_count}")
    print(f"Skipped (no content):     {skipped_count}")
    print(f"Errors during processing: {error_count}")
    print(f"Embeddings saved to:      {output_dir}")
    print(f"----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for text chunks and save them as JSON.")
    parser.add_argument(
        "-i", "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing the chunk .txt files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the embedding .json files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-m", "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        choices=["st-minilm", "st-mpnet"],
        help=f"Embedding model ID from embedding_factory.py (default: {DEFAULT_EMBEDDING_MODEL})"
    )

    args = parser.parse_args()

    try:
        print("Initializing embedding handler...")
        embedding_handler = get_embedding_handler(args.embedding_model)
        if hasattr(embedding_handler, 'model'):
            _ = embedding_handler.model
            print(f"Embedding model '{embedding_handler.model_name}' loaded.")
        else:
            print("Could not explicitly load model, will load on first use.")

    except Exception as e:
        print(f"\nError initializing embedding handler: {e}")
        exit(1)

    try:
        generate_and_save_embeddings(args.input_dir, args.output_dir, embedding_handler)
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\nEmbedding generation process finished.")

def embed_chunks():
    try:
        print("Initializing embedding handler...")
        embedding_handler = get_embedding_handler(DEFAULT_EMBEDDING_MODEL)
        if hasattr(embedding_handler, 'model'):
            _ = embedding_handler.model
            print(f"Embedding model '{embedding_handler.model_name}' loaded.")
        else:
            print("Could not explicitly load model, will load on first use.")

    except Exception as e:
        print(f"\nError initializing embedding handler: {e}")
        exit(1)

    try:
        generate_and_save_embeddings(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, embedding_handler)
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\nEmbedding generation process finished.")