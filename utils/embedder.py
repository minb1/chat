# generate_embeddings.py
import os
import argparse
import re
import json
from pathlib import Path
from typing import List, Dict, Any

# --- Assume chatRAG modules are importable ---
# Adjust sys.path if needed, e.g., if running from a different directory
# import sys
# sys.path.append('../') # Example if chatRAG is one level up
try:
    # We only need the embedding parts for this script
    from embedding.embedding_factory import get_embedding_handler
    from embedding.base_handler import BaseEmbeddingHandler
    from embedding.sentence_transformer_handler import SentenceTransformerHandler # Optional for type hint

except ImportError as e:
    print(f"Error importing chatRAG embedding modules: {e}")
    print("Please ensure the script is run from a location where 'chatRAG' is importable,")
    print("or adjust the Python path.")
    exit(1)

# --- Constants ---
DEFAULT_INPUT_DIR = "chunks_optimized"
DEFAULT_OUTPUT_DIR = "textembeddings"
DEFAULT_EMBEDDING_MODEL = "st-minilm" # Use SentenceTransformer MiniLM

# --- Helper Function ---
def parse_chunk_file(filepath: Path) -> str:
    """Reads a chunk file and returns the content after the YAML frontmatter."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading file {filepath}: {e}")
        return ""

    content_lines = []
    in_content = False
    sep_count = 0
    for line in lines:
        if line.strip() == '---':
            sep_count += 1
            if sep_count == 2:
                in_content = True
            continue # Skip the separator lines

        if in_content:
            content_lines.append(line)

    # Handle files with no frontmatter (shouldn't happen with the generator script)
    if sep_count < 2 and not content_lines:
         print(f"  Warning: No frontmatter found in {filepath}. Using all lines.")
         return "".join(lines).strip()
    elif sep_count == 0 and content_lines: # Case where only content exists
         return "".join(content_lines).strip()
    elif sep_count >= 2:
         return "".join(content_lines).strip()
    else: # Only frontmatter found
         print(f"  Warning: Only frontmatter found in {filepath}. No content to embed.")
         return ""

# --- Main Embedding Generation Logic ---
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

    # Ensure the base output directory exists
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {root_output_dir}")

    print("\nScanning for .txt chunk files...")
    chunk_files = list(root_input_dir.rglob("*.txt")) # Recursively find all .txt files
    print(f"Found {len(chunk_files)} chunk files.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for filepath in chunk_files:
        relative_path = filepath.relative_to(root_input_dir)
        print(f"\nProcessing [{processed_count + skipped_count + error_count + 1}/{len(chunk_files)}]: {relative_path}")

        # 1. Parse Content
        content = parse_chunk_file(filepath)
        if not content:
            print(f"  Skipping empty file or file with no content: {relative_path}")
            skipped_count += 1
            continue

        # 2. Generate Embedding
        try:
            print(f"  Generating embedding for {len(content)} characters...")
            embedding = embedding_handler.get_embedding(content)
            if not embedding or len(embedding) != embedding_handler.dimension:
                 print(f"  Error: Invalid embedding generated (Dim: {len(embedding) if embedding else 'None'} vs Expected: {embedding_handler.dimension}). Skipping.")
                 error_count += 1
                 continue
            print(f"  Embedding generated (dimension: {len(embedding)}).")

        except Exception as e:
            print(f"  Error generating embedding for {relative_path}: {e}")
            error_count += 1
            continue

        # 3. Prepare Output Path and Data
        output_subpath = relative_path.with_suffix(".json")
        output_filepath = root_output_dir / output_subpath
        output_data = {
            "file_path": str(relative_path).replace('\\', '/'), # Store relative path with forward slashes
            "embedding": embedding # Store the list of floats
        }

        # Ensure the specific output subdirectory exists
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        # 4. Save Embedding to JSON
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, indent=None) # Use None for compact storage
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

# --- Command Line Interface ---
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
        # You might want to expand choices later if needed
        choices=["st-minilm", "st-mpnet"], # Add other sentence-transformer models if desired
        help=f"Embedding model ID from embedding_factory.py (default: {DEFAULT_EMBEDDING_MODEL})"
    )

    args = parser.parse_args()

    # --- Initialization ---
    try:
        print("Initializing embedding handler...")
        embedding_handler = get_embedding_handler(args.embedding_model)
        # Trigger model loading here if desired, or let it lazy load on first use
        if hasattr(embedding_handler, 'model'):
             _ = embedding_handler.model # Access property to trigger download/load
             print(f"Embedding model '{embedding_handler.model_name}' loaded.")
        else:
             print("Could not explicitly load model, will load on first use.")


    except Exception as e:
        print(f"\nError initializing embedding handler: {e}")
        exit(1)

    # --- Run Generation ---
    try:
        generate_and_save_embeddings(args.input_dir, args.output_dir, embedding_handler)
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\nEmbedding generation process finished.")


def embed_chunks():
    # --- Initialization ---
    try:
        print("Initializing embedding handler...")
        embedding_handler = get_embedding_handler(DEFAULT_EMBEDDING_MODEL)
        # Trigger model loading here if desired, or let it lazy load on first use
        if hasattr(embedding_handler, 'model'):
            _ = embedding_handler.model  # Access property to trigger download/load
            print(f"Embedding model '{embedding_handler.model_name}' loaded.")
        else:
            print("Could not explicitly load model, will load on first use.")


    except Exception as e:
        print(f"\nError initializing embedding handler: {e}")
        exit(1)

    # --- Run Generation ---
    try:
        generate_and_save_embeddings(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, embedding_handler)
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\nEmbedding generation process finished.")