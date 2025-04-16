import os
import sys
import yaml
from typing import Tuple, Dict, Any
from logius.models import Document

def parse_chunk_file_for_db(raw_content: str) -> Tuple[str, Dict[str, Any] | None]:
    """
    Lightweight parser to extract YAML metadata and content from the raw file string.

    Args:
        raw_content: The raw content of a chunk file.

    Returns:
        Tuple of (content string without frontmatter, metadata dictionary or None).
    """
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
                metadata = None
        except yaml.YAMLError:
            metadata = None

    return content_string, metadata

def insert_chunks_to_postgres(chunks_directory: str = "chunks_optimized") -> int:
    """
    Insert text chunks into PostgreSQL database using Django's ORM.

    Args:
        chunks_directory: Directory containing chunk files.

    Returns:
        Number of processed documents.
    """
    processed_count = 0
    skipped_meta_count = 0

    print(f"Scanning directory {chunks_directory} for PostgreSQL insertion...")
    for dirpath, _, filenames in os.walk(chunks_directory):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(dirpath, filename)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                    content_string, metadata = parse_chunk_file_for_db(raw_content)

                    if not metadata or 'file_path' not in metadata or 'doc_tag' not in metadata:
                        print(f"  Skipping {full_path}: Missing required metadata (file_path, doc_tag).")
                        skipped_meta_count += 1
                        continue

                    file_path_meta = metadata['file_path']
                    doc_tag_meta = metadata['doc_tag']

                    Document.objects.update_or_create(
                        file_path=file_path_meta,
                        defaults={
                            'doc_tag': doc_tag_meta,
                            'content': content_string
                        }
                    )
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"  ... inserted/updated {processed_count} documents into PostgreSQL.")
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

    print(f"Finished PostgreSQL processing. Inserted/Updated: {processed_count}, Skipped (Metadata): {skipped_meta_count}")
    return processed_count

if __name__ == "__main__":
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
    import django
    django.setup()
    chunks_dir = "chunks_optimized"
    if len(sys.argv) > 1:
        chunks_dir = sys.argv[1]
    insert_chunks_to_postgres(chunks_directory=chunks_dir)