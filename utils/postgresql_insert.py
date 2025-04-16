import os
import sys
import yaml
from typing import Tuple, Dict, Any
from django.db import transaction
from django.utils import timezone
from datetime import timedelta
from logius.models import Document, ChatQuery, Feedback

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

def insert_chunks_to_postgres(chunks_directory: str = "chunks_optimized", clear_old_feedback: bool = False) -> int:
    """
    Insert text chunks into PostgreSQL database using Django's ORM and handle feedback cleanup.

    Args:
        chunks_directory: Directory containing chunk files.
        clear_old_feedback: If True, delete feedback older than 30 days.

    Returns:
        Number of processed documents.
    """
    processed_count = 0
    skipped_meta_count = 0

    print(f"Scanning directory {chunks_directory} for PostgreSQL insertion...")
    with transaction.atomic():
        # Track existing file_paths
        old_file_paths = set(Document.objects.values_list('file_path', flat=True))

        # Delete existing documents
        deleted_docs = Document.objects.all().delete()
        print(f"Deleted {deleted_docs[1].get('logius.Document', 0)} documents")

        # Insert new documents
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
                        original_url = metadata.get('original_url')
                        chunk_url = metadata.get('chunk_url')

                        Document.objects.create(
                            file_path=file_path_meta,
                            doc_tag=doc_tag_meta,
                            content=content_string,
                            original_url=original_url,
                            chunk_url=chunk_url
                        )
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print(f"  ... inserted {processed_count} documents into PostgreSQL.")
                    except Exception as e:
                        print(f"Error processing {full_path}: {e}")

        # Check for changed file_paths
        new_file_paths = set(Document.objects.values_list('file_path', flat=True))
        changed_paths = old_file_paths - new_file_paths
        if changed_paths:
            print(f"Detected {len(changed_paths)} changed file_paths. Clearing related queries and feedback...")
            queries_to_clear = ChatQuery.objects.filter(file_paths__overlap=list(changed_paths))
            feedback_count = Feedback.objects.filter(query__in=queries_to_clear).count()
            query_count = queries_to_clear.count()
            queries_to_clear.delete()  # CASCADE deletes associated Feedback
            print(f"Deleted {query_count} queries and {feedback_count} feedback entries for changed file_paths")

        # Clear old feedback if requested
        if clear_old_feedback:
            old_date = timezone.now() - timedelta(days=30)
            deleted_feedback = Feedback.objects.filter(created_at__lt=old_date).delete()
            print(f"Deleted {deleted_feedback[1].get('logius.Feedback', 0)} feedback entries older than 30 days")

    print(f"Finished PostgreSQL processing. Inserted: {processed_count}, Skipped (Metadata): {skipped_meta_count}")
    return processed_count

if __name__ == "__main__":
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
    import django
    django.setup()
    chunks_dir = "chunks_optimized"
    clear_old = False
    if len(sys.argv) > 1:
        chunks_dir = sys.argv[1]
    if "--clear-old" in sys.argv:
        clear_old = True
    insert_chunks_to_postgres(chunks_directory=chunks_dir, clear_old_feedback=clear_old)