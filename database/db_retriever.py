import sys
from typing import List
from logius.models import Document

def retrieve_chunks_from_db(relative_paths: List[str]) -> List[Document]:
    """
    Retrieves Document objects for given relative file paths from the PostgreSQL database using Django ORM.

    Args:
        relative_paths: A list of file paths as stored in the database.

    Returns:
        A list of Document objects corresponding to the provided file paths.
        Returns an empty list if no paths are provided or an error occurs.
    """
    if not relative_paths:
        return []

    try:
        documents = Document.objects.filter(file_path__in=relative_paths)
        found_paths = {doc.file_path for doc in documents}
        missing_paths = [p for p in relative_paths if p not in found_paths]
        if missing_paths:
            print(f"Warning: Content not found in DB for paths: {missing_paths}", file=sys.stderr)
        return list(documents)
    except Exception as e:
        print(f"Error retrieving chunks from database: {e}", file=sys.stderr)
        return []