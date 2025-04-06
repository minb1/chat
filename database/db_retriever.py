# database/db_retriever.py

import os
import psycopg2
import sys
from typing import List, Dict, Tuple

# Database connection parameters (reuse or centralize these)
# Consider moving these to a config file or central settings module
DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "logius-standaarden")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

# --- Database Connection ---
# Note: Opening/closing connections per query can be inefficient.
# Consider using a connection pool (e.g., psycopg2.pool) for applications
# that handle many requests. For simplicity here, we connect/disconnect.

def connect_db() -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        # No autocommit needed for SELECT queries
        # conn.autocommit = True # Only needed if you were doing INSERT/UPDATE/DELETE without explicit commit
        return conn, conn.cursor()
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}", file=sys.stderr)
        # Decide how to handle connection errors. Exit? Raise? Return None?
        # For a RAG app, perhaps returning empty results is better than crashing.
        # sys.exit(1) # Avoid exiting the whole app if possible
        raise ConnectionError(f"Could not connect to database: {e}")

# --- Retrieval Functions ---

def retrieve_chunks_from_db(relative_paths: List[str]) -> Dict[str, str]:
    """
    Retrieves content for given relative file paths from the PostgreSQL database.

    Args:
        relative_paths: A list of file paths as stored in the database (e.g., from Qdrant).

    Returns:
        A dictionary where keys are the found file paths and values are their content.
        Returns an empty dictionary if no paths are provided, no content is found,
        or a DB connection error occurs. Includes messages for paths not found in the DB.
    """
    if not relative_paths:
        return {}

    content_map: Dict[str, str] = {}
    conn = None
    cur = None

    try:
        conn, cur = connect_db() # Make sure this function exists and works

        # Use ANY(%s) for efficient querying with a list of paths
        # Ensure the list is passed correctly (as a list or tuple)
        query = """
        SELECT file_path, content
        FROM documents
        WHERE file_path = ANY(%s);
        """
        # Pass the list as a tuple or list for psycopg2
        cur.execute(query, (list(relative_paths),))
        results = cur.fetchall()

        # Create a dictionary for quick lookup of retrieved content
        content_map = {row[0]: row[1] for row in results}

        # You could optionally add markers for paths not found, but it might be
        # cleaner to just return what was found. The frontend will only display
        # items present in the returned dictionary.
        # Log missing paths if needed for debugging:
        found_paths = set(content_map.keys())
        missing_paths = [p for p in relative_paths if p not in found_paths]
        if missing_paths:
             print(f"Warning: Content not found in DB for paths: {missing_paths}", file=sys.stderr)


    except (Exception, psycopg2.Error) as e:
        print(f"Error retrieving chunks from database: {e}", file=sys.stderr)
        return {} # Return empty dict on error
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    # Return the dictionary directly
    return content_map