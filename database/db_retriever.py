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

def retrieve_chunks_from_db(relative_paths: List[str]) -> str:
    """
    Retrieves content for given relative file paths from the PostgreSQL database
    and formats it as a single string for the RAG context.

    Args:
        relative_paths: A list of file paths as stored in the database (e.g., from Qdrant).

    Returns:
        A formatted string containing the content of the found documents,
        or an empty string if no paths are provided or a DB connection error occurs.
        Includes messages for paths not found in the DB.
    """
    if not relative_paths:
        return ""

    formatted_output = []
    conn = None
    cur = None

    try:
        conn, cur = connect_db()

        # Use ANY(%s) for efficient querying with a list of paths
        # Ensure the list is passed as a tuple for psycopg2
        query = """
        SELECT file_path, content
        FROM documents
        WHERE file_path = ANY(%s);
        """
        cur.execute(query, (relative_paths,)) # Pass the list as a tuple
        results = cur.fetchall()

        # Create a dictionary for quick lookup of retrieved content
        content_map: Dict[str, str] = {row[0]: row[1] for row in results}

        # Iterate through the original paths to maintain order and handle misses
        for rel_path in relative_paths:
            content = content_map.get(rel_path)
            if content:
                # Mimic the original formatting
                formatted_output.append(f"**{rel_path}**:\n\"{content}\"\n")
            else:
                # Indicate if a specific path wasn't found in the DB
                formatted_output.append(f"**{rel_path}**:\n(Content not found in database)\n")

    except (Exception, psycopg2.Error) as e:
        print(f"Error retrieving chunks from database: {e}", file=sys.stderr)
        # Return a generic error message or empty string depending on desired behavior
        return "Error retrieving context from database."
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return "\n".join(formatted_output)