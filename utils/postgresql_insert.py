import os
import psycopg2
import sys
from typing import Optional, Tuple, Dict


def insert_chunks_to_postgres(
        chunks_directory: str = "chunks_optimized",
        db_host: Optional[str] = None,
        db_port: Optional[str] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None
) -> int:
    """
    Insert text chunks into PostgreSQL database.

    Args:
        chunks_directory: Directory containing text chunks
        db_host: Database host address (defaults to DB_HOST env var or "db")
        db_port: Database port (defaults to DB_PORT env var or "5432")
        db_name: Database name (defaults to DB_NAME env var or "logius-standaarden")
        db_user: Database user (defaults to DB_USER env var or "postgres")
        db_password: Database password (defaults to DB_PASSWORD env var or "postgres")

    Returns:
        Number of documents processed
    """
    # Database connection parameters (use provided values or environment variables)
    DB_HOST = db_host or os.environ.get("DB_HOST", "db")
    DB_PORT = db_port or os.environ.get("DB_PORT", "5432")
    DB_NAME = db_name or os.environ.get("DB_NAME", "logius-standaarden")
    DB_USER = db_user or os.environ.get("DB_USER", "postgres")
    DB_PASSWORD = db_password or os.environ.get("DB_PASSWORD", "postgres")

    # Connect to database
    conn, cur = connect_db(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)

    # Create table if it doesn't exist
    create_table(cur)

    # Process files and insert to database
    processed_count = process_files(cur, chunks_directory)

    # Close connections
    cur.close()
    conn.close()
    print(f"Finished processing {processed_count} text files.")

    return processed_count


def connect_db(host, port, dbname, user, password) -> Tuple:
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        conn.autocommit = True
        return conn, conn.cursor()
    except Exception as e:
        print("Error connecting to PostgreSQL:", e)
        sys.exit(1)


def create_table(cur):
    """Create documents table if it doesn't exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        file_path TEXT UNIQUE,
        content TEXT,
        inserted_at TIMESTAMP DEFAULT NOW()
    );
    """
    cur.execute(create_table_query)


def insert_or_update_document(cur, file_path, content):
    """Insert a new document or update if the file_path already exists."""
    insert_query = """
    INSERT INTO documents (file_path, content)
    VALUES (%s, %s)
    ON CONFLICT (file_path)
    DO UPDATE SET content = EXCLUDED.content, inserted_at = NOW();
    """
    try:
        cur.execute(insert_query, (file_path, content))
        return True
    except Exception as e:
        print(f"Error inserting/updating {file_path}: {e}")
        return False


def process_files(cur, root_dir) -> int:
    """Process text files and insert them into the database."""
    processed_count = 0

    # Walk through the directory recursively
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(dirpath, filename)
                # Get the relative path (this will serve as your identifier)
                rel_path = os.path.relpath(full_path, root_dir)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if insert_or_update_document(cur, rel_path, content):
                        processed_count += 1
                        print(f"Inserted/Updated: {rel_path}")
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

    return processed_count


# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    # Default values when run directly
    chunks_dir = "chunks_optimized"

    # Allow overriding chunks directory via command line
    if len(sys.argv) > 1:
        chunks_dir = sys.argv[1]

    insert_chunks_to_postgres(chunks_directory=chunks_dir)