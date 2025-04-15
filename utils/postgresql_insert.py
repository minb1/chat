import os
import psycopg2
import sys
from typing import Optional, Tuple, Dict, Any
import yaml
import re

def parse_chunk_file_for_db(raw_content: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Lightweight parser to extract YAML metadata and content from the raw file string.
    Returns:
        Tuple[str, Optional[Dict[str, Any]]]: (full_content_string, metadata_dict or None)
    """
    metadata = None
    try:
        # Find YAML frontmatter boundaries
        match = re.match(r'^---\s*$(.*?)^---\s*$(.*)', raw_content, re.MULTILINE | re.DOTALL)
        if match:
            yaml_string = match.group(1).strip()
            # content_string = match.group(2).strip() # This would be just the markdown part
            try:
                metadata = yaml.safe_load(yaml_string)
                if not isinstance(metadata, dict):
                    metadata = None
            except yaml.YAMLError:
                metadata = None
        # else: # No frontmatter found
            # content_string = raw_content.strip()

    except Exception:
        # Fallback in case of regex or parsing error
        metadata = None
        # content_string = raw_content.strip()

    # Return the full original content and the parsed metadata
    return raw_content, metadata

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
    processed_count = process_files(cur, conn, chunks_directory)  # Add conn argument
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
    """Create documents table and add column/index separately if needed."""
    table_exists = False
    try:
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'documents');")
        table_exists = cur.fetchone()[0]
    except Exception as e:
        print(f"Error checking if table exists: {e}")
        # Handle error appropriately, maybe exit

    if not table_exists:
        print("Table 'documents' does not exist. Creating...")
        create_table_query = """
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            file_path TEXT UNIQUE,
            doc_tag TEXT, -- Add doc_tag column during creation
            content TEXT,
            inserted_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """
        try:
            cur.execute(create_table_query)
            print("Table 'documents' created successfully.")
        except Exception as e:
            print(f"FATAL: Error creating table 'documents': {e}")
            # Exit or raise because subsequent steps will fail
            sys.exit(1)
    else:
        print("Table 'documents' already exists. Checking for 'doc_tag' column...")
        column_exists = False
        try:
            cur.execute("SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name='documents' AND column_name='doc_tag');")
            column_exists = cur.fetchone()[0]
        except Exception as e:
             print(f"Error checking if 'doc_tag' column exists: {e}")
             # Decide how to proceed, maybe assume it doesn't exist and try to add

        if not column_exists:
            print("'doc_tag' column missing. Attempting to add...")
            try:
                cur.execute("ALTER TABLE documents ADD COLUMN doc_tag TEXT;")
                print("'doc_tag' column added successfully.")
            except Exception as e:
                print(f"FATAL: Error adding 'doc_tag' column: {e}")
                # Exit or raise
                sys.exit(1)
        else:
            print("'doc_tag' column already exists.")

    # --- Now handle index ---
    index_exists = False
    try:
         cur.execute("SELECT EXISTS (SELECT FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_documents_doc_tag' AND n.nspname = 'public');") # Check index existence
         index_exists = cur.fetchone()[0]
    except Exception as e:
         print(f"Warning: Error checking if index 'idx_documents_doc_tag' exists: {e}")

    if not index_exists:
        print("Index 'idx_documents_doc_tag' missing. Creating...")
        try:
            cur.execute("CREATE INDEX idx_documents_doc_tag ON documents (doc_tag);")
            print("Index 'idx_documents_doc_tag' created successfully.")
        except Exception as e:
            print(f"Warning: Error creating index 'idx_documents_doc_tag': {e}") # Non-fatal warning is okay
    else:
         print("Index 'idx_documents_doc_tag' already exists.")

    # --- Trigger for updated_at (keep as before) ---
    try:
        trigger_setup = """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
           NEW.updated_at = NOW();
           RETURN NEW;
        END;
        $$ language 'plpgsql';

        DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
        CREATE TRIGGER update_documents_updated_at
        BEFORE UPDATE ON documents
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
        """
        cur.execute(trigger_setup)
        print("Ensured 'updated_at' trigger exists.")
    except Exception as e:
        print(f"Warning: Error setting up 'updated_at' trigger: {e}")


def insert_or_update_document(cur, file_path, doc_tag, content):
    """Insert a new document or update if the file_path already exists, including doc_tag."""
    insert_query = """
    INSERT INTO documents (file_path, doc_tag, content, inserted_at, updated_at)
    VALUES (%s, %s, %s, NOW(), NOW())
    ON CONFLICT (file_path)
    DO UPDATE SET
        content = EXCLUDED.content,
        doc_tag = EXCLUDED.doc_tag, -- Update doc_tag too, just in case
        updated_at = NOW(); -- Let trigger handle this, or set explicitly
    """
    try:
        cur.execute(insert_query, (file_path, doc_tag, content))
        return True
    except Exception as e:
        print(f"Error inserting/updating {file_path} (tag: {doc_tag}): {e}")
        return False


def process_files(cur, conn, root_dir) -> int:
    """Process text files and insert them into the database."""
    processed_count = 0
    skipped_meta_count = 0

    print(f"Scanning directory {root_dir} for PostgreSQL insertion...")
    # Walk through the directory recursively
    # --- *** MODIFIED: Use correct walk pattern *** ---
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(dirpath, filename)
                # Don't calculate rel_path here, get it from metadata
                # rel_path = os.path.relpath(full_path, root_dir)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                    # Parse content for metadata
                    content_for_db, metadata = parse_chunk_file_for_db(raw_content)

                    if not metadata or 'file_path' not in metadata or 'doc_tag' not in metadata:
                        print(f"  Skipping {full_path}: Missing required metadata (file_path, doc_tag).")
                        skipped_meta_count += 1
                        continue

                    file_path_meta = metadata['file_path']
                    doc_tag_meta = metadata['doc_tag']

                    if insert_or_update_document(cur, file_path_meta, doc_tag_meta, content_for_db):
                        processed_count += 1
                        # print(f"Inserted/Updated: {file_path_meta} (Tag: {doc_tag_meta})")
                        if processed_count % 100 == 0:
                             print(f"  ... inserted/updated {processed_count} documents into PostgreSQL.")
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

    conn.commit() # Commit remaining transactions
    print(f"Finished PostgreSQL processing. Inserted/Updated: {processed_count}, Skipped (Metadata): {skipped_meta_count}")
    return processed_count


# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    # Default values when run directly
    chunks_dir = "chunks_optimized"

    # Allow overriding chunks directory via command line
    if len(sys.argv) > 1:
        chunks_dir = sys.argv[1]

    insert_chunks_to_postgres(chunks_directory=chunks_dir)