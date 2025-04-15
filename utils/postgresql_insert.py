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
        Tuple[str, Optional[Dict[str, Any]]]: (content_string without frontmatter, metadata_dict or None)
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
    """
    DB_HOST = db_host or os.environ.get("DB_HOST", "db")
    DB_PORT = db_port or os.environ.get("DB_PORT", "5432")
    DB_NAME = db_name or os.environ.get("DB_NAME", "logius-standaarden")
    DB_USER = db_user or os.environ.get("DB_USER", "postgres")
    DB_PASSWORD = db_password or os.environ.get("DB_PASSWORD", "postgres")

    conn, cur = connect_db(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)

    create_table(cur)

    processed_count = process_files(cur, conn, chunks_directory)

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

    if not table_exists:
        print("Table 'documents' does not exist. Creating...")
        create_table_query = """
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            file_path TEXT UNIQUE,
            doc_tag TEXT,
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
            sys.exit(1)
    else:
        print("Table 'documents' already exists. Checking for 'doc_tag' column...")
        column_exists = False
        try:
            cur.execute("SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name='documents' AND column_name='doc_tag');")
            column_exists = cur.fetchone()[0]
        except Exception as e:
            print(f"Error checking if 'doc_tag' column exists: {e}")

        if not column_exists:
            print("'doc_tag' column missing. Attempting to add...")
            try:
                cur.execute("ALTER TABLE documents ADD COLUMN doc_tag TEXT;")
                print("'doc_tag' column added successfully.")
            except Exception as e:
                print(f"FATAL: Error adding 'doc_tag' column: {e}")
                sys.exit(1)
        else:
            print("'doc_tag' column already exists.")

    index_exists = False
    try:
        cur.execute("SELECT EXISTS (SELECT FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_documents_doc_tag' AND n.nspname = 'public');")
        index_exists = cur.fetchone()[0]
    except Exception as e:
        print(f"Warning: Error checking if index 'idx_documents_doc_tag' exists: {e}")

    if not index_exists:
        print("Index 'idx_documents_doc_tag' missing. Creating...")
        try:
            cur.execute("CREATE INDEX idx_documents_doc_tag ON documents (doc_tag);")
            print("Index 'idx_documents_doc_tag' created successfully.")
        except Exception as e:
            print(f"Warning: Error creating index 'idx_documents_doc_tag': {e}")
    else:
        print("Index 'idx_documents_doc_tag' already exists.")

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
    doc_tag = EXCLUDED.doc_tag,
    updated_at = NOW();
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
    for dirpath, dirnames, filenames in os.walk(root_dir):
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

                    if insert_or_update_document(cur, file_path_meta, doc_tag_meta, content_string):
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print(f"  ... inserted/updated {processed_count} documents into PostgreSQL.")
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

    conn.commit()
    print(f"Finished PostgreSQL processing. Inserted/Updated: {processed_count}, Skipped (Metadata): {skipped_meta_count}")
    return processed_count

if __name__ == "__main__":
    chunks_dir = "chunks_optimized"
    if len(sys.argv) > 1:
        chunks_dir = sys.argv[1]
    insert_chunks_to_postgres(chunks_directory=chunks_dir)