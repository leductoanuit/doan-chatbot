"""
PostgreSQL metadata storage for Vietnamese legal documents.
Handles document metadata and chunk index persistence.
"""

import os
import uuid
from contextlib import contextmanager
from datetime import date
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

# Connection URL from environment, default to local dev DB
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/chatbot")


@contextmanager
def get_connection():
    """Yield a psycopg2 connection that auto-closes on exit."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    """Create tables and indexes if they do not already exist."""
    create_documents = """
    CREATE TABLE IF NOT EXISTS documents (
        id            UUID        PRIMARY KEY,
        title         TEXT,
        document_number TEXT,
        issue_date    DATE,
        effective_date DATE,
        issuing_body  TEXT,
        document_type TEXT,
        source_file   TEXT,
        source_url    TEXT,
        system_type   TEXT,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    create_chunks = """
    CREATE TABLE IF NOT EXISTS chunks (
        id             TEXT        PRIMARY KEY,
        document_id    UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_number    INT,
        chunk_index    INT,
        content_preview TEXT,
        token_count    INT,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    create_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_documents_issue_date    ON documents(issue_date);",
        "CREATE INDEX IF NOT EXISTS idx_documents_document_type ON documents(document_type);",
        "CREATE INDEX IF NOT EXISTS idx_documents_system_type   ON documents(system_type);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id      ON chunks(document_id);",
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_documents)
            cur.execute(create_chunks)
            for idx_sql in create_indexes:
                cur.execute(idx_sql)
        conn.commit()


def insert_document(metadata: dict) -> str:
    """
    Insert a document record and return its UUID string.
    `metadata` keys mirror the documents table columns (id is auto-generated).
    """
    doc_id = str(uuid.uuid4())

    sql = """
    INSERT INTO documents (
        id, title, document_number, issue_date, effective_date,
        issuing_body, document_type, source_file, source_url, system_type
    ) VALUES (
        %(id)s, %(title)s, %(document_number)s, %(issue_date)s, %(effective_date)s,
        %(issuing_body)s, %(document_type)s, %(source_file)s, %(source_url)s, %(system_type)s
    )
    ON CONFLICT (id) DO NOTHING;
    """

    row = {
        "id": doc_id,
        "title": metadata.get("title"),
        "document_number": metadata.get("document_number"),
        "issue_date": metadata.get("issue_date"),
        "effective_date": metadata.get("effective_date"),
        "issuing_body": metadata.get("issuing_body"),
        "document_type": metadata.get("document_type"),
        "source_file": metadata.get("source_file"),
        "source_url": metadata.get("source_url"),
        "system_type": metadata.get("system_type"),
    }

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, row)
        conn.commit()

    return doc_id


def insert_chunks_batch(chunks: list[dict]):
    """
    Batch-insert chunk records using execute_values for efficiency.
    Each dict must contain: id, document_id, page_number, chunk_index,
    content_preview, token_count.
    """
    if not chunks:
        return

    sql = """
    INSERT INTO chunks (id, document_id, page_number, chunk_index, content_preview, token_count)
    VALUES %s
    ON CONFLICT (id) DO NOTHING;
    """

    rows = [
        (
            c.get("id"),
            c.get("document_id"),
            c.get("page_number"),
            c.get("chunk_index"),
            c.get("content_preview"),
            c.get("token_count"),
        )
        for c in chunks
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()


def query_documents(
    doc_type: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    system_type: Optional[str] = None,
) -> list[dict]:
    """
    Filter documents by optional criteria.
    Returns list of dicts with all document columns.
    """
    conditions = []
    params: dict = {}

    if doc_type:
        conditions.append("document_type = %(doc_type)s")
        params["doc_type"] = doc_type
    if date_from:
        conditions.append("issue_date >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        conditions.append("issue_date <= %(date_to)s")
        params["date_to"] = date_to
    if system_type:
        conditions.append("system_type = %(system_type)s")
        params["system_type"] = system_type

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM documents {where_clause} ORDER BY issue_date DESC NULLS LAST;"

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def get_document_by_id(doc_id: str) -> Optional[dict]:
    """Fetch a single document by its UUID. Returns None if not found."""
    sql = "SELECT * FROM documents WHERE id = %(doc_id)s;"

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"doc_id": doc_id})
            row = cur.fetchone()
            return dict(row) if row else None
