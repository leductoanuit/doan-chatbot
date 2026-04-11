"""Chat history persistence — sessions and messages stored in PostgreSQL.

Tables: chat_sessions, chat_messages (created via init_chat_schema)
"""

import json
import uuid
from typing import Optional

from src.storage.postgres_metadata import get_connection


def init_chat_schema() -> None:
    """Create chat_sessions and chat_messages tables if not exist."""
    sql_sessions = """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        title      TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    sql_messages = """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID        NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role       TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
        content    TEXT        NOT NULL,
        sources    JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    sql_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);",
        "CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at DESC);",
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_sessions)
            cur.execute(sql_messages)
            for idx in sql_indexes:
                cur.execute(idx)
        conn.commit()


def create_session(title: str = "") -> str:
    """Create a new chat session. Returns session UUID string."""
    session_id = str(uuid.uuid4())
    sql = "INSERT INTO chat_sessions (id, title) VALUES (%s, %s);"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, title or "Hội thoại mới"))
        conn.commit()
    return session_id


def update_session_title(session_id: str, title: str) -> None:
    """Update session title (set from first user message, truncated to 60 chars)."""
    sql = """
    UPDATE chat_sessions
    SET title = %s, updated_at = NOW()
    WHERE id = %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (title[:60], session_id))
        conn.commit()


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
) -> None:
    """Save a single chat message. sources is optional RAG citation list."""
    sql = """
    INSERT INTO chat_messages (session_id, role, content, sources)
    VALUES (%s, %s, %s, %s);
    """
    # Bump session updated_at so list_sessions orders correctly
    bump_sql = "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s;"
    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, role, content, sources_json))
            cur.execute(bump_sql, (session_id,))
        conn.commit()


def get_session_messages(session_id: str) -> list[dict]:
    """Load all messages for a session ordered by creation time."""
    sql = """
    SELECT role, content, sources, created_at
    FROM chat_messages
    WHERE session_id = %s
    ORDER BY created_at ASC;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
            rows = cur.fetchall()

    return [
        {
            "role": row[0],
            "content": row[1],
            "sources": row[2],  # psycopg2 auto-deserializes JSONB to Python object
            "created_at": row[3],
        }
        for row in rows
    ]


def list_sessions(limit: int = 20) -> list[dict]:
    """Return recent sessions ordered by last activity."""
    sql = """
    SELECT id, title, created_at, updated_at
    FROM chat_sessions
    ORDER BY updated_at DESC
    LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

    return [
        {
            "id": str(row[0]),
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
        }
        for row in rows
    ]
