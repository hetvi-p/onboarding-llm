from typing import Optional, Any
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def upsert_source(
    conn: Connection,
    *,
    source_type: str,
    external_id: str,
    title: Optional[str],
    content_hash: str,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    path: Optional[str] = None,
    language: Optional[str] = None,
    doc_space: Optional[str] = None,
    page_id: Optional[str] = None,
    url: Optional[str] = None,
    updated_at=None,
    meta: Optional[dict[str, Any]] = None,
) -> int:
    meta = meta or {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO sources (
              source_type, external_id, title, content_hash,
              repo, branch, path, language,
              doc_space, page_id, url, updated_at, meta, indexed_at
            )
            VALUES (
              %(source_type)s, %(external_id)s, %(title)s, %(content_hash)s,
              %(repo)s, %(branch)s, %(path)s, %(language)s,
              %(doc_space)s, %(page_id)s, %(url)s, %(updated_at)s,
              %(meta)s, now()
            )
            ON CONFLICT (external_id)
            DO UPDATE SET
              title = EXCLUDED.title,
              content_hash = EXCLUDED.content_hash,
              repo = EXCLUDED.repo,
              branch = EXCLUDED.branch,
              path = EXCLUDED.path,
              language = EXCLUDED.language,
              doc_space = EXCLUDED.doc_space,
              meta = EXCLUDED.meta,
              page_id = EXCLUDED.page_id,
              url = EXCLUDED.url,
              updated_at = EXCLUDED.updated_at,
              indexed_at = now()
            RETURNING id;
            """,
            {
                "source_type": source_type,
                "external_id": external_id,
                "title": title,
                "content_hash": content_hash,
                "repo": repo,
                "branch": branch,
                "path": path,
                "language": language,
                "doc_space": doc_space,
                "page_id": page_id,
                "url": url,
                "updated_at": updated_at,
                "meta": Jsonb(meta),
            },
        )
        return int(cur.fetchone()["id"])

def delete_chunks_for_source(conn: Connection, source_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id = %s;", (source_id,))

def insert_chunks_bulk(conn: Connection, rows: list[dict]) -> int:
    """
    rows: list of dicts with keys matching insert statement.
    """
    if not rows:
        return 0

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (
              source_id, chunk_type, symbol, signature, heading_chain,
              content, embedding, position, content_hash
            )
            VALUES (
              %(source_id)s, %(chunk_type)s, %(symbol)s, %(signature)s, %(heading_chain)s,
              %(content)s, %(embedding)s, %(position)s, %(content_hash)s
            )
            ON CONFLICT (source_id, content_hash) DO NOTHING;
            """,
            rows,
        )
        return cur.rowcount