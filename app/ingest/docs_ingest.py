from pathlib import Path
from typing import Any
from psycopg import Connection

from ..utils_hash import sha256_text
from ..embeddings import embed_texts
from ..summarizer import summarize
from .chunk_docs_markdown import chunk_markdown
from .db_ops import upsert_source, delete_chunks_for_source, insert_chunks_bulk

def ingest_markdown_file(
    conn: Connection,
    *,
    path: Path,
    doc_space: str | None = None,
    meta: dict[str, Any] | None = None,
    force_reindex: bool = False,
) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    normalized = raw.strip()
    source_hash = sha256_text(normalized)

    external_id = f"local_md:{str(path)}"
    source_id = upsert_source(
        conn,
        source_type="docs",
        external_id=external_id,
        title=path.name,
        content_hash=source_hash,
        doc_space=doc_space,
        page_id=None,
        url=None,
        updated_at=None,
        meta=meta or {},
    )

    # check if unchanged (TODO: optional optimization — here reindex unless we wire a check)
    if force_reindex:
        delete_chunks_for_source(conn, source_id)

    doc_chunks = chunk_markdown(normalized)
    # build embedding texts (include heading chain + summary)
    embed_inputs = []
    prepared_rows = []

    for i, ch in enumerate(doc_chunks):
        summary = summarize(ch.content)
        embed_text = f"{' > '.join(ch.heading_chain)}\n\n{summary}\n\n{ch.content}"
        embed_inputs.append(embed_text)

        prepared_rows.append({
            "source_id": source_id,
            "chunk_type": "section",
            "symbol": None,
            "signature": None,
            "heading_chain": ch.heading_chain,
            "content": ch.content,
            "embedding": None,   # fill after embedding
            "position": i,
            "content_hash": sha256_text(ch.content),
            "meta": {
                **(meta or {}),
                "summary": summary,
                "doc_space": doc_space,
            },
        })

    embs = embed_texts(embed_inputs)
    for row, emb in zip(prepared_rows, embs):
        row["embedding"] = emb

    inserted = insert_chunks_bulk(conn, prepared_rows)
    return {"source_id": source_id, "chunks_inserted": inserted, "chunks_total": len(prepared_rows)}