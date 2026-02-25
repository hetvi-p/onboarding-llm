from pathlib import Path
from typing import Any
from psycopg import Connection

from ..utils_hash import sha256_text
from ..embeddings import embed_texts
from ..summarizer import summarize
from .db_ops import upsert_source, delete_chunks_for_source, insert_chunks_bulk
from .code_parse_python import parse_python_symbols
from .chunk_code_fallback import chunk_code_fallback

def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
    }.get(ext, "unknown")

def ingest_code_file(
    conn: Connection,
    *,
    file_path: Path,
    repo: str,
    branch: str,
    meta: dict[str, Any] | None = None,
    force_reindex: bool = False,
) -> dict:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    normalized = raw.strip()
    source_hash = sha256_text(normalized)

    language = detect_language(file_path)
    external_id = f"repo:{repo}@{branch}:{str(file_path)}"

    source_id = upsert_source(
        conn,
        source_type="code",
        external_id=external_id,
        title=file_path.name,
        content_hash=source_hash,
        repo=repo,
        branch=branch,
        path=str(file_path),
        language=language,
        meta=meta or {},
    )

    if force_reindex:
        delete_chunks_for_source(conn, source_id)

    if language == "python":
        symbol_chunks = parse_python_symbols(normalized, path=str(file_path))
    else:
        symbol_chunks = chunk_code_fallback(normalized, path=str(file_path))

    embed_inputs = []
    prepared_rows = []

    for i, ch in enumerate(symbol_chunks):
        # ingest-time summary (heuristic or local LLM)
        summary = summarize(ch.content)

        # embedding text: path + signature + summary + content excerpt
        embed_text = f"{ch.signature or ''}\n\nSUMMARY:\n{summary}\n\n{ch.content}"
        embed_inputs.append(embed_text)

        prepared_rows.append({
            "source_id": source_id,
            "chunk_type": ch.chunk_type if ch.chunk_type in ("function","class","module","block") else "block",
            "symbol": ch.symbol,
            "signature": ch.signature,
            "heading_chain": None,
            "content": ch.content,
            "embedding": None,
            "position": i,
            "content_hash": sha256_text(ch.content),
            "meta": {
                **(meta or {}),
                "summary": summary,
                "repo": repo,
                "branch": branch,
                "path": str(file_path),
                "language": language,
            },
        })

    embs = embed_texts(embed_inputs)
    for row, emb in zip(prepared_rows, embs):
        row["embedding"] = emb

    inserted = insert_chunks_bulk(conn, prepared_rows)
    return {"source_id": source_id, "chunks_inserted": inserted, "chunks_total": len(prepared_rows)}