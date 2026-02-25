from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Literal
import re
import math

from psycopg import Connection
from psycopg.rows import dict_row

from ..embeddings import embed_texts


SourceType = Literal["docs", "code", "all"]


@dataclass
class SearchFilters:
    source_type: SourceType = "all"         # docs | code | all
    repo: Optional[str] = None
    branch: Optional[str] = None
    path_prefix: Optional[str] = None
    doc_space: Optional[str] = None

    # meta tags
    service: Optional[str] = None
    component: Optional[str] = None
    owners: Optional[str] = None

    # optional: only include chunks newer than this (uses sources.updated_at)
    updated_after_iso: Optional[str] = None


@dataclass
class Candidate:
    chunk_id: int
    source_id: int
    source_type: str
    chunk_type: str
    symbol: Optional[str]
    signature: Optional[str]
    heading_chain: Optional[list[str]]
    content: str
    meta: dict[str, Any]
    # ranking info
    score: float
    rank: int
    channel: str  # e.g. "docs_vec", "code_kw"


def _intent_signals(q: str) -> dict[str, bool]:
    ql = q.lower()
    return {
        "code_heavy": bool(re.search(r"`|\.py\b|\.ts\b|\.js\b|traceback|stack trace|exception|error:|where is|defined in|function|class|import ", ql)),
        "docs_heavy": bool(re.search(r"how do i|steps|runbook|guide|deploy|setup|install|troubleshoot|policy|procedure", ql)),
    }


def _rewrite_queries(q: str) -> list[str]:
    """
    query rewrite/expansion:
    - original
    - keywordy variant (keep identifiers, strip filler)
    - semantic-ish variant (add 'explain' / 'how to')
    """
    q = q.strip()
    if not q:
        return []

    keywordy = re.sub(r"[^A-Za-z0-9_\-./: ]+", " ", q)
    keywordy = re.sub(r"\s+", " ", keywordy).strip()

    semantic = q
    if not re.search(r"\bhow\b|\bwhy\b|\bexplain\b", q.lower()):
        semantic = f"Explain: {q}"

    rewrites = [q]
    if keywordy != q:
        rewrites.append(keywordy)
    if semantic != q and semantic != keywordy:
        rewrites.append(semantic)
    return rewrites[:3]


def _build_filter_sql(filters: SearchFilters) -> tuple[str, dict[str, Any]]:
    """
    Returns a SQL fragment and params to apply to both vector + keyword queries.
    We filter via sources + chunks fields.
    """
    clauses = []
    params: dict[str, Any] = {}

    if filters.source_type != "all":
        clauses.append("s.source_type = %(source_type)s::source_type")
        params["source_type"] = filters.source_type

    if filters.repo:
        clauses.append("s.repo = %(repo)s")
        params["repo"] = filters.repo

    if filters.branch:
        clauses.append("s.branch = %(branch)s")
        params["branch"] = filters.branch

    if filters.path_prefix:
        clauses.append("s.path LIKE %(path_prefix)s")
        params["path_prefix"] = filters.path_prefix.rstrip("/") + "%"

    if filters.doc_space:
        clauses.append("s.doc_space = %(doc_space)s")
        params["doc_space"] = filters.doc_space

    # meta filters (jsonb)
    if filters.service:
        clauses.append("s.meta->>'service' = %(service)s OR c.meta->>'service' = %(service)s")
        params["service"] = filters.service

    if filters.component:
        clauses.append("s.meta->>'component' = %(component)s OR c.meta->>'component' = %(component)s")
        params["component"] = filters.component

    if filters.owners:
        clauses.append("s.meta->>'owners' = %(owners)s OR c.meta->>'owners' = %(owners)s")
        params["owners"] = filters.owners

    if filters.updated_after_iso:
        clauses.append("(s.updated_at IS NOT NULL AND s.updated_at >= %(updated_after)s::timestamptz)")
        params["updated_after"] = filters.updated_after_iso

    if not clauses:
        return "TRUE", params

    return " AND ".join(f"({c})" for c in clauses), params


def _fetch_vector_candidates(
    conn: Connection,
    query_emb: list[float],
    *,
    filters: SearchFilters,
    top_k: int,
    channel: str,
) -> list[Candidate]:
    """
    Uses cosine distance. Since embeddings are normalized, cosine distance is fine.
    pgvector returns distance where smaller is better. We convert to score = 1 - distance.
    """
    filter_sql, params = _build_filter_sql(filters)
    params = dict(params)
    params["emb"] = query_emb
    params["k"] = top_k

    sql = f"""
    SELECT
      c.id AS chunk_id,
      c.source_id,
      s.source_type::text AS source_type,
      c.chunk_type::text AS chunk_type,
      c.symbol,
      c.signature,
      c.heading_chain,
      c.content,
      c.meta,
      (1 - (c.embedding <=> %(emb)s::vector)) AS score
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    WHERE c.embedding IS NOT NULL
      AND {filter_sql}
    ORDER BY c.embedding <=> %(emb)s::vector
    LIMIT %(k)s;
    """

    out: list[Candidate] = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    for i, r in enumerate(rows, start=1):
        out.append(
            Candidate(
                chunk_id=int(r["chunk_id"]),
                source_id=int(r["source_id"]),
                source_type=r["source_type"],
                chunk_type=r["chunk_type"],
                symbol=r.get("symbol"),
                signature=r.get("signature"),
                heading_chain=r.get("heading_chain"),
                content=r["content"],
                meta=r.get("meta") or {},
                score=float(r["score"] or 0.0),
                rank=i,
                channel=channel,
            )
        )
    return out


def _fetch_keyword_candidates(
    conn: Connection,
    query_text: str,
    *,
    filters: SearchFilters,
    top_k: int,
    channel: str,
) -> list[Candidate]:
    """
    Full-text retrieval via tsvector + websearch_to_tsquery.
    Score: ts_rank_cd (higher is better).
    """
    filter_sql, params = _build_filter_sql(filters)
    params = dict(params)
    params["q"] = query_text
    params["k"] = top_k

    sql = f"""
    SELECT
      c.id AS chunk_id,
      c.source_id,
      s.source_type::text AS source_type,
      c.chunk_type::text AS chunk_type,
      c.symbol,
      c.signature,
      c.heading_chain,
      c.content,
      c.meta,
      ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(q)s)) AS score
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    WHERE c.tsv IS NOT NULL
      AND c.tsv @@ websearch_to_tsquery('english', %(q)s)
      AND {filter_sql}
    ORDER BY score DESC
    LIMIT %(k)s;
    """

    out: list[Candidate] = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    for i, r in enumerate(rows, start=1):
        out.append(
            Candidate(
                chunk_id=int(r["chunk_id"]),
                source_id=int(r["source_id"]),
                source_type=r["source_type"],
                chunk_type=r["chunk_type"],
                symbol=r.get("symbol"),
                signature=r.get("signature"),
                heading_chain=r.get("heading_chain"),
                content=r["content"],
                meta=r.get("meta") or {},
                score=float(r["score"] or 0.0),
                rank=i,
                channel=channel,
            )
        )
    return out


def _rrf_fuse(lists: list[list[Candidate]], *, k: int = 60) -> list[Candidate]:
    """
    Reciprocal Rank Fusion:
      RRF(doc) = sum_i 1 / (k + rank_i)
    We compute fused_score per chunk_id across channels.
    """
    fused: dict[int, dict[str, Any]] = {}

    for cand_list in lists:
        for c in cand_list:
            if c.chunk_id not in fused:
                fused[c.chunk_id] = {
                    "cand": c,          # keep first copy as base payload
                    "rrf": 0.0,
                    "channels": set(),
                }
            fused[c.chunk_id]["rrf"] += 1.0 / (k + c.rank)
            fused[c.chunk_id]["channels"].add(c.channel)

    results: list[Candidate] = []
    for v in fused.values():
        base: Candidate = v["cand"]
        base.score = float(v["rrf"])
        base.channel = ",".join(sorted(v["channels"]))
        results.append(base)

    results.sort(key=lambda x: x.score, reverse=True)
    return results


def search_hybrid_rrf(
    conn: Connection,
    query: str,
    *,
    filters: SearchFilters,
    top_k_vec: int = 30,
    top_k_kw: int = 30,
    fused_top_k: int = 20,
) -> dict[str, Any]:
    """
    1) rewrite queries
    2) run (docs vec/kw) + (code vec/kw)
    3) fuse with RRF
    """
    rewrites = _rewrite_queries(query)
    if not rewrites:
        return {"query": query, "rewrites": [], "results": []}

    signals = _intent_signals(query)

    # determine corpora weights by adjusting k's (simple but effective)
    # if code-heavy: search more code candidates
    # if docs-heavy: search more docs candidates
    code_boost = 1.4 if signals["code_heavy"] and not signals["docs_heavy"] else 1.0
    docs_boost = 1.4 if signals["docs_heavy"] and not signals["code_heavy"] else 1.0

    all_lists: list[list[Candidate]] = []

    for i, rq in enumerate(rewrites):
        # vector embedding for this rewrite
        emb = embed_texts([rq])[0]

        # docs
        docs_filters = filters
        if filters.source_type == "all":
            docs_filters = SearchFilters(**{**filters.__dict__, "source_type": "docs"})
        all_lists.append(_fetch_vector_candidates(conn, emb, filters=docs_filters, top_k=int(top_k_vec * docs_boost), channel=f"docs_vec_q{i}"))
        all_lists.append(_fetch_keyword_candidates(conn, rq, filters=docs_filters, top_k=int(top_k_kw * docs_boost), channel=f"docs_kw_q{i}"))

        # code
        code_filters = filters
        if filters.source_type == "all":
            code_filters = SearchFilters(**{**filters.__dict__, "source_type": "code"})
        all_lists.append(_fetch_vector_candidates(conn, emb, filters=code_filters, top_k=int(top_k_vec * code_boost), channel=f"code_vec_q{i}"))
        all_lists.append(_fetch_keyword_candidates(conn, rq, filters=code_filters, top_k=int(top_k_kw * code_boost), channel=f"code_kw_q{i}"))

    fused = _rrf_fuse(all_lists, k=60)[:fused_top_k]

    # return structured payload
    results = []
    for c in fused:
        results.append({
            "chunk_id": c.chunk_id,
            "source_id": c.source_id,
            "source_type": c.source_type,
            "chunk_type": c.chunk_type,
            "symbol": c.symbol,
            "signature": c.signature,
            "heading_chain": c.heading_chain,
            "score": c.score,
            "channels": c.channel,
            "meta": c.meta,
            "content": c.content,
        })

    return {
        "query": query,
        "rewrites": rewrites,
        "signals": signals,
        "results": results,
    }