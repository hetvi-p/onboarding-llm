# app/answering/answer.py
from __future__ import annotations

from typing import Any, Optional

from ..settings import settings
from ..llm.ollama_client import ollama_generate
from ..retrieval.rerank import rerank_cross_encoder, rerank_ollama_judge


def build_context_block(results: list[dict[str, Any]], *, max_chars_each: int = 2600) -> str:
    """
    Formats chunks into a strict citation-friendly context block to prevent halluncination.
    Each chunk gets a stable citation key like [C1], [C2], ...
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        cite = f"[C{i}]"
        title_bits = []
        if r.get("source_type") == "code":
            # Try to show path if present in meta
            path = (r.get("meta") or {}).get("path")
            repo = (r.get("meta") or {}).get("repo")
            if repo and path:
                title_bits.append(f"{repo}:{path}")
            elif path:
                title_bits.append(str(path))
            if r.get("symbol"):
                title_bits.append(r["symbol"])
        else:
            # docs
            hc = r.get("heading_chain")
            if hc:
                title_bits.append(" > ".join(hc))
            url = (r.get("meta") or {}).get("url") or r.get("meta", {}).get("source_url")
            if url:
                title_bits.append(str(url))

        header = " | ".join([b for b in title_bits if b]) or "chunk"
        content = (r.get("content") or "")[:max_chars_each]

        blocks.append(
            f"{cite} {header}\n"
            f"chunk_id={r['chunk_id']} source_id={r['source_id']} type={r['chunk_type']}\n"
            f"{content}\n"
        )
    return "\n---\n".join(blocks)


def answer_with_ollama(
    query: str,
    *,
    context_results: list[dict[str, Any]],
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Produce a grounded answer with citations [C1], [C2], ...
    """
    context_block = build_context_block(context_results)

    prompt = (
        "You are an onboarding + technical assistant.\n"
        "You MUST answer using ONLY the provided CONTEXT.\n"
        "If the context is insufficient, say what is missing and ask 1-2 specific follow-up questions.\n"
        "Cite sources inline using [C#] tags.\n"
        "Do NOT invent file paths, function names, configs, commands, or behavior.\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        "RESPONSE FORMAT:\n"
        "1) Direct answer (with citations)\n"
        "2) Steps / code pointers (with citations)\n"
        "3) If uncertain: what’s missing + follow-up question(s)\n"
    )

    text = ollama_generate(prompt, model=model, temperature=0.2, max_tokens=900)
    return {"answer": text, "context_used": context_results}


def rerank_results(
    query: str,
    retrieved: list[dict[str, Any]],
    *,
    rerank_mode: str,
    rerank_top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    Takes retrieved results (already fused) and reranks to top N.
    """
    if rerank_mode == "none" or not retrieved:
        return retrieved[:rerank_top_n]

    candidates = retrieved[: min(len(retrieved), 40)]  # rerank only top 40 for speed

    if rerank_mode == "ollama_judge":
        reranked = rerank_ollama_judge(query, candidates)
    else:
        reranked = rerank_cross_encoder(query, candidates)

    # map chunk_id -> candidate
    by_id = {int(c["chunk_id"]): c for c in candidates}

    out = []
    for r in reranked:
        c = by_id.get(int(r.chunk_id))
        if not c:
            continue
        c = dict(c)
        c["rerank_score"] = r.score
        if r.reason:
            c["rerank_reason"] = r.reason
        out.append(c)

    # fallback: if something went wrong
    if not out:
        return retrieved[:rerank_top_n]
    return out[:rerank_top_n]