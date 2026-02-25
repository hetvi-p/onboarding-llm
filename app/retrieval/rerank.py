from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Literal

from ..settings import settings
from ..llm.ollama_client import ollama_generate

RerankMode = Literal["cross_encoder", "ollama_judge", "none"]

@dataclass
class Reranked:
    chunk_id: int
    score: float
    reason: Optional[str] = None


# --------- cross-encoder reranker ----------
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        model_name = getattr(settings, "reranker_model_name", None) or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder

def rerank_cross_encoder(query: str, candidates: list[dict[str, Any]]) -> list[Reranked]:
    """
    candidates: list of {"chunk_id":..., "content":..., "symbol":..., "signature":..., "heading_chain":...}
    returns scores where higher = more relevant.
    """
    ce = _get_cross_encoder()
    pairs = []
    ids = []
    for c in candidates:
        ids.append(int(c["chunk_id"]))
        text = _candidate_text_for_rerank(c)
        pairs.append((query, text))

    scores = ce.predict(pairs)  # float scores
    out = [Reranked(chunk_id=i, score=float(s)) for i, s in zip(ids, scores)]
    out.sort(key=lambda x: x.score, reverse=True)
    return out


# --------- ollama LLM judge reranker option  ----------
def rerank_ollama_judge(query: str, candidates: list[dict[str, Any]], *, model: str | None = None) -> list[Reranked]:
    """
    asks a Ollama to score each candidate 0..1; batched in one prompt for efficiency.
    """
    items = []
    for idx, c in enumerate(candidates, start=1):
        items.append(
            f"[{idx}] chunk_id={c['chunk_id']}\n"
            f"{_candidate_text_for_rerank(c)[:1200]}\n"
        )
    prompt = (
        "You are a strict relevance judge for technical onboarding QA.\n"
        "Given a USER QUESTION and several CONTEXT SNIPPETS, score each snippet from 0 to 10.\n"
        "10 = directly answers the question with key details. 0 = unrelated.\n"
        "Return ONLY valid JSON as a list of objects: "
        '[{"chunk_id":123,"score":7,"reason":"..."}].\n\n'
        f"USER QUESTION:\n{query}\n\n"
        f"CONTEXT SNIPPETS:\n" + "\n---\n".join(items)
    )
    raw = ollama_generate(prompt, model=model, temperature=0.0, max_tokens=800)
    # robustish JSON parse
    import json
    try:
        data = json.loads(raw)
        out = []
        for obj in data:
            out.append(Reranked(chunk_id=int(obj["chunk_id"]), score=float(obj["score"]), reason=obj.get("reason")))
        out.sort(key=lambda x: x.score, reverse=True)
        return out
    except Exception:
        # if parsing fails, return in original order with neutral scores
        return [Reranked(chunk_id=int(c["chunk_id"]), score=0.0) for c in candidates]


def _candidate_text_for_rerank(c: dict[str, Any]) -> str:
    parts = []
    if c.get("source_type"):
        parts.append(f"SOURCE_TYPE: {c['source_type']}")
    if c.get("symbol"):
        parts.append(f"SYMBOL: {c['symbol']}")
    if c.get("signature"):
        parts.append(f"SIGNATURE: {c['signature']}")
    hc = c.get("heading_chain")
    if hc:
        parts.append(f"HEADING: {' > '.join(hc)}")
    parts.append("CONTENT:\n" + (c.get("content") or ""))
    return "\n".join(parts)