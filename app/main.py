from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Literal


from .db import get_conn
from .ingest.docs_ingest import ingest_markdown_file
from .ingest.code_ingest import ingest_code_file
from .retrieval.search import SearchFilters, search_hybrid_rrf
from .db import get_conn
from .answering.answer import answer_with_ollama, rerank_results
from .settings import settings


app = FastAPI(title="Onboarding RAG - Ingest v2")

class IngestDocsRequest(BaseModel):
    path: str
    doc_space: str | None = None
    meta: dict | None = None
    force_reindex: bool = False

class IngestCodeRequest(BaseModel):
    file_path: str
    repo: str
    branch: str = "main"
    meta: dict | None = None
    force_reindex: bool = False


class SearchRequest(BaseModel):
    query: str

    # retrieval params
    top_k_vec: int = 30
    top_k_kw: int = 30
    fused_top_k: int = 12

    # filters
    source_type: Literal["docs", "code", "all"] = "all"
    repo: Optional[str] = None
    branch: Optional[str] = None
    path_prefix: Optional[str] = None
    doc_space: Optional[str] = None

    service: Optional[str] = None
    component: Optional[str] = None
    owners: Optional[str] = None

    updated_after_iso: Optional[str] = None


class AnswerRequest(BaseModel):
    query: str

    # retrieval params
    top_k_vec: int = 30
    top_k_kw: int = 30
    fused_top_k: int = 18

    # rerank params
    rerank_top_n: int = 10
    rerank_mode: Optional[str] = None  # overrides env RERANK_MODE if set

    # answer model
    llm_model: Optional[str] = None  # defaults to OLLAMA_MODEL

    # filters (same as /search)
    source_type: Literal["docs", "code", "all"] = "all"
    repo: Optional[str] = None
    branch: Optional[str] = None
    path_prefix: Optional[str] = None
    doc_space: Optional[str] = None
    service: Optional[str] = None
    component: Optional[str] = None
    owners: Optional[str] = None
    updated_after_iso: Optional[str] = None



@app.post("/ingest/docs/markdown")
def ingest_docs(req: IngestDocsRequest):
    p = Path(req.path)
    with get_conn() as conn:
        out = ingest_markdown_file(
            conn,
            path=p,
            doc_space=req.doc_space,
            meta=req.meta,
            force_reindex=req.force_reindex,
        )
        conn.commit()
    return out

@app.post("/ingest/code/file")
def ingest_code(req: IngestCodeRequest):
    p = Path(req.file_path)
    with get_conn() as conn:
        out = ingest_code_file(
            conn,
            file_path=p,
            repo=req.repo,
            branch=req.branch,
            meta=req.meta,
            force_reindex=req.force_reindex,
        )
        conn.commit()
    return out



@app.post("/search")
def search(req: SearchRequest):
    filters = SearchFilters(
        source_type=req.source_type,
        repo=req.repo,
        branch=req.branch,
        path_prefix=req.path_prefix,
        doc_space=req.doc_space,
        service=req.service,
        component=req.component,
        owners=req.owners,
        updated_after_iso=req.updated_after_iso,
    )

    with get_conn() as conn:
        out = search_hybrid_rrf(
            conn,
            req.query,
            filters=filters,
            top_k_vec=req.top_k_vec,
            top_k_kw=req.top_k_kw,
            fused_top_k=req.fused_top_k,
        )
    return out



@app.post("/answer")
def answer(req: AnswerRequest):
    filters = SearchFilters(
        source_type=req.source_type,
        repo=req.repo,
        branch=req.branch,
        path_prefix=req.path_prefix,
        doc_space=req.doc_space,
        service=req.service,
        component=req.component,
        owners=req.owners,
        updated_after_iso=req.updated_after_iso,
    )

    with get_conn() as conn:
        retrieval = search_hybrid_rrf(
            conn,
            req.query,
            filters=filters,
            top_k_vec=req.top_k_vec,
            top_k_kw=req.top_k_kw,
            fused_top_k=req.fused_top_k,
        )

    retrieved = retrieval.get("results", [])
    mode = (req.rerank_mode or settings.rerank_mode).lower()

    reranked = rerank_results(
        req.query,
        retrieved,
        rerank_mode=mode,
        rerank_top_n=req.rerank_top_n,
    )

    answer_payload = answer_with_ollama(
        req.query,
        context_results=reranked,
        model=req.llm_model or settings.ollama_model,
    )

    return {
        "query": req.query,
        "signals": retrieval.get("signals"),
        "rewrites": retrieval.get("rewrites"),
        "rerank_mode": mode,
        "results": reranked,
        "answer": answer_payload["answer"],
    }
