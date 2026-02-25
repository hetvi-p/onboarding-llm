"""
onboarding_demo_service.py

Purpose:
- A realistic, chunk-friendly demo file for testing your ingestion pipeline.
- Designed to link directly to /docs/onboarding.md so retrieval can fuse code + docs.

How to use:
- Put this file in your repo (or a test repo) and ingest both:
  - /docs/onboarding.md
  - this file
- Then ask questions like:
  - "How does ingestion work in code?"
  - "Where do we store chunks?"
  - "How do docs + code get fused?"
  - "Show the ingestion pipeline steps and where they happen in code."

Chunking note:
- This file is structured by "symbols" (classes/functions) with clear docstrings.
- If your chunker uses AST parsing, each function/class should become a chunk.
- If it uses regex or simple heuristics, the "SECTION" headers help.

Security note:
- No real secrets, only example env var names.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import os


# =========================
# SECTION: Docs Cross-Refs
# =========================

class DocsRef(str, Enum):
    """
    Canonical references into /docs/onboarding.md.

    Your chatbot can answer with:
    - "See onboarding.md → 5. Ingestion Pipeline"
    - "See onboarding.md → 6. Query Flow"
    """
    SYSTEM_OVERVIEW = "onboarding.md#1-system-overview"
    ARCHITECTURE = "onboarding.md#2-high-level-architecture"
    REPO_STRUCTURE = "onboarding.md#3-repository-structure"
    LOCAL_DEV = "onboarding.md#4-local-development-setup"
    INGESTION_PIPELINE = "onboarding.md#5-ingestion-pipeline"
    QUERY_FLOW = "onboarding.md#6-query-flow"
    TROUBLESHOOTING = "onboarding.md#8-troubleshooting"
    SECURITY = "onboarding.md#11-security-notes"


# =========================
# SECTION: Core Data Models
# =========================

@dataclass(frozen=True)
class SourceDocument:
    """
    Represents a document to ingest.

    Maps to onboarding.md:
    - "Supported Sources" (onboarding.md → 5. Ingestion Pipeline)
    """
    source_id: str                 # stable id (path/url/etc.)
    source_type: str               # "markdown" | "code" | "pdf" | ...
    title: str
    text: str
    uri: Optional[str] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Chunk:
    """
    Represents a chunk produced from a SourceDocument.

    Maps to onboarding.md:
    - "Chunk" definition (onboarding.md → 10. Operational Concepts)
    - "Chunk metadata includes ..." (onboarding.md → 5. Ingestion Pipeline)
    """
    chunk_id: str
    source_id: str
    chunk_index: int
    content: str
    # These fields are intentionally RAG-friendly:
    heading_path: Optional[List[str]] = None     # docs heading stack
    symbol_path: Optional[str] = None            # code symbol like "Class.method"
    language: Optional[str] = None               # "python"
    file_path: Optional[str] = None              # path in repo
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    Represents a chunk + its embedding vector.

    Maps to onboarding.md:
    - "Embedding" definition (onboarding.md → 10. Operational Concepts)
    """
    chunk: Chunk
    embedding: List[float]


# =========================
# SECTION: Utilities
# =========================

def sha256_text(s: str) -> str:
    """
    Compute a stable hash for IDs and deduping.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    """
    Return timezone-aware current time.
    """
    return datetime.now(timezone.utc)


# ==================================
# SECTION: Chunkers (Docs + Code)
# ==================================

class SimpleDocChunker:
    """
    Minimal markdown chunker for testing:
    - splits on headings (#, ##, ###)
    - builds a heading_path stack for metadata

    Maps to onboarding.md:
    - "Chunk by heading sections" (onboarding.md → 1) & (onboarding.md → 5)

    NOTE:
    Your real system probably has a better chunker. This is test-friendly.
    """
    def __init__(self, max_chars: int = 900, overlap: int = 150) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        lines = doc.text.splitlines()
        chunks: List[Chunk] = []

        heading_stack: List[str] = []
        buffer: List[str] = []
        buffer_heading_path: List[str] = []

        def flush_buffer() -> None:
            nonlocal buffer, buffer_heading_path
            text_block = "\n".join(buffer).strip()
            if not text_block:
                buffer = []
                return

            # character windowing inside each section
            start = 0
            idx = len(chunks)
            while start < len(text_block):
                end = min(len(text_block), start + self.max_chars)
                slice_ = text_block[start:end].strip()
                if slice_:
                    chunk_id = sha256_text(f"{doc.source_id}:{idx}:{slice_[:40]}")
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            source_id=doc.source_id,
                            chunk_index=idx,
                            content=slice_,
                            heading_path=list(buffer_heading_path),
                            language=None,
                            file_path=doc.uri or doc.source_id,
                            metadata={
                                "doc_ref": DocsRef.INGESTION_PIPELINE.value,
                                "source_type": doc.source_type,
                            },
                        )
                    )
                    idx += 1
                start = end - self.overlap if end < len(text_block) else end

            buffer = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                # new section: flush previous
                flush_buffer()
                # update heading stack
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped.lstrip("#").strip()
                # maintain stack by heading level
                heading_stack = heading_stack[: max(0, level - 1)]
                heading_stack.append(title)
                buffer_heading_path = list(heading_stack)
                buffer.append(line)
            else:
                buffer.append(line)

        flush_buffer()
        return chunks


class CodeSymbolChunker:
    """
    Pretend "symbol chunker" for Python.

    In a real system you'd use `ast` to chunk:
    - classes
    - functions
    - methods

    Here we keep it simple and very test-friendly:
    - chunk by 'class ' and 'def ' boundaries
    - attach symbol_path + file_path metadata

    Maps to onboarding.md:
    - "Chunk by symbol" (onboarding.md → 1) & (onboarding.md → 5)
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        text = doc.text
        # A lightweight boundary finder
        boundaries: List[int] = [0]
        for token in ("\nclass ", "\ndef "):
            i = 0
            while True:
                i = text.find(token, i + 1)
                if i == -1:
                    break
                boundaries.append(i + 1)  # keep newline alignment

        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))

        chunks: List[Chunk] = []
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            block = text[start:end].strip()
            if not block:
                continue

            # naive symbol name extraction
            symbol_path = self._extract_symbol_path(block)
            chunk_id = sha256_text(f"{doc.source_id}:{idx}:{symbol_path or 'block'}")

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=doc.source_id,
                    chunk_index=len(chunks),
                    content=block,
                    heading_path=None,
                    symbol_path=symbol_path,
                    language="python",
                    file_path=self.file_path,
                    metadata={
                        "doc_ref": DocsRef.QUERY_FLOW.value,
                        "source_type": doc.source_type,
                    },
                )
            )

        return chunks

    def _extract_symbol_path(self, block: str) -> Optional[str]:
        first_line = block.splitlines()[0].strip()
        if first_line.startswith("class "):
            name = first_line.replace("class ", "").split("(")[0].split(":")[0].strip()
            return name
        if first_line.startswith("def "):
            name = first_line.replace("def ", "").split("(")[0].split(":")[0].strip()
            return name
        return None


# ==================================
# SECTION: Embedding + Storage APIs
# ==================================

class EmbeddingModel:
    """
    Abstract embedding model interface.

    Your ingestion mechanism likely wraps sentence-transformers,
    OpenAI embeddings, or another local model.

    Maps to onboarding.md:
    - "Embeddings" + "Vector database" (onboarding.md → 2, 5, 10)
    """
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class FakeEmbeddingModel(EmbeddingModel):
    """
    Deterministic fake embeddings for testing end-to-end ingestion:
    - no ML dependencies
    - stable vectors so you can debug storage & retrieval

    Replace with SentenceTransformer in real use.
    """
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for t in texts:
            h = sha256_text(t)
            # turn hash into dim floats in [0, 1)
            vals = []
            for i in range(self.dim):
                chunk = h[i * 8 : i * 8 + 8]
                vals.append(int(chunk, 16) / 0xFFFFFFFF)
            vectors.append(vals)
        return vectors


class ChunkStore:
    """
    Abstract chunk store API.

    In your system, this is likely Postgres + pgvector.
    This interface mirrors what you probably already have:
    - upsert chunks
    - upsert embeddings

    Maps to onboarding.md:
    - "Store metadata" / "Insert into vector database" (onboarding.md → 5)
    """
    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError

    def upsert_embeddings(self, embedded: List[EmbeddedChunk]) -> None:
        raise NotImplementedError


class InMemoryChunkStore(ChunkStore):
    """
    In-memory store for smoke-testing your ingestion flow without a DB.
    Useful for validating that chunk IDs and metadata look correct.
    """
    def __init__(self) -> None:
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c

    def upsert_embeddings(self, embedded: List[EmbeddedChunk]) -> None:
        for e in embedded:
            self.embeddings[e.chunk.chunk_id] = e.embedding


# =========================
# SECTION: Ingestion Service
# =========================

class IngestionService:
    """
    Coordinates ingestion end-to-end:

    1) Load document
    2) Split into chunks
    3) Generate embeddings
    4) Store metadata
    5) Insert into vector database

    Exactly matches onboarding.md → "Ingestion Pipeline".

    This is intentionally simple so your chatbot can reference it as
    "the ingestion pipeline in code" during answers.
    """
    def __init__(self, store: ChunkStore, embedder: EmbeddingModel) -> None:
        self.store = store
        self.embedder = embedder

    def ingest_docs(self, docs: List[SourceDocument]) -> Tuple[int, int]:
        """
        Ingest a mixture of docs + code documents.
        Returns (num_chunks, num_embeddings).

        Ask your bot:
        - "What does ingest_docs do?"
        - "Where are chunk IDs created?"
        """
        all_chunks: List[Chunk] = []
        for doc in docs:
            all_chunks.extend(self._chunk_one(doc))

        self.store.upsert_chunks(all_chunks)

        vectors = self.embedder.embed([c.content for c in all_chunks])
        embedded = [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(all_chunks, vectors)]
        self.store.upsert_embeddings(embedded)

        return (len(all_chunks), len(embedded))

    def _chunk_one(self, doc: SourceDocument) -> List[Chunk]:
        """
        Route to the right chunker based on doc type / extension.
        """
        if doc.source_type == "markdown":
            return SimpleDocChunker().chunk(doc)

        if doc.source_type == "code" and (doc.uri or doc.source_id).endswith(".py"):
            # treat python as symbol-chunked
            file_path = doc.uri or doc.source_id
            return CodeSymbolChunker(file_path=file_path).chunk(doc)

        # fallback: one chunk
        chunk_id = sha256_text(f"{doc.source_id}:fallback:{doc.text[:40]}")
        return [
            Chunk(
                chunk_id=chunk_id,
                source_id=doc.source_id,
                chunk_index=0,
                content=doc.text.strip(),
                file_path=doc.uri or doc.source_id,
                metadata={"doc_ref": DocsRef.INGESTION_PIPELINE.value},
            )
        ]


# =========================
# SECTION: Demo Fixtures
# =========================

def load_demo_docs() -> List[SourceDocument]:
    """
    Build two documents:
    1) a mini excerpt of onboarding.md (docs)
    2) this file itself (code)

    In your real pipeline, you'd load from disk.
    This is for testing ingestion wiring.

    Your chatbot can then "piece them together":
    - Docs: the conceptual 5-step pipeline
    - Code: IngestionService implements the exact 5 steps
    """
    onboarding_excerpt = """# Ingestion Pipeline

Documents are processed using the following steps:

1. Load document
2. Split into chunks
3. Generate embeddings
4. Store metadata
5. Insert into vector database

## Chunk metadata includes

- source file
- heading hierarchy
- repository
- document type
"""

    this_file_path = os.path.abspath(__file__)
    try:
        with open(this_file_path, "r", encoding="utf-8") as f:
            this_file_text = f.read()
    except Exception:
        # If __file__ isn't available in your environment, we still provide a placeholder
        this_file_text = "# Could not load self; paste file content for real ingestion."

    return [
        SourceDocument(
            source_id="docs/onboarding.md",
            source_type="markdown",
            title="Engineering Onboarding Guide",
            text=onboarding_excerpt,
            uri="docs/onboarding.md",
            updated_at=utcnow(),
            metadata={"doc_ref": DocsRef.INGESTION_PIPELINE.value},
        ),
        SourceDocument(
            source_id="ingestion/demo_repo/onboarding_demo_service.py",
            source_type="code",
            title="Onboarding Demo Service",
            text=this_file_text,
            uri="ingestion/demo_repo/onboarding_demo_service.py",
            updated_at=utcnow(),
            metadata={"doc_ref": DocsRef.QUERY_FLOW.value},
        ),
    ]


def run_smoke_test() -> None:
    """
    Run a minimal ingestion smoke test without a database.
    """
    store = InMemoryChunkStore()
    embedder = FakeEmbeddingModel(dim=8)
    service = IngestionService(store=store, embedder=embedder)

    docs = load_demo_docs()
    n_chunks, n_embeds = service.ingest_docs(docs)

    print("Ingestion completed.")
    print(f"- chunks stored: {n_chunks}")
    print(f"- embeddings stored: {n_embeds}")

    # Print a few chunks to verify metadata looks correct
    sample = list(store.chunks.values())[:5]
    for i, c in enumerate(sample, start=1):
        print("\n---")
        print(f"Chunk #{i}")
        print(f"chunk_id: {c.chunk_id}")
        print(f"source_id: {c.source_id}")
        print(f"file_path: {c.file_path}")
        print(f"heading_path: {c.heading_path}")
        print(f"symbol_path: {c.symbol_path}")
        print(f"metadata: {c.metadata}")
        print("content preview:")
        print(c.content[:200])


if __name__ == "__main__":
    run_smoke_test()