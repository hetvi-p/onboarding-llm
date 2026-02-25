# Onboarding AI Assistant (RAG Chatbot)

A **local-first Retrieval Augmented Generation (RAG) chatbot** designed to answer onboarding and technical questions using internal documentation and code repositories.

This system ingests engineering documentation and source code, indexes them using semantic embeddings **(sentence-transformers)** + keyword search, and answers questions using a **locally hosted LLM (Ollama)** with grounded citations.

Currently implemented as a **backend API (FastAPI)**.

---

## Features

- Documentation ingestion (Markdown / text)
- Code repository ingestion (symbol-aware chunking)
- Hybrid search (Vector + Keyword / BM25)
- Code + Docs retrieval fusion (RRF ranking)
- Local embeddings (**SentenceTransformers**)
- Local LLM answering via **Ollama**
- Reranking for improved relevance
- Metadata filtering (repo, service, doc space, etc.)
- Incremental indexing (skips unchanged files)

---

# Setup

## 1. Install Dependencies

Create virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install packages:

```
pip install fastapi uvicorn psycopg sentence-transformers
```

---

## 2. Setup PostgreSQL + pgvector

Create database.

Run migration:

```
psql "$DATABASE_URL" -f migrations/001_init.sql
```

Required extensions:

* vector
* pg_trgm

---

## 3. Environment Variables

Example:

```
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db

EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

SUMMARIZER_MODE=heuristic
RERANK_MODE=cross_encoder

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

---

## 4. Install Ollama (Local LLM)

Install:

https://ollama.com

Run:

```
ollama serve
ollama pull llama3.1:8b
```

---

# Running the API

Start server:

```
uvicorn app.main:app --reload
```

Swagger UI available at:

```
http://localhost:8000/docs
```

---

# Ingestion

The system supports both single-file and bulk ingestion.

## Ingest Documentation Folder

```
POST /ingest/docs/folder
```

Example:

```json
{
  "folder": "data/docs",
  "doc_space": "Engineering Onboarding",
  "meta": {
    "team": "platform"
  }
}
```

---

## Ingest Code Repository

```
POST /ingest/code/repo
```

Example:

```json
{
  "repo_root": "repos/backend",
  "repo": "backend-api",
  "branch": "main",
  "meta": {
    "service": "auth"
  }
}
```

---

# Searching

Retrieve relevant chunks without generating an answer.

```
POST /search
```

Example:

```json
{
  "query": "Where is JWT verification implemented?"
}
```

Returns retrieved documentation and code snippets.

---

# Asking Questions

Main chatbot endpoint.

```
POST /answer
```

Example:

```json
{
  "query": "How do I deploy the backend service?"
}
```

Response includes:

* grounded answer
* citations
* retrieved sources

---

## Retrieval Strategy

The system combines multiple retrieval methods:

### Vector Search

Semantic similarity using embeddings.

### Keyword Search

Postgres full-text search for identifiers and errors.

### Fusion

Reciprocal Rank Fusion (RRF).

### Reranking

Cross-encoder improves final relevance.

---

## Metadata Filtering

Supported filters include:

* repository
* branch
* doc space
* service
* component
* owners

Example:

```
Only backend repo
Only runbooks
Only auth service
```

---

## Performance Notes

Recommended ingestion setup:

```
SUMMARIZER_MODE=heuristic
```

After bulk ingestion:

```
ANALYZE sources;
ANALYZE chunks;
```

---


