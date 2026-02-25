# Engineering Onboarding Guide

Welcome to the engineering team. This document provides everything required to
understand the system architecture, development workflow, and operational setup.

---

# 1. System Overview

## What This System Does

This platform provides an AI-powered onboarding assistant capable of answering
technical and operational questions by searching internal documentation and
code repositories.

Core capabilities:

- Documentation ingestion
- Code repository indexing
- Semantic search
- Retrieval-Augmented Generation (RAG)
- Source-grounded answers

---

## High-Level Architecture

Main components:

1. Ingestion Service
2. Vector Database
3. Retrieval Engine
4. LLM Answer Service
5. API Layer
6. Frontend Interface

System Flow:

User Question → Retriever → Context Fusion → LLM → Answer + Citations

---

# 2. Technology Stack

## Backend

- Python
- FastAPI
- PostgreSQL
- pgvector

## AI / ML

- Sentence Transformers (local embeddings)
- Local LLM (Ollama)
- Hybrid Retrieval (Vector + Keyword)

## Infrastructure

- Docker
- Docker Compose
- GitHub Actions

---

# 3. Repository Structure
project/
├── api/
├── ingestion/
├── retrieval/
├── llm/
├── database/
├── data/
│ └── docs/
└── docker/


Important folders:

### ingestion/

Responsible for:

- document parsing
- chunking
- embedding generation
- metadata extraction

### retrieval/

Handles:

- semantic search
- ranking
- result fusion

### llm/

Responsible for:

- prompt construction
- grounding responses
- citation formatting

---

# 4. Local Development Setup

## Requirements

Install:

- Docker
- Python 3.11+
- PostgreSQL

---

## Setup Steps

Clone repository:
```git clone <repo>
cd project
```

Start services:
`docker compose up --build`

Run API:
`uvicorn api.main:app --reload`


---

# 5. Ingestion Pipeline

Documents are processed using the following steps:

1. Load document
2. Split into chunks
3. Generate embeddings
4. Store metadata
5. Insert into vector database

Chunk metadata includes:

- source file
- heading hierarchy
- repository
- document type

---

## Supported Sources

Currently supported:

- Markdown
- Engineering docs
- Runbooks
- Internal guides
- Code repositories

Future:

- Notion
- Confluence
- GitHub ingestion

---

# 6. Query Flow

When a user asks a question:

1. Query embedding generated
2. Retrieve documentation chunks
3. Retrieve code chunks
4. Fuse results
5. Send grounded context to LLM
6. Generate answer

---

# 7. Common Developer Tasks

## Re-ingest Documentation
`python ingestion/run_ingestion.py`

---

## Reset Database
`docker compose down -v`

---

## View Stored Chunks
`SELECT * FROM chunks LIMIT 20;`

---

# 8. Troubleshooting

## Embeddings Not Generated

Check:

- embedding model downloaded
- database connection
- ingestion logs

---

## Poor Answers

Possible causes:

- chunk size too large
- missing metadata
- outdated ingestion

Re-run ingestion.

---

# 9. Coding Guidelines

Follow:

- modular services
- typed Python
- clear docstrings
- small functions

Every public function should include:

- purpose
- inputs
- outputs

---

# 10. Operational Concepts

Important terms:

### Chunk

Small section of documentation indexed for retrieval.

### Embedding

Vector representation of text meaning.

### Retrieval

Finding relevant chunks for a question.

### Grounding

Restricting LLM answers to retrieved sources.

---

# 11. Security Notes

Do not ingest:

- secrets
- API keys
- credentials

Sensitive files should be excluded during ingestion.

---

# 12. Contacts

Engineering Owner:
engineering@example.com

Infrastructure:
infra@example.com

---

# 13. First Week Checklist

New engineers should:

- Setup local environment
- Run ingestion once
- Ask chatbot system questions
- Review architecture docs
- Deploy locally

Recommended questions:

- "How does ingestion work?"
- "Where are embeddings stored?"
- "How does retrieval ranking work?"

---

End of onboarding guide.





