CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------- enums ----------
DO $$ BEGIN
  CREATE TYPE source_type AS ENUM ('docs', 'code');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE chunk_type AS ENUM (
    -- docs
    'section', 'table', 'list', 'faq', 'runbook_step_block',
    -- code
    'function', 'class', 'module', 'block'
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------- sources ----------
CREATE TABLE IF NOT EXISTS sources (
  id            BIGSERIAL PRIMARY KEY,
  source_type   source_type NOT NULL,

  -- code fields
  repo          TEXT,
  branch        TEXT,
  path          TEXT,
  language      TEXT,

  -- docs fields
  doc_space     TEXT,
  page_id       TEXT,
  url           TEXT,
  updated_at    TIMESTAMPTZ,

  -- common fields
  title         TEXT,
  external_id   TEXT NOT NULL,          -- stable key per doc page / file path + repo/branch, etc.
  content_hash  TEXT NOT NULL,          -- hash of normalized source content for dedupe
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT sources_external_id_unique UNIQUE (external_id)
);

CREATE INDEX IF NOT EXISTS sources_type_idx ON sources(source_type);
CREATE INDEX IF NOT EXISTS sources_repo_branch_idx ON sources(repo, branch);
CREATE INDEX IF NOT EXISTS sources_path_idx ON sources(path);
CREATE INDEX IF NOT EXISTS sources_doc_space_idx ON sources(doc_space);
CREATE INDEX IF NOT EXISTS sources_meta_gin_idx ON sources USING GIN (meta);

-- ---------- chunks ----------
CREATE TABLE IF NOT EXISTS chunks (
  id            BIGSERIAL PRIMARY KEY,
  source_id     BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,

  chunk_type    chunk_type NOT NULL,

  -- code-only fields
  symbol        TEXT,         -- e.g., verify_token
  signature     TEXT,         -- e.g., verify_token(token: str) -> Claims

  -- docs-only fields
  heading_chain TEXT[],       -- e.g., {"Runbooks","Deploy","Rollback"}

  -- content shown/cited
  content       TEXT NOT NULL,

  -- embeddings
  embedding     vector(384), -- all-MiniLM-L6-v2 model is 384

  -- keyword search (full text)
  tsv           tsvector,

  -- ordering + dedupe
  position      INT NOT NULL DEFAULT 0,
  content_hash  TEXT NOT NULL,

  -- flexible metadata
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- dedup chunks within a source by hash (prevents re-inserting same chunk)
CREATE UNIQUE INDEX IF NOT EXISTS chunks_source_hash_unique
  ON chunks(source_id, content_hash);

-- vector index (cosine distance)
CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat_idx
  ON chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100); -- NOTE: adjust lists based on scale (start with 100)

-- full-text index
CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx
  ON chunks USING GIN (tsv);

-- filters
CREATE INDEX IF NOT EXISTS chunks_type_idx ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS chunks_symbol_trgm_idx ON chunks USING GIN (symbol gin_trgm_ops);
CREATE INDEX IF NOT EXISTS chunks_meta_gin_idx ON chunks USING GIN (meta);

-- ---------- chunk edges ----------
CREATE TABLE IF NOT EXISTS chunk_edges (
  id            BIGSERIAL PRIMARY KEY,
  from_chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  to_chunk_id   BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  edge_type     TEXT NOT NULL, -- e.g. "imports", "calls", "doc_links", "same_symbol"
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS chunk_edges_from_idx ON chunk_edges(from_chunk_id);
CREATE INDEX IF NOT EXISTS chunk_edges_to_idx ON chunk_edges(to_chunk_id);
CREATE INDEX IF NOT EXISTS chunk_edges_type_idx ON chunk_edges(edge_type);

-- ---------- trigger to auto-build tsvector ----------
CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
BEGIN
  NEW.tsv :=
      setweight(to_tsvector('english', coalesce(NEW.symbol, '')), 'A')
    || setweight(to_tsvector('english', coalesce(NEW.signature, '')), 'A')
    || setweight(to_tsvector('english', array_to_string(coalesce(NEW.heading_chain, '{}'), ' > ')), 'B')
    || setweight(to_tsvector('english', coalesce(NEW.content, '')), 'C');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks;
CREATE TRIGGER trg_chunks_tsv
BEFORE INSERT OR UPDATE OF symbol, signature, heading_chain, content
ON chunks
FOR EACH ROW
EXECUTE FUNCTION chunks_tsv_update();