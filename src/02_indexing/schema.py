# schema.py is just SQL wrapped in Python

from __future__ import annotations

from psycopg import Connection

from config import settings

# potential issue: % settings.embedding_dimensions string interpolation for the vector dimension. This works but is 
# technically string formatting into a SQL statement, which is the pattern SQL injection comes from. 
# It's safe here because embedding_dimensions is an integer from config, not user input

# runs a series of CREATE TABLES / INDEX IF NOT EXIST against postgres database
# IF NOT ensures repeated calls are safe

# documents - one row per ingested page. stores metadata from NormalizedDocument
# chunks - one row per chunk; both parents and children. stores actual text, rich content flags, 
# char/token offsets, embedding vector, and denormalized fields from documents so retrieval doesn't have to do joins

# indexes make queries fast; prevents scan of every row
# HNSW index is for the vector similarity search - graph based data stucture to allow pgvector to find nearest embedding vectors
def init_schema(conn: Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                url TEXT NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT NULL,
                description TEXT NULL,
                language TEXT NULL,
                status_code INTEGER NULL,
                published_time TEXT NULL,
                modified_time TEXT NULL,
                doc_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL,
                indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                depth INTEGER NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS documents_source_url_key
            ON documents (source_url);
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_id UUID NULL REFERENCES chunks(id) ON DELETE CASCADE,
                chunk_level TEXT NOT NULL CHECK (chunk_level IN ('parent', 'child')),
                chunk_index INTEGER NOT NULL,
                section_heading TEXT NULL,
                chunk_text TEXT NOT NULL,
                html_text TEXT NULL,
                has_table BOOLEAN NOT NULL DEFAULT FALSE,
                has_code BOOLEAN NOT NULL DEFAULT FALSE,
                has_math BOOLEAN NOT NULL DEFAULT FALSE,
                has_definition_list BOOLEAN NOT NULL DEFAULT FALSE,
                has_admonition BOOLEAN NOT NULL DEFAULT FALSE,
                has_steps BOOLEAN NOT NULL DEFAULT FALSE,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                token_start INTEGER NOT NULL,
                token_end INTEGER NOT NULL,
                embedding VECTOR(%d) NULL,
                source_url TEXT NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL,
                depth INTEGER NOT NULL,
                title TEXT NULL
            );
            """
            % settings.embedding_dimensions
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_document_id_idx
            ON chunks (document_id);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_parent_id_idx
            ON chunks (parent_id);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_chunk_level_idx
            ON chunks (chunk_level);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_source_url_idx
            ON chunks (source_url);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WHERE chunk_level = 'child' AND embedding IS NOT NULL;
            """
        )

        # ── link_candidates ─────────────────────────────────────
        # Stores outgoing links found on ingested pages.  Rows are
        # inserted URL-only at index time (title/description NULL),
        # then enriched via the Firecrawl /map endpoint on-demand
        # when orchestration needs link scoring metadata.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS link_candidates (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                source_url TEXT NOT NULL,
                target_url TEXT NOT NULL,
                title TEXT NULL,
                description TEXT NULL,
                discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                enriched_at TIMESTAMPTZ NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                UNIQUE (source_document_id, target_url)
            );
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_link_candidates_source
            ON link_candidates (source_document_id);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_link_candidates_target
            ON link_candidates (target_url);
            """
        )

    conn.commit()
