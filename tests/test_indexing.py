from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import os
import time
from typing import Any

import pytest

from config import settings
from src.ingestion.service import NormalizedDocument, ingest
from src.indexing.indexer import index_batch
from src.indexing.schema import init_schema


SCIKIT_URL = "https://scikit-learn.org/stable/modules/ensemble.html#ensemble"
FACTUAL_URL = "https://docs.python.org/3/glossary.html"
UNRELATED_URL = os.getenv(
    "TEST_UNRELATED_URL",
    # Lighter unrelated default for faster integration runs.
    "https://simple.wikipedia.org/wiki/Mount_Everest",
)
RUN_SLOW_INTEGRATION = os.getenv("RUN_SLOW_INTEGRATION", "").strip() == "1"


def _source_url(doc: NormalizedDocument) -> str:
    return doc.source_url or doc.url


@pytest.fixture(scope="module")
def indexed_corpus(db_conn, reset_index_tables) -> dict[str, Any]:
    reset_index_tables()
    print("indexing fixture: reset tables complete")

    try:
        urls = [SCIKIT_URL, FACTUAL_URL, UNRELATED_URL]
        docs: list[NormalizedDocument] = []
        for url in urls:
            started = time.perf_counter()
            print(f"indexing fixture: ingesting {url}")
            # ingest() is now async — run in a fresh event loop.
            doc = asyncio.run(ingest(url))
            elapsed = time.perf_counter() - started
            print(f"indexing fixture: ingested {url} in {elapsed:.1f}s")
            docs.append(doc)
    except RuntimeError as exc:
        if "FIRECRAWL_API_KEY" in str(exc):
            pytest.skip(f"Skipping indexing integration tests: {exc}")
        raise

    depths = [0, 0, 2]
    started_index = time.perf_counter()
    print("indexing fixture: running index_batch")
    # index_batch() is now async — run in a fresh event loop.
    asyncio.run(index_batch(docs, depths))
    print(f"indexing fixture: index_batch finished in {time.perf_counter() - started_index:.1f}s")

    source_urls = [_source_url(doc) for doc in docs]

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents;")
        total_documents = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks;")
        total_chunks = cur.fetchone()[0]
        cur.execute(
            """
            SELECT COUNT(*) FROM chunks
            WHERE chunk_level = 'child' AND embedding IS NOT NULL
            """
        )
        embedded_children = cur.fetchone()[0]
        cur.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN has_code THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN has_table THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN has_math THEN 1 ELSE 0 END), 0)
            FROM chunks
            WHERE chunk_level = 'child'
            """
        )
        has_code_count, has_table_count, has_math_count = cur.fetchone()

    summary = {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "total_child_chunks_with_embeddings": embedded_children,
        "distinct_source_urls": source_urls,
        "flag_distribution": {
            "has_code": has_code_count,
            "has_table": has_table_count,
            "has_math": has_math_count,
        },
    }
    print("indexing summary:", summary)

    return {
        "docs": docs,
        "source_urls": source_urls,
        "depths": depths,
    }


def test_schema_validation(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                to_regclass('public.documents') IS NOT NULL,
                to_regclass('public.chunks') IS NOT NULL
            """
        )
        exists_row = cur.fetchone()
    if not (exists_row and exists_row[0] and exists_row[1]):
        init_schema(db_conn)

    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('documents', 'chunks')
            """
        )
        tables = {row[0] for row in cur.fetchall()}
        assert "documents" in tables
        assert "chunks" in tables

        cur.execute(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'chunks'
              AND indexname = 'chunks_embedding_hnsw_idx'
            """
        )
        row = cur.fetchone()
        assert row is not None
        assert "USING hnsw" in row[1]
        assert "vector_cosine_ops" in row[1]


@pytest.mark.usefixtures("indexed_corpus")
class TestIndexingIntegration:
    def test_document_rows_created(self, db_conn, indexed_corpus):
        for source_url in indexed_corpus["source_urls"]:
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM documents WHERE source_url = %s", (source_url,)
                )
                assert cur.fetchone()[0] == 1

    def test_chunk_counts_nonzero_and_sane(self, db_conn, indexed_corpus):
        scikit_source = indexed_corpus["source_urls"][0]
        for source_url in indexed_corpus["source_urls"]:
            with db_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        SUM(CASE WHEN chunk_level = 'child' THEN 1 ELSE 0 END) AS child_count,
                        SUM(CASE WHEN chunk_level = 'parent' THEN 1 ELSE 0 END) AS parent_count
                    FROM chunks
                    WHERE source_url = %s
                    """,
                    (source_url,),
                )
                child_count, parent_count = cur.fetchone()
                child_count = child_count or 0
                parent_count = parent_count or 0
                min_children = 5 if source_url == scikit_source else 1
                assert child_count >= min_children
                assert parent_count >= 1
                assert child_count > parent_count

    def test_parent_child_linkage_integrity(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunks child
                LEFT JOIN chunks parent
                  ON child.parent_id = parent.id
                 AND parent.chunk_level = 'parent'
                 AND parent.document_id = child.document_id
                WHERE child.chunk_level = 'child'
                  AND (child.parent_id IS NULL OR parent.id IS NULL)
                """
            )
            orphaned = cur.fetchone()[0]
        assert orphaned == 0

    def test_embeddings_populated_on_children_only(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM chunks
                WHERE chunk_level = 'child' AND embedding IS NULL
                """
            )
            assert cur.fetchone()[0] == 0

            cur.execute(
                """
                SELECT COUNT(*) FROM chunks
                WHERE chunk_level = 'parent' AND embedding IS NOT NULL
                """
            )
            assert cur.fetchone()[0] == 0

    def test_embedding_dimensionality(self, db_conn, indexed_corpus):
        scikit_source = indexed_corpus["source_urls"][0]
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT vector_dims(embedding)
                FROM chunks
                WHERE source_url = %s
                  AND chunk_level = 'child'
                  AND embedding IS NOT NULL
                LIMIT 10
                """,
                (scikit_source,),
            )
            dims = [row[0] for row in cur.fetchall()]
        assert len(dims) > 0
        assert all(dim == settings.embedding_dimensions for dim in dims)

    def test_rich_content_flags_on_scikit(self, db_conn, indexed_corpus):
        scikit_source = indexed_corpus["source_urls"][0]
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    SUM(CASE WHEN has_code THEN 1 ELSE 0 END),
                    SUM(CASE WHEN has_table THEN 1 ELSE 0 END)
                FROM chunks
                WHERE source_url = %s AND chunk_level = 'child'
                """,
                (scikit_source,),
            )
            code_count, table_count = cur.fetchone()
        assert (code_count or 0) > 0
        assert (table_count or 0) > 0

    def test_parent_rich_content_propagated_on_scikit(self, db_conn, indexed_corpus):
        scikit_source = indexed_corpus["source_urls"][0]
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE source_url = %s
                  AND chunk_level = 'parent'
                  AND (
                    has_table
                    OR has_code
                    OR has_math
                    OR has_definition_list
                    OR has_admonition
                  )
                """,
                (scikit_source,),
            )
            flagged_parent_count = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE source_url = %s
                  AND chunk_level = 'parent'
                  AND (
                    has_table
                    OR has_code
                    OR has_math
                    OR has_definition_list
                    OR has_admonition
                  )
                  AND html_text IS NULL
                """,
                (scikit_source,),
            )
            flagged_missing_html = cur.fetchone()[0]

        assert flagged_parent_count > 0
        assert flagged_missing_html == 0

    def test_depth_persisted_correctly(self, db_conn, indexed_corpus):
        scikit_source, factual_source, unrelated_source = indexed_corpus["source_urls"]
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE source_url = %s AND depth <> 2",
                (unrelated_source,),
            )
            assert cur.fetchone()[0] == 0

            cur.execute(
                """
                SELECT COUNT(*) FROM chunks
                WHERE source_url IN (%s, %s) AND depth <> 0
                """,
                (scikit_source, factual_source),
            )
            assert cur.fetchone()[0] == 0

    def test_denormalized_fields_populated(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.source_url, c.fetched_at, c.title, c.depth,
                       d.source_url, d.fetched_at, d.title, d.depth
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                LIMIT 25
                """
            )
            rows = cur.fetchall()

        assert len(rows) > 0
        for row in rows:
            c_source, c_fetched, c_title, c_depth, d_source, d_fetched, d_title, d_depth = row
            assert c_source is not None
            assert c_fetched is not None
            assert c_depth is not None
            assert c_source == d_source
            assert c_fetched == d_fetched
            assert c_title == d_title
            assert c_depth == d_depth

    def test_dedup_same_hash_noop(self, db_conn, indexed_corpus):
        scikit_doc = indexed_corpus["docs"][0]
        source_url = _source_url(scikit_doc)
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE source_url = %s", (source_url,))
            before = cur.fetchone()[0]

        # index_batch() is now async.
        asyncio.run(index_batch([scikit_doc], [0]))

        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE source_url = %s", (source_url,))
            after = cur.fetchone()[0]
        assert before == after

    @pytest.mark.skipif(
        not RUN_SLOW_INTEGRATION,
        reason="Set RUN_SLOW_INTEGRATION=1 to run expensive reindex mutation test.",
    )
    def test_dedup_changed_hash_triggers_reindex(self, db_conn, indexed_corpus):
        original = indexed_corpus["docs"][0]
        source_url = _source_url(original)

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT id::text, COUNT(*) FROM documents WHERE source_url = %s GROUP BY id",
                (source_url,),
            )
            old_doc_id, _ = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM chunks WHERE source_url = %s", (source_url,))
            old_chunk_count = cur.fetchone()[0]

        mutated_markdown = (original.markdown or "") + "\n\nIndexing test mutation sentence."
        mutated_hash = hashlib.sha256(mutated_markdown.encode("utf-8")).hexdigest()
        mutated_doc = replace(
            original,
            markdown=mutated_markdown,
            content_hash=mutated_hash,
        )

        # index_batch() is now async.
        asyncio.run(index_batch([mutated_doc], [0]))

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT id::text, COUNT(*) FROM documents WHERE source_url = %s GROUP BY id",
                (source_url,),
            )
            new_doc_id, row_count = cur.fetchone()
            assert row_count == 1
            assert new_doc_id != old_doc_id

            cur.execute("SELECT COUNT(*) FROM documents WHERE id = %s", (old_doc_id,))
            assert cur.fetchone()[0] == 0

            cur.execute("SELECT COUNT(*) FROM chunks WHERE source_url = %s", (source_url,))
            new_chunk_count = cur.fetchone()[0]

        assert new_chunk_count > 0
        assert new_chunk_count <= max(2 * old_chunk_count, 1)

        # restore original document state for downstream tests if needed
        # index_batch() is now async.
        asyncio.run(index_batch([original], [0]))

    def test_char_offsets_valid(self, db_conn, indexed_corpus):
        markdown_lengths = {
            _source_url(doc): len(doc.markdown or "") for doc in indexed_corpus["docs"]
        }
        with db_conn.cursor() as cur:
            cur.execute("SELECT source_url, char_start, char_end FROM chunks")
            rows = cur.fetchall()

        assert len(rows) > 0
        for source_url, char_start, char_end in rows:
            assert char_start >= 0
            assert char_end > char_start
            assert char_end <= markdown_lengths[source_url]

    def test_token_offsets_valid_and_non_overlapping_within_parent(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT token_start, token_end
                FROM chunks
                """
            )
            all_rows = cur.fetchall()

            cur.execute(
                """
                SELECT parent_id, token_start, token_end
                FROM chunks
                WHERE chunk_level = 'child'
                ORDER BY parent_id, token_start, token_end
                """
            )
            child_rows = cur.fetchall()

        assert len(all_rows) > 0
        for token_start, token_end in all_rows:
            assert token_start >= 0
            assert token_end >= token_start

        current_parent = None
        previous_end = None
        for parent_id, token_start, token_end in child_rows:
            if parent_id != current_parent:
                current_parent = parent_id
                previous_end = None
            if previous_end is not None:
                assert token_start >= previous_end
            previous_end = token_end

    def test_section_headings_present_on_parents(self, db_conn, indexed_corpus):
        scikit_source = indexed_corpus["source_urls"][0]
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_parents,
                    SUM(
                        CASE
                            WHEN section_heading IS NOT NULL
                             AND btrim(section_heading) <> ''
                            THEN 1 ELSE 0
                        END
                    ) AS headed_parents
                FROM chunks
                WHERE source_url = %s AND chunk_level = 'parent'
                """,
                (scikit_source,),
            )
            total_parents, headed_parents = cur.fetchone()

        assert total_parents > 0
        ratio = (headed_parents or 0) / total_parents
        assert ratio >= 0.5

    def test_empty_batch_noop(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            docs_before = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunks_before = cur.fetchone()[0]

        # index_batch() is now async.
        asyncio.run(index_batch([], []))

        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            docs_after = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunks_after = cur.fetchone()[0]

        assert docs_before == docs_after
        assert chunks_before == chunks_after
