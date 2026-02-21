"""Link candidate persistence and enrichment.

Provides two public functions for the orchestration layer:

  enrich_link_candidates(source_document_id, source_url, conn, ...)
      Calls the Firecrawl /map endpoint and updates link_candidates rows
      with title/description metadata.  Idempotent — skips if already
      enriched.

  get_link_candidates(source_document_id, conn, ...)
      Fetches persisted link candidates for a source document with
      optional filtering (enriched-only, URL exclusion).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from psycopg import Connection

from config import settings
from src.ingestion.service import LinkCandidate, discover_links
from src.retrieval.models import PersistedLinkCandidate


def enrich_link_candidates(
    source_document_id: UUID,
    source_url: str,
    conn: Connection,
    limit: int = settings.ingest_discover_links_default_limit,
) -> int:
    """Enrich link_candidates rows with /map metadata for a source document.

    Calls discover_links() (Firecrawl /map endpoint) to get title and
    description for outgoing links, then UPDATEs matching rows in the
    link_candidates table.  Any /map URLs not already in the table are
    INSERTed as new fully-enriched rows (the /map endpoint may discover
    URLs not found on the specific scraped page).

    Idempotent: if any rows for this source_document_id already have
    enriched_at set, assumes enrichment was already done and returns 0
    without making an API call.

    Args:
        source_document_id: UUID of the document whose links to enrich.
        source_url: The page URL to pass to discover_links().
        conn: A psycopg Connection (caller manages lifecycle).
        limit: Max links to request from the /map endpoint.

    Returns:
        Number of rows enriched (updated + newly inserted).
    """
    # Check if enrichment already happened for this document.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM link_candidates
                WHERE source_document_id = %s
                  AND enriched_at IS NOT NULL
            )
            """,
            (source_document_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            return 0

    # Call Firecrawl /map endpoint.
    candidates: list[LinkCandidate] = discover_links(
        source_url, limit=limit
    )

    if not candidates:
        return 0

    enriched_count = 0
    now = datetime.now(timezone.utc)

    with conn.cursor() as cur:
        for candidate in candidates:
            # Try to UPDATE an existing row first (URL discovered during scraping).
            cur.execute(
                """
                UPDATE link_candidates
                SET title = %s,
                    description = %s,
                    enriched_at = %s
                WHERE source_document_id = %s
                  AND target_url = %s
                  AND enriched_at IS NULL
                """,
                (
                    candidate.title,
                    candidate.description,
                    now,
                    source_document_id,
                    candidate.url,
                ),
            )
            if cur.rowcount > 0:
                enriched_count += cur.rowcount
            else:
                # URL from /map not already in table — insert as fully enriched.
                # Use ON CONFLICT DO NOTHING in case of a race or if the row
                # was enriched between the UPDATE and this INSERT.
                cur.execute(
                    """
                    INSERT INTO link_candidates (
                        source_document_id, source_url, target_url,
                        title, description, enriched_at, depth
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, (
                        SELECT depth FROM documents WHERE id = %s
                    ))
                    ON CONFLICT (source_document_id, target_url) DO NOTHING
                    """,
                    (
                        source_document_id,
                        source_url,
                        candidate.url,
                        candidate.title,
                        candidate.description,
                        now,
                        source_document_id,
                    ),
                )
                enriched_count += cur.rowcount

    conn.commit()
    return enriched_count


def get_link_candidates(
    source_document_id: UUID,
    conn: Connection,
    *,
    enriched_only: bool = False,
    exclude_urls: set[str] | None = None,
) -> list[PersistedLinkCandidate]:
    """Fetch link candidates for a source document.

    Args:
        source_document_id: UUID of the document to get links for.
        conn: A psycopg Connection (caller manages lifecycle).
        enriched_only: If True, only return rows that have been enriched
            (enriched_at IS NOT NULL).
        exclude_urls: Set of target_url values to exclude (e.g. already
            visited URLs).

    Returns:
        List of PersistedLinkCandidate, ordered by discovered_at ASC.
    """
    clauses = ["source_document_id = %s"]
    params: list[object] = [source_document_id]

    if enriched_only:
        clauses.append("enriched_at IS NOT NULL")

    if exclude_urls:
        clauses.append("target_url != ALL(%s)")
        params.append(list(exclude_urls))

    where = " AND ".join(clauses)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                id, source_document_id, source_url, target_url,
                title, description, discovered_at, enriched_at, depth
            FROM link_candidates
            WHERE {where}
            ORDER BY discovered_at ASC
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        PersistedLinkCandidate(
            id=row[0],
            source_document_id=row[1],
            source_url=row[2],
            target_url=row[3],
            title=row[4],
            description=row[5],
            discovered_at=row[6],
            enriched_at=row[7],
            depth=row[8],
        )
        for row in rows
    ]
