"""Link scoring and expansion execution.

Handles candidate discovery, scoring, selection, scraping, and indexing
to grow the corpus during the orchestration loop.
"""

from __future__ import annotations

import asyncio
import re
import warnings
from datetime import datetime, timezone
from urllib.parse import urlparse
from uuid import UUID, uuid4

from psycopg import AsyncConnection

from config import settings
from src.ingestion.links import enrich_link_candidates, get_link_candidates
from src.ingestion.service import NormalizedDocument, ingest
from src.indexing.indexer import index_batch
from src.orchestration.models import (
    ExpansionOutcome,
    QueryAnalysis,
    ScoredCandidate,
)
from src.retrieval.models import PersistedLinkCandidate

# ── Scoring weights ───────────────────────────────────────────────
# These weights control candidate link ranking during expansion.
# Title and description carry the most signal because they directly
# describe the page content.  URL path helps for docs sites with
# structured paths.  In-degree and depth are weaker structural signals.

_W_URL_PATH = 0.15
_W_TITLE = 0.40
_W_DESCRIPTION = 0.30
_W_IN_DEGREE = 0.05
_W_DEPTH = 0.10

# ── Public interface ──────────────────────────────────────────────


async def score_candidates(
    candidates: list[PersistedLinkCandidate],
    query: str,
    query_analysis: QueryAnalysis,
    already_ingested_urls: set[str],
    *,
    in_degree_map: dict[str, int] | None = None,
) -> list[ScoredCandidate]:
    """Score link candidates for expansion relevance.

    Returns candidates sorted by score descending, excluding any URL
    already in *already_ingested_urls*.
    """
    key_concepts = set(query_analysis.key_concepts)
    query_tokens = _tokenize(query)

    scored: list[ScoredCandidate] = []
    for cand in candidates:
        if cand.target_url in already_ingested_urls:
            continue

        url_path_rel = _url_path_relevance(cand.target_url, key_concepts)
        title_rel = _text_relevance(cand.title, query_tokens) if cand.title else 0.0
        desc_rel = _text_relevance(cand.description, query_tokens) if cand.description else 0.0
        in_deg = _in_degree_signal(
            cand.target_url, in_degree_map or {},
        )
        depth_fresh = _depth_freshness(cand.depth)

        total = (
            _W_URL_PATH * url_path_rel
            + _W_TITLE * title_rel
            + _W_DESCRIPTION * desc_rel
            + _W_IN_DEGREE * in_deg
            + _W_DEPTH * depth_fresh
        )

        scored.append(
            ScoredCandidate(
                link_candidate=cand,
                score=total,
                score_breakdown={
                    "url_path": url_path_rel,
                    "title": title_rel,
                    "description": desc_rel,
                    "in_degree": in_deg,
                    "depth": depth_fresh,
                },
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


async def expand(
    seed_url: str,
    query: str,
    query_analysis: QueryAnalysis,
    conn: AsyncConnection,
    *,
    already_ingested_urls: set[str],
    current_depth: int,
) -> ExpansionOutcome:
    """Execute one expansion iteration.

    Steps:
      1. Fetch link_candidates for all ingested source documents.
      2. Filter out already-ingested URLs.
      3. Enrich un-enriched candidates via ``enrich_link_candidates()``.
      4. Generate parent URL candidates from seed URL path derivation.
      5. Score all candidates.
      6. Select top ``max_candidates_per_iteration``.
      7. Scrape selected URLs concurrently.
      8. Index scraped documents.
      9. Return ``ExpansionOutcome``.
    """
    new_depth = current_depth + 1

    # 1. Get all source document IDs for ingested URLs.
    source_doc_ids = await _get_source_document_ids(conn, already_ingested_urls)

    # 2–3. Fetch and enrich link candidates.
    all_candidates: list[PersistedLinkCandidate] = []
    for doc_id, source_url in source_doc_ids:
        # Enrich if not yet done.
        try:
            await enrich_link_candidates(
                doc_id, source_url, conn,
                limit=settings.expansion_map_limit,
            )
        except Exception as exc:
            warnings.warn(
                f"Enrichment failed for {source_url}: {exc!r}",
                stacklevel=2,
            )

        # Fetch (enriched and un-enriched).
        candidates = await get_link_candidates(
            doc_id, conn,
            exclude_urls=already_ingested_urls,
        )
        all_candidates.extend(candidates)

    # 4. Parent URL candidates.
    # For docs sites, stripping path segments often surfaces index pages
    # (e.g. /docs/api/foo → /docs/api/) that link to related content.
    parent_urls = _derive_parent_urls(seed_url)
    for purl in parent_urls:
        if purl not in already_ingested_urls and not any(
            c.target_url == purl for c in all_candidates
        ):
            all_candidates.append(
                _synthetic_candidate(purl, seed_url, new_depth)
            )

    # Deduplicate by target_url.
    seen_urls: set[str] = set()
    deduped: list[PersistedLinkCandidate] = []
    for c in all_candidates:
        if c.target_url not in seen_urls:
            seen_urls.add(c.target_url)
            deduped.append(c)

    # Build in-degree map for scoring.
    in_degree_map = _build_in_degree_map(deduped)

    # 5. Score candidates (limit to candidates_to_score_per_iteration).
    to_score = deduped[: settings.candidates_to_score_per_iteration]
    scored = await score_candidates(
        to_score, query, query_analysis, already_ingested_urls,
        in_degree_map=in_degree_map,
    )

    # 6. Select top candidates.
    selected = scored[: settings.max_candidates_per_iteration]
    if not selected:
        return ExpansionOutcome(
            urls_attempted=[],
            urls_ingested=[],
            urls_failed=[],
            chunks_added=0,
            candidates_scored=len(scored),
            candidates_selected=0,
            depth=new_depth,
        )

    selected_urls = [s.link_candidate.target_url for s in selected]

    # 7. Scrape concurrently.
    scrape_tasks = [ingest(url) for url in selected_urls]
    results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    docs: list[NormalizedDocument] = []
    urls_ingested: list[str] = []
    urls_failed: list[str] = []

    for url, result in zip(selected_urls, results):
        if isinstance(result, Exception):
            warnings.warn(f"Scrape failed for {url}: {result!r}", stacklevel=2)
            urls_failed.append(url)
        elif isinstance(result, NormalizedDocument):
            docs.append(result)
            urls_ingested.append(url)
        else:
            urls_failed.append(url)

    # 8. Index all successful scrapes.
    chunks_added = 0
    if docs:
        depths = [new_depth] * len(docs)
        await index_batch(docs, depths)
        # Lower-bound estimate: actual chunk count depends on document
        # length and chunking config, but 1 doc ≥ 1 chunk always.
        chunks_added = len(docs)

    return ExpansionOutcome(
        urls_attempted=selected_urls,
        urls_ingested=urls_ingested,
        urls_failed=urls_failed,
        chunks_added=chunks_added,
        candidates_scored=len(scored),
        candidates_selected=len(selected),
        depth=new_depth,
    )


# ── Parent URL derivation ────────────────────────────────────────


def _derive_parent_urls(url: str) -> list[str]:
    """Derive parent URLs by stripping path segments.

    Example::

        "https://scikit-learn.org/stable/modules/ensemble.html"
        → ["https://scikit-learn.org/stable/modules/",
           "https://scikit-learn.org/stable/"]

    Excludes the root domain itself (too generic).
    """
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]

    if len(segments) <= 1:
        return []

    parents: list[str] = []
    # Strip from the end, one segment at a time, keeping at least 1 segment.
    for i in range(len(segments) - 1, 0, -1):
        parent_path = "/" + "/".join(segments[:i]) + "/"
        parent_url = f"{parsed.scheme}://{parsed.netloc}{parent_path}"
        parents.append(parent_url)

    return parents


# ── Scoring helpers ───────────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenize on word boundaries."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _url_path_relevance(url: str, key_concepts: set[str]) -> float:
    """Jaccard overlap between URL path tokens and key_concepts."""
    parsed = urlparse(url)
    path_tokens = set(re.findall(r"[a-z0-9]+", parsed.path.lower()))
    if not path_tokens or not key_concepts:
        return 0.0
    intersection = path_tokens & key_concepts
    union = path_tokens | key_concepts
    return len(intersection) / len(union)


def _text_relevance(text: str, query_tokens: set[str]) -> float:
    """Token overlap between candidate text and query tokens."""
    text_tokens = _tokenize(text)
    if not text_tokens or not query_tokens:
        return 0.0
    intersection = text_tokens & query_tokens
    union = text_tokens | query_tokens
    return len(intersection) / len(union)


def _in_degree_signal(target_url: str, in_degree_map: dict[str, int]) -> float:
    """Normalised in-degree: min(1.0, in_degree / 5)."""
    return min(1.0, in_degree_map.get(target_url, 0) / 5)


def _depth_freshness(depth: int) -> float:
    """Prefer shallower candidates: max(0.0, 1.0 - depth * 0.2)."""
    return max(0.0, 1.0 - depth * 0.2)


def _build_in_degree_map(
    candidates: list[PersistedLinkCandidate],
) -> dict[str, int]:
    """Count how many distinct source documents point to each target URL."""
    counts: dict[str, set[UUID]] = {}
    for c in candidates:
        counts.setdefault(c.target_url, set()).add(c.source_document_id)
    return {url: len(sources) for url, sources in counts.items()}


def _synthetic_candidate(
    url: str, source_url: str, depth: int,
) -> PersistedLinkCandidate:
    """Create a synthetic PersistedLinkCandidate for parent URL derivation."""
    return PersistedLinkCandidate(
        id=uuid4(),
        source_document_id=uuid4(),  # Placeholder — not persisted.
        source_url=source_url,
        target_url=url,
        title=None,
        description=None,
        discovered_at=datetime.now(timezone.utc),
        enriched_at=None,
        depth=depth,
    )


async def _get_source_document_ids(
    conn: AsyncConnection,
    ingested_urls: set[str],
) -> list[tuple[UUID, str]]:
    """Get (document_id, source_url) for all ingested URLs."""
    if not ingested_urls:
        return []

    placeholders = ", ".join(["%s"] * len(ingested_urls))
    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT id, source_url FROM documents
            WHERE source_url IN ({placeholders})
            """,  # noqa: S608
            list(ingested_urls),
        )
        rows = await cur.fetchall()

    return [(row[0], row[1]) for row in rows]
