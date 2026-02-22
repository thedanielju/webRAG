from __future__ import annotations

from dataclasses import asdict
from pprint import pformat

import pytest


HTML_URL = "https://scikit-learn.org/stable/modules/ensemble.html"
PDF_URL = "https://arxiv.org/pdf/1706.03762"
MAP_URL = "https://scikit-learn.org"
INVALID_URL = "https://this-url-does-not-exist-xyz.com"


def _safe_preview(text: str, size: int) -> str:
    return text[:size].encode("cp1252", errors="replace").decode("cp1252")


def _import_or_skip():
    try:
        from src.ingestion.service import LinkCandidate, NormalizedDocument
        from src.ingestion.service import discover_links, ingest, ingest_batch

        return LinkCandidate, NormalizedDocument, discover_links, ingest, ingest_batch
    except RuntimeError as exc:
        if "FIRECRAWL_API_KEY" in str(exc):
            pytest.skip(f"Skipping service smoke test: {exc}")
        raise


@pytest.mark.asyncio
async def test_ingest_html():
    LinkCandidate, NormalizedDocument, discover_links, ingest, ingest_batch = _import_or_skip()

    result = await ingest(HTML_URL)

    assert isinstance(result, NormalizedDocument)
    assert result.doc_type == "html"
    assert result.markdown is not None and len(result.markdown) > 0
    assert result.html is not None and len(result.html) > 0
    assert isinstance(result.links, list)
    assert result.content_hash is not None and len(result.content_hash) > 0
    assert result.fetched_at is not None

    doc_dump = asdict(result)
    doc_dump.pop("markdown", None)
    doc_dump.pop("html", None)
    print("ingest(html) fields (without markdown/html):\n", pformat(doc_dump))
    print("ingest(html) markdown preview:\n", result.markdown[:500])
    print("ingest(html) links count:", len(result.links))


@pytest.mark.asyncio
async def test_ingest_pdf():
    LinkCandidate, NormalizedDocument, discover_links, ingest, ingest_batch = _import_or_skip()

    result = await ingest(PDF_URL)

    assert isinstance(result, NormalizedDocument)
    assert result.doc_type == "pdf"
    assert result.markdown is not None and len(result.markdown) > 0

    print("ingest(pdf) doc_type:", result.doc_type)
    print("ingest(pdf) title:", result.title)
    print("ingest(pdf) markdown preview:\n", _safe_preview(result.markdown, 500))
    print("ingest(pdf) links count:", len(result.links))


@pytest.mark.asyncio
async def test_ingest_batch():
    LinkCandidate, NormalizedDocument, discover_links, ingest, ingest_batch = _import_or_skip()

    results = await ingest_batch([HTML_URL, INVALID_URL])

    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], NormalizedDocument)
    assert results[1] is None

    valid_result = results[0]
    print("ingest_batch valid doc_type:", valid_result.doc_type)
    print("ingest_batch valid title:", valid_result.title)
    print("ingest_batch valid content_hash:", valid_result.content_hash)
    print("ingest_batch valid markdown preview:\n", (valid_result.markdown or "")[:300])


@pytest.mark.asyncio
async def test_discover_links():
    LinkCandidate, NormalizedDocument, discover_links, ingest, ingest_batch = _import_or_skip()

    candidates = await discover_links(MAP_URL, limit=10)

    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert all(isinstance(item, LinkCandidate) for item in candidates)
    assert all(isinstance(item.url, str) for item in candidates)

    print("discover_links first 5 candidates:")
    for candidate in candidates[:5]:
        print(
            {
                "url": candidate.url,
                "title": candidate.title,
                "description": candidate.description,
            }
        )


@pytest.mark.asyncio
async def test_discover_links_exclude():
    LinkCandidate, NormalizedDocument, discover_links, ingest, ingest_batch = _import_or_skip()

    initial_candidates = await discover_links(MAP_URL, limit=20)
    exclude_set = {candidate.url for candidate in initial_candidates[:5]}

    filtered_candidates = await discover_links(MAP_URL, limit=20, exclude=exclude_set)
    filtered_urls = {candidate.url for candidate in filtered_candidates}

    assert all(excluded_url not in filtered_urls for excluded_url in exclude_set)

    print("Excluded URLs:")
    for excluded_url in sorted(exclude_set):
        print(excluded_url)
    print("Confirmed excluded URLs are absent from second discover_links call")
