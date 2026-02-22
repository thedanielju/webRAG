from __future__ import annotations

from typing import Any

import pytest


VALID_URL = "https://scikit-learn.org/stable/modules/ensemble.html"
INVALID_URL = "https://this-url-does-not-exist-xyz.com"


def _import_client_or_skip():
    try:
        from src.ingestion import firecrawl_client

        return firecrawl_client
    except RuntimeError as exc:
        if "FIRECRAWL_API_KEY" in str(exc):
            pytest.skip(f"Skipping Firecrawl smoke test: {exc}")
        raise


def _as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {}


@pytest.mark.asyncio
async def test_firecrawl_batch_output():
    firecrawl_client = _import_client_or_skip()

    # batch_scrape() is now async.
    results = await firecrawl_client.batch_scrape([VALID_URL, INVALID_URL])

    print("Batch results count:", len(results))
    for index, item in enumerate(results):
        label = "None" if item is None else type(item).__name__
        print(f"Result {index} type:", label)

    valid_doc = results[0]
    valid_payload = _as_dict(valid_doc)
    print("Valid document keys:", sorted(valid_payload.keys()))

    markdown = valid_payload.get("markdown")
    markdown_preview = markdown[:500] if isinstance(markdown, str) else ""
    print("Valid markdown preview (first 500 chars):\n", markdown_preview)

    links = valid_payload.get("links")
    links_count = len(links) if isinstance(links, list) else 0
    print("Valid links count:", links_count)

    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is None
