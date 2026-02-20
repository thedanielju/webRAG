from __future__ import annotations

from pprint import pformat
from typing import Any

import pytest


TEST_SCRAPE_URL = "https://scikit-learn.org/stable/modules/ensemble.html#ensemble"
TEST_MAP_URL = "https://scikit-learn.org"


# test firecrawl_client
def _import_client_or_skip():
    try:
        from src.ingestion import firecrawl_client

        return firecrawl_client
    except RuntimeError as exc:
        if "FIRECRAWL_API_KEY" in str(exc):
            pytest.skip(f"Skipping Firecrawl smoke test: {exc}")
        raise


# SDK discovery check
def _as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()  # pydantic model
    if hasattr(result, "dict"):
        return result.dict()  # generic model-like object
    raise AssertionError(f"Unexpected scrape response type: {type(result)!r}")


# dump metadata fields for ingestion schema discovery
def _metadata_dump(result: Any, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None)
    if metadata is None:
        metadata = payload.get("metadata")

    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if hasattr(metadata, "model_dump"):
        return metadata.model_dump()
    if hasattr(metadata, "dict"):
        return metadata.dict()
    if hasattr(metadata, "__dict__"):
        return dict(metadata.__dict__)
    return {"raw_metadata_repr": repr(metadata)}


# calls the scrape wrapper, convert result to dict, print keys and a preview
# and if the response isn't empty
def test_firecrawl_scrape_smoke():
    firecrawl_client = _import_client_or_skip()

    result = firecrawl_client.scrape(TEST_SCRAPE_URL)
    payload = _as_dict(result)
    metadata_dump = _metadata_dump(result, payload)

    print("Scrape response keys:", sorted(payload.keys()))
    print("Scrape response preview:\n", pformat(payload)[:2000])
    print("Scrape metadata dump:\n", pformat(metadata_dump))

    assert payload, "Scrape returned an empty payload"
    assert metadata_dump, "Scrape metadata is empty or missing"
    assert any(
        key in payload for key in ("data", "markdown", "html", "links", "success")
    ), f"Unexpected scrape payload shape: {sorted(payload.keys())}"


def test_firecrawl_map_smoke():
    firecrawl_client = _import_client_or_skip()

    raw_map_result = firecrawl_client.app.map(TEST_MAP_URL)
    print("Raw SDK map type:", type(raw_map_result))

    if hasattr(raw_map_result, "links"):
        raw_links = raw_map_result.links
    elif isinstance(raw_map_result, dict):
        raw_links = raw_map_result.get("links", [])
    elif isinstance(raw_map_result, list):
        raw_links = raw_map_result
    else:
        raw_links = []

    print("Raw SDK map first 5 links:\n", pformat(raw_links[:5]))
    assert isinstance(raw_links, list), "SDK map did not return a list-like links payload"

    try:
        wrapper_links = firecrawl_client.map(TEST_MAP_URL)
    except ValueError as exc:
        pytest.xfail(f"Wrapper map() does not yet handle SDK MapData response: {exc}")

    print("Wrapper map link count:", len(wrapper_links))
    print("First link item:", pformat(raw_links[0]))
    assert isinstance(wrapper_links, list)
