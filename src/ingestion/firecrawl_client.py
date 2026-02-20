# exception contract not specified yet as im the only caller; above layers might necessitate this
# client handles only API access and exposes scrape and map

from __future__ import annotations

import os
from urllib.parse import urldefrag, urlparse
from typing import Any

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()

# api key management
_FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not _FIRECRAWL_API_KEY:
    raise RuntimeError(
        "Missing FIRECRAWL_API_KEY - please set it in your environment or .env"
    )

app = FirecrawlApp(api_key=_FIRECRAWL_API_KEY)

DEFAULT_SCRAPE_OPTIONS: dict[str, Any] = {
    "formats": [{"type": "markdown"}, {"type": "html"}, {"type": "links"}],
    "only_main_content": True,
    "parsers": [{"type": "pdf"}],
    "timeout": 30000, # 30s
    "block_ads": True,
    "remove_base64_images": True,
    "proxy": "auto",
}

# URL string input, optional dict of options to override defaults
# options parameters exists so callers can later override specific defaults
# very unlikely service will override anything

def scrape(url: str, options: dict[str, Any] | None = None) -> dict[str, Any]: # no options defaults to None
    merged_options = {**DEFAULT_SCRAPE_OPTIONS, **(options or {})}
    return app.scrape(url, **merged_options)

# add ingest_batch at orchestration step

def _normalize_url_for_match(url: str | None) -> str:
    if not url:
        return ""
    without_fragment = urldefrag(url).url
    parsed = urlparse(without_fragment)
    normalized_path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{normalized_path}"


def _extract_batch_documents(response: Any) -> list[Any]:
    if isinstance(response, list):
        return response

    if isinstance(response, dict):
        data = response.get("data")
        if isinstance(data, list):
            return data

    if hasattr(response, "data") and isinstance(response.data, list):
        return response.data

    raise ValueError("Unexpected response shape from Firecrawl batch_scrape endpoint.")


def batch_scrape(urls: list[str], options: dict[str, Any] | None = None) -> list[Any]:
    merged_options = {**DEFAULT_SCRAPE_OPTIONS, **(options or {})}
    response = app.batch_scrape(urls, **merged_options)
    documents = _extract_batch_documents(response)

    if len(documents) == len(urls):
        return documents

    normalized_to_docs: dict[str, list[Any]] = {}
    for doc in documents:
        metadata = getattr(doc, "metadata", None)
        source_url = getattr(metadata, "source_url", None)
        fallback_url = getattr(metadata, "url", None) or getattr(doc, "url", None)
        normalized = _normalize_url_for_match(source_url or fallback_url)
        if normalized:
            normalized_to_docs.setdefault(normalized, []).append(doc)

    ordered_documents: list[Any] = []
    used_doc_ids: set[int] = set()
    for url in urls:
        normalized_input = _normalize_url_for_match(url)
        candidates = normalized_to_docs.get(normalized_input, [])
        if candidates:
            matched_doc = candidates.pop(0)
            ordered_documents.append(matched_doc)
            used_doc_ids.add(id(matched_doc))
            continue
        ordered_documents.append(None)

    remaining_documents = [doc for doc in documents if id(doc) not in used_doc_ids]
    for index, doc in enumerate(ordered_documents):
        if doc is None and remaining_documents:
            ordered_documents[index] = remaining_documents.pop(0)

    return ordered_documents

def map(url: str, limit: int = 100) -> list[Any]:
    response = app.map(url, limit=limit)  # cap candidate link volume for orchestration

    if isinstance(response, list):
        return response

    if isinstance(response, dict):
        links = response.get("links")
        if isinstance(links, list):
            return links

    if hasattr(response, "links"):
        return response.links

    raise ValueError("Unexpected response shape from Firecrawl map endpoint.")
