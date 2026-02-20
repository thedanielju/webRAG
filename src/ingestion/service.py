from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone # needed for fetched_at timestamp
import hashlib # generates content_hash
from typing import Any
from urllib.parse import urldefrag, urlparse

from config import settings
from src.ingestion import firecrawl_client


@dataclass
class NormalizedDocument:
    url: str
    source_url: str | None
    title: str | None
    description: str | None
    language: str | None
    status_code: int | None
    published_time: str | None
    modified_time: str | None
    markdown: str | None
    html: str | None
    links: list[Any]
    fetched_at: datetime
    content_hash: str
    doc_type: str


@dataclass
class LinkCandidate:
    url: str
    title: str | None
    description: str | None

# sometimes, Firecrawl returns different response shapes

# converts anything into a plain dict
def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}

# safely reads single fields from metadata regardless of metadata type
def _metadata_value(metadata: Any, key: str) -> Any:
    if metadata is None:
        return None # no metadata, doesn't raise errors
    if isinstance(metadata, dict):
        return metadata.get(key) # dict access
    return getattr(metadata, key, None) 

# Keep URL-based detection strict to avoid false positives like /pdf-guides/.
# helper method for _detect_doc_type
def _is_pdf_url(url: str | None) -> bool:
    if not isinstance(url, str):
        return False
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")

# Metadata signals are more reliable than URL heuristics, especially for
# endpoints like arXiv that serve PDF content without a .pdf suffix.
def _detect_doc_type(input_url: str, result: Any | None = None) -> str:

    if result is None:
        return "pdf" if _is_pdf_url(input_url) else "html"

    metadata = getattr(result, "metadata", None)
    content_type = _metadata_value(metadata, "content_type")
    num_pages = _metadata_value(metadata, "num_pages")
    source_url = _metadata_value(metadata, "source_url")

    # Strongest signal when available.
    if isinstance(content_type, str) and "application/pdf" in content_type.lower():
        return "pdf"
    # Firecrawl sets page count for parsed PDFs, which is a good fallback.
    if isinstance(num_pages, int) and num_pages > 0:
        return "pdf"
    # Last resort: suffix checks on source and input URL.
    if _is_pdf_url(source_url) or _is_pdf_url(input_url):
        return "pdf"

    return "html"


def _normalize_links(links: Any) -> list[str]:
    if not isinstance(links, list):
        return []

    normalized_links: list[str] = []
    seen: set[str] = set()
    for link in links:
        if not isinstance(link, str):
            continue
        # Strip anchor fragments so page and page#section map to one canonical URL.
        normalized_link = urldefrag(link).url
        if not normalized_link:
            continue
        if normalized_link in seen:
            continue
        seen.add(normalized_link)
        normalized_links.append(normalized_link)

    return normalized_links


def _normalize_document(result: Any, input_url: str, doc_type: str) -> NormalizedDocument:
    # extract fields from scrape result, getattr reads directly off Pydantic document
    payload = _as_dict(result)
    metadata = getattr(result, "metadata", None)
    markdown = getattr(result, "markdown", None)
    html = getattr(result, "html", None)
    links = getattr(result, "links", None)

    # fallback - read from dict version again
    if metadata is None:
        metadata = payload.get("metadata")
    if markdown is None:
        markdown = payload.get("markdown")
    if html is None:
        html = payload.get("html")
    if links is None:
        links = payload.get("links")

    # compute content_hash
    markdown_value = markdown if isinstance(markdown, str) else None
    content_hash = (
        hashlib.sha256(markdown_value.encode("utf-8")).hexdigest()
        if markdown_value
        else ""
    )

    # build NormalizeDocument with straightforward field mapping
    return NormalizedDocument(
        url=input_url,
        source_url=_metadata_value(metadata, "source_url"),
        title=_metadata_value(metadata, "title"),
        description=_metadata_value(metadata, "description"),
        language=_metadata_value(metadata, "language"),
        status_code=_metadata_value(metadata, "status_code"),
        published_time=_metadata_value(metadata, "published_time"),
        modified_time=_metadata_value(metadata, "modified_time"),
        markdown=markdown_value,
        html=html if isinstance(html, str) else None, # typeguards ensure non-strings are not stored as html
        links=_normalize_links(links),
        fetched_at=datetime.now(timezone.utc),
        content_hash=content_hash,
        doc_type=doc_type,
    )


def ingest(url: str) -> NormalizedDocument:
    # Scrape first so doc_type can use metadata-based detection.
    result = firecrawl_client.scrape(url)
    doc_type = _detect_doc_type(url, result)
    return _normalize_document(result, input_url=url, doc_type=doc_type)


def ingest_batch(urls: list[str]) -> list[NormalizedDocument | None]:
    results = firecrawl_client.batch_scrape(urls)

    normalized_documents: list[NormalizedDocument | None] = []
    for url, result in zip(urls, results):
        # Preserve positional alignment with input URLs for orchestration.
        if result is None:
            normalized_documents.append(None)
            continue
        doc_type = _detect_doc_type(url, result)
        normalized_documents.append(
            _normalize_document(result, input_url=url, doc_type=doc_type)
        )

    return normalized_documents


# Takes a URL, optional limit for how many links to fetch, optional set of URLs to exclude. Returns a list of LinkCandidate objects.
def discover_links(
    url: str, limit: int = settings.ingest_discover_links_default_limit, exclude: set[str] | None = None
) -> list[LinkCandidate]:
    links = firecrawl_client.map(url, limit=limit)

# Calls map, initializes empty results list, converts exclude to an empty set if None was passed, so in excluded check always works without needing a None check later.
    candidates: list[LinkCandidate] = []
    excluded = exclude or set()

    for link in links:
        # map() may return model objects or plain dicts depending on SDK shape.
        link_url = getattr(link, "url", None)
        title = getattr(link, "title", None)
        description = getattr(link, "description", None)

        if link_url is None and isinstance(link, dict):
            link_url = link.get("url")
            title = link.get("title")
            description = link.get("description")

        # two filters - skip anything without a valid URL string, anything in exclude
        if not isinstance(link_url, str):
            continue
        if link_url in excluded:
            continue

        candidates.append(
            LinkCandidate(url=link_url, title=title, description=description)
        )

    return candidates
