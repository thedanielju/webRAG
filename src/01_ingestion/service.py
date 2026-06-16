from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone # needed for fetched_at timestamp
import hashlib # generates content_hash
import logging
from typing import Any
from urllib.parse import urldefrag, urlparse

from config import settings
from src.ingestion import firecrawl_client, raw_fetch_client
from src.ingestion.politeness import PolitenessGate

logger = logging.getLogger(__name__)


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


class IngestionSkipped(Exception):
    """A URL was deliberately not ingested (robots.txt disallow or a
    non-HTML/binary response on the raw path).  Raised instead of
    returning an empty document so callers can distinguish a skip from a
    real document and surface it.  Both orchestration callers already
    treat ingest() exceptions as a per-URL skip without crashing the run.
    """


# ── Provider selection ────────────────────────────────────────────
# "firecrawl" forces the paid client; "raw" forces the key-free path;
# "auto" prefers Firecrawl when a key is present and falls back to raw.
def _resolve_provider() -> str:
    provider = (settings.ingestion_provider or "auto").strip().lower()
    if provider == "firecrawl":
        return "firecrawl"
    if provider == "raw":
        return "raw"
    # auto
    return "firecrawl" if settings.firecrawl_api_key else "raw"


# A single politeness gate guards all fetches in this process.  Built
# lazily from settings; tests can swap it via set_politeness_gate().
_politeness_gate: PolitenessGate | None = None


def _get_politeness_gate() -> PolitenessGate:
    global _politeness_gate
    if _politeness_gate is None:
        _politeness_gate = PolitenessGate()
    return _politeness_gate


def set_politeness_gate(gate: PolitenessGate | None) -> None:
    """Inject (or reset) the process-wide politeness gate.  Primarily for
    tests; passing None forces a rebuild from current settings."""
    global _politeness_gate
    _politeness_gate = gate

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


async def _scrape_one(url: str) -> Any:
    """Fetch one URL via the selected provider, applying the politeness
    gate (robots + rate limit) first.  Returns the provider's raw scrape
    result for _normalize_document.  Raises IngestionSkipped when robots
    disallows the URL or the raw path hit a non-HTML/binary response."""
    gate = _get_politeness_gate()
    if not await gate.check_and_wait(url):
        raise IngestionSkipped(f"robots.txt disallows {url}")

    if _resolve_provider() == "raw":
        result = await raw_fetch_client.scrape(url)
        if getattr(result, "skipped", False):
            reason = getattr(result, "skip_reason", None) or "unsupported content"
            logger.info("raw ingest skipped %s: %s", url, reason)
            raise IngestionSkipped(f"{url}: {reason}")
        return result

    # Firecrawl path — first so doc_type can use metadata-based detection.
    return await firecrawl_client.scrape(url)


async def ingest(url: str) -> NormalizedDocument:
    result = await _scrape_one(url)
    doc_type = _detect_doc_type(url, result)
    return _normalize_document(result, input_url=url, doc_type=doc_type)


async def ingest_batch(urls: list[str]) -> list[NormalizedDocument | None]:
    # The raw path has no batch endpoint, and both paths must pass through
    # the politeness gate per URL, so route batches through _scrape_one.
    # Firecrawl's batch_scrape is still used directly when provider is
    # firecrawl and robots permits every URL, preserving its throughput.
    if _resolve_provider() == "raw":
        normalized_documents: list[NormalizedDocument | None] = []
        for url in urls:
            try:
                result = await _scrape_one(url)
            except IngestionSkipped as exc:
                logger.info("ingest_batch skipped %s", exc)
                normalized_documents.append(None)
                continue
            except Exception as exc:  # noqa: BLE001 — one bad URL must not kill the batch
                logger.warning("ingest_batch failed for %s: %r", url, exc)
                normalized_documents.append(None)
                continue
            doc_type = _detect_doc_type(url, result)
            normalized_documents.append(
                _normalize_document(result, input_url=url, doc_type=doc_type)
            )
        return normalized_documents

    # Firecrawl path: gate each URL (robots + rate limit), drop disallowed
    # URLs from the batch call, and re-expand results back to positional
    # alignment with the input list.
    gate = _get_politeness_gate()
    allowed_urls: list[str] = []
    allowed_flags: list[bool] = []
    for url in urls:
        permitted = await gate.check_and_wait(url)
        allowed_flags.append(permitted)
        if permitted:
            allowed_urls.append(url)

    fetched = await firecrawl_client.batch_scrape(allowed_urls) if allowed_urls else []
    fetched_iter = iter(fetched)

    normalized_documents = []
    for url, permitted in zip(urls, allowed_flags):
        if not permitted:
            normalized_documents.append(None)
            continue
        result = next(fetched_iter, None)
        if result is None:
            normalized_documents.append(None)
            continue
        doc_type = _detect_doc_type(url, result)
        normalized_documents.append(
            _normalize_document(result, input_url=url, doc_type=doc_type)
        )

    return normalized_documents


# Takes a URL, optional limit for how many links to fetch, optional set of URLs to exclude. Returns a list of LinkCandidate objects.
async def discover_links(
    url: str, limit: int = settings.ingest_discover_links_default_limit, exclude: set[str] | None = None
) -> list[LinkCandidate]:
    # The /map reachability endpoint is Firecrawl-only.  On the key-free
    # raw path there is no site-map service, so link discovery degrades to
    # an empty candidate set (deep-mode reachability is opt-in and the
    # orchestrator already tolerates an empty frontier).
    if _resolve_provider() == "raw":
        return []

    links = await firecrawl_client.map(url, limit=limit)

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
