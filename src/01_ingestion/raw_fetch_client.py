# Key-free fetch path: plain HTTP via httpx + main-content extraction via
# BeautifulSoup.  This is the fallback when no FIRECRAWL_API_KEY is set.
#
# The single contract this module owns: scrape() must return the SAME
# shape the Firecrawl path feeds into service._normalize_document, so the
# rest of the pipeline (indexing, retrieval) cannot tell which path ran.
# Concretely it returns a RawScrapeResult with .markdown / .html / .links
# / .metadata attributes mirroring a Firecrawl document, plus .status_code.
#
# Unlike Firecrawl, the raw path PRESERVES LaTeX ($...$, \(...\), \[...\])
# because it does not round-trip content through a markdown renderer that
# strips math.  The chunker's _detect_markdown_flags re-enables LaTeX
# detection accordingly.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urldefrag, urlparse

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag

from config import settings


# A User-Agent that identifies the crawler honestly.  Some origins block
# the default httpx UA; a descriptive UA is also the polite thing to send
# alongside robots.txt compliance.
_USER_AGENT = "WebRAG/1.0 (+https://github.com/; raw-fetch)"

# Content types we can extract text from.  Anything else (images, video,
# archives, octet-stream) is skipped — see scrape()'s binary guard.
_TEXT_CONTENT_TYPES = ("text/html", "application/xhtml", "text/plain")

# Tags whose contents are never page prose.  Stripped before extraction.
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "form",
    "button",
)

# Containers that, when present, are the best signal for "main content".
# Checked in priority order; first match wins.
_MAIN_CONTENT_SELECTORS = ("main", "article", '[role="main"]')


@dataclass
class RawMetadata:
    """Mirror of the subset of Firecrawl document.metadata that
    service._metadata_value reads.  Attribute access only — service uses
    getattr/_metadata_value, never dict keys, on the metadata object."""

    source_url: str | None = None
    title: str | None = None
    description: str | None = None
    language: str | None = None
    status_code: int | None = None
    published_time: str | None = None
    modified_time: str | None = None
    content_type: str | None = None
    num_pages: int | None = None


@dataclass
class RawScrapeResult:
    """Firecrawl-document-shaped result for the raw path.

    service._normalize_document reads .metadata / .markdown / .html /
    .links off the scrape result via getattr, then falls back to a dict
    view.  Exposing those four attributes is enough for full parity; the
    extra .status_code / .skipped fields help the service layer log and
    skip without re-parsing.
    """

    markdown: str | None
    html: str | None
    links: list[str] = field(default_factory=list)
    metadata: RawMetadata = field(default_factory=RawMetadata)
    status_code: int | None = None
    # Set when the response was non-HTML/binary and could not be extracted.
    # service.py surfaces this as a skip rather than indexing empty content.
    skipped: bool = False
    skip_reason: str | None = None


def _is_text_response(content_type: str | None) -> bool:
    if not content_type:
        # No Content-Type header — assume text and let the parser decide.
        # Empty/garbage bodies just produce empty markdown, which the
        # service layer treats as a skip.
        return True
    lowered = content_type.lower()
    return any(token in lowered for token in _TEXT_CONTENT_TYPES)


def _meta_content(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str | None:
    if name is not None:
        tag = soup.find("meta", attrs={"name": name})
    else:
        tag = soup.find("meta", attrs={"property": prop})
    if isinstance(tag, Tag):
        value = tag.get("content")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_title(soup: BeautifulSoup) -> str | None:
    og_title = _meta_content(soup, prop="og:title")
    if og_title:
        return og_title
    if soup.title and soup.title.string:
        stripped = soup.title.string.strip()
        if stripped:
            return stripped
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        text = h1.get_text(" ", strip=True)
        if text:
            return text
    return None


def _extract_language(soup: BeautifulSoup) -> str | None:
    html_tag = soup.find("html")
    if isinstance(html_tag, Tag):
        lang = html_tag.get("lang")
        if isinstance(lang, str) and lang.strip():
            return lang.strip()
    return _meta_content(soup, prop="og:locale")


def _main_content_root(soup: BeautifulSoup) -> Tag:
    """Return the most content-bearing subtree, falling back to <body>.

    Mirrors Firecrawl's only_main_content=True intent: prefer semantic
    landmarks, then the whole body if none exist."""
    for selector in _MAIN_CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if isinstance(node, Tag) and node.get_text(strip=True):
            return node
    body = soup.body
    if isinstance(body, Tag):
        return body
    return soup  # degenerate fragment with no <body>


def _extract_links(root: Tag, base_url: str) -> list[str]:
    """Collect absolute, fragment-stripped, deduped hrefs from the page.

    service._normalize_links re-runs dedup/defrag, but doing it here too
    keeps the raw result self-consistent and bounds list size early."""
    seen: set[str] = set()
    links: list[str] = []
    for anchor in root.find_all("a", href=True):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        href = href.strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        normalized = urldefrag(absolute).url
        parsed = urlparse(normalized)
        if parsed.scheme not in ("http", "https"):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links


# ── Markdown-ish rendering ────────────────────────────────────────
# Goal is not a faithful HTML->Markdown converter (Firecrawl's job) but a
# clean, heading-aware text surface the chunker can section on (## / #)
# while PRESERVING inline LaTeX.  We walk block-level elements and emit
# markdown headings + paragraph breaks; inline math survives untouched
# because get_text() returns the raw $...$ / \(...\) source verbatim.

_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "blockquote",
    "pre",
    "td",
    "th",
    "tr",
    "figcaption",
    "dd",
    "dt",
}
_HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


def _render_markdownish(root: Tag) -> str:
    lines: list[str] = []

    def _walk(node: Tag) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in _HEADING_TAGS:
                text = child.get_text(" ", strip=True)
                if text:
                    lines.append(f"{_HEADING_TAGS[name]} {text}")
                continue
            if name == "pre":
                code = child.get_text("\n", strip=False).strip("\n")
                if code:
                    lines.append("```")
                    lines.append(code)
                    lines.append("```")
                continue
            if name in {"ul", "ol"}:
                for item in child.find_all("li", recursive=False):
                    item_text = item.get_text(" ", strip=True)
                    if item_text:
                        lines.append(f"- {item_text}")
                continue
            if name in _BLOCK_TAGS:
                # If the block has block-level children, recurse so nested
                # headings/lists are not flattened into one paragraph.
                if child.find(list(_HEADING_TAGS) + ["ul", "ol", "pre", "p"], recursive=False):
                    _walk(child)
                else:
                    text = child.get_text(" ", strip=True)
                    if text:
                        lines.append(text)
                continue
            # Unknown container — descend to find block content within.
            _walk(child)

    _walk(root)

    # Collapse runs of blank lines; join paragraphs with a blank line so
    # _paragraph_blocks (split on \n\s*\n) sees clean boundaries.
    rendered = "\n\n".join(line for line in lines if line.strip())
    return rendered.strip()


def _build_metadata(
    soup: BeautifulSoup,
    *,
    final_url: str,
    status_code: int,
    content_type: str | None,
) -> RawMetadata:
    return RawMetadata(
        source_url=final_url,
        title=_extract_title(soup),
        description=(
            _meta_content(soup, name="description")
            or _meta_content(soup, prop="og:description")
        ),
        language=_extract_language(soup),
        status_code=status_code,
        published_time=(
            _meta_content(soup, prop="article:published_time")
            or _meta_content(soup, name="date")
        ),
        modified_time=_meta_content(soup, prop="article:modified_time"),
        content_type=content_type,
        num_pages=None,
    )


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


async def scrape(url: str, *, client: httpx.AsyncClient | None = None) -> RawScrapeResult:
    """Fetch *url* and return a Firecrawl-document-shaped result.

    Non-HTML / binary responses return a skipped result (markdown=None)
    rather than raising, so a single bad URL never crashes a crawl.  An
    injectable *client* lets tests drive a mock transport with no network.
    """
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=settings.raw_fetch_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )

    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        # Network-level failure (DNS, connect, timeout) — skip, don't crash.
        return RawScrapeResult(
            markdown=None,
            html=None,
            status_code=None,
            skipped=True,
            skip_reason=f"fetch failed: {exc.__class__.__name__}",
        )
    finally:
        if owns_client:
            await client.aclose()

    content_type = response.headers.get("content-type")
    final_url = str(response.url)
    status_code = response.status_code

    if not _is_text_response(content_type):
        # Binary or unsupported (PDF, image, zip, ...).  The raw path does
        # not parse PDFs; surface a clean skip so service.py can log it.
        return RawScrapeResult(
            markdown=None,
            html=None,
            links=[],
            metadata=RawMetadata(
                source_url=final_url,
                status_code=status_code,
                content_type=content_type,
            ),
            status_code=status_code,
            skipped=True,
            skip_reason=f"non-HTML content-type: {content_type}",
        )

    html = response.text
    soup = _soup(html)

    # Drop noise before extraction so it pollutes neither markdown nor links.
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    root = _main_content_root(soup)
    markdown = _render_markdownish(root) or None
    links = _extract_links(root, final_url)
    metadata = _build_metadata(
        soup, final_url=final_url, status_code=status_code, content_type=content_type
    )

    return RawScrapeResult(
        markdown=markdown,
        html=html,
        links=links,
        metadata=metadata,
        status_code=status_code,
        skipped=markdown is None,
        skip_reason=None if markdown else "no extractable content",
    )
