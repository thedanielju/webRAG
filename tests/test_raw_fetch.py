from __future__ import annotations

from dataclasses import fields

import httpx
import pytest

from src.ingestion import raw_fetch_client, service
from src.ingestion.politeness import PolitenessGate
from src.ingestion.service import NormalizedDocument


# ── Fixtures / helpers ────────────────────────────────────────────


def _mock_client(handler) -> httpx.AsyncClient:
    """Build an AsyncClient backed by httpx.MockTransport (no network)."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, follow_redirects=True)


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <title>Sample Article</title>
  <meta name="description" content="A test page about math." />
  <meta property="og:title" content="Sample Article (OG)" />
</head>
<body>
  <nav><a href="/home">Home</a></nav>
  <main>
    <h1>Sample Article</h1>
    <h2>Section One</h2>
    <p>Some prose here with a link to <a href="/related">related material</a>.</p>
    <p>An external reference: <a href="https://other.example.com/x">x</a></p>
    <h2>Math Section</h2>
    <p>The mass-energy relation is $$E = mc^2$$ in display form.</p>
    <p>Inline form: \\(a^2 + b^2 = c^2\\).</p>
  </main>
  <footer><a href="/home#contact">Contact</a></footer>
</body>
</html>
"""


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch):
    """Default to an allow-all, no-wait politeness gate so raw-fetch tests
    exercise the fetcher, not robots/rate-limit behaviour."""
    async def _allow_all(_origin: str):
        return None  # None body => allow-all

    gate = PolitenessGate(
        respect_robots=True,
        rate_limit_rps=0,  # no spacing
        robots_fetcher=_allow_all,
    )
    service.set_politeness_gate(gate)
    yield
    service.set_politeness_gate(None)


# ── raw_fetch_client.scrape ───────────────────────────────────────


@pytest.mark.asyncio
async def test_raw_scrape_extracts_markdown_title_links():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, text=HTML_PAGE
        )

    async with _mock_client(handler) as client:
        result = await raw_fetch_client.scrape(
            "https://example.com/article", client=client
        )

    assert result.skipped is False
    assert result.status_code == 200
    assert result.markdown is not None
    # Heading structure preserved for the chunker to section on.
    assert "## Section One" in result.markdown
    assert "## Math Section" in result.markdown
    # OG title preferred over <title>.
    assert result.metadata.title == "Sample Article (OG)"
    assert result.metadata.description == "A test page about math."
    assert result.metadata.language == "en"

    # Links: absolute, deduped, fragment-stripped; nav/footer outside <main>
    # are excluded because extraction is scoped to the main-content root.
    assert "https://example.com/related" in result.links
    assert "https://other.example.com/x" in result.links
    # /home appears only in nav (outside <main>) and footer with a fragment;
    # neither should leak in.
    assert "https://example.com/home" not in result.links


@pytest.mark.asyncio
async def test_raw_scrape_preserves_latex():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=HTML_PAGE)

    async with _mock_client(handler) as client:
        result = await raw_fetch_client.scrape("https://example.com/m", client=client)

    assert "$$E = mc^2$$" in result.markdown
    assert "\\(a^2 + b^2 = c^2\\)" in result.markdown


@pytest.mark.asyncio
async def test_raw_scrape_non_html_is_skipped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4..."
        )

    async with _mock_client(handler) as client:
        result = await raw_fetch_client.scrape("https://example.com/doc.pdf", client=client)

    assert result.skipped is True
    assert result.markdown is None
    assert "non-HTML" in (result.skip_reason or "")


@pytest.mark.asyncio
async def test_raw_scrape_network_error_skips_not_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with _mock_client(handler) as client:
        result = await raw_fetch_client.scrape("https://example.com/down", client=client)

    assert result.skipped is True
    assert result.markdown is None
    assert result.status_code is None


# ── NormalizedDocument parity through the service layer ───────────


@pytest.mark.asyncio
async def test_raw_path_produces_normalized_document_shape(monkeypatch):
    monkeypatch.setattr(service.settings, "ingestion_provider", "raw")

    async def fake_scrape(url, *, client=None):
        # Return a real RawScrapeResult so _normalize_document/_detect_doc_type
        # exercise the actual getattr-based field mapping.
        return raw_fetch_client.RawScrapeResult(
            markdown="# Title\n\nBody with $$x^2$$ math.",
            html="<html><body><h1>Title</h1><p>Body</p></body></html>",
            links=["https://example.com/a", "https://example.com/a#frag"],
            metadata=raw_fetch_client.RawMetadata(
                source_url="https://example.com/p",
                title="Title",
                description="desc",
                language="en",
                status_code=200,
                content_type="text/html",
            ),
            status_code=200,
        )

    monkeypatch.setattr(raw_fetch_client, "scrape", fake_scrape)

    doc = await service.ingest("https://example.com/p")

    assert isinstance(doc, NormalizedDocument)
    # Every NormalizedDocument field is populated with the right type.
    field_names = {f.name for f in fields(NormalizedDocument)}
    assert field_names == {
        "url",
        "source_url",
        "title",
        "description",
        "language",
        "status_code",
        "published_time",
        "modified_time",
        "markdown",
        "html",
        "links",
        "fetched_at",
        "content_hash",
        "doc_type",
    }
    assert doc.url == "https://example.com/p"
    assert doc.source_url == "https://example.com/p"
    assert doc.title == "Title"
    assert doc.description == "desc"
    assert doc.language == "en"
    assert doc.status_code == 200
    assert doc.doc_type == "html"
    assert isinstance(doc.markdown, str) and "$$x^2$$" in doc.markdown
    assert isinstance(doc.html, str)
    # Links normalized (fragment dedup) by the shared service helper.
    assert doc.links == ["https://example.com/a"]
    assert isinstance(doc.content_hash, str) and len(doc.content_hash) == 64
    assert doc.fetched_at is not None


@pytest.mark.asyncio
async def test_raw_path_skip_raises_ingestion_skipped(monkeypatch):
    monkeypatch.setattr(service.settings, "ingestion_provider", "raw")

    async def fake_scrape(url, *, client=None):
        return raw_fetch_client.RawScrapeResult(
            markdown=None,
            html=None,
            skipped=True,
            skip_reason="non-HTML content-type: image/png",
        )

    monkeypatch.setattr(raw_fetch_client, "scrape", fake_scrape)

    with pytest.raises(service.IngestionSkipped):
        await service.ingest("https://example.com/pic.png")
