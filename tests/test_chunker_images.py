from __future__ import annotations

from datetime import datetime, timezone

import pytest

from config import settings
import src.indexing.chunker as chunker


@pytest.fixture(autouse=True)
def _fast_token_counter(monkeypatch):
    """Avoid loading real tokenizers in chunker unit tests."""
    monkeypatch.setattr(chunker, "_TOKEN_COUNTER", lambda text: max(1, len(text.split())))
    monkeypatch.setattr(settings, "child_target_tokens", 10_000)
    monkeypatch.setattr(settings, "parent_max_tokens", 10_000)


def _build(markdown: str, html: str | None):
    return chunker.build_chunks(
        markdown=markdown,
        html=html,
        source_url="https://example.com/page",
        fetched_at=datetime.now(timezone.utc),
        title="Example",
        depth=0,
    )


def test_chunks_with_no_images_have_no_image_signal():
    markdown = """# Title

This is a plain paragraph with no images.
"""
    html = "<html><body><h1>Title</h1><p>This is a plain paragraph with no images.</p></body></html>"

    parents, children = _build(markdown, html)
    assert len(parents) > 0
    assert len(children) > 0

    assert all(child.has_image is False for child in children)
    assert all(child.image_refs == [] for child in children)
    assert all(child.image_context_text is None for child in children)

    assert all(parent.has_image is False for parent in parents)
    assert all(parent.image_refs == [] for parent in parents)
    assert all(parent.image_context_text is None for parent in parents)


def test_decorative_empty_alt_image_preserves_ref_but_no_meaningful_signal():
    markdown = """# Title

![](https://example.com/decorative.png)

Decorative separator image only.
"""
    html = """
    <html><body>
      <h1>Title</h1>
      <p><img src="https://example.com/decorative.png" alt="" /></p>
      <p>Decorative separator image only.</p>
    </body></html>
    """

    parents, children = _build(markdown, html)
    assert len(children) > 0

    image_children = [child for child in children if child.has_image]
    assert len(image_children) >= 1

    # Image metadata is preserved for rendering, but empty-alt decorative images
    # should not create meaningful text signal for embedding/reranking.
    assert any(child.image_context_text is None for child in image_children)
    assert any(any(ref.url.endswith("decorative.png") for ref in child.image_refs) for child in image_children)


def test_multiple_images_captured_and_deduped_at_parent_level():
    markdown = """# Figures

![Architecture diagram](https://example.com/a.png)
![Architecture diagram](https://example.com/a.png)
![Deployment chart](https://example.com/b.png)

System overview and rollout notes.
"""
    html = """
    <html><body>
      <h1>Figures</h1>
      <figure>
        <img src="https://example.com/a.png" alt="Architecture diagram" />
        <figcaption>Architecture diagram</figcaption>
      </figure>
      <figure>
        <img src="https://example.com/a.png" alt="Architecture diagram" />
        <figcaption>Architecture diagram</figcaption>
      </figure>
      <figure>
        <img src="https://example.com/b.png" alt="Deployment chart" />
        <figcaption>Deployment chart</figcaption>
      </figure>
      <p>System overview and rollout notes.</p>
    </body></html>
    """

    parents, children = _build(markdown, html)
    assert len(parents) > 0
    assert len(children) > 0

    image_children = [child for child in children if child.has_image]
    assert len(image_children) > 0

    # Child-level refs are deduped by URL/metadata.
    child_urls = set()
    for child in image_children:
        for ref in child.image_refs:
            child_urls.add(ref.url)
    assert "https://example.com/a.png" in child_urls
    assert "https://example.com/b.png" in child_urls

    # Parent aggregates and dedupes image refs across children.
    parent = parents[0]
    assert parent.has_image is True
    parent_urls = [ref.url for ref in parent.image_refs]
    assert parent_urls.count("https://example.com/a.png") == 1
    assert parent_urls.count("https://example.com/b.png") == 1
    assert parent.image_context_text is not None
    assert "Architecture diagram" in parent.image_context_text
    assert "Deployment chart" in parent.image_context_text

