"""OrchestrationResult → structured plain-text response.

Assembles the [SOURCES], [EVIDENCE], [IMAGES], [EXPANSION TRACE],
[CITATIONS], and [STATS] sections that the reasoning model reads
as a research brief.  Manages a soft token budget so the response
stays within ``mcp_response_token_budget``.

Design notes:
  - Sections use bracketed headers ([SOURCES], [EVIDENCE], etc.) rather
    than Markdown headings because MCP tool output is plain text, not
    rendered Markdown.  Bracketed headers are unambiguous for LLM parsing.
  - Token budget priority ensures essential context (sources + stats)
    is never truncated.  Evidence fills the remaining space and is the
    only section that can be partially included (with a truncation note).
  - Token counting uses tiktoken's cl100k_base encoding — the same
    tokenizer used by the embedding model — so budget estimates align
    with the model's actual token consumption.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import tiktoken

from config import settings
from src.mcp_server.html_converter import ImageRef, convert_html
from src.orchestration.models import (
    ExpansionStep,
    OrchestrationResult,
    RankedChunk,
)
from src.retrieval.citations import CitationSpan

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy-initialised tokenizer (module-level singleton).
# We avoid loading tiktoken at import time because it downloads
# the encoding file on first use, which would slow cold starts.
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(settings.embedding_tokenizer_name)
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ── Public entry point ────────────────────────────────────────


def format_result(result: OrchestrationResult) -> str:
    """Format an ``OrchestrationResult`` into the full MCP response text.

    Sections are assembled in budget-priority order (§6.3):
      1. [SOURCES] — always included in full.
      2. [STATS]   — always included in full.
      3. [EVIDENCE] — fills remaining budget; truncated if needed.
      4. [EXPANSION TRACE] — included if budget allows.
      5. [IMAGES]  — included if budget allows.
      6. [CITATIONS] — included if budget allows.
    """
    budget = settings.mcp_response_token_budget

    # ── Build source index ────────────────────────────────────
    source_map, sources_section = _build_sources(result.chunks)
    sources_tokens = _count_tokens(sources_section)

    # ── Build stats ───────────────────────────────────────────
    stats_section = _build_stats(result)
    stats_tokens = _count_tokens(stats_section)

    remaining = budget - sources_tokens - stats_tokens

    # ── Build evidence (fills remaining budget) ───────────────
    evidence_section, evidence_images, evidence_tokens = _build_evidence(
        result.chunks, source_map, remaining,
    )
    remaining -= evidence_tokens

    # ── Build expansion trace ─────────────────────────────────
    trace_section = ""
    if result.expansion_steps and remaining > 200:
        trace_section = _build_expansion_trace(result.expansion_steps)
        trace_tokens = _count_tokens(trace_section)
        if trace_tokens <= remaining:
            remaining -= trace_tokens
        else:
            n = len(result.expansion_steps)
            trace_section = (
                f"[EXPANSION TRACE]\n"
                f"[Expansion trace omitted — {n} iterations, see stats above]"
            )
            remaining -= _count_tokens(trace_section)

    # ── Build images section ──────────────────────────────────
    images_section = ""
    if evidence_images and remaining > 100:
        images_section = _build_images(evidence_images, source_map)
        img_tokens = _count_tokens(images_section)
        if img_tokens <= remaining:
            remaining -= img_tokens
        else:
            images_section = ""

    # ── Build citations ───────────────────────────────────────
    citations_section = ""
    if result.citations and remaining > 100:
        citations_section = _build_citations(result.citations, source_map)
        cit_tokens = _count_tokens(citations_section)
        if cit_tokens <= remaining:
            remaining -= cit_tokens
        else:
            citations_section = ""

    # ── Assemble ──────────────────────────────────────────────
    parts = [sources_section, evidence_section]
    if images_section:
        parts.append(images_section)
    if trace_section:
        parts.append(trace_section)
    if citations_section:
        parts.append(citations_section)
    parts.append(stats_section)

    response = "\n\n".join(parts)

    logger.debug(
        "formatted response: %d tokens, %d chars (budget %d)",
        _count_tokens(response), len(response), budget,
    )

    return response


# ── Source index ──────────────────────────────────────────────

SourceMap = dict[str, int]  # source_url → 1-based reference number


def _build_sources(
    chunks: list[RankedChunk],
) -> tuple[SourceMap, str]:
    """Build [SOURCES] section and a url→number mapping.

    Sources are numbered in first-seen order (matching chunk ordering)
    so the model can reference them as [1], [2], etc.  Section headings
    are aggregated per URL for sub-section attribution.
    """
    # First-seen order matching chunk ordering.
    source_map: SourceMap = {}
    # url → {title, set of section headings}
    source_info: dict[str, dict] = {}

    for rc in chunks:
        c = rc.chunk
        url = c.source_url
        if url not in source_map:
            source_map[url] = len(source_map) + 1
            source_info[url] = {
                "title": c.title or "Untitled",
                "headings": set(),
            }
        if c.section_heading:
            source_info[url]["headings"].add(c.section_heading)

    if not source_map:
        return source_map, "[SOURCES]\n(none)"

    lines = ["[SOURCES]"]
    for url, num in source_map.items():
        info = source_info[url]
        lines.append(f"[{num}] {info['title']} — {url}")
        for heading in sorted(info["headings"]):
            lines.append(f"    § {heading}")

    return source_map, "\n".join(lines)


# ── Evidence ──────────────────────────────────────────────────

# Per-chunk image context: (ImageRef, source_number)
_ImageEntry = tuple[ImageRef, int]


def _build_evidence(
    chunks: list[RankedChunk],
    source_map: SourceMap,
    token_budget: int,
) -> tuple[str, list[_ImageEntry], int]:
    """Build [EVIDENCE] section, respecting token budget.

    Chunks are sorted by reranked_score (highest first) so the most
    relevant evidence appears first.  Each chunk block includes:
      - Source reference and relevance score
      - Confidence score (if from a provider that returns it)
      - Locality flag (if expanded from adjacent sibling chunks)
      - Sub-query attribution (which decomposed query matched)
      - Converted text content (HTML chunks go through html_converter)

    If the budget is exhausted mid-way, a truncation message shows
    how many chunks were included out of the total.

    Returns (section_text, extracted_images, tokens_used).
    """
    if not chunks:
        section = "[EVIDENCE]\n(no evidence chunks)"
        return section, [], _count_tokens(section)

    # Sort by reranked score descending.
    ordered = sorted(chunks, key=lambda rc: rc.reranked_score, reverse=True)

    lines = ["[EVIDENCE]"]
    tokens_used = _count_tokens("[EVIDENCE]\n")
    included = 0
    all_images: list[_ImageEntry] = []
    seen_texts: set[str] = set()  # dedup identical selected_text across parents

    for rc in ordered:
        c = rc.chunk

        # Skip duplicate content (e.g. multiple parents sharing the same
        # html_text because they matched the same HTML context element).
        if c.selected_text in seen_texts:
            continue
        seen_texts.add(c.selected_text)

        source_num = source_map.get(c.source_url, 0)

        # Build header line.
        header_parts = [f"Source [{source_num}] (relevance: {rc.reranked_score:.2f}"]
        if rc.confidence is not None:
            header_parts.append(f", confidence: {rc.confidence:.2f}")
        header_parts.append(")")
        if rc.is_locality_expanded:
            header_parts.append(" [adjacent context]")
        if rc.source_sub_query:
            header_parts.append(f' [sub-query: "{rc.source_sub_query}"]')
        header = "".join(header_parts)

        # Convert content.
        text, chunk_images = _format_chunk_text(rc)
        for img in chunk_images:
            all_images.append((img, source_num))

        block = f"\n{header}:\n{text}"
        block_tokens = _count_tokens(block)

        if tokens_used + block_tokens > token_budget:
            remaining_count = len(ordered) - included
            lines.append(
                f"\n... (showing {included} of {len(ordered)} chunks "
                f"— remaining {remaining_count} omitted due to response budget)"
            )
            break

        lines.append(block)
        tokens_used += block_tokens
        included += 1

    section = "\n".join(lines)
    return section, all_images, _count_tokens(section)


def _format_chunk_text(rc: RankedChunk) -> tuple[str, list[ImageRef]]:
    """Convert a single chunk's text, handling HTML/markdown surfaces."""
    c = rc.chunk
    if c.surface == "html":
        result = convert_html(c.selected_text, base_url=c.source_url)
        return result.text, result.images
    # Markdown passes through as-is.
    return c.selected_text, []


# ── Images ────────────────────────────────────────────────────


def _build_images(
    image_entries: list[_ImageEntry],
    source_map: SourceMap,
) -> str:
    """Build [IMAGES] section from extracted image metadata."""
    lines = ["[IMAGES]"]
    for img, source_num in image_entries:
        alt = img.alt or "Image"
        entry = f"- [{alt}]({img.url})"
        if img.caption:
            entry += f" — {img.caption}"
        entry += f" (from Source [{source_num}])"
        lines.append(entry)
    return "\n".join(lines)


# ── Expansion trace ──────────────────────────────────────────


def _build_expansion_trace(steps: list[ExpansionStep]) -> str:
    """Build [EXPANSION TRACE] as a parent→child ASCII tree.

    Reconstructs a tree from the flat ``ExpansionStep`` list:
    each step's ``source_url`` is the parent, and its
    ``candidates_expanded`` / ``candidates_failed`` are children.
    URLs sharing a common prefix are truncated for readability.

    Per the patch spec, the trace includes depth and chunk-count
    annotations but omits per-node timing and scores (those belong
    in [STATS]).
    """
    if not steps:
        return ""

    # ── Collect all URLs for common-prefix computation ────
    all_urls: list[str] = []
    for s in steps:
        all_urls.append(s.source_url)
        all_urls.extend(s.candidates_expanded)
        all_urls.extend(s.candidates_failed)

    common_prefix = _common_url_prefix(all_urls)

    def _trunc(url: str) -> str:
        if common_prefix and url.startswith(common_prefix):
            suffix = url[len(common_prefix):]
            return f"/{suffix}" if not suffix.startswith("/") else suffix
        return url

    # ── Build tree nodes ──────────────────────────────────
    # Each node: (url, depth, chunks_added, status, children[])
    # status: "ok" | "failed" | "stopped: <reason>"
    #
    # We walk the steps in order.  Each step produces child nodes
    # under the source_url parent.  We track nodes by URL so deeper
    # iterations can attach to the correct parent.

    class _Node:
        __slots__ = ("url", "depth", "chunks_added", "status", "children")

        def __init__(
            self, url: str, depth: int, chunks_added: int = 0, status: str = "ok",
        ) -> None:
            self.url = url
            self.depth = depth
            self.chunks_added = chunks_added
            self.status = status
            self.children: list[_Node] = []

    # Map url → node for parent lookups.
    node_map: dict[str, _Node] = {}
    seed_url = steps[0].source_url
    root = _Node(seed_url, depth=0)
    node_map[seed_url] = root

    for s in steps:
        parent = node_map.get(s.source_url)
        if parent is None:
            # Source URL not yet in tree — add as root-level.
            parent = _Node(s.source_url, depth=s.depth - 1)
            root.children.append(parent)
            node_map[s.source_url] = parent

        for url in s.candidates_expanded:
            child = _Node(url, depth=s.depth, chunks_added=s.chunks_added)
            parent.children.append(child)
            node_map[url] = child

        for url in s.candidates_failed:
            child = _Node(url, depth=s.depth, status="failed")
            parent.children.append(child)
            node_map[url] = child

    # Tag the stop reason on the last step's parent node.
    last = steps[-1]
    last_parent = node_map.get(last.source_url)
    if last_parent and last.reason:
        # Attach stop annotation to the deepest expanded child,
        # or to the parent itself if nothing was expanded.
        target = last_parent.children[-1] if last_parent.children else last_parent
        if target.status == "ok":
            target.status = f"stopped: {last.reason}"

    # ── Render ASCII tree ─────────────────────────────────
    lines = ["[EXPANSION TRACE]"]
    if common_prefix:
        lines.append(f"Base: {common_prefix}")
    lines.append("")

    def _render(node: _Node, prefix: str, is_last: bool, is_root: bool) -> None:
        # Connector.
        if is_root:
            connector = ""
        else:
            connector = "└── " if is_last else "├── "

        # Annotation.
        parts: list[str] = []
        if not is_root:
            parts.append(f"depth {node.depth}")
        if node.chunks_added > 0:
            parts.append(f"+{node.chunks_added} chunks")
        if node.status == "failed":
            parts.append("failed")
        elif node.status.startswith("stopped:"):
            parts.append(f"[{node.status}]")

        annotation = f" ({', '.join(parts)})" if parts else ""
        label = _trunc(node.url) if not is_root else f"{_trunc(node.url)} (seed)"
        lines.append(f"{prefix}{connector}{label}{annotation}")

        # Recurse into children.
        child_prefix = prefix + ("    " if is_last or is_root else "│   ")
        for i, child in enumerate(node.children):
            _render(child, child_prefix, is_last=(i == len(node.children) - 1), is_root=False)

    _render(root, "", is_last=True, is_root=True)
    return "\n".join(lines)


def _common_url_prefix(urls: list[str]) -> str:
    """Find the longest common URL prefix (up to the last ``/``)."""
    if not urls:
        return ""
    if len(urls) == 1:
        # Return up to last /.
        idx = urls[0].rfind("/")
        return urls[0][:idx + 1] if idx > 0 else ""

    prefix = os.path.commonprefix(urls)
    # Trim to last /.
    idx = prefix.rfind("/")
    if idx > 0:
        return prefix[:idx + 1]
    return ""


# ── Citations ─────────────────────────────────────────────────


def _build_citations(
    citations: list[CitationSpan],
    source_map: SourceMap,
) -> str:
    """Build [CITATIONS] section with verbatim quotes (deduplicated)."""
    lines = ["[CITATIONS]"]
    seen_verbatim: set[str] = set()
    citation_num = 0
    for cit in citations:
        # Truncate very long verbatim text.
        verbatim = cit.verbatim_text
        if len(verbatim) > 300:
            verbatim = verbatim[:297] + "..."
        # Skip duplicate verbatim quotes.
        if verbatim in seen_verbatim:
            continue
        seen_verbatim.add(verbatim)
        citation_num += 1
        source_num = source_map.get(cit.source_url, 0)
        heading_note = f" § {cit.section_heading}" if cit.section_heading else ""
        title = cit.title or "Untitled"
        lines.append(
            f'[{citation_num}] "{verbatim}"\n'
            f"    — {title}, {cit.source_url}{heading_note}"
        )
    return "\n\n".join(lines)


# ── Stats ─────────────────────────────────────────────────────


def _build_stats(result: OrchestrationResult) -> str:
    """Build [STATS] summary section."""
    t = result.timing
    mode_desc = result.mode
    if result.mode == "full_context":
        mode_desc = "full_context (entire corpus fits within token budget)"

    lines = [
        "[STATS]",
        f"Mode: {mode_desc}",
        f"Documents searched: {result.corpus_stats.total_documents}",
        f"Documents matched: {len(result.corpus_stats.documents_matched)}",
        f"Chunks evaluated: {result.corpus_stats.total_parent_chunks}",
        f"Expansion iterations: {result.total_iterations}",
        f"URLs ingested: {result.total_urls_ingested}",
        f"Stop reason: {result.final_decision.reason}",
        (
            f"Total time: {t.total_ms:.0f}ms "
            f"(analysis: {t.query_analysis_ms:.0f}ms, "
            f"retrieval: {t.retrieval_ms:.0f}ms, "
            f"reranking: {t.reranking_ms:.0f}ms, "
            f"expansion: {t.expansion_ms:.0f}ms, "
            f"locality: {t.locality_ms:.0f}ms, "
            f"merge: {t.merge_ms:.0f}ms)"
        ),
    ]

    # Contextual notes.
    if (
        result.total_iterations == 0
        and result.total_urls_ingested <= 1
    ):
        lines.append("Note: No expansion performed (single iteration).")

    if result.expansion_steps:
        last = result.expansion_steps[-1]
        max_depth = settings.max_expansion_depth
        if last.depth >= max_depth:
            lines.append(
                f"Note: Expansion reached maximum depth ({max_depth}). "
                "Results may benefit from a more specific query or a "
                "different seed URL."
            )

    return "\n".join(lines)
