from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A context chunk returned by retrieval for downstream orchestration."""

    # Identity
    chunk_id: UUID
    document_id: UUID
    parent_id: UUID | None
    source_url: str
    title: str | None

    # Content selected by retrieval surface rules.
    selected_text: str
    surface: Literal["html", "markdown"]
    section_heading: str | None
    chunk_index: int

    # Offsets in the original document markdown.
    char_start: int
    char_end: int
    token_start: int
    token_end: int

    # Scoring.
    score: float
    raw_similarity: float | None
    depth: int

    # Rich-content flags denormalized from indexed chunks.
    has_table: bool
    has_code: bool
    has_math: bool
    has_definition_list: bool
    has_admonition: bool
    has_steps: bool

    # Metadata
    fetched_at: datetime


class CorpusStats(BaseModel):
    """Corpus-level retrieval stats used by orchestration heuristics."""

    total_documents: int
    total_parent_chunks: int
    total_tokens: int
    documents_matched: list[str]


class TimingInfo(BaseModel):
    """Timing breakdown for retrieval observability."""

    embed_ms: float
    search_ms: float
    total_ms: float


class RetrievalResult(BaseModel):
    """Top-level retrieval output contract consumed by orchestration."""

    mode: Literal["full_context", "chunk"]
    chunks: list[RetrievedChunk]
    query_embedding: list[float]
    corpus_stats: CorpusStats
    timing: TimingInfo

    @property
    def is_empty(self) -> bool:
        """True when retrieval found no chunks."""
        return len(self.chunks) == 0

    @property
    def top_score(self) -> float:
        """Maximum score among returned chunks (0.0 for empty results)."""
        if not self.chunks:
            return 0.0
        return max(chunk.score for chunk in self.chunks)

    @property
    def source_urls(self) -> list[str]:
        """Unique source URLs in first-seen order."""
        seen: set[str] = set()
        ordered_urls: list[str] = []
        for chunk in self.chunks:
            if chunk.source_url in seen:
                continue
            seen.add(chunk.source_url)
            ordered_urls.append(chunk.source_url)
        return ordered_urls


class PersistedLinkCandidate(BaseModel):
    """A link candidate row from the link_candidates table.

    Rows start URL-only (title/description/enriched_at NULL) and are
    populated later via enrich_link_candidates() when orchestration
    needs /map metadata for link scoring.
    """

    id: UUID
    source_document_id: UUID
    source_url: str
    target_url: str
    title: str | None
    description: str | None
    discovered_at: datetime
    enriched_at: datetime | None
    depth: int
