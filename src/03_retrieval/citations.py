from __future__ import annotations

from pydantic import BaseModel

from src.retrieval.models import RetrievedChunk


class CitationSpan(BaseModel):
    """A verbatim quote tied to source metadata for citation output."""

    verbatim_text: str
    source_url: str
    section_heading: str | None
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    title: str | None


def extract_citation(
    chunk: RetrievedChunk,
    quote_start: int,
    quote_end: int,
) -> CitationSpan | None:
    """Extract a citation span from a retrieved chunk using absolute char offsets."""
    if quote_start < chunk.char_start:
        return None
    if quote_end > chunk.char_end:
        return None
    if quote_end <= quote_start:
        return None

    relative_start = quote_start - chunk.char_start
    relative_end = quote_end - chunk.char_start
    if relative_start < 0:
        return None
    if relative_end > len(chunk.selected_text):
        return None

    verbatim = chunk.selected_text[relative_start:relative_end]
    if not verbatim:
        return None

    return CitationSpan(
        verbatim_text=verbatim,
        source_url=chunk.source_url,
        section_heading=chunk.section_heading,
        char_start=quote_start,
        char_end=quote_end,
        token_start=chunk.token_start,
        token_end=chunk.token_end,
        title=chunk.title,
    )
