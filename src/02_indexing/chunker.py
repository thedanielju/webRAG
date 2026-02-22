# build_chunks() with private helpers

from __future__ import annotations

"""Chunk construction for indexing.

Core responsibilities:
- Split markdown into parent and child chunks.
- Detect rich-content signals from markdown + HTML context.
- Attach child-level html_text snippets when rich content exists.
- Propagate rich-content metadata to parents so retrieval can choose the
  correct render surface (html vs markdown) at parent return time.
"""

import re
from datetime import datetime
from typing import Callable

from bs4 import BeautifulSoup
from bs4.element import Tag

from config import settings
from src.indexing.models import Chunk, ChunkLevel, RichContentFlags


_TOKEN_COUNTER: Callable[[str], int] | None = None

# tokenization - loads whichever tokenizer is configured (tiktoken or HuggingFace) lazily on first calls, cached in _TOKEN_COUNTER

def _build_token_counter() -> Callable[[str], int]:
    if settings.embedding_tokenizer_kind == "tiktoken":
        import tiktoken
        encoding = tiktoken.get_encoding(settings.embedding_tokenizer_name)
        return lambda text: len(encoding.encode(text))

    if settings.embedding_tokenizer_kind == "huggingface":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(settings.embedding_tokenizer_name)
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    raise ValueError(
        f"Unsupported tokenizer kind: {settings.embedding_tokenizer_kind}. "
        "Expected 'tiktoken' or 'huggingface'."
    )

# size decisions go through token_count - informs chunker if a section is "too big"

def _token_count(text: str) -> int:
    global _TOKEN_COUNTER
    if _TOKEN_COUNTER is None:
        _TOKEN_COUNTER = _build_token_counter()
    return _TOKEN_COUNTER(text)

# html utilities support HTML alignment problem
# markdown and HTML are different representations of the same content with no structural correspondence
# _text_overlap_score counts shared words between a markdown chunk and an HTML element's text to find the best match. 
# _markdown_to_plain_text strips markdown syntax first so the comparison is fair.

def _soup_from_html(text: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(text, "lxml")
    except Exception:
        return BeautifulSoup(text, "html.parser")

# Flag detection (_detect_markdown_flags, _detect_html_flags, _merge_flags) — two passes over each chunk:
# one regex pass over the markdown text, one BS4 pass over the HTML context. Results are merged with OR logic:
# if either pass detects a table, has_table is true. _detect_html_flags now finds the best-matching HTML element first 
# via _best_matching_html_tag, then checks that element for tables, code, definition lists, admonitions.

def _detect_markdown_flags(text: str) -> RichContentFlags:
    has_table = bool(re.search(r"(?m)^\|.+\|\s*$", text))
    has_code = "```" in text

    # Math: Firecrawl strips all LaTeX ($/$$ syntax) from markdown.
    # Detection is HTML-only via _has_math_elements(); always False here.
    # If a non-Firecrawl ingestion path is added later (e.g. raw HTML fetch,
    # local file import), re-add a LaTeX regex here: r"\$\$.+?\$\$|\$.+?\$"
    has_math = False

    # Definition lists: Firecrawl does not produce markdown definition-list
    # syntax (term:\n    def).  Detection is HTML-only via <dl> tags.
    has_definition_list = False

    # Admonitions: Firecrawl renders admonitions as a bare keyword line
    # ("Note", "Warning", etc.) followed by a blank-line-separated body,
    # NOT as blockquotes (> **Note:**).  Match the bare keyword format.
    has_admonition = bool(
        re.search(
            r"(?im)^(Note|Warning|Important|Tip|Caution|Danger|Info|Success|Example|See also|Deprecated since)\s*$",
            text,
        )
    )

    has_steps = bool(
        re.search(r"(?m)^\s*\d+\.\s+", text)
        or re.search(r"(?m)^\s*\d+\)\s+", text)
    )

    return RichContentFlags(
        has_table=has_table,
        has_code=has_code,
        has_math=has_math,
        has_definition_list=has_definition_list,
        has_admonition=has_admonition,
        has_steps=has_steps,
    )


def _has_admonition_classes(node: Tag) -> bool:
    class_pattern = re.compile(
        r"(admonition|note|warning|tip|important|caution|danger|info)", re.IGNORECASE
    )

    for tag in [node] + list(node.find_all(True)):
        classes = tag.get("class") or []
        if any(class_pattern.search(cls) for cls in classes):
            return True
    return False


def _collect_html_candidates(soup: BeautifulSoup) -> list[Tag]:
    candidates: list[Tag] = []
    for name in ("section", "article", "div", "p", "li", "table", "pre", "code", "dl", "blockquote"):
        candidates.extend(soup.find_all(name))
    if candidates:
        return candidates
    return list(soup.find_all(True))


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) > 2}


def _prepare_candidate_tokens(candidates: list[Tag]) -> list[tuple[Tag, set[str]]]:
    prepared: list[tuple[Tag, set[str]]] = []
    for tag in candidates:
        prepared.append((tag, _token_set(tag.get_text(" ", strip=True))))
    return prepared


def _best_matching_html_tag(
    markdown_chunk_text: str,
    soup: BeautifulSoup | None,
    candidates: list[Tag] | None = None,
    prepared_candidates: list[tuple[Tag, set[str]]] | None = None,
) -> Tag | None:
    if soup is None:
        return None

    chunk_tokens = _token_set(_markdown_to_plain_text(markdown_chunk_text))
    if not chunk_tokens:
        return None

    if prepared_candidates is None:
        local_candidates = candidates if candidates is not None else _collect_html_candidates(soup)
        prepared_candidates = _prepare_candidate_tokens(local_candidates)

    best_tag: Tag | None = None
    best_score = 0
    for tag, tag_tokens in prepared_candidates:
        if not tag_tokens:
            continue
        score = len(chunk_tokens & tag_tokens)
        if score > best_score:
            best_score = score
            best_tag = tag

    return best_tag


def _has_math_elements(node: Tag) -> bool:
    """Detect MathJax, KaTeX, or MathML elements inside an HTML context tag."""
    # MathML <math> tags (MathJax assistive layer, native MathML)
    if node.name == "math" or node.find("math") is not None:
        return True
    # MathJax 3 custom elements: <mjx-math>, <mjx-container>, etc.
    for tag in [node] + list(node.find_all(True)):
        if tag.name and tag.name.startswith("mjx-"):
            return True
    # Class-based detection: MathJax wrappers, KaTeX containers,
    # and Sphinx/docutils "math" class on <span> or <div>.
    math_class_pattern = re.compile(r"(MathJax|katex|^math$)", re.IGNORECASE)
    for tag in [node] + list(node.find_all(True)):
        classes = tag.get("class") or []
        if any(math_class_pattern.search(cls) for cls in classes):
            return True
    return False


def _detect_html_flags(
    markdown_chunk_text: str,
    soup: BeautifulSoup | None,
    candidates: list[Tag] | None = None,
    prepared_candidates: list[tuple[Tag, set[str]]] | None = None,
) -> RichContentFlags:
    context_tag = _best_matching_html_tag(
        markdown_chunk_text, soup, candidates, prepared_candidates
    )
    if context_tag is None:
        return RichContentFlags()

    return RichContentFlags(
        has_table=context_tag.name == "table" or context_tag.find("table") is not None,
        has_code=(
            context_tag.name in {"pre", "code"}
            or context_tag.find("pre") is not None
            or context_tag.find("code") is not None
        ),
        has_math=_has_math_elements(context_tag),
        has_definition_list=context_tag.name == "dl" or context_tag.find("dl") is not None,
        has_admonition=_has_admonition_classes(context_tag)
        or context_tag.name == "blockquote"
        or context_tag.find("blockquote") is not None,
        has_steps=context_tag.name == "ol" or context_tag.find("ol") is not None,
    )


def _merge_flags(a: RichContentFlags, b: RichContentFlags) -> RichContentFlags:
    return RichContentFlags(
        has_table=a.has_table or b.has_table,
        has_code=a.has_code or b.has_code,
        has_math=a.has_math or b.has_math,
        has_definition_list=a.has_definition_list or b.has_definition_list,
        has_admonition=a.has_admonition or b.has_admonition,
        has_steps=a.has_steps or b.has_steps,
    )


def _aggregate_flags_from_children(children: list[Chunk]) -> RichContentFlags:
    """Aggregate rich-content flags from child chunks to parent-level truth."""
    return RichContentFlags(
        has_table=any(child.flags.has_table for child in children),
        has_code=any(child.flags.has_code for child in children),
        has_math=any(child.flags.has_math for child in children),
        has_definition_list=any(child.flags.has_definition_list for child in children),
        has_admonition=any(child.flags.has_admonition for child in children),
        has_steps=any(child.flags.has_steps for child in children),
    )


def _should_store_html(flags: RichContentFlags) -> bool:
    return (
        flags.has_table
        or flags.has_code
        or flags.has_math
        or flags.has_definition_list
        or flags.has_admonition
    )


def _markdown_to_plain_text(text: str) -> str:
    plain = re.sub(r"`{1,3}", " ", text)
    plain = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", plain)
    plain = re.sub(r"!\[(.*?)\]\((.*?)\)", r"\1", plain)
    plain = re.sub(r"[*_>#|\\-]+", " ", plain)
    plain = re.sub(r"\s+", " ", plain)
    return plain.strip().lower()


def _text_overlap_score(a: str, b: str) -> int:
    a_tokens = {token for token in re.findall(r"\w+", a.lower()) if len(token) > 2}
    b_tokens = {token for token in re.findall(r"\w+", b.lower()) if len(token) > 2}
    if not a_tokens or not b_tokens:
        return 0
    return len(a_tokens & b_tokens)

# HTML extraction: when _should_store_html is true (table, code, math, definition list, or admonition present),
# finds the best-matching HTML element, then narrows further to the specific flagged sub-element 
# (e.g. the <table> inside it) and returns its raw HTML string - stored in html_text.
#
# NOTE: Multiple child chunks under the same parent section often share the same
# HTML context element, so their html_text values may be identical.  This is
# intentional — deduplication is a retrieval-layer concern (e.g. deduplicate
# before sending to the LLM) rather than an indexing-layer concern.

def _extract_html_snippet(
    markdown_chunk_text: str,
    soup: BeautifulSoup | None,
    flags: RichContentFlags,
    candidates: list[Tag] | None = None,
    prepared_candidates: list[tuple[Tag, set[str]]] | None = None,
) -> str | None:
    if soup is None:
        return None

    context_tag = _best_matching_html_tag(
        markdown_chunk_text, soup, candidates, prepared_candidates
    )
    if context_tag is None:
        return None

    sub_candidates: list[Tag] = []

    if flags.has_table:
        sub_candidates.extend(context_tag.find_all("table"))
        if context_tag.name == "table":
            sub_candidates.append(context_tag)
    if flags.has_code:
        sub_candidates.extend(context_tag.find_all("pre"))
        sub_candidates.extend(context_tag.find_all("code"))
        if context_tag.name in {"pre", "code"}:
            sub_candidates.append(context_tag)
    if flags.has_definition_list:
        sub_candidates.extend(context_tag.find_all("dl"))
        if context_tag.name == "dl":
            sub_candidates.append(context_tag)
    if flags.has_math:
        # Extract clean MathML <math> tags only — these contain semantic
        # markup (<mi>, <mo>, <mn>, <mrow>, etc.) that renderers understand.
        # MathJax visual elements (<mjx-*>) are rendering artifacts and are
        # intentionally excluded from the snippet.
        math_tags = list(context_tag.find_all("math"))
        if context_tag.name == "math":
            math_tags.append(context_tag)
        if math_tags:
            # Return all MathML tags directly, skip overlap scoring.
            return " ".join(str(t) for t in math_tags)
        # Fallback: no <math> tags found — use the context tag itself.
        return str(context_tag)
    if flags.has_admonition:
        class_pattern = re.compile(
            r"(admonition|note|warning|tip|important|caution|danger|info)",
            re.IGNORECASE,
        )
        for tag in [context_tag] + list(context_tag.find_all(True)):
            classes = tag.get("class") or []
            if any(class_pattern.search(cls) for cls in classes):
                sub_candidates.append(tag)
        if context_tag.name == "blockquote":
            sub_candidates.append(context_tag)
        sub_candidates.extend(context_tag.find_all("blockquote"))

    # As a fallback, consider generic containers.
    if not sub_candidates:
        sub_candidates.append(context_tag)

    chunk_plain = _markdown_to_plain_text(markdown_chunk_text)
    best_tag = None
    best_score = 0
    for tag in sub_candidates:
        tag_text = tag.get_text(" ", strip=True)
        score = _text_overlap_score(chunk_plain, tag_text)
        if score > best_score:
            best_score = score
            best_tag = tag

    if best_tag is None:
        return None
    return str(best_tag)


def _enrich_parent_chunks_with_rich_content(
    parents: list[Chunk],
    children: list[Chunk],
    full_html: str | None,
) -> None:
    """Propagate rich-content metadata and HTML surface to parent chunks.

    Parents are the retrieval context unit, so they must carry:
    - aggregated rich-content flags (any child flag -> parent flag),
    - html_text when rich content exists, with child snippet fallback.
    """
    if not parents or not children:
        return

    children_by_parent: dict[object, list[Chunk]] = {}
    for child in children:
        if child.parent_id is None:
            continue
        children_by_parent.setdefault(child.parent_id, []).append(child)

    html_soup = _soup_from_html(full_html) if full_html else None
    html_candidates: list[Tag] | None = None
    prepared_candidates: list[tuple[Tag, set[str]]] | None = None
    if html_soup:
        html_candidates = _collect_html_candidates(html_soup)
        prepared_candidates = _prepare_candidate_tokens(html_candidates)

    for parent in parents:
        parent_children = children_by_parent.get(parent.id, [])
        if not parent_children:
            continue

        parent.flags = _aggregate_flags_from_children(parent_children)
        if not _should_store_html(parent.flags):
            parent.html_text = None
            continue

        parent_html = _extract_html_snippet(
            parent.chunk_text,
            html_soup,
            parent.flags,
            html_candidates,
            prepared_candidates,
        )
        if parent_html is None:
            for child in parent_children:
                if child.html_text:
                    parent_html = child.html_text
                    break
        parent.html_text = parent_html

# Text splitting (_recursive_token_split, _greedy_word_split, _split_keep_separator)
# fallback chain for oversized text. Tries splitting on \n\n, then \n, then . , then  . If even single words exceed the token limit 
# (rare but possible with code), _greedy_word_split handles it. _split_keep_separator keeps the separator attached to the preceding to not lose paragraph breaks.

def _split_keep_separator(text: str, separator: str) -> list[str]:
    if separator not in text:
        return [text]

    parts = text.split(separator)
    result: list[str] = []
    for index, part in enumerate(parts):
        if index < len(parts) - 1:
            result.append(part + separator)
        else:
            result.append(part)
    return result


def _greedy_word_split(text: str, max_tokens: int) -> list[str]:
    words = text.split(" ")
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = (" ".join(current + [word])).strip()
        if candidate and _token_count(candidate) <= max_tokens:
            current.append(word)
            continue

        if current:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            chunks.append(word.strip())

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _recursive_token_split(text: str, max_tokens: int) -> list[str]:
    if _token_count(text) <= max_tokens:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]

    def _split_inner(value: str, level: int) -> list[str]:
        if _token_count(value) <= max_tokens:
            return [value]
        if level >= len(separators):
            return _greedy_word_split(value, max_tokens)

        pieces = _split_keep_separator(value, separators[level])
        if len(pieces) == 1:
            return _split_inner(value, level + 1)

        out: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            out.extend(_split_inner(piece, level + 1))
        return out

    return [chunk for chunk in _split_inner(text, 0) if chunk.strip()]

# Section finding (_find_sections, _paragraph_blocks)
#  _find_sections walks the markdown looking for ## boundaries (falls back to #, then treats the whole doc as one section)
#  _paragraph_blocks splits a section's text on blank lines and returns each paragraph with its character offsets relative to the section start

def _paragraph_blocks(text: str) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []

    pieces = re.split(r"\n\s*\n+", text)
    blocks: list[tuple[str, int, int]] = []
    search_pos = 0

    for piece in pieces:
        if not piece.strip():
            continue
        start = text.find(piece, search_pos)
        if start == -1:
            continue
        end = start + len(piece)
        blocks.append((piece, start, end))
        search_pos = end

    return blocks

def _find_sections(markdown: str) -> list[tuple[str | None, str, int, int]]:
    h2_matches = list(re.finditer(r"(?m)^##\s+.+$", markdown))
    if h2_matches:
        boundaries = h2_matches
    else:
        boundaries = list(re.finditer(r"(?m)^#\s+.+$", markdown))

    if not boundaries:
        return [(None, markdown, 0, len(markdown))]

    sections: list[tuple[str | None, str, int, int]] = []
    for index, match in enumerate(boundaries):
        start = match.start()
        end = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(markdown)
        text = markdown[start:end]
        heading = match.group(0).strip()
        sections.append((heading, text, start, end))

    return sections


# Parent construction (_parent_chunks_from_sections)
# walks sections, accumulates paragraphs greedily until adding 
# the next one would exceed parent_max_tokens, then flushes a parent chunk. 
# Each parent carries the section heading, absolute char offsets, and all the denormalized metadata.

def _parent_chunks_from_sections(
    markdown: str,
    source_url: str,
    fetched_at: datetime,
    title: str | None,
    depth: int,
) -> list[Chunk]:
    parent_chunks: list[Chunk] = []
    parent_index = 0

    for heading, section_text, section_start, _section_end in _find_sections(markdown):
        paragraph_blocks = _paragraph_blocks(section_text)
        if not paragraph_blocks:
            continue

        current_texts: list[str] = []
        current_start: int | None = None
        current_end: int | None = None

        for block_text, rel_start, rel_end in paragraph_blocks:
            abs_start = section_start + rel_start
            abs_end = section_start + rel_end

            candidate_text = "\n\n".join(current_texts + [block_text]).strip()
            if current_texts and _token_count(candidate_text) > settings.parent_max_tokens:
                full_text = "\n\n".join(current_texts).strip()
                parent_chunks.append(
                    Chunk(
                        chunk_level=ChunkLevel.PARENT,
                        chunk_index=parent_index,
                        section_heading=heading,
                        chunk_text=full_text,
                        char_start=current_start or 0,
                        char_end=current_end or 0,
                        source_url=source_url,
                        fetched_at=fetched_at,
                        depth=depth,
                        title=title,
                    )
                )
                parent_index += 1
                current_texts = [block_text]
                current_start = abs_start
                current_end = abs_end
                continue

            if not current_texts:
                current_start = abs_start
            current_texts.append(block_text)
            current_end = abs_end

        if current_texts:
            full_text = "\n\n".join(current_texts).strip()
            parent_chunks.append(
                Chunk(
                    chunk_level=ChunkLevel.PARENT,
                    chunk_index=parent_index,
                    section_heading=heading,
                    chunk_text=full_text,
                    char_start=current_start or 0,
                    char_end=current_end or 0,
                    source_url=source_url,
                    fetched_at=fetched_at,
                    depth=depth,
                    title=title,
                )
            )
            parent_index += 1

    if parent_chunks:
        return parent_chunks

    return [
        Chunk(
            chunk_level=ChunkLevel.PARENT,
            chunk_index=0,
            section_heading=None,
            chunk_text=markdown,
            char_start=0,
            char_end=len(markdown),
            source_url=source_url,
            fetched_at=fetched_at,
            depth=depth,
            title=title,
        )
    ]

# Child construction (_child_chunks_from_parent)
# for each paragraph in a parent, splits to child_target_tokens, detects flags, extracts HTML snippet if needed, and creates 
# child chunks pointing back to the parent via parent_id.

def _child_chunks_from_parent(
    parent: Chunk,
    html: str | None,
) -> list[Chunk]:
    paragraph_blocks = _paragraph_blocks(parent.chunk_text)
    children: list[Chunk] = []
    child_index = 0
    html_soup = _soup_from_html(html) if html else None
    html_candidates: list[Tag] | None = None
    prepared_candidates: list[tuple[Tag, set[str]]] | None = None
    if html_soup:
        # Scope matching to the parent's best HTML context, then reuse pre-tokenized
        # candidate text across all child chunks to avoid repeated expensive scans.
        global_candidates = _collect_html_candidates(html_soup)
        global_prepared = _prepare_candidate_tokens(global_candidates)
        parent_context = _best_matching_html_tag(
            parent.chunk_text,
            html_soup,
            global_candidates,
            global_prepared,
        )
        scoped_root = parent_context if parent_context is not None else html_soup
        html_candidates = _collect_html_candidates(scoped_root)
        prepared_candidates = _prepare_candidate_tokens(html_candidates)

    for block_text, rel_start, _rel_end in paragraph_blocks:
        abs_start = parent.char_start + rel_start

        split_texts = _recursive_token_split(block_text, settings.child_target_tokens)
        split_search = 0

        for split_text in split_texts:
            local_pos = block_text.find(split_text, split_search)
            if local_pos == -1:
                local_pos = split_search

            chunk_start = abs_start + local_pos
            chunk_end = chunk_start + len(split_text)
            split_search = local_pos + len(split_text)

            markdown_flags = _detect_markdown_flags(split_text)
            html_flags = _detect_html_flags(
                split_text, html_soup, html_candidates, prepared_candidates
            )
            flags = _merge_flags(markdown_flags, html_flags)

            html_text = None
            if _should_store_html(flags):
                html_text = _extract_html_snippet(
                    split_text,
                    html_soup,
                    flags,
                    html_candidates,
                    prepared_candidates,
                )

            children.append(
                Chunk(
                    parent_id=parent.id,
                    chunk_level=ChunkLevel.CHILD,
                    chunk_index=child_index,
                    section_heading=parent.section_heading,
                    chunk_text=split_text,
                    html_text=html_text,
                    flags=flags,
                    char_start=chunk_start,
                    char_end=chunk_end,
                    source_url=parent.source_url,
                    fetched_at=parent.fetched_at,
                    depth=parent.depth,
                    title=parent.title,
                )
            )
            child_index += 1

    return children

# Fallback path (_fallback_parent_grouping) 
# for documents with no headings: the whole document becomes one big parent
# children are split from it normally
# then every 4 children get grouped under a synthetic parent
# this gives retrieval the same parent-child structure to work with regardless of document quality.

def _fallback_parent_grouping(
    children: list[Chunk],
    source_url: str,
    fetched_at: datetime,
    title: str | None,
    depth: int,
) -> list[Chunk]:
    if not children:
        return []

    grouped_parents: list[Chunk] = []
    parent_index = 0

    for start in range(0, len(children), 4):
        group = children[start : start + 4]
        parent = Chunk(
            chunk_level=ChunkLevel.PARENT,
            chunk_index=parent_index,
            section_heading=None,
            chunk_text="\n\n".join(child.chunk_text for child in group),
            char_start=group[0].char_start,
            char_end=group[-1].char_end,
            source_url=source_url,
            fetched_at=fetched_at,
            depth=depth,
            title=title,
        )
        for child in group:
            child.parent_id = parent.id
        grouped_parents.append(parent)
        parent_index += 1

    return grouped_parents

# build_chunks()
# the entry point that wires it all together
# checks for heading structure, routes to heading path or fallback path, returns (parents, children) as a tuple.

def build_chunks(
    markdown: str,
    html: str | None,
    source_url: str,
    fetched_at: datetime,
    title: str | None,
    depth: int,
) -> tuple[list[Chunk], list[Chunk]]:
    """Split one document into parent+child chunks and enrich parent metadata.

    Returns:
    - parents: context units returned by retrieval.
    - children: embedding/search units linked to parents by parent_id.
    """
    if not markdown:
        return [], []

    sections = _find_sections(markdown)
    has_heading_structure = any(section_heading is not None for section_heading, *_ in sections)

    if not has_heading_structure:
        base_parent = Chunk(
            chunk_level=ChunkLevel.PARENT,
            chunk_index=0,
            section_heading=None,
            chunk_text=markdown,
            char_start=0,
            char_end=len(markdown),
            source_url=source_url,
            fetched_at=fetched_at,
            depth=depth,
            title=title,
        )
        fallback_children = _child_chunks_from_parent(base_parent, html)
        fallback_parents = _fallback_parent_grouping(
            fallback_children,
            source_url=source_url,
            fetched_at=fetched_at,
            title=title,
            depth=depth,
        )
        _enrich_parent_chunks_with_rich_content(fallback_parents, fallback_children, html)
        return fallback_parents, fallback_children

    parent_chunks = _parent_chunks_from_sections(
        markdown=markdown,
        source_url=source_url,
        fetched_at=fetched_at,
        title=title,
        depth=depth,
    )

    child_chunks: list[Chunk] = []
    for parent in parent_chunks:
        child_chunks.extend(_child_chunks_from_parent(parent, html))

    _enrich_parent_chunks_with_rich_content(parent_chunks, child_chunks, html)

    return parent_chunks, child_chunks
