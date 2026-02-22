"""HTML surface → readable plain text conversion.

Stateless functions that convert HTML elements into text the
reasoning model can parse.  Called by ``formatter.py`` when
processing chunks whose ``surface == "html"``.

Why this exists:
  Retrieval returns chunks in either markdown or HTML "surface"
  format (chosen per-parent by the indexer based on rich-content
  flags).  HTML is kept when the source page has tables, code blocks,
  math, or images that markdown can't faithfully represent.  But raw
  HTML is noisy for LLMs — tags, attributes, and entities waste
  tokens and confuse reasoning.  This module converts the HTML into
  clean, readable plain text while preserving semantic structure.

Conversion pipeline:
  1. Parse with BeautifulSoup + lxml.
  2. Walk the DOM tree top-down.
  3. Dispatch recognised elements to specialised converters
     (table → pipe-delimited, code → fenced block, math → LaTeX, etc.).
  4. Fall through for generic block/inline elements, preserving text.
  5. Clean up excessive whitespace.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


# ── Public data types ─────────────────────────────────────────


@dataclass
class ImageRef:
    """Image metadata extracted from an HTML chunk."""

    url: str
    alt: str | None = None
    caption: str | None = None


@dataclass
class ConversionResult:
    """Output of :func:`convert_html`."""

    text: str
    images: list[ImageRef] = field(default_factory=list)


# ── Top-level entry point ────────────────────────────────────


def convert_html(html: str, *, base_url: str | None = None) -> ConversionResult:
    """Convert an HTML fragment to model-readable plain text.

    Parameters
    ----------
    html:
        Raw HTML string (typically ``RetrievedChunk.selected_text``
        when ``surface == "html"``).
    base_url:
        Used to resolve relative ``<img src>`` URLs.

    Returns
    -------
    ConversionResult
        Converted text and any extracted image metadata.
    """
    soup = BeautifulSoup(html, "lxml")
    images: list[ImageRef] = []
    text = _walk(soup, images=images, base_url=base_url)
    text = _clean_whitespace(text)
    return ConversionResult(text=text, images=images)


# ── DOM walker ────────────────────────────────────────────────


def _walk(
    node: Tag | NavigableString,
    *,
    images: list[ImageRef],
    base_url: str | None,
) -> str:
    """Recursively walk the DOM, dispatching to element converters.

    The walker checks the tag name against a dispatch table of
    specialised converters.  If no converter matches, it recurses
    into children and wraps block-level elements in paragraph breaks
    so the plain-text output stays readable.
    """
    if isinstance(node, NavigableString):
        return str(node)

    tag_name = node.name if isinstance(node, Tag) else ""

    # Dispatch to specific converters.
    if tag_name == "table":
        return _convert_table(node)
    if tag_name == "pre":
        return _convert_code_block(node)
    if tag_name == "code" and (not node.parent or node.parent.name != "pre"):
        # Inline code — wrap in backticks.
        return f"`{node.get_text()}`"
    if tag_name == "math":
        return _convert_math(node)
    if tag_name == "dl":
        return _convert_definition_list(node)
    if tag_name == "img":
        return _convert_image(node, images=images, base_url=base_url)
    if tag_name == "div" and _is_admonition(node):
        return _convert_admonition(node, images=images, base_url=base_url)
    if tag_name in ("br",):
        return "\n"
    if tag_name in ("hr",):
        return "\n---\n"

    # Block-level elements: add paragraph breaks.
    block_tags = {
        "p", "div", "section", "article", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "figure", "figcaption",
        "header", "footer", "nav", "main", "aside",
    }

    parts: list[str] = []
    for child in node.children:
        parts.append(_walk(child, images=images, base_url=base_url))

    joined = "".join(parts)

    if tag_name in block_tags:
        return f"\n\n{joined.strip()}\n\n"

    return joined


# ── Element converters ────────────────────────────────────────


def _convert_table(table: Tag) -> str:
    """Convert ``<table>`` to a pipe-delimited text table.

    Produces a GitHub-flavoured Markdown table with a separator row
    after the header.  Columns are padded to align pipes visually.
    """
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(["th", "td"]):
            cells.append(cell.get_text(strip=True))
        if cells:
            rows.append(cells)

    if not rows:
        return table.get_text()

    # Normalise column count.
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    # Compute column widths.
    widths = [
        max(len(rows[i][j]) for i in range(len(rows)))
        for j in range(max_cols)
    ]
    # Minimum width of 3 for readability.
    widths = [max(w, 3) for w in widths]

    lines: list[str] = []
    for idx, row in enumerate(rows):
        line = "| " + " | ".join(
            cell.ljust(widths[j]) for j, cell in enumerate(row)
        ) + " |"
        lines.append(line)
        if idx == 0:
            sep = "| " + " | ".join("-" * w for w in widths) + " |"
            lines.append(sep)

    return "\n\n" + "\n".join(lines) + "\n\n"


def _convert_code_block(pre: Tag) -> str:
    """Convert ``<pre><code>`` to a fenced code block.

    Detects the language from class attributes like
    ``class="language-python"`` or ``class="highlight-js"``.
    """
    code_tag = pre.find("code")
    if code_tag and isinstance(code_tag, Tag):
        lang = _detect_language(code_tag) or _detect_language(pre)
        text = code_tag.get_text()
    else:
        lang = _detect_language(pre)
        text = pre.get_text()

    # Strip a single leading/trailing newline if present.
    text = text.strip("\n")
    lang_hint = lang or ""
    return f"\n\n```{lang_hint}\n{text}\n```\n\n"


def _detect_language(tag: Tag) -> str | None:
    """Extract language hint from ``class`` attributes."""
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    for cls in classes:
        for prefix in ("language-", "highlight-", "lang-"):
            if cls.startswith(prefix):
                return cls[len(prefix):]
    return None


def _convert_math(math_tag: Tag) -> str:
    """Convert ``<math>`` (MathML) to LaTeX notation.

    Three-tier fallback strategy:
      1. Use the ``alttext`` attribute if present — many renderers
         (e.g. MathJax, KaTeX) embed the original LaTeX source here.
      2. Attempt structural MathML → LaTeX conversion via
         ``_mathml_to_latex()`` for common elements (fractions,
         superscripts, subscripts, sqrt, matrices).
      3. Extract plain text as a last resort.
    """
    # Prefer alttext attribute (many renderers embed the LaTeX source).
    alttext = math_tag.get("alttext") or math_tag.get("altText")
    if alttext:
        display = math_tag.get("display") == "block"
        if display:
            return f"\n\n$${alttext}$$\n\n"
        return f"${alttext}$"

    # Attempt basic structural conversion.
    latex = _mathml_to_latex(math_tag)
    if latex:
        display = math_tag.get("display") == "block"
        if display:
            return f"\n\n$${latex}$$\n\n"
        return f"${latex}$"

    # Fallback: extract text content.
    return math_tag.get_text()


def _mathml_to_latex(node: Tag) -> str | None:
    """Best-effort MathML → LaTeX for common elements."""
    try:
        return _mathml_node(node)
    except Exception:
        logger.debug("MathML conversion failed, falling back to text", exc_info=True)
        return None


def _mathml_node(node: Tag | NavigableString) -> str:
    r"""Recursively convert a single MathML node to LaTeX.

    Handles common MathML elements:
      - ``<mfrac>``    → ``\frac{num}{den}``
      - ``<msup>``     → ``base^{exp}``
      - ``<msub>``     → ``base_{sub}``
      - ``<msqrt>``    → ``\sqrt{inner}``
      - ``<mover>``    → ``\overset{over}{base}``
      - ``<munder>``   → ``\underset{under}{base}``
      - ``<mtable>``   → ``\begin{matrix}...\end{matrix}``
      - ``<mi/mn/mo>`` → literal text

    Unknown tags fall through to recursive child concatenation.
    """
    if isinstance(node, NavigableString):
        return str(node).strip()

    tag = node.name
    children = [c for c in node.children if not (isinstance(c, NavigableString) and not c.strip())]

    if tag in ("math", "mrow", "mstyle", "mpadded"):
        return "".join(_mathml_node(c) for c in children)
    if tag == "mi" or tag == "mn" or tag == "mo" or tag == "mtext":
        return node.get_text(strip=True)
    if tag == "mfrac":
        if len(children) >= 2:
            num = _mathml_node(children[0])
            den = _mathml_node(children[1])
            return rf"\frac{{{num}}}{{{den}}}"
    if tag == "msup":
        if len(children) >= 2:
            base = _mathml_node(children[0])
            exp = _mathml_node(children[1])
            return rf"{base}^{{{exp}}}"
    if tag == "msub":
        if len(children) >= 2:
            base = _mathml_node(children[0])
            sub = _mathml_node(children[1])
            return rf"{base}_{{{sub}}}"
    if tag == "msubsup":
        if len(children) >= 3:
            base = _mathml_node(children[0])
            sub = _mathml_node(children[1])
            sup = _mathml_node(children[2])
            return rf"{base}_{{{sub}}}^{{{sup}}}"
    if tag == "msqrt":
        inner = "".join(_mathml_node(c) for c in children)
        return rf"\sqrt{{{inner}}}"
    if tag == "mover":
        if len(children) >= 2:
            base = _mathml_node(children[0])
            over = _mathml_node(children[1])
            return rf"\overset{{{over}}}{{{base}}}"
    if tag == "munder":
        if len(children) >= 2:
            base = _mathml_node(children[0])
            under = _mathml_node(children[1])
            return rf"\underset{{{under}}}{{{base}}}"
    if tag == "mtable":
        rows_out: list[str] = []
        for mtr in node.find_all("mtr", recursive=False):
            cells = [_mathml_node(mtd) for mtd in mtr.find_all("mtd", recursive=False)]
            rows_out.append(" & ".join(cells))
        return r"\begin{matrix}" + r" \\ ".join(rows_out) + r"\end{matrix}"

    # Fallback for unknown tags.
    return "".join(_mathml_node(c) for c in children)


def _convert_definition_list(dl: Tag) -> str:
    """Convert ``<dl>`` to labelled plain text."""
    parts: list[str] = []
    for child in dl.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "dt":
            parts.append(f"\n\n**{child.get_text(strip=True)}**: ")
        elif child.name == "dd":
            parts.append(child.get_text(strip=True))
    return "".join(parts) + "\n\n"


def _convert_image(
    img: Tag,
    *,
    images: list[ImageRef],
    base_url: str | None,
) -> str:
    """Extract image metadata and return an inline placeholder.

    Images are collected into the ``images`` accumulator so the
    formatter can build a separate [IMAGES] section.  In the text
    flow, a brief ``[alt text]`` placeholder keeps the reading
    position clear without wasting tokens on data URIs.
    """
    src = img.get("src", "")
    if not src:
        return ""

    if base_url:
        src = urljoin(base_url, str(src))
    else:
        src = str(src)

    alt = img.get("alt")
    alt = str(alt).strip() if alt else None

    # Look for a nearby <figcaption>.
    caption: str | None = None
    parent = img.parent
    if parent and isinstance(parent, Tag) and parent.name == "figure":
        figcap = parent.find("figcaption")
        if figcap:
            caption = figcap.get_text(strip=True)

    images.append(ImageRef(url=src, alt=alt, caption=caption))

    # Return a brief inline placeholder so the text flow stays readable.
    label = alt or "image"
    return f"[{label}]"


# ── Admonition detection ──────────────────────────────────────
#
# Admonitions are styled callout boxes used in documentation frameworks
# (Sphinx, MkDocs, Docusaurus, etc.).  They're rendered as <div> elements
# with CSS classes like "admonition warning" or "note".  We detect them
# by regex and convert to labelled text blocks with emoji prefixes.

_ADMONITION_RE = re.compile(
    r"\b(admonition|warning|note|tip|caution|danger|important|hint|attention|error)\b",
    re.IGNORECASE,
)

_ADMONITION_ICONS: dict[str, str] = {
    "warning": "\u26a0\ufe0f WARNING",
    "caution": "\u26a0\ufe0f CAUTION",
    "danger": "\u26a0\ufe0f DANGER",
    "error": "\u274c ERROR",
    "note": "\u2139\ufe0f NOTE",
    "tip": "\U0001f4a1 TIP",
    "hint": "\U0001f4a1 HINT",
    "important": "\u2757 IMPORTANT",
    "attention": "\u2757 ATTENTION",
}


def _is_admonition(div: Tag) -> bool:
    """Detect admonition-style ``<div>`` elements."""
    classes = div.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    return any(_ADMONITION_RE.search(cls) for cls in classes)


def _convert_admonition(
    div: Tag,
    *,
    images: list[ImageRef],
    base_url: str | None,
) -> str:
    """Convert an admonition ``<div>`` to a labelled text block.

    When a div has multiple classes (e.g. ``class="admonition warning"``),
    we prefer the specific type ("warning") over the generic ("admonition")
    to get the right emoji icon.
    """
    classes = div.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()

    kind = "NOTE"
    # Iterate all classes; prefer a specific type (warning, note, etc.)
    # over the generic "admonition" class name.
    found_generic = False
    for cls in classes:
        m = _ADMONITION_RE.search(cls)
        if m:
            key = m.group(1).lower()
            if key == "admonition":
                found_generic = True
                continue
            kind = _ADMONITION_ICONS.get(key, key.upper())
            break
    # If only "admonition" was found (no specific type), keep default.

    # Get text content (recurse through walker for nested elements).
    inner_parts: list[str] = []
    for child in div.children:
        inner_parts.append(_walk(child, images=images, base_url=base_url))
    inner = "".join(inner_parts).strip()

    return f"\n\n{kind}: {inner}\n\n"


# ── Whitespace cleanup ───────────────────────────────────────


def _clean_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trim trailing whitespace."""
    # Collapse 3+ consecutive newlines to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()
