"""Query analysis and decomposition.

Three modes controlled by ``settings.decomposition_mode``:

* ``"llm"``        – structured-output call to the orchestration LLM.
* ``"rule_based"`` – regex/pattern matching (no API calls).
* ``"none"``       – pass-through (original query is the sole sub-query).

If ``"llm"`` is configured but ``orchestration_llm_api_key`` is ``None``,
falls back to ``"rule_based"`` with a warning.

The goal is to turn a complex user query into a list of simpler
sub-queries, each of which retrieves independently.  The merger
then unions the results.  For simple queries, ``"none"`` is fastest.
"""

from __future__ import annotations

import json
import re
import warnings

from openai import AsyncOpenAI

from config import settings
from src.orchestration.models import QueryAnalysis

# ── Prompt ────────────────────────────────────────────────────────

DECOMPOSITION_SYSTEM_PROMPT = """\
You are a query analysis assistant for a retrieval system.
Given a user query, produce a JSON object with:
- "sub_queries": list of 1-4 focused sub-queries that together cover \
the original query's information needs.  If the query is simple and \
focused, return just the original query.  Only decompose when the query \
genuinely contains multiple distinct information needs.
- "query_type": one of "factual", "comparison", "how_to", "exploratory"
- "complexity": one of "simple", "moderate", "complex"
- "key_concepts": list of 2-6 key topic/entity keywords extracted from \
the query

Respond with ONLY the JSON object, no other text."""


# ── Public interface ──────────────────────────────────────────────


async def analyze_query(
    query: str,
    *,
    intent: str | None = None,
    known_context: str | None = None,
    constraints: list[str] | None = None,
    rewrite_context: str | None = None,
) -> QueryAnalysis:
    """Analyse and optionally decompose *query*.

    Parameters
    ----------
    query:
        The original user query.
    intent:
        MCP hint from the reasoning model.  Used as ``query_type``
        directly if provided.
    known_context:
        What the model already knows — used in the LLM prompt to
        avoid retrieving already-known information.
    constraints:
        E.g. ``["must include code examples"]``.  Appended to
        sub-queries where relevant.
    rewrite_context:
        Feedback from a previous expansion loop iteration.  When
        present the LLM receives it as additional instruction to
        produce a *different* decomposition.
    """
    mode = settings.decomposition_mode.lower()

    if mode == "llm":
        if settings.orchestration_llm_api_key is None:
            warnings.warn(
                "decomposition_mode='llm' but orchestration_llm_api_key is "
                "not set.  Falling back to rule_based.",
                stacklevel=2,
            )
            mode = "rule_based"

    if mode == "llm":
        return await _llm_decompose(
            query,
            intent=intent,
            known_context=known_context,
            constraints=constraints,
            rewrite_context=rewrite_context,
        )

    if mode == "rule_based":
        return _rule_based_analyze(
            query, intent=intent, constraints=constraints,
        )

    # mode == "none" or unknown
    return QueryAnalysis(
        original_query=query,
        sub_queries=_apply_constraints([query], constraints),
        query_type=intent or _infer_query_type(query),
        complexity="simple",
        key_concepts=_extract_key_concepts(query),
    )


# ── LLM decomposition ────────────────────────────────────────────


async def _llm_decompose(
    query: str,
    *,
    intent: str | None,
    known_context: str | None,
    constraints: list[str] | None,
    rewrite_context: str | None,
) -> QueryAnalysis:
    """Structured-output call to the orchestration LLM."""
    client = AsyncOpenAI(
        base_url=settings.orchestration_llm_base_url,
        api_key=settings.orchestration_llm_api_key,
    )

    # Build user message.
    user_parts: list[str] = [f"Query: {query}"]
    if known_context:
        user_parts.append(f"Already known context (avoid retrieving this): {known_context}")
    if constraints:
        user_parts.append(f"Constraints: {', '.join(constraints)}")
    if rewrite_context:
        user_parts.append(f"Feedback from previous attempt: {rewrite_context}")

    user_message = "\n".join(user_parts)

    try:
        response = await client.chat.completions.create(
            model=settings.orchestration_llm_model,
            messages=[
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content or ""
        # Strip markdown fences if present.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)

        data = json.loads(raw)

        sub_queries = data.get("sub_queries", [query])
        if not sub_queries:
            sub_queries = [query]
        sub_queries = _apply_constraints(sub_queries, constraints)

        return QueryAnalysis(
            original_query=query,
            sub_queries=sub_queries,
            query_type=intent or data.get("query_type", "factual"),
            complexity=data.get("complexity", "moderate"),
            key_concepts=data.get("key_concepts", _extract_key_concepts(query)),
        )

    except Exception as exc:
        warnings.warn(
            f"LLM decomposition failed ({exc!r}), falling back to rule_based.",
            stacklevel=2,
        )
        return _rule_based_analyze(query, intent=intent, constraints=constraints)


# ── Rule-based decomposition ─────────────────────────────────────

# Pre-compiled patterns.
_VS_PATTERN = re.compile(
    r"(?:^|\s)(.+?)\s+(?:vs\.?|versus|compared?\s+to|difference\s+between)\s+(.+)",
    re.IGNORECASE,
)
_PROS_CONS_PATTERN = re.compile(
    r"(?:pros?\s+(?:and|&)\s+cons?|advantages?\s+(?:and|&)\s+disadvantages?)\s+(?:of\s+)?(.+)",
    re.IGNORECASE,
)
_AND_PATTERN = re.compile(
    r"^(.+?)\s+and\s+(.+?)\s+(?:for|in|with|of)\s+(.+)$",
    re.IGNORECASE,
)
_MULTI_QUESTION = re.compile(
    r"(.+?)\s+and\s+((?:when|how|what|why|where)\s+.+)",
    re.IGNORECASE,
)


def _rule_based_analyze(
    query: str,
    *,
    intent: str | None = None,
    constraints: list[str] | None = None,
) -> QueryAnalysis:
    """Pattern-based query decomposition fallback."""
    sub_queries, query_type = _rule_based_decompose(query)
    sub_queries = _apply_constraints(sub_queries, constraints)

    return QueryAnalysis(
        original_query=query,
        sub_queries=sub_queries,
        query_type=intent or query_type,
        complexity="simple" if len(sub_queries) == 1 else "moderate",
        key_concepts=_extract_key_concepts(query),
    )


def _rule_based_decompose(query: str) -> tuple[list[str], str]:
    """Return (sub_queries, query_type) from pattern matching."""

    # "X vs Y" / "X compared to Y" / "difference between X and Y"
    m = _VS_PATTERN.search(query)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        return [a, b], "comparison"

    # "pros and cons of X"
    m = _PROS_CONS_PATTERN.search(query)
    if m:
        topic = m.group(1).strip()
        return [f"advantages of {topic}", f"disadvantages of {topic}"], "comparison"

    # "X and Y for Z" (distinct noun phrases)
    m = _AND_PATTERN.match(query)
    if m:
        x, y, z = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        return [f"{x} for {z}", f"{y} for {z}"], "exploratory"

    # "how does X work and when should I use it"
    m = _MULTI_QUESTION.search(query)
    if m:
        first_part = m.group(1).strip()
        second_part = m.group(2).strip()
        return [first_part, second_part], "how_to"

    # No match — single query.
    return [query], _infer_query_type(query)


# ── Helpers ───────────────────────────────────────────────────────


def _infer_query_type(query: str) -> str:
    """Heuristic query-type classification from question words."""
    q = query.lower().strip()
    if any(q.startswith(w) for w in ("how do", "how to", "how can", "how does")):
        return "how_to"
    if any(w in q for w in (" vs ", " versus ", "compare", "difference between")):
        return "comparison"
    if any(q.startswith(w) for w in ("what is", "what are", "who is", "who are", "when", "where")):
        return "factual"
    return "exploratory"


def _extract_key_concepts(query: str) -> list[str]:
    """Cheap keyword extraction: drop stop-words and keep meaningful tokens."""
    _STOP = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "have", "has", "had", "having",
        "i", "me", "my", "we", "our", "you", "your", "it", "its",
        "and", "or", "but", "if", "of", "at", "by", "for", "with",
        "to", "from", "in", "on", "up", "out", "off", "into", "onto",
        "about", "as", "that", "this", "these", "those", "what", "which",
        "who", "whom", "how", "when", "where", "why", "not", "no",
        "can", "could", "should", "would", "will", "shall", "may",
        "might", "must", "vs", "versus", "between", "compared",
    }
    tokens = re.findall(r"\b[a-zA-Z0-9_.-]+\b", query.lower())
    concepts = [t for t in tokens if t not in _STOP and len(t) > 1]
    # Deduplicate preserving order.
    seen: set[str] = set()
    result: list[str] = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result[:6]


def _apply_constraints(
    sub_queries: list[str],
    constraints: list[str] | None,
) -> list[str]:
    """Append constraints to each sub-query if provided."""
    if not constraints:
        return sub_queries
    suffix = " " + " ".join(constraints)
    return [sq + suffix for sq in sub_queries]
