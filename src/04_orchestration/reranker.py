"""Modular reranker abstraction.

Supports ZeroEntropy (zerank-2), Cohere (rerank-v3.5), Jina (HTTP API),
and a passthrough ("none") provider.  All providers normalise output to
``list[RerankResult]`` sorted by ``relevance_score`` descending.

Design notes:
  - Provider SDKs are imported lazily (inside each function) so the
    system starts without installing unused provider packages.
  - Any provider failure falls back to passthrough with a warning,
    ensuring retrieval never hard-fails on a reranker outage.
"""

from __future__ import annotations

import logging
import warnings

from config import settings
from src.orchestration.models import RerankResult

logger = logging.getLogger(__name__)


# ── Public interface ──────────────────────────────────────────────


async def rerank(
    query: str,
    passages: list[str],
    *,
    instruction: str | None = None,
    top_n: int | None = None,
    original_scores: list[float] | None = None,
) -> list[RerankResult]:
    """Rerank *passages* against *query* using the configured provider.

    Parameters
    ----------
    query:
        The search query.
    passages:
        Texts to rerank.
    instruction:
        Optional instruction hint.  Passed natively to ZeroEntropy;
        prepended to the query for Cohere/Jina; ignored for "none".
    top_n:
        Maximum results to return.  Defaults to ``settings.reranker_top_n``.
    original_scores:
        Required when ``reranker_provider == "none"`` — the retrieval
        scores used as-is for passthrough ordering.

    Returns
    -------
    list[RerankResult]
        Sorted by ``relevance_score`` descending.  Length ≤ *top_n*.
    """
    if not passages:
        return []

    effective_top_n = top_n or settings.reranker_top_n
    provider = settings.reranker_provider.lower()

    logger.debug("rerank: %d passages sent to '%s' (top_n=%d)",
                 len(passages), provider, effective_top_n)

    dispatch = {
        "zeroentropy": _rerank_zeroentropy,
        "cohere": _rerank_cohere,
        "jina": _rerank_jina,
    }

    if provider == "none":
        return _passthrough_rerank(passages, original_scores, effective_top_n)

    handler = dispatch.get(provider)
    if handler is None:
        warnings.warn(
            f"Unknown reranker_provider '{provider}', falling back to passthrough.",
            stacklevel=2,
        )
        return _passthrough_rerank(passages, original_scores, effective_top_n)

    try:
        return await handler(query, passages, instruction, effective_top_n)
    except Exception as exc:
        warnings.warn(
            f"Reranker '{provider}' failed ({exc!r}), falling back to passthrough.",
            stacklevel=2,
        )
        return _passthrough_rerank(passages, original_scores, effective_top_n)


# ── ZeroEntropy ───────────────────────────────────────────────────


async def _rerank_zeroentropy(
    query: str,
    passages: list[str],
    instruction: str | None,
    top_n: int,
) -> list[RerankResult]:
    """ZeroEntropy zerank-2 reranking.

    Uses ``AsyncZeroEntropy().models.rerank()`` with keyword-only arguments.
    ZeroEntropy returns calibrated 0–1 scores.
    """
    from zeroentropy import AsyncZeroEntropy

    client = AsyncZeroEntropy(api_key=settings.reranker_api_key)

    response = await client.models.rerank(
        model=settings.reranker_model,
        query=query,
        documents=passages,
        top_n=top_n,
    )

    results = [
        RerankResult(
            index=r.index,
            relevance_score=r.relevance_score,
            # confidence is not in the current SDK (v0.1.0a8).
            confidence=getattr(r, "confidence", None),
        )
        for r in response.results
    ]

    return sorted(results, key=lambda r: r.relevance_score, reverse=True)


# ── Cohere ────────────────────────────────────────────────────────


async def _rerank_cohere(
    query: str,
    passages: list[str],
    instruction: str | None,
    top_n: int,
) -> list[RerankResult]:
    """Cohere Rerank v3.5.

    Uses ``cohere.AsyncClientV2().rerank()``.  Instruction is prepended
    to the query string (best-effort instruction injection).
    """
    import cohere

    client = cohere.AsyncClientV2(api_key=settings.reranker_api_key)

    effective_query = f"{instruction}. {query}" if instruction else query

    response = await client.rerank(
        model=settings.reranker_model,
        query=effective_query,
        documents=passages,
        top_n=top_n,
    )

    results = [
        RerankResult(
            index=r.index,
            relevance_score=r.relevance_score,
        )
        for r in response.results
    ]

    return sorted(results, key=lambda r: r.relevance_score, reverse=True)


# ── Jina ──────────────────────────────────────────────────────────


async def _rerank_jina(
    query: str,
    passages: list[str],
    instruction: str | None,
    top_n: int,
) -> list[RerankResult]:
    """Jina Reranker v2 via HTTP API.

    Uses ``httpx.AsyncClient`` to POST to Jina's rerank endpoint.
    Instruction is prepended to the query string.
    """
    import httpx

    effective_query = f"{instruction}. {query}" if instruction else query

    payload = {
        "model": settings.reranker_model,
        "query": effective_query,
        "documents": passages,
        "top_n": top_n,
    }

    headers = {
        "Authorization": f"Bearer {settings.reranker_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.jina.ai/v1/rerank",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    results = [
        RerankResult(
            index=r["index"],
            relevance_score=r["relevance_score"],
        )
        for r in data["results"]
    ]

    return sorted(results, key=lambda r: r.relevance_score, reverse=True)


# ── Passthrough ───────────────────────────────────────────────────


def _passthrough_rerank(
    passages: list[str],
    original_scores: list[float] | None,
    top_n: int,
) -> list[RerankResult]:
    """No-op reranker for ``reranker_provider="none"``.

    Preserves the original ordering and scores.  When *original_scores*
    is ``None``, assigns monotonically decreasing scores from 1.0.
    """
    if original_scores is None:
        n = len(passages)
        original_scores = [1.0 - (i / max(n, 1)) for i in range(n)]

    results = [
        RerankResult(index=i, relevance_score=s)
        for i, s in enumerate(original_scores)
    ]

    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results[:top_n]
