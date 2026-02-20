from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from openai import OpenAI

from config import settings
from src.indexing.models import Chunk

# ────────────────────────────────────────────────────────────────
# embedder.py — Embedding + tokenization for the indexing layer
#
# Responsibilities:
#   1. embed_texts()  — convert a list of chunk texts into vectors
#   2. annotate_token_offsets() — stamp token_start/token_end onto
#      Chunk objects using a char→token boundary map
#   3. token_count()  — count tokens for a text string
#
# Concurrency model (embed_texts):
#   Texts are split into batches of EMBEDDING_BATCH_SIZE (default 256)
#   and dispatched to the embedding API via a ThreadPoolExecutor with
#   at most EMBEDDING_MAX_WORKERS concurrent threads (default 4).
#
#   Why batching:
#     A single API call with 1200 texts takes ~70s round-trip due to
#     payload serialization and server-side processing.  Splitting
#     into 5-6 batches of 256 and sending them concurrently cuts
#     wall-clock time to ~10-15s — the batches overlap on the wire.
#
#   Why default 4 workers:
#     OpenAI enforces per-minute rate limits (RPM) that vary by API
#     tier.  4 concurrent requests is safe for all paid tiers and
#     most trial/free tiers.  Higher-tier plans or local servers
#     (Ollama, LM Studio) can safely increase EMBEDDING_MAX_WORKERS
#     to 8-12 via config.  Local servers have no rate limits, so the
#     practical ceiling is CPU/GPU core count.
#
#   Failure handling:
#     If any single batch raises (timeout, 429, network error, etc.),
#     the exception propagates immediately and fails the entire
#     embed_texts() call.  No partial results are returned.  This is
#     intentional — index_batch() treats embedding as all-or-nothing;
#     partial vectors would leave chunks in an inconsistent state.
# ────────────────────────────────────────────────────────────────

# four cached singletons - nothing loads at import time

_CLIENT: OpenAI | None = None
_TOKENIZER: Callable[[str], list[int]] | None = None
_TIKTOKEN_ENCODING: Any | None = None
_HF_TOKENIZER: Any | None = None

# build the OpenAI client once using base_url from config; optional API key since local models don't need one
# handles all providers

def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        client_kwargs = {"base_url": settings.embedding_base_url}
        if settings.embedding_api_key:
            client_kwargs["api_key"] = settings.embedding_api_key
        _CLIENT = OpenAI(**client_kwargs)
    return _CLIENT

# builds simple text -> list[int] callable from whichever configured backend
# used by token_count(), returns token IDs, not the boundary map

def _build_tokenizer() -> Callable[[str], list[int]]:
    if settings.embedding_tokenizer_kind == "tiktoken":
        encoding = _get_tiktoken_encoding()
        return lambda text: encoding.encode(text)

    if settings.embedding_tokenizer_kind == "huggingface":
        tokenizer = _get_hf_tokenizer()
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    raise ValueError(
        f"Unsupported tokenizer kind: {settings.embedding_tokenizer_kind}. "
        "Expected 'tiktoken' or 'huggingface'."
    )

# lazy loaders for the two tokenizer backends
# separated from _build_tokenizer so boundary map functions
# can access raw encoding/tokenizer objects directly

def _encode_tokens(text: str) -> list[int]:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _build_tokenizer()
    return _TOKENIZER(text)


def _get_tiktoken_encoding() -> Any:
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        import tiktoken
        _TIKTOKEN_ENCODING = tiktoken.get_encoding(settings.embedding_tokenizer_name)
    return _TIKTOKEN_ENCODING


def _get_hf_tokenizer() -> Any:
    global _HF_TOKENIZER
    if _HF_TOKENIZER is None:
        from transformers import AutoTokenizer
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(settings.embedding_tokenizer_name)
    return _HF_TOKENIZER


def token_count(text: str) -> int:
    return len(_encode_tokens(text))

def _token_boundary_map_tiktoken(markdown: str) -> list[int]:
    encoding = _get_tiktoken_encoding()
    token_ids = encoding.encode(markdown)
    if not token_ids:
        return [0] * (len(markdown) + 1)

    # Official API gap: encode()/encode_ordinary() do not expose offset spans.
    # We use decode_with_offsets/decode_tokens_bytes to derive char-aligned
    # boundaries in a supported way, while keeping Python str index semantics.
    decoded_text, _token_start_chars = encoding.decode_with_offsets(token_ids)
    if decoded_text != markdown:
        raise ValueError(
            "tiktoken decode_with_offsets roundtrip mismatch while building boundaries."
        )

    token_bytes = encoding.decode_tokens_bytes(token_ids)
    token_end_chars: list[int] = []
    text_len = 0
    for token in token_bytes:
        # Mirrors tiktoken's UTF-8 continuation-byte handling for char positions.
        text_len += sum(1 for byte in token if not 0x80 <= byte < 0xC0)
        token_end_chars.append(text_len)

    boundaries = [0] * (len(markdown) + 1)
    token_count_so_far = 0
    for char_index in range(len(markdown) + 1):
        while (
            token_count_so_far < len(token_end_chars)
            and token_end_chars[token_count_so_far] <= char_index
        ):
            token_count_so_far += 1
        boundaries[char_index] = token_count_so_far

    return boundaries

# huggingface tokenizers support offset mapping returns
# which returns (char_start, char_end) pairs

def _token_boundary_map_huggingface(markdown: str) -> list[int]:
    tokenizer = _get_hf_tokenizer()
    encoded = tokenizer(
        markdown,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "HuggingFace tokenizer did not provide offset_mapping; "
            "cannot compute token boundaries accurately."
        )

    # Ignore zero-length spans (special/no-op tokens) so boundary counts stay stable.
    token_end_chars = [end for start, end in offsets if end > start]
    boundaries = [0] * (len(markdown) + 1)

    token_count_so_far = 0
    for char_index in range(len(markdown) + 1):
        while (
            token_count_so_far < len(token_end_chars)
            and token_end_chars[token_count_so_far] <= char_index
        ):
            token_count_so_far += 1
        boundaries[char_index] = token_count_so_far

    return boundaries

# dispatcher routing to tiktoken or HuggingFace path based on config

def _token_boundary_map(markdown: str) -> list[int]:
    if settings.embedding_tokenizer_kind == "tiktoken":
        return _token_boundary_map_tiktoken(markdown)
    if settings.embedding_tokenizer_kind == "huggingface":
        return _token_boundary_map_huggingface(markdown)
    raise ValueError(
        f"Unsupported tokenizer kind: {settings.embedding_tokenizer_kind}. "
        "Expected 'tiktoken' or 'huggingface'."
    )

# ── Debug flag (same env var as indexer.py) ───────────────────
_EMBEDDING_DEBUG = os.getenv("INDEXING_DEBUG", "").strip() == "1"

# sends list of texts to the embedding API in concurrent batches;
# validates every response has the right count and dimensionality.
#
# Concurrency uses ThreadPoolExecutor (stdlib, no asyncio) so the
# indexing layer stays regular-def per the spec.  Each thread calls
# _embed_single_batch() which is an isolated, stateless HTTP request.
#
# Failure semantics: if ANY batch raises, the exception propagates
# out of the executor and fails the entire call — no partial vectors
# are ever returned.

def _embed_single_batch(batch: list[str]) -> list[list[float]]:
    """Embed a single batch of texts.  Called from worker threads."""
    client = _get_client()
    response = client.embeddings.create(model=settings.embedding_model, input=batch)
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(batch):
        raise ValueError(
            "Embedding response size mismatch: "
            f"expected {len(batch)} vectors, got {len(embeddings)}"
        )
    for vector in embeddings:
        if len(vector) != settings.embedding_dimensions:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {settings.embedding_dimensions}, got {len(vector)}"
            )
    return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert chunk texts to embedding vectors via concurrent API batches."""
    if not texts:
        return []

    batch_size = settings.embedding_batch_size
    max_workers = settings.embedding_max_workers

    # Fast path: if everything fits in one batch, skip the executor overhead.
    if len(texts) <= batch_size:
        return _embed_single_batch(texts)

    # Split texts into sequential batches, preserving order.
    batches: list[list[str]] = [
        texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
    ]
    if _EMBEDDING_DEBUG:
        print(
            f"embed_texts: dispatching {len(texts)} texts "
            f"in {len(batches)} batches (size={batch_size}, workers={max_workers})"
        )

    # results_by_index preserves input ordering regardless of completion order.
    results_by_index: dict[int, list[list[float]]] = {}

    # ThreadPoolExecutor.submit + as_completed gives us:
    #   - concurrent HTTP requests (threads release the GIL on I/O)
    #   - fail-fast: first exception is raised immediately, cancelling
    #     remaining futures and preventing partial-result return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_embed_single_batch, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            # .result() re-raises any exception from the worker thread.
            results_by_index[idx] = future.result()

    # Reassemble in original order.
    all_embeddings: list[list[float]] = []
    for idx in range(len(batches)):
        all_embeddings.extend(results_by_index[idx])

    return all_embeddings

# public function indexer.py will call - builds boundary map once for the full document markdown
# for each chunk does 2 O(1) look ups (boundaries[char_start], boundaries[char_end]) to set token_start and token_end

def annotate_token_offsets(chunks: list[Chunk], markdown: str) -> None:
    """Stamp token_start/token_end on each chunk using the document's full markdown."""
    if not chunks:
        return

    # Token offsets are absolute over the full document token stream.
    # Build char-index -> token-count map once per document for O(chunks) assignment.
    boundaries = _token_boundary_map(markdown)
    max_index = len(boundaries) - 1

    for chunk in chunks:
        start_index = min(max(chunk.char_start, 0), max_index)
        end_index = min(max(chunk.char_end, 0), max_index)
        if end_index < start_index:
            end_index = start_index
        chunk.token_start = boundaries[start_index]
        chunk.token_end = boundaries[end_index]


def embed_query(text: str) -> list[float]:
    """Embed a single retrieval query string."""
    return embed_texts([text])[0]
