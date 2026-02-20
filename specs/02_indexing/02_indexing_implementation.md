### IMPLEMENTATION
Added post-design-session. These decisions are finalized and should be treated as constraints, not suggestions.

## Scope
The indexing layer takes a NormalizedDocument (or a list of them) from ingestion and produces queryable, citation-ready records in Postgres. It owns:

The DB schema (documents + chunks tables, HNSW index)
Chunking (splitting markdown into parent-child chunks with offsets)
Rich content flag detection
Embedding (calling the embedding API, computing token offsets)
Writing to DB (dedup check, atomic delete+reinsert on hash mismatch)

It does not own:

Fetching or normalizing web content (ingestion's job)
Deciding which documents to index or in what order (orchestration's job)
Querying or retrieving chunks (retrieval layer's job)
The MCP server interface (separate layer)

# Public interface is exactly two functions: 
- index_document(doc: NormalizedDocument, depth: int) -> None
- index_batch(docs: list[NormalizedDocument], depths: list[int]) -> None

index_batch batches all child chunk texts across the entire input list and embeds them via concurrent HTTP batches, then bulk-writes all records. This matches ingestion's ingest_batch pattern. Nothing else crosses the boundary.

### Libraries
## Database
psycopg[binary]     # psycopg3, binary package for better performance
pgvector            # pgvector Python adapter

## Embeddings
openai              # embedding calls for all providers (OpenAI-compatible interface)
tiktoken            # tokenization for OpenAI models (text-embedding-3-small, etc.)
transformers        # tokenization for local models (nomic-embed-text, bge-m3)

## Text processing
re                  # stdlib: markdown pattern detection (headings, code fences, admonition keywords, ordered lists)
beautifulsoup4      # HTML pattern detection for rich content flags
lxml                # fast C-based backend for bs4; graceful fallback to bs4's html.parser on parse errors

## Data / types
dataclasses         # stdlib: internal chunk structures
uuid                # stdlib
datetime            # stdlib

## Config
pydantic            # config objects only
python-dotenv       # env var loading

### Outline
src/indexing/
├── models.py     # Dataclasses — Chunk, ChunkLevel enum, RichContentFlags
├── schema.py     # CREATE TABLE / CREATE INDEX — called programmatically with IF NOT EXISTS guard
├── chunker.py    # Parent-child splitting logic. does flag detection as well: finding <table>, $$, etc
├── embedder.py   # Embedding + tokenization — accepts list of texts, returns list of vectors + token offsets
└── indexer.py    # Entry point — index_document() and index_batch(), wires everything together, stores, handles dedup. 

root/
├── config.py     # Pydantic config — DB, embedding provider, model, dims, tokenizer

## Module Notes
# Async
All indexing functions are regular def — not async def. The ingestion layer uses the sync Firecrawl client and its entry points are regular def. Indexing matches this. If a future async caller (e.g. MCP server) needs to call indexing, it wraps via asyncio.to_thread() at that boundary. Do not introduce asyncio inside the indexing layer.

Concurrency within indexing uses stdlib concurrent.futures.ThreadPoolExecutor — this is compatible with regular def functions and does not require asyncio. Threads are used for two purposes:
1. embed_texts() fans out HTTP embedding requests across EMBEDDING_MAX_WORKERS threads
2. index_batch() overlaps chunking (main thread) with embedding (background thread) for multi-doc batches

# Embedding Concurrency
embedder.py splits the full text list into batches of EMBEDDING_BATCH_SIZE (default 256) and dispatches them concurrently via ThreadPoolExecutor with at most EMBEDDING_MAX_WORKERS threads (default 4).

Batch size rationale: 256 texts per request balances HTTP overhead (fewer requests) against per-request latency and memory. OpenAI's embedding endpoint handles payloads up to ~8M tokens, so 256 chunks of ~256 tokens each (~65K tokens) is well within limits while keeping individual request latency manageable (~10-12s).

Worker count rationale: Default 4 is conservative to avoid 429 rate-limit errors on lower OpenAI API tiers (free/tier-1 typically allow 60-200 RPM). Paid plans with higher RPM allowances can safely increase to 8-12. Local embedding servers (Ollama, LM Studio) have no rate limits — the practical ceiling is CPU/GPU core count.

Both values are configurable via EMBEDDING_BATCH_SIZE and EMBEDDING_MAX_WORKERS in config.py / .env.

Failure handling: if any single batch raises (timeout, 429, network error, dimension mismatch), the exception propagates immediately and fails the entire embed_texts() call. No partial results are returned. index_batch() treats embedding as all-or-nothing — partial vectors would leave chunks in an inconsistent state.

# Chunking/Embedding Overlap
For multi-doc index_batch() calls, chunking of subsequent documents proceeds in the main thread while embedding of already-chunked documents runs in a single background thread. This is safe because:
- Chunking is pure CPU work operating on independent NormalizedDocument objects
- embed_texts() operates on an independent list of strings, no shared mutable state
- The background thread's embed_texts() internally fans out to its own ThreadPoolExecutor

For single-document calls (index_document), the overlap adds no benefit and is skipped — embed_texts() is called synchronously.

# DB Write Performance
_insert_chunks uses cursor.executemany() instead of per-row cursor.execute() loops. executemany sends the parameterised INSERT once and streams all row data in a single client-server round-trip, reducing Phase 3 write time from ~3s to <0.5s for 1000+ chunk batches. The semantic behaviour is identical: one INSERT per chunk, same column set, same parameter mapping.

# Expected Timing Improvements
Benchmarked on 3 large documentation pages (1243 child chunks total):
| Phase | Before | After |
|---|---|---|
| Embedding (1243 chunks) | ~70s (single API call) | ~10-15s (concurrent batches) |
| DB writes (1317 rows) | ~3s (row-by-row) | <0.5s (executemany) |
| Total index_batch | ~77s | ~15-20s |

For a single average page (50-150 chunks): ~3-5s total.

# Schema Init
schema.py called programmatically with IF NOT EXISTS guard, not a standalone migration script. schema.py is not a standalone migration script. It exposes an init_schema(conn) function that runs CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS for all tables and indexes. indexer.py calls this once at startup before any indexing operations.

# HTML Parser
BS4 with lxml as the backend parser for rich content flag detection. Fallback to html.parser (stdlib) on parse errors. Pattern matching on markdown (via re) handles code fences, admonitions (Firecrawl's bare keyword format), tables, and ordered lists. bs4 handles all HTML-specific patterns — math (MathML/MathJax), definition lists (`<dl>`), admonitions (CSS classes), and tables.

# Rich Content Detection Rules
Detection rules updated after auditing real Firecrawl output (Feb 2026). Firecrawl strips LaTeX and does not produce markdown definition-list syntax. Detection relies on HTML for math, definition lists, and (primarily) admonitions.

- Math:
  - HTML only: MathML `<math>` tags, MathJax 3 `<mjx-*>` elements, `MathJax`/`katex`/`math` CSS classes
  - Markdown: no detection (Firecrawl strips all `$`/`$$` LaTeX syntax)
  - Snippet extraction: targets clean MathML `<math>` tags exclusively; MathJax visual elements excluded
- Definition lists:
  - HTML only: `<dl><dt><dd>`
  - Markdown: no detection (Firecrawl does not produce definition-list syntax)
- Admonitions:
  - Markdown: bare keyword line (`Note`, `Warning`, `Important`, `Tip`, `Caution`, `Danger`, `Info`, `Success`, `Example`, `See also`, `Deprecated since`) — matches Firecrawl's actual rendering
  - HTML classes containing: `admonition`, `note`, `warning`, `tip`, `important`, `caution`, `danger`, `info`
- Steps:
  - Ordered list markers `1.` and `1)`, including nested ordered lists

# Config
There is a single config file at config.py (project root) that all layers import from. It loads all env vars via python-dotenv and exposes a single Pydantic config object. Indexing-specific fields (embedding model, dims, tokenizer, DB connection string) live there alongside ingestion fields. No layer has its own config module.

# Retrieval-Facing Schema Additions
The following fields are denormalized onto the chunks table so retrieval never needs to join documents at query time:

- depth (int): copied from the depth argument passed to index_document
- title (text | null): copied from NormalizedDocument.title

These join the already-denormalized source_url and fetched_at fields. The complete set of fields retrieval can read from a chunk row without any join: source_url, fetched_at, depth, title, section_heading, chunk_index, has_* flags, char_start, char_end, chunk_text, html_text.
