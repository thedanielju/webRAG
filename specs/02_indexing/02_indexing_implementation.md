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
- index_documentindex_document(doc: NormalizedDocument, depth: int) -> None
- index_batch(docs: list[NormalizedDocument], depths: list[int]) -> None

index_batch batches all child chunk texts across the entire input list into a single embedding API call, then writes all records. This matches ingestion's ingest_batch pattern.(doc, depth) and index_batch(docs, depths). Nothing else crosses the boundary.

### Libraries
## Database
psycopg[binary]     # psycopg3, binary package for better performance
pgvector            # pgvector Python adapter

## Embeddings
openai              # embedding calls for all providers (OpenAI-compatible interface)
tiktoken            # tokenization for OpenAI models (text-embedding-3-small, etc.)
transformers        # tokenization for local models (nomic-embed-text, bge-m3)

## Text processing
re                  # stdlib: markdown pattern detection (headings, math, code fences)
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

# Schema Init
schema.py called programmatically with IF NOT EXISTS guard, not a standalone migration script. schema.py is not a standalone migration script. It exposes an init_schema(conn) function that runs CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS for all tables and indexes. indexer.py calls this once at startup before any indexing operations.

# HTML Parser
BS4 with lxml as the backend parser for rich content flag detection. Fallback to html.parser (stdlib) on parse errors. Pattern matching on markdown (via re) handles math blocks, code fences, and admonitions — bs4 is only used for HTML-specific patterns (<table>, <dl>).

# Config
There is a single config file at src/config.py that all layers import from. It loads all env vars via python-dotenv and exposes a single Pydantic config object. Indexing-specific fields (embedding model, dims, tokenizer, DB connection string) live there alongside ingestion fields. No layer has its own config module.

# Retrieval-Facing Schema Additions
The following fields are denormalized onto the chunks table so retrieval never needs to join documents at query time:

- depth (int): copied from the depth argument passed to index_document
- title (text | null): copied from NormalizedDocument.title

These join the already-denormalized source_url and fetched_at fields. The complete set of fields retrieval can read from a chunk row without any join: source_url, fetched_at, depth, title, section_heading, chunk_index, has_* flags, char_start, char_end, chunk_text, html_text.