# WebRAG Architecture

## System Overview

WebRAG is a research memory system that enables LLMs to reason over evolving web corpora. It recursively and selectively expands, indexes, retrieves, and cites evidence from web sources. The system cleanly separates:

- **Reasoning layer** → the chat model (Claude, GPT, etc.)
- **Knowledge layer** → WebRAG memory system

The model queries WebRAG via MCP tools. WebRAG handles corpus construction, retrieval, and citation grounding — the model never needs to scrape, embed, or search on its own.

## System Goals

1. Attach websites as structured knowledge corpora
2. Support iterative corpus expansion via link and sublink traversal
3. Retrieve evidence-grounded passages for reasoning
4. Produce citation-grade quotes tied to source URLs and section headings
5. Maintain an evolving research memory over time

## Architecture Layers

```
User Query
  ↓
LLM (Reasoning layer — Claude, GPT, etc.)
  ↓
Layer 5: MCP Server (tools.py, server.py, formatter.py)
  ↓
Layer 4: Orchestration (engine.py, evaluator, expander, merger)
  ↓
Layer 3: Retrieval (search.py, citations.py)
  ↓
Layer 2: Indexing (chunker.py, embedder.py, indexer.py)
  ↓
Layer 1: Ingestion (service.py, firecrawl_client.py, links.py)
  ↓
Postgres + pgvector
```

Each layer has a numbered directory (`src/01_ingestion/` through `src/05_mcp_server/`) and is mapped to clean import paths via `pyproject.toml` (e.g. `from src.orchestration.engine import OrchestratorEngine`).

## Core Architectural Principles

1. **Firecrawl for reliable acquisition** — Firecrawl handles bot compliance, browser rendering, rate limiting, proxy rotation, and PDF parsing. WebRAG consumes its output rather than reimplementing web scraping.
2. **Evolving corpus, not static snapshots** — Key features include selective recursion, corpus growth strategy, retrieval policy, citation rigor, and intelligent stopping criteria.
3. **Configuration in one place** — All settings live in `config.py` via pydantic-settings. Feature modules import typed constants and never read env vars directly.
4. **Async everywhere** — All I/O (database, embedding API, reranker API, Firecrawl) is async. The orchestration loop uses `asyncio.gather()` for concurrent sub-query retrieval.
5. **Clean error boundaries** — Each tool returns structured text, never exceptions. Errors are formatted as readable diagnostics that instruct the model not to hallucinate.

---

## Layer 1: Ingestion (`src/01_ingestion/`)

**Responsibilities:** Fetch pages, extract content, discover sublinks, prevent duplicate ingestion.

| Module               | Role |
|----------------------|------|
| `service.py`         | Top-level `ingest(url)` → `NormalizedDocument`. Coordinates Firecrawl client, link extraction, and content normalization. |
| `firecrawl_client.py`| Thin async wrapper around the Firecrawl API. Handles scrape, map, and batch operations. |
| `links.py`           | URL canonicalization, tracking parameter stripping, trailing slash normalization, content hashing for dedup. |

**Output:** `NormalizedDocument` — a Pydantic model containing URL, title, metadata, markdown text, HTML text, outgoing links, and a content hash.

## Layer 2: Indexing (`src/02_indexing/`)

**Responsibilities:** Chunk documents, generate embeddings, store in Postgres.

| Module        | Role |
|---------------|------|
| `chunker.py`  | Splits markdown by semantic structure (headings/paragraphs) into parent and child chunks. Detects rich-content signals (tables, code, math, images) from both markdown and aligned HTML. |
| `embedder.py` | Concurrent async embedding via OpenAI-compatible API. Batches texts and dispatches with `asyncio.gather()`. Supports tiktoken and HuggingFace tokenizers. |
| `indexer.py`  | Pipeline entry point: dedup check → chunk → embed → transactional DB write. Uses `executemany` for bulk inserts. |
| `models.py`   | `Chunk`, `ChunkLevel`, `RichContentFlags`, `ChunkImageRef` data models. |
| `schema.py`   | DDL for `documents` and `chunks` tables, plus HNSW vector index. Idempotent (`IF NOT EXISTS`). |

### Chunking Strategy

- **Parent chunks** (~1000 tokens max): Heading-bounded sections of the document. Returned by retrieval for context.
- **Child chunks** (~256 tokens target): Sub-sections of parents. Embedded and searched via ANN. Scores roll up to their parent.
- **Rich content detection**: When a parent contains tables, code blocks, math, or images, the indexer preserves the HTML surface alongside markdown so retrieval can choose the best representation.

### Storage

- Postgres with pgvector extension
- HNSW index for approximate nearest neighbor search on child embeddings
- Chunk text, metadata, and embeddings stored together (no joins needed at retrieval time)

## Layer 3: Retrieval (`src/03_retrieval/`)

**Responsibilities:** Semantic search, passage selection, citation extraction.

| Module         | Role |
|----------------|------|
| `search.py`    | `retrieve(conn, query)` → `RetrievalResult`. Two modes: full-context (small corpus, return everything) and chunk-mode (ANN search → parent aggregation → budget trim). |
| `citations.py` | `extract_citation(chunk, start, end)` → `CitationSpan`. Reconstructs verbatim quotes using stored character offsets. |
| `models.py`    | `RetrievedChunk`, `RetrievalResult`, `CorpusStats`, `TimingInfo`. |

### Retrieval Modes

- **Full-context mode**: When total corpus tokens < `RETRIEVAL_FULL_CONTEXT_THRESHOLD` (30k default), returns all parent chunks ordered by position. No embedding query needed — fast and complete.
- **Chunk mode**: Embeds the query, runs HNSW ANN search on child chunks, aggregates scores to parents, applies depth decay and similarity floor, trims to token budget.

**Surface selection**: Each returned parent gets its `surface` field set to `"html"` or `"markdown"` based on rich-content flags, so downstream formatting knows how to render it.

The retrieval layer supports both modes. The MCP layer now defaults to `chunk` mode for faster user-facing behavior and better citation/image visibility. Full-context remains available as an explicit caller choice.

## Layer 4: Orchestration (`src/04_orchestration/`)

**Responsibilities:** Drive the retrieve-evaluate-expand loop. Decide when the corpus is "good enough" and when to grow it.

| Module              | Role |
|---------------------|------|
| `engine.py`         | Top-level entry point. Owns connection pool, drives the iteration loop, assembles `OrchestrationResult`. |
| `query_analyzer.py` | Decomposes a user query into sub-queries (LLM-based, rule-based, or passthrough). |
| `reranker.py`       | Provider-agnostic reranking (ZeroEntropy, Cohere, Jina, or passthrough). |
| `evaluator.py`      | Computes quality signals from score distributions and applies an 11-rule decision matrix (stop / expand_breadth / expand_recall / expand_intent). |
| `expander.py`       | Scores candidate links (5 weighted signals), scrapes top picks, indexes them. |
| `locality.py`       | Fetches adjacent sibling chunks around high-scoring hits (cheap DB query, no API calls). |
| `merger.py`         | Unions chunks from multiple sub-queries, deduplicates with MMR (Jaccard text overlap), enforces token budget. |
| `models.py`         | Pydantic data contracts: `QueryAnalysis`, `RankedChunk`, `EvaluationSignals`, `ExpansionStep`, `OrchestrationResult`, etc. |

### Orchestration Loop

```
1. Ensure seed URL is ingested and indexed
2. Analyse / decompose query into sub-queries
3. Retrieve → rerank → evaluate quality signals
4. Decision:
   ├── STOP → proceed to output
   ├── EXPAND_BREADTH → score candidate links, ingest top picks, re-retrieve
   ├── EXPAND_RECALL → re-retrieve with doubled token budget
   └── EXPAND_INTENT → re-analyse query with feedback, re-retrieve
5. Repeat steps 3–4 up to max_expansion_depth iterations
6. Locality expansion (adjacent sibling chunks)
7. Final merge, MMR dedup, token-budget trim, citation extraction
```

### Stop Conditions (priority order)

1. Max expansion depth reached (safety cap)
2. Token budget filled with good-quality, non-redundant chunks
3. Diminishing returns (recall proxy barely improved iteration-over-iteration)
4. Good plateau (low score variance, mean above mediocre floor)

### Connection Strategy

- **Linux/macOS**: `AsyncConnectionPool` for efficient connection reuse (epoll/kqueue handle pool background workers).
- **Windows**: Direct `AsyncConnection` per operation. The pool's background workers corrupt Windows' `SelectorEventLoop` fd set, causing `OSError [WinError 10038]`.
- **Per-sub-query isolation**: Each sub-query in `asyncio.gather()` gets its own connection to avoid `OutOfOrderTransactionNesting` from concurrent cursor operations on a shared connection.

## Layer 5: MCP Server (`src/05_mcp_server/`)

**Responsibilities:** Expose WebRAG to reasoning models via the Model Context Protocol.

| Module              | Role |
|---------------------|------|
| `server.py`         | FastMCP setup, lifespan context manager (engine lifecycle), transport config, CLI entry point. |
| `tools.py`          | Three MCP tool handlers: `answer`, `search`, `status`. |
| `formatter.py`      | `OrchestrationResult` → structured plain-text response with token budget management. |
| `html_converter.py` | HTML → readable plain text (tables, code blocks, MathML, admonitions, images). |
| `errors.py`         | Error response templates that instruct the model not to hallucinate. |

### MCP Tools

| Tool     | Purpose | Speed |
|----------|---------|-------|
| `answer` | Full pipeline: ingest URL(s) → decompose → retrieve → rerank → expand → format. | Slow (seconds to minutes) |
| `search` | Query existing corpus. Skips ingestion and expansion. Direct retrieve + rerank. | Fast (sub-second) |
| `status` | Report corpus contents: document count, tokens, URLs, timestamps. | Instant |

### MCP Default Behavior

- `answer` defaults to a fast pass: chunked retrieval and no expansion unless explicitly requested.
- `answer` supports explicit overrides such as `research_mode="deep"` and `retrieval_mode="full_context"`.
- Every tool response starts with a `[PRESENTATION GUIDE]` — inline directives that tell the model exactly which sections to include in its answer (ACI poka-yoke pattern).
- Every tool response ends with `[FOLLOW-UP OPTIONS]` — context-sensitive suggestions (deep search, continued expansion, full context, follow-up search, query refinement) that the model presents to the user.
- Citations are guaranteed (never dropped by the token budget). The formatter builds citations before allocating space to evidence.
- The orchestration engine emits richer phase and iteration progress callbacks that the MCP layer surfaces as progress notifications.

### Response Format

Tool responses use bracketed section headers for unambiguous LLM parsing. Every response includes 5 guaranteed sections and up to 3 conditional sections:

```
[PRESENTATION GUIDE]                    ← guaranteed, inline directives for the model
Your response to the user MUST include:
1. A synthesized answer ...
2. Inline citations for every claim ...
3. Use the [CITATIONS] section below ...
...

[SOURCES]                               ← guaranteed
[1] Page Title — https://example.com/page
    § Section Heading

[EVIDENCE]                              ← budget-dependent, fills remaining space
Source [1] (relevance: 0.87, confidence: 0.92):
Relevant text content here...

[IMAGES]                                ← conditional (if images found + budget)
- [Alt text](https://example.com/image.png) (from Source [1])

[EXPANSION TRACE]                       ← conditional (if expansion occurred + budget)
Base: https://example.com/
/page (seed)
├── /subpage (depth 1, +12 chunks)

[CITATIONS]                             ← guaranteed (never dropped by budget)
[1] "Verbatim quote from the source"
    — Page Title, https://example.com/page § Section

[STATS]                                 ← guaranteed
Mode: chunk
Documents searched: 3
Chunks evaluated: 45
Expansion iterations: 1
Total time: 5200ms

[FOLLOW-UP OPTIONS]                     ← guaranteed, context-sensitive
- DEEP SEARCH: This was a fast single-page pass ...
- FOLLOW-UP SEARCH: The indexed content is available ...
- REFINE QUERY: If results are too broad ...
```

### Token Budget Strategy

The formatter manages a soft token budget (`MCP_RESPONSE_TOKEN_BUDGET`, default 30k) to keep responses within model context limits. Sections are split into guaranteed (never dropped) and budget-dependent (fills remaining space):

**Guaranteed (built first, subtracted from budget):**
1. **[PRESENTATION GUIDE]** — inline directives for the model (~80–120 tokens)
2. **[SOURCES]** — always included in full (compact, essential for attribution)
3. **[STATS]** — always included in full (small, useful for diagnostics)
4. **[CITATIONS]** — always included in full (promoted from budget-dependent to guaranteed so citations are never silently dropped)
5. **[FOLLOW-UP OPTIONS]** — context-sensitive next-step suggestions (~80–150 tokens)

**Budget-dependent (fills remaining space):**
6. **[EVIDENCE]** — fills remaining budget, sorted by relevance; truncates with a count note
7. **[EXPANSION TRACE]** — included if budget allows, otherwise replaced with summary
8. **[IMAGES]** — included if budget allows

### Transport Modes

- **stdio** (default): For desktop MCP clients (Claude Desktop, Cursor). Client launches the server as a subprocess. All logging goes to stderr — stdout is reserved for the MCP JSON-RPC protocol stream.
- **streamable-http**: For remote/hosted deployments. Listens on `MCP_HOST:MCP_PORT`.

---

## Targeted Expansion Strategy

### Seed Set

- Root URL provided by the model
- Sublinks discovered via Firecrawl's map endpoint

### Link Scoring

Candidate links are scored using 5 weighted signals:
- URL path heuristics (documentation patterns, API references)
- Candidate title/description relevance to the query
- Page title relevance
- Recency (weak signal)
- Domain allowlist constraints (when configured)

Expansion also applies a minimum scored-candidate threshold before expensive scrape/index work begins. This avoids low-yield expansion rounds by default.

### Termination

- Token budget saturation (primary signal)
- Diminishing retrieval improvement (recall proxy)
- Score plateau detection (variance + floor checks)
- Max depth / max candidates safety caps

Depth and page count are **weak** constraints — the evaluator's 11-rule decision matrix drives termination, not arbitrary limits.

## Citation Model

WebRAG produces two citation layers:

**1. Reference Layer** — Indexed source references:
```
[1] Page Title — https://example.com/page § Section Heading
```

**2. Evidence Layer** — Verbatim snippets reconstructed from stored character offsets:
```
"exact retrieved span from the original document"
```

Citations are reconstructed from stored chunk offsets to guarantee fidelity — no paraphrasing or summarization.

## Intended Usage Model

WebRAG enables "docs-as-context" workflows where users can:
- Query web documentation conversationally
- Verify answers against source text
- Explore linked material iteratively

The system acts as a persistent research memory rather than a one-shot retrieval tool.
