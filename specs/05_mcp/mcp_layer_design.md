# WebRAG MCP Layer — Design Document

> **Layer:** `src/05_mcp_server/` (`src.mcp_server`)

---

## 1. Purpose

The MCP layer is the final layer of WebRAG. It wraps the `OrchestratorEngine` and exposes it as tools over the Model Context Protocol (MCP) for reasoning LLMs (Claude, ChatGPT, OpenWebUI). It is primarily a **transport + formatting** layer — it does not contain retrieval logic, reranking, or expansion decisions.

### Responsibilities

1. Start and manage the `OrchestratorEngine` lifecycle (create → start → reuse → stop).
2. Define MCP tools with typed input schemas.
3. Call `OrchestratorEngine.run()` and receive `OrchestrationResult`.
4. Transform the structured result into formatted text the reasoning model can parse — including citations, rich content conversion, traversal diagrams, and stats.
5. Handle errors gracefully, surfacing them as readable text rather than letting the model hallucinate.

### Non-responsibilities

- No retrieval, reranking, or expansion logic.
- No LLM calls — the MCP layer never invokes a language model.
- No corpus management UI in v1.

---

## 2. Transport

### 2.1 Primary: stdio (default)

stdio is the standard transport for Claude Desktop and local MCP clients. The client spawns the WebRAG server as a subprocess and communicates over stdin/stdout. This is the simplest transport and works on all platforms (Windows, Linux, macOS).

**Use when:** Running locally with Claude Desktop, Claude Code, or for development/testing.

### 2.2 Secondary: Streamable HTTP

Streamable HTTP runs the server as a long-lived HTTP process. Claude's custom connectors (Settings > Connectors > "Add custom connector") connect to the server URL over HTTP. This is required for remote access — e.g., running WebRAG on a Linux server and connecting from Claude.ai or Claude Desktop on another machine.

**Use when:** Deploying as a remote service, using Claude custom connectors, or multi-client access.

### 2.3 Transport Selection

A single entry point supports both transports via configuration:

- **CLI flag:** `--transport stdio` (default) or `--transport streamable-http`
- **Environment variable:** `MCP_TRANSPORT=streamable-http`
- **Config:** `mcp_transport` setting in `.env`

The `FastMCP` SDK from the official `mcp` Python package handles both transports natively. The same tool definitions work for either transport — only the server startup changes.

### 2.4 Streamable HTTP Deployment Notes

When using Streamable HTTP:

- The server binds to `MCP_HOST` (default `0.0.0.0`) and `MCP_PORT` (default `8765`).
- For production, deploy behind an nginx reverse proxy with TLS termination and Cloudflare DNS — matching the existing server infrastructure pattern.
- The MCP endpoint URL will be something like `https://webrag.example.com/mcp`.
- Optional bearer token auth via `MCP_AUTH_TOKEN` env var (see §8 Security).

---

## 3. Engine Lifecycle

### 3.1 Current Engine Behavior

`OrchestratorEngine` (from `src/04_orchestration/engine.py`):

- **Constructor:** Parameterless `OrchestratorEngine()`.
- **`start()`:** Creates `AsyncConnectionPool` on non-Windows (Linux/macOS), or marks ready on Windows (uses per-request `AsyncConnection`).
- **`stop()`:** Closes the pool (non-Windows) or marks stopped (Windows).
- **`run()`:** Full orchestration pipeline — corpus prep, query analysis, retrieve-rerank-expand loop, locality expansion, citation assembly. Returns `OrchestrationResult`.

### 3.2 MCP Lifecycle Integration

The MCP server manages the engine via FastMCP's `lifespan` context manager:

```
Server startup:
  1. engine = OrchestratorEngine()
  2. await engine.start()      # Pool created on Linux, no-op marker on Windows
  3. Store engine in app state  # Accessible to tool handlers

Server running:
  4. Tool calls → engine.run() # Reuses the single engine instance + pool

Server shutdown:
  5. await engine.stop()        # Pool closed cleanly
```

The engine instance is created once and shared across all tool invocations. Connection pooling happens inside the engine (pool size 1–4 on Linux; direct connections on Windows).

### 3.3 Lifespan Implementation Pattern

Use FastMCP's lifespan context manager to wrap engine start/stop:

```
@asynccontextmanager
async def lifespan(app):
    engine = OrchestratorEngine()
    await engine.start()
    # store engine so tool handlers can access it
    yield
    await engine.stop()
```

For stdio transport, the lifespan runs for the duration of the subprocess. For Streamable HTTP, it runs for the lifetime of the HTTP server process.

---

## 4. MCP Interaction Model

Before specifying tools, it's important to understand how MCP tool calls flow — because the MCP layer does not control the model's final output to the user. It provides structured context that the model uses to synthesize its response.

### 4.1 How MCP Tools Work

1. **User sends a message** to the reasoning model (Claude, ChatGPT, etc.).
2. **The model decides whether to call a tool.** It sees the tool's name, description, and input schema — that's all it knows about the server. Based on the description, it decides whether this query needs WebRAG and constructs the parameters itself (picks the URL, formulates the query, etc.).
3. **The model sends a tool call request** to the MCP server. The user sees a "using tool" indicator but not the raw request.
4. **The MCP server runs orchestration and returns text.** This is the formatted response — [SOURCES], [EVIDENCE], [CITATIONS], [STATS] blocks. It goes back to the model as tool result content.
5. **The model reads the tool result and writes its response to the user.** The model reads the formatted text like a person reading a research brief, then synthesizes its answer — citing sources, quoting verbatim text, referencing scores and metadata as it sees fit.

### 4.2 What This Means for Formatting

The tool response is a **research brief for the model**, not a direct response to the user. The MCP layer cannot control what the model outputs. However, well-structured formatting with explicit metadata (relevance scores, source numbers, verbatim quotes) strongly influences the model's behavior:

- When we write `Source [1] (relevance: 0.85):` followed by text, the model naturally prioritizes it over `Source [5] (relevance: 0.31):`.
- When we include `[CITATIONS]` with verbatim quotes and source URLs, the model naturally uses those in its response.
- When we write `[ERROR] ... Do not attempt to answer from memory`, the model follows that instruction.
- When we include `[IMAGES]` with URLs and alt text, the model can embed those in its response.

The better the formatting, the better the model's response. This is why the formatting logic is the most important part of the MCP layer.

### 4.3 Progress Notifications

MCP supports server-to-client **notifications** that are separate from the tool result. These appear in the client UI *during* tool execution (e.g., "Analyzing query...", "Expanding to linked pages..."). See §5.6 for details.

---

## 5. Tool Definitions

### 5.1 `answer` — Primary Retrieval Tool

**Description for the model:**
> Query one or more web pages and their linked content for information. WebRAG will scrape each URL if not already indexed, decompose the query, retrieve and rerank relevant chunks, optionally expand to linked pages, and return cited evidence with source attribution. Use this when you need factual information grounded in specific web sources.

**Input Schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | `str \| list[str]` | Yes | One or more web page URLs to use as primary sources. When multiple URLs are provided, all are ingested and searched together. |
| `query` | `str` | Yes | Question or information need. |
| `intent` | `str \| None` | No | Hint: `"factual"`, `"comparison"`, `"how_to"`, or `"exploratory"`. If omitted, query analysis infers it. |
| `known_context` | `str \| None` | No | What the model already knows — helps avoid redundant retrieval. |
| `constraints` | `list[str] \| None` | No | E.g., `["must include code examples"]`, `["focus on performance benchmarks"]`. |
| `expansion_budget` | `int \| None` | No | Max expansion iterations. `0` = no expansion (seed pages only). `None` = auto (engine default, up to `max_expansion_depth`). |

**Multi-URL behavior:** When `url` is a list, the MCP layer calls `engine.run()` for the first URL (the primary seed), but ensures all other URLs are ingested into the corpus first so they're available for retrieval. This allows comparative queries like "Compare how scikit-learn and XGBoost handle categorical features" with `url=["https://scikit-learn.org/...", "https://xgboost.readthedocs.io/..."]`.

**Implementation:** When multiple URLs are provided, the MCP layer:
1. Ingests all URLs in parallel (or serially if on Windows) by calling `engine._ensure_ingested()` for each additional URL before the main `engine.run()` call on the first URL.
2. Since orchestration searches the full corpus after expansion starts, the additional URLs' content is naturally included in retrieval.

**Note:** Multi-URL support requires a thin shim in the MCP layer to pre-ingest the extra URLs. The orchestration engine itself does not need changes — it already searches the full corpus after the first iteration. If the shim proves impractical due to `_ensure_ingested` being a private method, an alternative is to expose a public `ensure_ingested(url)` method on the engine.

### 5.2 `search` — Corpus Search Tool

**Description for the model:**
> Search the existing WebRAG corpus for information without scraping new pages or expanding to linked content. Use this when you know the content has already been indexed (e.g., from a previous `answer` call) and want to ask a different question about the same material. Faster than `answer` because it skips ingestion and expansion.

**Input Schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `str` | Yes | Question or information need. |
| `source_urls` | `list[str] \| None` | No | Restrict search to these URLs. If omitted, searches the entire corpus. |
| `intent` | `str \| None` | No | Query type hint (same as `answer`). |
| `top_k` | `int \| None` | No | Max chunks to return. Default: 20. |

**Implementation:** The `search` tool bypasses the full orchestration loop. It:
1. Acquires a connection from the engine's pool.
2. Calls `retrieve()` directly (from `src.retrieval.search`) with the query.
3. Optionally calls `rerank()` on the results.
4. Extracts citations and formats the response using the same formatter.

The response format is identical to `answer` but without `[EXPANSION TRACE]` (no expansion occurs), and `[STATS]` reflects the simpler pipeline.

**Why this exists:** After the model calls `answer` for a URL, the content is indexed in Postgres. If the user asks a follow-up question about the same content, `search` avoids re-scraping and re-expanding — it goes straight to vector search + reranking. Much faster for iterative research.

### 5.3 `status` — Corpus Status Tool

**Description for the model:**
> Check what content WebRAG currently has indexed. Returns document count, total tokens, indexed URLs with titles, and last-fetched timestamps. Use this to understand what's available before deciding whether to call `answer` (which ingests new content) or `search` (which queries existing content).

**Input Schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_url` | `str \| None` | No | Check status for a specific URL. If omitted, returns corpus-wide stats. |
| `include_urls` | `bool` | No | Include the list of all indexed URLs. Default: `True`. |

**Implementation:** Direct database query against the `documents` and `chunks` tables:

```sql
-- Corpus-wide stats:
SELECT COUNT(DISTINCT d.id), COUNT(c.id), SUM(c.token_count)
FROM documents d LEFT JOIN chunks c ON d.id = c.document_id;

-- Per-URL: document title, chunk count, total tokens, fetched_at
-- URL listing: all indexed source_urls with titles
```

**Response format:**
```
[CORPUS STATUS]
Documents indexed: 12
Total chunks: 847
Total tokens: 198,432

Indexed URLs:
- Ensemble Methods — https://scikit-learn.org/stable/modules/ensemble.html
  (47 chunks, 12,340 tokens, fetched 2026-02-22 14:30 UTC)
- Gradient Boosting — https://scikit-learn.org/stable/modules/gradient_boosting.html
  (31 chunks, 8,210 tokens, fetched 2026-02-22 14:31 UTC)
...
```

**Why this exists:** The model can check whether content is already indexed before deciding to call `answer` (slow, ingests + expands) vs `search` (fast, existing content only). This enables smarter tool selection by the reasoning model.

### 5.4 Future Tools (not implemented in v1)

- **`ingest`** — Explicitly scrape and index a URL without querying it. Useful for pre-loading content the model expects to query later. Trivial to add — calls `ingest()` + `index_batch()` directly.

---

## 6. Response Formatting

The MCP layer transforms `OrchestrationResult` into structured plain text. MCP tool responses are text-only (no markdown rendering within the tool result), but the reasoning model reads this text and can render rich formatting in its response to the user.

### 6.1 Response Structure

```
[SOURCES]
[1] {title} — {url}
    § {section_heading}
[2] {title} — {url}
    § {section_heading}

[EVIDENCE]
Source [1] (relevance: 0.85, confidence: 0.92):
{formatted chunk text}

Source [2] (relevance: 0.71):
{formatted chunk text}

[IMAGES]
- [Alt text](full_url) — Caption (from Source [1])
- [Alt text](full_url) (from Source [3])

[EXPANSION TRACE]
{ascii traversal diagram}

[STATS]
Mode: chunk (selective retrieval)
Documents searched: 5
Chunks evaluated: 47
Expansion iterations: 2
URLs ingested: 3
Stop reason: score plateau — top-10 variance below threshold
Total time: 2340ms (analysis: 180ms, retrieval: 890ms, reranking: 620ms, expansion: 450ms, locality: 120ms, merge: 80ms)
```

### 6.2 Section Details

#### [SOURCES] — Reference List

One entry per unique source URL across all chunks, numbered sequentially. A chunk's source number is determined by first-seen order (matching the chunk ordering from orchestration — grouped by `source_url`, then by `chunk_index`).

Format:
```
[N] {title or "Untitled"} — {full_url}
    § {section_heading}
```

If multiple chunks share the same `source_url` but have different `section_heading` values, list each heading:
```
[1] Ensemble Methods — https://scikit-learn.org/stable/modules/ensemble.html
    § Gradient Boosted Trees
    § Random Forests
```

#### [EVIDENCE] — Chunk Content

One block per chunk, ordered by reranked score descending. Each block includes:

- **Source reference number** (from [SOURCES]).
- **Relevance score** (`reranked_score`, 2 decimal places).
- **Confidence score** (if available from ZeroEntropy, shown as `confidence: X.XX`).
- **Sub-query attribution** (if multiple sub-queries, show which produced this chunk).
- **Locality indicator** (if `is_locality_expanded=True`, note `[adjacent context]`).
- **Formatted chunk text** (see §7 Rich Content Conversion).

Format:
```
Source [1] (relevance: 0.85, confidence: 0.92):
{formatted text — may be multi-line, includes tables, code blocks, etc.}

Source [2] (relevance: 0.71) [adjacent context]:
{formatted text}

Source [3] (relevance: 0.68, sub-query: "how does gradient boosting handle missing values"):
{formatted text}
```

#### [IMAGES] — Image References

Extracted from HTML-surface chunks during rich content conversion (see §7.5). Listed here so the reasoning model can reference them in its response. Only present if images were found.

Format:
```
- [Alt text](full_url) — Caption text (from Source [N])
- [Diagram of architecture](https://example.com/img/arch.png) (from Source [2])
```

#### [EXPANSION TRACE] — ASCII Traversal Diagram

Only rendered when `expansion_steps` is non-empty (i.e., expansion actually occurred).

**URL truncation rules:**
1. Find the common URL prefix across all URLs in the trace (e.g., `https://scikit-learn.org/stable/modules/`).
2. Display only the differing suffix for URLs that share the prefix.
3. Display full URLs for URLs from a completely different domain/path.

**Format:**
```
Base: https://scikit-learn.org/stable/modules/

/ensemble.html (seed, score: 0.63 → 0.63)
├── /gradient_boosting.html (depth 1, +12 chunks, score: 0.63 → 0.71, 1200ms)
│   └── /histogram_gradient_boosting.html (depth 2, +8 chunks, score: 0.71 → 0.74, 980ms)
├── /random_forests.html (depth 1, +6 chunks, score: 0.63 → 0.67, 1100ms)
└── https://xgboost.readthedocs.io/en/latest/ (external, depth 1, +4 chunks, 890ms)
    [stopped: score plateau]
```

Each node shows:
- Truncated URL (or full URL if external).
- Depth level.
- Chunks added (`+N chunks`).
- Score progression (`score: before → after`).
- Duration in ms.
- Failed URLs shown with `[failed: {reason}]`.
- Terminal decision shown as `[stopped: {reason}]` on the last node.

**Building the tree:** `ExpansionStep` records are flat (one per iteration). The tree structure is reconstructed from `depth` values and `source_url` to build parent-child relationships. The seed URL is always the root.

#### [STATS] — Summary Block

Always present. Provides the model with context about retrieval quality and performance.

```
Mode: {mode}
Documents searched: {corpus_stats.total_documents}
Documents matched: {corpus_stats.documents_matched | length}
Chunks evaluated: {corpus_stats.total_parent_chunks}
Expansion iterations: {total_iterations}
URLs ingested: {total_urls_ingested}
Stop reason: {final_decision.reason}
Total time: {timing.total_ms:.0f}ms (analysis: {timing.query_analysis_ms:.0f}ms, retrieval: {timing.retrieval_ms:.0f}ms, reranking: {timing.reranking_ms:.0f}ms, expansion: {timing.expansion_ms:.0f}ms, locality: {timing.locality_ms:.0f}ms, merge: {timing.merge_ms:.0f}ms)
```

Additional notes appended when relevant:
- `Mode: full_context (entire corpus fits within token budget)` — when mode is `"full_context"`.
- `Note: No expansion performed (single iteration).` — when `total_iterations == 0` and `expansion_budget != 0`.
- `Note: Expansion budget exhausted (reached {N} of {N} iterations).` — when expansion hit the budget limit.
- `Note: Expansion reached maximum depth ({N}). Results may benefit from a more specific query or a different seed URL.` — when depth equals `max_expansion_depth`.

### 6.3 Response Token Budget

The formatted response is governed by `mcp_response_token_budget` (default: 30,000 tokens, configurable). This is the **formatted output** budget, not the orchestration context budget.

**Budget allocation priority (highest to lowest):**

1. **[SOURCES]** — Always included in full. Typically small (< 500 tokens).
2. **[STATS]** — Always included in full. Tiny (< 200 tokens).
3. **[EVIDENCE]** — Fills remaining budget. Chunks are included in reranked-score order. When the budget is approached, stop adding chunks.
4. **[EXPANSION TRACE]** — Included if budget allows after evidence. Omitted with a note if truncated: `[Expansion trace omitted — {N} iterations, see stats above]`.
5. **[IMAGES]** — Included if budget allows after evidence.

If chunks are truncated:
```
[EVIDENCE]
... (showing 14 of 22 chunks — remaining 8 omitted due to response budget)
```

Token counting uses the same tokenizer configured for embeddings (`cl100k_base` / tiktoken by default) to avoid adding dependencies.

### 6.4 Verbatim Citations

The `CitationSpan` objects from orchestration contain verbatim text tied to source metadata. These are included at the bottom of the response so the model can quote directly with proper attribution:

```
[CITATIONS]
[1] "Gradient boosting builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions."
    — Ensemble Methods, https://scikit-learn.org/stable/modules/ensemble.html § Gradient Boosted Trees

[2] "The key advantage of histogram-based gradient boosting is that it bins continuous features into discrete buckets, reducing the number of splitting points."
    — Histogram Gradient Boosting, https://scikit-learn.org/stable/modules/ensemble.html § Histogram-Based Gradient Boosting
```

Each citation corresponds to a chunk and uses `CitationSpan.verbatim_text`. These give the model exact quotes it can use when answering the user, with full provenance.

### 6.5 Progress Notifications

MCP supports server-to-client notifications that appear in the client UI *during* tool execution. These are not part of the tool result — they're real-time status updates.

The MCP layer sends progress notifications at each major orchestration phase:

| Phase | Notification Text |
|-------|-------------------|
| Ingestion start | `"Checking if {url} is indexed..."` or `"Scraping {url}..."` |
| Ingestion complete | `"Indexed {url} ({N} chunks, {T} tokens)"` |
| Query analysis | `"Analyzing query — decomposed into {N} sub-queries"` |
| Retrieval | `"Searching {N} documents..."` |
| Reranking | `"Reranking {N} chunks..."` |
| Expansion decision | `"Expanding to linked pages (iteration {i}/{max})..."` |
| Expansion URL | `"Scraping {truncated_url}..."` |
| Locality expansion | `"Grabbing adjacent context..."` |
| Final merge | `"Assembling results ({N} chunks, {T} tokens)..."` |

**Implementation:** The MCP SDK's `Context` object (available in tool handlers via `ctx` parameter) provides `await ctx.report_progress(progress, total)` for numeric progress and logging for text notifications. The tool handler wraps `engine.run()` with a callback or observer pattern that emits notifications at each phase.

**Note:** This requires the orchestration engine to support some form of progress callback. The simplest approach: the MCP tool handler instruments the engine by wrapping key internal calls and emitting notifications between them. Since `engine.run()` is a single async call, the MCP layer cannot interleave notifications mid-execution unless the engine supports it. Two options:

1. **Phase-level notifications only:** Emit notifications before and after `engine.run()` (coarse-grained). Simple, no engine changes.
2. **Fine-grained notifications via callback:** Add an optional `progress_callback: Callable | None` parameter to `engine.run()` that the engine calls at each phase boundary. The MCP layer passes a callback that emits MCP notifications. This is the preferred approach but requires a small engine change.

The design doc recommends option 2 (callback) for v1. The callback signature:
```
async def on_progress(phase: str, detail: str, iteration: int | None = None, total: int | None = None) -> None
```

### 6.6 Streaming Responses

MCP supports streaming tool results via Server-Sent Events within the Streamable HTTP transport. This allows the user to see partial results as they become available rather than waiting for the entire orchestration pipeline to complete.

**Streaming strategy:**

The tool result is assembled incrementally. As each section becomes available, it's streamed to the client:

| Order | Section | Available after |
|-------|---------|-----------------|
| 1 | `[SOURCES]` (partial) | Initial retrieval + reranking |
| 2 | `[EVIDENCE]` (partial, top chunks) | Initial retrieval + reranking |
| 3 | `[EXPANSION TRACE]` (growing) | Each expansion iteration |
| 4 | `[SOURCES]` (updated) | Expansion adds new sources |
| 5 | `[EVIDENCE]` (updated, more chunks) | Post-expansion reranking |
| 6 | `[IMAGES]` | Final formatting |
| 7 | `[CITATIONS]` | Final formatting |
| 8 | `[STATS]` | Pipeline complete |

**Implementation considerations:**

- Streaming requires the orchestration engine to yield intermediate results rather than returning a single `OrchestrationResult` at the end. This is a significant engine change.
- For v1, a pragmatic approach: stream progress notifications (§6.5) during execution, then send the full formatted result at the end. This gives the user real-time visibility without requiring engine changes.
- For v2, refactor `engine.run()` to yield `OrchestrationResult` snapshots at phase boundaries (after initial retrieval, after each expansion iteration, after final merge). The MCP layer formats and streams each snapshot incrementally.

**stdio limitation:** stdio transport does not support streaming tool results. Progress notifications work over stdio, but the tool result is delivered as a single message. Streaming is only available with Streamable HTTP transport.

---

## 7. Rich Content Conversion

When `RetrievedChunk.surface == "html"`, the `selected_text` contains raw HTML. The MCP layer converts this to model-readable plain text. When `surface == "markdown"`, the text passes through as-is.

### 7.1 HTML Tables → Text Tables

Convert `<table>` elements to aligned text tables. Use simple column-aligned format:

```
| Column A     | Column B  | Column C |
|--------------|-----------|----------|
| Value 1      | 42        | Yes      |
| Value 2      | 17        | No       |
```

**Implementation:** Use BeautifulSoup (already a project dependency) to parse `<table>`, extract `<th>` and `<td>` cells, and format as pipe-delimited text with padding. Handle `colspan`/`rowspan` on a best-effort basis (flatten to the simplest readable representation).

### 7.2 Code Blocks

Convert `<pre><code>` elements to fenced code blocks:

````
```python
def gradient_boost(X, y, n_estimators=100):
    ...
```
````

**Language detection:** Check for `class="language-*"` or `class="highlight-*"` attributes on the `<code>` or `<pre>` element. Fall back to no language hint if not detectable.

### 7.3 MathML → LaTeX

Convert `<math>` elements to LaTeX notation wrapped in `$...$` (inline) or `$$...$$` (display). This is best-effort — MathML to LaTeX conversion is lossy.

**Strategy:**
1. Check for an `alttext` attribute on the `<math>` element (many renderers include LaTeX source here). Use it if available.
2. If no alttext, attempt basic structural conversion of common MathML elements (`<mfrac>` → `\frac{}{}`, `<msup>` → `^{}`, `<msqrt>` → `\sqrt{}`, etc.).
3. Fall back to extracting the text content of the `<math>` element as readable text.

### 7.4 Definition Lists

Convert `<dl>` / `<dt>` / `<dd>` elements to a readable format:

```
**Term 1**: Definition text for term 1.

**Term 2**: Definition text for term 2.
```

### 7.5 Image Extraction

Extract image metadata from `<img>` tags within HTML chunks:

- `src` attribute → full URL (resolve relative URLs against `source_url`).
- `alt` attribute → alt text.
- Nearby `<figcaption>` text → caption.

Images are not embedded in the tool response (MCP custom connectors don't support image content). Instead, image metadata is collected and rendered in the `[IMAGES]` section (see §5.2).

### 7.6 Admonitions and Callouts

Convert admonition-style HTML (e.g., `<div class="admonition warning">`, `<div class="note">`, GitHub-style alerts) to labeled text blocks:

```
⚠️ WARNING: This method is deprecated in version 1.2. Use `new_method()` instead.
```

```
ℹ️ NOTE: This only applies when `n_estimators > 100`.
```

### 7.7 Fallback: Generic HTML Stripping

For any HTML element not covered by the above rules, strip tags and preserve text content. Collapse excessive whitespace. Preserve paragraph boundaries as double newlines.

### 7.8 Conversion Pipeline

The converter processes `selected_text` through a pipeline:
1. Parse with BeautifulSoup.
2. Walk the DOM tree.
3. Apply specific converters for recognized elements (table, pre/code, math, dl, img, admonition divs).
4. Strip remaining tags, preserving text.
5. Clean up whitespace.

Return the converted text and a list of extracted image metadata (URL, alt, caption, source reference number).

---

## 8. Error Handling

### 8.1 Design Principle

**Fail transparently.** The reasoning model should always receive a text response — either formatted results or a clear error explanation. The model can then inform the user of what went wrong rather than hallucinating an answer.

### 8.2 Error Categories

#### Full Failure

`OrchestratorEngine.run()` raises an exception (API down, database unreachable, ingestion failed).

**Response:**
```
[ERROR]
WebRAG encountered an error during retrieval.

Error type: {exception class name}
Details: {exception message}

The knowledge retrieval service was unable to complete this query. Possible causes:
- The URL may be unreachable or return an error.
- An external API (embedding, reranker) may be temporarily unavailable.
- The database connection may have failed.

Please inform the user of this error. Do not attempt to answer from memory — the tool was called because specific source material was needed.
```

The tool response is marked as `is_error=True` in the MCP response so the model knows to treat it as an error.

#### Timeout

Orchestration exceeds `mcp_tool_timeout` (default: 120 seconds).

**Response:**
```
[ERROR]
WebRAG timed out after {timeout}s.

The orchestration pipeline did not complete within the allowed time. This may indicate:
- A very large page requiring extensive indexing.
- Many expansion iterations with slow external APIs.
- Network latency to external services.

Suggestion: Retry with expansion_budget=0 to skip expansion, or try a more specific URL.
```

#### Empty Results

Orchestration completes but returns zero chunks.

**Response:** Normal format with an advisory note:
```
[SOURCES]
(none)

[EVIDENCE]
No relevant content was found for this query in the specified URL.

[STATS]
Mode: chunk
Documents searched: 1
Chunks evaluated: 12
Expansion iterations: 0
Stop reason: No chunks above relevance threshold.
Total time: 450ms
```

#### Partial Degradation

Orchestration returns results but with quality concerns (e.g., all scores below a low threshold, reranker fell back to retrieval scores). This is handled within orchestration already — the MCP layer formats whatever `OrchestrationResult` it receives. Quality signals in `[STATS]` (stop reason, scores, mode) give the model enough context to judge reliability.

### 8.3 Timeout Implementation

Wrap the `engine.run()` call with `asyncio.wait_for()`:

```
result = await asyncio.wait_for(
    engine.run(url, query, ...),
    timeout=settings.mcp_tool_timeout,
)
```

Catch `asyncio.TimeoutError` and return the timeout error response.

---

## 9. Security

### 9.1 stdio Transport

No network exposure. The MCP client (Claude Desktop) spawns the server as a subprocess. Security is inherent — only the local user's MCP client can communicate with it.

### 9.2 Streamable HTTP Transport

When exposed over HTTP, the server should be protected:

**Primary protection: Network-level security.** Deploy behind existing infrastructure:
- WireGuard VPN (only accessible from VPN peers).
- Cloudflare Access (identity-aware proxy).
- nginx reverse proxy with TLS termination.

**Secondary protection: Optional bearer token.** Set `MCP_AUTH_TOKEN` as an environment variable. When set, the server validates that incoming requests include a matching `Authorization: Bearer {token}` header. When unset, the server runs authless (relying on network-level security).

This matches Claude's custom connector model — connectors support optional OAuth or bearer token auth. For a self-hosted personal server, network-level security is usually sufficient, with the bearer token as an extra layer.

### 9.3 Input Validation

All tool parameters are validated by FastMCP's schema validation (Pydantic-backed). The `url` parameter is passed directly to orchestration's `_ensure_ingested()`, which handles it via Firecrawl. No additional URL sanitization is needed at the MCP layer — Firecrawl handles the actual HTTP request.

---

## 10. Configuration

### 10.1 New Settings

Add to the existing `Settings` class in `config.py`:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `mcp_transport` | `MCP_TRANSPORT` | `"stdio"` | Transport: `"stdio"` or `"streamable-http"` |
| `mcp_host` | `MCP_HOST` | `"0.0.0.0"` | Bind address for Streamable HTTP |
| `mcp_port` | `MCP_PORT` | `8765` | Port for Streamable HTTP |
| `mcp_auth_token` | `MCP_AUTH_TOKEN` | `None` | Optional bearer token. If set, requests must include it. |
| `mcp_response_token_budget` | `MCP_RESPONSE_TOKEN_BUDGET` | `30000` | Max tokens for formatted tool response |
| `mcp_tool_timeout` | `MCP_TOOL_TIMEOUT` | `120` | Seconds before tool call times out |
| `mcp_log_level` | `MCP_LOG_LEVEL` | `"INFO"` | Logging verbosity for MCP layer |

### 10.2 Existing Settings Reused

- `database_url` — Passed through to `OrchestratorEngine` (which reads it from `settings` directly).
- `embedding_tokenizer_kind` / `embedding_tokenizer_name` — Reused for token counting in the response budget logic.
- All orchestration settings — Engine reads them directly from `settings`.

---

## 11. Module Structure

```
src/05_mcp_server/
├── server.py           # FastMCP setup, lifespan, transport config, entry point
├── tools.py            # Tool definitions (answer, search, status handlers)
├── formatter.py        # OrchestrationResult → structured text response
├── html_converter.py   # HTML surface → readable text (tables, code, math, images)
└── errors.py           # Error response formatting
```

### 11.1 Module Responsibilities

**`server.py`** — Entry point. Creates `FastMCP` instance, registers the lifespan context manager (engine start/stop), registers tools, and runs the server with the configured transport. Contains `__main__` block and CLI argument parsing.

**`tools.py`** — Defines all tool functions (`answer`, `search`, `status`). The `answer` handler calls `engine.run()` with timeout wrapping and progress notifications, handles multi-URL pre-ingestion, and passes the `OrchestrationResult` to the formatter. The `search` handler calls `retrieve()` + `rerank()` directly. The `status` handler queries the database. All handlers handle their error categories (§8) and return appropriate MCP responses.

**`formatter.py`** — The core formatting logic. Takes `OrchestrationResult` and produces the final text response. Orchestrates the section assembly: [SOURCES], [EVIDENCE], [IMAGES], [EXPANSION TRACE], [CITATIONS], [STATS]. Manages the token budget, deciding how many chunks to include. Builds the ASCII traversal diagram from `ExpansionStep` records.

**`html_converter.py`** — Stateless conversion functions for HTML → readable text. One function per element type (tables, code blocks, MathML, definition lists, admonitions, images). Also extracts image metadata. Called by `formatter.py` when processing chunks with `surface="html"`.

**`errors.py`** — Error response templates. Functions that produce formatted error text for each error category (full failure, timeout, empty results). Keeps error formatting consistent and testable.

### 11.2 Dependency Graph

```
server.py
  └── tools.py
        ├── formatter.py
        │     └── html_converter.py
        ├── errors.py
        ├── OrchestratorEngine (from src.orchestration.engine)  ← answer tool
        ├── retrieve(), rerank() (from src.retrieval/orchestration) ← search tool
        └── database queries (via engine pool)                   ← status tool
  └── Settings (from config)
```

### 11.3 pyproject.toml Update

Add the `mcp` package dependency and the new module mapping:

```toml
# In [project] dependencies, add:
"mcp[cli]",

# In [tool.setuptools.package-dir], add:
"src.mcp_server" = "src/05_mcp_server"
```

---

## 12. Logging

### 12.1 stdio Transport Constraint

**Critical:** When using stdio transport, the server MUST NOT write to stdout — stdout is reserved for MCP protocol messages. All logging must go to stderr.

Configure Python logging to write to stderr:
```python
logging.basicConfig(stream=sys.stderr, level=settings.mcp_log_level)
```

This also applies to any library that might print to stdout. Suppress or redirect accordingly.

### 12.2 Log Points

Key events to log at INFO level:
- Server startup (transport, host/port if HTTP, engine started).
- Tool call received (tool name, URL, query — truncated).
- Orchestration completed (timing, chunk count, mode).
- Tool response sent (response size in tokens).
- Server shutdown.

At DEBUG level:
- Full tool parameters.
- Formatted response preview (first 500 chars).
- Token budget allocation breakdown.
- Individual section sizes.

At WARNING level:
- Timeout approaching (e.g., at 80% of budget).
- Empty results.
- HTML conversion failures (fallback to text stripping).

At ERROR level:
- Engine start failure.
- Orchestration exceptions.
- Timeout.

---

## 13. Testing

### 13.1 Unit Tests

**Formatter tests** (`test_formatter.py`):
- Build mock `OrchestrationResult` objects with known chunks, citations, expansion steps.
- Verify the formatted output contains expected sections, source numbering, score formatting.
- Test token budget truncation — ensure chunks are dropped in correct order.
- Test edge cases: zero chunks, single chunk, full-context mode, no expansion.

**HTML converter tests** (`test_html_converter.py`):
- Test each converter function in isolation: table → text table, code → fenced block, MathML → LaTeX, definition list → formatted text, image → metadata extraction.
- Test nested/complex HTML structures.
- Test fallback behavior for unrecognized elements.

**Traversal diagram tests** (`test_formatter.py` or separate):
- Build mock `ExpansionStep` lists and verify ASCII tree output.
- Test URL truncation logic — common prefix detection, external URL handling.
- Test single-depth, multi-depth, and failed-expansion cases.

**Error formatting tests** (`test_errors.py`):
- Verify error response templates for each error category.
- Ensure `is_error` flag is set correctly.

### 13.2 Integration Tests

**MCP client test** (`test_integration.py`):
- Use the `mcp` SDK's client to connect to the server (stdio transport).
- Call `list_tools()` and verify `answer`, `search`, and `status` tool schemas are correct.
- Call the `answer` tool with a known URL and query — verify well-formed response with expected sections.
- Call the `status` tool — verify corpus stats are returned.
- Call `answer` then `search` on the same corpus — verify `search` returns results without re-ingesting.
- Test multi-URL `answer` call — verify both URLs are ingested and searched.
- Gate behind a marker (e.g., `@pytest.mark.integration`) since it requires a running database, API keys, and network access.

**MCP Inspector**:
- The `mcp` CLI includes `mcp dev` which launches a web UI for interactive testing.
- Not automated, but useful for manual verification during development.

### 13.3 Test Fixtures

Create reusable fixtures that build `OrchestrationResult` objects with:
- Varying numbers of chunks (0, 1, 5, 20).
- Mixed surfaces (html and markdown).
- With and without expansion steps.
- With and without citations.
- Full-context vs chunk mode.
- Various error scenarios.

---

## 14. Implementation Notes

### 14.1 Platform Compatibility

The MCP layer must work on both Windows and Linux:
- **Windows:** stdio transport is primary. The engine uses direct `AsyncConnection` (no pool). The event loop is `SelectorEventLoop`. No special handling needed in the MCP layer — the engine manages this internally.
- **Linux:** Both transports work. The engine uses `AsyncConnectionPool` for efficient connection reuse.

### 14.2 Import Path

The engine currently uses `from config import settings` (not a package-relative import). The MCP server module will need to ensure the Python path includes the project root so that `config` and `src.*` imports resolve correctly. The entry point should handle this (e.g., `sys.path` adjustment or running from the project root).

### 14.3 Concurrency

The MCP server handles one tool call at a time per client session (MCP is request-response). However, with Streamable HTTP and `stateless_http=True`, multiple clients could call the tool concurrently. The engine's connection pool (on Linux) handles concurrent database access. The orchestration pipeline is async and re-entrant. No additional concurrency controls are needed in the MCP layer.

### 14.4 Image Survival Through Reranking

Images embedded in HTML chunks survive the reranking pipeline because they're part of `selected_text`. The reranker scores the full passage text — `<img>` tags are just part of the HTML content. The MCP layer's HTML converter extracts image metadata during formatting. No special reranker handling is needed.

---

## 15. Future Work

Items deferred from v1 but architecturally anticipated:

- **`ingest` tool** — Explicitly scrape and index a URL without querying. Calls `ingest()` + `index_batch()` directly. Trivial to add.
- **Direct image support** — If MCP custom connectors add image content support, upgrade the [IMAGES] section to pass images inline rather than as URL references.
- **Full streaming responses** — v1 streams progress notifications but delivers the tool result as a single message. v2 should refactor `engine.run()` to yield intermediate `OrchestrationResult` snapshots, enabling incremental streaming of [SOURCES] and [EVIDENCE] as they become available.
- **Conversation-aware corpus** — Track which URLs were ingested in the current conversation so the model can make smarter `search` vs `answer` decisions without calling `status` each time.

---

## 16. Summary

The MCP layer is the thinnest layer in the WebRAG stack. It:
1. Manages the engine lifecycle (start/stop).
2. Defines three tools: `answer` (full orchestration with multi-URL support), `search` (fast corpus query), and `status` (corpus inspection).
3. Formats `OrchestrationResult` into structured text with citations, rich content, traversal diagrams, and stats — acting as a research brief for the reasoning model.
4. Sends real-time progress notifications during orchestration so the user sees what's happening.
5. Handles errors transparently, surfacing them as readable text so the model can inform the user rather than hallucinate.
6. Supports both stdio (default, for local use) and Streamable HTTP (for remote/connector use) transports.

The formatting logic in `formatter.py` and `html_converter.py` is the bulk of the implementation work. The `search` and `status` tools are lightweight — they bypass orchestration and talk directly to the retrieval layer and database respectively. Progress notifications require a small callback addition to the orchestration engine.
