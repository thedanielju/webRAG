<h1 align="center">🌐 WebRAG</h1>

<p align="center">
  <strong>Give your LLM a research memory.</strong><br>
  WebRAG indexes web pages and retrieves cited evidence so AI models can reason over real sources - not training data.
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white">
  <img alt="MCP" src="https://img.shields.io/badge/protocol-MCP-green">
  <img alt="Postgres + pgvector" src="https://img.shields.io/badge/database-pgvector-336791?logo=postgresql&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## What is WebRAG?

WebRAG is a **Model Context Protocol (MCP)** server that turns web pages into searchable, citable knowledge for LLMs like Claude and GPT. Point it at a URL, ask a question, and get back evidence-grounded answers with source citations.

**How it works:**
1. You ask your LLM a question and provide a URL
2. The LLM calls WebRAG's `answer` tool via MCP
3. WebRAG scrapes the page, chunks it, embeds it, and stores it in Postgres
4. It retrieves the most relevant passages and can optionally follow links to expand the corpus
5. The LLM receives cited evidence - verbatim quotes with source URLs - and reasons over it

**Why not just paste the page into the chat?**
- Pages can be too large for context windows
- Multi-page research requires following links and building a corpus over time
- WebRAG handles chunking, embedding, retrieval, reranking, and citation extraction automatically
- The corpus persists between conversations; ask follow-up questions without re-scraping
- Higher quality, informed responses with rich html content and relevant image links. 

---

## Features

| Feature | Description |
|---------|-------------|
| **MCP Integration** | Three tools (`answer`, `search`, `status`) exposed via the Model Context Protocol. Fast defaults plus explicit deep/full-context controls. Works with Claude Desktop, Cursor, and any MCP-compatible client. |
| **Smart Corpus Expansion** | Default MCP behavior is no expansion for speed. Deep expansion is available on demand and uses link candidate scoring (URL heuristics, title/description relevance) to avoid low-yield crawls. |
| **Semantic Chunking** | Splits pages by heading structure into parent/child chunks. Parents provide context; children are embedded for precise ANN search. |
| **Rich Content Handling** | Preserves tables, code blocks, math (MathML → LaTeX), and images through HTML surface detection. The formatter converts these to clean, readable text for the model. |
| **Provider-Agnostic Reranking** | Supports ZeroEntropy, Cohere, Jina, or no reranking. Dramatically improves result quality by rescoring passages with a cross-encoder. |
| **Query Decomposition** | Complex questions are split into sub-queries (via LLM or rule-based patterns) and retrieved concurrently. Results are merged with MMR deduplication. |
| **Citation Fidelity** | Verbatim quotes reconstructed from stored character offsets. No paraphrasing - every citation maps to exact source text. |
| **Intelligent Stopping** | 11-rule decision matrix evaluates score distributions to decide when to stop expanding. Token budget saturation, plateau detection, and diminishing returns - not arbitrary depth limits. |
| **Persistent Memory** | Indexed content lives in Postgres. Ask follow-up questions hours later without re-scraping. |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/thedanielju/webRAG.git
cd webRAG
pip install -e .
```

### 2. Start the database

WebRAG stores indexed knowledge in Postgres with pgvector. The included Docker Compose file sets everything up:

```bash
docker compose up -d
```

Verify it's running:

```bash
docker compose ps
# Should show webrag-postgres as "healthy"
```

> **Default connection:** `postgresql://webrag:webrag@localhost:5432/webrag`

### 3. Configure your `.env`

Copy the template and fill in your API keys:

```bash
cp blank.env .env
```

**Required keys:**

| Key | What it's for | Where to get it |
|-----|---------------|-----------------|
| `FIRECRAWL_API_KEY` | Web scraping via Firecrawl | [firecrawl.dev](https://firecrawl.dev) |
| `DATABASE_URL` | Postgres connection | Pre-filled for Docker setup |
| `EMBEDDING_API_KEY` | Text embeddings | [OpenAI API keys](https://platform.openai.com/api-keys) |

**Optional keys (improve quality):**

| Key | What it's for | Default without it |
|-----|---------------|--------------------|
| `RERANKER_API_KEY` | Cross-encoder reranking | No reranking (embedding similarity only) |
| `ORCHESTRATION_LLM_API_KEY` | LLM-based query decomposition | Falls back to rule-based decomposition |

### 4. Connect to your MCP client

#### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "webrag": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/webRAG"
    }
  }
}
```

#### Cursor

Add to your MCP settings:

```json
{
  "webrag": {
    "command": "python",
    "args": ["-m", "src.mcp_server.server"],
    "cwd": "/path/to/webRAG"
  }
}
```

#### HTTP mode (remote/hosted)

```bash
python -m src.mcp_server.server --transport streamable-http
# Listens on 0.0.0.0:8765 by default
```

---

## MCP Tools

### `answer` - Full research pipeline

Scrapes a URL (if needed), decomposes your question, retrieves and reranks evidence, optionally expands to linked pages, and returns cited results.

Default MCP behavior is optimized for responsiveness:
- chunked retrieval
- no expansion unless explicitly requested
- tool may return a recommendation asking the model to ask the user before a slower deep pass

```
answer(
    url: "https://docs.python.org/3/library/asyncio.html",
    query: "How does asyncio handle cancellation?",
    research_mode: "fast",      # optional: fast | auto | deep (default: fast)
    retrieval_mode: "chunk"     # optional: chunk | full_context | auto (default: chunk)
)
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `url` | `str` or `list[str]` | URL(s) to research. First URL is the primary seed; additional URLs are pre-ingested. |
| `query` | `str` | Your question about the content. |
| `intent` | `str` (optional) | Hint for the reranker (e.g. "compare", "explain", "find examples"). |
| `known_context` | `str` (optional) | Context the model already has, to avoid redundant retrieval. |
| `constraints` | `list[str]` (optional) | Hard constraints on what to include/exclude. |
| `research_mode` | `str` (optional) | `fast`, `auto`, or `deep`. `fast` disables expansion by default. `deep` enables slower multi-page expansion from the start. |
| `retrieval_mode` | `str` (optional) | `chunk`, `full_context`, or `auto`. MCP defaults to `chunk` for better speed and citation visibility. |
| `expansion_budget` | `int` (optional) | Max expansion iterations (overrides `MAX_EXPANSION_DEPTH`). Set to 0 to skip expansion. |

### `search` - Fast corpus query

Searches already-indexed content without scraping. Use this for follow-up questions about material you've already asked about.

```
search(
    query: "What are the best practices for task groups?",
    source_urls: ["https://docs.python.org/3/library/asyncio.html"],
    retrieval_mode: "chunk"  # optional: chunk | full_context | auto
)
```

`search` also defaults to `chunk` mode so results stay compact and citations are more likely to survive the MCP response budget.

### `status` — Corpus introspection

Check what WebRAG has indexed:

```
status()                                    # Everything indexed
status(source_url: "https://example.com")   # Specific URL
```

---

## Configuration Reference

All settings live in `config.py` and can be overridden via environment variables in `.env`. See `blank.env` for a fully commented template.

### Embeddings

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible embedding endpoint. Change for local models (Ollama, LM Studio). |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name. |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimensions. Must match your model's output. |
| `EMBEDDING_TOKENIZER_KIND` | `tiktoken` | `tiktoken` (OpenAI) or `huggingface` (local models). |
| `EMBEDDING_TOKENIZER_NAME` | `cl100k_base` | Tokenizer identifier for chunk sizing. |
| `EMBEDDING_BATCH_SIZE` | `256` | Texts per embedding API request. |
| `EMBEDDING_MAX_WORKERS` | `4` | Concurrent embedding batches. Keep low for OpenAI (rate limits); increase for local servers. |

<details>
<summary><strong>Using local embeddings (Ollama, LM Studio)</strong></summary>

```env
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_API_KEY=
EMBEDDING_DIMENSIONS=768
EMBEDDING_TOKENIZER_KIND=huggingface
EMBEDDING_TOKENIZER_NAME=nomic-ai/nomic-embed-text-v1
```

Local servers aren't rate-limited, so you can safely increase `EMBEDDING_MAX_WORKERS` to 8–12.

</details>

### Chunking

| Setting | Default | Description |
|---------|---------|-------------|
| `CHILD_TARGET_TOKENS` | `256` | Target size for child chunks (embedded for search). |
| `PARENT_MAX_TOKENS` | `1000` | Max size for parent chunks (returned as context). |

### Retrieval

| Setting | Default | Description |
|---------|---------|-------------|
| `RETRIEVAL_FULL_CONTEXT_THRESHOLD` | `30000` | Below this corpus size (tokens), return everything instead of searching. |
| `RETRIEVAL_CONTEXT_BUDGET` | `40000` | Max tokens of context to return. |
| `RETRIEVAL_TOP_K_CHILDREN_LIMIT` | `60` | Max child chunks from HNSW before parent aggregation. |
| `RETRIEVAL_SIMILARITY_FLOOR` | `0.3` | Min cosine similarity for a child hit to survive filtering. |
| `RETRIEVAL_HNSW_EF_SEARCH` | `100` | pgvector HNSW recall knob (higher = better recall, slightly slower). |

### Reranking

| Setting | Default | Description |
|---------|---------|-------------|
| `RERANKER_PROVIDER` | `zeroentropy` | `zeroentropy`, `cohere`, `jina`, or `none`. |
| `RERANKER_API_KEY` | — | API key for your chosen reranker provider. |
| `RERANKER_MODEL` | `zerank-2` | Model identifier (provider-specific). |
| `RERANKER_TOP_N` | `20` | Max passages sent to the reranker per sub-query. |

### Orchestration

| Setting | Default | Description |
|---------|---------|-------------|
| `DECOMPOSITION_MODE` | `llm` | `llm` (best quality), `rule_based` (no API calls), or `none`. |
| `MAX_EXPANSION_DEPTH` | `5` | Hard cap on expansion iterations. |
| `MAX_CANDIDATES_PER_ITERATION` | `5` | Links to scrape per expansion round. |
| `LOCALITY_EXPANSION_ENABLED` | `true` | Grab sibling chunks adjacent to high-scoring hits. |

### MCP Server

| Setting | Default | Description |
|---------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | `stdio` (desktop clients) or `streamable-http` (remote). |
| `MCP_HOST` | `0.0.0.0` | Bind address for HTTP transport. |
| `MCP_PORT` | `8765` | Port for HTTP transport. |
| `MCP_RESPONSE_TOKEN_BUDGET` | `30000` | Soft ceiling on response size. Higher = more evidence, more tokens. |
| `MCP_DEFAULT_RESEARCH_MODE` | `fast` | Default MCP behavior. `fast` disables expansion unless explicitly requested. |
| `MCP_DEFAULT_RETRIEVAL_MODE` | `chunk` | Default retrieval mode for MCP tools. Keeps responses smaller and improves citation visibility. |
| `MCP_CITATIONS_RESERVED_TOKENS` | `1500` | Reserved response budget for `[CITATIONS]`. |
| `MCP_IMAGES_RESERVED_TOKENS` | `500` | Reserved response budget for `[IMAGES]`. |
| `MCP_TOOL_TIMEOUT` | `120` | Seconds before the `answer` tool times out. |
| `MCP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, or `WARNING`. |

---

## Project Structure

```
webRAG/
├── config.py                   # Centralized settings (pydantic-settings)
├── blank.env                   # Annotated config template
├── pyproject.toml              # Dependencies and package mappings
├── docker-compose.yml          # Postgres + pgvector container
│
├── src/
│   ├── 01_ingestion/           # Layer 1: Web scraping and content extraction
│   │   ├── service.py          #   Top-level ingest(url) → NormalizedDocument
│   │   ├── firecrawl_client.py #   Firecrawl API wrapper
│   │   └── links.py            #   URL normalization and dedup
│   │
│   ├── 02_indexing/            # Layer 2: Chunking, embedding, storage
│   │   ├── chunker.py          #   Semantic chunking (parent/child)
│   │   ├── embedder.py         #   Async batch embedding
│   │   ├── indexer.py          #   Pipeline: dedup → chunk → embed → store
│   │   ├── models.py           #   Chunk, RichContentFlags, etc.
│   │   └── schema.py           #   Postgres DDL (documents, chunks, indexes)
│   │
│   ├── 03_retrieval/           # Layer 3: Search and citation extraction
│   │   ├── search.py           #   retrieve(conn, query) → RetrievalResult
│   │   ├── citations.py        #   Verbatim quote extraction
│   │   └── models.py           #   RetrievedChunk, CorpusStats, etc.
│   │
│   ├── 04_orchestration/       # Layer 4: Retrieve-evaluate-expand loop
│   │   ├── engine.py           #   OrchestratorEngine (main entry point)
│   │   ├── query_analyzer.py   #   Query decomposition (LLM/rule-based)
│   │   ├── reranker.py         #   Provider-agnostic reranking
│   │   ├── evaluator.py        #   11-rule quality decision matrix
│   │   ├── expander.py         #   Link scoring and corpus expansion
│   │   ├── locality.py         #   Adjacent chunk expansion
│   │   ├── merger.py           #   Sub-query merge + MMR dedup
│   │   └── models.py           #   OrchestrationResult, RankedChunk, etc.
│   │
│   └── 05_mcp_server/          # Layer 5: MCP interface
│       ├── server.py           #   FastMCP setup, lifespan, entry point
│       ├── tools.py            #   answer, search, status tool handlers
│       ├── formatter.py        #   Result → structured text with token budget
│       ├── html_converter.py   #   HTML → readable plain text
│       └── errors.py           #   Error response templates
│
├── tests/                      # Unit and integration tests (~170 tests)
├── docs/
│   └── architecture.md         # Detailed architecture documentation
└── specs/                      # Design specifications
```

---

## How the Pipeline Works

```
                    ┌─────────────────────────────────┐
                    │         MCP Client (LLM)        │
                    │   Claude Desktop / Cursor / etc │
                    └──────────────┬──────────────────┘
                                   │ answer(url, query)
                    ┌──────────────▼──────────────────┐
                    │        MCP Server (Layer 5)     │
                    │   tools.py → formatter.py       │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     Orchestration (Layer 4)     │
                    │                                 │
                    │  ┌─ Ingest seed URL             │
                    │  ├─ Decompose query             │
                    │  ├─ Retrieve + Rerank           │
                    │  ├─ Evaluate quality signals    │
                    │  ├─ Expand corpus (if needed)──┐│
                    │  │  ├─ Score candidate links   ││
                    │  │  ├─ Scrape top picks        ││
                    │  │  └─ Re-index + re-retrieve  ││
                    │  ├─────────────────────────────┘│
                    │  ├─ Locality expansion          │
                    │  └─ Merge + dedup + cite        │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
         ┌────────────┐     ┌────────────┐      ┌────────────┐
         │ Retrieval  │     │  Indexing  │      │ Ingestion  │
         │ (Layer 3)  │     │ (Layer 2)  │      │ (Layer 1)  │
         │ ANN search │     │ Chunk+Embed│      │ Firecrawl  │
         └──────┬─────┘     └──────┬─────┘      └──────┬─────┘
                │                  │                   │
                └──────────────────┼───────────────────┘
                                   ▼
                        ┌─────────────────────────┐
                        │   Postgres + pgvector   │
                        │   documents | chunks    │
                        │   HNSW vector index     │
                        └─────────────────────────┘
```
---
