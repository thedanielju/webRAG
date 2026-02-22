# WebRAG Setup

WebRAG is stateful. It stores indexed knowledge in Postgres (`pgvector`), so local setup includes one database container.

## 1) Clone and install

```bash
pip install -e .
```

## 2) Create `.env`

Copy `blank.env` to `.env`, then fill required fields.

Required now:
- `FIRECRAWL_API_KEY`
- `DATABASE_URL` (must use `postgresql://...` for `psycopg.connect`)
- `EMBEDDING_API_KEY` (if using OpenAI embeddings)

Default Docker DB URL (matches included compose file):

```env
DATABASE_URL=postgresql://webrag:webrag@localhost:5432/webrag
```

## 3) Start database

```bash
docker compose up -d
```

Check health:

```bash
docker compose ps
```

You should see `webrag-postgres` as `healthy`.

## 4) Notes on embeddings

OpenAI (default):
- `EMBEDDING_BASE_URL=https://api.openai.com/v1`
- `EMBEDDING_MODEL=text-embedding-3-small`
- `EMBEDDING_API_KEY=...`
- `EMBEDDING_DIMENSIONS=1536`
- `EMBEDDING_TOKENIZER_KIND=tiktoken`
- `EMBEDDING_TOKENIZER_NAME=cl100k_base`

Local endpoint is supported too (OpenAI-compatible base URL). Update model, dimensions, and tokenizer fields accordingly.

## 5) Orchestration setup

The orchestration layer handles corpus expansion (retrieve → evaluate → expand loop). Most defaults work out of the box; configure these as needed:

**Reranker** (optional — improves result quality):
- `RERANKER_PROVIDER` — `zeroentropy`, `cohere`, `jina`, or `none` (default: `none`).
- Set the matching API key: `ZEROENTROPY_API_KEY`, `COHERE_API_KEY`, or `JINA_API_KEY`.

**Query decomposition** (optional — helps with complex queries):
- `DECOMPOSITION_MODE` — `llm`, `rule_based`, or `none` (default: `rule_based`).
- If using `llm` mode: set `ORCHESTRATION_LLM_API_KEY` and optionally `ORCHESTRATION_LLM_BASE_URL` / `ORCHESTRATION_LLM_MODEL`.

**Expansion tuning**:
- `MAX_EXPANSION_DEPTH` — hard cap on expansion iterations (default: 3).
- `RETRIEVAL_CONTEXT_BUDGET` — target token count for the context window (default: 4096).
- `MAX_CANDIDATES_PER_ITERATION` — how many links to scrape per expansion round (default: 3).

All orchestration settings are documented in `blank.env`.

## 6) MCP usage flow

Typical local flow:
1. Clone repo
2. Copy `blank.env` to `.env` and fill keys
3. `docker compose up -d`
4. Point `claude_desktop_config.json` to the MCP server entry point
