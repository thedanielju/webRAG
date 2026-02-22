## SYSTEM OVERVIEW

WebRAG is a research memory system that enables LLMs to reason over evolving web corpora by if necessary, recursively and selectively expanding, indexing, retrieving, and citing evidence from web sources. The system separates:

Reasoning layer → chat model (ChatGPT, Claude, OpenWebUI)

Knowledge layer → WebRAG memory system

The model queries WebRAG via MCP tools, and WebRAG performs corpus construction, retrieval, and citation grounding.

## SYSTEM GOALS

WebRAG is designed to:
1. Attach websites as structured knowledge corpora
2. Support iterative corpus expansion via link and sublink traversal
3. Retrieve evidence-grounded passages for reasoning
4. Produce citation-grade quotes tied to source URLs
5. Maintain an evolving research memory over time (not sure on this)

## HIGH LEVEL ARCHITECTURE
1. User Query
   ↓
2. LLM (Reasoning layer)
   ↓
3. WebRAG MCP tools
   ↓
4. WebRAG Orchestrator
   ↓
5. Retrieval → Expansion → Re-index → Retrieval
   ↓
6. Answer + Evidence citations

## CORE ARCHITECTURAL PRINCIPLES
1. Firecrawl solves the reliable acquiistion of web content and is integrated in WebRAG
2. WebRAG reasons over evolving web corpora - key features include selective recursion, corpus growth strategy, retrieval policy, citation rigor, and stopping criteria

## CONFIGURATION LAYER
Configuration is centralized in `config.py` and imported by feature modules.

Current ingestion-related config includes:
- `FIRECRAWL_API_KEY` loading/validation
- Firecrawl default scrape options (formats, PDF parser, timeout, proxy, ad/image cleanup)
- default map/discovery limits used by ingestion wrappers

Principle:
- configuration and secrets live in one place
- ingestion/indexing/retrieval modules only consume typed constants and do not read env vars directly

## INGESTION LAYER
Responsibilities:
- Fetch pages permitted for retrieval
- Extract primary content
- Discover and normalize sublinks
- Prevent duplicate ingestion

Components:
1. Ingest: Managed by firecrawl (bot, scraping, rate limiting compliance, browser rendering, etc)

2. Extract and Store:
- url
- title
- published / updated metadata
- fetched_at timestamp
- cleaned main text
- outgoing links

3. Link Discovery:
- canonicalize URLs
- strip tracking parameters
- normalize trailing slashes
- content hashing for deduplication

all managed by firecrawl - subject to specs in ingestion.md

## INDEX Layer
Chunking:
Chunk documents by semantic structure (headings / paragraphs).

Chunk metadata:
1. source_url
2. chunk_id
3. section_heading
4. title
5. fetched_at
6. char offsets
7. token boundaries
8. chunk text

Vector Store:
- Qdrant or Postgres
- Raw chunk text stored alongside embeddings
- metadata stored for citation reconstruction

## RETRIEVAL LAYER
Responsibilities:
- semantic search over indexed chunks
- optional reranking
- passage selection for evidence
- citation snippet generation
- Citation generation uses stored offsets to return verbatim spans.

## ORCHESTRATION LAYER
The orchestration layer implements the retrieve-evaluate-expand loop that
decides when the corpus is "good enough" and when to grow it.

### Module breakdown
| Module            | Responsibility |
|-------------------|----------------|
| `engine.py`       | Top-level entry point. Owns connection pool, drives the iteration loop, assembles the final result. |
| `query_analyzer.py` | Decomposes a user query into sub-queries (LLM, rule-based, or passthrough). |
| `reranker.py`     | Provider-agnostic reranking (ZeroEntropy, Cohere, Jina, or passthrough). |
| `evaluator.py`    | Computes quality signals from score distributions and applies an 11-rule decision matrix (stop / expand_breadth / expand_recall / expand_intent). |
| `expander.py`     | Scores candidate links (5 weighted signals), scrapes the top picks, and indexes them. |
| `locality.py`     | Fetches adjacent sibling chunks around high-scoring hits (cheap DB query, no API calls). |
| `merger.py`       | Unions chunks from multiple sub-queries, deduplicates with MMR (Jaccard text overlap), enforces a token budget. |
| `models.py`       | Pydantic data contracts shared across the layer (QueryAnalysis, RankedChunk, EvaluationSignals, etc.). |

### Loop behaviour
1. Ensure seed URL is ingested and indexed.
2. Analyse / decompose query into sub-queries.
3. Retrieve → rerank → evaluate quality signals.
4. If evaluator says **stop**: proceed to output.  
   If **expand_breadth**: score candidate links, ingest top picks, re-retrieve over the larger corpus.  
   If **expand_recall**: re-retrieve with a doubled token budget.  
   If **expand_intent**: re-analyse the query with feedback from the failed retrieval.
5. Repeat step 3–4 up to `max_expansion_depth` iterations.
6. Optionally expand locality (adjacent parent chunks).
7. Final merge, MMR dedup, token-budget trim, citation extraction.

### Stop conditions (priority order)
- Max expansion depth reached (safety cap).
- Token budget filled with good-quality, non-redundant chunks.
- Diminishing returns (recall proxy barely improved iteration-over-iteration).
- Good plateau (low score variance, mean above mediocre floor).

### Key configuration
All orchestration config lives in `config.py` with env-var overrides documented in `blank.env`.  Critical knobs:
- `RERANKER_PROVIDER` — which reranker to use (zeroentropy / cohere / jina / none).
- `DECOMPOSITION_MODE` — query decomposition strategy (llm / rule_based / none).
- `MAX_EXPANSION_DEPTH` — hard cap on expansion iterations.
- `RETRIEVAL_CONTEXT_BUDGET` — target token count for the final context window.

## TARGETED EXPANSION STRATEGY
Seed Set:
- root URL
- sitemap.xml if provided
- documentation hubs

Link Scoring:
Candidate links are scored using:
- URL path heuristics
- anchor text relevance
- page title relevance
- recency (honestly a weak signal)
- allowlist constraints (if i care enough to implement it)

Batch Expansion:
Top-K candidates are ingested per iteration.

Termination Conditions:
- token budget saturation (may be worth implementing various token budgets according to need)
- diminishing retrieval improvement
- coverage completion heuristics
- model-identified knowledge gaps

Depth and page count are considered weak constraints and are not primary termination signals!

## CITATION MODEL
WebRAG produces two citation layers (this is subject to change)

1. Reference Layer
Indexed references:
- [1] URL + section metadata

2. Evidence Layer
Verbatim snippet:
- “exact retrieved span”

Citations are reconstructed from stored chunk offsets to guarantee fidelity.

## INTERFACE LAYER
WebRAG exposes MCP tools - LLMs act as clients; WebRAG is the tool provider

For now, primary tool call is:
- answer(url, query)

Behavior:
1. ingest URL
2. expand corpus if needed (depends on if we want to preserve corpus vs redo everytime, see Intended Usage Model)
3. retrieve evidence
4. generate answer
5. return citations

## PERMISSION ROUTERS
WebRAG supports ingestion via:
- public web crawling (primary focus)

and will eventually support:
- official APIs 
- browser session captures
- credentialed access (user-provided)

## INTENDED USAGE MODEL
WebRAG enables “docs-as-context” workflows where users can:
- query web documentation conversationally
- verify answers against source text
- explore linked material iteratively

The system acts as a persistent research memory rather than a one-shot retrieval tool
