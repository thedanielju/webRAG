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
The orchestration layer implements the corpus expansion loop

Loop behavior:
1. Retrieve evidence from existing corpus
2. If insufficient coverage:
- identify candidate sublinks
- score link relevance
- ingest top candidates
3. Re-index new content
4. Re-run retrieval
5. Terminate when stopping conditions are met

Expansion always progresses one hierarchical level per iteration.

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