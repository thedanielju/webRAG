# WebRAG Orchestration Layer — Design Document

> **Version:** 1.0  
> **Date:** 2026-02-21  
> **Status:** Ready for implementation  
> **Scope:** Orchestration layer design — the coordination brain between MCP tool calls and the ingestion/indexing/retrieval layers.

---

## Table of Contents

1. [Purpose & Role](#1-purpose--role)
2. [Libraries & Dependencies](#2-libraries--dependencies)
3. [Configuration Additions](#3-configuration-additions)
4. [Module Structure](#4-module-structure)
5. [Data Flow Overview](#5-data-flow-overview)
6. [Orchestration Loop Design](#6-orchestration-loop-design)
7. [Query Analysis & Decomposition](#7-query-analysis--decomposition)
8. [Reranking Layer](#8-reranking-layer)
9. [Stopping & Expansion Decision Engine](#9-stopping--expansion-decision-engine)
10. [Link Scoring & Expansion](#10-link-scoring--expansion)
11. [Locality Expansion](#11-locality-expansion)
12. [Output Contract](#12-output-contract)
13. [Error Handling & Edge Cases](#13-error-handling--edge-cases)
14. [MCP Layer Interface Notes](#14-mcp-layer-interface-notes)

---

## 1. Purpose & Role

The orchestration layer is the **coordination brain** of WebRAG. It sits between MCP tool calls (incoming requests from a reasoning LLM) and the three existing layers (ingestion, indexing, retrieval). It does NOT perform reasoning or answer synthesis — it produces evidence + citations for the LLM to use.

### Responsibilities

1. **Corpus management** — ensure requested URLs are ingested and indexed before retrieval.
2. **Query analysis** — decompose complex queries, classify intent, estimate complexity.
3. **Retrieve-evaluate-expand loop** — call retrieval, evaluate result quality via reranking, decide whether and how to expand, execute expansion, re-retrieve until stopping conditions are met.
4. **Reranking** — apply cross-encoder reranking to calibrate relevance scores from retrieval.
5. **Locality expansion** — grab adjacent chunks around high-scoring hits for fuller context.
6. **Citation assembly** — produce structured citation metadata from retrieved chunks.
7. **Output assembly** — produce a rich structured payload for the MCP layer to format and return.

### What It Does NOT Do

- Answer synthesis or reasoning (that's the LLM client).
- MCP protocol transport or tool schema definition (that's the MCP layer).
- Raw web scraping or chunk embedding (that's ingestion/indexing).
- Vector similarity search (that's the retrieval layer).

---

## 2. Libraries & Dependencies

### Existing (already in the codebase)

| Library | Used For | Layer |
|---------|----------|-------|
| `psycopg` (v3) | Async DB connections, queries | All layers |
| `pgvector` | Vector similarity search via `register_vector_async` | Retrieval |
| `firecrawl-py` | `AsyncFirecrawlApp` for scrape, map, batch_scrape | Ingestion |
| `openai` | `AsyncOpenAI` for embedding (text-embedding-3-small) | Indexing |
| `pydantic` | Data models, validation, serialization | All layers |

### New (required for orchestration)

| Library | Used For | Install |
|---------|----------|---------|
| `zeroentropy` | Reranking via zerank-2 (primary reranker provider) | `pip install zeroentropy` |
| `cohere` | Reranking via Cohere Rerank (optional provider) | `pip install cohere` |
| `openai` (reuse) | Query analysis LLM calls via `AsyncOpenAI` chat completions | Already installed |

### Optional / Future

| Library | Used For | When |
|---------|----------|------|
| `jina` / HTTP client | Jina Reranker (optional provider) | If user configures Jina |

### Notes

- The `openai` SDK is reused for query analysis LLM calls (gpt-4o-mini default). It shares the same SDK as the embedding client but uses a separate client instance pointed at the orchestration LLM endpoint (which may differ from the embedding endpoint).
- ZeroEntropy SDK provides `AsyncZeroEntropy` with async methods mirroring the sync client. Use `zclient.models.rerank()` for standalone reranking.
- Cohere SDK provides `AsyncClientV2` (or equivalent) with `rerank()`.
- Jina Reranker is accessed via HTTP API — use `httpx.AsyncClient` or the `jina` SDK if available.

---

## 3. Configuration Additions

Add to `config.py` under `Settings(BaseSettings)`:

```python
# ── Orchestration: Query Analysis ─────────────────────────────────
orchestration_llm_base_url: str = "https://api.openai.com/v1"
orchestration_llm_api_key: str | None = None
orchestration_llm_model: str = "gpt-4o-mini"
decomposition_mode: str = "llm"
# Valid values: "llm" | "rule_based" | "none"
# - "llm": Uses orchestration LLM for query decomposition (best quality).
#   Falls back to "rule_based" if orchestration_llm_api_key is not set.
# - "rule_based": Regex/pattern-based decomposition. Handles comparisons,
#   conjunctions, pro/con splits. No API calls.
# - "none": No decomposition. Single query passed directly to retrieval.

# ── Orchestration: Reranking ──────────────────────────────────────
reranker_provider: str = "zeroentropy"
# Valid values: "zeroentropy" | "cohere" | "jina" | "none"
# - "zeroentropy": ZeroEntropy zerank-2. Calibrated 0–1 scores, instruction-
#   following, confidence scores. $0.025/1M tokens. Recommended default.
# - "cohere": Cohere Rerank v3.5. ~$1/1000 queries.
# - "jina": Jina Reranker v2. Free tier available.
# - "none": Skip reranking. Stopping criteria use raw embedding similarity
#   scores (less calibrated but functional).
reranker_api_key: str | None = None
reranker_model: str = "zerank-2"
# Provider-specific model identifiers:
#   zeroentropy: "zerank-2" (default), "zerank-1", "zerank-1-small"
#   cohere: "rerank-v3.5" or "rerank-english-v3.0"
#   jina: "jina-reranker-v2-base-multilingual"
reranker_top_n: int = 20
# Maximum number of results returned after reranking.

# ── Orchestration: Expansion ──────────────────────────────────────
max_expansion_depth: int = 5
# Absolute safety ceiling. In practice, score-based stopping criteria
# trigger well before this. Prevents infinite recursion if heuristics fail.
max_candidates_per_iteration: int = 5
# Number of top-scored candidate URLs to actually scrape per expansion round.
# Candidates are scored from the link_candidates table (enriched via /map).
candidates_to_score_per_iteration: int = 20
# Number of candidate URLs to evaluate (score) per iteration before
# selecting the top max_candidates_per_iteration to scrape.
expansion_map_limit: int = 100
# Limit passed to discover_links() / Firecrawl /map endpoint when
# enriching link candidates. Controls how many URLs /map returns.

# ── Orchestration: Locality Expansion ─────────────────────────────
locality_expansion_enabled: bool = True
# When True, grab sibling parent chunks adjacent to high-scoring hits.
locality_expansion_radius: int = 1
# Number of sibling parent chunks to grab in each direction (before/after)
# around each high-scoring hit. 1 = grab the immediately preceding and
# following parent chunk. Set to 0 to effectively disable.

# ── Orchestration: Stopping Criteria ──────────────────────────────
token_budget_saturation_ratio: float = 0.8
# STOP signal: if (non-redundant tokens / context budget) exceeds this
# ratio, we have enough material. Applied after redundancy filtering.
redundancy_ceiling: float = 0.85
# Maximum pairwise cosine similarity between two chunks before one is
# filtered as redundant. Used in MMR-style deduplication.
score_cliff_threshold: float = 0.15
# Score gap between rank-1 and rank-K chunk. A cliff above this value
# indicates narrow relevance — one strong hit but thin coverage.
score_cliff_rank_k: int = 5
# Which rank to compare against rank-1 for cliff detection.
plateau_variance_threshold: float = 0.02
# Score variance among top-N chunks. Low variance = plateau.
# Plateau + high scores = broad coverage (STOP).
# Plateau + mediocre scores = everything vaguely related (expand intent).
plateau_top_n: int = 10
# Number of top chunks used for plateau variance calculation.
diminishing_return_delta: float = 0.03
# Minimum improvement in recall proxy (count of chunks above near-top
# band) across iterations to justify another expansion. If improvement
# is below this, STOP (diminishing returns).
mediocre_score_floor: float = 0.5
# Threshold below which a score plateau is considered "mediocre" rather
# than "good." Mediocre plateau triggers intent expansion, not stop.
# NOTE: This threshold is meaningful with calibrated reranker scores
# (zerank-2). With raw embedding similarity, lower to ~0.35.
confidence_floor: float = 0.3
# ZeroEntropy-specific: minimum average confidence across top-N chunks.
# Below this, the corpus likely doesn't have what we need. Triggers
# aggressive expansion or early termination with low-confidence flag.
```

### Environment Variables (add to `.env`)

```bash
# Orchestration LLM (query analysis / decomposition)
ORCHESTRATION_LLM_BASE_URL=https://api.openai.com/v1
ORCHESTRATION_LLM_API_KEY=sk-...
ORCHESTRATION_LLM_MODEL=gpt-4o-mini

# Reranker
RERANKER_PROVIDER=zeroentropy
RERANKER_API_KEY=ze-...
RERANKER_MODEL=zerank-2

# Expansion
MAX_EXPANSION_DEPTH=5
MAX_CANDIDATES_PER_ITERATION=5

# Stopping
TOKEN_BUDGET_SATURATION_RATIO=0.8
REDUNDANCY_CEILING=0.85
```

---

## 4. Module Structure

```
src/
├── orchestration/
│   ├── __init__.py
│   ├── engine.py            # Main orchestration loop (OrchestratorEngine)
│   ├── query_analyzer.py    # Query decomposition & classification
│   ├── reranker.py          # Modular reranker abstraction (zerank/cohere/jina)
│   ├── evaluator.py         # Retrieval quality evaluation & stop/expand decision
│   ├── expander.py          # Link scoring, candidate selection, expansion execution
│   ├── locality.py          # Adjacent chunk expansion
│   ├── merger.py            # Multi-subquery result merging & MMR deduplication
│   └── models.py            # Orchestration-specific Pydantic models
├── ingestion/               # Existing — scrape, discover_links, map_site
├── indexing/                 # Existing — index_batch, embedder, chunker
├── retrieval/               # Existing — retrieve, search, citations, models
└── config.py                # Extended with orchestration settings
```

### Module Responsibilities

#### `engine.py` — OrchestratorEngine

The top-level entry point. Owns the full orchestration lifecycle for a single request.

```python
class OrchestratorEngine:
    """Coordinates the full retrieve-evaluate-expand loop.
    
    Owns:
    - Connection pool management
    - Query analysis dispatch
    - The main iteration loop
    - Output assembly
    
    Does NOT own:
    - Individual evaluation logic (evaluator.py)
    - Reranking calls (reranker.py)
    - Expansion execution (expander.py)
    """

    async def run(
        self,
        url: str,
        query: str,
        *,
        intent: str | None = None,
        known_context: str | None = None,
        constraints: list[str] | None = None,
        expansion_budget: int | None = None,
    ) -> OrchestrationResult:
        """Execute the full orchestration pipeline for a single request."""
        ...
```

#### `query_analyzer.py` — Query Analysis & Decomposition

Handles query understanding before retrieval begins.

```python
class QueryAnalysis(BaseModel):
    """Output of query analysis."""
    original_query: str
    sub_queries: list[str]           # Decomposed sub-queries (may be just [original_query])
    query_type: str                  # "factual" | "comparison" | "how_to" | "exploratory"
    complexity: str                  # "simple" | "moderate" | "complex"
    key_concepts: list[str]          # Extracted topic/entity keywords

async def analyze_query(
    query: str,
    *,
    intent: str | None = None,       # MCP hint from reasoning model
    known_context: str | None = None, # What the model already knows
    constraints: list[str] | None = None,
) -> QueryAnalysis:
    """Analyze and optionally decompose a query.
    
    Mode selection (from config.decomposition_mode):
    - "llm": Single structured-output call to orchestration LLM.
    - "rule_based": Pattern matching (see _rule_based_decompose).
    - "none": Returns original query as sole sub-query.
    
    If "llm" is configured but orchestration_llm_api_key is None,
    falls back to "rule_based" with a warning.
    """
    ...

def _rule_based_decompose(query: str) -> list[str]:
    """Pattern-based query decomposition fallback.
    
    Handles:
    - Comparisons: "X vs Y", "X compared to Y", "difference between X and Y"
      → ["X", "Y"] or ["X properties", "Y properties"]
    - Conjunctions of distinct topics: "X and Y for Z"
      → ["X for Z", "Y for Z"]
    - Pro/con patterns: "pros and cons of X", "advantages and disadvantages"
      → ["advantages of X", "disadvantages of X"]
    - Multi-part questions: "how does X work and when should I use it"
      → ["how X works", "when to use X"]
    
    Returns [query] unchanged if no decomposition pattern matches.
    """
    ...
```

**LLM decomposition prompt** (used when `decomposition_mode == "llm"`):

```python
DECOMPOSITION_SYSTEM_PROMPT = """You are a query analysis assistant for a retrieval system.
Given a user query, produce a JSON object with:
- "sub_queries": list of 1-4 focused sub-queries that together cover the original query's information needs. If the query is simple and focused, return just the original query. Only decompose when the query genuinely contains multiple distinct information needs.
- "query_type": one of "factual", "comparison", "how_to", "exploratory"
- "complexity": one of "simple", "moderate", "complex"
- "key_concepts": list of 2-6 key topic/entity keywords extracted from the query

Respond with ONLY the JSON object, no other text."""
```

#### `reranker.py` — Modular Reranker Abstraction

Provider-agnostic reranking interface.

```python
class RerankResult(BaseModel):
    """Single reranked result."""
    index: int                      # Original position in input list
    relevance_score: float          # 0.0–1.0, calibrated for zerank-2
    confidence: float | None = None # ZeroEntropy-specific confidence score

async def rerank(
    query: str,
    passages: list[str],
    *,
    instruction: str | None = None,  # Passed to zerank-2 instruction field
    top_n: int | None = None,        # Defaults to config.reranker_top_n
) -> list[RerankResult]:
    """Rerank passages using the configured provider.
    
    Provider dispatch (from config.reranker_provider):
    - "zeroentropy": AsyncZeroEntropy().models.rerank(). Supports instruction.
    - "cohere": AsyncClientV2().rerank(). Instruction prepended to query.
    - "jina": HTTP POST to Jina API. Instruction prepended to query.
    - "none": Returns input order with original scores (no-op passthrough).
    
    All providers normalize output to list[RerankResult] sorted by
    relevance_score descending.
    """
    ...

async def _rerank_zeroentropy(
    query: str,
    passages: list[str],
    instruction: str | None,
    top_n: int,
) -> list[RerankResult]:
    """ZeroEntropy zerank-2 reranking.
    
    Uses: zclient.models.rerank(model=config.reranker_model,
          query=query, documents=passages, top_n=top_n)
    
    The instruction parameter is passed to zerank-2's instruction field
    for context-aware reranking (e.g., "prioritize code examples",
    "looking for comparison content").
    
    ZeroEntropy returns calibrated scores where 0.8 ≈ 80% relevance,
    plus a confidence score per result.
    """
    ...

async def _rerank_cohere(query: str, passages: list[str],
                          instruction: str | None, top_n: int) -> list[RerankResult]:
    ...

async def _rerank_jina(query: str, passages: list[str],
                        instruction: str | None, top_n: int) -> list[RerankResult]:
    ...

def _passthrough_rerank(passages: list[str],
                         original_scores: list[float]) -> list[RerankResult]:
    """No-op reranker for provider="none". Preserves original order/scores."""
    ...
```

#### `evaluator.py` — Retrieval Quality Evaluation

Analyzes reranked results and decides: stop, or which type of expansion to perform.

```python
class EvaluationSignals(BaseModel):
    """Computed signals from a reranked retrieval result."""
    top_score: float
    score_at_k: float                    # Score at rank K (for cliff detection)
    score_cliff: float                   # top_score - score_at_k
    score_variance: float                # Variance of top N scores
    score_mean: float                    # Mean of top N scores
    chunks_above_threshold: int          # Count above (top_score - delta)
    token_fill_ratio: float              # Non-redundant tokens / budget
    redundancy_ratio: float              # Fraction of chunks flagged redundant
    source_document_count: int           # Distinct source URLs in results
    avg_confidence: float | None         # ZeroEntropy confidence average
    is_plateau: bool                     # Low variance in top N
    is_cliff: bool                       # Large gap between rank 1 and rank K
    is_saturated: bool                   # Token budget sufficiently filled
    is_mediocre_plateau: bool            # Plateau but scores are low
    has_high_redundancy: bool            # Many near-duplicate chunks

class ExpansionDecision(BaseModel):
    """The evaluator's recommendation."""
    action: str
    # Valid values:
    #   "stop"            — sufficient coverage, return results
    #   "expand_breadth"  — follow more outgoing links, scrape new pages
    #   "expand_recall"   — adjust retrieval params (lower threshold, higher k)
    #   "expand_intent"   — rewrite/decompose query, try alternative phrasings
    #   "expand_locality" — fetch adjacent chunks (handled separately, always runs)
    reason: str               # Human-readable explanation for the log
    confidence: str           # "high" | "medium" | "low"

async def evaluate(
    reranked_chunks: list[RankedChunk],
    context_budget: int,
    iteration: int,
    previous_signals: EvaluationSignals | None,
    query_analysis: QueryAnalysis,
) -> tuple[EvaluationSignals, ExpansionDecision]:
    """Evaluate retrieval quality and decide next action.
    
    Decision logic (applied on reranked scores):
    
    1. Token budget saturated + low redundancy + not mediocre plateau → STOP
    2. Iteration > 0 and recall proxy didn't improve by diminishing_return_delta → STOP
    3. Cliff detected:
       a. High redundancy among top hits → EXPAND_INTENT (rewrite query)
       b. Low redundancy → EXPAND_BREADTH (more sources needed)
    4. Mediocre plateau (low variance + low mean score) → EXPAND_INTENT
    5. Good plateau (low variance + high mean score) → STOP
    6. Low scores everywhere → EXPAND_INTENT
    7. Good scores but low token fill → EXPAND_LOCALITY (then BREADTH if still low)
    8. Default fallback → STOP
    
    Cross-iteration signals:
    - Recall proxy delta: count of chunks above (top_score - δ) vs previous
    - Diversity gain: new source documents discovered this round
    - Score distribution improvement: overall distribution shift
    """
    ...

def _compute_redundancy(
    chunks: list[RankedChunk],
    query_embedding: list[float],
) -> tuple[float, list[int]]:
    """Compute pairwise redundancy and return (ratio, indices_to_drop).
    
    Uses MMR-style greedy selection:
    1. Start with highest-scored chunk.
    2. For each remaining chunk, compute max similarity to already-selected set.
    3. If max similarity > redundancy_ceiling, mark as redundant.
    4. Return ratio of redundant chunks and their indices.
    
    Similarity computed from query_embedding cosine against chunk embeddings
    when available, or falls back to text overlap heuristic.
    """
    ...
```

#### `expander.py` — Link Scoring & Expansion Execution

Handles candidate discovery, scoring, and expansion execution.

```python
async def score_candidates(
    candidates: list[PersistedLinkCandidate],
    query: str,
    query_analysis: QueryAnalysis,
    already_ingested_urls: set[str],
) -> list[ScoredCandidate]:
    """Score link candidates for expansion relevance.
    
    Scoring signals (combined as weighted sum):
    1. URL path relevance (0.0–1.0): Token overlap between URL path
       segments and query key_concepts. Weight: 0.15
    2. Title relevance (0.0–1.0): If title available (from /map enrichment),
       token overlap or embedding similarity with query. Weight: 0.40
    3. Description relevance (0.0–1.0): If description available,
       token overlap with query. Weight: 0.30
    4. In-degree signal (0.0–1.0): Normalized count of how many
       already-ingested pages link to this candidate URL. Weight: 0.05
    5. Depth penalty (0.0–1.0): Decay by candidate depth. Weight: 0.10
    
    Candidates already in already_ingested_urls are excluded.
    Returns sorted by score descending.
    """
    ...

class ScoredCandidate(BaseModel):
    """A link candidate with computed expansion relevance score."""
    link_candidate: PersistedLinkCandidate
    score: float
    score_breakdown: dict[str, float]  # Signal name → individual score

async def expand(
    seed_url: str,
    query: str,
    query_analysis: QueryAnalysis,
    conn: AsyncConnection,
    *,
    already_ingested_urls: set[str],
    current_depth: int,
) -> ExpansionOutcome:
    """Execute one expansion iteration.
    
    Steps:
    1. Fetch link_candidates for all currently-ingested source documents.
    2. Filter out already_ingested_urls.
    3. Enrich candidates via discover_links() if not already enriched
       (calls /map once per source URL, caches in link_candidates table).
    4. Generate parent URL candidates from seed_url path derivation
       (strip path segments to discover parent/sibling pages).
    5. Score all candidates.
    6. Select top max_candidates_per_iteration.
    7. Scrape selected candidates concurrently (asyncio.gather).
    8. Index scraped content (index_batch).
    9. Return ExpansionOutcome with ingested URLs and stats.
    """
    ...

class ExpansionOutcome(BaseModel):
    """Result of a single expansion iteration."""
    urls_attempted: list[str]
    urls_ingested: list[str]
    urls_failed: list[str]
    chunks_added: int
    candidates_scored: int
    candidates_selected: int
    depth: int

def _derive_parent_urls(url: str) -> list[str]:
    """Derive parent URLs by stripping path segments.
    
    Example: "https://scikit-learn.org/stable/modules/ensemble.html"
    → ["https://scikit-learn.org/stable/modules/",
       "https://scikit-learn.org/stable/"]
    
    Excludes the root domain itself (too generic).
    These are added to the candidate pool and scored alongside
    link-discovered candidates — no special priority boost.
    """
    ...
```

#### `locality.py` — Adjacent Chunk Expansion

```python
async def expand_locality(
    high_score_chunks: list[RankedChunk],
    conn: AsyncConnection,
    *,
    radius: int = 1,
    min_score_for_expansion: float = 0.5,
) -> list[RetrievedChunk]:
    """Fetch sibling parent chunks adjacent to high-scoring hits.
    
    For each chunk with reranked score >= min_score_for_expansion:
    1. Query chunks table for same document_id, chunk_level='parent',
       chunk_index in [hit.chunk_index - radius, hit.chunk_index + radius].
    2. Exclude chunks already in the result set.
    3. Return new chunks with a flag indicating locality-expanded origin.
    
    This is a cheap DB query — no embedding or API calls.
    Runs after every retrieval, before final output assembly.
    """
    ...
```

#### `merger.py` — Result Merging & Deduplication

```python
async def merge_subquery_results(
    subquery_results: list[SubQueryResult],
    context_budget: int,
) -> list[RankedChunk]:
    """Merge results from multiple sub-query retrievals.
    
    Steps:
    1. Union all chunks across sub-query results.
    2. For chunks appearing in multiple results, keep max reranked score.
    3. Apply MMR deduplication (redundancy_ceiling from config).
    4. Sort by score descending.
    5. Apply token budget.
    6. Return merged, deduplicated, budget-fit result set.
    """
    ...

class SubQueryResult(BaseModel):
    """Result from a single sub-query's retrieve+rerank cycle."""
    sub_query: str
    retrieval_result: RetrievalResult
    reranked_chunks: list[RankedChunk]

class RankedChunk(BaseModel):
    """A RetrievedChunk with reranked relevance score."""
    chunk: RetrievedChunk
    reranked_score: float
    confidence: float | None = None
    is_locality_expanded: bool = False
    source_sub_query: str | None = None
```

#### `models.py` — Orchestration-Specific Models

```python
class OrchestrationResult(BaseModel):
    """Top-level output contract from orchestration to MCP layer."""
    
    # Core content
    chunks: list[RankedChunk]
    citations: list[CitationSpan]
    
    # Query analysis
    query_analysis: QueryAnalysis
    
    # Expansion history (for traversal diagram)
    expansion_steps: list[ExpansionStep]
    
    # Aggregated stats
    corpus_stats: CorpusStats
    timing: OrchestrationTiming
    
    # Metadata
    mode: str                          # "full_context" | "chunk"
    final_decision: ExpansionDecision  # The terminal stop decision
    total_iterations: int
    total_urls_ingested: int

class ExpansionStep(BaseModel):
    """Record of a single expansion iteration for observability."""
    iteration: int
    depth: int
    source_url: str
    candidates_scored: int
    candidates_expanded: list[str]     # URLs actually scraped
    candidates_failed: list[str]       # URLs that failed to scrape
    chunks_added: int
    top_score_before: float            # Best reranked score before this expansion
    top_score_after: float             # Best reranked score after re-retrieval
    decision: str                      # "expand_breadth" | "expand_intent" | "stop" | etc.
    reason: str                        # Human-readable reason
    
class OrchestrationTiming(BaseModel):
    """End-to-end timing breakdown."""
    query_analysis_ms: float
    retrieval_ms: float                # Cumulative across iterations
    reranking_ms: float                # Cumulative across iterations
    expansion_ms: float                # Scraping + indexing time
    locality_ms: float
    merge_ms: float
    total_ms: float
```

---

## 5. Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Tool Call                            │
│  answer(url, query, intent?, known_context?, constraints?,      │
│         expansion_budget?)                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OrchestratorEngine.run()                     │
│                                                                 │
│  1. Acquire async DB connection from pool                       │
│  2. Corpus check: is URL ingested?                              │
│     ├─ NO → scrape(url) → index_batch([doc]) → proceed         │
│     └─ YES → proceed                                            │
│  3. Query analysis → QueryAnalysis (sub_queries, type, etc.)    │
│  4. For each sub_query (parallel via asyncio.gather):           │
│     └─ retrieve(conn, sub_query, source_urls=[url])             │
│        → RetrievalResult                                        │
│  5. Rerank each sub_query's chunks                              │
│     └─ rerank(sub_query, passages, instruction=intent)          │
│        → list[RerankResult]                                     │
│  6. Merge sub-query results → merged RankedChunks               │
│  7. Evaluate merged results → EvaluationSignals + Decision      │
│                                                                 │
│  ┌─── Expansion Loop (if decision != "stop") ───────────────┐  │
│  │  8. Execute expansion based on decision type:             │  │
│  │     ├─ expand_breadth → score links → scrape → index      │  │
│  │     ├─ expand_recall → adjust retrieval params → retrieve │  │
│  │     └─ expand_intent → rewrite query → retrieve           │  │
│  │  9. Re-retrieve with expanded corpus                      │  │
│  │ 10. Re-rerank                                             │  │
│  │ 11. Re-evaluate → new Decision                            │  │
│  │ 12. Log ExpansionStep                                     │  │
│  │ 13. Loop or break                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│ 14. Locality expansion (grab adjacent sibling chunks)           │
│ 15. Final MMR deduplication                                     │
│ 16. Citation extraction (extract_citation for each chunk)       │
│ 17. Assemble OrchestrationResult                                │
│ 18. Return to MCP layer                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Types at Each Boundary

| Step | Function | Input | Output |
|------|----------|-------|--------|
| Corpus check | `scrape(url)` | `str` | `NormalizedDocument` |
| Index | `index_batch([doc], conn)` | `list[NormalizedDocument]` | `None` (side effect: DB rows) |
| Retrieve | `retrieve(conn, query, ...)` | `str` + params | `RetrievalResult` |
| Rerank | `rerank(query, passages, ...)` | `str` + `list[str]` | `list[RerankResult]` |
| Evaluate | `evaluate(chunks, budget, ...)` | `list[RankedChunk]` + context | `(EvaluationSignals, ExpansionDecision)` |
| Expand | `expand(url, query, ...)` | URL + query + context | `ExpansionOutcome` |
| Locality | `expand_locality(chunks, conn)` | `list[RankedChunk]` | `list[RetrievedChunk]` |
| Merge | `merge_subquery_results(...)` | `list[SubQueryResult]` | `list[RankedChunk]` |
| Citation | `extract_citation(chunk, ...)` | `RetrievedChunk` + offsets | `CitationSpan \| None` |

---

## 6. Orchestration Loop Design

The orchestration loop is a **state machine** with explicit state tracked across iterations.

### State Object

```python
class OrchestrationState:
    """Mutable state tracked across the orchestration loop."""
    
    # Query context (immutable after init)
    original_query: str
    query_analysis: QueryAnalysis
    seed_url: str
    intent: str | None
    constraints: list[str] | None
    expansion_budget: int | None       # None = auto (use heuristics)
    
    # Iteration tracking
    iteration: int = 0
    current_depth: int = 0
    
    # Corpus state (grows across iterations)
    ingested_urls: set[str] = set()
    
    # Result state (updated each iteration)
    current_chunks: list[RankedChunk] = []
    current_signals: EvaluationSignals | None = None
    
    # History
    expansion_steps: list[ExpansionStep] = []
    all_retrieval_results: list[RetrievalResult] = []
```

### Loop Pseudocode

```python
async def run(self, url, query, **hints) -> OrchestrationResult:
    state = OrchestrationState(seed_url=url, original_query=query, **hints)
    conn = await self._acquire_connection()
    
    try:
        # ── Phase 1: Corpus Preparation ───────────────────────
        await self._ensure_ingested(url, conn, state)
        
        # ── Phase 2: Query Analysis ───────────────────────────
        state.query_analysis = await analyze_query(
            query, intent=hints.get("intent"),
            known_context=hints.get("known_context"),
            constraints=hints.get("constraints"),
        )
        
        # ── Phase 3: Initial Retrieval + Reranking ────────────
        state.current_chunks = await self._retrieve_and_rerank(
            state.query_analysis.sub_queries, conn, state
        )
        state.current_signals, decision = await evaluate(
            state.current_chunks, context_budget, 0, None, state.query_analysis
        )
        
        # ── Phase 4: Expansion Loop ───────────────────────────
        max_iterations = state.expansion_budget or config.max_expansion_depth
        
        while decision.action != "stop" and state.iteration < max_iterations:
            state.iteration += 1
            top_score_before = state.current_signals.top_score
            
            if decision.action == "expand_breadth":
                outcome = await expand(
                    state.seed_url, query, state.query_analysis, conn,
                    already_ingested_urls=state.ingested_urls,
                    current_depth=state.current_depth,
                )
                state.ingested_urls.update(outcome.urls_ingested)
                state.current_depth = outcome.depth
                # Re-retrieve over expanded corpus (no source_url filter now)
                state.current_chunks = await self._retrieve_and_rerank(
                    state.query_analysis.sub_queries, conn, state
                )
                
            elif decision.action == "expand_recall":
                # Re-retrieve with relaxed parameters
                state.current_chunks = await self._retrieve_and_rerank(
                    state.query_analysis.sub_queries, conn, state,
                    relaxed=True,  # lower threshold, higher top_k
                )
                
            elif decision.action == "expand_intent":
                # Rewrite/decompose query differently
                new_analysis = await analyze_query(
                    query,
                    intent=f"Previous retrieval insufficient. "
                           f"Reason: {decision.reason}. "
                           f"Try alternative decomposition.",
                )
                state.query_analysis = new_analysis
                state.current_chunks = await self._retrieve_and_rerank(
                    new_analysis.sub_queries, conn, state
                )
            
            # Re-evaluate
            state.current_signals, decision = await evaluate(
                state.current_chunks, context_budget,
                state.iteration, state.current_signals,
                state.query_analysis,
            )
            
            # Log expansion step
            state.expansion_steps.append(ExpansionStep(
                iteration=state.iteration,
                depth=state.current_depth,
                source_url=state.seed_url,
                candidates_scored=outcome.candidates_scored if decision.action == "expand_breadth" else 0,
                candidates_expanded=outcome.urls_ingested if decision.action == "expand_breadth" else [],
                candidates_failed=outcome.urls_failed if decision.action == "expand_breadth" else [],
                chunks_added=outcome.chunks_added if decision.action == "expand_breadth" else 0,
                top_score_before=top_score_before,
                top_score_after=state.current_signals.top_score,
                decision=decision.action,
                reason=decision.reason,
            ))
        
        # ── Phase 5: Locality Expansion ───────────────────────
        if config.locality_expansion_enabled:
            locality_chunks = await expand_locality(
                state.current_chunks, conn,
                radius=config.locality_expansion_radius,
            )
            state.current_chunks.extend(locality_chunks)
        
        # ── Phase 6: Final Dedup + Citation Assembly ──────────
        final_chunks = await merge_subquery_results(
            [SubQueryResult(sub_query="final", ...)], context_budget
        )
        citations = [
            extract_citation(c.chunk, c.chunk.char_start, c.chunk.char_end)
            for c in final_chunks
        ]
        
        return OrchestrationResult(
            chunks=final_chunks,
            citations=[c for c in citations if c is not None],
            query_analysis=state.query_analysis,
            expansion_steps=state.expansion_steps,
            corpus_stats=...,
            timing=...,
            mode=...,
            final_decision=decision,
            total_iterations=state.iteration,
            total_urls_ingested=len(state.ingested_urls),
        )
    finally:
        await self._release_connection(conn)
```

### Connection Management

```python
# OrchestratorEngine.__init__
self._pool: AsyncConnectionPool = AsyncConnectionPool(
    conninfo=config.database_url,
    min_size=2,
    max_size=10,
    open=False,
)

async def _acquire_connection(self) -> AsyncConnection:
    conn = await self._pool.getconn()
    await register_vector_async(conn)
    return conn

async def _release_connection(self, conn: AsyncConnection) -> None:
    await self._pool.putconn(conn)
```

---

## 7. Query Analysis & Decomposition

### Three Modes

| Mode | Config Value | How It Works | When to Use |
|------|-------------|--------------|-------------|
| LLM | `"llm"` | Structured output call to gpt-4o-mini (or configured model). ~200ms, ~0.1¢ per call. | Default. Best decomposition quality. |
| Rule-based | `"rule_based"` | Regex/pattern matching. Zero API calls. | When no LLM key is configured, or for cost-sensitive deployments. |
| None | `"none"` | Pass-through. Original query becomes the sole sub-query. | Maximum speed, simple factual queries. |

### Rule-Based Patterns

```
Pattern: "X vs Y" / "X compared to Y" / "difference between X and Y"
→ sub_queries: ["X", "Y"]
→ query_type: "comparison"

Pattern: "pros and cons of X" / "advantages and disadvantages of X"  
→ sub_queries: ["advantages of X", "disadvantages of X"]
→ query_type: "comparison"

Pattern: "how does X work and when should I use it"
→ sub_queries: ["how X works", "when to use X"]
→ query_type: "how_to"

Pattern: "X and Y for Z" (where X, Y are distinct noun phrases)
→ sub_queries: ["X for Z", "Y for Z"]
→ query_type: "exploratory"

No match:
→ sub_queries: [original_query]
→ query_type: inferred from question words (what/how/why/compare)
```

### MCP Hint Integration

When the reasoning model provides `intent`, `known_context`, or `constraints` via MCP tool parameters, these modify query analysis:

- `intent` → Used as `query_type` directly if provided, skipping classification.
- `known_context` → Passed to LLM decomposition prompt as context to avoid retrieving already-known information.
- `constraints` → Appended to sub-queries where relevant (e.g., constraint "must include code examples" → sub-query becomes "gradient boosting Python code examples").

---

## 8. Reranking Layer

### Pipeline Position

```
retrieve() → list[RetrievedChunk]   (coarse, embedding similarity)
    ↓
rerank()  → list[RerankResult]      (fine, cross-encoder calibrated)
    ↓
evaluate() → ExpansionDecision      (decisions on calibrated scores)
```

### Why Retrieval Is NOT Moot

The retrieval layer and reranker serve complementary roles in a funnel:

| Layer | Purpose | Scale | Speed |
|-------|---------|-------|-------|
| Retrieval | **Recall** — find candidate chunks from thousands via HNSW | O(log n) | ~900ms |
| Reranker | **Precision** — re-score 20-40 candidates with cross-encoder | O(k) per pair | ~200ms |

Retrieval's depth decay, parent dedup, surface selection, token budget enforcement, full-context mode switching, and at-least-one guarantee all remain essential. The reranker only re-scores the relevance ordering within the set retrieval already assembled.

### ZeroEntropy zerank-2 Advantages

1. **Calibrated scores**: A score of 0.8 ≈ 80% relevance, consistently. This makes absolute thresholds meaningful for stopping criteria.
2. **Instruction-following**: Pass MCP intent hints and constraints directly into the reranker instruction field for context-aware scoring.
3. **Confidence scores**: Per-result confidence measure. Low average confidence across results signals the corpus may not contain relevant content.
4. **Cost**: $0.025 per 1M tokens — negligible for 20-40 chunks per query.

### Provider Interface

All providers normalize to the same `list[RerankResult]` output. The `instruction` parameter is:
- **ZeroEntropy**: Passed natively to zerank-2's instruction field.
- **Cohere/Jina**: Prepended to the query string (best-effort instruction injection).
- **None**: Ignored.

---

## 9. Stopping & Expansion Decision Engine

### Signal Computation

After each reranking, the evaluator computes these signals from the reranked score distribution:

| Signal | Computation | What It Means |
|--------|-------------|---------------|
| `score_cliff` | `top_score - score_at_k` (k=5) | Large gap = narrow relevance pocket |
| `score_variance` | Variance of top N scores (N=10) | Low = plateau, high = diverse relevance |
| `score_mean` | Mean of top N scores | Absolute quality of results |
| `token_fill_ratio` | Non-redundant tokens / budget | How much useful content we have |
| `redundancy_ratio` | Fraction of chunks flagged as near-duplicate | Content repetition level |
| `source_document_count` | Distinct source URLs | Breadth of evidence |
| `avg_confidence` | Mean zerank-2 confidence (if available) | Reranker's self-assessed certainty |
| `chunks_above_threshold` | Count above `(top_score - δ)` | Recall proxy — how many "good" chunks |

### Decision Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    STOP CONDITIONS (any → stop)                 │
├─────────────────────────────────────────────────────────────────┤
│ 1. Saturated + not mediocre:                                    │
│    token_fill_ratio > 0.8 AND NOT is_mediocre_plateau           │
│    AND redundancy_ratio < redundancy_ceiling                    │
│                                                                 │
│ 2. Diminishing returns (iteration > 0):                         │
│    chunks_above_threshold did not increase by                   │
│    diminishing_return_delta since last iteration                │
│                                                                 │
│ 3. Good plateau:                                                │
│    is_plateau AND score_mean >= mediocre_score_floor             │
│                                                                 │
│ 4. Max depth reached (safety cap):                              │
│    iteration >= max_expansion_depth                             │
├─────────────────────────────────────────────────────────────────┤
│                    EXPAND CONDITIONS                            │
├─────────────────────────────────────────────────────────────────┤
│ 5. Cliff + high redundancy → EXPAND_INTENT                     │
│    Diagnosis: stuck in one pocket, need different angle         │
│                                                                 │
│ 6. Cliff + low redundancy → EXPAND_BREADTH                     │
│    Diagnosis: found one good source, need more sources          │
│                                                                 │
│ 7. Mediocre plateau → EXPAND_INTENT                             │
│    Diagnosis: everything vaguely related, query too broad       │
│                                                                 │
│ 8. Low scores everywhere → EXPAND_INTENT                        │
│    Diagnosis: query misaligned with corpus terminology          │
│                                                                 │
│ 9. Good scores + low token fill → EXPAND_BREADTH                │
│    Diagnosis: quality hits exist but need more volume            │
│                                                                 │
│ 10. Low confidence (zerank-2) → EXPAND_BREADTH                  │
│     Diagnosis: reranker uncertain, corpus may lack coverage     │
│                                                                 │
│ 11. Default fallback → STOP                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-Iteration Tracking

The evaluator compares current signals against `previous_signals` to detect:

- **Recall proxy delta**: `current.chunks_above_threshold - previous.chunks_above_threshold`. If < `diminishing_return_delta`, expansion isn't helping.
- **Diversity gain**: `current.source_document_count - previous.source_document_count`. Zero gain means expansion found content from already-known sources.
- **Score improvement**: `current.top_score - previous.top_score`. Weak signal alone (breadth matters more than peak), but combined with recall proxy gives full picture.

### Score Normalization

When using raw embedding similarity (reranker_provider="none"), scores are not calibrated — 0.6 might mean "good" in one query and "mediocre" in another. In this mode:

- Absolute thresholds (mediocre_score_floor, confidence_floor) should be lowered. Use `mediocre_score_floor=0.35` instead of 0.5.
- Relative signals (cliff, variance, plateau) remain reliable regardless of calibration.
- The evaluator detects reranker_provider from config and adjusts threshold behavior accordingly.

---

## 10. Link Scoring & Expansion

### Expansion Flow

```
1. Fetch link_candidates for all ingested source_document_ids
       ↓
2. Filter out already-ingested URLs (state.ingested_urls)
       ↓
3. Enrich un-enriched candidates via discover_links() / map
   (one /map call per source URL, cached in link_candidates table)
       ↓
4. Generate parent URL candidates from seed URL path derivation
       ↓
5. Score all candidates → sorted list
       ↓
6. Select top max_candidates_per_iteration (default 5)
       ↓
7. Scrape selected URLs concurrently (asyncio.gather)
       ↓
8. Index scraped documents (index_batch)
       ↓
9. Return ExpansionOutcome
```

### Link Scoring Formula

```
score = (0.15 × url_path_relevance) +
        (0.40 × title_relevance) +
        (0.30 × description_relevance) +
        (0.05 × in_degree_signal) +
        (0.10 × depth_freshness)
```

| Signal | Range | How Computed |
|--------|-------|-------------|
| `url_path_relevance` | 0.0–1.0 | Jaccard overlap between URL path tokens (split on `/`, `-`, `_`) and `query_analysis.key_concepts` |
| `title_relevance` | 0.0–1.0 | Token overlap between link candidate title and query. 0.0 if title is None (un-enriched). |
| `description_relevance` | 0.0–1.0 | Token overlap between link candidate description and query. 0.0 if None. |
| `in_degree_signal` | 0.0–1.0 | `min(1.0, in_degree / 5)` where in_degree = count of distinct source documents in link_candidates that point to this target URL. |
| `depth_freshness` | 0.0–1.0 | `max(0.0, 1.0 - candidate.depth * 0.2)`. Prefers shallower candidates. |

**When candidates lack title/description** (un-enriched, only URL available): the title and description weights (0.40 + 0.30 = 0.70) effectively become 0. This heavily penalizes un-enriched candidates, which incentivizes calling `enrich_link_candidates()` before scoring. However, if enrichment is skipped (e.g., to save credits), URL path relevance and in-degree still provide some signal.

### Parent URL Derivation

```python
"https://scikit-learn.org/stable/modules/ensemble.html"
→ "https://scikit-learn.org/stable/modules/"      # parent directory
→ "https://scikit-learn.org/stable/"               # grandparent
# Root "https://scikit-learn.org/" excluded (too generic)
```

Parent URLs are added to the candidate pool as synthetic `PersistedLinkCandidate` entries with `title=None`, `description=None`. They compete on scoring like any other candidate. If they score well (URL path matches query concepts), they get expanded. If not, they're naturally filtered out.

### Concurrency in Expansion

```python
# Scrape candidates concurrently
scrape_tasks = [scrape(url) for url in selected_candidate_urls]
results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

# Filter successes, log failures
docs = [r for r in results if isinstance(r, NormalizedDocument)]
failed = [url for url, r in zip(urls, results) if isinstance(r, Exception)]

# Index all successful scrapes in one batch
await index_batch(docs, conn)
```

---

## 11. Locality Expansion

### When It Runs

Locality expansion runs **once**, after the expansion loop terminates and before final output assembly. It operates on the terminal set of reranked chunks.

### Logic

```python
for chunk in reranked_chunks:
    if chunk.reranked_score < min_score_for_expansion:
        continue
    
    # Fetch sibling parents within radius
    siblings = await conn.execute("""
        SELECT * FROM chunks
        WHERE document_id = %s
          AND chunk_level = 'parent'
          AND chunk_index BETWEEN %s AND %s
          AND id != %s
    """, (chunk.document_id, 
          chunk.chunk_index - radius,
          chunk.chunk_index + radius,
          chunk.id))
    
    for sibling in siblings:
        if sibling.id not in already_in_results:
            expanded_chunks.append(
                _row_to_retrieved_chunk(sibling, score=0.0, raw_similarity=None)
                # score=0.0 marks these as locality-expanded, not similarity-matched
            )
```

### Result Integration

Locality-expanded chunks are added to the result set with:
- `reranked_score = 0.0` (they were not scored by retrieval or reranker)
- `is_locality_expanded = True` flag
- They are included in the token budget calculation
- They are NOT included in stopping criteria score analysis (they'd skew distributions)
- They appear in the output adjacent to their source chunk (preserving document reading order)

---

## 12. Output Contract

### OrchestrationResult → MCP Layer

The MCP layer receives an `OrchestrationResult` and is responsible for formatting it into the MCP tool response. The output is intentionally rich so the MCP layer has maximum flexibility.

```python
class OrchestrationResult(BaseModel):
    """Complete orchestration output. See models.py for full definition."""
    
    chunks: list[RankedChunk]
    # Ordered by: source_url groups, then chunk_index within groups.
    # Each RankedChunk contains:
    #   - chunk: RetrievedChunk (full content, offsets, metadata)
    #   - reranked_score: float (calibrated if zerank-2)
    #   - confidence: float | None
    #   - is_locality_expanded: bool
    #   - source_sub_query: str | None
    
    citations: list[CitationSpan]
    # One CitationSpan per chunk. Produced by extract_citation().
    # Contains: verbatim_text, source_url, section_heading,
    #           char_start, char_end, token_start, token_end, title
    
    query_analysis: QueryAnalysis
    # sub_queries, query_type, complexity, key_concepts
    
    expansion_steps: list[ExpansionStep]
    # Full traversal history. Each step contains:
    #   iteration, depth, source_url, candidates_scored,
    #   candidates_expanded (URLs), chunks_added,
    #   top_score_before, top_score_after, decision, reason
    # MCP layer renders this as ASCII traversal diagram.
    
    corpus_stats: CorpusStats
    # total_documents, total_parent_chunks, total_tokens, documents_matched
    
    timing: OrchestrationTiming
    # Breakdown: query_analysis_ms, retrieval_ms, reranking_ms,
    #            expansion_ms, locality_ms, merge_ms, total_ms
    
    mode: str                          # "full_context" | "chunk"
    final_decision: ExpansionDecision  # Terminal stop decision with reason
    total_iterations: int
    total_urls_ingested: int
```

### Citation Generation

Orchestration calls `extract_citation()` from the existing citations module for each chunk in the final result set:

```python
citations = []
for ranked_chunk in final_chunks:
    chunk = ranked_chunk.chunk
    span = extract_citation(
        chunk,
        quote_start=chunk.char_start,
        quote_end=chunk.char_end,
    )
    if span is not None:
        citations.append(span)
```

This produces full-chunk citations. The MCP layer can extract shorter verbatim snippets from within these spans as needed for inline citation formatting.

---

## 13. Error Handling & Edge Cases

### Failure Modes and Handling

| Scenario | Handling |
|----------|---------|
| **Scrape fails** (Firecrawl error, rate limit, 4xx/5xx) | Log failure, skip URL, continue with remaining candidates. Track in `ExpansionOutcome.urls_failed`. Do NOT retry in same iteration — failed URLs are excluded from future candidate pools. |
| **Embedding API fails** | Retry once with backoff. If still fails, return partial results with error flag. Do NOT expand further. |
| **Reranker API fails** | Fall back to `_passthrough_rerank()` (preserve retrieval ordering). Log warning. Stopping criteria operate on raw similarity scores with adjusted thresholds. |
| **Orchestration LLM fails** | Fall back to `_rule_based_decompose()`. Log warning. |
| **Empty corpus** | Retrieval returns `is_empty=True`. Orchestration skips evaluation, proceeds directly to expansion (scrape seed URL if not yet ingested, or return empty result). |
| **All candidates already ingested** | No expansion possible. STOP regardless of evaluation signals. Log "no expansion candidates available." |
| **Circular links** | `state.ingested_urls` set prevents re-scraping. `link_candidates` UNIQUE constraint prevents duplicate rows. |
| **Very large pages** (token count exceeds budget) | Retrieval's existing token budget enforcement handles this. Orchestration inherits the behavior. |
| **Adversarial / malformed URLs** | URL validation before scraping: must be valid http/https, not localhost, not IP address, not file://. Reject with error. |
| **Firecrawl rate limits** | Respect Firecrawl's rate limit headers. If rate-limited during expansion, reduce `max_candidates_per_iteration` for this iteration and log. |
| **DB connection failure** | Retry connection acquisition once. If pool exhausted, return error immediately. Do NOT queue. |
| **Expansion exceeds expansion_budget** | Check `state.iteration < max_iterations` at loop top. Hard stop, return best results so far. |

### Timeout Strategy

```python
ORCHESTRATION_TIMEOUT = 120  # seconds, absolute ceiling
EXPANSION_ITERATION_TIMEOUT = 30  # seconds per expansion round

# Applied via asyncio.wait_for() around the expansion loop
try:
    result = await asyncio.wait_for(
        self._expansion_loop(state, conn),
        timeout=ORCHESTRATION_TIMEOUT,
    )
except asyncio.TimeoutError:
    # Return best results accumulated so far
    return self._assemble_partial_result(state, timeout=True)
```

### Graceful Degradation

The system is designed to degrade gracefully when optional components are unavailable:

| Component | Degraded Behavior |
|-----------|-------------------|
| No reranker API key | Passthrough reranking, use embedding similarity scores |
| No orchestration LLM key | Rule-based decomposition |
| No Firecrawl API key | Cannot ingest new URLs. Retrieval over existing corpus only. |
| DB connection pool exhausted | Immediate error return |

---

## 14. MCP Layer Interface Notes

This section captures design decisions and features relevant to the MCP layer, which is the next layer to be implemented after orchestration.

### MCP Tool Schema

The primary tool exposed to reasoning LLMs:

```python
# Tool: answer
# Description: Ingest a web page, retrieve evidence, and return
#              cited context for answering a query.
{
    "name": "answer",
    "parameters": {
        "url": {
            "type": "string",
            "description": "The web page URL to use as primary source.",
            "required": True,
        },
        "query": {
            "type": "string", 
            "description": "The question or information need to answer.",
            "required": True,
        },
        "intent": {
            "type": "string",
            "enum": ["factual", "comparison", "how_to", "exploratory"],
            "description": "Query intent classification. If provided, skips "
                           "orchestration's own classification.",
            "required": False,
        },
        "known_context": {
            "type": "string",
            "description": "What the model already knows about this topic. "
                           "Helps orchestration avoid retrieving redundant info.",
            "required": False,
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Retrieval constraints like 'must include code examples' "
                           "or 'prioritize recent content'.",
            "required": False,
        },
        "expansion_budget": {
            "type": "integer",
            "description": "Max expansion iterations. None = auto. "
                           "Use 0 for no expansion (single retrieval pass). "
                           "Use 1 for quick queries. Higher for thorough research.",
            "required": False,
        },
    }
}
```

### What the MCP Layer Receives

The MCP layer receives `OrchestrationResult` and is responsible for:

1. **Formatting context blocks** — Assembling the actual text payload the LLM will read from `chunks[].chunk.selected_text`.

2. **Citation formatting** — Converting `CitationSpan` objects into the two-layer citation format:
   - **Reference layer**: `[1] Title — URL § Section Heading`
   - **Evidence layer**: Verbatim snippet from `citation.verbatim_text`

3. **ASCII traversal diagram** — Rendering `expansion_steps` into a visual tree for the model response. URLs truncated to show only the distinguishing path segment:
   ```
   /ensemble.html (seed, score: 0.63)
   ├── /gradient_boosting.html (depth 1, score: 0.71)
   │   └── /histogram_gradient_boosting.html (depth 2, score: 0.58)
   ├── /random_forests.html (depth 1, score: 0.67)
   └── /modules/ (parent, score: 0.42) [skipped: low relevance]
   ```

4. **Timing/stats summary** — Human-readable performance breakdown from `OrchestrationTiming`.

5. **Error/warning surfacing** — Communicating partial results, degraded components, or timeout situations to the reasoning model.

### Additional MCP Tools (Future Consideration)

| Tool | Purpose | Priority |
|------|---------|----------|
| `answer` | Primary tool — full orchestration pipeline | v1 |
| `search_corpus` | Query existing indexed content without expansion | v2 |
| `ingest` | Explicitly ingest a URL without querying | v2 |
| `corpus_status` | Return stats about what's indexed | v2 |

### MCP Hints the Reasoning Model Can Provide

| Hint | How Orchestration Uses It |
|------|--------------------------|
| `intent` | Skips query type classification. Passed to reranker instruction field. |
| `known_context` | Passed to LLM decomposition to avoid redundant retrieval. |
| `constraints` | Appended to sub-queries. Passed to reranker instruction. |
| `expansion_budget` | Overrides max iterations. `0` = no expansion. `None` = auto. |

### Context Block Assembly (MCP Layer Responsibility)

The MCP layer should assemble the LLM-readable context block from orchestration output. Suggested structure:

```
[SOURCES]
[1] {title} — {url} § {section_heading}
[2] ...

[EVIDENCE]
Source [1] (relevance: 0.85):
{chunk.selected_text}

Source [2] (relevance: 0.71):
{chunk.selected_text}

[EXPANSION TRACE]
{ascii_diagram}

[STATS]
Documents searched: {corpus_stats.total_documents}
Chunks evaluated: {corpus_stats.total_parent_chunks}
Expansion iterations: {total_iterations}
Total time: {timing.total_ms:.0f}ms
```

The exact formatting is the MCP layer's decision — orchestration provides all the raw materials.

---

## Implementation Order

Recommended implementation sequence for the agent:

1. **`models.py`** — All Pydantic models first (OrchestrationResult, ExpansionStep, QueryAnalysis, RankedChunk, EvaluationSignals, ExpansionDecision, etc.). These define all data contracts.

2. **`reranker.py`** — Modular reranker abstraction. Can be tested independently with mock data.

3. **`query_analyzer.py`** — Query analysis with all three modes. Can be tested independently.

4. **`evaluator.py`** — Stopping criteria logic. Testable with synthetic score distributions.

5. **`locality.py`** — Adjacent chunk expansion. Simple DB queries, testable in isolation.

6. **`expander.py`** — Link scoring and expansion execution. Depends on link_candidates table (already implemented).

7. **`merger.py`** — Result merging and MMR deduplication.

8. **`engine.py`** — Main orchestration loop. Integrates all modules. Integration tested.

9. **Config additions** — Add new settings to config.py and .env.

10. **Tests** — Unit tests for each module, integration test for full pipeline.
