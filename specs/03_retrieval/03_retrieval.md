# Retrieval Layer Design — src/retrieval/

**Status:** Design complete, ready for implementation
**Depends on:** Ingestion (complete), Indexing (complete)
**Consumed by:** Orchestration layer (next)

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Libraries and Dependencies](#2-libraries-and-dependencies)
3. [Configuration Additions](#3-configuration-additions)
4. [Module Structure](#4-module-structure)
5. [Query Embedding](#5-query-embedding)
6. [Mode Determination](#6-mode-determination)
7. [Full-Context Mode Pipeline](#7-full-context-mode-pipeline)
8. [Chunk Mode Pipeline](#8-chunk-mode-pipeline)
9. [Surface Selection](#9-surface-selection)
10. [Return Contract](#10-return-contract)
11. [Citation Reconstruction](#11-citation-reconstruction)
12. [Image Handling at Retrieval Time](#12-image-handling-at-retrieval-time)
13. [SQL Queries Reference](#13-sql-queries-reference)
14. [Testing Strategy](#14-testing-strategy)
15. [Future Considerations](#15-future-considerations)

---

## 1. Purpose and Scope

Retrieval is a **pure search function**: given a query string and an existing indexed corpus, return the best available evidence with everything needed for citation reconstruction and presentation.

    retrieve(query, **filters) -> RetrievalResult

Retrieval does **not**:
- Make LLM calls
- Decide if coverage is sufficient
- Trigger corpus expansion
- Generate answers

Those responsibilities belong to the orchestration layer, which consumes retrieval's output.

### Pipeline Summary

1. **Embed the query** — single OpenAI-compatible API call via embed_query() wrapper
2. **Decide mode** — full-context (small corpus) vs chunk (large corpus) based on total corpus token span
3. **Execute search** — full-context loads all parents; chunk mode runs HNSW + parent aggregation
4. **Apply surface selection** — flag-based html_text vs chunk_text per chunk
5. **Build return object** — rich RetrievalResult with chunks, metadata, corpus stats, timing
6. **Return to orchestration** — orchestration decides next action

### Architectural Pattern: Small-to-Large Retrieval

Retrieval uses the **small-to-large** pattern (also called "parent document retrieval" in LlamaIndex/LangChain literature):

- **Children are the search index.** HNSW finds them by embedding similarity. Small chunks (target 256 tokens) produce precise, focused embeddings that match specific query intents well.
- **Parents are the context unit.** What actually gets returned to the LLM. Larger chunks (up to 1000 tokens) give the LLM enough surrounding context to reason correctly — a child hit about "learning rate" pulls in the full "Gradient Boosted Trees" section.
- **Deduplication happens at the parent level.** If 5 children from the same parent all match, the parent is returned once.
- **Scoring uses max aggregation.** The best child score represents the parent. Max, not average — we want the strongest signal, not a diluted one.
- **Citation offsets remain precise.** Because children carry char_start/char_end from the original document, the specific matched child within a parent can still be cited exactly. The parent provides context; the child provides citation precision.

This is the right pattern for WebRAG because:
- Embedding precision scales inversely with chunk size — small chunks embed more precisely
- LLM reasoning quality scales with available context — large chunks reason better
- Citation reconstruction requires character-level offsets — children carry these from indexing
- The schema already supports it: children have embeddings + parent_id, parents have full section text

---

## 2. Libraries and Dependencies

### Required (already in project)

| Library | Version | Usage in Retrieval |
|---|---|---|
| psycopg[binary] | 3.x | All database queries (connection, parameterized SQL, cursor iteration) |
| pgvector.psycopg | — | register_vector(conn) for vector type adaptation (required on every connection) |
| openai | >=1.0 | Query embedding via embed_query() — reuses the existing embedder.py client |
| pydantic | 2.x | RetrievalResult and RetrievedChunk models for the return contract |
| pydantic-settings | 2.x | Config additions to Settings class |

### Required (new)

None. Retrieval introduces **zero new dependencies**. All functionality is built on psycopg3 (vector similarity via pgvector SQL operators), the existing OpenAI client, and stdlib.

**Justification:** Retrieval's core operations are a single embedding API call and parameterized SQL queries. pgvector's <=> operator handles cosine distance natively in SQL. Parent aggregation and surface selection are straightforward Python. No ORM, no additional vector client library, no search framework needed. This keeps the dependency surface minimal and avoids abstracting away the SQL that controls retrieval behavior.

### Not Used in v1

| Library | Reason for exclusion |
|---|---|
| cohere / rerank APIs | Reranking is a v2 enhancement. Cosine similarity + depth decay is the v1 scoring pipeline. |
| sentence-transformers | Cross-encoder reranking alternative. Same rationale — v2. |
| sqlalchemy | Project uses psycopg3 directly. No ORM layer. Consistent with indexing. |
| numpy | Vector operations are handled by pgvector in SQL. No Python-side vector math needed. |

---

## 3. Configuration Additions

Add to config.py Settings class and document in blank.env.

### config.py additions

```python
# -- Retrieval ---------------------------------------------------------------

# Total corpus tokens below this threshold -> full-context mode
# (pass entire corpus to LLM, skip vector search).
# Above -> chunk mode (HNSW search, parent aggregation).
# Default 30000 is roughly one large documentation page.
# NOTE: Must be <= retrieval_context_budget. If set higher, retrieval
# clamps it to context_budget and logs a warning.
retrieval_full_context_threshold: int = Field(
    default=30000, validation_alias="RETRIEVAL_FULL_CONTEXT_THRESHOLD"
)

# Max total tokens of parent chunks to return in chunk mode.
# Controls how much context retrieval sends to the LLM.
# Default 40000 fits comfortably in any modern model's context window
# while leaving room for system prompt + answer generation.
# Users with larger context windows can increase this.
retrieval_context_budget: int = Field(
    default=40000, validation_alias="RETRIEVAL_CONTEXT_BUDGET"
)

# Hard ceiling on child chunks retrieved from HNSW search before
# parent aggregation. Prevents absurd result sets on vague queries.
retrieval_top_k_children_limit: int = Field(
    default=60, validation_alias="RETRIEVAL_TOP_K_CHILDREN_LIMIT"
)

# Minimum cosine similarity for a child chunk to be considered.
# Anything below this floor is noise and discarded before parent
# aggregation. Range: 0.0-1.0. Lower values are more permissive.
retrieval_similarity_floor: float = Field(
    default=0.3, validation_alias="RETRIEVAL_SIMILARITY_FLOOR"
)

# Depth scoring: linear decay per depth level.
# Score multiplier = max(1.0 - depth * decay_rate, depth_floor).
# Default 0.05 means: depth 0 -> 1.0x, depth 1 -> 0.95x,
# depth 2 -> 0.90x, depth 4+ -> 0.80x (capped at floor).
retrieval_depth_decay_rate: float = Field(
    default=0.05, validation_alias="RETRIEVAL_DEPTH_DECAY_RATE"
)

# Minimum depth multiplier -- prevents deep pages from being
# penalized too heavily. Deep pages with strong similarity
# should still surface.
retrieval_depth_floor: float = Field(
    default=0.80, validation_alias="RETRIEVAL_DEPTH_FLOOR"
)

# HNSW ef_search parameter -- controls recall vs speed tradeoff
# at query time. Higher = better recall, slightly slower.
# pgvector default is 40. We use 100 because parent aggregation
# after search means a missed child can cause an entire parent
# to be absent from results. For WebRAG's corpus sizes (hundreds
# to low thousands of chunks), the latency difference between
# 40 and 100 is sub-millisecond. Configurable for users with
# very large corpora who want to trade recall for speed.
retrieval_hnsw_ef_search: int = Field(
    default=100, validation_alias="RETRIEVAL_HNSW_EF_SEARCH"
)
```

### blank.env additions

```env
# -- Retrieval ---------------------------------------------------------------

# Corpus token threshold for full-context vs chunk mode (default: 30000)
# Below this: entire corpus passed to LLM. Above: HNSW vector search.
# Must be <= RETRIEVAL_CONTEXT_BUDGET. If higher, gets clamped to budget.
# RETRIEVAL_FULL_CONTEXT_THRESHOLD=30000

# Max tokens of retrieved context to return (default: 40000)
# Must be >= RETRIEVAL_FULL_CONTEXT_THRESHOLD.
# RETRIEVAL_CONTEXT_BUDGET=40000

# Max child chunks from HNSW search before parent aggregation (default: 60)
# RETRIEVAL_TOP_K_CHILDREN_LIMIT=60

# Cosine similarity floor -- children below this are discarded (default: 0.3)
# RETRIEVAL_SIMILARITY_FLOOR=0.3

# Depth decay rate per level (default: 0.05)
# RETRIEVAL_DEPTH_DECAY_RATE=0.05

# Minimum depth multiplier floor (default: 0.80)
# RETRIEVAL_DEPTH_FLOOR=0.80

# HNSW ef_search for recall tuning (default: 100, pgvector default: 40)
# Higher = better recall, negligible latency cost at typical corpus sizes.
# RETRIEVAL_HNSW_EF_SEARCH=100
```

### Configuration Validation

At retrieve() call time (not at Settings init, since both values may not yet be set):

```python
def _get_effective_threshold(context_budget_override: int | None = None) -> int:
    """Validate and return the effective full-context threshold.

    Clamps threshold to context budget if misconfigured.
    """
    budget = context_budget_override or settings.retrieval_context_budget
    threshold = settings.retrieval_full_context_threshold

    if threshold > budget:
        import warnings
        warnings.warn(
            f"RETRIEVAL_FULL_CONTEXT_THRESHOLD ({threshold}) "
            f"exceeds RETRIEVAL_CONTEXT_BUDGET ({budget}). "
            f"Clamping threshold to budget value.",
            stacklevel=2,
        )
        return budget
    return threshold
```

Implementation note: _get_effective_threshold() returns the effective threshold (clamped if needed). retrieve() uses the returned value, not settings.retrieval_full_context_threshold directly.

---

## 4. Module Structure

```
src/retrieval/
    models.py          # Pydantic models: RetrievalResult, RetrievedChunk,
                       #   CorpusStats, CitationSpan, TimingInfo
    search.py          # Core search logic: mode decision, HNSW query,
                       #   parent aggregation, depth scoring, token budget,
                       #   surface selection, result assembly
    citations.py       # Citation reconstruction from chunk offsets
                       #   and verbatim span extraction
```

No __init__.py required — project uses pyproject.toml with implicit namespace packages. Public API is imported directly from submodules:

```python
from src.retrieval.search import retrieve
from src.retrieval.models import RetrievalResult, RetrievedChunk
```

### Embedding: no new file

Query embedding uses a thin wrapper added to the existing src/indexing/embedder.py file (see Section 5). Retrieval imports it. No retrieval-specific embedding module.

### Why models.py is separate from search.py

The return contract types (RetrievalResult, RetrievedChunk, etc.) will be imported by both retrieval and orchestration. Keeping them in a dedicated models file avoids circular imports and makes the contract explicit.

---

## 5. Query Embedding

### Required addition to src/indexing/embedder.py

Add a single public function at the bottom of the existing embedder.py:

```python
def embed_query(text: str) -> list[float]:
    """Embed a single query string for retrieval.

    Thin wrapper over embed_texts() for semantic clarity.
    Retrieval embeds exactly one query per call -- this avoids
    the caller needing to index into a list.

    Args:
        text: The query string to embed.

    Returns:
        Embedding vector of length settings.embedding_dimensions.
    """
    return embed_texts([text])[0]
```

**This is a required addition, not optional.** All retrieval code must call embed_query(), never embed_texts([q])[0] directly.

**Justification:** Reuses the existing OpenAI client, config, and error handling from embedder.py. No new HTTP client, no new config. embed_texts([text]) with a single-element list hits the fast path in embed_texts() (batch size 1, no ThreadPoolExecutor overhead). Performance is identical to a hypothetical direct client call.

---

## 6. Mode Determination

### Logic

```python
def _determine_mode(
    conn: Connection,
    source_urls: list[str] | None = None,
    context_budget_override: int | None = None,
) -> tuple[str, int]:
    """Check total corpus token span to decide retrieval mode.

    Args:
        conn: Active psycopg3 connection with register_vector() already called.
        source_urls: Optional list of source URLs to scope the corpus.
            If provided, only documents matching these URLs are considered.
        context_budget_override: Optional override for the context budget,
            allowing orchestration to pass a model-aware budget if known.

    Returns:
        ("full_context", total_tokens) or ("chunk", total_tokens)
    """
    effective_threshold = _get_effective_threshold(context_budget_override)

    if source_urls:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(token_end - token_start), 0) AS total_tokens
            FROM chunks
            WHERE chunk_level = 'parent' AND source_url = ANY(%s)
            """,
            [source_urls],
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(token_end - token_start), 0) AS total_tokens
            FROM chunks
            WHERE chunk_level = 'parent'
            """,
        ).fetchone()

    total_tokens = row[0]

    if total_tokens <= effective_threshold:
        return ("full_context", total_tokens)
    return ("chunk", total_tokens)
```

**Why sum parent token spans, not child:** Parents partition the document without overlap. Children subdivide parents and may have slight overlap at boundaries. Summing parent spans gives the true corpus size. This query is a single aggregate scan over the chunk_level index — fast.

**Why a fixed threshold vs something smarter:** A fixed threshold is predictable, configurable, and requires zero additional signals. The user knows: "below 30k tokens, my entire corpus goes to the LLM; above, vector search kicks in." No surprises. The alternative — estimating coverage confidence before searching — requires running the search first (circular) or an LLM call (expensive). The threshold is the simplest correct solution.

**context_budget_override parameter:** Orchestration may know the model's context window (e.g., from MCP session metadata or user config). If so, it can pass a derived budget. Otherwise the default config kicks in. This is a forward-looking parameter — currently MCP does not expose model metadata, but the interface is ready when it does.

**Edge case:** If total_tokens is 0 (empty corpus), mode is "full_context" and retrieval returns an empty result set. Orchestration handles this — it sees zero chunks returned and triggers ingestion.

---

## 7. Full-Context Mode Pipeline

When the corpus fits within the token threshold, skip vector search entirely and return all content in document reading order.

### Steps

1. **Load all parent chunks**, ordered by (depth ASC, source_url, chunk_index ASC).
   - Depth ordering: seed pages first (depth 0), then discovered pages.
   - Within a document: chunk_index preserves the author's reading order.
   - This ensures the LLM reads content in a natural, structured sequence.

2. **Apply surface selection** per chunk (see Section 9).

3. **Group by source document.** Each group carries source_url, title, depth. Within each group, chunks are in chunk_index order.

4. **Respect context budget as hard cap.** Even in full-context mode, if the total tokens exceed retrieval_context_budget (should not happen if config validation passed, but defense-in-depth), truncate by dropping the lowest-depth-scored documents until within budget.

5. **Build RetrievalResult** with mode="full_context", all parents as RetrievedChunk objects, and corpus stats.

### SQL (Q3: Full-Context Load All Parents)

**Without source_url filter:**

```sql
SELECT
    id, document_id, parent_id, chunk_level,
    chunk_index, section_heading,
    chunk_text, html_text,
    has_table, has_code, has_math,
    has_definition_list, has_admonition, has_steps,
    char_start, char_end, token_start, token_end,
    source_url, fetched_at, depth, title
FROM chunks
WHERE chunk_level = 'parent'
ORDER BY depth ASC, source_url, chunk_index ASC;
```

**With source_url filter:**

```sql
SELECT
    id, document_id, parent_id, chunk_level,
    chunk_index, section_heading,
    chunk_text, html_text,
    has_table, has_code, has_math,
    has_definition_list, has_admonition, has_steps,
    char_start, char_end, token_start, token_end,
    source_url, fetched_at, depth, title
FROM chunks
WHERE chunk_level = 'parent'
  AND source_url = ANY(%s)
ORDER BY depth ASC, source_url, chunk_index ASC;
```

**No scoring in full-context mode.** All content is returned; relevance scoring is unnecessary when the LLM sees everything. The score field on RetrievedChunk is set to 1.0 for all chunks in full-context mode (signals "all content included, no filtering applied").

---

## 8. Chunk Mode Pipeline

When the corpus exceeds the full-context threshold, run the small-to-large retrieval pipeline.

### Step 8.1: Set HNSW ef_search

Before running the HNSW query, set the recall parameter for this transaction:

```python
conn.execute(
    "SET LOCAL hnsw.ef_search = %s",
    [settings.retrieval_hnsw_ef_search],
)
```

SET LOCAL scopes the setting to the current transaction, so it doesn't affect other connections or subsequent queries outside the transaction. This requires the retrieval queries to run inside a transaction block.

**Why ef_search=100 (default):** Parent aggregation after HNSW search means a missed child can cause an entire parent to be absent from results. Higher ef_search means the HNSW graph explores more neighbors, increasing the chance all relevant parents get at least one child hit. For WebRAG's typical corpus sizes (hundreds to low thousands of chunks), the latency difference between ef_search=40 (pgvector default) and 100 is sub-millisecond. The recall gain justifies the negligible cost.

### Step 8.2: HNSW Search Over Children

**Without source_url filter:**

```sql
SELECT
    id, parent_id, document_id,
    chunk_index, section_heading,
    chunk_text, html_text,
    has_table, has_code, has_math,
    has_definition_list, has_admonition, has_steps,
    char_start, char_end, token_start, token_end,
    source_url, fetched_at, depth, title,
    (embedding <=> %s::vector) AS distance
FROM chunks
WHERE chunk_level = 'child'
  AND embedding IS NOT NULL
  AND (embedding <=> %s::vector) < %s
ORDER BY distance ASC
LIMIT %s;
```

**With source_url filter:**

```sql
SELECT
    id, parent_id, document_id,
    chunk_index, section_heading,
    chunk_text, html_text,
    has_table, has_code, has_math,
    has_definition_list, has_admonition, has_steps,
    char_start, char_end, token_start, token_end,
    source_url, fetched_at, depth, title,
    (embedding <=> %s::vector) AS distance
FROM chunks
WHERE chunk_level = 'child'
  AND embedding IS NOT NULL
  AND (embedding <=> %s::vector) < %s
  AND source_url = ANY(%s)
ORDER BY distance ASC
LIMIT %s;
```

Parameters:
- %s::vector — the query embedding (passed twice: once for distance calc in SELECT, once for WHERE filter)
- %s (WHERE distance threshold) — 1.0 - settings.retrieval_similarity_floor (cosine distance = 1 - similarity; floor of 0.3 similarity = distance threshold of 0.7)
- %s (LIMIT) — settings.retrieval_top_k_children_limit (default 60)

**Cosine distance vs similarity — critical conversion:**
pgvector's <=> operator returns **distance** (0 = identical, 2 = opposite). Our config uses **similarity** (1 = identical, 0 = orthogonal). Conversion:

    distance = 1.0 - similarity
    similarity = 1.0 - distance

All user-facing config and return values use **similarity**. SQL uses **distance** internally. search.py handles the conversion at query construction and result parsing.

**Why the distance WHERE clause in SQL (not just app-layer filtering):**
The SQL distance filter gives pgvector's HNSW index a pruning hint — it can terminate graph traversal early when remaining candidates exceed the threshold. This is a meaningful performance optimization. The threshold of 0.7 distance (0.3 similarity) is extremely permissive — a truly relevant chunk will not score below 0.3 cosine similarity against a reasonable query.

**Justification for SQL-level filtering over app-layer-only:** For v1, the simplicity of a single SQL query with the exact threshold outweighs the marginal recall risk at the boundary. HNSW is approximate, so in theory a relevant chunk could be slightly misranked and excluded. In practice, at a 0.3 floor, this risk is negligible — a chunk at the boundary is barely related to the query. If recall issues surface during testing, the mitigation is to move the floor to the app layer (fetch more candidates from SQL, filter in Python). This is flagged as a tuning option, not a v1 requirement.

**Why adaptive instead of fixed top_k:** The similarity floor + hard ceiling approach is naturally adaptive:
- Precise query on well-indexed corpus: maybe 15 children above floor — 15 returned
- Vague query: maybe 55 children above floor — 55 returned (up to ceiling)
- Off-topic query: maybe 2 children above floor — 2 returned (orchestration sees sparse results)
- HNSW search cost is logarithmic regardless of k, so over-fetching is cheap

**Index usage:** This query uses the partial HNSW index chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops) WHERE chunk_level = 'child' AND embedding IS NOT NULL. The WHERE clause in the query matches the index predicate exactly, ensuring pgvector uses the index.

### Step 8.3: Parent Aggregation

After HNSW returns child results:

```python
from collections import defaultdict
from uuid import UUID

# Group children by parent_id
parent_groups: dict[UUID, list[ChildResult]] = defaultdict(list)
for child in child_results:
    parent_groups[child.parent_id].append(child)

# Score each parent by best child similarity, with depth decay
scored_parents = []
for parent_id, children in parent_groups.items():
    best_child = max(children, key=lambda c: c.similarity)
    depth = best_child.depth  # denormalized on chunk row
    depth_weight = max(
        1.0 - depth * settings.retrieval_depth_decay_rate,
        settings.retrieval_depth_floor,
    )
    adjusted_score = best_child.similarity * depth_weight
    scored_parents.append((parent_id, adjusted_score, best_child, children))

# Sort by adjusted score descending
scored_parents.sort(key=lambda x: x[1], reverse=True)
```

**Why max aggregation, not average:** If a parent has one highly relevant child and four irrelevant ones, the average dilutes the signal. Max says "the best evidence in this section is X good" — which is what matters for deciding if this section should be included in context. Average penalizes sections that contain both relevant and irrelevant subsections, which is common in documentation.

### Step 8.4: Depth Scoring Details

Depth scoring adjusts **ranking order, not inclusion**. A chunk passes or fails based on its raw cosine similarity against the floor. Depth only reorders the surviving results.

Score formula: adjusted_score = similarity * max(1.0 - depth * decay_rate, floor)

With defaults (decay_rate=0.05, floor=0.80):

| Depth | Multiplier | Effect |
|---|---|---|
| 0 (seed page) | 1.00 | No penalty — user explicitly asked about this |
| 1 | 0.95 | Slight penalty — one link away from seed |
| 2 | 0.90 | Moderate penalty |
| 3 | 0.85 | |
| 4+ | 0.80 | Floor — deep pages not penalized further |

**Worked example:** A depth-3 chunk with 0.90 raw similarity scores 0.90 * 0.85 = 0.765. A depth-0 chunk with 0.72 raw similarity scores 0.72 * 1.0 = 0.72. The deeper chunk still wins — similarity dominates. Depth only matters when similarity is close.

**Multi-page scenario:** If the corpus has 6 pages and all 6 have relevant chunks above the similarity floor, all 6 contribute parent chunks. Depth scoring only determines the *ordering* among them. If the token budget accommodates all of them, all are returned. If the budget forces truncation, the highest-scored parents (combining similarity and depth) survive.

### Step 8.5: Token Budget Accumulation

After scoring and sorting parents, accumulate until the token budget is exhausted:

```python
selected_parents = []
accumulated_tokens = 0

for parent_id, score, best_child, children in scored_parents:
    parent_tokens = parent_row.token_end - parent_row.token_start
    if accumulated_tokens + parent_tokens > settings.retrieval_context_budget:
        # Always return at least one result, even if it exceeds budget.
        # An oversized result is better than empty -- orchestration can handle it.
        if not selected_parents:
            selected_parents.append(...)
            accumulated_tokens += parent_tokens
        break
    selected_parents.append(...)
    accumulated_tokens += parent_tokens
```

**Why accumulate at parent level:** Parents are the context unit — the text that actually goes to the LLM. Budgeting at the parent level gives predictable, bounded output. The alternative (budgeting at child level then fetching parents) could overshoot wildly if a small child pulls in a large parent.

**Always return at least one result.** Even if the single best parent exceeds the budget, return it. An oversized result is better than an empty one — orchestration can handle the edge case.

### Step 8.6: Fetch Full Parent Rows

The HNSW query in Step 8.2 returns children, not parents. After parent aggregation identifies which parents to include, fetch the full parent chunk rows:

```sql
SELECT
    id, document_id, chunk_index, section_heading,
    chunk_text, html_text,
    has_table, has_code, has_math,
    has_definition_list, has_admonition, has_steps,
    char_start, char_end, token_start, token_end,
    source_url, fetched_at, depth, title
FROM chunks
WHERE id = ANY(%s);
```

Parameter: list of selected parent UUIDs.

**Why a second query instead of JOINing in Step 8.2:** The HNSW index is on children only (partial index: WHERE chunk_level = 'child'). Joining parents into the HNSW query would bypass the index. Two queries (HNSW on children, then fetch parents by ID) is cleaner and lets the index do its job.

### Step 8.7: Group by Source Document

After fetching parents, group them by source_url for coherent presentation:

```python
from collections import defaultdict

# Group by source document
doc_groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
for chunk in selected_parents:
    doc_groups[chunk.source_url].append(chunk)

# Within each document, sort chunks by chunk_index (reading order)
for url in doc_groups:
    doc_groups[url].sort(key=lambda c: c.chunk_index)

# Order document groups by their best parent score (highest first)
ordered_groups = sorted(
    doc_groups.items(),
    key=lambda item: max(c.score for c in item[1]),
    reverse=True,
)
```

**Why group by source:** Interleaved chunks from different pages are harder for the LLM to reason about. Grouping by source document means the LLM processes one source at a time, maintaining coherence. The MCP/orchestration layer can format each group with a document header (title, source_url) followed by its chunks in reading order.

**Compatibility with indexing:** This grouping is fully compatible with how indexing stores data. Each chunk carries denormalized source_url, title, and depth, so grouping requires no joins against the documents table. The chunk_index field preserves the original reading order within each document, which indexing sets sequentially per document.

---

## 9. Surface Selection

For each chunk (parent in both modes), retrieval decides whether to return chunk_text (markdown) or html_text (raw HTML).

### Decision Rule

```python
# These flags indicate the markdown representation is lossy.
# When any is True, html_text preserves structure that markdown loses.
HTML_PREFERRED_FLAGS = (
    "has_table",
    "has_code",
    "has_math",
    "has_definition_list",
    "has_admonition",
)

def _select_surface(chunk_row) -> tuple[str, str]:
    """Return the best text representation for this chunk.

    Prefers html_text when rich content flags indicate markdown is lossy.
    Falls back to chunk_text when html_text is unavailable or no flags trigger.

    Returns:
        Tuple of (selected_text, surface_label) where surface_label is
        "html" or "markdown".
    """
    if any(getattr(chunk_row, flag) for flag in HTML_PREFERRED_FLAGS):
        if chunk_row.html_text is not None:
            return (chunk_row.html_text, "html")
    return (chunk_row.chunk_text, "markdown")
```

### Flag Semantics (from indexing spec)

| Flag | Prefer html_text? | Reason |
|---|---|---|
| has_table | Yes | Markdown loses alignment, merged cells, complex cell content |
| has_code | Yes | Rendering fidelity for syntax highlighting, indentation |
| has_math | Yes | Firecrawl strips all LaTeX from markdown; MathML in HTML is the only representation |
| has_definition_list | Yes | Firecrawl doesn't produce markdown definition-list syntax; HTML dl preserves structure |
| has_admonition | Yes | HTML preserves intent via div class="admonition ..." with CSS classes |
| has_steps | No | Markdown ordered lists are sufficient; flag is informational only |

**has_steps does NOT trigger html_text.** Steps chunks that have html_text got it from a co-occurring flag (e.g., has_code + has_steps). The has_steps flag signals retrieval/MCP for UX treatment only.

### has_math and html_text Storage — Confirmed Correct

**This is verified ground truth from the live codebase and database:**
1. _should_store_html() in indexing includes has_math — indexing stores html_text for math chunks.
2. _extract_html_snippet() has a has_math branch targeting MathML math tags.
3. Retrieval's HTML_PREFERRED_FLAGS includes has_math — surface selection prefers html_text.

All three layers (indexing storage, HTML extraction, retrieval selection) are consistent. A future page with math content but no co-occurring code/table/admonition flags will still have html_text stored and selected correctly. The live database confirms: all 397 math-flagged chunks have html_text populated.

### html_text Deduplication

In the current retrieval design, deduplication is not a concern: we return **parent chunks**, and each parent has a unique html_text if present. Multiple child chunks under the same parent often share the same html_text value (inherited from the same HTML context element during indexing), but since we aggregate to parents before returning, this overlap is naturally eliminated.

If a future retrieval path returns children directly (e.g., for targeted citation extraction), deduplication would be needed: when returning multiple sibling children from the same parent, send the shared html_text once rather than repeating it per child.

---

## 10. Return Contract

### src/retrieval/models.py

All return types are Pydantic BaseModel subclasses. No dataclasses.

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A parent chunk selected for inclusion in the LLM context.

    Contains all fields needed for:
    - Presentation (selected_text with surface selection applied)
    - Citation reconstruction (char/token offsets, source_url, section_heading)
    - Orchestration decisions (score, depth, document grouping)

    In full-context mode, score is 1.0 for all chunks (no filtering applied).
    In chunk mode, score is the depth-adjusted similarity of the best matching child.
    """

    # Identity
    chunk_id: UUID
    document_id: UUID
    parent_id: UUID | None       # None for parents (they are the parent)
    source_url: str
    title: str | None

    # Content (after surface selection)
    selected_text: str           # html_text or chunk_text, per surface selection
    surface: Literal["html", "markdown"]  # which was selected
    section_heading: str | None
    chunk_index: int             # reading order within document

    # Citation offsets (on the original document markdown)
    char_start: int
    char_end: int
    token_start: int
    token_end: int

    # Scoring
    score: float                 # 1.0 in full-context; adjusted similarity in chunk mode
    raw_similarity: float | None # pre-depth-adjustment similarity; None in full-context
    depth: int

    # Rich content flags (for downstream formatting decisions)
    has_table: bool
    has_code: bool
    has_math: bool
    has_definition_list: bool
    has_admonition: bool
    has_steps: bool

    # Metadata
    fetched_at: datetime


class CorpusStats(BaseModel):
    """Corpus-level statistics for orchestration decision-making.

    Orchestration uses these to judge whether the corpus is sufficient
    or needs expansion. Retrieval computes them; orchestration interprets.
    """

    total_documents: int
    total_parent_chunks: int
    total_tokens: int            # sum of parent token spans
    documents_matched: list[str] # source_urls that contributed at least one result


class TimingInfo(BaseModel):
    """Retrieval timing breakdown for observability and optimization."""

    embed_ms: float              # query embedding API call time
    search_ms: float             # HNSW search + parent aggregation + fetch time
    total_ms: float              # end-to-end retrieval time


class RetrievalResult(BaseModel):
    """Complete retrieval output -- the contract between retrieval and orchestration.

    Designed to be rich enough that orchestration never needs to call back
    into retrieval for additional metadata. All coverage/sufficiency
    decisions are orchestration's responsibility based on this data.
    """

    mode: Literal["full_context", "chunk"]
    chunks: list[RetrievedChunk]  # grouped by source_url, ordered by chunk_index within group
    query_embedding: list[float]  # cached for potential reuse (e.g., orchestration re-retrieval)
    corpus_stats: CorpusStats
    timing: TimingInfo

    # Convenience accessors for orchestration
    @property
    def is_empty(self) -> bool:
        """No results found -- corpus may be empty or query may be off-topic."""
        return len(self.chunks) == 0

    @property
    def top_score(self) -> float:
        """Highest chunk score. 1.0 in full-context mode."""
        if not self.chunks:
            return 0.0
        return max(c.score for c in self.chunks)

    @property
    def source_urls(self) -> list[str]:
        """Unique source URLs that contributed chunks, preserving order."""
        seen: set[str] = set()
        urls: list[str] = []
        for c in self.chunks:
            if c.source_url not in seen:
                seen.add(c.source_url)
                urls.append(c.source_url)
        return urls
```

### Design Decisions for the Return Contract

**selected_text + surface instead of both texts:** Retrieval applies surface selection before returning. The consumer gets the correct text representation directly, plus a surface label indicating which was chosen. This avoids every downstream consumer re-implementing the flag-checking logic.

**query_embedding included:** Orchestration may need it for re-retrieval after corpus expansion. Caching it avoids a redundant API call. Cost: ~6KB per result (1536 floats * 4 bytes). Negligible.

**raw_similarity separate from score:** Orchestration might want to see both the pre-adjustment similarity and the post-depth-adjustment score. This supports debugging and allows orchestration to apply its own scoring adjustments without reverse-engineering depth decay.

**documents_matched in CorpusStats:** Orchestration needs to know which pages contributed evidence vs which were searched but yielded nothing. This drives expansion decisions: "I have 6 pages indexed but only 2 contributed results — should I explore more links from the contributing pages?"

**All Pydantic BaseModel, no dataclasses.** Consistent with the rest of the codebase (config.py uses pydantic-settings). Pydantic provides validation, serialization, and model_dump() for free.

---

## 11. Citation Reconstruction

### src/retrieval/citations.py

Citation reconstruction is **deterministic string lookup** using stored chunk offsets. No LLM involved.

### CitationSpan Model

```python
class CitationSpan(BaseModel):
    """A verbatim quote tied to its source for citation output.

    Used to produce the citations block at the bottom of the LLM response.
    If the verbatim text already appears in the answer prose, the citations
    block supplies source_url and section_heading as attribution only
    (no re-quoting -- avoids the "double citation" problem).
    """

    verbatim_text: str
    source_url: str
    section_heading: str | None
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    title: str | None
```

### Core Function

```python
def extract_citation(
    chunk: RetrievedChunk,
    quote_start: int,
    quote_end: int,
) -> CitationSpan | None:
    """Extract a verbatim citation span from a chunk.

    Given character offsets within the original document, verify that the
    span falls within this chunk's char_start..char_end range and return
    the verbatim text with source attribution.

    This is a deterministic O(1) lookup -- no fuzzy matching, no LLM.
    The offsets come from indexing and survive through the entire pipeline:
    ingestion -> indexing -> retrieval -> citation.

    Args:
        chunk: The parent chunk containing the quoted text.
        quote_start: Character offset in the original document where the quote begins.
        quote_end: Character offset in the original document where the quote ends.

    Returns:
        CitationSpan with verbatim text and source metadata, or None if
        the offsets don't fall within this chunk's range.
    """
    # Verify the quote span falls within this chunk's range
    if quote_start < chunk.char_start or quote_end > chunk.char_end:
        return None

    # Extract verbatim text from chunk_text using relative offsets
    relative_start = quote_start - chunk.char_start
    relative_end = quote_end - chunk.char_start
    verbatim = chunk.selected_text[relative_start:relative_end]

    return CitationSpan(
        verbatim_text=verbatim,
        source_url=chunk.source_url,
        section_heading=chunk.section_heading,
        char_start=quote_start,
        char_end=quote_end,
        token_start=chunk.token_start,  # approximate; chunk-level granularity
        token_end=chunk.token_end,
        title=chunk.title,
    )
```

### How Citations Work End-to-End

1. **Retrieval** returns RetrievedChunk objects with char_start, char_end, source_url, section_heading.
2. **The LLM** generates an answer. When it quotes text, it should reference the chunk's char offsets (the MCP layer instructs it to do this via system prompt).
3. **Citation verification** uses extract_citation() to confirm the LLM's quoted span actually exists in the source material at the claimed offsets. This is a deterministic O(1) lookup — not fuzzy matching, not LLM-based.
4. **The response** includes a citations block at the bottom with [source_url#section_heading] for each cited span. If a verbatim quote already appears in the answer prose, the citation supplies attribution only without re-quoting.

### Why Char Offsets Are Critical

Char offsets prevent hallucinated citations. Without them, you'd need fuzzy string matching to verify quotes (slow, unreliable) or trust the LLM's self-reported sources (risky). With offsets, citation verification is: "does document_text[char_start:char_end] match the quoted span? Yes/no." Binary, fast, trustworthy.

This is enabled by indexing: every chunk carries char_start and char_end relative to the original document markdown. These survive through the entire pipeline — ingestion to indexing to retrieval to citation.

---

## 12. Image Handling at Retrieval Time

Indexing preserves ![alt text](url) as-is in chunk_text. No has_image flag exists — images don't need HTML fallback since Firecrawl preserves them faithfully in markdown with absolute URLs.

### Retrieval's Responsibility

Retrieval passes image markdown through as-is in selected_text. No image processing, fetching, or vision API calls at retrieval time.

### MCP/Orchestration Layer Responsibility (noted here for completeness)

At response assembly time, the MCP layer decides per-image:
- If the image URL appears in a retrieved chunk being sent to the LLM **and** alt text is meaningful or the image is contextually relevant: fetch and pass alongside as a vision input
- Otherwise: pass alt text + URL as a text reference

**Alt text quality varies by source:** Wikipedia often has empty alt text; scikit-learn uses filename-based alt text. The MCP layer should not assume alt text is always informative. This is an MCP/orchestration concern, not a retrieval concern — noted here so the design is complete.

---

## 13. SQL Queries Reference

All queries use parameterized psycopg3 syntax (%s placeholders). No string interpolation in SQL. All queries are housed in search.py.

Database connections use postgresql:// DSN format for psycopg.connect(), consistent with the rest of the codebase. Not postgresql+psycopg:// (that is SQLAlchemy syntax and is wrong for this project).

### Q1: Corpus Token Count (mode determination)

**Without filter:**

```sql
SELECT COALESCE(SUM(token_end - token_start), 0) AS total_tokens
FROM chunks
WHERE chunk_level = 'parent';
```

**With source_url filter:**

```sql
SELECT COALESCE(SUM(token_end - token_start), 0) AS total_tokens
FROM chunks
WHERE chunk_level = 'parent'
  AND source_url = ANY(%s);
```

**Index used:** chunks_chunk_level_idx

### Q2: Corpus Document Count

**Without filter:**

```sql
SELECT COUNT(*) AS total_documents FROM documents;
```

**With source_url filter:**

```sql
SELECT COUNT(*) AS total_documents
FROM documents
WHERE source_url = ANY(%s);
```

**Index used:** documents_source_url_key (unique index)

### Q3: Full-Context — Load All Parents

See Section 7 for both variants (with/without filter).

**Index used:** chunks_chunk_level_idx

### Q4: Chunk Mode — HNSW Search Over Children

See Section 8, Step 8.2 for both variants (with/without filter).

**Index used:** chunks_embedding_hnsw_idx (partial HNSW index on children with embeddings)

### Q5: Fetch Parent Chunks by ID

```sql
SELECT
    id, document_id, chunk_index, section_heading,
    chunk_text, html_text,
    has_table, has_code, has_math,
    has_definition_list, has_admonition, has_steps,
    char_start, char_end, token_start, token_end,
    source_url, fetched_at, depth, title
FROM chunks
WHERE id = ANY(%s);
```

**Index used:** Primary key index on chunks.id

### Q6: Parent Chunk Count

**Without filter:**

```sql
SELECT COUNT(*) FROM chunks WHERE chunk_level = 'parent';
```

**With source_url filter:**

```sql
SELECT COUNT(*) FROM chunks
WHERE chunk_level = 'parent'
  AND source_url = ANY(%s);
```

**Index used:** chunks_chunk_level_idx

---

## 14. Testing Strategy

### Unit Tests (tests/test_retrieval.py)

Test with the three existing indexed documents (scikit-learn ensemble, Python glossary, Mount Everest Wikipedia).

**Mode determination tests:**
- Verify full-context mode activates when corpus tokens < threshold
- Verify chunk mode activates when corpus tokens > threshold
- Verify source_url filtering scopes the token count correctly
- Verify config validation: threshold > budget gets clamped with warning

**Full-context mode tests:**
- Returns all parent chunks for the corpus
- Chunks are grouped by source_url
- Chunks within each group are in chunk_index order
- Surface selection applies correctly (scikit-learn chunks with has_math return html_text)
- Score is 1.0 for all chunks

**Chunk mode tests:**
- Query "gradient boosting" returns scikit-learn parents with highest scores
- Query "Mount Everest" returns Wikipedia parents
- Parent deduplication: multiple children from the same parent produce one parent result
- Depth scoring: seed page chunks score higher than depth-2 chunks at similar raw similarity
- Token budget: result set doesn't exceed configured budget
- Similarity floor: low-quality matches are excluded
- At least one result always returned (even if it exceeds budget)
- ef_search is set correctly for the query session

**Surface selection tests:**
- Chunk with has_math=True and non-null html_text returns html_text, surface="html"
- Chunk with has_table=True and null html_text falls back to chunk_text, surface="markdown"
- Chunk with no flags returns chunk_text, surface="markdown"
- Chunk with has_steps=True only returns chunk_text (has_steps doesn't trigger html)

**Citation tests:**
- extract_citation() returns correct verbatim text for valid offsets
- Returns None for offsets outside chunk range
- Returned span matches exact characters from original document

**Return contract tests:**
- RetrievalResult.is_empty is True when no chunks returned
- RetrievalResult.top_score returns highest score
- RetrievalResult.source_urls returns deduplicated URLs in order
- CorpusStats fields are accurate against known test corpus
- TimingInfo fields are non-negative

### Integration Tests

- End-to-end: retrieve("What is gradient boosting?") against indexed scikit-learn page returns relevant parents with scikit-learn source_url
- Mode switch: index a single small page then full-context mode; add more pages to exceed threshold then chunk mode
- Empty corpus: retrieve("anything") returns empty result with zero corpus stats

---

## 15. Future Considerations

### 15.1 Reranking (v2)

After HNSW retrieval and parent aggregation, add an optional reranking pass:
- Cross-encoder model (sentence-transformers) or API (Cohere Rerank)
- Rerank the selected parent chunks (not all children — too many)
- Config: RETRIEVAL_RERANK_ENABLED=false, RETRIEVAL_RERANK_MODEL, RETRIEVAL_RERANK_API_KEY
- Reranking only applies in chunk mode (full-context returns everything)

### 15.2 Hybrid Search (v2+)

Combine vector similarity with keyword matching (BM25). Useful for queries containing exact terms (function names, class names, error codes) where keyword match is stronger than semantic similarity. pgvector can be combined with PostgreSQL's built-in full-text search (tsvector/tsquery) in a single query with score fusion.

### 15.3 Orchestration Loop Ownership

**FLAG FOR ORCHESTRATION DESIGN — do not resolve in retrieval.**

Who drives the retrieve then expand then re-retrieve loop?

**Option A: Internal orchestration.** WebRAG's orchestrator makes its own LLM API calls to judge coverage sufficiency and drive expansion. WebRAG controls the loop end-to-end. Cost: additional API calls and latency. Benefit: self-contained system.

**Option B: LLM-driven orchestration.** WebRAG exposes retrieve() and expand() as separate MCP tools. The reasoning LLM (Claude/ChatGPT — the user's subscription model) calls retrieve(), examines results, decides if expansion is needed, calls expand(), then retrieve() again. The sufficiency judgment is "free" for subscription users because it happens in their existing chat session.

**Option C: Hybrid.** Score-based heuristics as fast gate; LLM judgment for ambiguous cases.

This decision fundamentally shapes orchestration architecture, MCP tool design, and cost model. Retrieval's return contract (Section 10) is designed to support all three options — it provides rich enough data for any coverage judgment strategy.

### 15.4 Math Rendering

Chunks with has_math=True have MathML in their html_text. The MCP layer may need to convert MathML to LaTeX for models that handle LaTeX better than raw HTML. This is an MCP/presentation concern, not retrieval.

### 15.5 Vision Model Pass for Images

Currently no image content is processed at retrieval time. A future enhancement could:
- Detect ![alt](url) patterns in retrieved chunks
- Fetch images and pass as vision inputs alongside text
- Particularly valuable when alt text is empty or uninformative
This is noted in Section 12 and is an MCP layer responsibility.

### 15.6 Model-Aware Context Budgeting

Currently MCP does not expose which model is calling WebRAG or its context window size. The context_budget_override parameter on retrieve() is a forward-looking interface for when this metadata becomes available. When it does, orchestration can derive the budget as model_context_window * 0.4 (conservative) and pass it to retrieval, replacing the static config default.

---

## Implementation Notes for Agent

### Connection Preconditions

retrieve() accepts a psycopg.Connection as its first parameter. The caller (orchestration/MCP layer) is responsible for:

1. **Connection is active** and connected to the WebRAG database using postgresql:// DSN format (e.g., postgresql://webrag:webrag@localhost:5432/webrag). Do NOT use postgresql+psycopg:// — that is SQLAlchemy syntax and is incorrect for this project.
2. **register_vector(conn) has been called** on this connection (from pgvector.psycopg). Without this, pgvector's vector type adaptation fails and embedding queries will error. The indexing layer already does this — use the same connection setup pattern.
3. **Schema exists** — init_schema(conn) has been run at least once (tables and indexes exist). Retrieval does not create or modify schema.

Example connection setup (for reference, not retrieval's responsibility):

```python
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect("postgresql://webrag:webrag@localhost:5432/webrag")
register_vector(conn)
```

### File Creation Order

1. src/retrieval/models.py — define all Pydantic models first (other modules import them)
2. Add embed_query() to src/indexing/embedder.py — single function addition, required
3. Add retrieval config fields to config.py — extend Settings class
4. Update blank.env — add commented retrieval config section
5. src/retrieval/citations.py — CitationSpan model + extract_citation()
6. src/retrieval/search.py — core search logic (imports models, embedder, config)

No __init__.py needed — project uses pyproject.toml with implicit namespace packages.

### Key Implementation Details

- **All database access uses psycopg3** with parameterized queries (%s placeholders). No string interpolation in SQL. No f-strings in queries.
- **Connection management:** retrieve() takes a psycopg.Connection as a parameter (same pattern as indexing). The caller manages connection lifecycle.
- **Transaction for ef_search:** The HNSW ef_search setting requires SET LOCAL, which is scoped to the current transaction. Wrap chunk mode queries in a transaction block (with conn.transaction():).
- **Timing:** Use time.perf_counter() around embed and search sections. Report in milliseconds.
- **UUID handling:** psycopg3 natively handles Python uuid.UUID to PostgreSQL UUID conversion. No string casting needed.
- **Vector passing:** psycopg3 with pgvector sends embeddings as Python lists of floats. Cast to ::vector in SQL. Requires register_vector(conn) (caller's responsibility, see Connection Preconditions above).
- **Error handling:** If the corpus is empty (no documents/chunks), return an empty RetrievalResult with mode="full_context", empty chunks, and zeroed corpus stats. Don't raise — let orchestration handle it.
- **No async:** Retrieval uses synchronous psycopg3, consistent with the rest of the codebase. The embedding API call is the only I/O besides DB queries, and it's a single request (no batching needed).

### Inline Comment and Docstring Standards

- Every public function gets a docstring explaining: what it does, why, and what the caller should know.
- SQL queries get inline comments explaining: what index they use, why the query is structured this way, and what the parameters are.
- Configuration values get inline comments explaining: what they control, what the default means, and when a user would change them.
- Non-obvious algorithmic choices (max vs average aggregation, depth decay formula, similarity floor vs distance threshold conversion) get brief inline comments explaining the reasoning.

### retrieve() Function Signature

```python
def retrieve(
    conn: Connection,
    query: str,
    *,
    source_urls: list[str] | None = None,
    context_budget_override: int | None = None,
) -> RetrievalResult:
    """Search the indexed corpus and return the best evidence for a query.

    This is the single public entry point for the retrieval layer.
    Returns a rich RetrievalResult that orchestration uses to decide
    next steps (answer directly, expand corpus, or re-retrieve).

    Args:
        conn: Active psycopg3 connection with register_vector() called.
        query: The user's query string to search for.
        source_urls: Optional list of source URLs to scope the search.
            If provided, only chunks from these documents are searched.
        context_budget_override: Optional token budget override.
            If provided, overrides RETRIEVAL_CONTEXT_BUDGET from config.
            Useful when orchestration knows the model's context window.

    Returns:
        RetrievalResult with ranked chunks, corpus stats, and timing.
    """
```
