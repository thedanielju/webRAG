# WebRAG Indexing Layer Design

## Scope

The indexing layer receives `NormalizedDocument` objects from ingestion and produces retrievable, citation-ready chunks stored in Postgres with pgvector. It does not fetch web pages, select links, or drive the corpus expansion loop. It owns the document record that orchestration uses for deduplication and change detection.

Indexing always runs regardless of retrieval mode. Full-context mode (passing whole pages to the LLM) is a retrieval-layer decision — the index must exist either way because citation reconstruction depends on stored chunk offsets.

---

## Inputs and Outputs

**Input:** `NormalizedDocument` (from ingestion)

**Outputs:**
- One document record per ingested page (Postgres `documents` table)
- One or more parent chunk records per document (Postgres `chunks` table, `chunk_level = 'parent'`)
- One or more child chunk records per parent (Postgres `chunks` table, `chunk_level = 'child'`)
- pgvector embedding per child chunk

---

## Why Index at All

Straightforward question worth answering explicitly, since WebRAG's corpus expansion is selective and corpora stay relatively small.

**Full-context mode works for small corpora.** When the total corpus fits within the LLM's context window, passing full page markdown directly produces better results than chunk retrieval — no recall risk from imprecise embedding matches. Indexing still runs, but retrieval uses stored documents rather than vector search.

**Chunk retrieval is necessary when:**
- The corpus grows past the token budget threshold (configurable, suggested default ~100K tokens)
- The user enables it explicitly
- Citation offset reconstruction is needed (always — see below)

**Citation fidelity requires chunk-level storage regardless of retrieval mode.** When the LLM quotes a span, that span must be verified against stored chunk text using char offsets to guarantee verbatim accuracy and prevent hallucinated citations. This is a deterministic string lookup — no LLM involved — and it requires indexed chunks to work even if the LLM received full-page context.

**Cross-page retrieval.** When the answer spans multiple pages, vector search over child chunks is more precise than passing 30 full pages and hoping the LLM finds the right passages.

**Corpus persistence over time.** WebRAG is designed as a persistent research memory. As corpora grow across sessions, an index becomes necessary.

---

## Data Model

### `documents` table

One record per ingested `NormalizedDocument`. Owned entirely by the indexing layer. Orchestration reads this table to check whether a URL has been indexed and whether its content has changed.

| Field | Type | Source | Notes |
|---|---|---|---|
| `id` | uuid | generated | primary key |
| `url` | text | NormalizedDocument.url | requested URL |
| `source_url` | text | NormalizedDocument.source_url | canonical post-redirect URL; used for citations |
| `title` | text \| null | NormalizedDocument.title | may be None for PDFs |
| `description` | text \| null | NormalizedDocument.description | extracted from HTML meta tags at no LLM cost; primary signal for traversal decisions and diagram node labels; URL path used as fallback when null |
| `language` | text \| null | NormalizedDocument.language | |
| `status_code` | int \| null | NormalizedDocument.status_code | |
| `published_time` | text \| null | NormalizedDocument.published_time | |
| `modified_time` | text \| null | NormalizedDocument.modified_time | |
| `doc_type` | text | NormalizedDocument.doc_type | `"html"` \| `"pdf"` |
| `content_hash` | text | NormalizedDocument.content_hash | sha256 of markdown; orchestration compares against stored value to detect changes |
| `fetched_at` | timestamptz | NormalizedDocument.fetched_at | when ingestion fetched the page; recency signal for retrieval scoring and staleness detection |
| `indexed_at` | timestamptz | generated | when indexing wrote this record |
| `depth` | int | orchestration-supplied | 0 = seed URL, 1 = first expansion, 2 = second, etc. Not derived by indexing — orchestration passes this value at index time. Used by retrieval for depth-based scoring. |

**Deduplication logic (called by orchestration before ingesting):**
- URL not in `documents` → ingest and index normally
- URL in `documents`, `content_hash` matches → skip, no-op
- URL in `documents`, `content_hash` differs → atomic delete of all child and parent chunks for this document, then reindex

### `chunks` table

Stores both parent and child chunks. Children reference their parent via `parent_id`. Both levels share this table; `chunk_level` distinguishes them.

| Field | Type | Notes |
|---|---|---|
| `id` | uuid | primary key |
| `document_id` | uuid | FK → documents.id |
| `parent_id` | uuid \| null | FK → chunks.id; null for parent chunks |
| `chunk_level` | text | `"parent"` \| `"child"` |
| `chunk_index` | int | position of this chunk within its parent (for children) or within the document (for parents); preserves reading order |
| `section_heading` | text \| null | nearest `#` or `##` heading above this chunk; carried forward onto all sub-chunks when a section is split |
| `chunk_text` | text | raw text of this chunk in markdown |
| `html_text` | text \| null | HTML representation of this chunk; populated only when one or more rich content flags require it; null otherwise |
| `has_table` | bool | true if chunk contains a markdown table or HTML `<table>` |
| `has_code` | bool | true if chunk contains a fenced code block |
| `has_math` | bool | true if chunk contains math content (detected via MathML `<math>` tags, MathJax `<mjx-*>` elements, or `MathJax`/`katex`/`math` CSS classes in HTML; Firecrawl strips LaTeX from markdown so detection is HTML-only) |
| `has_definition_list` | bool | true if chunk contains a definition list (detected via HTML `<dl>` tag; Firecrawl does not produce markdown definition-list syntax) |
| `has_admonition` | bool | true if chunk contains a callout or admonition (markdown: bare keyword line like `Note`, `Warning`, `Tip`, etc.; HTML: `<div class="admonition ...">` or elements with `admonition`/`note`/`warning`/`tip`/`important`/`caution`/`danger`/`info` CSS classes) |
| `has_steps` | bool | true if chunk contains a numbered step sequence (ordered list used procedurally) |
| `char_start` | int | character offset of chunk_text start within the document's full markdown |
| `char_end` | int | character offset of chunk_text end within the document's full markdown |
| `token_start` | int | token offset start; tokenizer matches the configured embedding model |
| `token_end` | int | token offset end |
| `embedding` | vector(N) | pgvector column; populated for child chunks only; null for parent chunks; N is set at schema creation time per the configured embedding model |
| `source_url` | text | denormalized from document for fast citation lookup without join |
| `fetched_at` | timestamptz | denormalized from document |

**Rich content flag detection** happens at chunk creation time via pattern matching on chunk text and HTML — no external calls. `html_text` is populated whenever any flag that benefits from it is true, giving retrieval a higher-fidelity rendering surface for those chunks.

---

## Chunking Strategy

### Primary Strategy: Heading-Based Parent-Child

Web documentation has reliable heading structure from Firecrawl's markdown output. This is exploited directly as the natural document hierarchy.

**Parent chunks** are entire `##` sections — the heading plus all content beneath it until the next `##`. If a document has no `##` headings, fall back to `#` boundaries. If no headings at all, the entire document is treated as one parent and the fallback strategy handles child splitting within it.

**Child chunks** are the individual paragraphs within each parent section. A paragraph is a block of text separated by blank lines in the markdown.

**Why parent-child works for web content:** A page on scikit-learn's `RandomForestClassifier` has sections `## Parameters`, `## Examples`, `## Notes`. Each section becomes a parent (~300–800 tokens). The paragraphs within `## Examples` become children (~100–250 tokens each). A query for "how to set max_depth in a random forest" hits the specific child paragraph mentioning `max_depth` via vector search. The retrieval layer returns the full `## Parameters` parent as LLM context. The child's `char_start`/`char_end` offsets anchor the verbatim citation. Precision from children, context from parents, no overlap needed.

**Oversized sections:** If a parent section exceeds the token ceiling (default: 1000 tokens), it is split at paragraph boundaries into sub-chunks. Each sub-chunk carries the original section heading in `section_heading` and a `chunk_index` indicating its position within the split. Reading order and heading context are preserved without truncation.

**Token targets (config, not hardcoded):**
- Child chunks: **256 tokens** default. NVIDIA benchmarks show factoid and precise queries — the dominant WebRAG pattern — peak at 256–512 tokens. Chroma research shows 400-token chunks with recursive splitting reach 88–89% recall without semantic chunking overhead. 256 is the right default for technical documentation queries.
- Parent ceiling: **1000 tokens** default. Covers most `##` sections comfortably. Any section exceeding this is split as described above, consistent with the 3–5x child-to-parent ratio recommended in parent-child RAG literature.

### Fallback Strategy: Recursive Character Splitting

Applied when a document or section has no heading structure — landing pages, narrative blog posts, some PDFs where Firecrawl degrades to flat text.

Split priority order: `\n\n` (paragraphs) → `\n` (lines) → `. ` (sentences) → ` ` (words). Target child chunk size: 256 tokens. Parent chunks in fallback mode are groups of 4 consecutive children (~800–1000 tokens), preserving local reading context.

`section_heading` will be null for fallback-chunked content. Retrieval handles null headings gracefully — it is a valid and expected state for unstructured pages.

### Rich Content Handling

Firecrawl returns both markdown and HTML. Markdown is the primary embedding surface. HTML is stored on chunks where it adds fidelity that markdown loses. All flags are set by indexing at chunk creation time via pattern matching — no external calls.

| Content type | Markdown behavior | Flag | HTML stored |
|---|---|---|---|
| Tables | Loses alignment and complex cell content | `has_table` | Yes — HTML is authoritative |
| Code blocks | Preserved as fenced blocks | `has_code` | Yes — for rendering fidelity |
| Math (LaTeX/MathJax) | Firecrawl strips all LaTeX syntax (`$`/`$$`) from markdown; equations appear as bare Unicode approximations or are dropped entirely. MathML (`<math>` tags with `<mi>`, `<mo>`, `<mn>` children) and MathJax elements are preserved in HTML only. | `has_math` | Yes — HTML is the only representation; snippet targets clean MathML `<math>` tags |
| Definition lists | Firecrawl does not produce markdown definition-list syntax; renders as plain paragraphs | `has_definition_list` | Yes — HTML `<dl><dt><dd>` preserves structure |
| Admonitions / callouts | Firecrawl renders as a bare keyword line (`Note`, `Warning`, etc.) followed by body text — not as blockquotes | `has_admonition` | Yes — HTML preserves intent via `<div class="admonition ...">` |
| Numbered step sequences | Preserved as ordered lists; order semantics may lose styling | `has_steps` | No — markdown sufficient; flag signals retrieval |
| Images | Inline as `![alt text](url)` | — | No — URL + alt text stored as-is in chunk text |

**Image handling:** Indexing preserves `![alt text](url)` as-is in `chunk_text`. The retrieval/MCP layer decides at response time whether to fetch and pass the image to the LLM: if the image URL appears within a retrieved chunk being sent as context, and the alt text is meaningful or the image is contextually embedded in relevant content, it is fetched and passed alongside. Otherwise alt text and URL are passed as a text reference. No vision model processing at indexing time.

Retrieval selects `html_text` over `chunk_text` for rendering when `has_table`, `has_code`, `has_math`, `has_definition_list`, or `has_admonition` are true. `html_text` is null when no flags requiring it are true, keeping storage lean.

**Detection coverage (finalized — updated after Firecrawl output audit):**
- Math: HTML-only detection. Firecrawl strips all LaTeX (`$`/`$$`) from markdown. Detection checks for MathML `<math>` tags, MathJax 3 `<mjx-*>` custom elements, and `MathJax`/`katex`/`math` CSS classes. HTML snippet extraction targets clean MathML `<math>` tags (containing `<mi>`, `<mo>`, `<mn>` etc.) and excludes MathJax visual rendering elements (`<mjx-math>`, `<mjx-container>`).
- Definition lists: HTML-only detection via `<dl>` tags. Firecrawl does not produce markdown definition-list syntax (`term:\n    definition`); it renders definitions as plain paragraphs.
- Admonitions: Markdown detection matches Firecrawl's bare keyword format — a standalone line containing `Note`, `Warning`, `Important`, `Tip`, `Caution`, `Danger`, `Info`, `Success`, `Example`, `See also`, or `Deprecated since`. HTML detection via CSS classes containing `admonition`, `note`, `warning`, `tip`, `important`, `caution`, `danger`, or `info`.
- Steps: detect ordered lists using `1.` and `1)` markers, including nested ordered lists.

---

## Embedding

### What Embedding Is

Each child chunk's text is passed through an embedding model — a neural network that maps text to a fixed-size float vector (e.g. 1536 numbers for OpenAI's default model). Semantic similarity maps to geometric proximity in that vector space: "how to set max_depth" and "max_depth parameter controls tree depth" end up close; "max_depth" and "French cuisine" end up far apart. At retrieval time, the query is embedded the same way and nearest-neighbor search over stored child embeddings finds the most relevant chunks. Parent chunks are not embedded — they are fetched by `parent_id` pointer after child retrieval.

### Provider Configuration

The indexing layer calls a single interface — `embed_texts(texts: list[str]) -> list[list[float]]` — and does not know or care which backend is running. Configuration is `base_url` + `model_name` + optional `api_key`, using the OpenAI-compatible `/v1/embeddings` endpoint that all major providers share. This means any OpenAI-compatible local server (Ollama, LM Studio, HuggingFace TGI) works as a drop-in with no code changes.

| Backend | base_url | model_name | api_key |
|---|---|---|---|
| OpenAI (default) | `api.openai.com` | `text-embedding-3-small` | required |
| Ollama (local) | `localhost:11434` | `nomic-embed-text` | not required |
| LM Studio / TGI / other | user-configured | user-configured | optional |

**Default: OpenAI `text-embedding-3-small`.** At worst-case WebRAG scale (thousands of pages, ~20 child chunks each, ~256 tokens per chunk), embedding cost is on the order of cents. Self-hosting on a capable GPU (RTX 5070 or equivalent, 12GB VRAM) is a legitimate zero-cost alternative — `nomic-embed-text` (768 dims, ~550MB VRAM) and `bge-m3` (1024 dims, ~2.3GB VRAM) match frontier embedding quality at WebRAG's scale and run comfortably within VRAM constraints.

### Transport and Concurrency

`embed_texts()` splits the full text list into batches of `EMBEDDING_BATCH_SIZE` (default 256 texts) and sends them concurrently via `ThreadPoolExecutor` with at most `EMBEDDING_MAX_WORKERS` threads (default 4). This stays within the sync `def` contract — no asyncio — while reducing wall-clock embedding time by ~5x for large chunk sets (e.g. 1200 chunks: ~70s → ~10-15s).

Single-batch calls (≤256 texts) skip the executor entirely to avoid overhead.

If any batch fails (timeout, 429, dimension mismatch), the entire `embed_texts()` call fails immediately — no partial vectors are returned. `index_batch()` depends on this all-or-nothing guarantee to prevent inconsistent chunk state in Postgres.

**Configuration** (via `.env` / `config.py`):
- `EMBEDDING_BATCH_SIZE` — texts per API request (default 256). Balances HTTP overhead against per-request latency.
- `EMBEDDING_MAX_WORKERS` — max concurrent threads (default 4). Conservative for OpenAI rate limits; increase for higher-tier plans or local servers.

### DB Write Performance

Chunk inserts use `cursor.executemany()` instead of per-row `cursor.execute()` loops. `executemany` sends the parameterised INSERT once and streams all row data in a single client-server round-trip, reducing write time from ~3s to <0.5s for 1000+ chunk batches. Semantic behaviour is identical.

### Chunking/Embedding Overlap

For multi-document `index_batch()` calls, chunking of subsequent documents proceeds in the main thread while embedding of already-chunked documents runs in a background thread. This is safe because chunking is pure CPU work with no shared mutable state, and `embed_texts()` operates on an independent list of strings. For single-document calls, the overlap is skipped.

### Tokenizer

The tokenizer used to compute `token_start`/`token_end` offsets must match the embedding model. Both are set together as a paired config decision and read at indexing startup, making mismatch impossible when configured correctly.

| Embedding model | Tokenizer |
|---|---|
| OpenAI `text-embedding-3-small` | `tiktoken` with `cl100k_base` encoding |
| `nomic-embed-text` | HuggingFace tokenizer for `nomic-ai/nomic-embed-text-v1` |
| `bge-m3` | HuggingFace tokenizer for `BAAI/bge-m3` |

### pgvector Dimension

The pgvector `embedding` column is defined as `VECTOR(N)` where N is the output dimensionality of the configured embedding model. This is set once at schema creation time by reading the configured dimension from config. Changing models requires dropping and recreating the column and re-embedding the entire corpus — there is no migration path.

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-small` | 1536 | Default; fits within pgvector's HNSW index limit of 2000 |
| `nomic-embed-text` | 768 | Smaller and faster; well-suited for WebRAG's scale |
| `bge-m3` | 1024 | Best multilingual support; 8192-token context window |
| `text-embedding-3-large` | 3072 | Exceeds HNSW limit of 2000; requires half-precision (`halfvec`) storage — not recommended |

**Index type: HNSW with cosine similarity** (`vector_cosine_ops`). HNSW is preferred over IVFFlat because it requires no training step, can be built on an empty table, and provides a better speed-recall tradeoff at WebRAG's corpus scale. Cosine similarity is standard for text embedding comparison. The default schema ships with dimension 1536; users running local models set their model's dimension in config before running schema setup.

---

## Re-indexing

When orchestration detects a `content_hash` mismatch on a URL already in `documents`:

1. Delete all `chunks` records where `document_id` matches (cascades child and parent)
2. Delete the `documents` record
3. Re-run the full indexing pipeline on the new `NormalizedDocument`
4. This operation is atomic — retrieval must not hit a half-replaced document mid-operation

When `content_hash` matches: no-op. Indexing returns immediately.

---

## Signals to Retrieval Layer

Indexing provides the following fields that retrieval should use for scoring, filtering, and surface selection. These are documented here as a contract — retrieval should not need to query any layer other than the `documents` and `chunks` tables to make all decisions.

| Field | Table | Purpose |
|---|---|---|
| `depth` | documents | Depth-based scoring; seed pages (depth=0) warrant higher base trust |
| `fetched_at` | documents | Recency signal; more recently fetched pages may be more current |
| `description` | documents | Page-level summary for traversal diagram labels; link scoring primary signal (URL path is fallback) |
| `has_table`, `has_code`, `has_math`, `has_definition_list`, `has_admonition`, `has_steps` | chunks | Surface selection and rendering hints; retrieval prefers `html_text` for `has_table`, `has_code`, `has_math`, `has_definition_list`, `has_admonition` |
| `chunk_index` | chunks | Reading order; adjacent children under the same parent can be merged before passing to LLM |
| `section_heading` | chunks | Coarse filtering and citation labeling; null for fallback-chunked content |
| `source_url` | chunks | Denormalized for fast citation lookup without join |

---

## End-to-End Response Flow

This section describes how all layers combine to produce the final user-facing response, clarifying what indexing must support at each step. The retrieval layer makes the case determination dynamically based on corpus size and coverage confidence — it is not determined from the initial prompt.

### Case 1: Full-Context Answer (small corpus)

The corpus fits within the token budget threshold and retrieval confidence is high.

1. **Ingestion** fetches the URL, returns `NormalizedDocument` with markdown, HTML, links, metadata.
2. **Indexing** chunks the document, detects and sets rich content flags, stores parent/child records with char and token offsets, embeds children, writes document record.
3. **Retrieval** (full-context mode): passes complete page markdown to the LLM, substituting `html_text` for flagged chunks. LLM generates a natural language answer with LaTeX rendered inline, code blocks preserved, tables from HTML where applicable.
4. **Citation reconstruction**: LLM's quoted spans are matched against stored chunk text via `char_start`/`char_end` — deterministic string lookup, no LLM involved. If a verbatim quote already appears in the answer prose, the citations block supplies `source_url` and `section_heading` as attribution only without re-quoting.
5. **Response to user**: natural answer + citations block at bottom.

### Case 2: Multi-Page Traversal Answer (large or expanding corpus)

Coverage confidence is insufficient on the existing corpus. Orchestration drives expansion.

1. **Orchestration** runs the expansion loop: seed URL ingested, retrieval finds insufficient coverage, candidate links scored — using `description` as primary signal, URL path pattern as fallback — selected, batch ingested, re-indexed, retrieval re-run. Orchestration records which URLs were visited and which yielded evidence.
2. **Indexing** runs on each batch of new pages, writing document records (with `depth` supplied by orchestration) and chunk records.
3. **Retrieval** (chunk mode, corpus now large): vector search over child embeddings finds relevant passages, returns parent chunks as LLM context.
4. **Citation reconstruction**: same deterministic offset lookup as Case 1, now across multiple `source_url` values.
5. **Traversal diagram**: orchestration emits a compact horizontal ASCII diagram using `source_url` and `title` from document records. The base domain is shown once; subsequent nodes show only the differing URL path segments. `[✓]` marks pages that yielded evidence; `[–]` marks visited but non-contributory pages. Example:

```
docs.example.com/intro → /ensemble-methods [✓] → /random-forests [✓]
                                            → /getting-started  [–]
```

The diagram is assembled by the orchestration/MCP layer. Indexing supplies the necessary fields — `title`, `source_url`, `depth`, `description` — on every document record.I want the changing part of the URL path, not the full string. So docs.example.com/api/ensemble → /random-forests [✓] → /decision-trees [✓] where the base domain is shown once and subsequent nodes show only the differing path segments. This is an orchestration/MCP layer detail but good to note. And for link scoring / traversal decisions: description first, URL path pattern second as fallback. Indexing just stores both cleanly — link scoring is orchestration's problem.

---

## Token Ceilings
The benchmark consensus for child chunks in a parent-child setup targeting technical documentation:
Child chunks: 256–512 tokens is the sweet spot. NVIDIA benchmarks show factoid/precise queries peak at 256–512; Chroma research shows 400 tokens with RecursiveCharacterTextSplitter hits 88–89% recall without semantic chunking overhead. For web docs where queries are often precise ("what parameter controls X"), 256 is the right default.
Parent chunks: 3–5x child size, so 768–1500 tokens. 1000 tokens is a defensible default — it covers most ## sections comfortably without becoming unwieldy.
Oversized section split ceiling: 1000 tokens for parents, which means any ## section exceeding that gets split at paragraph boundaries into sub-chunks.

## Image Handling
Store the URL and alt text inline in markdown - retrieval/MCP layer is sound: if an image URL appears within a retrieved chunk that's being passed to the LLM, and the alt text is meaningful or the image is contextually embedded in relevant content, download and pass it alongside. Otherwise fall back to the caption/alt text as a text reference. This is a retrieval-layer concern — indexing just preserves Show Image as-is in chunk text. Nothing changes in indexing.

## Open Questions
- Whether `description` from HTML meta tags is reliably populated across real-world pages, or whether URL path will need to serve as the primary traversal diagram label more often than expected — will become clear during integration testing
- Whether `has_steps` warrants storing `html_text` (currently flagged markdown-sufficient) — some documentation frameworks render ordered lists with semantic styling that markdown loses; revisit after seeing real Firecrawl output on procedural pages
