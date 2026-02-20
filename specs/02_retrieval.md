Vector search — query child chunk embeddings, get back parent chunk text as context. The schema supports this: children have embeddings, parent_id links to the full section.
Citation reconstruction — given a chunk, get source_url, char_start, char_end, section_heading. All denormalized on the chunk, so one row lookup.
Surface selection — prefer html_text over chunk_text when has_table, has_code, has_math, has_definition_list, or has_admonition are true. Flags are on the chunk row, so retrieval can do this inline.
Depth-based scoring — depth is on the documents table, not denormalized to chunks. Retrieval needs a join to score by depth. Worth denormalizing depth onto chunks the same way source_url and fetched_at are, so retrieval never has to join documents at query time.
Traversal diagram — needs title and source_url per document, plus which chunks contributed to the answer. title is only on documents, not chunks. Same argument for denormalizing it.

natural answer + traversal diagram + verbatim citations block. Retrieval can handle "double citation" by only surfacing the citations block without re-quoting what already appears verbatim in the answer.

---

## Retrieval-Relevant Signals from Indexing

Signals the retrieval layer should use, all available on the `chunks` and `documents` tables without external calls.

### Surface Selection

When presenting a chunk to the LLM, retrieval chooses between `chunk_text` (markdown) and `html_text` (raw HTML snippet):

| Flag | Prefer html_text? | Reason |
|---|---|---|
| `has_table` | Yes | Markdown loses alignment, merged cells, complex cell content |
| `has_code` | Yes | Rendering fidelity for syntax highlighting, indentation |
| `has_math` | Yes | Firecrawl strips all LaTeX from markdown; MathML in HTML is the only representation |
| `has_definition_list` | Yes | Firecrawl doesn't produce markdown definition-list syntax; HTML `<dl>` preserves structure |
| `has_admonition` | Yes | HTML preserves intent via `<div class="admonition ...">` with CSS classes |
| `has_steps` | No | Markdown ordered lists are sufficient; flag signals retrieval for UX treatment only |

When multiple flags are true, `html_text` is always preferred (it covers all flagged content).

**html_text deduplication:** Multiple child chunks under the same parent section often share the same HTML context element, producing identical `html_text` values. Retrieval must deduplicate before sending to the LLM — e.g. when returning multiple sibling children from the same section, send the shared `html_text` once rather than repeating it per chunk.

### Image Handling at Retrieval Time

Indexing preserves `![alt text](url)` as-is in `chunk_text`. No `has_image` flag exists — images don't need HTML fallback since Firecrawl preserves them faithfully in markdown with absolute URLs.

At response time, retrieval/MCP layer decides per-image:
- If the image URL appears in a retrieved chunk being sent to the LLM and alt text is meaningful or the image is contextually relevant → fetch and pass alongside as a vision input
- Otherwise → pass alt text + URL as a text reference

Alt text quality varies by source (Wikipedia: many empty; scikit-learn: filename-based). Retrieval should not assume alt text is always informative.

### Scoring Signals

| Signal | Source | Usage |
|---|---|---|
| `depth` | documents | Seed pages (depth=0) get higher base trust; diminishing weight at higher depths |
| `fetched_at` | documents (denormalized to chunks) | Recency signal; more recently fetched pages may be more current |
| `section_heading` | chunks | Coarse filtering; citation labeling; null for fallback-chunked content |
| `chunk_index` | chunks | Reading order; adjacent children under the same parent can be merged before passing to LLM |
| `description` | documents | Traversal diagram node labels; link scoring primary signal |

