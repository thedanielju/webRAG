Vector search — query child chunk embeddings, get back parent chunk text as context. The schema supports this: children have embeddings, parent_id links to the full section.
Citation reconstruction — given a chunk, get source_url, char_start, char_end, section_heading. All denormalized on the chunk, so one row lookup.
Surface selection — prefer html_text over chunk_text when has_table, has_code, has_definition_list, or has_admonition are true. Flags are on the chunk row, so retrieval can do this inline.
Depth-based scoring — depth is on the documents table, not denormalized to chunks. Retrieval needs a join to score by depth. Worth denormalizing depth onto chunks the same way source_url and fetched_at are, so retrieval never has to join documents at query time.
Traversal diagram — needs title and source_url per document, plus which chunks contributed to the answer. title is only on documents, not chunks. Same argument for denormalizing it.

natural answer + traversal diagram + verbatim citations block. Retrieval can handle "double citation" by only surfacing the citations block without re-quoting what already appears verbatim in the answer.
