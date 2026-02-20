## workspace execution tracking

- decisions contains architectural, contractual design choices for reference
- specs contains exact design specifications

1. set up firecrawl
2. design ingestion

## long term thoughts:
1. implement corpus identity:

corpus_id
version
root_url
crawl_timestamp

2. link scoring is going to be the biggest subproblem!

3. citation offsets are super high leverage - prevents hallucinated quotes

4. +1 hierarchy is very good - constraint prevents crawl explosion(?)

5. Most relevant beyond PDFs:
- Pages that trigger file downloads (Content-Disposition: attachment) but are not PDFs.
- Office docs (.docx, .pptx, .xlsx) and similar binaries.
- Plain text/CSV/XML endpoints that aren’t normal HTML pages.
- Media-heavy URLs (images/video/audio) that may yield sparse or no useful markdown.

6. No image content is processed at indexing time. A vision model pass is a future enhancement.

7. Re-add LaTeX markdown regex (`$$`/`$`) to `_detect_markdown_flags` if a non-Firecrawl ingestion path is added (e.g. raw HTML fetch, local file import). Currently hardcoded to False because Firecrawl strips all LaTeX from markdown.

## general design plan:
building is slice-based - one thin end to end path first, then deepen layers

1. ingestion returning clean text + links
2. minimal indexing that stores chunks somewhere (i'd rather do the database before this, but i guess embeddings are paid)
3. minimal retrieval to return relevant chunks
4. minimal citations (stored chunk quotations)
5. orchestrator loop - expand if insufficient
6. harden each layer (rate limits, dedupe, rerank, etc)