## ingest(url: str) -> NormalizedDocument
## ingest_batch(urls: list[str]) -> list[NormalizedDocument]  
future: swap internals to batch_scrape firecrawl endpoint, ingest should accept a list of URLs, not just one - orchestration has a ingest top candidates step to pass a batch
current: sequential scrape calls

Configuration note:
- API key and Firecrawl defaults are not handled in service.py; they are centralized in `config.py` and consumed by `firecrawl_client.py`.

1. Detect doc_type: check URL suffix, default to html
- we can't check content-type before scrape; and already handled after scrape
2. Call firecrawl_client.scrape(url)
3. Build NormalizedDocument:
   - url: from input
   - title, language, status_code: from result.metadata
   - markdown, html: from result directly
   - links: from result.links (default to [] if None)
   - fetched_at: datetime.utcnow()
   - content_hash: sha256 of markdown (or empty string if None)
        a. Hash the markdown field specifically, not the full response. markdown is canonical content; hashing it lets orchestration skip re-ingesting pages whose content hasn't changed.
        b. orchestration will compare new content_hash against stored one; match skips re-index; differ updated content
   - doc_type: from step 1
4. Return NormalizedDocument
5. we should think ahead and make this usable for the orchestration layer recursion

## discover_links(url: str, limit: int = 500, exclude: set[str] = None) -> list[LinkCandidate]
orchestration tracks visited URLs to avoid re-ingesting. rather than making orchestration filter after the fact, pass a seen set into discover_links():
Filter out any LinkCandidate whose URL is in exclude before returning. Keeps deduplication logic close to where links are discovered.

Default `limit` should come from `config.py` so operators can tune discovery volume without touching service logic.

1. Call firecrawl_client.map(url, limit=limit)
    - accept limit and pass it through to firecrawl_client.map() so orchestration can control volume
2. Map each LinkResult to LinkCandidate:
   - url: required
   - title: pass through, may be None
   - description: pass through, may be None
   
3. Return list[LinkCandidate]

## DATA CLASSES
defining these explicitly means indexing and orchestration layers get typed objects, not raw dicts they have to guess the shape of

@dataclass
class NormalizedDocument:
    url:            input url
    source_url:     result.metadata.source_url # canonical post-redirect URL; used by indexing for citations
    title:          result.metadata.title
    description:    result.metadata.description # indexing can use as summary field for chunk metadata without re-processing full markdown
    language:       result.metadata.language
    status_code:    result.metadata.status_code
    published_time: result.metadata.published_time
    modified_time:  result.metadata.modified_time
    markdown:       result.markdown
    html:           result.html
    links:          result.links or []
    fetched_at:     datetime.utcnow()
    content_hash:   sha256(markdown) or ""
    doc_type:       "pdf" if url.endswith(".pdf") else "html"

@dataclass
class LinkCandidate:
    url: str
    title: str | None
    description: str | None

## TEST IMPLEMENTATION
PDF test — noted, add a PDF URL test case to scratch tests when we get to service.py testing. A good public PDF to test against: https://arxiv.org/pdf/1706.03762 (the Attention Is All You Need paper).
