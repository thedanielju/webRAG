
ingest(root_url)           # single page, get NormalizedDocument
discover_links(root_url)   # get candidate LinkCandidates with title/description
                           # orchestration scores them
ingest_batch(top_k_urls)   # ingest best candidates as a batch
discover_links(each)       # find next layer of candidates
                           # repeat

One hierarchical level per iteration lets retrieval quality inform whether to go deeper before committing more API credits. Blindly ingesting two levels deep upfront often pulls in irrelevant pages.

ingest_batch(urls: list[str]) -> list[NormalizedDocument]
- firecrawl batch_scrape endpoint (one API call)
- returns NormalizedDocument for each URL
- failed URLs return None or are skipped with a warning (don't let one failure kill the batch)

ingest(url)              -> NormalizedDocument
ingest_batch(urls)       -> list[NormalizedDocument | None]

user controlled depth:
depth=0  → ingest root only
depth=1  → ingest root + one batch expansion (default)
depth=auto → orchestration decides based on retrieval quality

Ingestion — gives you LinkCandidate objects with url, title, description
Orchestration — scores them using those fields plus query context and corpus state, picks top-K, decides whether to expand
Indexing — never sees LinkCandidate at all, only receives NormalizedDocument
Retrieval — operates purely over indexed chunks, has no awareness of the expansion loop