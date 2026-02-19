
Firecrawl works as Page acquisition
1. returns data.markdown
2. data.links
3. data.metadata (title/sourceURL/statusCode/etc)

WebRAG needs to
1. normalization
2. dedupe (why)
3. chunking
4. embeddings
5. storage
6. retrieval + citations

## Firecrawl options:
1. onlyMainContent: true
2. formats: [{"type":"markdown"},{"type":"links"}]
3. removeBase64Images: true
4. blockAds: true
5. proxy: "auto"
    - for self-hosting, might not want proxies at all; decide later

store the following:
- source_url = metadata.sourceURL (not the request URL)
- fetched_at = your timestamp
- title = metadata.title
- status_code = metadata.statusCode
- markdown = data.markdown
- outgoing_links = data.links (if requested)
- content_hash = hash(markdown)

## Firecrawl Citations:
Firecrawl returns markdown as a string; treat Firecrawl markdown as immutable “page text artifact.”
Chunking produces char_start/char_end offsets into that exact markdown string

citations must include:
url
chunk_id
section (if tracked)
quote snippet = markdown[char_start:char_end] (or a sub-span inside chunk)

should give true verbatim citations without needing HTML offsets; reason to store HTML too: tables/code blocks that markdown mangles


## endpoints
scrape: single known URL -> clean markdown + metadata
map: discover URLs on a site fast before deciding what to ingest
batch scrape / crrawl / search 
