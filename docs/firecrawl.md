Firecrawl = "eyes"; gives models a look at webpages

WebRAG = "memory + reasoning over knowledge"
- accumulate knowledge over time; asking another question tommorow = reference
- retrieval policies
- selective recursion
- maintain citations
- deterministic expansion loops

- adds selective recursion, link scoring, diminishing returns logic, corpus growth strategy, citation reconstruction
- policy and memory layer above firecrawl

Firecrawl does not chunk, embed, vector index, or retrieve semantically - each query requires re-fetching and raw content scanning

## firecrawl capabilities
Tool	Best for	Returns
scrape	Single page content	JSON (preferred) or markdown
batch_scrape	Multiple known URLs	JSON (preferred) or markdown[]
map	Discovering URLs on a site	URL[]
crawl	Multi-page extraction (with limits)	markdown/html[]
search	Web search for info	results[]
agent	Complex multi-source research	JSON (structured data)
browser	Interactive multi-step automation	Session with live browser

## Firestarter: 
https://github.com/firecrawl/firestarter
similar in using firecrawl:
- to crawl
- upstash to chunk, embed, vector search
- streaming chat, openAI endpoint

differences:
- WebRAG is concerned with being a general purpose MCP tool surface; not an app
- "chat with a website" vs "reusable ingestion/retrieval service that agents can call with consistent contracts, logging, storage, and policy"


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


## ingestion specs
1. endpoints
