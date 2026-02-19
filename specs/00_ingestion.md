## SCOPE
ingestion layer is concerned with just the ingestion step, which will be recursed if necessary as determined by the orchestration layer (so not handled here)
returns a URL into a NormalizedDocument, handles link discovery
no recursion, no link selection, no persistence

## SCRIPTS

# firecrawl_client.py: 
Firecrawl API wrapper, config, auth
1. load API key from env
2. Default request config
3. expose scrape(url, options), map(url)

default scrape formats: ["markdown", "html", "links"]
onlyMainContent: true  (strips nav/footer noise)
parsers: [{type: "pdf"}]  (PDF → markdown by default)
blockAds: true
removeBase64Images: true  (base64 images are huge and useless)
proxy: "auto"

will NOT request: rawHtml, screenshot, json (LLM extraction), 
summary (LLM-based), images (format) - image URLs appear inline in markdown

# service.py:
ingestion logic, normalization, output schema:

ingest(url):
1. detect if PDF or url, detection logic is trival (checking if URL ends with .pdf / response ContentType is application/pdf)
2. call firecrawl_client.scrape() with appropriate options
3. normalize response -> normalizedDocument
- filtering, transformations and adding my own fields
- thin transform: enforce schema, add fetched_at + content_hash + doc_type, strip Firecrawl envelope
4. return normalizedDocument (no persistence here)

discover_links(url):
1. call firecrawl_client.map(url)
2. return raw candidate link list
- sitemaps endpoint returns url objects with options for
- title, description (which are necessary for link scoring later)

## OUTPUTS

From Firecrawl (pass-through):
- url
- markdown
- html  
- links (list of outgoing URLs)
- title, language, statusCode (from metadata)

Added by ingestion:
- fetched_at (timestamp)
- content_hash (md5/sha256 of markdown, for deduplication)
- doc_type ("html" | "pdf")

Excluded from Firecrawl response (noise):
- rawHtml, screenshot, ogImage, ogTitle, ogDescription

LinkCandidate:
- url (required)
- title (optional)
- description (optional)

Persistence is handled by indexing layer.
saving is managed by indexing, ingestion just returns a NormalizedDocument in memory

firecrawl also supports returning images; clients don't render images from tool responses
- store image URLS and alt text as part of normalized metadata; pass as markdown links in retrieval

firecrawl also supports returning HTML for table and code blocks
- firecrawl can return formats: ["markdown", "html"]
- always request markdown and HTML - can't know if a page is table heavy until it's fetched

## ERROR HANDLING
- per endpoint (scrape, map)
- HTTP errors, Firecrawl API errors, malformed responses

## WHY NOT CRAWL?
crawl is better for simple cases, handling concurrency, deduplication, depth management. it's genuinely good; however, i decided to implement my own
version because crawl's stopping logic is structural (page count and depth). the concept of retrieval quality, diminishing returns, and semantic relevance
are stronger constraints. no way to force crawl to stop when these criteria are met; ingests everything it discovers breadth-first. link scoring and selective
expansion are the core differentiator of WebRAG as research memory rather than a bulk scraper.

may borrow includePaths, excludePaths regex filtering, and allowExternalLinks.
good ideas for me to implement in orchestration's link scoring; no need to reinvent them from scratch

## Markdown vs JSON
Markdown for primary content - JSON (type: json) in Firecrawl is LLM-based structured extraction; provide a schema and a prompt, and Firecrawl uses an LLM to extract structured data from the page. not what I want at ingestion time; it's expensive, lossy, and schema-dependent. we want raw content that the retrieval layer can search over freely.