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