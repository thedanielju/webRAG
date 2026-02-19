
# brief outline, translated into clearer decisions

- firecrawl ingestion should be light; heavy only for JS rendering and headless browsers, concurrency, etc
- mostly CPU/RAM/network heavy

- chunking:
a. deterministic
- headings / paragraphs / token length
- cheap, CPU-light

b. LLM-based
- scraper outputs weird fragmentation
- can deal with conceptual blocks
- tables, data, etc
- can get expensive

- embeddings: converts text into vectors for semantic search
- GPU matters if we embed a lot of texts
- hosted; easiest + predictable, external dependency

reranking:
- CPU-feasible at small top-K (20-50 candidates), GPU helps a lot
- worth it when corpus is large / heterogenous
- higher precision for citations, similar pages
- skip for: small corpus, clear headings, speed / minimal infra

A. i think we should "route" depending on the use case
B. does firecrawl "chunk"?