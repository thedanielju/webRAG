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

## general design plan:
building is slice-based - one thin end to end path first, then deepen layers

1. ingestion returning clean text + links
2. minimal indexing that stores chunks somewhere (i'd rather do the database before this, but i guess embeddings are paid)
3. minimal retrieval to return relevant chunks
4. minimal citations (stored chunk quotations)
5. orchestrator loop - expand if insufficient
6. harden each layer (rate limits, dedupe, rerank, etc)