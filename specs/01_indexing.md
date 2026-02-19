## Indexing
turns documents into retrievable chunks
1. parse sections/heading if possible
2. chunk
3. embed
4. store in vector DB + metadata store
- does not fetch web pages
- does not decide which new pages to ingest
- does not loop / stop