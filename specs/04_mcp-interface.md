utility from firecrawl:
AsyncFirecrawl

This is an MCP layer question but worth answering now since it affects how retrieval should package its output. The answer is system prompt engineering plus structured tool output. The MCP tool response includes: the retrieved chunks (or full pages) as labeled context blocks, explicit instruction to ground every claim in provided context, and a citation format instruction. Something like:

You have been provided the following web content. Answer using only 
this content. For every factual claim, cite the source chunk by URL 
and section. At the end, list verbatim evidence spans.

[CONTEXT]
source: example.com/docs/ensemble — section: Random Forests
"...chunk text..."

[CONTEXT]
source: example.com/docs/api — section: Parameters  
"...chunk text..."

The model follows this reliably when the context is well-structured and the instruction is explicit. Reranking (returning only the top-K most relevant chunks) is important here — if you flood the context with 40 marginally relevant chunks, answer quality degrades. Retrieval should return focused, high-confidence chunks, not everything it finds.
