pgvector column you write embedding VECTOR(1536) — that number is the dimensionality of your embedding model's output and must match exactly. You set it once at schema creation time. If you change models, you drop and recreate the column and re-embed everything — there's no migration path.

The practical choices per model:

text-embedding-3-small (OpenAI default): 1536 dimensions. This is the standard. pgvector handles it fine, HNSW index works cleanly below the 2000-dimension limit.
text-embedding-3-large: 3072 dimensions. Exceeds pgvector's HNSW index limit of 2000 — you'd need half-precision vectors (halfvec) to make it work. Not worth the complexity.
nomic-embed-text (local): 768 dimensions. Smaller, faster, completely fine for WebRAG's scale.
bge-m3 (local, multilingual): 1024 dimensions. Good balance.

Recommendation: design the schema so the vector dimension is a config value read at setup time, not hardcoded in the migration. The default schema ships with 1536 (matching text-embedding-3-small). Users running local models set their model's dimension in config and run the schema setup once. The HNSW index type (faster approximate search, no training step needed, better recall tradeoff than IVFFlat) is the right default for WebRAG's scale — create it with vector_cosine_ops since cosine similarity is standard for text embeddings.

