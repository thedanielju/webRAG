from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Ingestion / Firecrawl
    firecrawl_api_key: str | None = Field(default=None, validation_alias="FIRECRAWL_API_KEY")
    firecrawl_default_scrape_options: dict[str, Any] = Field(
        default_factory=lambda: {
            "formats": [{"type": "markdown"}, {"type": "html"}, {"type": "links"}],
            "only_main_content": True,
            "parsers": [{"type": "pdf"}],
            "timeout": 30000,
            "block_ads": True,
            "remove_base64_images": True,
            "proxy": "auto",
        }
    )
    firecrawl_map_default_limit: int = Field(
        default=100, validation_alias="FIRECRAWL_MAP_DEFAULT_LIMIT"
    )
    ingest_discover_links_default_limit: int = Field(
        default=500, validation_alias="INGEST_DISCOVER_LINKS_DEFAULT_LIMIT"
    )

    # Indexing / Embeddings
    database_url: str = Field(default="", validation_alias="DATABASE_URL")
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="EMBEDDING_BASE_URL"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL"
    )
    embedding_api_key: str | None = Field(default=None, validation_alias="EMBEDDING_API_KEY")
    embedding_dimensions: int = Field(default=1536, validation_alias="EMBEDDING_DIMENSIONS")
    embedding_tokenizer_kind: str = Field(
        default="tiktoken", validation_alias="EMBEDDING_TOKENIZER_KIND"
    )
    embedding_tokenizer_name: str = Field(
        default="cl100k_base", validation_alias="EMBEDDING_TOKENIZER_NAME"
    )
    # Concurrent embedding batch settings.
    # Texts are split into batches of this size and sent to the embedding API
    # in parallel using up to EMBEDDING_MAX_WORKERS threads.
    embedding_batch_size: int = Field(
        default=256, validation_alias="EMBEDDING_BATCH_SIZE"
    )
    # Max concurrent threads for embedding API calls.  Default 4 is conservative
    # to avoid 429 rate-limit errors on lower OpenAI API tiers.  Paid plans with
    # higher RPM allowances can safely increase this to 8-12.  Local embedding
    # servers (Ollama, LM Studio) are not rate-limited and can use higher values,
    # though returns diminish past the number of physical CPU/GPU cores.
    embedding_max_workers: int = Field(
        default=4, validation_alias="EMBEDDING_MAX_WORKERS"
    )

    child_target_tokens: int = Field(default=256, validation_alias="CHILD_TARGET_TOKENS")
    parent_max_tokens: int = Field(default=1000, validation_alias="PARENT_MAX_TOKENS")

    # Retrieval
    # Total corpus tokens below this threshold switch retrieval to full-context mode.
    # Must be <= retrieval_context_budget; retrieval clamps at runtime if misconfigured.
    retrieval_full_context_threshold: int = Field(
        default=30000, validation_alias="RETRIEVAL_FULL_CONTEXT_THRESHOLD"
    )
    # Hard cap on total context tokens retrieval is allowed to return.
    # Must be >= retrieval_full_context_threshold.
    retrieval_context_budget: int = Field(
        default=40000, validation_alias="RETRIEVAL_CONTEXT_BUDGET"
    )
    # Ceiling on child hits pulled from HNSW before parent aggregation.
    retrieval_top_k_children_limit: int = Field(
        default=60, validation_alias="RETRIEVAL_TOP_K_CHILDREN_LIMIT"
    )
    # Minimum cosine similarity required for a child hit to survive filtering.
    retrieval_similarity_floor: float = Field(
        default=0.3, validation_alias="RETRIEVAL_SIMILARITY_FLOOR"
    )
    # Linear penalty per crawl depth level when ranking parent candidates.
    retrieval_depth_decay_rate: float = Field(
        default=0.05, validation_alias="RETRIEVAL_DEPTH_DECAY_RATE"
    )
    # Minimum multiplier for depth penalty so deep pages are not zeroed out.
    retrieval_depth_floor: float = Field(
        default=0.80, validation_alias="RETRIEVAL_DEPTH_FLOOR"
    )
    # pgvector HNSW recall knob (higher = better recall, slightly slower).
    retrieval_hnsw_ef_search: int = Field(
        default=100, validation_alias="RETRIEVAL_HNSW_EF_SEARCH"
    )

    # ── Orchestration: Query Analysis ─────────────────────────────
    orchestration_llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="ORCHESTRATION_LLM_BASE_URL",
    )
    orchestration_llm_api_key: str | None = Field(
        default=None, validation_alias="ORCHESTRATION_LLM_API_KEY"
    )
    orchestration_llm_model: str = Field(
        default="gpt-4o-mini", validation_alias="ORCHESTRATION_LLM_MODEL"
    )
    decomposition_mode: str = Field(
        default="llm", validation_alias="DECOMPOSITION_MODE"
    )

    # ── Orchestration: Reranking ──────────────────────────────────
    reranker_provider: str = Field(
        default="zeroentropy", validation_alias="RERANKER_PROVIDER"
    )
    reranker_api_key: str | None = Field(
        default=None, validation_alias="RERANKER_API_KEY"
    )
    reranker_model: str = Field(
        default="zerank-2", validation_alias="RERANKER_MODEL"
    )
    reranker_top_n: int = Field(
        default=20, validation_alias="RERANKER_TOP_N"
    )

    # ── Orchestration: Expansion ──────────────────────────────────
    max_expansion_depth: int = Field(
        default=5, validation_alias="MAX_EXPANSION_DEPTH"
    )
    max_candidates_per_iteration: int = Field(
        default=5, validation_alias="MAX_CANDIDATES_PER_ITERATION"
    )
    candidates_to_score_per_iteration: int = Field(
        default=20, validation_alias="CANDIDATES_TO_SCORE_PER_ITERATION"
    )
    expansion_map_limit: int = Field(
        default=100, validation_alias="EXPANSION_MAP_LIMIT"
    )

    # ── Orchestration: Locality Expansion ─────────────────────────
    locality_expansion_enabled: bool = Field(
        default=True, validation_alias="LOCALITY_EXPANSION_ENABLED"
    )
    locality_expansion_radius: int = Field(
        default=1, validation_alias="LOCALITY_EXPANSION_RADIUS"
    )

    # ── Orchestration: Stopping Criteria ──────────────────────────
    token_budget_saturation_ratio: float = Field(
        default=0.8, validation_alias="TOKEN_BUDGET_SATURATION_RATIO"
    )
    redundancy_ceiling: float = Field(
        default=0.85, validation_alias="REDUNDANCY_CEILING"
    )
    score_cliff_threshold: float = Field(
        default=0.15, validation_alias="SCORE_CLIFF_THRESHOLD"
    )
    score_cliff_rank_k: int = Field(
        default=5, validation_alias="SCORE_CLIFF_RANK_K"
    )
    plateau_variance_threshold: float = Field(
        default=0.02, validation_alias="PLATEAU_VARIANCE_THRESHOLD"
    )
    plateau_top_n: int = Field(
        default=10, validation_alias="PLATEAU_TOP_N"
    )
    diminishing_return_delta: float = Field(
        default=0.03, validation_alias="DIMINISHING_RETURN_DELTA"
    )
    mediocre_score_floor: float = Field(
        default=0.5, validation_alias="MEDIOCRE_SCORE_FLOOR"
    )
    confidence_floor: float = Field(
        default=0.3, validation_alias="CONFIDENCE_FLOOR"
    )


settings = Settings()
