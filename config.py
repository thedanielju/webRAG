from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file's directory (the project root),
# not the working directory.  This ensures the .env is found even when
# the process is launched with a different cwd (e.g. Claude Desktop).
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), case_sensitive=False)

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
    # Chunks with raw_similarity below this floor are excluded before
    # reranking.  These are near-noise that passed retrieval's broader
    # threshold but aren't worth reranker tokens.  Set to 0.0 to disable.
    reranker_similarity_floor: float = Field(
        default=0.15, validation_alias="RERANKER_SIMILARITY_FLOOR"
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
    # Skip expensive expansion ingestion when the best scored link candidate is
    # weaker than this threshold. Helps avoid low-yield expansion rounds.
    expansion_min_candidate_score: float = Field(
        default=0.12, validation_alias="EXPANSION_MIN_CANDIDATE_SCORE"
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
        default=0.25, validation_alias="MEDIOCRE_SCORE_FLOOR"
    )
    confidence_floor: float = Field(
        default=0.3, validation_alias="CONFIDENCE_FLOOR"
    )
    min_score_for_expansion: float = Field(
        default=0.3, validation_alias="MIN_SCORE_FOR_EXPANSION"
    )

    # ── MCP Server ────────────────────────────────────────────
    # These settings control how the MCP server communicates with
    # reasoning models (Claude, GPT, etc.) via the Model Context Protocol.

    # Transport protocol: "stdio" for desktop clients (Claude Desktop,
    # Cursor) that launch the server as a subprocess, or "streamable-http"
    # for remote/hosted deployments accessible over the network.
    mcp_transport: str = Field(
        default="stdio", validation_alias="MCP_TRANSPORT"
    )
    # Network bind address for streamable-http transport (ignored for stdio).
    mcp_host: str = Field(
        default="0.0.0.0", validation_alias="MCP_HOST"
    )
    # Port for streamable-http transport (ignored for stdio).
    mcp_port: int = Field(
        default=8765, validation_alias="MCP_PORT"
    )
    # Optional bearer token for authenticating MCP clients (future use).
    mcp_auth_token: str | None = Field(
        default=None, validation_alias="MCP_AUTH_TOKEN"
    )
    # Soft ceiling on response size (in tokens).  The formatter fits as
    # many evidence chunks as possible within this budget, then truncates
    # with a note.  Higher = more context for the model, but slower and
    # more expensive for token-billed APIs.
    mcp_response_token_budget: int = Field(
        default=30000, validation_alias="MCP_RESPONSE_TOKEN_BUDGET"
    )
    # Default MCP behavior should favor responsiveness. "fast" uses chunked
    # retrieval and no expansion unless the caller explicitly asks for more.
    mcp_default_research_mode: str = Field(
        default="fast", validation_alias="MCP_DEFAULT_RESEARCH_MODE"
    )
    # MCP retrieval default. "chunk" avoids giant full-context payloads and
    # preserves room for citations/images in the formatted response.
    mcp_default_retrieval_mode: str = Field(
        default="chunk", validation_alias="MCP_DEFAULT_RETRIEVAL_MODE"
    )
    # When enabled, MCP responses append a recommendation block prompting the
    # model to ask the user before running a slower deep-expansion pass.
    mcp_enable_expansion_recommendations: bool = Field(
        default=True, validation_alias="MCP_ENABLE_EXPANSION_RECOMMENDATIONS"
    )
    # Reserve response tokens for citations so they survive large evidence blocks.
    mcp_citations_reserved_tokens: int = Field(
        default=1500, validation_alias="MCP_CITATIONS_RESERVED_TOKENS"
    )
    # Reserve response tokens for the [IMAGES] section.
    mcp_images_reserved_tokens: int = Field(
        default=500, validation_alias="MCP_IMAGES_RESERVED_TOKENS"
    )
    # Hard timeout (seconds) for the answer tool's orchestration run.
    # Prevents runaway expansion loops from blocking the MCP connection.
    mcp_tool_timeout: int = Field(
        default=120, validation_alias="MCP_TOOL_TIMEOUT"
    )
    # Logging verbosity for the MCP server process (DEBUG, INFO, WARNING).
    mcp_log_level: str = Field(
        default="INFO", validation_alias="MCP_LOG_LEVEL"
    )


settings = Settings()
