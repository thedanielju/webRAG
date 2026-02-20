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


settings = Settings()
