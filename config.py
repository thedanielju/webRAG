from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Load .env once at startup so all modules read a consistent environment.
load_dotenv()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing {name} - please set it in your environment or .env")


# Firecrawl API authentication.
FIRECRAWL_API_KEY = get_required_env("FIRECRAWL_API_KEY")

# Firecrawl scrape defaults used across single and batch ingestion.
FIRECRAWL_DEFAULT_SCRAPE_OPTIONS: dict[str, Any] = {
    "formats": [{"type": "markdown"}, {"type": "html"}, {"type": "links"}],
    "only_main_content": True,
    "parsers": [{"type": "pdf"}],
    "timeout": 30000,
    "block_ads": True,
    "remove_base64_images": True,
    "proxy": "auto",
}

# Sensible discovery defaults for orchestration.
FIRECRAWL_MAP_DEFAULT_LIMIT = 100
INGEST_DISCOVER_LINKS_DEFAULT_LIMIT = 500
