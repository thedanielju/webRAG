# exception contract not specified yet as im the only caller; above layers might necessitate this
# client handles only API access and exposes scrape and map

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()

# api key management
_FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not _FIRECRAWL_API_KEY:
    raise RuntimeError(
        "Missing FIRECRAWL_API_KEY - please set it in your environment or .env"
    )

app = FirecrawlApp(api_key=_FIRECRAWL_API_KEY)

DEFAULT_SCRAPE_OPTIONS: dict[str, Any] = {
    "formats": [{"type": "markdown"}, {"type": "html"}, {"type": "links"}],
    "only_main_content": True,
    "parsers": [{"type": "pdf"}],
    "timeout": 30000, # 30s
    "block_ads": True,
    "remove_base64_images": True,
    "proxy": "auto",
}

# URL string input, optional dict of options to override defaults
# options parameters exists so callers can later override specific defaults
# very unlikely service will override anything

def scrape(url: str, options: dict[str, Any] | None = None) -> dict[str, Any]: # no options defaults to None
    merged_options = {**DEFAULT_SCRAPE_OPTIONS, **(options or {})}
    return app.scrape(url, **merged_options)


def map(url: str, limit: int = 100) -> list[Any]:
    response = app.map(url, limit=limit)  # cap candidate link volume for orchestration

    if isinstance(response, list):
        return response

    if isinstance(response, dict):
        links = response.get("links")
        if isinstance(links, list):
            return links

    if hasattr(response, "links"):
        return response.links

    raise ValueError("Unexpected response shape from Firecrawl map endpoint.")
