# Polite-crawler controls applied at the service layer, before any fetch
# on EITHER the Firecrawl or raw path:
#
#   1. robots.txt — fetched once per origin, cached, parsed with the
#      stdlib urllib.robotparser.  is_allowed() returns False for
#      disallowed paths; the service skips those URLs (logged, surfaced).
#   2. per-origin rate limit — an async token-spacing limiter that
#      guarantees at least 1/CRAWL_RATE_LIMIT_RPS seconds between fetches
#      to the same origin, regardless of concurrency.
#
# Both are injectable (the time/sleep functions and the robots fetcher)
# so tests can assert spacing and parsing with no real network or sleeps.

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable
from urllib import robotparser
from urllib.parse import urlparse, urlunparse

import httpx

from config import settings

logger = logging.getLogger(__name__)

# Same UA the raw fetcher sends; robots rules are matched against it.
_USER_AGENT = "WebRAG/1.0 (+https://github.com/; raw-fetch)"

# Type of the injectable robots.txt body fetcher: origin -> robots text
# (or None if it could not be fetched, which is treated as "allow all").
RobotsFetcher = Callable[[str], Awaitable[str | None]]


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def _robots_url(origin: str) -> str:
    return origin.rstrip("/") + "/robots.txt"


async def _default_robots_fetcher(origin: str) -> str | None:
    """Fetch <origin>/robots.txt over HTTP.  Returns None on any failure,
    which the caller interprets as 'no robots file -> allow all'."""
    try:
        async with httpx.AsyncClient(
            timeout=settings.raw_fetch_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(_robots_url(origin))
    except httpx.HTTPError:
        return None
    # 4xx/5xx (commonly 404) means no usable rules — allow all.
    if response.status_code >= 400:
        return None
    return response.text


class PolitenessGate:
    """Per-origin robots.txt cache + rate limiter.

    One instance guards a whole crawl run.  ``is_allowed`` answers the
    robots question (cached per origin); ``wait_for_slot`` blocks until
    the origin's rate-limit window opens.  The service layer calls
    ``check_and_wait`` which combines both: returns False (skip) when
    robots disallows, otherwise waits out the rate limit and returns True.
    """

    def __init__(
        self,
        *,
        respect_robots: bool | None = None,
        rate_limit_rps: float | None = None,
        robots_fetcher: RobotsFetcher | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._respect_robots = (
            settings.respect_robots_txt if respect_robots is None else respect_robots
        )
        rps = settings.crawl_rate_limit_rps if rate_limit_rps is None else rate_limit_rps
        # Minimum seconds between consecutive fetches to one origin.
        # rps <= 0 disables spacing entirely.
        self._min_interval = (1.0 / rps) if rps and rps > 0 else 0.0
        self._robots_fetcher = robots_fetcher or _default_robots_fetcher
        self._monotonic = monotonic
        self._sleep = sleep

        # origin -> parsed RobotFileParser (or None when robots is allow-all).
        self._robots_cache: dict[str, robotparser.RobotFileParser | None] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}
        # origin -> monotonic timestamp the next fetch is permitted at.
        self._next_allowed_at: dict[str, float] = {}
        self._rate_locks: dict[str, asyncio.Lock] = {}

    # ── robots.txt ────────────────────────────────────────────────
    async def _get_parser(self, origin: str) -> robotparser.RobotFileParser | None:
        if origin in self._robots_cache:
            return self._robots_cache[origin]

        lock = self._robots_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            # Re-check inside the lock — another coroutine may have filled it.
            if origin in self._robots_cache:
                return self._robots_cache[origin]

            body = await self._robots_fetcher(origin)
            if body is None:
                parser = None  # allow-all sentinel
            else:
                parser = robotparser.RobotFileParser()
                parser.parse(body.splitlines())
            self._robots_cache[origin] = parser
            return parser

    async def is_allowed(self, url: str) -> bool:
        """True if robots.txt permits fetching *url* (or robots is off)."""
        if not self._respect_robots:
            return True
        parser = await self._get_parser(_origin(url))
        if parser is None:
            return True
        return parser.can_fetch(_USER_AGENT, url)

    # ── rate limiting ─────────────────────────────────────────────
    async def wait_for_slot(self, url: str) -> None:
        """Block until the origin's rate-limit window allows a fetch."""
        if self._min_interval <= 0:
            return
        origin = _origin(url)
        lock = self._rate_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            now = self._monotonic()
            next_at = self._next_allowed_at.get(origin, 0.0)
            wait = next_at - now
            if wait > 0:
                await self._sleep(wait)
                now = next_at
            # Reserve the next slot one interval out from this fetch.
            self._next_allowed_at[origin] = now + self._min_interval

    # ── combined gate used by the service layer ───────────────────
    async def check_and_wait(self, url: str) -> bool:
        """Robots + rate-limit gate for one URL.

        Returns False when robots disallows the URL (caller skips it, no
        wait incurred).  Otherwise waits out the rate limit and returns
        True.  Disallowed URLs are logged so a run can surface the skip.
        """
        if not await self.is_allowed(url):
            logger.info("politeness: robots.txt disallows %s — skipping", url)
            return False
        await self.wait_for_slot(url)
        return True
