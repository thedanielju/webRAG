from __future__ import annotations

import pytest

from src.ingestion.politeness import PolitenessGate


ROBOTS_BLOCK_PRIVATE = """User-agent: *
Disallow: /private/
Allow: /
"""

ROBOTS_BLOCK_ALL = """User-agent: *
Disallow: /
"""


def _fetcher(body: str | None):
    async def _fetch(_origin: str):
        return body

    return _fetch


# ── robots.txt parsing ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_robots_allows_and_denies_by_path():
    gate = PolitenessGate(
        respect_robots=True,
        rate_limit_rps=0,
        robots_fetcher=_fetcher(ROBOTS_BLOCK_PRIVATE),
    )

    assert await gate.is_allowed("https://site.example/public/page") is True
    assert await gate.is_allowed("https://site.example/private/secret") is False


@pytest.mark.asyncio
async def test_robots_missing_file_allows_all():
    # Fetcher returns None (e.g. 404) => allow everything.
    gate = PolitenessGate(
        respect_robots=True,
        rate_limit_rps=0,
        robots_fetcher=_fetcher(None),
    )
    assert await gate.is_allowed("https://site.example/anything") is True


@pytest.mark.asyncio
async def test_robots_respect_flag_off_bypasses_parsing():
    calls = {"n": 0}

    async def _fetch(_origin: str):
        calls["n"] += 1
        return ROBOTS_BLOCK_ALL

    gate = PolitenessGate(
        respect_robots=False,
        rate_limit_rps=0,
        robots_fetcher=_fetch,
    )
    # Even a block-all robots file is ignored, and never fetched.
    assert await gate.is_allowed("https://site.example/private/x") is True
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_robots_cached_per_origin():
    calls = {"n": 0}

    async def _fetch(_origin: str):
        calls["n"] += 1
        return ROBOTS_BLOCK_PRIVATE

    gate = PolitenessGate(
        respect_robots=True,
        rate_limit_rps=0,
        robots_fetcher=_fetch,
    )
    await gate.is_allowed("https://site.example/a")
    await gate.is_allowed("https://site.example/b")
    await gate.is_allowed("https://site.example/private/c")
    # One fetch for the origin, reused across all three checks.
    assert calls["n"] == 1


# ── check_and_wait: disallowed URLs are skipped ───────────────────


@pytest.mark.asyncio
async def test_check_and_wait_skips_disallowed():
    gate = PolitenessGate(
        respect_robots=True,
        rate_limit_rps=0,
        robots_fetcher=_fetcher(ROBOTS_BLOCK_PRIVATE),
    )
    assert await gate.check_and_wait("https://site.example/private/x") is False
    assert await gate.check_and_wait("https://site.example/ok") is True


# ── rate limiter spacing (fake clock, no real sleeps) ─────────────


class _FakeClock:
    """Monotonic-style clock that only advances when the injected sleep is
    awaited.  Lets us assert spacing without wall-clock time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.asyncio
async def test_rate_limiter_spaces_same_origin():
    clock = _FakeClock()
    gate = PolitenessGate(
        respect_robots=False,
        rate_limit_rps=2.0,  # min interval 0.5s
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    url = "https://site.example/page"
    # First call: window is open, no sleep.
    await gate.wait_for_slot(url)
    assert clock.sleeps == []

    # Second call immediately after: must wait one full interval.
    await gate.wait_for_slot(url)
    assert clock.sleeps == [0.5]

    # Third call: another full interval (clock advanced by the sleep).
    await gate.wait_for_slot(url)
    assert clock.sleeps == [0.5, 0.5]


@pytest.mark.asyncio
async def test_rate_limiter_independent_per_origin():
    clock = _FakeClock()
    gate = PolitenessGate(
        respect_robots=False,
        rate_limit_rps=2.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    # Two different origins never block each other on the first hit.
    await gate.wait_for_slot("https://a.example/x")
    await gate.wait_for_slot("https://b.example/y")
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_rate_limiter_disabled_when_rps_zero():
    clock = _FakeClock()
    gate = PolitenessGate(
        respect_robots=False,
        rate_limit_rps=0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    url = "https://site.example/page"
    await gate.wait_for_slot(url)
    await gate.wait_for_slot(url)
    assert clock.sleeps == []
