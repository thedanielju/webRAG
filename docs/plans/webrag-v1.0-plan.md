# WebRAG v1.0 — Public Release Plan

Status: approved scope, execution in progress. Direct commits to `main`. Two human gates:
(1) before `git push` to the public GitHub remote, (2) before tagging `v1.0.0`.

## Baseline (verified 2026-06-15)

- Full suite: 191 passed, 1 failed, 3 slow-integration skipped (`RUN_SLOW_INTEGRATION=1` to run).
- The one failure is `tests/test_firecrawl_client.py::test_firecrawl_map_smoke`, a live test referencing a stale API (`firecrawl_client.app.map()`); the module actually exposes `firecrawl_client.map(url)`. Fixed in Phase 1.
- Env ready: Docker `webrag-postgres` healthy at `postgresql://webrag:webrag@localhost:5432/webrag`; Ollama up with `nomic-embed-text`; real `.env` (Firecrawl + OpenAI) at repo root.

## Scope

In: multi-level recursion; Firecrawl-`map` reachability + coverage; wire both into the live MCP `deep` path; two-tier crawl stop; raw-HTTP fetch fallback (zero-key + re-enable LaTeX); robots.txt + rate limiting on both fetch paths; verified free/local path; GitHub Actions CI; full docs scrub + LICENSE + README/spec updates + version bump.

Deferred to v2 (do not build): named corpora, vision/image pass, hybrid search, streaming responses, PyPI publish.

## Defaults already decided

Shipped default config stays paid/best-quality (Firecrawl + OpenAI). Firecrawl `map` runs only in `deep` mode and is cached. robots.txt respected on both fetch paths. `fast` stays the MCP default; `deep` is opt-in genuine recursion.

---

## Phase 1 — Reachability frontier + coverage

Objective: orchestration enumerates the reachable page universe from the seed via `firecrawl_client.map()`, draws expansion candidates from it, and reports coverage.

| Area | Work |
|---|---|
| `src/01_ingestion/firecrawl_client.py` | `map()` already exists; confirm signature and return shape. Add a thin cache keyed by normalized seed origin so repeated `deep` calls in one answer reuse the map. |
| `src/04_orchestration/expander.py` | Add a reachability source: when in `deep` mode, fetch the seed's reachable set once, normalize/dedup (reuse `links.py`), and use it as the candidate universe that link scoring ranks against (today candidates come only from scraped page bodies). |
| `src/04_orchestration/models.py` | Extend the result model with coverage fields: `reachable_total`, `indexed_count`, `coverage_ratio`. |
| `src/05_mcp_server/formatter.py` | Surface coverage in the response (e.g. a `[COVERAGE]` line: "indexed 12 of ~40 reachable pages"). |
| `config.py` + `blank.env` | Confirm `firecrawl_map_default_limit`; add `REACHABILITY_ENABLED` (default true, deep-only) and map-cache TTL if needed. |

Tests: fix `tests/test_firecrawl_client.py::test_firecrawl_map_smoke` to call `firecrawl_client.map(url)`. Add unit tests for frontier dedup/normalization and coverage math (mock the map call — no live network in CI).
Acceptance: `pytest tests/test_orchestration_unit.py tests/test_firecrawl_client.py -q` green (live smoke may be marked/skipped in CI).
Judgment level: creative (candidate-universe design touches scoring).

## Phase 2 — Multi-level recursion + two-tier stop

Objective: genuine N-level recursive expansion driven by the evaluator, reachable through MCP `deep` mode; conservative hard safety backstops.

| Area | Work |
|---|---|
| `src/04_orchestration/engine.py` | Make the retrieve→evaluate→expand loop descend multiple levels (links of links), not a single +1 hop. Track depth and per-iteration provenance. |
| `src/04_orchestration/evaluator.py` | Keep the 11-rule quality stop as the PRIMARY halt (diminishing returns, coverage). |
| Two-tier backstop (`config.py` + `blank.env`) | Add hard ceilings, conservative defaults: `MAX_PAGES_PER_ANSWER`, `MAX_TOKENS_INDEXED_PER_ANSWER`, `ANSWER_WALLCLOCK_BUDGET_SECONDS` (and/or a credit budget). Enforced in `engine.py`; halting on a backstop is logged and surfaced. |
| `src/05_mcp_server/tools.py` / `formatter.py` | Ensure `research_mode="deep"` actually triggers multi-level recursion end to end; report stop reason (quality vs. backstop) and depth reached. |

Tests: extend `tests/test_orchestration_unit.py` with multi-level descent cases and backstop-trip cases (mock ingestion/retrieval).
Acceptance: `pytest tests/test_orchestration_unit.py tests/test_orchestration_integration.py -q`.
Judgment level: creative.

## Phase 3 — Raw-HTTP fetch fallback + robots + rate limiting

Objective: ingest without a Firecrawl key; re-enable LaTeX; be a polite crawler on both paths.

| Area | Work |
|---|---|
| `src/01_ingestion/` (new `raw_fetch_client.py`) | httpx + BeautifulSoup fetcher returning the same NormalizedDocument shape as the Firecrawl path. |
| `src/01_ingestion/service.py` | Provider selection: `INGESTION_PROVIDER=firecrawl|raw|auto`; `auto` falls back to raw when `FIRECRAWL_API_KEY` is empty. |
| `src/02_indexing/chunker.py` | Re-enable LaTeX flag detection in `_detect_markdown_flags` when content comes from the raw path (Firecrawl strips LaTeX; raw HTML preserves it). See progress.md note #7. |
| robots + rate limit (new `politeness.py`) | Respect robots.txt (cache per origin) and apply a default per-origin rate limit on BOTH fetch paths. Config: `RESPECT_ROBOTS_TXT` (default true), `CRAWL_RATE_LIMIT_RPS`. |
| `config.py` + `blank.env` | Add the keys above. |

Tests: new `tests/test_raw_fetch.py` (mock httpx), robots parsing tests, LaTeX-flag test for raw content.
Acceptance: `pytest tests/test_raw_fetch.py tests/test_service.py tests/test_chunker_images.py -q`.
Judgment level: routine→creative (NormalizedDocument parity and robots edge cases need care).

## Phase 4 — Free/local path wiring + verification

Objective: prove an end-to-end answer using only free components.

Free config: `EMBEDDING_BASE_URL=http://localhost:11434/v1`, `EMBEDDING_MODEL=nomic-embed-text`, `EMBEDDING_DIMENSIONS=768`, huggingface tokenizer; `RERANKER_PROVIDER=none`; `DECOMPOSITION_MODE=rule_based`; `INGESTION_PROVIDER=raw`. Verify dimension handling (schema is 1536 by default — confirm reindex/dimension switch works or document the constraint).
Acceptance (live, local-only, no paid keys): a real `answer()` returns correct citations. Documented as a `blank.env` profile block.
Judgment level: routine.

## Phase 5 — GitHub Actions CI

Objective: clean-checkout install + full non-live pytest on every push.

`.github/workflows/ci.yml`: matrix on Python 3.11/3.12; `services: postgres` using `pgvector/pgvector:pg16`; `pip install -e .`; run the suite with live-network tests deselected/marked. Add a CI status badge to README. Mark live tests (`test_firecrawl_*` live smokes) with a `live` marker so CI skips them deterministically.
Acceptance: workflow green on a test push (gate: do not push to public remote without sign-off).
Judgment level: routine.

## Phase 6 — Docs scrub + LICENSE + README/specs + version bump (LAST)

Natural-sounding human edit (not a regex purge) of all tracked/shipping markdown: `README.md`, `docs/*.md`, `SECURITY.md`, `specs/*.md`. Blocklist: em/en dashes (— –), "not X but Y" reframes, rule-of-three triads, AI-vocab tells (delve, leverage, seamless, robust, comprehensive, boasts, "it's worth noting", "plays a crucial role", "a testament to"), hype adjectives, bold-overuse, "let's dive in"/"in conclusion". Keep the single 🌐 title emoji + badges; strip other emoji. Light-touch on code comments only. Leave gitignored `agent_documentation/` alone.
Plus: add `LICENSE` (MIT, copyright "Daniel Ju"); document recursion/reachability/raw-fetch/robots/free-path and all new config keys in README + config reference; bump `pyproject.toml` version to `1.0.0`.
Acceptance: docs read naturally; `git grep -nP "[—–]"` over tracked markdown returns only intentional cases; full suite green.

## Final verification gate (definition of done)

Full `pytest` green on clean checkout; free-path live `answer()` with correct citations; paid-path live `answer()` plus a multi-level recursion run that demonstrably descends multiple levels and reports coverage. Paid test crawls self-capped to small URLs / single-digit credits, spend reported.

Then human gate 1 (push to public GitHub), then human gate 2 (tag `v1.0.0` + release notes).
