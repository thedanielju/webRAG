# Security Policy

This document defines the security posture for WebRAG as a self-hosted, stateful MCP server.

## Scope

WebRAG currently includes:
- ingestion from external URLs via Firecrawl
- indexing into Postgres + pgvector
- retrieval over indexed content
- local/self-hosted deployment via `.env` and Docker Compose

This project is currently designed for **single-user, trusted-operator deployments**. It is not hardened as a multi-tenant internet-facing SaaS (or tbh, ever will be).

## Security Model

Current assumptions:
- Operator controls infrastructure and runtime environment.
- Users are expected to submit useful/trusted URLs.
- Postgres is run locally (or in a trusted private network).
- MCP server is not exposed publicly without additional controls.

If your deployment violates these assumptions, apply the hardening guidance below before production use.

## Threat Model and Controls

### 1) Untrusted URL ingestion (SSRF and internal network exposure)

Risk:
- Any ingestion system that fetches arbitrary URLs can be abused to access internal resources (metadata endpoints, private services, localhost).

Current state:
- No strict domain allowlist is enforced by default (intended for flexible research workflows).

Recommended controls:
- Run WebRAG in a network-restricted environment.
- Block RFC1918/private ranges and link-local addresses at firewall/proxy level.
- Deny access to cloud metadata endpoints (for example `169.254.169.254`).
- Enforce crawl depth/link limits for orchestration.
- If operating in higher-risk environments, add explicit domain allow/deny policy.

### 2) Prompt injection and malicious page content

Risk:
- Retrieved web content can contain adversarial instructions that attempt to override system behavior.

Current state:
- Retrieval is citation-oriented and preserves source provenance, but hostile text may still be returned.

Recommended controls:
- Treat retrieved text as untrusted data, never trusted instructions.
- Keep strong system-level prompt boundaries in MCP orchestration.
- Require source-cited outputs for high-risk actions.
- Add post-retrieval policy checks for sensitive tool invocations.

### 3) HTML content handling and rendering safety

Risk:
- HTML content in `html_text` may contain unsafe markup if later rendered in a UI.

Current state:
- WebRAG stores HTML for fidelity (tables/code/math structure), but core server flow does not require direct browser rendering. The retrieval layer generally filters for rich html matching said content.

Recommended controls:
- If rendering HTML in any UI, sanitize with a strict allowlist sanitizer before display.
- Prefer markdown/plain-text rendering where possible.
- Disallow inline script/event handlers in any downstream renderer.

### 4) Secrets handling (`.env`, API keys, DB credentials)

Risk:
- Secret leakage through source control, logs, stack traces, or shell history.

Current state:
- Keys are loaded from `.env` via `pydantic-settings`.

Recommended controls:
- Never commit `.env`; keep `.env` in `.gitignore`.
- Rotate keys immediately if exposed.
- Avoid logging full configuration objects.
- Redact secret-like fields in debug output.
- Use separate keys per environment.

### 5) Database exposure and integrity

Risk:
- Unauthorized DB access, data tampering, or data exfiltration.

Current state:
- Local Docker Compose setup is supported and expected.

Recommended controls:
- Do not expose Postgres publicly.
- Use non-default strong credentials outside local dev.
- Restrict network access to DB port.
- Enable TLS and least-privilege DB roles for remote deployments.
- Backup and encrypt data if persistence is business-critical.

### 6) Cost and availability abuse

Risk:
- Excessive ingestion or retrieval traffic can inflate external API cost (Firecrawl, embedding APIs) and degrade availability.

Recommended controls:
- Enforce per-request URL count/depth/token budgets.
- Cap embedding concurrency and batch sizes.
- Add request rate limiting at the MCP boundary.
- Track usage metrics and alert on unusual spikes.

### 7) Dependency and supply chain risk

Risk:
- Vulnerabilities in Python dependencies or transitive packages.

Recommended controls:
- Pin critical dependencies where possible.
- Run periodic vulnerability scanning (e.g., `pip-audit`).
- Keep `firecrawl-py`, `psycopg`, `pgvector`, and model clients updated.
- Review dependency updates before promotion to production.

## Deployment Hardening Checklist

Before production-like use, please verify:
- [ ] MCP server is not publicly reachable without authentication.
- [ ] Postgres is private and credentialed.
- [ ] `.env` is not committed and secrets are rotated regularly.
- [ ] Network egress restrictions are in place for URL fetching.
- [ ] Rate limits and traversal limits are configured.
- [ ] HTML rendering path (if any) uses sanitization.
- [ ] Logging excludes secrets and sensitive payloads.

## Reporting Security Issues

Please do **not** open public issues for suspected vulnerabilities.

Preferred process:
1. Open a private GitHub Security Advisory for this repository, if available.
2. If advisories are unavailable, contact the maintainer privately with:
   - vulnerability description
   - impact assessment
   - reproduction steps
   - suggested remediation (if known)