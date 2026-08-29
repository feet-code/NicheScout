# NicheScout

NicheScout is a **finite, restart-safe research tournament** that front-loads the selection of an initial SEO micro-SaaS portfolio. Its default job is to research 1,500 candidates for 72 active hours, underwrite the strongest candidates more deeply, and emit exactly **500 products grouped into variable-size, audience-coherent websites**.

It deliberately stops at `exports/ideas.json`. It does not deploy sites and it does not use GSC, PostHog, signups, or revenue. Those signals belong to the downstream deployment/feedback loop.

## Quick start on Windows

Install Python 3.11 or newer, clone the repo, and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
notepad .env
```

Replace `GEMINI_API_KEY=replace-me` with a free Google AI Studio key, then verify and start:

```powershell
.\.venv\Scripts\python.exe -m niche_scout doctor
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

Press `Ctrl+C` whenever you are done for the day. Run the same command tomorrow; it resumes automatically. No `--resume` flag is required, although `niche-scout resume` is provided as an explicit alias.

## Quick start on macOS/Linux

```bash
./scripts/setup.sh
$EDITOR .env
.venv/bin/python -m niche_scout doctor
./scripts/run.sh
```

If the shell files lost their executable bit, use `sh scripts/setup.sh` and `sh scripts/run.sh`.

## Safe smoke test

This calls no external API and writes to an isolated mock database, so it cannot contaminate real research:

```bash
niche-scout run --mock --max-actions 2
niche-scout status --mock
```

## What the agent does

```text
current web evidence
        |
        v
1,500 broad candidates -- deterministic hard gates / scoring
        |
        v
top 800: pain, SEO, commercial, feasibility validation
        |
        v
top 600: adversarial red-team research
        |
        v
remaining time: highest-value uncertainty near the cutoff
        |
        v
500 qualified products -- deterministic audience-fit grouping --> variable site count
```

Each action has two intentionally separate Gemini stages:

1. A Google-Search-grounded model gathers a cited dossier and negative evidence.
2. A free text model converts only that dossier into a strict schema.

Separating grounding from structured output makes citations inspectable and lets the synthesis fallback chain use more free models. The last ranking and grouping step is deterministic Python, not another subjective model vote. `products_per_site` is a soft size/diversity target; it is not a quota. A site closes when no remaining idea clears the audience-similarity threshold, and an unusually coherent audience can grow up to `max_products_per_site`.

Discovery strategies are selected with an upper-confidence-bound bandit. Product quality rewards increase exploitation of productive evidence sources while the uncertainty term preserves exploration. After required tournament coverage is complete, the planner spends remaining time on the highest expected value of information near the 500th-place cutoff.

## Gemini fallback and free-tier behavior (important)

The grounded fallback chain is `gemini-3.7-flash` -> `gemini-3.6-flash` -> `gemini-3.5-flash` -> `gemini-2.5-flash`. Google documents Search grounding support for all four, so each attempt receives the Google Search tool rather than using the 3.x models only for ungrounded synthesis.

Grounding capability is separate from API-tier entitlement. The model IDs have a free text tier, but Google's current Developer API pricing page lists Gemini 3.x Search grounding as unavailable on the free tier (it can be tested in AI Studio). It lists `gemini-2.5-flash` Search grounding as free for up to **500 requests per day**, shared with Flash-Lite. On an unbilled project, unavailable 3.x attempts fall through to the 2.5 anchor; if billing is enabled, verify Google's current quota and pricing before a long run. To make the API chain strictly free-tier-only, set `grounded_models = ["gemini-2.5-flash"]`.

When every usable model is limited, NicheScout records the cooldown in SQLite and waits with a 30-second heartbeat. Quota waiting does **not** consume the configured 72 active research hours. You can safely stop instead and restart later.

Official references: [Gemini Python SDK](https://googleapis.github.io/python-genai/), [pricing](https://ai.google.dev/gemini-api/docs/pricing), [Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search), and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## Search-volume policy

NicheScout does not pretend that AI can reverse-engineer exact keyword volume for free. It records:

- query intent and cluster breadth;
- observed SERP weakness and mismatch;
- current complaints, alternatives, pricing, and workflow evidence;
- numeric volume/CPC/difficulty only when a cited source explicitly measured it.

Unmeasured keyword hypotheses remain `null` and receive less confidence. Your later blog probes, GSC impressions/clicks, signups, and PostHog behavior will provide the real demand calibration after deployment.

## Restart safety

- SQLite runs in WAL mode and commits every candidate, citation, review, score, cooldown, and heartbeat.
- A grounded dossier and its structured response are stage-cached before database materialization.
- Interrupted/deferred missions reuse the same action ID, preventing duplicate review passes.
- Raw action artifacts are saved under `state/artifacts/NNNNNN-mode/`.
- `ideas.json` and the readable report are written atomically.

A hard shutdown can lose only the request that was literally in flight. The last completed stage and all prior research resume automatically.

## Observe and debug

These commands are safe while the agent is stopped; `status` is also useful from another terminal while it runs:

```bash
niche-scout status
niche-scout leaderboard --limit 30
niche-scout show 123
niche-scout errors --limit 20
```

Useful files:

- `state/logs/events.jsonl` — complete machine-readable event stream;
- `state/niche_scout.db` — authoritative state;
- `state/artifacts/` — mission, dossier, sources, schema result, and normalized batch;
- `exports/ideas.json` — final editable handoff;
- `exports/finalists.md` — readable variable-site/500-product ranking.

Use `--verbose` before the command for full console JSON:

```bash
niche-scout --verbose run --max-actions 1 --no-wait
```

`--max-actions 1 --no-wait` is the fastest real-API diagnostic. If it fails, the terminal traceback plus `niche-scout errors` and the JSONL log retain the exact model/status/mission context.

## Finishing and handoff

At the time limit, NicheScout exports only if at least 500 ideas satisfy every hard feasibility, source-count, validation, and red-team gate. It refuses to pad the file with junk; extend `target_active_hours` and rerun if necessary.

To intentionally export an early snapshot (all quality/count gates still apply):

```bash
niche-scout finalize --force
```

The version-2 contract is human editable:

```json
{
  "version": 2,
  "sites": [
    {
      "id": "portfolio-001-example-audience",
      "audience": "example audience",
      "productIds": ["one", "two", "three"]
    }
  ],
  "ideas": [
    {
      "id": "one",
      "siteId": "portfolio-001-example-audience",
      "name": "Example product",
      "research": {"evidence": [], "keywords": [], "reviews": []}
    }
  ]
}
```

You can reorder, edit, remove, or add your own ideas before importing it into the SEO deployment repository. Keep each `site.productIds` list and the matching idea `siteId` consistent. Sites may contain any non-empty number of products; the configured maximum is an optimizer guardrail, not a downstream schema restriction.

## Configuration

`scripts/setup.*` copies `config.example.toml` to the ignored `config.toml`. The defaults implement the requested 72-hour, 500-finalist tournament. Environment variables can override any key with the `NICHE_SCOUT_` prefix; list values are comma separated.

Examples:

```text
NICHE_SCOUT_TARGET_ACTIVE_HOURS=96
NICHE_SCOUT_HEARTBEAT_SECONDS=60
NICHE_SCOUT_SYNTHESIS_MODELS=gemini-3.5-flash-lite,gemini-2.5-flash-lite
```

The model policy fails fast when a configured model lacks a documented free text tier or when a grounded-chain model lacks Search-grounding support. Google Search billing/entitlement is controlled separately by the Google Cloud project, so keep billing disabled or use only `gemini-2.5-flash` when a hard zero-dollar ceiling matters.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite exercises an end-to-end miniature tournament/export, variable audience-driven grouping, hard feasibility kills, interrupted-action reuse, config parsing, and Gemini rate-limit fallback without making network calls.
