# Shandu - Open-Source Deep Research Agent (CLI + GUI)

Shandu is a Python deep research agent: ask a question, and a team of LLM agents
plans research loops, runs web searches, scrapes pages and documents, scores every
source for credibility, and writes a long-form markdown report with numbered
citations. It is open source and runs on your own machine with any model
LiteLLM supports: DeepSeek, OpenRouter, Anthropic, OpenAI, or a local endpoint.
Built on the Blackgeorge agent framework, usable from a terminal CLI or a
Gradio web GUI.

- Architecture deep dive: [`ARCH.md`](ARCH.md)
- Example long-form output: see the `examples` directory.
- DeepSeek Flash example: [`examples/deepseek-flash.md`](examples/deepseek-flash.md)

## How It Works

- The lead turns your question into a research brief: scope, concrete
  sub-questions, time window, and the shape of answer you want.
- It plans iterative research loops against the sub-questions still open, and
  parallel researchers search the web and, when papers can answer a task, arXiv,
  OpenAlex, and PubMed.
- Researchers read whole pages and documents, score every source for
  credibility, and extract claims with verbatim quotes that code checks against
  the page. The checked claims drive the coverage rating and the citation audit.
  An agentic researcher (`--researcher agentic`) also follows links to the
  primary sources a page cites.
- After each loop, code rates every sub-question by how many independent credible
  sources back it, and research continues until each has two or the budget ends.
- One call writes the report; an auditor then checks every cited sentence
  against its sources and removes unsupported citations.
- Every run is saved as it happens, so you can resume it or rewrite its report
  without searching again.
- The CLI streams one aligned line per event; the web GUI shows the run live and
  the finished report with its references.

## Installation

Recommended for end users (no manual venv management):

```bash
pipx install shandu
```

Standard pip install:

```bash
pip install shandu
```

Install latest from GitHub:

```bash
pipx install "git+https://github.com/jolovicdev/shandu.git@main"
```

## Quick Start

```bash
uv sync --dev
source .venv/bin/activate
cp .env.example .env
# edit .env with your provider/model settings
```

## API Key Configuration (LiteLLM Style)

`shandu configure` asks for:

- `Default model` (example: `deepseek/deepseek-flash`, `openrouter/minimax/minimax-m2.5`)
- `API key env var name` (example: `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `ANYSUPPORTED_API_KEY`)
- `API key value` (hidden input)

Shandu saves these in user config storage and exports the configured env var at runtime for LiteLLM if it is not already set in your shell.

Examples:

```bash
# DeepSeek
shandu configure
# model: deepseek/deepseek-flash
# env var name: DEEPSEEK_API_KEY
# key value: <your key>

# OpenRouter
shandu configure
# model: openrouter/minimax/minimax-m2.5
# env var name: OPENROUTER_API_KEY
# key value: <your key>
```

You can still configure keys only through shell env vars if you prefer:

```bash
export OPENROUTER_API_KEY="your_real_key"
```

## Environment Variables (Without `shandu configure`)

If you prefer not to use interactive configuration, set env vars directly.

Provider/model:

- `SHANDU_MODEL` (primary model selector, example `deepseek/deepseek-flash`)
- `OPENAI_MODEL_NAME` (compatibility fallback if `SHANDU_MODEL` is not set)

Provider API key routing:

- `SHANDU_API_KEY_ENV` (name of provider key env var, example `OPENROUTER_API_KEY`)
- `SHANDU_API_KEY` (actual key value that Shandu exports into `SHANDU_API_KEY_ENV` at runtime if missing)

Direct LiteLLM-style provider key env vars (examples):

- `DEEPSEEK_API_KEY`
- `OPENROUTER_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- Any other provider key name LiteLLM supports, for example `ANYSUPPORTED_API_KEY`

Generation/runtime controls:

- `SHANDU_TEMPERATURE` (default `0.2`)
- `SHANDU_MAX_TOKENS` (default `32768`)
- `SHANDU_STORAGE_DIR` (default `~/.shandu/storage`)
- `SHANDU_PROXY` (optional proxy for scraping)
- `SHANDU_SEARCH_BACKEND` (default `ddgs`): web search through `ddgs`, or a
  search API: `tavily` (`TAVILY_API_KEY`), `brave` (`BRAVE_API_KEY`), `exa`
  (`EXA_API_KEY`), or `searxng` (`SEARXNG_URL`, an instance with JSON output
  enabled). A backend without its variable falls back to `ddgs` with a warning.
- `OPENALEX_API_KEY`, `NCBI_API_KEY` (optional): raise the OpenAlex and PubMed
  rate limits for paper search.
- `SHANDU_MODEL_PLANNER`, `SHANDU_MODEL_RESEARCHER`, `SHANDU_MODEL_EXTRACTOR`,
  `SHANDU_MODEL_SYNTHESIZER`, `SHANDU_MODEL_WRITER`, `SHANDU_MODEL_AUDITOR`
  (optional): a model for one role, overriding the default model. The same
  overrides live under `api.models` in `~/.shandu/config.json`. Extraction makes
  most of the calls, so a cheaper extractor model cuts cost the most. A role on
  another provider needs that provider's key in the environment, for example
  `ANTHROPIC_API_KEY`.

Precedence:

1. If your provider key env var (for example `OPENROUTER_API_KEY`) is already set in shell, Shandu uses it.
2. Otherwise, Shandu uses `SHANDU_API_KEY_ENV` + `SHANDU_API_KEY` from config/env.

## CLI

```bash
shandu run "What did the SELECT trial find about semaglutide and cardiovascular events?" \
  --effort medium \
  --researcher agentic \
  --output report.md
```

Research options for `shandu run`:

- `--effort low|medium|high|max`: iterations, parallel tasks, pages and searches
  per task, and results per query together. Explicit flags below override it.
- `--max-iterations`, `--parallelism`, `--max-pages-per-task`,
  `--max-results-per-query`: the individual budgets. `--parallelism` caps how
  many tasks run at once inside each iteration.
- `--researcher pipeline|agentic`: a fixed search, rank, and read pass per task,
  or a tool loop that searches, reads, and follows links.
- `--include-domain`, `--exclude-domain` (repeatable), `--since YYYY-MM-DD`:
  source filters.
- `--clarify`: answer up to three clarifying questions before research starts.
- `--max-cost USD`: stop researching once this much is spent, then write the report.
  A model with no known price cannot be budgeted; the run says so and continues.
- `--no-audit`: skip checking citations against their sources.

During `shandu run`, progress events stream live in the terminal:

- `BOOTSTRAP` / `PLAN` / `SEARCH` / `SYNTHESIZE` / `CITE` / `REPORT` / `COMPLETE`
- Per-task search events (`Task <id> started` and `Task <id> completed`) with metrics
- Iteration index and task IDs for long-running model calls
- Run summary includes model call count across lead, subagents, and extraction
- Metered calls/tokens/cost appear when provider exposes billing/usage metrics

```bash
shandu aisearch "latest state of open-source browser automation in 2026" \
  --max-results 8 \
  --max-pages 3 \
  --detail-level high \
  --output aisearch.md
```

`aisearch` is the quick mode: a single answer built from live web search, with
source citations, when a full deep-research run is more than you need.

Citation behavior:

- Final reports enforce numeric citation markers (`[1]`, `[2]`, ...).
- Raw internal evidence IDs are removed from the rendered markdown.
- The final `## References` section is rendered from the citation ledger to keep numbering stable.

Other commands:

- `shandu info`
- `shandu configure`
- `shandu gui`
- `shandu aisearch <query>`
- `shandu list`: stored runs, newest first
- `shandu show <run_id>`: a stored run's summary and report
- `shandu report <run_id> [--detail-level] [--model] [--no-audit] [--output]`: rewrite the report from stored evidence, with no new searches
- `shandu resume <run_id>`: continue a stopped run after its last completed iteration
- `shandu inspect <run_id>`
- `shandu clean`: delete runtime caches and keep stored runs; `--all` also deletes the runs and their reports

Each run lives in `<storage_dir>/runs/<run_id>/`: the request, live events,
evidence checkpointed after every task, per-iteration plans and syntheses, the
citation ledger, `report.md`, and run stats.

### GUI

Launch the visual control room:

```bash
shandu gui --host 127.0.0.1 --port 7860
```

`gradio` ships with the default Shandu install, so `shandu gui` works out of the box.

GUI features:

- one screen that moves through three states: ask, watch the run, read the report
- run strip with stage, iteration progress, evidence, pages read, model calls, cost, and run id
- stage sequence (plan, search, synthesize, cite, write) marking the step in flight
- activity feed and a subagent board (focus, last query, hits, pages, evidence)
- report set at one reading measure, with a contents list and its reference list
- markdown download, plus recent reports listed with their question, citations, and cost
- settings in a sidebar (model, key env var, key, iterations, parallelism, detail, depth, search limits)
- light and dark, following the browser setting; `?__theme=dark` forces one

### GUI Preview

#### Running a query

![Shandu GUI during a run](assets/main.png)

#### Reading the report

![Shandu GUI report view](assets/report.png)

#### References

![Shandu GUI reference list](assets/citations.png)

## Python API

```python
from shandu import ResearchRequest, ShanduEngine

engine = ShanduEngine.from_config()
result = engine.run_sync(
    ResearchRequest(
        query="AI inference infrastructure 2026",
        max_iterations=2,
        parallelism=3,
    )
)
print(result.report_markdown)
engine.close()  # release the shared HTTP session
```

## Development

```bash
uv run ruff check .
uv run pytest -q
```

## Web Scraping Pipeline

- Three-layer HTML extraction: trafilatura → readability-lxml → BS4.
- Document-format support: PDF, DOCX, XLSX, CSV, plaintext, markdown.
- Structured blocks preserve headings, tables, code, blockquotes, and list items.
- Per-domain rate limiting with exponential backoff; 3 retry attempts with jitter.
- Fetch-error detection: paywall, captcha, empty JS shell, blocked, login-required.
- Publication-date extraction from OpenGraph, JSON-LD, DC, prism, sailthru, parsely meta tags.
- In-flight deduplication prevents concurrent duplicate fetches of the same URL.
- Redirect-aware: pages tracked by requested URL so redirects don't cause false misses.

## Source-Quality Enforcement

Shandu grades every page it reads instead of treating all search results as
equal, so reports lean on primary sources and peer-reviewed work rather than
undated blogs and marketing pages.

- The per-page extractor classifies each source on the existing extraction call (primary, official, peer-reviewed, journalism, corporate, community, personal blog, advocacy/marketing, aggregator, social profile, or unknown) and records authorship, dating, and whether the page only summarizes work it does not contain.
- Each evidence record carries a `source_class`, a `credibility_score` derived from those signals, and `quality_flags` such as `undated`, `no_author`, `promotional`, `snippet_only`, or `unassessed`.
- The synthesizer and reporter weight claims by credibility and name weak-only support explicitly; undated evidence is treated as stale for time-sensitive queries.
- The adaptive loop treats high-confidence but low-credibility evidence as weak, so weak-source corpora keep searching; run stats report source-class counts and the fraction of dated evidence.
- The citation ledger excludes evidence below the credibility bar (weak pages can still inform caveats in prose), merges URL variants of the same work (abstract/HTML/PDF) into one reference, and sanitizes boilerplate titles.
- Claims carry verbatim quotes; a claim whose quote is not in the page it came from is dropped before it can reach the report.
- After writing, every cited sentence, bullet, and table row is checked against the evidence behind its markers. Unsupported citations are removed, and the run summary counts supported, partial, and removed citations.

MIT license.
