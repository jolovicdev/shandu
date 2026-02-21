# Shandu Package Architecture (3.0)

## Component Map

- CLI and Python API: user-facing entrypoints for execution and inspection.
- Engine: stable facade for run, stream, inspect, and AI-search flows.
- Orchestrator: controls iterative research loop and concurrency.
- Lead agent: builds plans, synthesizes findings, drafts final reports.
- Search subagents: gather evidence through search, scrape, and extraction.
- Citation agent: builds and normalizes citation ledger.
- Services layer: search backend, scraping pipeline, memory access, report rendering, one-shot AI search.
- Runtime layer: model setup, desk/memory wiring, and async runner bridge.
- UI layer: rich terminal rendering for events, summaries, and final outputs.

## Runtime Boundaries

- Runtime owns model execution, run store, and persistent memory.
- Orchestrator owns control flow, fan-out, merge, and completion.
- Agents own reasoning logic only; no direct CLI responsibilities.
- Services own external I/O and transformation boundaries.
- Engine composes all components and exposes a stable interface.

## Design Intent

- Replaceable black-box modules with explicit contracts.
- Async-first execution with deterministic sync bridge.
- Persistent run state and memory through Blackgeorge storage.
- UI concerns separated from orchestration and runtime logic.
