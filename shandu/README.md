# Shandu Package Architecture (4.0.0)

## Component Map

- CLI and Python API: user-facing entry points for running, resuming, re-reporting, and inspecting research.
- Engine: stable facade over the orchestrator, the run store, and one-shot AI search.
- Orchestrator: research brief, iterative loop, task fan-out, coverage-driven stopping, report assembly, citation audit.
- Lead agent: brief, clarifying questions, planning, synthesis, and the single and sectioned writers.
- Researchers: the pipeline and agentic researchers gather evidence per task; both extract and score every page the same way.
- Citation agent and auditor: build the citation ledger, then check every cited sentence against its sources.
- Services layer: web and scholarly search, the scraping pipeline, findings and coverage, report rendering, run store, one-shot AI search.
- Runtime layer: model setup, per-role models, desk wiring, and the async runner bridge.
- UI layer: rich terminal and Gradio rendering for events, summaries, and final reports.

## Runtime Boundaries

- Runtime owns model execution and the blackgeorge desk.
- The run store owns everything a run leaves on disk.
- Orchestrator owns control flow, fan-out, merge, and completion.
- Agents own reasoning logic only; no direct CLI responsibilities.
- Services own external I/O and transformation boundaries.
- Engine composes all components and exposes a stable interface.

## Design Intent

- Replaceable black-box modules with explicit contracts.
- Async-first execution with a deterministic sync bridge.
- Every run persisted as it happens, so it can be resumed or re-reported.
- Model output checked by code wherever it can be: quotes, coverage, citations.
- UI concerns separated from orchestration and runtime logic.
