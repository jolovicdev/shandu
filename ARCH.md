# Shandu 3.0 Architecture

This document describes how Shandu 3.0 executes research runs end-to-end.

## 1) System Topology

```mermaid
flowchart TD
    user[User CLI and API] --> engine[Engine]
    engine --> orch[Orchestrator]

    orch --> lead[Lead agent]
    orch --> suba[Search subagent A]
    orch --> subb[Search subagent B]
    orch --> cite[Citation agent]

    suba --> search[Search service]
    suba --> scrape[Scrape service]
    subb --> search
    subb --> scrape

    orch --> mem[Memory service]
    orch --> report[Report service]
```

## 2) Iterative Run Loop

```mermaid
flowchart TD
    start([Start run]) --> plan[Lead agent creates plan]
    plan --> fanout[Run subagent tasks in parallel]
    fanout --> synth[Lead agent synthesizes evidence]
    synth --> decision{Continue loop?}
    decision -->|Yes| plan
    decision -->|No| cite[Citation agent builds ledger]
    cite --> draft[Lead agent builds final draft]
    draft --> render[Report service renders markdown]
    render --> done([Complete and persist result])
```

## 3) Control Flow (Vertical)

```mermaid
flowchart TD
    a[User submits query] --> b[Engine runs request]
    b --> c[Orchestrator bootstraps run and writes memory]
    c --> d[Lead agent creates iteration plan]
    d --> e[Search subagents execute in parallel]
    e --> f[Evidence merged]
    f --> g[Lead agent synthesizes iteration]
    g --> h{Need another loop}
    h -->|Yes| d
    h -->|No| i[Citation agent builds citations]
    i --> j[Lead agent builds final report]
    j --> k[Report service renders markdown]
    k --> l[Persist result and events]
    l --> m[Return research result]
```

## 4) Data Model Pipeline

```mermaid
flowchart TD
    req[Research request]
    plan[Iteration plan]
    ev[Evidence records]
    syn[Iteration summaries]
    cit[Citation entries]
    draft[Final report draft]
    result[Research result]

    req --> plan
    plan --> ev
    ev --> syn
    ev --> cit
    syn --> draft
    cit --> draft
    draft --> result
```

## 5) Parallelism Model

- `--parallelism` is the hard upper bound for concurrent subagent task execution per iteration.
- The planner attempts to generate enough independent tasks to use requested parallelism.
- The orchestrator enforces concurrency with an async semaphore.

```mermaid
flowchart TD
    p[Parallelism N] --> sem[Semaphore limit N]
    sem --> t1[Task 1]
    sem --> t2[Task 2]
    sem --> t3[Task 3]
    sem --> tx[More tasks]
    t1 --> out[Evidence merged]
    t2 --> out
    t3 --> out
    tx --> out
```

## 6) Citation Guarantees

- Final rendered reports use numeric citation markers only (`[1]`, `[2]`, ...).
- Internal evidence IDs are removed from rendered markdown.
- References are rebuilt from citation ledger ordering for stable output.

## 7) Module Boundaries (Black Box View)

- Engine: public runtime entrypoint (`run`, `stream`, `inspect`, `ai_search`).
- Orchestrator: iterative loop control, task fan-out, progress events, and result assembly.
- Lead agent: planning, synthesis, and final report drafting.
- Search subagents: evidence retrieval and extraction for each planned task.
- Citation agent: citation ledger generation and normalization.
- Search service: web search backend abstraction.
- Scrape service: URL canonicalization, page fetch, and content extraction.
- Memory service: persistent run memory and retrieval.
- Report service: citation normalization and final markdown rendering.
- AI search service: one-shot search + explanation workflow.
- Runtime bootstrap: model/runtime/memory wiring.
- Terminal UI: live progress, dashboards, and final display panels.

## 8) Example Output

- Example long-form report: see the `examples` directory.
