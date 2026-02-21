# Shandu 3.0 Architecture

This document describes how Shandu 3.0 executes research runs end-to-end.

## 1) System Topology

```mermaid
flowchart TD
    user[User / CLI / Python API] --> engine[ShanduEngine]
    engine --> orch[LeadOrchestrator]

    orch --> lead[LeadAgent]
    orch --> suba[SearchSubagent A]
    orch --> subb[SearchSubagent B]
    orch --> cite[CitationAgent]

    suba --> search[SearchService]
    suba --> scrape[ScrapeService]
    subb --> search
    subb --> scrape

    orch --> mem[MemoryService]
    orch --> report[ReportService]
```

## 2) Iterative Run Loop

```mermaid
flowchart TD
    start([Start Run]) --> plan[LeadAgent creates plan]
    plan --> fanout[Run subagent tasks in parallel]
    fanout --> synth[LeadAgent synthesizes evidence]
    synth --> decision{Continue loop?}
    decision -->|Yes| plan
    decision -->|No| cite[CitationAgent builds ledger]
    cite --> draft[LeadAgent builds final draft]
    draft --> render[ReportService renders markdown]
    render --> done([Complete + persist result])
```

## 3) Control Flow (Vertical)

```mermaid
flowchart TD
    a[User submits query] --> b[Engine.run(request)]
    b --> c[Orchestrator bootstrap + memory write]
    c --> d[LeadAgent create_iteration_plan]
    d --> e[SearchSubagents execute in parallel]
    e --> f[EvidenceRecord[] merged]
    f --> g[LeadAgent synthesize_iteration]
    g --> h{Need another loop?}
    h -->|Yes| d
    h -->|No| i[CitationAgent build_citations]
    i --> j[LeadAgent build_final_report]
    j --> k[ReportService render markdown]
    k --> l[Persist result + events]
    l --> m[ResearchRunResult returned]
```

## 4) Data Model Pipeline

```mermaid
flowchart TD
    req[ResearchRequest]
    plan[IterationPlan]
    ev[EvidenceRecord[]]
    syn[IterationSynthesis[]]
    cit[CitationEntry[]]
    draft[FinalReportDraft]
    result[ResearchRunResult]

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
    p[parallelism = N] --> sem[Semaphore(N)]
    sem --> t1[Task 1]
    sem --> t2[Task 2]
    sem --> t3[Task 3]
    sem --> tx[Task ...]
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
