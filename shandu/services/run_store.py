from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import config
from ..contracts import (
    CitationEntry,
    CitationVerdict,
    EvidenceRecord,
    IterationPlan,
    IterationSynthesis,
    ResearchBrief,
    ResearchRequest,
    ResearchRunResult,
    RunEvent,
)


class RunNotFoundError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class CompletedIteration:
    index: int
    plan: IterationPlan
    synthesis: IterationSynthesis
    counters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StoredRun:
    run_id: str
    request: ResearchRequest
    brief: ResearchBrief | None
    status: dict[str, Any]
    evidence: list[tuple[int, EvidenceRecord]]
    iterations: list[CompletedIteration]


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    query: str
    status: str
    created_at: str
    updated_at: str
    citation_count: int | None
    usd_spent: float | None
    report_path: Path | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    # Write-then-rename so a crash never leaves a half-written file that
    # would make the run unloadable.
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # A crash mid-append can leave a truncated last line.
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


class RunStore:
    """One directory per research run under ``root``.

    Evidence is checkpointed after every task and each iteration after its
    synthesis, so an interrupted run can be resumed or re-reported without
    repeating searches.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def run_dir(self, run_id: str) -> Path:
        if not run_id or run_id != Path(run_id).name or run_id.startswith("."):
            raise RunNotFoundError(run_id)
        return self.root / run_id

    def report_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "report.md"

    def create(
        self, run_id: str, request: ResearchRequest, settings: dict[str, Any] | None
    ) -> None:
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "request.json", request.model_dump(mode="json"))
        if settings is not None:
            _write_json(directory / "settings.json", settings)
        created = _now()
        _write_json(
            directory / "status.json",
            {"status": "running", "created_at": created, "updated_at": created},
        )

    def set_status(self, run_id: str, status: str, error: str | None = None) -> None:
        path = self.run_dir(run_id) / "status.json"
        payload = _read_json(path) or {}
        payload.update({"status": status, "updated_at": _now()})
        payload.pop("error", None)
        if error:
            payload["error"] = error
        _write_json(path, payload)

    def save_brief(self, run_id: str, brief: ResearchBrief) -> None:
        _write_json(self.run_dir(run_id) / "brief.json", brief.model_dump(mode="json"))

    def mark_research_done(self, run_id: str) -> None:
        path = self.run_dir(run_id) / "status.json"
        payload = _read_json(path) or {}
        payload["research_done"] = True
        _write_json(path, payload)

    def append_event(self, run_id: str, event: RunEvent) -> None:
        with (self.run_dir(run_id) / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def append_evidence(
        self, run_id: str, iteration: int, records: list[EvidenceRecord]
    ) -> None:
        if not records:
            return
        with (self.run_dir(run_id) / "evidence.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                row = {"iteration": iteration, **record.model_dump(mode="json")}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_iteration(self, run_id: str, iteration: CompletedIteration) -> None:
        path = self.run_dir(run_id) / "iterations.json"
        rows = [
            row
            for row in _read_json(path) or []
            if isinstance(row, dict) and row.get("iteration") != iteration.index
        ]
        rows.append(
            {
                "iteration": iteration.index,
                "plan": iteration.plan.model_dump(mode="json"),
                "synthesis": iteration.synthesis.model_dump(mode="json"),
                "counters": iteration.counters,
            }
        )
        _write_json(path, rows)

    def save_report(
        self,
        run_id: str,
        markdown: str,
        citations: list[CitationEntry],
        run_stats: dict[str, Any],
        citation_audit: list[CitationVerdict] | None = None,
    ) -> None:
        directory = self.run_dir(run_id)
        (directory / "report.md").write_text(markdown, encoding="utf-8")
        _write_json(
            directory / "audit.json",
            [verdict.model_dump(mode="json") for verdict in citation_audit or []],
        )
        _write_json(
            directory / "citations.json",
            [entry.model_dump(mode="json") for entry in citations],
        )
        _write_json(directory / "result.json", {"run_stats": run_stats})

    def exists(self, run_id: str) -> bool:
        try:
            return (self.run_dir(run_id) / "request.json").is_file()
        except RunNotFoundError:
            return False

    def load(self, run_id: str) -> StoredRun:
        directory = self.run_dir(run_id)
        request_payload = _read_json(directory / "request.json")
        if not isinstance(request_payload, dict):
            raise RunNotFoundError(run_id)
        evidence = [
            (int(row.get("iteration", 0)), EvidenceRecord.model_validate(row))
            for row in _read_jsonl(directory / "evidence.jsonl")
        ]
        iterations = sorted(
            (
                CompletedIteration(
                    index=int(row["iteration"]),
                    plan=IterationPlan.model_validate(row["plan"]),
                    synthesis=IterationSynthesis.model_validate(row["synthesis"]),
                    counters=dict(row.get("counters") or {}),
                )
                for row in _read_json(directory / "iterations.json") or []
                if isinstance(row, dict)
            ),
            key=lambda item: item.index,
        )
        brief_payload = _read_json(directory / "brief.json")
        return StoredRun(
            run_id=run_id,
            request=ResearchRequest.model_validate(request_payload),
            brief=ResearchBrief.model_validate(brief_payload)
            if isinstance(brief_payload, dict)
            else None,
            status=_read_json(directory / "status.json") or {},
            evidence=evidence,
            iterations=iterations,
        )

    def load_result(self, run_id: str) -> ResearchRunResult:
        stored = self.load(run_id)
        directory = self.run_dir(run_id)
        report = directory / "report.md"
        citations = _read_json(directory / "citations.json") or []
        audit = _read_json(directory / "audit.json") or []
        result = _read_json(directory / "result.json") or {}
        return ResearchRunResult(
            run_id=run_id,
            request=stored.request,
            brief=stored.brief,
            report_markdown=report.read_text(encoding="utf-8") if report.is_file() else "",
            citations=[CitationEntry.model_validate(entry) for entry in citations],
            citation_audit=[CitationVerdict.model_validate(entry) for entry in audit],
            evidence=[record for _, record in stored.evidence],
            iteration_summaries=[item.synthesis for item in stored.iterations],
            run_stats=dict(result.get("run_stats") or {}),
        )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.run_dir(run_id) / "events.jsonl")

    def settings(self, run_id: str) -> dict[str, Any] | None:
        payload = _read_json(self.run_dir(run_id) / "settings.json")
        return payload if isinstance(payload, dict) else None

    def list_runs(self) -> list[RunSummary]:
        if not self.root.is_dir():
            return []
        summaries: list[RunSummary] = []
        for directory in self.root.iterdir():
            request = _read_json(directory / "request.json")
            if not isinstance(request, dict):
                continue
            status = _read_json(directory / "status.json") or {}
            stats = (_read_json(directory / "result.json") or {}).get("run_stats") or {}
            citations = stats.get("citation_count")
            cost = stats.get("usd_spent")
            report = directory / "report.md"
            summaries.append(
                RunSummary(
                    run_id=directory.name,
                    query=str(request.get("query", "")),
                    status=str(status.get("status", "unknown")),
                    created_at=str(status.get("created_at", "")),
                    updated_at=str(status.get("updated_at", "")),
                    citation_count=citations if isinstance(citations, int) else None,
                    usd_spent=float(cost)
                    if isinstance(cost, (int, float)) and cost > 0
                    else None,
                    report_path=report if report.is_file() else None,
                )
            )
        summaries.sort(key=lambda item: item.created_at, reverse=True)
        return summaries


def default_run_store() -> RunStore:
    return RunStore(Path(str(config.get("runtime", "storage_dir"))) / "runs")
