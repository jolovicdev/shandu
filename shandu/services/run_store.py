from __future__ import annotations

import json
import os
import tempfile
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
from .findings import build_findings


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
    citation_count: int | None
    usd_spent: float | None
    report_path: Path | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text_atomic(path: Path, content: str) -> None:
    # Unique temp file + rename so concurrent writers never collide and a crash
    # never leaves a half-written file.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_json(path: Path, payload: Any) -> None:
    _write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


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
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_lines:
        # A crash mid-append can leave a truncated last line (even mid-UTF-8).
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, lines: list[str]) -> None:
    if not lines:
        return
    needs_newline = False
    if path.is_file():
        try:
            size = path.stat().st_size
            if size > 0:
                with path.open("rb") as probe:
                    probe.seek(-1, os.SEEK_END)
                    needs_newline = probe.read(1) != b"\n"
        except OSError:
            needs_newline = False
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        for line in lines:
            handle.write(line + "\n")


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
        self._patch_status(run_id, status=status, updated_at=_now(), error=error or None)

    def save_brief(self, run_id: str, brief: ResearchBrief) -> None:
        _write_json(self.run_dir(run_id) / "brief.json", brief.model_dump(mode="json"))

    def mark_research_done(self, run_id: str) -> None:
        self._patch_status(run_id, research_done=True)

    def record_spend(self, run_id: str, llm_totals: dict[str, Any]) -> None:
        """Keep the run's metered totals current, so a resume after an
        interruption still counts what the interrupted iteration spent."""
        self._patch_status(run_id, llm_totals=dict(llm_totals))

    def _patch_status(self, run_id: str, **fields: Any) -> None:
        """Update status.json fields; a None value removes the field."""
        path = self.run_dir(run_id) / "status.json"
        raw = _read_json(path)
        payload = dict(raw) if isinstance(raw, dict) else {}
        for key, value in fields.items():
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
        _write_json(path, payload)

    def append_event(self, run_id: str, event: RunEvent) -> None:
        _append_jsonl(
            self.run_dir(run_id) / "events.jsonl",
            [event.model_dump_json()],
        )

    def drop_evidence_from(self, run_id: str, iteration: int) -> None:
        """Remove evidence rows from ``iteration`` on, before it runs again."""
        path = self.run_dir(run_id) / "evidence.jsonl"
        kept: list[dict[str, Any]] = []
        for row in _read_jsonl(path):
            try:
                row_iter = int(row.get("iteration", 0))
            except (TypeError, ValueError):
                continue
            if row_iter < iteration:
                kept.append(row)
        _write_text_atomic(
            path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        )

    def append_evidence(
        self, run_id: str, iteration: int, records: list[EvidenceRecord]
    ) -> None:
        if not records:
            return
        _append_jsonl(
            self.run_dir(run_id) / "evidence.jsonl",
            [
                json.dumps(
                    {"iteration": iteration, **record.model_dump(mode="json")},
                    ensure_ascii=False,
                )
                for record in records
            ],
        )

    def save_iteration(self, run_id: str, iteration: CompletedIteration) -> None:
        path = self.run_dir(run_id) / "iterations.json"
        existing = _read_json(path)
        rows = [
            row
            for row in (existing if isinstance(existing, list) else [])
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
        _write_json(
            directory / "audit.json",
            [verdict.model_dump(mode="json") for verdict in citation_audit or []],
        )
        _write_json(
            directory / "citations.json",
            [entry.model_dump(mode="json") for entry in citations],
        )
        _write_json(directory / "result.json", {"run_stats": run_stats})
        # Write report.md last and atomically so list_runs/load_result never see
        # a report without matching citations/stats or a half-written file.
        _write_text_atomic(directory / "report.md", markdown)

    def load(self, run_id: str) -> StoredRun:
        directory = self.run_dir(run_id)
        request_payload = _read_json(directory / "request.json")
        if not isinstance(request_payload, dict):
            raise RunNotFoundError(run_id)
        evidence: list[tuple[int, EvidenceRecord]] = []
        for row in _read_jsonl(directory / "evidence.jsonl"):
            try:
                evidence.append(
                    (int(row.get("iteration", 0)), EvidenceRecord.model_validate(row))
                )
            except (TypeError, ValueError):
                continue
        raw_iterations = _read_json(directory / "iterations.json")
        parsed_iterations: list[CompletedIteration] = []
        for row in raw_iterations if isinstance(raw_iterations, list) else []:
            if not isinstance(row, dict):
                continue
            try:
                parsed_iterations.append(
                    CompletedIteration(
                        index=int(row["iteration"]),
                        plan=IterationPlan.model_validate(row["plan"]),
                        synthesis=IterationSynthesis.model_validate(row["synthesis"]),
                        counters=dict(row.get("counters") or {}),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        iterations = sorted(parsed_iterations, key=lambda item: item.index)
        brief_payload = _read_json(directory / "brief.json")
        raw_status = _read_json(directory / "status.json")
        return StoredRun(
            request=ResearchRequest.model_validate(request_payload),
            brief=ResearchBrief.model_validate(brief_payload)
            if isinstance(brief_payload, dict)
            else None,
            status=raw_status if isinstance(raw_status, dict) else {},
            evidence=evidence,
            iterations=iterations,
        )

    def load_result(self, run_id: str) -> ResearchRunResult:
        stored = self.load(run_id)
        directory = self.run_dir(run_id)
        report = directory / "report.md"
        raw_citations = _read_json(directory / "citations.json")
        citations = raw_citations if isinstance(raw_citations, list) else []
        raw_audit = _read_json(directory / "audit.json")
        audit = raw_audit if isinstance(raw_audit, list) else []
        raw_result = _read_json(directory / "result.json")
        result = raw_result if isinstance(raw_result, dict) else {}
        evidence = [record for _, record in stored.evidence]
        return ResearchRunResult(
            run_id=run_id,
            request=stored.request,
            brief=stored.brief,
            report_markdown=report.read_text(encoding="utf-8") if report.is_file() else "",
            citations=[CitationEntry.model_validate(entry) for entry in citations],
            citation_audit=[CitationVerdict.model_validate(entry) for entry in audit],
            evidence=evidence,
            findings=build_findings(evidence),
            iteration_summaries=[item.synthesis for item in stored.iterations],
            run_stats=dict(result.get("run_stats") or {})
            if isinstance(result.get("run_stats"), dict)
            else {},
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
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            request = _read_json(directory / "request.json")
            if not isinstance(request, dict):
                continue
            raw_status = _read_json(directory / "status.json")
            status = raw_status if isinstance(raw_status, dict) else {}
            raw_result = _read_json(directory / "result.json")
            raw_stats = (
                raw_result.get("run_stats") if isinstance(raw_result, dict) else None
            )
            stats = raw_stats if isinstance(raw_stats, dict) else {}
            citations = stats.get("citation_count")
            cost = stats.get("usd_spent")
            report = directory / "report.md"
            summaries.append(
                RunSummary(
                    run_id=directory.name,
                    query=str(request.get("query", "")),
                    status=str(status.get("status", "unknown")),
                    created_at=str(status.get("created_at", "")),
                    citation_count=citations if isinstance(citations, int) else None,
                    usd_spent=float(cost)
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0
                    else None,
                    report_path=report if report.is_file() else None,
                )
            )
        summaries.sort(key=lambda item: item.created_at, reverse=True)
        return summaries


def default_run_store() -> RunStore:
    return RunStore(Path(str(config.get("runtime", "storage_dir"))) / "runs")
