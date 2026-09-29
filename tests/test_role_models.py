from __future__ import annotations

import asyncio
from types import SimpleNamespace

from shandu.agents.lead import LeadAgent, _PlanPayload
from shandu.agents.search_subagent import SearchSubagent, _ExtractionPayload
from shandu.config import Config, config
from shandu.contracts import ResearchRequest, SubagentTask
from shandu.runtime.bootstrap import RuntimeBootstrap
from shandu.services.scrape import ScrapedPage
from shandu.services.search import SearchHit


class _RecordingDesk:
    def __init__(self) -> None:
        self.models: dict[str, str] = {}

    async def arun(self, worker, job):
        self.models[worker.name] = worker.model
        if worker.name == "SubagentExtractor":
            data = _ExtractionPayload(snippet="s", extracted_text="t")
        else:
            data = _PlanPayload(subagent_tasks=[SubagentTask(task_id="t", focus="f")])
        return SimpleNamespace(status="completed", data=data, metrics={})


class _Search:
    async def search(self, query, max_results):
        return [SearchHit(query=query, url="https://a.org", title="A", snippet="s")]


class _Scrape:
    async def scrape_many(self, urls):
        return [
            ScrapedPage(requested_url=url, url=url, title="A", text="text", domain="a.org")
            for url in urls
        ], 0


def test_each_role_calls_its_configured_model() -> None:
    desk = _RecordingDesk()
    runtime = SimpleNamespace(
        settings=SimpleNamespace(model="base", role_models={"extractor": "cheap"}),
        desk=desk,
    )
    request = ResearchRequest(query="q")

    asyncio.run(LeadAgent(runtime).create_iteration_plan(request, 0, [], []))
    asyncio.run(
        SearchSubagent(runtime, _Search(), _Scrape()).execute_task(
            "run:r", SubagentTask(task_id="t", focus="f", search_queries=["q"]), request
        )
    )

    assert desk.models == {"LeadPlanner": "base", "SubagentExtractor": "cheap"}


def test_role_model_env_override_reaches_runtime_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHANDU_MODEL_WRITER", "anthropic/claude-sonnet")
    loaded = Config()
    assert loaded.get("api", "models") == {"writer": "anthropic/claude-sonnet"}

    monkeypatch.setitem(config._config["api"], "models", {"writer": "w", "unknown": "x", "auditor": " "})
    monkeypatch.setattr(RuntimeBootstrap, "__init__", lambda self, settings: setattr(self, "settings", settings))

    assert RuntimeBootstrap.from_config().settings.role_models == {"writer": "w"}
