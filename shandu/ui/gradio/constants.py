from __future__ import annotations

from ...interfaces import DepthPolicy, DetailLevel

DETAIL_LEVELS: tuple[DetailLevel, ...] = ("concise", "standard", "high")
DEPTH_POLICIES: tuple[DepthPolicy, ...] = ("adaptive", "fixed")

TIMELINE_HEADERS = ["Time", "Stage", "Task", "Message", "Metrics"]
TASK_HEADERS = [
    "Task",
    "Status",
    "Focus",
    "Last Query",
    "Hits",
    "Scraped",
    "Evidence",
    "Last Update",
]
TRACE_HEADERS = ["Time", "Task", "Trace", "Query", "URL", "Details"]
CITATION_HEADERS = ["#", "Publisher", "Title", "URL", "Accessed"]
