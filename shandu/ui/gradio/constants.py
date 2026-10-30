from __future__ import annotations

from ...contracts import ReportLength
from ...interfaces import DepthPolicy

DETAIL_LEVELS: tuple[ReportLength, ...] = ("auto", "concise", "standard", "high")
DEPTH_POLICIES: tuple[DepthPolicy, ...] = ("adaptive", "fixed")
