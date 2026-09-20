from __future__ import annotations

from ...interfaces import DepthPolicy, DetailLevel

DETAIL_LEVELS: tuple[DetailLevel, ...] = ("concise", "standard", "high")
DEPTH_POLICIES: tuple[DepthPolicy, ...] = ("adaptive", "fixed")
