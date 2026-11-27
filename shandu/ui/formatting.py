from __future__ import annotations

_AUDIT_LABELS = {
    "supported": "supported",
    "partial": "partial",
    "unsupported": "removed",
    "unchecked": "unchecked",
}


def audit_summary(counts: object) -> str:
    """Citation audit counts as "36 supported · 2 removed"; empty without an audit."""
    if not isinstance(counts, dict):
        return ""
    return " · ".join(
        f"{counts[key]} {label}" for key, label in _AUDIT_LABELS.items() if counts.get(key)
    )
