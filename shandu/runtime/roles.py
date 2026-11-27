from __future__ import annotations

from typing import Any

from ..config import Role


def role_model(settings: Any, role: Role) -> str:
    """The model configured for a role, or the runtime's default model."""
    override = (getattr(settings, "role_models", None) or {}).get(role)
    if isinstance(override, str) and override.strip():
        return override.strip()
    return str(settings.model)
