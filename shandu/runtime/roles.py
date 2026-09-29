from __future__ import annotations

from typing import Any

from ..config import Role


def role_model(settings: Any, role: Role) -> str:
    """The model configured for a role, or the runtime's default model."""
    return (getattr(settings, "role_models", None) or {}).get(role) or settings.model
