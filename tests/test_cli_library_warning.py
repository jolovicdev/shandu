from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# primp reads a logger's level once, on its first message, so each check runs
# in a fresh interpreter that has not logged anything yet.
_BUILD_CLIENT_WITH_UNKNOWN_PROFILE = (
    "primp.Client(impersonate='no_such_browser_99', impersonate_os='linux')\n"
)


def _stderr_of(code: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stderr


def test_primp_warns_about_an_unknown_browser_profile_on_its_own() -> None:
    assert "does not exist" in _stderr_of("import primp\n" + _BUILD_CLIENT_WITH_UNKNOWN_PROFILE)


def test_the_search_service_silences_that_warning() -> None:
    code = "import shandu.services.search, primp\n" + _BUILD_CLIENT_WITH_UNKNOWN_PROFILE

    assert "does not exist" not in _stderr_of(code)
