from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_loads_dotenv_from_working_directory(tmp_path) -> None:
    (tmp_path / ".env").write_text("SHANDU_MODEL=probe/x\n", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("from shandu.cli import cli\ncli()\n", encoding="utf-8")
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SHANDU_MODEL", None)
    env.pop("OPENAI_MODEL_NAME", None)

    completed = subprocess.run(
        [sys.executable, str(runner), "info"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0
    assert "probe/x" in completed.stdout
