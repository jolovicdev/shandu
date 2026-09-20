from __future__ import annotations

from click.testing import CliRunner

from shandu.cli import cli
from shandu.config import config


def _run_clean(storage: str, home: str):
    previous = config.get("runtime", "storage_dir")
    config.set("runtime", "storage_dir", storage)
    try:
        import os

        old_home = os.environ.get("HOME")
        os.environ["HOME"] = home
        try:
            return CliRunner().invoke(cli, ["clean", "--force"])
        finally:
            if old_home is None:
                del os.environ["HOME"]
            else:
                os.environ["HOME"] = old_home
    finally:
        config.set("runtime", "storage_dir", previous)


def test_clean_force_refuses_home_directory(tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sentinel = home / "keep.txt"
    sentinel.write_text("x", encoding="utf-8")

    result = _run_clean(str(home), str(home))

    assert result.exit_code != 0
    assert sentinel.exists()
    assert "efus" in result.output


def test_clean_force_refuses_unmarked_directory(tmp_path) -> None:
    target = tmp_path / "plain"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("x", encoding="utf-8")

    result = _run_clean(str(target), str(tmp_path / "home"))

    assert result.exit_code != 0
    assert sentinel.exists()
    assert "marker" in result.output.lower() or "efus" in result.output


def test_clean_force_removes_marked_storage(tmp_path) -> None:
    target = tmp_path / "store"
    target.mkdir()
    (target / "memory.db").write_text("x", encoding="utf-8")

    result = _run_clean(str(target), str(tmp_path / "home"))

    assert result.exit_code == 0
    assert not target.exists()


def test_storage_dir_resolves_absolute_at_load(tmp_path, monkeypatch) -> None:
    import os

    from shandu.config import Config

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHANDU_STORAGE_DIR", ".blackgeorge")
    loaded = Config()
    assert os.path.isabs(str(loaded.get("runtime", "storage_dir")))
