from __future__ import annotations

from shandu.services.search import SearchService


def test_search_service_constructs() -> None:
    service = SearchService()
    assert service is not None


def test_search_backends_exclude_stale_names() -> None:
    from shandu.services.search import _TEXT_BACKENDS

    assert "lite" not in _TEXT_BACKENDS
    assert "html" not in _TEXT_BACKENDS
    assert "duckduckgo" in _TEXT_BACKENDS
