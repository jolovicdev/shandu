from __future__ import annotations

from shandu.services.search import SearchService


def test_search_service_constructs() -> None:
    service = SearchService()
    assert service is not None


def test_search_backends_pair_engines_with_auto_fallback() -> None:
    from shandu.services.search import _TEXT_BACKENDS

    assert "lite" not in _TEXT_BACKENDS
    assert "html" not in _TEXT_BACKENDS
    primary = _TEXT_BACKENDS[0].split(",")
    assert "brave" in primary
    assert "duckduckgo" in primary
    assert _TEXT_BACKENDS[-1] == "auto"


def test_search_cache_evicts_oldest() -> None:
    from shandu.services.search import _SEARCH_CACHE_MAX

    service = SearchService()
    for i in range(_SEARCH_CACHE_MAX + 5):
        service._set_cached(f"k{i}", [])
    assert len(service._cache) == _SEARCH_CACHE_MAX
    assert "k0" not in service._cache
    assert f"k{_SEARCH_CACHE_MAX + 4}" in service._cache
