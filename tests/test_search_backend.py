from __future__ import annotations

from shandu.services.search import SearchService


def test_search_service_constructs() -> None:
    service = SearchService()
    assert service is not None
