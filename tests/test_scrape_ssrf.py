from __future__ import annotations

import asyncio
import socket

import pytest
from aiohttp import web
from aiohttp.resolver import DefaultResolver

from shandu.services.scrape.service import (
    _PublicAddressConnector,
    _PrivateAddressError,
    ScrapeService,
)

_BODY = " ".join(["informative content sentence"] * 60)


async def _started_server(counter: list[int]) -> tuple[web.AppRunner, int]:
    async def handler(request: web.Request) -> web.Response:
        counter.append(1)
        return web.Response(
            text=f"<html><head><title>Local</title></head><body><p>{_BODY}</p></body></html>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = getattr(site, "_server", None).sockets or []
    port = sockets[0].getsockname()[1]
    return runner, port


def test_direct_private_url_yields_fetch_error_without_request() -> None:
    async def run():
        counter: list[int] = []
        runner, port = await _started_server(counter)
        try:
            service = ScrapeService()
            page = await service.scrape(f"http://127.0.0.1:{port}/")
            return page, len(counter)
        finally:
            await runner.cleanup()

    page, count = asyncio.run(run())
    assert page.fetch_error is not None
    assert count == 0


def test_resolver_double_for_public_name_yields_fetch_error_without_request(
    monkeypatch,
) -> None:
    async def run():
        counter: list[int] = []
        runner, port = await _started_server(counter)
        service = ScrapeService()
        try:
            page = await service.scrape(f"http://public.example:{port}/")
            return page, len(counter)
        finally:
            await runner.cleanup()

    async def fake_resolve(self, host, port=0, family=socket.AF_INET):
        del self, family
        return [
            {
                "hostname": host,
                "host": "10.9.8.7",
                "port": port,
                "family": socket.AF_INET,
                "proto": 0,
                "flags": 0,
            }
        ]

    monkeypatch.setattr(DefaultResolver, "resolve", fake_resolve)
    page, count = asyncio.run(run())
    assert page.fetch_error is not None
    assert count == 0


class _RedirectSession:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        del kwargs
        self.requested.append(url)
        session = self

        class FakeResponse:
            url = "https://public.example/start"
            headers = {"location": "http://127.0.0.1:9/private"}
            status = 302

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        del session
        return FakeResponse()


def test_public_to_private_redirect_yields_fetch_error_without_second_request() -> None:
    async def run():
        service = ScrapeService()
        session = _RedirectSession()
        page = await service.scrape("https://public.example/start", session=session)
        return page, session.requested

    page, requested = asyncio.run(run())
    assert page.fetch_error is not None
    assert requested == ["https://public.example/start"]


def test_connector_blocks_private_records_and_keeps_public(monkeypatch) -> None:
    async def fake_resolve(self, host, port=0, family=socket.AF_INET):
        del self, family
        if host == "mixed.example":
            hosts = ["10.9.8.7", "93.184.216.34"]
        else:
            hosts = ["10.9.8.7"]
        return [
            {
                "hostname": host,
                "host": ip,
                "port": port,
                "family": socket.AF_INET,
                "proto": 0,
                "flags": 0,
            }
            for ip in hosts
        ]

    monkeypatch.setattr(DefaultResolver, "resolve", fake_resolve)

    async def run():
        connector = _PublicAddressConnector()
        try:
            with pytest.raises(_PrivateAddressError):
                await connector._resolve_host("127.0.0.1", 80)
            with pytest.raises(_PrivateAddressError):
                await connector._resolve_host("2130706433", 80)
            with pytest.raises(_PrivateAddressError):
                await connector._resolve_host("private.example", 443)
            kept = await connector._resolve_host("mixed.example", 443)
            assert [record["host"] for record in kept] == ["93.184.216.34"]
        finally:
            await connector.close()

    asyncio.run(run())
