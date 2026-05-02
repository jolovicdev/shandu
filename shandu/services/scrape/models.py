from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    type: Literal["heading", "paragraph", "list_item", "table", "blockquote", "code", "caption"]
    text: str


class ScrapedPage(BaseModel):
    requested_url: str
    url: str
    title: str
    text: str
    blocks: list[ContentBlock] = Field(default_factory=list)
    domain: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: str | None = None
    content_type: str | None = None
    fetch_error: str | None = None
    http_status: int | None = None


class _ExtractionResult:
    def __init__(
        self,
        title: str = "",
        text: str = "",
        blocks: list[ContentBlock] | None = None,
        published_at: str | None = None,
    ) -> None:
        self.title = title
        self.text = text
        self.blocks = blocks or []
        self.published_at = published_at


class _FetchResult:
    def __init__(
        self,
        page: ScrapedPage,
        status: int | None = None,
        fetch_error: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.page = page
        self.status = status
        self.fetch_error = fetch_error
        self.retryable = retryable


class _ParseError(Exception):
    def __init__(self, fetch_error: str, message: str = "") -> None:
        self.fetch_error = fetch_error
        super().__init__(message)
