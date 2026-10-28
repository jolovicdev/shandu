from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from ..interfaces import RuntimeExecutionLike

# A structured call returns within a few minutes even on a slow reasoning
# model. A provider that accepts the request and then stalls would hold the
# run for the HTTP client's ten-minute timeout on each of its retries, so
# these calls get a shorter deadline and the caller's fallback takes over.
STRUCTURED_CALL_SECONDS = 420
# A long report from a slow reasoning model takes several minutes, so the
# writer gets more room before its next attempt runs.
WRITER_CALL_SECONDS = 900
# An agentic task makes many model calls. A loop that runs past this stops;
# the pages it read stay in the run and only its memo is lost.
RESEARCHER_LOOP_SECONDS = 1200

RejectionKind = Literal["key", "model", "credit"]

_REJECTION_SUMMARIES: dict[RejectionKind, str] = {
    "key": "The model provider rejected the API key",
    "model": "The model provider does not know this model",
    "credit": "The model provider account is out of credit",
}
# Providers word these replies differently, and some send a rejected key as a
# bad request, so the reply text is matched rather than the error class. A
# quota that runs out answers as a rate limit, so credit is matched before
# rate limits are set aside as passing.
_CREDIT = re.compile(
    r"insufficient[ _](?:balance|quota|credit)|payment required|credit balance|more credits",
    re.IGNORECASE,
)
_RATE_LIMIT = re.compile(r"rate[ _]?limit|too many requests", re.IGNORECASE)
_KEY = re.compile(
    r"authenticat|api[ _]?key|unauthori[sz]ed|no auth credentials|permission ?denied",
    re.IGNORECASE,
)
_MODEL = re.compile(
    r"llm provider not provided|model[ _]not[ _]found|supported api model names"
    r"|notfounderror|\bmodel\b[^.]{0,80}\b(?:does not exist|not exist)",
    re.IGNORECASE,
)
_PROVIDER_MESSAGE = re.compile(r'"message"\s*:\s*"([^"]+)"')


class ProviderRejectedError(RuntimeError):
    """A model call failed in a way no retry fixes: a rejected API key, an
    unknown model, or an account out of credit. Every later call would fail
    the same way, so the run stops instead of finishing on fallbacks."""

    def __init__(self, kind: RejectionKind, detail: str) -> None:
        super().__init__(f"{_REJECTION_SUMMARIES[kind]}: {detail}")
        self.kind: RejectionKind = kind


def _rejection(text: str) -> ProviderRejectedError | None:
    # Only a provider's reply counts. Other failures, such as a structured
    # answer that does not validate, can quote page text about API keys.
    provider_error = re.match(r"\s*litellm\.\w+:\s*", text)
    if provider_error is None:
        return None
    kind: RejectionKind | None = None
    if _CREDIT.search(text):
        kind = "credit"
    elif _RATE_LIMIT.search(text):
        return None
    elif _KEY.search(text):
        kind = "key"
    elif _MODEL.search(text):
        kind = "model"
    if kind is None:
        return None
    message = _PROVIDER_MESSAGE.search(text)
    detail = message.group(1) if message else text[provider_error.end() :].splitlines()[0]
    detail = re.sub(r"\s*\(request_id: [^)]*\)", "", detail).strip()
    return ProviderRejectedError(kind, detail[:200])


async def _arun(runtime: RuntimeExecutionLike, worker: Any, job: Any, seconds: float) -> Any:
    try:
        report = await asyncio.wait_for(runtime.desk.arun(worker, job), seconds)
    except Exception as exc:
        rejected = _rejection(str(exc))
        if rejected is not None:
            raise rejected from exc
        raise
    if getattr(report, "status", None) == "failed":
        for error in getattr(report, "errors", None) or []:
            rejected = _rejection(str(error))
            if rejected is not None:
                raise rejected
    return report


async def structured_call(runtime: RuntimeExecutionLike, worker: Any, job: Any) -> Any:
    return await _arun(runtime, worker, job, STRUCTURED_CALL_SECONDS)


async def writer_call(runtime: RuntimeExecutionLike, worker: Any, job: Any) -> Any:
    return await _arun(runtime, worker, job, WRITER_CALL_SECONDS)


async def researcher_call(runtime: RuntimeExecutionLike, worker: Any, job: Any) -> Any:
    return await _arun(runtime, worker, job, RESEARCHER_LOOP_SECONDS)
