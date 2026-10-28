from __future__ import annotations

import asyncio
from typing import Any

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


async def structured_call(runtime: RuntimeExecutionLike, worker: Any, job: Any) -> Any:
    return await asyncio.wait_for(runtime.desk.arun(worker, job), STRUCTURED_CALL_SECONDS)


async def writer_call(runtime: RuntimeExecutionLike, worker: Any, job: Any) -> Any:
    return await asyncio.wait_for(runtime.desk.arun(worker, job), WRITER_CALL_SECONDS)


async def researcher_call(runtime: RuntimeExecutionLike, worker: Any, job: Any) -> Any:
    return await asyncio.wait_for(runtime.desk.arun(worker, job), RESEARCHER_LOOP_SECONDS)
