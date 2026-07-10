"""
InkGraph backend — rate limiting and concurrent run semaphore.

Prevents overloading the Groq API and Supabase by limiting
simultaneous LangGraph workflow executions.
"""

import asyncio
from functools import wraps

from config import settings

# Global semaphore to cap concurrent workflow runs.
# Configured via MAX_CONCURRENT_RUNS env var (default: 20).
_workflow_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    """Lazily initialise and return the global workflow semaphore."""
    global _workflow_semaphore
    if _workflow_semaphore is None:
        _workflow_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_RUNS)
    return _workflow_semaphore


def with_rate_limit(coro):
    """Async decorator that acquires the workflow semaphore before executing."""
    @wraps(coro)
    async def wrapper(*args, **kwargs):
        async with get_semaphore():
            return await coro(*args, **kwargs)
    return wrapper
