from contextlib import contextmanager
from threading import BoundedSemaphore, Lock
from time import monotonic
from collections import OrderedDict
from typing import Any

from multimodal_rag.app.config import settings


# Shared limit across all model calls in this application process.
_model_semaphore = BoundedSemaphore(
    settings.max_concurrent_model_calls
)


@contextmanager
def model_call_slot():
    """
    Limit the number of simultaneous model calls.

    Raises TimeoutError if a slot cannot be acquired in time.
    """

    acquired = _model_semaphore.acquire(
        timeout=settings.model_request_timeout_seconds
    )

    if not acquired:
        raise TimeoutError(
            "The model service is busy. Please retry shortly."
        )

    try:
        yield
    finally:
        _model_semaphore.release()


class TTLResponseCache:
    """Thread-safe in-memory cache with TTL and maximum-entry limits."""

    def __init__(self, max_entries: int, ttl_seconds: int):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, Any]] = (
            OrderedDict()
        )
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return None

            expires_at, value = entry

            if monotonic() >= expires_at:
                del self._entries[key]
                return None

            # Mark this entry as recently used.
            self._entries.move_to_end(key)

            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            expires_at = monotonic() + self.ttl_seconds

            self._entries[key] = (expires_at, value)
            self._entries.move_to_end(key)

            # Remove expired entries.
            now = monotonic()

            expired_keys = [
                cache_key
                for cache_key, (expiry, _) in self._entries.items()
                if now >= expiry
            ]

            for cache_key in expired_keys:
                self._entries.pop(cache_key, None)

            # Enforce the maximum number of entries.
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


response_cache = TTLResponseCache(
    max_entries=settings.response_cache_max_entries,
    ttl_seconds=settings.response_cache_ttl_seconds,
)