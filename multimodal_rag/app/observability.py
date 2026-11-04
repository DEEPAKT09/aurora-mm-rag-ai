"""Small, dependency-light observability helpers.

Use `request_context(request_id)` around a request or workflow operation to attach
an ID to log records. Use `measure_time()` to measure a named operation. This
module does not configure global logging handlers; configure handlers once in
your application entry point.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from time import perf_counter
from uuid import uuid4
import logging

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


@contextmanager
def request_context(request_id: str | None = None):
    value = request_id or uuid4().hex
    token = request_id_var.set(value)
    try:
        yield value
    finally:
        request_id_var.reset(token)


@contextmanager
def measure_time(operation: str, logger: logging.Logger | None = None):
    active_logger = logger or logging.getLogger("rag.performance")
    started = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - started) * 1000
        active_logger.info(
            "operation_timing operation=%s elapsed_ms=%.2f request_id=%s",
            operation, elapsed_ms, request_id_var.get(),
        )


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True
