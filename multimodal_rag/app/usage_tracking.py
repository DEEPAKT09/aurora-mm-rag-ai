from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from multimodal_rag.app.config import settings


@dataclass
class RequestUsage:
    """Usage information collected during one RAG request."""

    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    calls_by_purpose: dict[str, int] = field(default_factory=dict)


_current_usage: ContextVar[RequestUsage | None] = ContextVar(
    "current_request_usage",
    default=None,
)

class ModelCallBudgetExceeded(RuntimeError):
    """Raised when a request exceeds its configured model-call budget."""


class ModelCallBudget:
    """Track and limit model calls made during one request."""

    def __init__(self, maximum_calls: int):
        self.maximum_calls = maximum_calls
        self._calls = 0
        self._lock = Lock()

    def consume(self, purpose: str) -> int:
        with self._lock:
            if self._calls >= self.maximum_calls:
                raise ModelCallBudgetExceeded(
                    "This request has reached its model-call limit."
                )

            self._calls += 1
            return self._calls

_current_budget: ContextVar[ModelCallBudget | None] = ContextVar(
    "current_model_call_budget",
    default=None,
)

def start_request_usage(
    maximum_calls: int | None = None,
) -> tuple[RequestUsage, Any, Any]:

    usage = RequestUsage()

    budget = ModelCallBudget(
        maximum_calls=(
            maximum_calls
            if maximum_calls is not None
            else settings.max_model_calls_per_request
        )
    )

    usage_token = _current_usage.set(usage)
    budget_token = _current_budget.set(budget)

    return usage, usage_token, budget_token


def finish_request_usage(
    usage_token: Any,
    budget_token: Any,
) -> None:

    _current_usage.reset(usage_token)
    _current_budget.reset(budget_token)


def consume_model_call(purpose: str) -> int:
    """Consume one call from the current request's model-call budget."""

    budget = _current_budget.get()

    if budget is None:
        # No request context exists, so there is no request-level budget.
        return 0

    return budget.consume(purpose)


def get_request_usage() -> RequestUsage | None:
    return _current_usage.get()

def calculate_estimated_cost(
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate estimated model cost in USD."""

    input_cost = (
        prompt_tokens
        * settings.model_input_cost_per_million_tokens
        / 1_000_000
    )

    output_cost = (
        completion_tokens
        * settings.model_output_cost_per_million_tokens
        / 1_000_000
    )

    return input_cost + output_cost

def record_model_call(
    purpose: str,
    response: Any,
) -> None:
    """Record token usage from a LangChain model response."""

    usage = _current_usage.get()

    if usage is None:
        return

    usage.model_calls += 1
    usage.calls_by_purpose[purpose] = (
        usage.calls_by_purpose.get(purpose, 0) + 1
    )

    metadata = getattr(response, "usage_metadata", None)

    if not metadata:
        response_metadata = getattr(response, "response_metadata", {})
        metadata = response_metadata.get("token_usage", {})

    if not metadata:
        return

    input_tokens = int(
        metadata.get("input_tokens", metadata.get("prompt_tokens", 0))
    )

    output_tokens = int(
        metadata.get("output_tokens", metadata.get("completion_tokens", 0))
    )

    total_tokens = int(
        metadata.get("total_tokens", input_tokens + output_tokens)
    )

    usage.prompt_tokens += input_tokens
    usage.completion_tokens += output_tokens
    usage.total_tokens += total_tokens
    usage.estimated_cost_usd += calculate_estimated_cost(
    prompt_tokens=input_tokens,
    completion_tokens=output_tokens,
)