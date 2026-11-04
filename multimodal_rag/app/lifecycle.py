from threading import Condition
from contextlib import contextmanager
from time import monotonic


class OperationTracker:
    """Track active RAG operations and coordinate application shutdown."""

    def __init__(self):
        self._condition = Condition()
        self._active_operations = 0
        self._accepting_operations = False

    def start(self):
        """Allow new operations after successful application startup."""
        with self._condition:
            self._accepting_operations = True

    def try_begin(self) -> bool:
        """Register an operation unless the application is shutting down."""
        with self._condition:
            if not self._accepting_operations:
                return False

            self._active_operations += 1
            return True

    def end(self):
        """Mark an operation as completed."""
        with self._condition:
            if self._active_operations <= 0:
                raise RuntimeError("No active operation to finish.")

            self._active_operations -= 1

            if self._active_operations == 0:
                self._condition.notify_all()

    def stop_accepting_and_wait(self, timeout: float) -> bool:
        """Reject new operations and wait for existing ones to finish."""
        deadline = monotonic() + timeout

        with self._condition:
            self._accepting_operations = False

            while self._active_operations > 0:
                remaining = deadline - monotonic()

                if remaining <= 0:
                    return False

                self._condition.wait(timeout=remaining)

            return True

    @property
    def active_operations(self) -> int:
        with self._condition:
            return self._active_operations

    @property
    def accepting_operations(self) -> bool:
        with self._condition:
            return self._accepting_operations


operation_tracker = OperationTracker()