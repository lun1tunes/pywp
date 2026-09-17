from __future__ import annotations

import inspect
import multiprocessing
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from multiprocessing.context import BaseContext
from typing import Any, TypeVar, cast

__all__ = [
    "CALCULATION_WORKER_LIMIT",
    "calculation_budgeted",
    "calculation_worker_budget",
    "process_pool_context",
    "process_pool_start_method",
]


# This limit is process-local by design: each deployed app instance has its
# own Python process.  It bounds concurrent heavy calculations inside that
# instance. Four matches the existing batch/AC automatic worker ceiling.
CALCULATION_WORKER_LIMIT = 4
_budget_condition = threading.Condition()
_budget_in_use = 0
_budget_waiters: deque[object] = deque()
_budget_local = threading.local()
_F = TypeVar("_F", bound=Callable[..., Any])


@contextmanager
def calculation_worker_budget(requested_workers: int) -> Iterator[int]:
    """Reserve the shared worker budget and yield the effective worker count.

    A serial calculation reserves one slot as well, so a parallel operation
    cannot overlap with an unbounded number of serial CPU-heavy operations.
    Nested calls from one operation reuse its reservation; this is needed for
    the anti-collision pipeline (well build -> pair analysis).
    """

    global _budget_in_use

    requested = max(int(requested_workers), 0)
    active_depth = int(getattr(_budget_local, "depth", 0) or 0)
    if active_depth:
        inherited = int(_budget_local.grant)
        grant = min(max(requested, 1), inherited)
        _budget_local.depth = active_depth + 1
        _budget_local.grant = grant
        try:
            yield min(requested, grant)
        finally:
            _budget_local.depth = active_depth
            _budget_local.grant = inherited
        return

    demand = max(requested, 1)
    ticket = object()
    with _budget_condition:
        _budget_waiters.append(ticket)
        try:
            while (
                _budget_waiters[0] is not ticket
                or _budget_in_use >= CALCULATION_WORKER_LIMIT
            ):
                _budget_condition.wait()
            grant = min(demand, CALCULATION_WORKER_LIMIT - _budget_in_use)
            _budget_in_use += grant
        finally:
            _budget_waiters.remove(ticket)
            _budget_condition.notify_all()
    _budget_local.depth = 1
    _budget_local.grant = grant
    try:
        yield min(requested, grant)
    finally:
        _budget_local.depth = 0
        _budget_local.grant = 0
        with _budget_condition:
            _budget_in_use -= grant
            _budget_condition.notify_all()


def calculation_budgeted(function: _F) -> _F:
    """Decorate a public heavy calculation with the shared worker budget."""

    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        requested = int(bound.arguments.get("parallel_workers", 0) or 0)
        with calculation_worker_budget(requested) as effective:
            if "parallel_workers" in bound.arguments:
                bound.arguments["parallel_workers"] = int(effective)
            return function(*bound.args, **bound.kwargs)

    return cast(_F, wrapped)


def process_pool_start_method(platform: str | None = None) -> str:
    """Return the safest process start method for Streamlit worker pools."""

    platform_name = sys.platform if platform is None else str(platform)
    if platform_name == "win32" or platform_name == "darwin":
        return "spawn"
    return "forkserver"


def process_pool_context(
    platform: str | None = None,
    *,
    allow_stdin_fork: bool = False,
) -> BaseContext:
    """Build a multiprocessing context with a spawn fallback."""

    main_file = str(getattr(sys.modules.get("__main__"), "__file__", "") or "")
    if (
        allow_stdin_fork
        and (not main_file or main_file.startswith("<"))
        and "fork" in multiprocessing.get_all_start_methods()
    ):
        return multiprocessing.get_context("fork")
    preferred_method = process_pool_start_method(platform)
    try:
        return multiprocessing.get_context(preferred_method)
    except ValueError:
        return multiprocessing.get_context("spawn")
