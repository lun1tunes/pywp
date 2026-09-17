from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pywp import parallel


def test_worker_budget_caps_nested_requests_and_releases_after_exception() -> None:
    with parallel.calculation_worker_budget(100) as outer:
        assert outer == parallel.CALCULATION_WORKER_LIMIT
        with parallel.calculation_worker_budget(2) as inner:
            assert inner == 2
            with parallel.calculation_worker_budget(100) as nested:
                assert nested == 2
        with pytest.raises(RuntimeError, match="failure"):
            with parallel.calculation_worker_budget(0) as serial:
                assert serial == 0
                with parallel.calculation_worker_budget(100) as nested:
                    assert nested == 1
                raise RuntimeError("failure")
        assert parallel._budget_in_use == outer
    assert parallel._budget_in_use == 0
    assert not parallel._budget_waiters


def test_worker_budget_waits_when_full_and_wakes_without_losing_request() -> None:
    started = threading.Event()
    entered = threading.Event()

    def queued_operation() -> int:
        started.set()
        with parallel.calculation_worker_budget(2) as workers:
            entered.set()
            return workers

    with ThreadPoolExecutor(max_workers=1) as pool:
        with parallel.calculation_worker_budget(100):
            future = pool.submit(queued_operation)
            assert started.wait(2)
            assert not entered.wait(0.1)
        assert future.result(timeout=2) == 2
    assert parallel._budget_in_use == 0


def test_concurrent_serial_calculation_reserves_one_slot() -> None:
    def serial_operation() -> tuple[int, int]:
        with parallel.calculation_worker_budget(0) as workers:
            return workers, parallel._budget_in_use

    with ThreadPoolExecutor(max_workers=1) as pool:
        with parallel.calculation_worker_budget(3):
            assert pool.submit(serial_operation).result(timeout=2) == (0, 4)
    assert parallel._budget_in_use == 0


def test_decorator_preserves_positional_signature_and_caps_workers() -> None:
    @parallel.calculation_budgeted
    def calculate(value: str, parallel_workers: int = 0) -> tuple[str, int]:
        return value, parallel_workers

    assert calculate("input", 100) == ("input", 4)
    assert calculate("input") == ("input", 0)
    assert parallel._budget_in_use == 0


def test_process_pool_start_method_is_platform_safe() -> None:
    assert parallel.process_pool_start_method("win32") == "spawn"
    assert parallel.process_pool_start_method("darwin") == "spawn"
    assert parallel.process_pool_start_method("linux") == "forkserver"


def test_process_pool_context_falls_back_to_spawn(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get_context(method: str) -> str:
        calls.append(str(method))
        if method == "forkserver":
            raise ValueError("forkserver unavailable")
        return f"context:{method}"

    monkeypatch.setattr(parallel.multiprocessing, "get_context", fake_get_context)

    assert parallel.process_pool_context("linux") == "context:spawn"
    assert calls == ["forkserver", "spawn"]


def test_process_pool_context_can_use_fork_for_stdin_main(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get_context(method: str) -> str:
        calls.append(str(method))
        return f"context:{method}"

    monkeypatch.setattr(parallel.multiprocessing, "get_context", fake_get_context)
    monkeypatch.setattr(
        parallel.sys.modules["__main__"], "__file__", "<stdin>", raising=False
    )

    assert (
        parallel.process_pool_context("linux", allow_stdin_fork=True)
        == "context:fork"
    )
    assert calls == ["fork"]


def test_process_pool_context_avoids_unavailable_stdin_fork(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get_context(method: str) -> str:
        calls.append(str(method))
        return f"context:{method}"

    monkeypatch.setattr(parallel.multiprocessing, "get_context", fake_get_context)
    monkeypatch.setattr(parallel.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    monkeypatch.setattr(
        parallel.sys.modules["__main__"], "__file__", "<stdin>", raising=False
    )

    assert (
        parallel.process_pool_context("win32", allow_stdin_fork=True)
        == "context:spawn"
    )
    assert calls == ["spawn"]


def test_backend_worker_import_does_not_preload_streamlit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; "
        "import pywp.welltrack_batch; "
        "print('streamlit' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )

    assert completed.stdout.strip() == "False"


def test_top_level_default_crs_access_does_not_preload_streamlit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; "
        "import pywp; "
        "_ = pywp.DEFAULT_CRS; "
        "print('streamlit' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )

    assert completed.stdout.strip() == "False"
