from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pywp import parallel


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
    monkeypatch.setattr(parallel.sys.modules["__main__"], "__file__", "<stdin>")

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
    monkeypatch.setattr(parallel.sys.modules["__main__"], "__file__", "<stdin>")

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
