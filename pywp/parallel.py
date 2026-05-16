from __future__ import annotations

import multiprocessing
import sys
from multiprocessing.context import BaseContext

__all__ = ["process_pool_context", "process_pool_start_method"]


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
