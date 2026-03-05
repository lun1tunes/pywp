#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from typing import Final


MODE_TO_EXPR: Final[dict[str, str]] = {
    "unit": "not integration and not slow",
    "fast": "not slow",
    "integration": "integration and not slow",
    "slow": "slow",
    "full": "slow or not slow",
}


def _has_xdist() -> bool:
    return importlib.util.find_spec("xdist") is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run pytest suites by marker profile. "
            "Default mode is 'fast' (all except @slow)."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=tuple(MODE_TO_EXPR.keys()),
        default="fast",
        help="Suite profile: unit | fast | integration | slow | full.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra pytest args (prefix with --, e.g. -- -q tests/test_planner.py).",
    )
    args = parser.parse_args()

    marker_expr = MODE_TO_EXPR[args.mode]
    cmd = [sys.executable, "-m", "pytest", "-m", marker_expr]

    # Parallelize expensive suites when xdist is available.
    expensive_mode = args.mode in {"integration", "slow", "full"}
    has_xdist = _has_xdist()
    if expensive_mode and has_xdist:
        cmd.extend(["-n", "auto", "--dist", "loadscope"])

    if args.pytest_args:
        cmd.extend(args.pytest_args)

    if expensive_mode and not has_xdist:
        print(
            "Note: pytest-xdist is not installed. Running in a single process.",
            flush=True,
        )

    print("Running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    return int(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    raise SystemExit(main())
