#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from streamlit.testing.v1 import AppTest


def _collect_page_files(project_root: Path) -> list[Path]:
    pages_dir = project_root / "pages"
    page_files = sorted(pages_dir.glob("*.py"))
    return [project_root / "app.py", *page_files]


def _check_page(path: Path) -> list[str]:
    at = AppTest.from_file(str(path)).run()
    errors: list[str] = []
    for exc in at.exception:
        message = str(exc.value).strip()
        if not message:
            message = "Unknown Streamlit exception"
        errors.append(message)
    return errors


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _streamlit_binary(project_root: Path) -> str:
    venv_streamlit = project_root / ".venv" / "bin" / "streamlit"
    if venv_streamlit.exists():
        return str(venv_streamlit)
    return "streamlit"


def _check_server_boot(project_root: Path, timeout_s: float = 25.0) -> str | None:
    port = _pick_free_port()
    cmd = [
        _streamlit_binary(project_root),
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    process = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{port}"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                return f"Streamlit exited during startup. Output:\\n{output.strip()}"
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:  # noqa: S310
                    if int(response.status) == 200:
                        return None
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.4)
        return f"Streamlit did not become ready within {timeout_s:.0f}s."
    finally:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    page_files = _collect_page_files(project_root=project_root)

    failed = False
    boot_error = _check_server_boot(project_root=project_root)
    if boot_error is not None:
        failed = True
        print(f"[FAIL] app.py server boot: {boot_error}")
    else:
        print("[ OK ] app.py server boot")

    for page_file in page_files:
        rel = page_file.relative_to(project_root)
        try:
            errors = _check_page(page_file)
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"[FAIL] {rel}: {exc}")
            continue

        if errors:
            failed = True
            print(f"[FAIL] {rel}: {errors[0]}")
        else:
            print(f"[ OK ] {rel}")

    if failed:
        print("\nStreamlit smoke-check failed.")
        return 1

    print("\nStreamlit smoke-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
