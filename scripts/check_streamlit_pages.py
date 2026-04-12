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

from pywp.models import TrajectoryConfig
from pywp.ui_calc_params import calc_param_defaults
from pywp.ui_utils import dls_to_pi

_WT_LEGACY_MIN_VALUES: dict[str, float] = {
    "wt_cfg_md_step_m": 1.0,
    "wt_cfg_md_step_control_m": 0.5,
    "wt_cfg_pos_tolerance_m": 0.1,
    "wt_cfg_entry_inc_target_deg": 70.0,
    "wt_cfg_entry_inc_tolerance_deg": 0.1,
    "wt_cfg_max_inc_deg": 80.0,
    "wt_cfg_max_total_md_postcheck_m": 100.0,
    "wt_cfg_kop_min_vertical_m": 0.0,
}


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


def _expected_calc_defaults() -> dict[str, float]:
    cfg = TrajectoryConfig()
    return {
        "Шаг MD, м": float(cfg.md_step_m),
        "Контрольный шаг MD, м": float(cfg.md_step_control_m),
        "Допуск по латерали, м": float(cfg.lateral_tolerance_m),
        "Допуск по вертикали, м": float(cfg.vertical_tolerance_m),
        "Целевой INC на t1, deg": float(cfg.entry_inc_target_deg),
        "Допуск INC на t1, deg": float(cfg.entry_inc_tolerance_deg),
        "Макс INC по стволу, deg": float(cfg.max_inc_deg),
        "Макс ПИ BUILD, deg/10m": float(dls_to_pi(cfg.dls_build_max_deg_per_30m)),
        "Мин VERTICAL до KOP, м": float(cfg.kop_min_vertical_m),
        "Макс итоговая MD (постпроверка), м": float(cfg.max_total_md_postcheck_m),
        "Макс рестартов решателя": float(cfg.turn_solver_max_restarts),
    }


def _find_number_value(at: AppTest, label: str) -> float | None:
    matches = [widget for widget in at.number_input if widget.label == label]
    if not matches:
        return None
    return float(matches[0].value)


def _find_selectbox_value(at: AppTest, label: str) -> str | None:
    matches = [widget for widget in at.selectbox if widget.label == label]
    if not matches:
        return None
    return str(matches[0].value)


def _check_calc_defaults_on_pages(project_root: Path) -> list[str]:
    expected = _expected_calc_defaults()
    errors: list[str] = []
    expected_turn_solver = str(TrajectoryConfig().turn_solver_mode)

    app_at = AppTest.from_file(str(project_root / "app.py")).run()
    for label, expected_value in expected.items():
        actual = _find_number_value(app_at, label)
        if actual is None:
            errors.append(f"app.py: input '{label}' not found.")
            continue
        if abs(float(actual) - float(expected_value)) > 1e-9:
            errors.append(
                f"app.py: '{label}'={actual} but expected {expected_value}."
            )
    turn_solver_label = "Метод решателя"
    app_turn_solver = _find_selectbox_value(app_at, turn_solver_label)
    if app_turn_solver is None:
        errors.append("app.py: solver method selectbox not found.")
    elif app_turn_solver != expected_turn_solver:
        errors.append(
            "app.py: "
            f"solver method is '{app_turn_solver}' but expected '{expected_turn_solver}'."
        )

    # Check PTC page instead of removed welltrack_import page.
    # PTC page only renders the calc-param form when records are loaded,
    # so we must provide a real welltrack source before clicking import.
    ptc_at = AppTest.from_file(str(project_root / "pages" / "03_ptc.py"))
    ptc_at.session_state["wt_source_mode"] = "Файл по пути"
    ptc_at.session_state["wt_source_path"] = "tests/test_data/WELLTRACKS.INC"
    ptc_at.run()
    import_buttons = [button for button in ptc_at.button if button.label == "Импорт целей"]
    if not import_buttons:
        errors.append("pages/03_ptc.py: import button not found.")
        return errors
    import_buttons[0].click()
    ptc_at.run()
    for label, expected_value in expected.items():
        actual = _find_number_value(ptc_at, label)
        if actual is None:
            errors.append(
                f"pages/03_ptc.py: input '{label}' not found after import."
            )
            continue
        if abs(float(actual) - float(expected_value)) > 1e-9:
            errors.append(
                "pages/03_ptc.py: "
                f"'{label}'={actual} but expected {expected_value}."
            )
    ptc_turn_solver = _find_selectbox_value(ptc_at, turn_solver_label)
    if ptc_turn_solver is None:
        errors.append(
            "pages/03_ptc.py: solver method selectbox not found after import."
        )
    elif ptc_turn_solver != expected_turn_solver:
        errors.append(
            "pages/03_ptc.py: "
            f"solver method is '{ptc_turn_solver}' but expected '{expected_turn_solver}'."
        )

    # Regression check: even with stale legacy keys in state, defaults must recover
    # without requiring manual "reset params" click.
    ptc_legacy = AppTest.from_file(
        str(project_root / "pages" / "03_ptc.py")
    )
    ptc_legacy.session_state["wt_source_mode"] = "Файл по пути"
    ptc_legacy.session_state["wt_source_path"] = "tests/test_data/WELLTRACKS.INC"
    calc_defaults = calc_param_defaults()
    for key, value in _WT_LEGACY_MIN_VALUES.items():
        ptc_legacy.session_state[key] = value
    ptc_legacy.session_state["wt_cfg___calc_param_defaults_signature__"] = tuple(
        (key, calc_defaults[key]) for key in sorted(calc_defaults.keys())
    )
    ptc_legacy.session_state["wt_cfg___calc_param_defaults_schema_version__"] = 2
    ptc_legacy.run()
    legacy_import_buttons = [
        button for button in ptc_legacy.button if button.label == "Импорт целей"
    ]
    if not legacy_import_buttons:
        errors.append(
            "pages/03_ptc.py: import button not found (legacy check)."
        )
        return errors
    legacy_import_buttons[0].click()
    ptc_legacy.run()
    for label, expected_value in expected.items():
        actual = _find_number_value(ptc_legacy, label)
        if actual is None:
            errors.append(
                "pages/03_ptc.py: "
                f"input '{label}' not found after import (legacy check)."
            )
            continue
        if abs(float(actual) - float(expected_value)) > 1e-9:
            errors.append(
                "pages/03_ptc.py legacy recovery: "
                f"'{label}'={actual} but expected {expected_value}."
            )
    legacy_turn_solver = _find_selectbox_value(ptc_legacy, turn_solver_label)
    if legacy_turn_solver is None:
        errors.append(
            "pages/03_ptc.py: solver method selectbox not found (legacy check)."
        )
    elif legacy_turn_solver != expected_turn_solver:
        errors.append(
            "pages/03_ptc.py legacy recovery: "
            f"solver method is '{legacy_turn_solver}' but expected '{expected_turn_solver}'."
        )
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

    defaults_errors = _check_calc_defaults_on_pages(project_root=project_root)
    if defaults_errors:
        failed = True
        print(f"[FAIL] calc defaults sync: {defaults_errors[0]}")
    else:
        print("[ OK ] calc defaults sync (app + ptc)")

    if failed:
        print("\nStreamlit smoke-check failed.")
        return 1

    print("\nStreamlit smoke-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
