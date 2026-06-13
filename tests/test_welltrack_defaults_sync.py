from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import TrajectoryConfig
from pywp.ui_calc_params import calc_param_defaults
from pywp.welltrack_batch import SuccessfulWellPlan

pytestmark = pytest.mark.integration


LEGACY_MIN_VALUES: dict[str, float] = {
    "wt_cfg_md_step_m": 1.0,
    "wt_cfg_md_step_control_m": 0.5,
    "wt_cfg_pos_tolerance_m": 0.1,
    "wt_cfg_entry_inc_target_deg": 70.0,
    "wt_cfg_entry_inc_tolerance_deg": 0.1,
    "wt_cfg_max_inc_deg": 80.0,
    "wt_cfg_max_total_md_postcheck_m": 100.0,
    "wt_cfg_kop_min_vertical_m": 0.0,
}


def _number_input_value(at: AppTest, label: str) -> float | None:
    matches = [widget for widget in at.number_input if widget.label == label]
    if not matches:
        return None
    return float(matches[0].value)


def _selectbox_value(at: AppTest, label: str) -> str | None:
    matches = [widget for widget in at.selectbox if widget.label == label]
    if not matches:
        return None
    return str(matches[0].value)


def _records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        )
    ]


def _successful_plan() -> SuccessfulWellPlan:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 90.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [0.0, 1000.0, 2000.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 0.0, 0.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0],
            "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
        }
    )
    return SuccessfulWellPlan(
        name="WELL-A",
        surface={"x": 0.0, "y": 0.0, "z": 0.0},
        t1={"x": 1000.0, "y": 0.0, "z": 0.0},
        t3={"x": 2000.0, "y": 0.0, "z": 0.0},
        stations=stations,
        summary={
            "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
            "trajectory_target_direction": "Цели в одном направлении",
            "well_complexity": "Обычная",
            "optimization_mode": "minimize_md",
            "azimuth_turn_deg": 0.0,
            "horizontal_length_m": 1000.0,
            "entry_inc_deg": 90.0,
            "hold_inc_deg": 90.0,
            "build_dls_selected_deg_per_30m": 3.0,
            "build1_dls_selected_deg_per_30m": 3.0,
            "build2_dls_selected_deg_per_30m": 3.0,
            "max_dls_total_deg_per_30m": 3.0,
            "kop_md_m": 560.0,
            "max_inc_actual_deg": 90.0,
            "max_inc_deg": 95.0,
            "md_total_m": 2000.0,
            "max_total_md_postcheck_m": 6500.0,
            "md_postcheck_excess_m": 0.0,
        },
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )


def test_welltrack_defaults_recover_from_legacy_keys() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    defaults = calc_param_defaults()
    for key, value in LEGACY_MIN_VALUES.items():
        at.session_state[key] = value
    at.session_state["wt_cfg___calc_param_defaults_signature__"] = tuple(
        (key, defaults[key]) for key in sorted(defaults.keys())
    )
    at.session_state["wt_cfg___calc_param_defaults_schema_version__"] = 4

    at.run()
    import_buttons = [button for button in at.button if button.label == "Импорт целей"]
    assert import_buttons, "Кнопка импорта целей не найдена."
    import_buttons[0].click()
    at.run()

    label_to_suffix = {
        "Шаг MD, м": "md_step",
        "Контрольный шаг MD, м": "md_control",
        "Допуск по латерали, м": "lateral_tol",
        "Допуск по вертикали, м": "vertical_tol",
        "Целевой INC на t1, deg": "entry_inc_target",
        "Допуск INC на t1, deg": "entry_inc_tol",
        "Макс INC по стволу, deg": "max_inc",
        "Макс ПИ BUILD, deg/10m": "dls_build_max",
        "Макс ПИ HORIZONTAL, deg/10m": "dls_horizontal_max",
        "Мин VERTICAL до KOP, м": "kop_min_vertical",
        "Макс итоговая MD (постпроверка), м": "max_total_md_postcheck",
        "Макс рестартов решателя": "turn_solver_max_restarts",
    }
    for label, suffix in label_to_suffix.items():
        actual = _number_input_value(at, label)
        assert actual is not None, f"Поле '{label}' не найдено."
        expected = float(defaults[suffix])
        assert abs(float(actual) - expected) < 1e-9, (
            f"Для '{label}' ожидалось {expected}, получено {actual}."
        )

    optimization_actual = _selectbox_value(at, "Оптимизация")
    assert optimization_actual is not None, "Поле оптимизации не найдено."
    assert optimization_actual == str(defaults["optimization_mode"])

    turn_solver_actual = _selectbox_value(at, "Метод решателя")
    assert turn_solver_actual is not None, "Поле метода решателя не найдено."
    assert turn_solver_actual == str(defaults["turn_solver_mode"])

    for key in LEGACY_MIN_VALUES:
        assert key not in at.session_state, f"Legacy-ключ не удален: {key}"


def test_ptc_calc_param_edit_persists_without_form_submit() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.run()
    import_buttons = [button for button in at.button if button.label == "Импорт целей"]
    assert import_buttons, "Кнопка импорта целей не найдена."
    import_buttons[0].click()
    at.run()

    md_step_inputs = [
        widget for widget in at.number_input if widget.label == "Шаг MD, м"
    ]
    assert md_step_inputs, "Поле 'Шаг MD, м' не найдено."
    expected_md_step = float(calc_param_defaults()["md_step"]) + 1.0
    md_step_inputs[0].set_value(expected_md_step)
    at.run()

    assert float(at.session_state["wt_cfg_md_step"]) == expected_md_step


def test_ptc_calc_param_edit_persists_with_results_loaded() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3}
    ]
    at.session_state["wt_successes"] = [_successful_plan()]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    md_step_inputs = [
        widget for widget in at.number_input if widget.label == "Шаг MD, м"
    ]
    assert md_step_inputs, "Поле 'Шаг MD, м' не найдено."
    expected_md_step = float(calc_param_defaults()["md_step"]) + 1.0
    md_step_inputs[0].set_value(expected_md_step)
    at.run(timeout=120)

    assert float(at.session_state["wt_cfg_md_step"]) == expected_md_step
