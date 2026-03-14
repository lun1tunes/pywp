from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from pywp.ui_calc_params import calc_param_defaults

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


def test_welltrack_defaults_recover_from_legacy_keys() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    defaults = calc_param_defaults()
    for key, value in LEGACY_MIN_VALUES.items():
        at.session_state[key] = value
    at.session_state["wt_cfg___calc_param_defaults_signature__"] = tuple(
        (key, defaults[key]) for key in sorted(defaults.keys())
    )
    at.session_state["wt_cfg___calc_param_defaults_schema_version__"] = 4

    at.run()
    parse_buttons = [button for button in at.button if button.label == "Прочитать WELLTRACK"]
    assert parse_buttons, "Кнопка парсинга WELLTRACK не найдена."
    parse_buttons[0].click()
    at.run()

    label_to_suffix = {
        "Шаг MD, м": "md_step",
        "Контрольный шаг MD, м": "md_control",
        "Допуск по позиции, м": "pos_tol",
        "Целевой INC на t1, deg": "entry_inc_target",
        "Допуск INC на t1, deg": "entry_inc_tol",
        "Макс INC по стволу, deg": "max_inc",
        "Макс ПИ BUILD, deg/10m": "dls_build_max",
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

    turn_solver_actual = _selectbox_value(at, "Метод решателя")
    assert turn_solver_actual is not None, "Поле метода решателя не найдено."
    assert turn_solver_actual == str(defaults["turn_solver_mode"])

    for key in LEGACY_MIN_VALUES:
        assert key not in at.session_state, f"Legacy-ключ не удален: {key}"
