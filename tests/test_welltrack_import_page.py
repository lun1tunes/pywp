from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord

pytestmark = pytest.mark.integration


def _records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-C",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=700.0, y=760.0, z=2200.0, md=2300.0),
                WelltrackPoint(x=1600.0, y=1960.0, z=2350.0, md=3350.0),
            ),
        ),
    ]


def _multiselect_value(at: AppTest, label: str) -> list[str] | None:
    matches = [widget for widget in at.multiselect if widget.label == label]
    if not matches:
        return None
    return [str(item) for item in matches[0].value]


def test_welltrack_page_shows_only_general_run_before_results() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run()

    assert _multiselect_value(at, "Скважины для расчета") == [
        "WELL-A",
        "WELL-B",
        "WELL-C",
    ]
    assert _multiselect_value(at, "Скважины для повторного расчета") is None


def test_welltrack_page_focuses_follow_up_selection_on_unresolved_wells() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {
            "Скважина": "WELL-A",
            "Точек": 3,
            "Статус": "OK",
            "Модель траектории": "Unified J Profile + Build + Azimuth Turn",
            "Классификация целей": "—",
            "Сложность": "—",
            "Горизонтальный отход t1, м": "—",
            "Длина HORIZONTAL, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс ПИ, deg/10m": "—",
            "Макс MD, м": "—",
            "Проблема": "",
        },
        {
            "Скважина": "WELL-B",
            "Точек": 3,
            "Статус": "Ошибка расчета",
            "Модель траектории": "—",
            "Классификация целей": "—",
            "Сложность": "—",
            "Горизонтальный отход t1, м": "—",
            "Длина HORIZONTAL, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс ПИ, deg/10m": "—",
            "Макс MD, м": "—",
            "Проблема": "Solver endpoint miss to t1.",
        },
    ]
    at.session_state["wt_selected_names"] = []
    at.session_state["wt_retry_selected_names"] = []

    at.run()

    expected = ["WELL-B", "WELL-C"]
    assert _multiselect_value(at, "Скважины для расчета") == expected
    assert _multiselect_value(at, "Скважины для повторного расчета") == expected
    checkbox_labels = [widget.label for widget in at.checkbox]
    assert "Использовать отдельные параметры для повторного расчета" in checkbox_labels
