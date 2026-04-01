from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import TrajectoryConfig
from pywp.reference_trajectories import parse_reference_trajectory_table
from pywp.welltrack_batch import SuccessfulWellPlan


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
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=620.0, y=820.0, z=2410.0, md=2410.0),
                WelltrackPoint(x=1520.0, y=2020.0, z=2510.0, md=3510.0),
            ),
        ),
    ]


def _successful_plan(*, name: str, y_offset_m: float) -> SuccessfulWellPlan:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 90.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [0.0, 1000.0, 2000.0],
            "Y_m": [y_offset_m, y_offset_m, y_offset_m],
            "Z_m": [0.0, 0.0, 0.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0],
            "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
        }
    )
    return SuccessfulWellPlan(
        name=name,
        surface={"x": 0.0, "y": y_offset_m, "z": 0.0},
        t1={"x": 1000.0, "y": y_offset_m, "z": 0.0},
        t3={"x": 2000.0, "y": y_offset_m, "z": 0.0},
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


def _reference_wells():
    return parse_reference_trajectory_table(
        [
            {
                "Wellname": "FACT-001",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "FACT-001",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 1200.0,
                "MD": 1200.0,
            },
            {
                "Wellname": "FACT-001",
                "Type": "actual",
                "X": 600.0,
                "Y": 0.0,
                "Z": 1300.0,
                "MD": 1900.0,
            },
            {
                "Wellname": "APP-001",
                "Type": "approved",
                "X": 30.0,
                "Y": 50.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "APP-001",
                "Type": "approved",
                "X": 30.0,
                "Y": 50.0,
                "Z": 1250.0,
                "MD": 1250.0,
            },
            {
                "Wellname": "APP-001",
                "Type": "approved",
                "X": 700.0,
                "Y": 80.0,
                "Z": 1360.0,
                "MD": 2050.0,
            },
        ]
    )


def test_ptc_page_shows_user_facing_import_and_run_controls() -> None:
    at = AppTest.from_file("pages/03_ptc.py")
    at.run()

    button_labels = {str(widget.label) for widget in at.button}
    assert "Импорт целей" in button_labels
    assert "Очистить импорт" not in button_labels


def test_ptc_page_hides_engineering_result_controls_and_single_well_debug_sections() -> None:
    at = AppTest.from_file("pages/03_ptc.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Траектории"

    at.run()

    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    button_labels = {str(widget.label) for widget in at.button}
    radio_labels = {str(widget.label) for widget in at.radio}
    assert "3D-режим отображения" not in selectbox_labels
    assert "3D backend" not in selectbox_labels
    assert "Пересоздать 3D viewer" not in button_labels
    assert "Режим отображения всех скважин" not in radio_labels

    view_mode_radio = next(
        widget for widget in at.radio if str(widget.label) == "Режим просмотра результатов"
    )
    view_mode_radio.set_value("Отдельная скважина")
    at.run()

    expander_labels = {str(widget.label) for widget in at.expander}
    assert "Контроль попадания и точность расчета" not in expander_labels
    assert "Технические параметры и диагностика решателя" not in expander_labels


def test_ptc_page_wraps_reference_well_table_into_expander() -> None:
    at = AppTest.from_file("pages/03_ptc.py")
    records = _records()
    reference_wells = _reference_wells()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_reference_actual_wells"] = [
        well for well in reference_wells if well.kind == "actual"
    ]
    at.session_state["wt_reference_approved_wells"] = [
        well for well in reference_wells if well.kind == "approved"
    ]

    at.run()

    expander_labels = {str(widget.label) for widget in at.expander}
    assert "Список загруженных фактических/ проектных скважин" in expander_labels
