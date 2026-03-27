from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd
import pytest
import sys
from streamlit.testing.v1 import AppTest

from pywp.anticollision import AntiCollisionAnalysis, AntiCollisionCorridor, build_anti_collision_well
from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
)
from pywp.anticollision_recommendations import (
    build_anti_collision_recommendation_clusters,
    build_anti_collision_recommendations,
)
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import Point3D, TrajectoryConfig
from pywp.plotly_config import DEFAULT_3D_CAMERA
from pywp.reference_trajectories import parse_reference_trajectory_table
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND,
    planning_uncertainty_model_for_preset,
)
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


def _click_button(at: AppTest, label: str) -> None:
    for widget in at.button:
        if widget.label == label:
            widget.click()
            return
    raise AssertionError(f"Button not found: {label}")


def _surface_points(records: list[WelltrackRecord]) -> list[tuple[float, float, float]]:
    return [
        (float(record.points[0].x), float(record.points[0].y), float(record.points[0].z))
        for record in records
        if record.points
    ]


def _load_welltrack_page_module():
    spec = importlib.util.spec_from_file_location(
        "wt_import_page_test_module",
        "pages/02_welltrack_import.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _successful_plan(
    *,
    name: str,
    y_offset_m: float,
    kop_md_m: float = 0.0,
    build1_dls_deg_per_30m: float = 0.0,
    build_dls_max_deg_per_30m: float = 3.0,
    optimization_mode: str = "none",
    stations: pd.DataFrame | None = None,
) -> SuccessfulWellPlan:
    if stations is None:
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
            "optimization_mode": optimization_mode,
            "azimuth_turn_deg": 0.0,
            "horizontal_length_m": 1000.0,
            "entry_inc_deg": 90.0,
            "hold_inc_deg": 90.0,
            "build_dls_selected_deg_per_30m": 0.0,
            "build1_dls_selected_deg_per_30m": build1_dls_deg_per_30m,
            "max_dls_total_deg_per_30m": 0.0,
            "kop_md_m": kop_md_m,
            "max_inc_actual_deg": 90.0,
            "max_inc_deg": 95.0,
            "md_total_m": 2000.0,
            "max_total_md_postcheck_m": 6500.0,
            "md_postcheck_excess_m": 0.0,
        },
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(
            optimization_mode=optimization_mode,
            dls_build_max_deg_per_30m=build_dls_max_deg_per_30m,
        ),
    )


def _vertical_successful_plan(
    *,
    name: str,
    y_offset_m: float,
    kop_md_m: float,
    lateral_y_t1_m: float,
    lateral_y_end_m: float,
) -> SuccessfulWellPlan:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "INC_deg": [0.0, 0.0, 0.0, 20.0, 55.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "X_m": [0.0, 0.0, 0.0, 60.0, 340.0],
            "Y_m": [y_offset_m, y_offset_m, y_offset_m, lateral_y_t1_m, lateral_y_end_m],
            "Z_m": [0.0, 500.0, 1000.0, 1450.0, 1750.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0, 3.0, 3.0],
            "segment": ["VERTICAL", "VERTICAL", "VERTICAL", "BUILD1", "BUILD1"],
        }
    )
    return _successful_plan(
        name=name,
        y_offset_m=y_offset_m,
        kop_md_m=kop_md_m,
        stations=stations,
    )


def _reference_wells():
    return tuple(
        parse_reference_trajectory_table(
            [
                {"Wellname": "FACT-1", "Type": "actual", "X": 0.0, "Y": 25.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "FACT-1", "Type": "actual", "X": 900.0, "Y": 25.0, "Z": 300.0, "MD": 950.0},
                {"Wellname": "FACT-1", "Type": "actual", "X": 1800.0, "Y": 25.0, "Z": 400.0, "MD": 1900.0},
                {"Wellname": "APP-1", "Type": "approved", "X": 0.0, "Y": -35.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "APP-1", "Type": "approved", "X": 850.0, "Y": -35.0, "Z": 250.0, "MD": 900.0},
                {"Wellname": "APP-1", "Type": "approved", "X": 1750.0, "Y": -35.0, "Z": 350.0, "MD": 1850.0},
            ]
        )
    )


def _far_reference_wells():
    return tuple(
        parse_reference_trajectory_table(
            [
                {"Wellname": "FACT-FAR", "Type": "actual", "X": 50000.0, "Y": 50000.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "FACT-FAR", "Type": "actual", "X": 52000.0, "Y": 50000.0, "Z": 300.0, "MD": 2100.0},
                {"Wellname": "APP-FAR", "Type": "approved", "X": -45000.0, "Y": -42000.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "APP-FAR", "Type": "approved", "X": -43000.0, "Y": -42000.0, "Z": 280.0, "MD": 2050.0},
            ]
        )
    )


def _horizontal_reference_well():
    return tuple(
        parse_reference_trajectory_table(
            [
                {"Wellname": "FACT-H", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "FACT-H", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 1000.0, "MD": 1000.0},
                {"Wellname": "FACT-H", "Type": "actual", "X": 100.0, "Y": 0.0, "Z": 1100.0, "MD": 1600.0},
                {"Wellname": "FACT-H", "Type": "actual", "X": 700.0, "Y": 0.0, "Z": 1150.0, "MD": 2200.0},
                {"Wellname": "FACT-H", "Type": "actual", "X": 1300.0, "Y": 0.0, "Z": 1200.0, "MD": 2800.0},
            ]
        )
    )


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


def test_welltrack_general_run_select_all_restores_full_selection() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_selected_names"] = ["WELL-A"]

    at.run()
    assert _multiselect_value(at, "Скважины для расчета") == ["WELL-A"]

    _click_button(at, "Выбрать все")
    at.run()

    assert _multiselect_value(at, "Скважины для расчета") == [
        "WELL-A",
        "WELL-B",
        "WELL-C",
    ]


def test_well_color_palette_is_large_unique_and_locally_contrasting() -> None:
    page = _load_welltrack_page_module()

    palette = tuple(str(color) for color in page.WELL_COLOR_PALETTE)
    assert len(palette) >= 50
    assert len(set(palette)) == len(palette)

    def _rgb_triplet(value: str) -> tuple[int, int, int]:
        normalized = value.lstrip("#")
        return (
            int(normalized[0:2], 16),
            int(normalized[2:4], 16),
            int(normalized[4:6], 16),
        )

    adjacent_distances = []
    for index in range(len(palette) - 1):
        red_a, green_a, blue_a = _rgb_triplet(palette[index])
        red_b, green_b, blue_b = _rgb_triplet(palette[index + 1])
        adjacent_distances.append(
            abs(red_a - red_b) + abs(green_a - green_b) + abs(blue_a - blue_b)
        )
    assert min(adjacent_distances) >= 120


def test_reference_trajectory_text_import_populates_reference_wells_state() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_reference_actual_source_mode"] = "Вставить XYZ/MD текст"
    at.session_state["wt_reference_actual_source_text"] = "\n".join(
        [
            "Wellname X Y Z MD",
            "FACT-1 0 25 0 0",
            "FACT-1 900 25 300 950",
            "FACT-1 1800 25 400 1900",
        ]
    )
    at.session_state["wt_reference_approved_source_mode"] = "Вставить XYZ/MD текст"
    at.session_state["wt_reference_approved_source_text"] = "\n".join(
        [
            "Wellname X Y Z MD",
            "APP-1 0 -35 0 0",
            "APP-1 850 -35 250 900",
            "APP-1 1750 -35 350 1850",
        ]
    )

    at.run()
    _click_button(at, "Импортировать фактические скважины")
    at.run()
    _click_button(at, "Импортировать проектные утвержденные скважины")
    at.run()

    reference_wells = tuple(at.session_state["wt_reference_wells"])
    assert len(reference_wells) == 2
    assert [str(item.name) for item in reference_wells] == ["FACT-1", "APP-1"]


def test_reference_trajectory_welltrack_path_import_populates_reference_wells_state(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    welltrack_path = tmp_path / "approved_wells.inc"
    welltrack_path.write_text(
        "\n".join(
            [
                "WELLTRACK 'APP-1'",
                "0 -35 0 0",
                "850 -35 250 900",
                "1750 -35 350 1850",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    at.session_state["wt_reference_approved_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_approved_welltrack_path"] = str(welltrack_path)

    at.run()
    _click_button(at, "Импортировать проектные утвержденные скважины")
    at.run(timeout=120)

    approved_wells = tuple(at.session_state["wt_reference_approved_wells"])
    assert len(approved_wells) == 1
    assert str(approved_wells[0].name) == "APP-1"


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

    at.run()

    expected = ["WELL-B", "WELL-C"]
    assert _multiselect_value(at, "Скважины для расчета") == expected
    assert [widget.label for widget in at.metric].count("Без замечаний") == 1
    assert [widget.label for widget in at.metric].count("С предупреждениями") == 1
    checkbox_labels = [widget.label for widget in at.checkbox]
    assert "Использовать отдельные параметры для повторного расчета" not in checkbox_labels


def test_welltrack_successful_batch_run_clears_stale_error_and_updates_selection_on_next_run() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_last_error"] = "Batch-расчет завершился ошибкой"

    at.run()
    _click_button(at, "Запустить / пересчитать выбранные скважины")
    at.run(timeout=120)

    assert [error.value for error in at.error] == []
    assert at.session_state["wt_last_error"] == ""

    metrics = {widget.label: widget.value for widget in at.metric}
    assert metrics["Без замечаний"] == "3"
    assert metrics["Ошибки"] == "0"

    at.run()
    assert _multiselect_value(at, "Скважины для расчета") == []


def test_welltrack_page_keeps_single_general_run_form_after_results_exist() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {
            "Скважина": "WELL-A",
            "Точек": 3,
            "Статус": "OK",
            "Рестарты решателя": "0",
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
    ]

    at.run()

    multiselect_labels = [widget.label for widget in at.multiselect]
    assert multiselect_labels.count("Скважины для расчета") == 1
    assert "Скважины для повторного расчета" not in multiselect_labels


def test_welltrack_import_auto_applies_pad_layout_for_shared_surface_and_can_reset() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    at.session_state["wt_source_mode"] = "Файл по пути"
    at.session_state["wt_source_path"] = "tests/test_data/WELLTRACKS2.INC"

    at.run(timeout=120)
    _click_button(at, "Прочитать WELLTRACK")
    at.run(timeout=120)

    original_records = at.session_state["wt_records_original"]
    current_records = at.session_state["wt_records"]
    original_surfaces = _surface_points(original_records)
    current_surfaces = _surface_points(current_records)

    assert len(original_surfaces) > 1
    assert len(set(original_surfaces)) == 1
    assert len(set(current_surfaces)) > 1
    assert at.session_state["wt_pad_last_applied_at"] != ""
    assert at.session_state["wt_pad_auto_applied_on_import"] is True
    assert any(
        "автоматически скорректированы" in str(widget.value).lower()
        for widget in at.info
    )

    _click_button(at, "Вернуть исходные устья")
    at.run(timeout=120)

    reset_surfaces = _surface_points(at.session_state["wt_records"])
    assert reset_surfaces == original_surfaces
    assert at.session_state["wt_pad_last_applied_at"] == ""
    assert at.session_state["wt_pad_auto_applied_on_import"] is False


def test_welltrack_import_accepts_tabular_point_editor_mode() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    at.session_state["wt_source_mode"] = "Вставить таблицу"
    at.session_state["wt_source_table_df"] = pd.DataFrame(
        [
            {"Wellname": "TAB-01", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "TAB-01", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {"Wellname": "TAB-01", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
            {"Wellname": "TAB-02", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "TAB-02", "Point": "t1", "X": 650.0, "Y": 780.0, "Z": 2300.0},
            {"Wellname": "TAB-02", "Point": "t3", "X": 1550.0, "Y": 1980.0, "Z": 2400.0},
        ]
    )

    at.run(timeout=120)
    _click_button(at, "Прочитать WELLTRACK")
    at.run(timeout=120)

    records = at.session_state["wt_records"]
    assert [record.name for record in records] == ["TAB-01", "TAB-02"]
    assert records[0].points[1].x == pytest.approx(600.0)
    assert records[1].points[2].y == pytest.approx(1980.0)


def test_welltrack_page_renders_anticollision_metrics_for_successful_batch() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()[:2]
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
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    metric_labels = [widget.label for widget in at.metric]
    assert "Проверено пар" in metric_labels
    assert "Минимальный SF" in metric_labels
    selectbox_labels = [widget.label for widget in at.selectbox]
    assert "Пресет неопределенности для anti-collision" in selectbox_labels


def test_welltrack_page_normalizes_invalid_anticollision_uncertainty_preset() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()[:2]
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
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    at.session_state["wt_anticollision_uncertainty_preset"] = "invalid_preset"
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    preset_widgets = [
        widget
        for widget in at.selectbox
        if widget.label == "Пресет неопределенности для anti-collision"
    ]
    assert preset_widgets
    assert str(preset_widgets[0].value) == DEFAULT_UNCERTAINTY_PRESET
    assert (
        str(at.session_state["wt_anticollision_uncertainty_preset"])
        == DEFAULT_UNCERTAINTY_PRESET
    )


def test_welltrack_page_shows_custom_actual_fund_uncertainty_preset_when_available() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()[:2]
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "OK"},
        {"Скважина": "WELL-B", "Статус": "OK"},
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    at.session_state["wt_actual_fund_custom_model"] = planning_uncertainty_model_for_preset(
        DEFAULT_UNCERTAINTY_PRESET
    )
    at.session_state["wt_anticollision_uncertainty_preset"] = (
        UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND
    )
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    preset_widgets = [
        widget
        for widget in at.selectbox
        if widget.label == "Пресет неопределенности для anti-collision"
    ]
    assert preset_widgets
    assert str(preset_widgets[0].value) == UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND
    assert "Пользовательская (по фактическому фонду)" in list(preset_widgets[0].options)


def test_focus_all_wells_anticollision_results_sets_result_view_state() -> None:
    page = _load_welltrack_page_module()

    page.st.session_state.clear()
    page._init_state()
    page._focus_all_wells_anticollision_results()

    assert page.st.session_state["wt_results_view_mode"] == "Все скважины"
    assert page.st.session_state["wt_results_all_view_mode"] == "Anti-collision"


def test_welltrack_page_respects_anticollision_result_focus_state() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {
            "Скважина": "WELL-A",
            "Точек": 3,
            "Статус": "OK",
            "Рестарты решателя": "0",
            "Модель траектории": "Unified J Profile + Build + Azimuth Turn",
            "Классификация целей": "В прямом направлении",
            "Сложность": "Обычная",
            "Горизонтальный отход t1, м": "1000.00",
            "KOP MD, м": "700.00",
            "Длина HORIZONTAL, м": "1000.00",
            "INC в t1, deg": "90.00",
            "ЗУ HOLD, deg": "90.00",
            "Макс ПИ, deg/10m": "0.00",
            "Макс MD, м": "2000.00",
            "Проблема": "",
        }
    ]
    at.session_state["wt_successes"] = [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run()

    result_mode = next(
        widget for widget in at.radio if widget.label == "Режим просмотра результатов"
    )
    all_wells_mode = next(
        widget for widget in at.radio if widget.label == "Режим отображения всех скважин"
    )
    assert str(result_mode.value) == "Все скважины"
    assert str(all_wells_mode.value) == "Anti-collision"


def test_welltrack_page_prepares_vertical_anticollision_rerun_plan() -> None:
    at = AppTest.from_file("pages/02_welltrack_import.py")
    records = _records()[:2]
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {
            "Скважина": "WELL-A",
            "Точек": 3,
            "Статус": "OK",
            "Рестарты решателя": "0",
            "Модель траектории": "Unified J Profile + Build + Azimuth Turn",
            "Классификация целей": "В прямом направлении",
            "Сложность": "Обычная",
            "Горизонтальный отход t1, м": "100.00",
            "KOP MD, м": "820.00",
            "Длина HORIZONTAL, м": "1000.00",
            "INC в t1, deg": "86.00",
            "ЗУ HOLD, deg": "45.00",
            "Макс ПИ, deg/10m": "1.00",
            "Макс MD, м": "4300.00",
            "Проблема": "",
        },
        {
            "Скважина": "WELL-B",
            "Точек": 3,
            "Статус": "OK",
            "Рестарты решателя": "0",
            "Модель траектории": "Unified J Profile + Build + Azimuth Turn",
            "Классификация целей": "В прямом направлении",
            "Сложность": "Обычная",
            "Горизонтальный отход t1, м": "100.00",
            "KOP MD, м": "780.00",
            "Длина HORIZONTAL, м": "1000.00",
            "INC в t1, deg": "86.00",
            "ЗУ HOLD, deg": "45.00",
            "Макс ПИ, deg/10m": "1.00",
            "Макс MD, м": "4300.00",
            "Проблема": "",
        },
    ]
    at.session_state["wt_successes"] = [
        _vertical_successful_plan(
            name="WELL-A",
            y_offset_m=0.0,
            kop_md_m=820.0,
            lateral_y_t1_m=80.0,
            lateral_y_end_m=320.0,
        ),
        _vertical_successful_plan(
            name="WELL-B",
            y_offset_m=5.0,
            kop_md_m=780.0,
            lateral_y_t1_m=150.0,
            lateral_y_end_m=380.0,
        ),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    selectbox_labels = [widget.label for widget in at.selectbox]
    assert "Подготовить пересчет по одному anti-collision событию" in selectbox_labels

    _click_button(at, "Подготовить одно событие")
    at.run(timeout=120)

    assert _multiselect_value(at, "Скважины для расчета") == ["WELL-A", "WELL-B"]
    override_message = at.session_state["wt_prepared_override_message"]
    assert "kop/build1" in str(override_message).lower()
    prepared = dict(at.session_state["wt_prepared_well_overrides"])
    assert set(prepared.keys()) == {"WELL-A", "WELL-B"}
    assert all(
        dict(value).get("update_fields", {}).get("optimization_mode")
        == "anti_collision_avoidance"
        for value in prepared.values()
    )


def test_build_prepared_trajectory_optimization_context_uses_pair_intervals() -> None:
    page = _load_welltrack_page_module()
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1100.0,
                md_a_end_m=1700.0,
                md_b_start_m=1200.0,
                md_b_end_m=1750.0,
                md_a_values_m=np.array([1100.0, 1700.0], dtype=float),
                md_b_values_m=np.array([1200.0, 1750.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.72, 0.69], dtype=float),
                overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0, kop_md_m=700.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0, kop_md_m=900.0),
    ]
    recommendation = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )[0]

    context = page._build_prepared_optimization_context(
        recommendation=recommendation,
        moving_success=successes[1],
        reference_success=successes[0],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        all_successes=successes,
    )

    assert context is not None
    assert context.candidate_md_start_m == pytest.approx(1200.0)
    assert context.candidate_md_end_m == pytest.approx(1750.0)
    assert context.references[0].well_name == "WELL-A"
    assert context.references[0].md_start_m == pytest.approx(1100.0)
    assert context.references[0].md_end_m == pytest.approx(1700.0)


def test_build_prepared_trajectory_optimization_context_includes_other_well_cones() -> None:
    page = _load_welltrack_page_module()
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1100.0,
                md_a_end_m=1700.0,
                md_b_start_m=1200.0,
                md_b_end_m=1750.0,
                md_a_values_m=np.array([1100.0, 1700.0], dtype=float),
                md_b_values_m=np.array([1200.0, 1750.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.72, 0.69], dtype=float),
                overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0, kop_md_m=700.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0, kop_md_m=900.0),
        _successful_plan(name="WELL-C", y_offset_m=120.0, kop_md_m=800.0),
    ]
    recommendation = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )[0]

    context = page._build_prepared_optimization_context(
        recommendation=recommendation,
        moving_success=successes[1],
        reference_success=successes[0],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        all_successes=successes,
    )

    assert context is not None
    assert {str(item.well_name) for item in context.references} == {"WELL-A", "WELL-C"}


def test_build_prepared_late_trajectory_context_marks_build2_adjustment() -> None:
    page = _load_welltrack_page_module()
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=4075.0,
                md_a_end_m=4375.0,
                md_b_start_m=4025.0,
                md_b_end_m=4275.0,
                md_a_values_m=np.array([4075.0, 4375.0], dtype=float),
                md_b_values_m=np.array([4025.0, 4275.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.31, 0.29], dtype=float),
                overlap_depth_values_m=np.array([105.0, 111.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.29,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0, kop_md_m=700.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0, kop_md_m=900.0),
    ]
    recommendation = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )[0]

    context = page._build_prepared_optimization_context(
        recommendation=recommendation,
        moving_success=successes[0],
        reference_success=successes[1],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        all_successes=successes,
    )

    assert context is not None
    assert context.prefer_keep_kop is True
    assert context.prefer_keep_build1 is True
    assert context.prefer_adjust_build2 is True


def test_prepared_override_rows_include_sf_before() -> None:
    page = _load_welltrack_page_module()
    page.st.session_state["wt_prepared_well_overrides"] = {
        "WELL-B": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "reason": "Prepared pairwise anti-collision rerun.",
        }
    }
    page.st.session_state["wt_prepared_recommendation_snapshot"] = {
        "before_sf": 0.69,
        "affected_wells": ("WELL-B",),
        "expected_maneuver": "Pre-entry azimuth turn / сдвиг HOLD до t1",
    }

    rows = page._prepared_override_rows()

    assert rows == [
        {
            "Порядок": "1",
            "Скважина": "WELL-B",
            "Маневр": "Pre-entry azimuth turn / сдвиг HOLD до t1",
            "Оптимизация": "Anti-collision avoidance",
            "SF до": "0.69",
            "Источник": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "Причина": "Prepared pairwise anti-collision rerun.",
        }
    ]


def test_format_prepared_override_scope_shows_local_mode_for_selected_wells() -> None:
    page = _load_welltrack_page_module()
    page.st.session_state["wt_prepared_well_overrides"] = {
        "WELL-B": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "reason": "Prepared pairwise anti-collision rerun.",
        },
        "WELL-C": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "ac-cluster-001 · WELL-A, WELL-B, WELL-C · событий 2 · SF 0.71",
            "reason": "Early collision: cone-aware rerun по KOP/BUILD1.",
        },
    }
    page.st.session_state["wt_prepared_recommendation_snapshot"] = {
        "kind": "cluster",
        "action_steps": (
            {
                "order_rank": 1,
                "well_name": "WELL-B",
                "expected_maneuver": "Сместить post-entry / HORIZONTAL",
            },
            {
                "order_rank": 2,
                "well_name": "WELL-C",
                "expected_maneuver": "Сместить ранний уход: KOP / BUILD1",
            },
        ),
    }

    rows = page._format_prepared_override_scope(selected_names=["WELL-A", "WELL-B", "WELL-C"])

    assert rows == [
        {
            "Скважина": "WELL-B",
            "Локальный режим": "Anti-collision avoidance",
            "Источник": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "Маневр": "Сместить post-entry / HORIZONTAL",
        },
        {
            "Скважина": "WELL-C",
            "Локальный режим": "Anti-collision avoidance",
            "Источник": "ac-cluster-001 · WELL-A, WELL-B, WELL-C · событий 2 · SF 0.71",
            "Маневр": "Сместить ранний уход: KOP / BUILD1",
        },
    ]


def test_sf_help_markdown_explains_thresholds() -> None:
    page = _load_welltrack_page_module()

    text = page._sf_help_markdown()

    assert "Separation Factor" in text
    assert "SF < 1" in text
    assert "SF > 1" in text


def test_prepared_override_rows_follow_cluster_action_step_order() -> None:
    page = _load_welltrack_page_module()
    page.st.session_state["wt_prepared_well_overrides"] = {
        "WELL-C": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "ac-cluster-001 · WELL-A, WELL-B, WELL-C · событий 2 · SF 0.71",
            "reason": "Trajectory collision against WELL-A.",
        },
        "WELL-B": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "ac-cluster-001 · WELL-A, WELL-B, WELL-C · событий 2 · SF 0.71",
            "reason": "Early collision: cone-aware rerun по KOP/BUILD1.",
        },
    }
    page.st.session_state["wt_prepared_recommendation_snapshot"] = {
        "kind": "cluster",
        "before_sf": 0.71,
        "action_steps": (
            {
                "order_rank": 1,
                "well_name": "WELL-C",
                "expected_maneuver": "Сместить post-entry / HORIZONTAL",
            },
            {
                "order_rank": 2,
                "well_name": "WELL-B",
                "expected_maneuver": "Сместить ранний уход: KOP / BUILD1",
            },
        ),
    }

    rows = page._prepared_override_rows()

    assert [row["Скважина"] for row in rows] == ["WELL-C", "WELL-B"]
    assert [row["Порядок"] for row in rows] == ["1", "2"]
    assert [row["Маневр"] for row in rows] == [
        "Сместить post-entry / HORIZONTAL",
        "Сместить ранний уход: KOP / BUILD1",
    ]


def test_build_last_anticollision_resolution_reports_sf_after() -> None:
    page = _load_welltrack_page_module()
    resolution = page._build_last_anticollision_resolution(
        snapshot={
            "well_a": "WELL-A",
            "well_b": "WELL-B",
            "source_label": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "action_label": "Подготовить anti-collision пересчет",
            "area_label": "траектория ↔ траектория",
            "before_sf": 0.69,
            "before_overlap_m": 8.0,
            "md_a_start_m": 1000.0,
            "md_a_end_m": 2000.0,
            "md_b_start_m": 1000.0,
            "md_b_end_m": 2000.0,
        },
        successes=[
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=400.0),
        ],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        uncertainty_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert resolution is not None
    assert float(resolution["after_sf"]) > 1.0
    assert float(resolution["delta_sf"]) > 0.0
    assert str(resolution["status"]) == "Конфликт снят"
    assert str(resolution["uncertainty_preset"]) == DEFAULT_UNCERTAINTY_PRESET


def test_cached_anti_collision_view_model_reuses_analysis_for_identical_inputs(monkeypatch) -> None:
    page = _load_welltrack_page_module()
    calls = {"analysis": 0}
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=0,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )
    monkeypatch.setattr(
        page,
        "_build_anti_collision_analysis",
        lambda successes, *, model, name_to_color=None, reference_wells=(): (
            calls.__setitem__("analysis", calls["analysis"] + 1) or analysis
        ),
    )
    monkeypatch.setattr(
        page,
        "build_anti_collision_recommendations",
        lambda analysis, *, well_context_by_name: (),
    )
    monkeypatch.setattr(
        page,
        "build_anti_collision_recommendation_clusters",
        lambda recommendations: (),
    )

    first = page._cached_anti_collision_view_model(
        successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        records=[],
    )
    second = page._cached_anti_collision_view_model(
        successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        records=[],
    )

    assert first[0] is analysis
    assert second[0] is analysis
    assert calls["analysis"] == 1


def test_prepare_rerun_from_cluster_builds_multi_reference_cluster_plan() -> None:
    page = _load_welltrack_page_module()
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-C",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1200.0,
                md_a_end_m=1700.0,
                md_b_start_m=1250.0,
                md_b_end_m=1750.0,
                md_a_values_m=np.array([1200.0, 1700.0], dtype=float),
                md_b_values_m=np.array([1250.0, 1750.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.78, 0.74], dtype=float),
                overlap_depth_values_m=np.array([7.0, 8.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="WELL-B",
                well_b="WELL-C",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1850.0,
                md_a_end_m=2300.0,
                md_b_start_m=1800.0,
                md_b_end_m=2250.0,
                md_a_values_m=np.array([1850.0, 2300.0], dtype=float),
                md_b_values_m=np.array([1800.0, 2250.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.76, 0.71], dtype=float),
                overlap_depth_values_m=np.array([9.0, 10.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.71,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0, kop_md_m=650.0),
        _successful_plan(name="WELL-B", y_offset_m=150.0, kop_md_m=700.0),
        _successful_plan(name="WELL-C", y_offset_m=5.0, kop_md_m=900.0),
    ]
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    page._prepare_rerun_from_cluster(
        clusters[0],
        successes=successes,
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    prepared = page.st.session_state["wt_prepared_well_overrides"]
    assert list(prepared.keys()) == ["WELL-C", "WELL-B", "WELL-A"]
    payload = dict(prepared["WELL-C"])
    context = payload["optimization_context"]
    assert context is not None
    assert float(context.candidate_md_start_m) == pytest.approx(1250.0)
    assert float(context.candidate_md_end_m) == pytest.approx(2250.0)
    assert {str(item.well_name) for item in context.references} == {"WELL-A", "WELL-B"}
    assert {
        str(item.well_name) for item in prepared["WELL-B"]["optimization_context"].references
    } == {"WELL-A", "WELL-C"}
    assert {
        str(item.well_name) for item in prepared["WELL-A"]["optimization_context"].references
    } == {"WELL-B", "WELL-C"}
    assert str(payload["source"]).startswith("ac-cluster-")
    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-C", "WELL-B", "WELL-A"]
    snapshot = dict(page.st.session_state["wt_prepared_recommendation_snapshot"])
    assert str(snapshot["kind"]) == "cluster"
    assert len(tuple(snapshot["items"])) == 2


def test_prepare_rerun_from_cluster_is_blocked_for_target_spacing_conflicts() -> None:
    page = _load_welltrack_page_module()
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="target-target",
                priority_rank=1,
                label_a="t1",
                label_b="t1",
                md_a_start_m=2400.0,
                md_a_end_m=2400.0,
                md_b_start_m=2350.0,
                md_b_end_m=2350.0,
                md_a_values_m=np.array([2400.0], dtype=float),
                md_b_values_m=np.array([2350.0], dtype=float),
                label_a_values=("t1",),
                label_b_values=("t1",),
                midpoint_xyz=np.array([[0.0, 0.0, 2400.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([8.0], dtype=float),
                separation_factor_values=np.array([0.63], dtype=float),
                overlap_depth_values_m=np.array([12.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="WELL-B",
                well_b="WELL-C",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1800.0,
                md_a_end_m=2250.0,
                md_b_start_m=1750.0,
                md_b_end_m=2200.0,
                md_a_values_m=np.array([1800.0, 2250.0], dtype=float),
                md_b_values_m=np.array([1750.0, 2200.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.79, 0.74], dtype=float),
                overlap_depth_values_m=np.array([7.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=1,
        worst_separation_factor=0.63,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0, kop_md_m=650.0),
        _successful_plan(name="WELL-B", y_offset_m=150.0, kop_md_m=700.0),
        _successful_plan(name="WELL-C", y_offset_m=5.0, kop_md_m=900.0),
    ]
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    page._prepare_rerun_from_cluster(
        clusters[0],
        successes=successes,
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    assert page.st.session_state["wt_prepared_well_overrides"] == {}
    assert page.st.session_state["wt_prepared_recommendation_snapshot"] is None
    assert page.st.session_state["wt_pending_selected_names"] is None
    assert "Cluster-level пересчет недоступен" in str(
        page.st.session_state["wt_prepared_override_message"]
    )
    assert "spacing целей" in str(page.st.session_state["wt_prepared_override_message"])


def test_prepare_rerun_from_cluster_merges_vertical_and_trajectory_into_mixed_well_plan() -> None:
    page = _load_welltrack_page_module()
    well_a_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "INC_deg": [0.0, 0.0, 0.0, 20.0, 50.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "X_m": [0.0, 0.0, 0.0, 50.0, 320.0],
            "Y_m": [0.0, 0.0, 0.0, 30.0, 120.0],
            "Z_m": [0.0, 500.0, 1000.0, 1450.0, 1775.0],
            "segment": ["VERTICAL", "VERTICAL", "VERTICAL", "BUILD1", "BUILD1"],
        }
    )
    well_b_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "INC_deg": [0.0, 0.0, 0.0, 20.0, 50.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "X_m": [0.0, 0.0, 0.0, 50.0, 320.0],
            "Y_m": [10.0, 10.0, 10.0, 40.0, 150.0],
            "Z_m": [0.0, 500.0, 1000.0, 1450.0, 1775.0],
            "segment": ["VERTICAL", "VERTICAL", "VERTICAL", "BUILD1", "BUILD1"],
        }
    )
    analysis = AntiCollisionAnalysis(
        wells=(
            build_anti_collision_well(
                name="WELL-A",
                color="#0B6E4F",
                stations=well_a_stations,
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(50.0, 30.0, 1450.0),
                t3=Point3D(320.0, 120.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="WELL-B",
                color="#3A86FF",
                stations=well_b_stations,
                surface=Point3D(0.0, 10.0, 0.0),
                t1=Point3D(50.0, 40.0, 1450.0),
                t3=Point3D(320.0, 150.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="WELL-C",
                color="#00798C",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 1000.0, 2000.0],
                        "INC_deg": [0.0, 90.0, 90.0],
                        "AZI_deg": [0.0, 90.0, 90.0],
                        "X_m": [0.0, 1000.0, 2000.0],
                        "Y_m": [0.0, 0.0, 0.0],
                        "Z_m": [0.0, 0.0, 0.0],
                        "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                    }
                ),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(1000.0, 0.0, 0.0),
                t3=Point3D(2000.0, 0.0, 0.0),
                azimuth_deg=90.0,
                md_t1_m=1000.0,
            ),
        ),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=400.0,
                md_a_end_m=700.0,
                md_b_start_m=410.0,
                md_b_end_m=710.0,
                md_a_values_m=np.array([400.0, 700.0], dtype=float),
                md_b_values_m=np.array([410.0, 710.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                separation_factor_values=np.array([0.82, 0.78], dtype=float),
                overlap_depth_values_m=np.array([5.0, 7.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-C",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1750.0,
                md_a_end_m=2050.0,
                md_b_start_m=1700.0,
                md_b_end_m=2000.0,
                md_a_values_m=np.array([1750.0, 2050.0], dtype=float),
                md_b_values_m=np.array([1700.0, 2000.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.76, 0.73], dtype=float),
                overlap_depth_values_m=np.array([7.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.73,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0, kop_md_m=700.0, stations=well_a_stations),
        _successful_plan(name="WELL-B", y_offset_m=10.0, kop_md_m=900.0, stations=well_b_stations),
        _successful_plan(name="WELL-C", y_offset_m=0.0, kop_md_m=650.0),
    ]
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    page._prepare_rerun_from_cluster(
        clusters[0],
        successes=successes,
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    prepared = page.st.session_state["wt_prepared_well_overrides"]
    assert "WELL-A" in prepared
    payload = dict(prepared["WELL-A"])
    context = payload["optimization_context"]
    assert context is not None
    assert bool(context.prefer_lower_kop) is True
    assert bool(context.prefer_higher_build1) is True
    assert str(payload["update_fields"]["optimization_mode"]) == "anti_collision_avoidance"
    assert "cone-aware rerun" in str(payload["reason"])
    assert "anti-collision avoidance rerun" in str(payload["reason"])


def test_prepare_rerun_from_cluster_stages_mixed_well_to_early_kop_build1_first() -> None:
    page = _load_welltrack_page_module()
    well_a_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "INC_deg": [0.0, 0.0, 0.0, 20.0, 50.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "X_m": [0.0, 0.0, 0.0, 50.0, 320.0],
            "Y_m": [0.0, 0.0, 0.0, 30.0, 120.0],
            "Z_m": [0.0, 500.0, 1000.0, 1450.0, 1775.0],
            "segment": ["VERTICAL", "VERTICAL", "VERTICAL", "BUILD1", "BUILD1"],
        }
    )
    well_b_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "INC_deg": [0.0, 0.0, 0.0, 20.0, 50.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "X_m": [0.0, 0.0, 0.0, 50.0, 320.0],
            "Y_m": [10.0, 10.0, 10.0, 40.0, 150.0],
            "Z_m": [0.0, 500.0, 1000.0, 1450.0, 1775.0],
            "segment": ["VERTICAL", "VERTICAL", "VERTICAL", "BUILD1", "BUILD1"],
        }
    )
    analysis = AntiCollisionAnalysis(
        wells=(
            build_anti_collision_well(
                name="WELL-A",
                color="#0B6E4F",
                stations=well_a_stations,
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(50.0, 30.0, 1450.0),
                t3=Point3D(320.0, 120.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="WELL-B",
                color="#3A86FF",
                stations=well_b_stations,
                surface=Point3D(0.0, 10.0, 0.0),
                t1=Point3D(50.0, 40.0, 1450.0),
                t3=Point3D(320.0, 150.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="WELL-C",
                color="#00798C",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 1000.0, 2000.0],
                        "INC_deg": [0.0, 90.0, 90.0],
                        "AZI_deg": [0.0, 90.0, 90.0],
                        "X_m": [0.0, 1000.0, 2000.0],
                        "Y_m": [0.0, 0.0, 0.0],
                        "Z_m": [0.0, 0.0, 0.0],
                        "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                    }
                ),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(1000.0, 0.0, 0.0),
                t3=Point3D(2000.0, 0.0, 0.0),
                azimuth_deg=90.0,
                md_t1_m=1000.0,
            ),
        ),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=400.0,
                md_a_end_m=700.0,
                md_b_start_m=410.0,
                md_b_end_m=710.0,
                md_a_values_m=np.array([400.0, 700.0], dtype=float),
                md_b_values_m=np.array([410.0, 710.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([6.0, 6.0], dtype=float),
                separation_factor_values=np.array([0.60, 0.55], dtype=float),
                overlap_depth_values_m=np.array([6.0, 7.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-C",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1750.0,
                md_a_end_m=2050.0,
                md_b_start_m=1700.0,
                md_b_end_m=2000.0,
                md_a_values_m=np.array([1750.0, 2050.0], dtype=float),
                md_b_values_m=np.array([1700.0, 2000.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                separation_factor_values=np.array([0.82, 0.80], dtype=float),
                overlap_depth_values_m=np.array([4.0, 4.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.55,
    )
    successes = [
        _successful_plan(
            name="WELL-A",
            y_offset_m=0.0,
            kop_md_m=700.0,
            build1_dls_deg_per_30m=2.0,
            build_dls_max_deg_per_30m=3.0,
            stations=well_a_stations,
        ),
        _successful_plan(
            name="WELL-B",
            y_offset_m=10.0,
            kop_md_m=900.0,
            build1_dls_deg_per_30m=2.0,
            build_dls_max_deg_per_30m=3.0,
            stations=well_b_stations,
        ),
        _successful_plan(name="WELL-C", y_offset_m=0.0),
    ]
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    page._prepare_rerun_from_cluster(
        clusters[0],
        successes=successes,
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    payload = dict(page.st.session_state["wt_prepared_well_overrides"]["WELL-A"])
    context = payload["optimization_context"]
    assert context is not None
    assert float(context.candidate_md_start_m) == pytest.approx(400.0)
    assert float(context.candidate_md_end_m) == pytest.approx(700.0)
    reference_by_name = {str(item.well_name): item for item in context.references}
    assert float(reference_by_name["WELL-C"].md_end_m) == pytest.approx(800.0)
    assert "WELL-C" not in str(payload["reason"])


def test_build_selected_optimization_contexts_rebuilds_prepared_context_from_current_successes() -> None:
    page = _load_welltrack_page_module()
    legacy_reference_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 45.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [999.0, 999.0, 999.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 1000.0, 2000.0],
        }
    )
    stale_context = AntiCollisionOptimizationContext(
        candidate_md_start_m=1000.0,
        candidate_md_end_m=2000.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        references=(
            build_anti_collision_reference_path(
                well_name="WELL-A",
                stations=legacy_reference_stations,
                md_start_m=1000.0,
                md_end_m=2000.0,
                sample_step_m=50.0,
                model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
            ),
        ),
    )
    page.st.session_state["wt_prepared_well_overrides"] = {
        "WELL-B": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "prepared",
            "reason": "stale reference",
            "optimization_context": stale_context,
        }
    }

    contexts = page._build_selected_optimization_contexts(
        selected_names={"WELL-B"},
        current_successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
    )

    assert "WELL-B" in contexts
    rebuilt_context = contexts["WELL-B"]
    assert float(rebuilt_context.references[0].xyz_m[-1, 0]) == pytest.approx(2000.0)
    stored_context = page.st.session_state["wt_prepared_well_overrides"]["WELL-B"]["optimization_context"]
    assert isinstance(stored_context, AntiCollisionOptimizationContext)
    assert float(stored_context.references[0].xyz_m[-1, 0]) == pytest.approx(2000.0)


def test_ensure_selected_success_baseline_populates_lazy_reference_and_caches_it(monkeypatch) -> None:
    page = _load_welltrack_page_module()
    optimized = _successful_plan(
        name="WELL-OPT",
        y_offset_m=0.0,
        optimization_mode="minimize_md",
    )
    page.st.session_state["wt_successes"] = [optimized]
    monkeypatch.setattr(
        page,
        "ensure_successful_plan_baseline",
        lambda *, success: success.validated_copy(
            baseline_summary={"md_total_m": 2200.0},
            baseline_runtime_s=0.42,
        ),
    )

    selected = page._ensure_selected_success_baseline(
        selected_name="WELL-OPT",
        successes=[optimized],
    )

    assert selected.baseline_summary is not None
    assert selected.baseline_runtime_s is not None
    cached = page.st.session_state["wt_successes"][0]
    assert cached.baseline_summary is not None
    assert cached.baseline_runtime_s is not None


def test_selected_execution_order_prioritizes_cluster_action_steps() -> None:
    page = _load_welltrack_page_module()
    page.st.session_state["wt_prepared_recommendation_snapshot"] = {
        "kind": "cluster",
        "action_steps": (
            {"order_rank": 1, "well_name": "WELL-C"},
            {"order_rank": 2, "well_name": "WELL-A"},
        ),
    }

    order = page._selected_execution_order(["WELL-A", "WELL-B", "WELL-C"])

    assert order == ["WELL-C", "WELL-A", "WELL-B"]


def test_build_last_anticollision_resolution_supports_cluster_snapshot() -> None:
    page = _load_welltrack_page_module()
    resolution = page._build_last_anticollision_resolution(
        snapshot={
            "kind": "cluster",
            "cluster_id": "ac-cluster-001",
            "source_label": "ac-cluster-001 · WELL-A, WELL-B, WELL-C · событий 2 · SF 0.71",
            "before_sf": 0.71,
            "items": (
                {
                    "kind": "recommendation",
                    "well_a": "WELL-A",
                    "well_b": "WELL-C",
                    "source_label": "WELL-A ↔ WELL-C · Траектория / review · SF 0.74",
                    "action_label": "Подготовить anti-collision пересчет",
                    "expected_maneuver": "Сместить post-entry / HORIZONTAL",
                    "area_label": "траектория ↔ траектория",
                    "before_sf": 0.74,
                    "before_overlap_m": 8.0,
                    "md_a_start_m": 1000.0,
                    "md_a_end_m": 2000.0,
                    "md_b_start_m": 1000.0,
                    "md_b_end_m": 2000.0,
                },
                {
                    "kind": "recommendation",
                    "well_a": "WELL-B",
                    "well_b": "WELL-C",
                    "source_label": "WELL-B ↔ WELL-C · Траектория / review · SF 0.71",
                    "action_label": "Подготовить anti-collision пересчет",
                    "expected_maneuver": "Сместить post-entry / HORIZONTAL",
                    "area_label": "траектория ↔ траектория",
                    "before_sf": 0.71,
                    "before_overlap_m": 10.0,
                    "md_a_start_m": 1000.0,
                    "md_a_end_m": 2000.0,
                    "md_b_start_m": 1000.0,
                    "md_b_end_m": 2000.0,
                },
            ),
        },
        successes=[
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=300.0),
            _successful_plan(name="WELL-C", y_offset_m=600.0),
        ],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        uncertainty_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert resolution is not None
    assert str(resolution["kind"]) == "cluster"
    assert float(resolution["after_sf"]) >= 1.0
    assert str(resolution["status"]) == "Конфликт снят"
    assert int(resolution["current_cluster_count"]) == 0
    assert tuple(resolution["items"]) == ()


def test_build_last_anticollision_resolution_cluster_rescans_current_conflicts_beyond_snapshot_windows() -> None:
    page = _load_welltrack_page_module()
    resolution = page._build_last_anticollision_resolution(
        snapshot={
            "kind": "cluster",
            "cluster_id": "ac-cluster-001",
            "source_label": "ac-cluster-001 · WELL-A, WELL-B, WELL-C · событий 2 · SF 0.71",
            "before_sf": 0.71,
            "well_names": ("WELL-A", "WELL-B", "WELL-C"),
            "items": (
                {
                    "kind": "recommendation",
                    "well_a": "WELL-A",
                    "well_b": "WELL-C",
                    "source_label": "WELL-A ↔ WELL-C · Траектория / review · SF 0.74",
                    "action_label": "Подготовить anti-collision пересчет",
                    "expected_maneuver": "Сместить post-entry / HORIZONTAL",
                    "area_label": "траектория ↔ траектория",
                    "before_sf": 0.74,
                    "before_overlap_m": 8.0,
                    "md_a_start_m": 1000.0,
                    "md_a_end_m": 2000.0,
                    "md_b_start_m": 1000.0,
                    "md_b_end_m": 2000.0,
                },
                {
                    "kind": "recommendation",
                    "well_a": "WELL-B",
                    "well_b": "WELL-C",
                    "source_label": "WELL-B ↔ WELL-C · Траектория / review · SF 0.71",
                    "action_label": "Подготовить anti-collision пересчет",
                    "expected_maneuver": "Сместить post-entry / HORIZONTAL",
                    "area_label": "траектория ↔ траектория",
                    "before_sf": 0.71,
                    "before_overlap_m": 10.0,
                    "md_a_start_m": 1000.0,
                    "md_a_end_m": 2000.0,
                    "md_b_start_m": 1000.0,
                    "md_b_end_m": 2000.0,
                },
            ),
        },
        successes=[
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=300.0),
            _successful_plan(name="WELL-C", y_offset_m=600.0),
            _successful_plan(name="WELL-D", y_offset_m=5.0),
        ],
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        uncertainty_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert resolution is not None
    assert str(resolution["kind"]) == "cluster"
    assert float(resolution["after_sf"]) < 1.0
    assert str(resolution["status"]) != "Конфликт снят"
    assert int(resolution["current_cluster_count"]) >= 1
    assert any(
        {
            str(item.get("well_a")),
            str(item.get("well_b")),
        }
        == {"WELL-A", "WELL-D"}
        for item in tuple(resolution["items"])
    )


def test_prepare_rerun_from_recommendation_skips_trajectory_override_without_context() -> None:
    page = _load_welltrack_page_module()
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1100.0,
                md_a_end_m=1700.0,
                md_b_start_m=1200.0,
                md_b_end_m=1750.0,
                md_a_values_m=np.array([1100.0, 1700.0], dtype=float),
                md_b_values_m=np.array([1200.0, 1750.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.72, 0.69], dtype=float),
                overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    successes = [_successful_plan(name="WELL-B", y_offset_m=5.0, kop_md_m=900.0)]
    recommendation = build_anti_collision_recommendations(
        analysis,
        well_context_by_name={
            "WELL-B": page._build_anticollision_well_contexts(successes)["WELL-B"],
        },
    )[0]

    page.st.session_state["wt_prepared_well_overrides"] = {"OLD": {"update_fields": {}}}
    page.st.session_state["wt_prepared_override_message"] = ""
    page.st.session_state["wt_prepared_recommendation_id"] = ""
    page.st.session_state["wt_pending_selected_names"] = ["OLD"]
    page._prepare_rerun_from_recommendation(
        recommendation,
        successes=successes,
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    assert page.st.session_state["wt_prepared_well_overrides"] == {}
    assert "контекст" in str(page.st.session_state["wt_prepared_override_message"]).lower()
    assert page.st.session_state["wt_pending_selected_names"] is None


def test_prepare_rerun_from_recommendation_builds_two_well_pair_plan() -> None:
    page = _load_welltrack_page_module()
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="WELL-A",
                well_b="WELL-B",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1100.0,
                md_a_end_m=1700.0,
                md_b_start_m=1200.0,
                md_b_end_m=1750.0,
                md_a_values_m=np.array([1100.0, 1700.0], dtype=float),
                md_b_values_m=np.array([1200.0, 1750.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.72, 0.69], dtype=float),
                overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0, kop_md_m=700.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0, kop_md_m=900.0),
        _successful_plan(name="WELL-C", y_offset_m=120.0, kop_md_m=800.0),
    ]
    recommendation = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )[0]

    page._prepare_rerun_from_recommendation(
        recommendation,
        successes=successes,
        uncertainty_model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    prepared = dict(page.st.session_state["wt_prepared_well_overrides"])
    assert set(prepared.keys()) == {"WELL-A", "WELL-B"}
    snapshot = dict(page.st.session_state["wt_prepared_recommendation_snapshot"])
    assert tuple(snapshot["affected_wells"]) == ("WELL-B", "WELL-A")
    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-B", "WELL-A"]


def test_anticollision_3d_figure_draws_terminal_cone_boundaries_per_well() -> None:
    page = _load_welltrack_page_module()
    figure = page._all_wells_anticollision_3d_figure(
        page._build_anti_collision_analysis(
            [
                _successful_plan(name="WELL-A", y_offset_m=0.0),
                _successful_plan(name="WELL-B", y_offset_m=5.0),
            ],
            model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        )
    )

    boundary_traces = [
        trace
        for trace in figure.data
        if str(trace.type) == "scatter3d"
        and str(trace.name).endswith(": граница конуса")
    ]

    assert len(boundary_traces) == 2
    assert {str(trace.line.color) for trace in boundary_traces} == {
        page._lighten_hex(page._well_color(0)),
        page._lighten_hex(page._well_color(1)),
    }
    assert all(float(trace.line.width) == 1.5 for trace in boundary_traces)
    assert figure.layout.scene.camera.to_plotly_json() == DEFAULT_3D_CAMERA


def test_anticollision_figures_draw_previous_trajectories_as_dashed_overlay() -> None:
    page = _load_welltrack_page_module()
    current_successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    previous_successes = {
        "WELL-A": _successful_plan(name="WELL-A", y_offset_m=-20.0),
    }
    analysis = page._build_anti_collision_analysis(
        current_successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    figure_3d = page._all_wells_anticollision_3d_figure(
        analysis,
        previous_successes_by_name=previous_successes,
    )
    figure_plan = page._all_wells_anticollision_plan_figure(
        analysis,
        previous_successes_by_name=previous_successes,
    )

    previous_3d = [
        trace for trace in figure_3d.data if str(trace.name).endswith(": до anti-collision")
    ]
    previous_plan = [
        trace for trace in figure_plan.data if str(trace.name).endswith(": до anti-collision")
    ]
    assert len(previous_3d) == 1
    assert len(previous_plan) == 1
    assert str(previous_3d[0].line.dash) == "dot"
    assert str(previous_plan[0].line.dash) == "dot"


def test_all_wells_3d_figure_uses_default_camera() -> None:
    page = _load_welltrack_page_module()
    figure = page._all_wells_3d_figure(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ]
    )

    assert figure.layout.scene.camera.to_plotly_json() == DEFAULT_3D_CAMERA


def test_all_wells_overview_figures_show_targets_for_failed_wells_without_fake_trajectory() -> None:
    page = _load_welltrack_page_module()
    records = _records()
    summary_rows = [
        {"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
        {
            "Скважина": "WELL-C",
            "Статус": "Ошибка расчета",
            "Проблема": "Solver endpoint miss to t1.",
        },
    ]
    target_only_wells = page._failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    color_map = page._well_color_map(records)

    figure_3d = page._all_wells_3d_figure(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        target_only_wells=target_only_wells,
        name_to_color=color_map,
    )
    figure_plan = page._all_wells_plan_figure(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        target_only_wells=target_only_wells,
        name_to_color=color_map,
    )

    failed_3d_traces = [
        trace
        for trace in figure_3d.data
        if str(trace.name) == "WELL-C: цели (без траектории)"
    ]
    failed_plan_traces = [
        trace
        for trace in figure_plan.data
        if str(trace.name) == "WELL-C: цели (без траектории)"
    ]
    failed_trajectory_3d = [
        trace for trace in figure_3d.data if str(trace.name) == "WELL-C"
    ]
    failed_trajectory_plan = [
        trace for trace in figure_plan.data if str(trace.name) == "WELL-C"
    ]

    assert len(target_only_wells) == 1
    assert failed_3d_traces
    assert failed_plan_traces
    assert not failed_trajectory_3d
    assert not failed_trajectory_plan
    assert "Статус: %{customdata[1]}" in str(failed_3d_traces[0].hovertemplate)
    assert "Точка: %{customdata[0]}" in str(failed_plan_traces[0].hovertemplate)
    assert str(failed_3d_traces[0].marker.symbol) == "cross"
    assert str(failed_plan_traces[0].marker.symbol) == "x"


def test_all_wells_overview_figures_include_reference_trajectories() -> None:
    page = _load_welltrack_page_module()
    reference_wells = _reference_wells()

    figure_3d = page._all_wells_3d_figure(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        reference_wells=reference_wells,
    )
    figure_plan = page._all_wells_plan_figure(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        reference_wells=reference_wells,
    )

    trace_names_3d = {str(trace.name) for trace in figure_3d.data}
    trace_names_plan = {str(trace.name) for trace in figure_plan.data}

    assert "FACT-1 (Фактическая)" in trace_names_3d
    assert "APP-1 (Проектная утвержденная)" in trace_names_3d
    assert "FACT-1 (Фактическая)" in trace_names_plan
    assert "APP-1 (Проектная утвержденная)" in trace_names_plan

    fact_3d = next(trace for trace in figure_3d.data if str(trace.name) == "FACT-1 (Фактическая)")
    approved_3d = next(trace for trace in figure_3d.data if str(trace.name) == "APP-1 (Проектная утвержденная)")
    assert str(fact_3d.line.color) == "#6B7280"
    assert str(approved_3d.line.color) == "#C62828"
    fact_label_3d = next(
        trace for trace in figure_3d.data if str(trace.name) == "Фактическая: подписи"
    )
    approved_label_3d = next(
        trace
        for trace in figure_3d.data
        if str(trace.name) == "Проектная утвержденная: подписи"
    )
    assert "FACT-1" in list(fact_label_3d.text)
    assert "APP-1" in list(approved_label_3d.text)
    assert str(fact_label_3d.textfont.color) == "#111111"
    assert str(approved_label_3d.textfont.color) == "#C62828"


def test_reference_well_labels_anchor_at_horizontal_entry_for_horizontal_wells() -> None:
    page = _load_welltrack_page_module()
    horizontal_well = _horizontal_reference_well()[0]

    anchor = page._reference_label_anchor_point(horizontal_well)

    assert anchor is not None
    assert float(anchor[0]) == pytest.approx(700.0)
    assert float(anchor[1]) == pytest.approx(0.0)
    assert float(anchor[2]) == pytest.approx(1150.0)


def test_all_wells_overview_figures_ignore_far_reference_wells_for_axis_zoom() -> None:
    page = _load_welltrack_page_module()
    successes = [_successful_plan(name="WELL-A", y_offset_m=0.0)]

    base_3d = page._all_wells_3d_figure(successes)
    base_plan = page._all_wells_plan_figure(successes)
    far_ref_3d = page._all_wells_3d_figure(
        successes,
        reference_wells=_far_reference_wells(),
    )
    far_ref_plan = page._all_wells_plan_figure(
        successes,
        reference_wells=_far_reference_wells(),
    )

    assert tuple(base_3d.layout.scene.xaxis.range) == tuple(far_ref_3d.layout.scene.xaxis.range)
    assert tuple(base_3d.layout.scene.yaxis.range) == tuple(far_ref_3d.layout.scene.yaxis.range)
    assert tuple(base_3d.layout.scene.zaxis.range) == tuple(far_ref_3d.layout.scene.zaxis.range)
    assert tuple(base_plan.layout.xaxis.range) == tuple(far_ref_plan.layout.xaxis.range)
    assert tuple(base_plan.layout.yaxis.range) == tuple(far_ref_plan.layout.yaxis.range)


def test_anticollision_figures_include_reference_trajectory_wells_without_target_markers() -> None:
    page = _load_welltrack_page_module()
    reference_wells = _reference_wells()
    analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=reference_wells,
    )

    figure_3d = page._all_wells_anticollision_3d_figure(analysis)
    figure_plan = page._all_wells_anticollision_plan_figure(analysis)

    assert any(str(trace.name) == "FACT-1 (Фактическая)" for trace in figure_3d.data)
    assert any(str(trace.name) == "APP-1 (Проектная утвержденная)" for trace in figure_3d.data)
    assert not any("FACT-1 (Фактическая): цели" == str(trace.name) for trace in figure_3d.data)
    assert not any("APP-1 (Проектная утвержденная): цели" == str(trace.name) for trace in figure_plan.data)
    assert any(str(trace.name) == "Фактическая: подписи" for trace in figure_3d.data)
    assert any(str(trace.name) == "Проектная утвержденная: подписи" for trace in figure_plan.data)


def test_anticollision_figures_ignore_far_reference_wells_for_axis_zoom() -> None:
    page = _load_welltrack_page_module()
    successes = [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    base_analysis = page._build_anti_collision_analysis(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    far_ref_analysis = page._build_anti_collision_analysis(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=_far_reference_wells(),
    )

    base_3d = page._all_wells_anticollision_3d_figure(base_analysis)
    base_plan = page._all_wells_anticollision_plan_figure(base_analysis)
    far_ref_3d = page._all_wells_anticollision_3d_figure(far_ref_analysis)
    far_ref_plan = page._all_wells_anticollision_plan_figure(far_ref_analysis)

    assert tuple(base_3d.layout.scene.xaxis.range) == tuple(far_ref_3d.layout.scene.xaxis.range)
    assert tuple(base_3d.layout.scene.yaxis.range) == tuple(far_ref_3d.layout.scene.yaxis.range)
    assert tuple(base_3d.layout.scene.zaxis.range) == tuple(far_ref_3d.layout.scene.zaxis.range)
    assert tuple(base_plan.layout.xaxis.range) == tuple(far_ref_plan.layout.xaxis.range)
    assert tuple(base_plan.layout.yaxis.range) == tuple(far_ref_plan.layout.yaxis.range)


def test_all_wells_3d_figure_aggregates_reference_wells_in_fast_mode() -> None:
    page = _load_welltrack_page_module()
    figure = page._all_wells_3d_figure(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        reference_wells=_reference_wells(),
        render_mode=page.WT_3D_RENDER_FAST,
    )

    trace_names = {str(trace.name) for trace in figure.data}
    assert "Фактическая (сводно)" in trace_names
    assert "Проектная утвержденная (сводно)" in trace_names
    assert "FACT-1 (Фактическая)" not in trace_names
    assert "APP-1 (Проектная утвержденная)" not in trace_names


def test_anticollision_3d_figure_aggregates_non_conflicting_reference_wells_in_fast_mode() -> None:
    page = _load_welltrack_page_module()
    analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=_reference_wells(),
    )

    figure = page._all_wells_anticollision_3d_figure(
        analysis,
        render_mode=page.WT_3D_RENDER_FAST,
    )

    trace_names = {str(trace.name) for trace in figure.data}
    assert "Фактическая (сводно)" in trace_names
    assert "Проектная утвержденная (сводно)" in trace_names
    assert "FACT-1 (Фактическая)" not in trace_names
    assert "APP-1 (Проектная утвержденная)" not in trace_names


def test_all_wells_and_anticollision_figures_show_well_names_at_t1() -> None:
    page = _load_welltrack_page_module()
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    records = _records()[:2]
    color_map = page._well_color_map(records)

    overview_3d = page._all_wells_3d_figure(successes, name_to_color=color_map)
    overview_plan = page._all_wells_plan_figure(successes, name_to_color=color_map)
    analysis = page._build_anti_collision_analysis(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        name_to_color=color_map,
    )
    anticollision_3d = page._all_wells_anticollision_3d_figure(analysis)
    anticollision_plan = page._all_wells_anticollision_plan_figure(analysis)

    for figure in (overview_3d, overview_plan, anticollision_3d, anticollision_plan):
        trace_names = {str(trace.name) for trace in figure.data}
        assert "WELL-A: t1 label" in trace_names
        assert "WELL-B: t1 label" in trace_names


def test_plotly_3d_figure_to_three_payload_preserves_overview_labels_and_legend() -> None:
    page = _load_welltrack_page_module()
    figure = page._all_wells_3d_figure(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        reference_wells=_reference_wells(),
    )

    payload = page._plotly_3d_figure_to_three_payload(figure)

    labels = {str(item["text"]) for item in payload["labels"]}
    legend_labels = {str(item["label"]) for item in payload["legend"]}
    assert payload["camera"] == DEFAULT_3D_CAMERA
    assert "WELL-A" in labels
    assert "FACT-1" in labels
    assert "APP-1" in labels
    assert "WELL-A" in legend_labels
    assert "FACT-1 (Фактическая)" in legend_labels


def test_plotly_3d_figure_to_three_payload_preserves_anticollision_meshes() -> None:
    page = _load_welltrack_page_module()
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    figure = page._all_wells_anticollision_3d_figure(analysis)

    payload = page._plotly_3d_figure_to_three_payload(figure)

    assert payload["meshes"]
    assert any(str(item["label"]) == "WELL-A" for item in payload["legend"])


def test_optimize_three_payload_merges_same_style_objects() -> None:
    page = _load_welltrack_page_module()
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_3D_CAMERA,
        "lines": [
            {
                "segments": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                "color": "#0B6E4F",
                "opacity": 1.0,
                "dash": "solid",
            },
            {
                "segments": [[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]],
                "color": "#0B6E4F",
                "opacity": 1.0,
                "dash": "solid",
            },
        ],
        "points": [
            {
                "points": [[0.0, 0.0, 0.0]],
                "color": "#111111",
                "opacity": 1.0,
                "size": 6.0,
                "symbol": "circle",
            },
            {
                "points": [[1.0, 0.0, 0.0]],
                "color": "#111111",
                "opacity": 1.0,
                "size": 6.0,
                "symbol": "circle",
            },
        ],
        "meshes": [
            {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#C62828",
                "opacity": 0.4,
            },
            {
                "vertices": [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#C62828",
                "opacity": 0.4,
            },
        ],
        "labels": [{"text": "A", "position": [0.0, 0.0, 0.0], "color": "#111111"}],
        "legend": [{"label": "A", "color": "#0B6E4F", "opacity": 1.0}],
    }

    optimized = page._optimize_three_payload(payload)

    assert len(optimized["lines"]) == 1
    assert len(optimized["lines"][0]["segments"]) == 2
    assert len(optimized["points"]) == 1
    assert len(optimized["points"][0]["points"]) == 2
    assert len(optimized["meshes"]) == 1
    assert len(optimized["meshes"][0]["vertices"]) == 6
    assert len(optimized["meshes"][0]["faces"]) == 2
    assert optimized["camera"] == DEFAULT_3D_CAMERA
    assert optimized["labels"] == payload["labels"]
    assert optimized["legend"] == payload["legend"]


def test_plotly_3d_figure_to_three_payload_preserves_custom_camera() -> None:
    page = _load_welltrack_page_module()
    figure = page._all_wells_3d_figure([_successful_plan(name="WELL-A", y_offset_m=0.0)])
    custom_camera = {
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.15, "y": -0.05, "z": 0.0},
        "eye": {"x": 1.6, "y": 0.9, "z": 1.1},
    }
    figure.update_layout(scene={"camera": custom_camera})

    payload = page._plotly_3d_figure_to_three_payload(figure)

    assert payload["camera"] == custom_camera


def test_batch_summary_display_df_reorders_and_shortens_summary_columns() -> None:
    page = _load_welltrack_page_module()
    source = pd.DataFrame(
        [
            {
                "Скважина": "WELL-A",
                "Точек": 3,
                "Рестарты решателя": "1",
                "Статус": "OK",
                "Проблема": "",
                "Классификация целей": "В прямом направлении",
                "Сложность": "Обычная",
                "Горизонтальный отход t1, м": "100.00",
                "KOP MD, м": "550.00",
                "Длина HORIZONTAL, м": "1200.00",
                "INC в t1, deg": "86.00",
                "ЗУ HOLD, deg": "45.00",
                "Макс ПИ, deg/10m": "1.00",
                "Макс MD, м": "4300.00",
                "Модель траектории": "Unified J Profile + Build + Azimuth Turn",
            }
        ]
    )

    display_df = page._batch_summary_display_df(source)

    assert list(display_df.columns[:8]) == [
        "Скважина",
        "Точек",
        "Цели",
        "Сложность",
        "Отход t1, м",
        "KOP MD, м",
        "HORIZONTAL, м",
        "INC в t1, deg",
    ]
    assert display_df.columns[-1] == "Модель траектории"
    assert "Рестарты" in display_df.columns
    assert "Классификация целей" not in display_df.columns
    assert "Рестарты решателя" not in display_df.columns


def test_anticollision_figures_render_overlap_corridor_and_red_conflict_segments() -> None:
    page = _load_welltrack_page_module()
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    figure_3d = page._all_wells_anticollision_3d_figure(analysis)
    figure_plan = page._all_wells_anticollision_plan_figure(analysis)

    overlap_3d_traces = [
        trace
        for trace in figure_3d.data
        if str(trace.name) == "Общая зона overlap"
    ]
    conflict_3d_traces = [
        trace
        for trace in figure_3d.data
        if str(trace.name) == "Конфликтный участок ствола"
    ]
    overlap_plan_traces = [
        trace
        for trace in figure_plan.data
        if str(trace.name) == "Общая зона overlap"
    ]
    conflict_plan_traces = [
        trace
        for trace in figure_plan.data
        if str(trace.name) == "Конфликтный участок ствола"
    ]

    assert overlap_3d_traces
    assert conflict_3d_traces
    assert overlap_plan_traces
    assert conflict_plan_traces
    assert all("198, 40, 40" in str(trace.line.color) for trace in conflict_3d_traces)
    assert all("198, 40, 40" in str(trace.line.color) for trace in conflict_plan_traces)
    assert all(str(getattr(trace, "hoverinfo", "")) == "skip" for trace in overlap_plan_traces)


def test_anticollision_3d_trajectory_hover_is_reserved_for_wells_targets_and_conflict_segments() -> None:
    page = _load_welltrack_page_module()
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    figure_3d = page._all_wells_anticollision_3d_figure(analysis)

    cone_meshes = [
        trace for trace in figure_3d.data if str(trace.type) == "mesh3d" and "cone" in str(trace.name)
    ]
    overlap_meshes = [
        trace for trace in figure_3d.data if str(trace.name) == "Общая зона overlap"
    ]
    well_lines = [
        trace for trace in figure_3d.data if str(trace.type) == "scatter3d" and str(trace.name) == "WELL-A"
    ]
    target_markers = [
        trace
        for trace in figure_3d.data
        if str(trace.type) == "scatter3d" and str(trace.name) == "WELL-A: цели"
    ]
    conflict_segments = [
        trace
        for trace in figure_3d.data
        if str(trace.type) == "scatter3d" and str(trace.name) == "Конфликтный участок ствола"
    ]

    assert cone_meshes
    assert overlap_meshes
    assert all(str(getattr(trace, "hoverinfo", "")) == "skip" for trace in cone_meshes)
    assert all(str(getattr(trace, "hoverinfo", "")) == "skip" for trace in overlap_meshes)
    assert well_lines
    assert "ПИ: %{customdata[1]:.2f} deg/10m" in str(well_lines[0].hovertemplate)
    assert "Сегмент: %{customdata[2]}" in str(well_lines[0].hovertemplate)
    assert target_markers
    assert "Точка: %{customdata[0]}" in str(target_markers[0].hovertemplate)
    assert conflict_segments
    assert "ПИ: %{customdata[1]:.2f} deg/10m" in str(conflict_segments[0].hovertemplate)
