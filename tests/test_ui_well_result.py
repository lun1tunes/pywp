from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from pywp.models import Point3D
from pywp.ui_well_result import (
    SUMMARY_TECH_HIDDEN_METRICS,
    SingleWellResultView,
    build_key_metrics_rows,
    build_target_validation_rows,
    collect_issue_messages,
    horizontal_offset_m,
    md_postcheck_issue_message,
    uncertainty_preset_key,
    uncertainty_toggle_key,
)


def test_md_postcheck_issue_message_is_empty_without_excess() -> None:
    summary = {
        "md_total_m": 6400.0,
        "max_total_md_postcheck_m": 6500.0,
        "md_postcheck_excess_m": 0.0,
    }
    assert md_postcheck_issue_message(summary) == ""


def test_collect_issue_messages_deduplicates_postcheck_message() -> None:
    summary = {
        "md_total_m": 6578.04,
        "max_total_md_postcheck_m": 6500.0,
        "md_postcheck_excess_m": 78.04,
    }
    md_message = md_postcheck_issue_message(summary)
    issues = collect_issue_messages(
        summary=summary,
        extra_messages=(md_message, "Пользовательская проверка"),
    )
    assert issues == (md_message, "Пользовательская проверка")


def test_horizontal_offset_m_uses_xy_distance_only() -> None:
    surface = Point3D(1000.0, 2000.0, 0.0)
    t1 = Point3D(1012.0, 2016.0, 3000.0)
    assert horizontal_offset_m(point=t1, reference=surface) == 20.0


def test_build_target_validation_rows_formats_exact_and_control_metrics() -> None:
    summary = {
        "distance_t1_m": 0.1234,
        "distance_t3_m": 1.75,
        "distance_t1_control_m": 0.4567,
        "distance_t3_control_m": 1.95,
        "control_gap_t1_m": 0.02,
        "control_gap_t3_m": 0.08,
        "entry_inc_deg": 86.0,
        "entry_inc_control_deg": 85.94,
        "t1_miss_dx_m": 0.10,
        "t1_miss_dy_m": -0.20,
        "t1_miss_dz_m": 0.30,
        "t3_miss_dx_m": 1.10,
        "t3_miss_dy_m": 0.20,
        "t3_miss_dz_m": -0.40,
    }

    rows = build_target_validation_rows(summary)
    assert rows[0] == {"Показатель": "Промах t1 (аналитический, 3D)", "Значение": "0.1234 m"}
    assert rows[1] == {"Показатель": "Промах t3 (аналитический, 3D)", "Значение": "1.75 m"}
    assert rows[6]["Показатель"] == "Компоненты промаха t1 (dX / dY / dZ)"
    assert rows[6]["Значение"] == "0.10 / -0.20 / 0.30 м"
    assert rows[-1] == {
        "Показатель": "INC на t1: analytic / control-grid",
        "Значение": "86.00 / 85.94 deg",
    }


def test_summary_tech_hidden_metrics_cover_raw_validation_fields() -> None:
    assert "distance_t1_m" in SUMMARY_TECH_HIDDEN_METRICS
    assert "distance_t3_control_m" in SUMMARY_TECH_HIDDEN_METRICS
    assert "t1_exact_x_m" in SUMMARY_TECH_HIDDEN_METRICS
    assert "t3_miss_dz_m" in SUMMARY_TECH_HIDDEN_METRICS


def test_single_well_result_view_accepts_model_like_inputs_from_stale_session_state() -> None:
    class LegacyPoint(BaseModel):
        x: float
        y: float
        z: float

    class LegacyConfig(BaseModel):
        md_step_m: float = 10.0
        md_step_control_m: float = 2.0
        pos_tolerance_m: float = 2.0
        entry_inc_target_deg: float = 86.0
        entry_inc_tolerance_deg: float = 2.0
        max_inc_deg: float = 95.0
        dls_build_min_deg_per_30m: float = 0.0
        dls_build_max_deg_per_30m: float = 3.0
        kop_min_vertical_m: float = 550.0
        max_total_md_m: float = 12000.0
        max_total_md_postcheck_m: float = 6500.0
        objective_mode: str = "minimize_total_md"
        turn_solver_mode: str = "least_squares"
        turn_solver_max_restarts: int = 2
        min_structural_segment_m: float = 30.0
        dls_limits_deg_per_30m: dict[str, float] = {
            "VERTICAL": 1.0,
            "BUILD1": 3.0,
            "HOLD": 2.0,
            "BUILD2": 3.0,
            "HORIZONTAL": 2.0,
        }

    view = SingleWellResultView(
        well_name="single_well",
        surface=LegacyPoint(x=0.0, y=0.0, z=0.0),
        t1=LegacyPoint(x=600.0, y=800.0, z=2400.0),
        t3=LegacyPoint(x=1500.0, y=2000.0, z=2500.0),
        stations=pd.DataFrame({"MD_m": [0.0], "X_m": [0.0], "Y_m": [0.0], "Z_m": [0.0]}),
        summary={"trajectory_type": "Unified J Profile + Build + Azimuth Turn"},
        config=LegacyConfig(),
        azimuth_deg=0.0,
        md_t1_m=1000.0,
    )

    assert isinstance(view.surface, Point3D)
    assert isinstance(view.t1, Point3D)
    assert isinstance(view.t3, Point3D)
    assert view.config.turn_solver_mode == "least_squares"


def test_uncertainty_toggle_key_is_stable_per_well_name() -> None:
    assert uncertainty_toggle_key(well_name="WELL-1") == "show_uncertainty_ellipses::WELL-1"


def test_uncertainty_preset_key_is_stable_per_well_name() -> None:
    assert uncertainty_preset_key(well_name="WELL-1") == "uncertainty_preset::WELL-1"


def test_build_key_metrics_rows_single_column_format() -> None:
    view = SingleWellResultView(
        well_name="WELL-1",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        stations=pd.DataFrame({"MD_m": [0.0], "X_m": [0.0], "Y_m": [0.0], "Z_m": [0.0]}),
        summary={
            "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
            "trajectory_target_direction": "Цели в одном направлении",
            "well_complexity": "Обычная",
            "optimization_mode": "minimize_md",
            "azimuth_turn_deg": 18.5,
            "entry_inc_deg": 86.0,
            "hold_inc_deg": 84.0,
            "build_dls_selected_deg_per_30m": 3.0,
            "build1_dls_selected_deg_per_30m": 3.0,
            "build2_dls_selected_deg_per_30m": 3.0,
            "max_dls_total_deg_per_30m": 3.0,
            "kop_md_m": 560.0,
            "horizontal_length_m": 1200.0,
            "max_inc_actual_deg": 88.0,
            "max_inc_deg": 95.0,
            "md_total_m": 4300.0,
            "max_total_md_postcheck_m": 6500.0,
            "md_postcheck_excess_m": 0.0,
            "solver_turn_restarts_used": 0,
            "solver_turn_max_restarts": 2,
        },
        config={"optimization_mode": "minimize_md"},
        azimuth_deg=0.0,
        md_t1_m=1000.0,
        runtime_s=1.23,
    )

    rows = build_key_metrics_rows(view)
    by_label = {row["Показатель"]: row for row in rows}

    # Only two columns: "Показатель" and "Значение"
    for row in rows:
        assert set(row.keys()) == {"Показатель", "Значение"}

    assert by_label["Проблемы"]["Значение"] == "-"
    assert by_label["Рестарты решателя"]["Значение"] == "0 / 2"
    assert by_label["Время расчета"]["Значение"] == "1.23 с"
    assert by_label["BUILD1 / BUILD2 / Макс ПИ"]["Значение"] == "1.00 / 1.00 / 1.00 deg/10m"
    assert by_label["Итоговая MD"]["Значение"] == "4300.00 m"
    assert by_label["Оптимизация"]["Значение"] == "Минимизация MD"
