from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from pywp import ptc_core as wt_import_module
from pywp import ptc_anticollision_params
from pywp.actual_fund_analysis import ActualFundKopDepthFunction
from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionCorridor,
    build_anti_collision_well,
)
from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
)
from pywp.anticollision_recommendations import (
    build_anti_collision_recommendation_clusters,
    build_anti_collision_recommendations,
)
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.mcm import compute_positions_min_curv
from pywp.models import Point3D, TrajectoryConfig
from pywp.reference_trajectories import parse_reference_trajectory_table
from pywp.three_config import DEFAULT_THREE_CAMERA
from pywp.ui_calc_params import (
    clear_kop_min_vertical_function,
    set_kop_min_vertical_function,
)
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC,
    UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC,
    planning_uncertainty_model_for_preset,
)
from pywp.well_pad import detect_well_pads
from pywp.welltrack_batch import SuccessfulWellPlan

pytestmark = pytest.mark.integration


def _records() -> list[WelltrackRecord]:
    records = [
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
    return records


def _bad_order_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="BAD-1",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=1200.0, y=0.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=500.0, y=0.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="OK-1",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=0.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1200.0, y=0.0, z=2500.0, md=3500.0),
            ),
        ),
    ]


def _two_bad_order_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="BAD-1",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=1200.0, y=0.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=500.0, y=0.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="BAD-2",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=1100.0, y=0.0, z=2450.0, md=2450.0),
                WelltrackPoint(x=650.0, y=0.0, z=2550.0, md=3550.0),
            ),
        ),
    ]


def _multi_pad_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="PAD1-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="PAD1-B",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="PAD2-A",
            points=(
                WelltrackPoint(x=5000.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=5600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=6500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="PAD2-B",
            points=(
                WelltrackPoint(x=5000.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=5650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=6550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
    ]


def _prepositioned_pad_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="P1-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="P1-B",
            points=(
                WelltrackPoint(x=25.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=625.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1525.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="P1-C",
            points=(
                WelltrackPoint(x=50.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=760.0, z=2200.0, md=2300.0),
                WelltrackPoint(x=1550.0, y=1960.0, z=2350.0, md=3350.0),
            ),
        ),
        WelltrackRecord(
            name="P2-A",
            points=(
                WelltrackPoint(x=2000.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=2600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=3500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="P2-B",
            points=(
                WelltrackPoint(x=2025.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=2625.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=3525.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
    ]


def _submeter_prepositioned_pad_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="SUB-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="SUB-B",
            points=(
                WelltrackPoint(x=0.2, y=0.1, z=0.0, md=0.0),
                WelltrackPoint(x=625.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1525.0, y=1980.0, z=2400.0, md=3400.0),
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
        (
            float(record.points[0].x),
            float(record.points[0].y),
            float(record.points[0].z),
        )
        for record in records
        if record.points
    ]


def test_records_overview_dataframe_uses_explicit_green_red_status_icons() -> None:
    page = wt_import_module
    incomplete = WelltrackRecord(
        name="WELL-X",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
        ),
    )

    overview_df = page._records_overview_dataframe([_records()[0], incomplete])

    assert list(overview_df.columns) == [
        "Скважина",
        "Точек",
        "Отход t1, м",
        "Длина ГС, м",
        "Примечание",
        "Статус",
        "Проблема",
    ]
    assert list(overview_df["Статус"]) == ["✅", "❌"]
    assert list(overview_df["Точек"]) == [2, 1]
    assert list(overview_df["Отход t1, м"]) == [1000.0, 1000.0]
    assert float(overview_df.iloc[0]["Длина ГС, м"]) == pytest.approx(1503.3296)
    assert pd.isna(overview_df.iloc[1]["Длина ГС, м"])
    assert str(overview_df.iloc[0]["Проблема"]) == "—"
    assert "Не хватает одной из точек" in str(overview_df.iloc[1]["Проблема"])


def test_analysis_reference_wells_rebuilds_surface_as_current_point3d() -> None:
    page = wt_import_module
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "INC_deg": [0.0, 90.0],
            "AZI_deg": [0.0, 90.0],
            "X_m": [100.0, 1100.0],
            "Y_m": [200.0, 200.0],
            "Z_m": [5.0, 5.0],
        }
    )
    analysis = AntiCollisionAnalysis(
        wells=(
            SimpleNamespace(
                name="REF-1",
                well_kind="approved",
                stations=stations,
                surface=SimpleNamespace(x=100.0, y=200.0, z=5.0),
                is_reference_only=True,
            ),
        ),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=0,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )

    reference_wells = page._analysis_reference_wells(analysis)

    assert len(reference_wells) == 1
    assert isinstance(reference_wells[0].surface, Point3D)
    assert reference_wells[0].surface == Point3D(x=100.0, y=200.0, z=5.0)


def test_records_overview_dataframe_detects_missing_surface_s_by_surface_z_heuristic() -> (
    None
):
    page = wt_import_module
    missing_surface = WelltrackRecord(
        name="NO-S",
        points=(
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            WelltrackPoint(x=2200.0, y=2900.0, z=2520.0, md=4400.0),
        ),
    )

    overview_df = page._records_overview_dataframe([missing_surface])

    assert list(overview_df["Статус"]) == ["❌"]
    assert list(overview_df["Точек"]) == [2]
    assert "Не найдена точка `S`" in str(overview_df.iloc[0]["Проблема"])


def test_records_overview_table_uses_collapsed_status_expander_without_problems(
    monkeypatch,
) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}

    class _DummyColumn:
        def metric(self, label, value, *args, **kwargs):
            captured.setdefault("metrics", []).append((label, value))
            return None

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_expander(label, *args, **kwargs):
        captured["label"] = label
        captured["expanded"] = kwargs.get("expanded")
        return _DummyExpander()

    def _fake_dataframe(frame, **kwargs):
        captured["frame"] = frame.copy()

    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn(), _DummyColumn()),
    )
    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)

    page._render_records_overview([_records()[0]])

    assert captured["label"] == "Статус загрузки целей"
    assert captured["expanded"] is False
    assert captured["metrics"] == [
        ("Скважин", "1"),
        ("Пилотов", "0"),
        ("Проблем", "0"),
    ]
    assert list(captured["frame"].columns) == [
        "Скважина",
        "Точек",
        "Отход t1, м",
        "Длина ГС, м",
        "Примечание",
        "Статус",
        "Проблема",
    ]
    assert list(captured["frame"]["Статус"]) == ["✅"]


def test_records_overview_status_expander_opens_when_import_has_problems(
    monkeypatch,
) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}
    incomplete = WelltrackRecord(
        name="WELL-X",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
        ),
    )

    class _DummyColumn:
        def metric(self, *args, **kwargs):
            return None

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_expander(label, *args, **kwargs):
        captured["label"] = label
        captured["expanded"] = kwargs.get("expanded")
        return _DummyExpander()

    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn(), _DummyColumn()),
    )
    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)

    page._render_records_overview([_records()[0], incomplete])

    assert captured["label"] == "Статус загрузки целей"
    assert captured["expanded"] is True


def test_records_overview_metrics_count_wells_and_pilots_separately(
    monkeypatch,
) -> None:
    page = wt_import_module
    captured: dict[str, object] = {"metrics": []}
    parent = WelltrackRecord(
        name="well_04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=300.0, y=0.0, z=1000.0, md=3.0),
        ),
    )
    pilot = WelltrackRecord(
        name="well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=50.0, y=0.0, z=800.0, md=2.0),
        ),
    )

    class _DummyColumn:
        def metric(self, label, value, *args, **kwargs):
            captured["metrics"].append((label, value))
            return None

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn(), _DummyColumn()),
    )
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)

    page._render_records_overview([parent, pilot])

    assert captured["metrics"] == [
        ("Скважин", "1"),
        ("Пилотов", "1"),
        ("Проблем", "0"),
    ]


def test_t1_t3_resolution_message_reports_fixed_and_kept_wells() -> None:
    page = wt_import_module

    page._clear_t1_t3_order_resolution_state()
    page._set_t1_t3_order_resolution(action="fixed", well_names={"WELL-B", "WELL-A"})
    assert page._t1_t3_order_resolution_message() == (
        "success",
        "Порядок t1/t3 изменился для скважин: WELL-A, WELL-B.",
    )

    page._set_t1_t3_order_resolution(action="kept", well_names={"WELL-C"})
    assert page._t1_t3_order_resolution_message() == (
        "info",
        "Порядок t1/t3 оставлен без изменений для скважин: WELL-C.",
    )


def test_welltrack_page_renders_t1_t3_order_actions_for_conflicting_wells() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _bad_order_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run(timeout=120)

    button_labels = {str(widget.label) for widget in at.button}
    assert "Исправить порядок для выбранных скважин" in button_labels
    assert "Оставить все точки без изменений" in button_labels


def test_welltrack_page_initial_crs_selectbox_has_no_session_state_default_warning() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")

    at.run(timeout=120)

    assert not any(
        "trajectory_crs_selectbox" in str(widget.value) for widget in at.warning
    )


def test_welltrack_page_keeps_t1_t3_order_panel_visible_when_no_issues() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run(timeout=120)

    assert any(
        "Проверка порядка t1/t3 — OK." in str(widget.value) for widget in at.success
    )


def test_welltrack_page_hides_t1_t3_warning_after_keep_action() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _bad_order_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run(timeout=120)
    _click_button(at, "Оставить все точки без изменений")
    at.run(timeout=120)

    assert any(
        "Порядок t1/t3 оставлен без изменений для скважин: BAD-1." in str(widget.value)
        for widget in at.info
    )
    assert not any(
        "Вероятно, порядок точек" in str(widget.value) for widget in at.warning
    )


def test_welltrack_page_shows_only_remaining_t1_t3_issue_after_partial_fix() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _two_bad_order_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_t1_t3_fix_BAD-1"] = True
    at.session_state["wt_t1_t3_fix_BAD-2"] = False

    at.run(timeout=120)
    _click_button(at, "Исправить порядок для выбранных скважин")
    at.run(timeout=120)

    assert any(
        "Порядок t1/t3 изменился для скважин: BAD-1." in str(widget.value)
        for widget in at.success
    )
    assert any("Вероятно, порядок точек" in str(widget.value) for widget in at.warning)
    markdown_values = [str(widget.value) for widget in at.markdown]
    assert any("`BAD-2`" in value for value in markdown_values)
    assert not any("`BAD-1`" in value for value in markdown_values)


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


def _successful_plan_xy(
    *,
    name: str,
    x_offset_m: float,
    y_offset_m: float,
) -> SuccessfulWellPlan:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 90.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [x_offset_m + 0.0, x_offset_m + 1000.0, x_offset_m + 2000.0],
            "Y_m": [y_offset_m, y_offset_m, y_offset_m],
            "Z_m": [0.0, 0.0, 0.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0],
            "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
        }
    )
    return SuccessfulWellPlan(
        name=name,
        surface={"x": x_offset_m + 0.0, "y": y_offset_m, "z": 0.0},
        t1={"x": x_offset_m + 1000.0, "y": y_offset_m, "z": 0.0},
        t3={"x": x_offset_m + 2000.0, "y": y_offset_m, "z": 0.0},
        stations=stations,
        summary={
            "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
            "trajectory_target_direction": "Цели в одном направлении",
            "well_complexity": "Обычная",
            "optimization_mode": "none",
            "azimuth_turn_deg": 0.0,
            "horizontal_length_m": 1000.0,
            "entry_inc_deg": 90.0,
            "hold_inc_deg": 90.0,
            "build_dls_selected_deg_per_30m": 0.0,
            "build1_dls_selected_deg_per_30m": 0.0,
            "max_dls_total_deg_per_30m": 0.0,
            "kop_md_m": 0.0,
            "max_inc_actual_deg": 90.0,
            "max_inc_deg": 95.0,
            "md_total_m": 2000.0,
            "max_total_md_postcheck_m": 6500.0,
            "md_postcheck_excess_m": 0.0,
        },
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(optimization_mode="none"),
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
            "Y_m": [
                y_offset_m,
                y_offset_m,
                y_offset_m,
                lateral_y_t1_m,
                lateral_y_end_m,
            ],
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
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 25.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 900.0,
                    "Y": 25.0,
                    "Z": 300.0,
                    "MD": 950.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 1800.0,
                    "Y": 25.0,
                    "Z": 400.0,
                    "MD": 1900.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": "approved",
                    "X": 0.0,
                    "Y": -35.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": "approved",
                    "X": 850.0,
                    "Y": -35.0,
                    "Z": 250.0,
                    "MD": 900.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": "approved",
                    "X": 1750.0,
                    "Y": -35.0,
                    "Z": 350.0,
                    "MD": 1850.0,
                },
            ]
        )
    )


def _far_reference_wells():
    return tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-FAR",
                    "Type": "actual",
                    "X": 50000.0,
                    "Y": 50000.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-FAR",
                    "Type": "actual",
                    "X": 52000.0,
                    "Y": 50000.0,
                    "Z": 300.0,
                    "MD": 2100.0,
                },
                {
                    "Wellname": "APP-FAR",
                    "Type": "approved",
                    "X": -45000.0,
                    "Y": -42000.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "APP-FAR",
                    "Type": "approved",
                    "X": -43000.0,
                    "Y": -42000.0,
                    "Z": 280.0,
                    "MD": 2050.0,
                },
            ]
        )
    )


def _horizontal_reference_well():
    return tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-H",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-H",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 1000.0,
                    "MD": 1000.0,
                },
                {
                    "Wellname": "FACT-H",
                    "Type": "actual",
                    "X": 100.0,
                    "Y": 0.0,
                    "Z": 1100.0,
                    "MD": 1600.0,
                },
                {
                    "Wellname": "FACT-H",
                    "Type": "actual",
                    "X": 700.0,
                    "Y": 0.0,
                    "Z": 1150.0,
                    "MD": 2200.0,
                },
                {
                    "Wellname": "FACT-H",
                    "Type": "actual",
                    "X": 1300.0,
                    "Y": 0.0,
                    "Z": 1200.0,
                    "MD": 2800.0,
                },
            ]
        )
    )


def _turned_horizontal_reference_well():
    survey = pd.DataFrame(
        {
            "MD_m": [0.0, 600.0, 900.0, 1200.0, 1600.0, 2000.0, 2350.0, 2800.0],
            "INC_deg": [0.0, 0.0, 20.0, 55.0, 55.0, 75.0, 90.0, 90.0],
            "AZI_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 90.0, 90.0],
        }
    )
    positioned = compute_positions_min_curv(survey, start=Point3D(0.0, 0.0, 0.0))
    return tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-TURN",
                    "Type": "actual",
                    "X": float(row["X_m"]),
                    "Y": float(row["Y_m"]),
                    "Z": float(row["Z_m"]),
                    "MD": float(row["MD_m"]),
                }
                for _, row in positioned.iterrows()
            ]
        )
    )


def test_welltrack_page_shows_only_general_run_before_results() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run()

    assert _multiselect_value(at, "Скважины для расчёта") == [
        "WELL-A",
        "WELL-B",
        "WELL-C",
    ]


def test_welltrack_general_run_select_all_restores_full_selection() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_selected_names"] = ["WELL-A"]

    at.run()
    assert _multiselect_value(at, "Скважины для расчёта") == ["WELL-A"]

    _click_button(at, "Выбрать все")
    at.run()

    assert _multiselect_value(at, "Скважины для расчёта") == [
        "WELL-A",
        "WELL-B",
        "WELL-C",
    ]


def test_welltrack_general_run_can_replace_selection_with_single_pad() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _multi_pad_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run()

    pad_select = next(widget for widget in at.selectbox if widget.label == "Куст")
    pad_select.set_value("PAD-02")
    at.run()

    _click_button(at, "Только куст")
    at.run()

    assert _multiselect_value(at, "Скважины для расчёта") == [
        "PAD2-A",
        "PAD2-B",
    ]


def test_welltrack_general_run_can_add_pad_to_existing_selection() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _multi_pad_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_selected_names"] = ["PAD1-A"]

    at.run()

    pad_select = next(widget for widget in at.selectbox if widget.label == "Куст")
    pad_select.set_value("PAD-02")
    at.run()

    _click_button(at, "Добавить куст")
    at.run()

    assert _multiselect_value(at, "Скважины для расчёта") == [
        "PAD1-A",
        "PAD2-A",
        "PAD2-B",
    ]


def test_trajectory_three_payload_overrides_build_tree_focus_targets_for_multi_pad() -> (
    None
):
    page = wt_import_module
    records = _multi_pad_records()
    successes = [
        _successful_plan_xy(name="PAD1-A", x_offset_m=0.0, y_offset_m=0.0),
        _successful_plan_xy(name="PAD1-B", x_offset_m=0.0, y_offset_m=50.0),
        _successful_plan_xy(name="PAD2-A", x_offset_m=5000.0, y_offset_m=0.0),
        _successful_plan_xy(name="PAD2-B", x_offset_m=5000.0, y_offset_m=50.0),
    ]
    name_to_color = {
        "PAD1-A": "#22c55e",
        "PAD1-B": "#2563eb",
        "PAD2-A": "#f59e0b",
        "PAD2-B": "#c026d3",
    }

    overrides = page._trajectory_three_payload_overrides(
        records=records,
        successes=successes,
        target_only_wells=[],
        name_to_color=name_to_color,
    )

    legend_tree = list(overrides["legend_tree"])
    focus_targets = dict(overrides["focus_targets"])
    hidden_labels = set(str(item) for item in overrides["hidden_flat_legend_labels"])
    first_surface_arrows = list(overrides["extra_meshes"])

    assert [str(item["label"]) for item in legend_tree] == [
        "Куст PAD-01",
        "Куст PAD-02",
    ]
    assert [str(child["label"]) for child in legend_tree[0]["children"]] == [
        "PAD1-A",
        "PAD1-B",
    ]
    assert [str(child["label"]) for child in legend_tree[1]["children"]] == [
        "PAD2-A",
        "PAD2-B",
    ]
    assert set(focus_targets) == {
        "pad::PAD-01",
        "pad::PAD-02",
        "well::PAD1-A",
        "well::PAD1-B",
        "well::PAD2-A",
        "well::PAD2-B",
    }
    assert hidden_labels == {"PAD1-A", "PAD1-B", "PAD2-A", "PAD2-B"}
    assert [str(item["well_name"]) for item in first_surface_arrows] == [
        "PAD1-A",
        "PAD2-A",
    ]
    assert {str(item["role"]) for item in first_surface_arrows} == {
        "pad_first_surface_arrow"
    }
    assert {str(item["color"]) for item in first_surface_arrows} == {"#475569"}
    assert {len(item["vertices"]) for item in first_surface_arrows} == {14}
    assert {len(item["faces"]) for item in first_surface_arrows} == {20}
    assert [str(item["end_well_name"]) for item in first_surface_arrows] == [
        "PAD1-B",
        "PAD2-B",
    ]
    assert first_surface_arrows[0]["start_position"] == [0.0, 0.0, 0.0]
    assert first_surface_arrows[1]["start_position"] == [5000.0, 0.0, 0.0]
    assert (
        np.linalg.norm(
            np.asarray(first_surface_arrows[0]["end_position"][:2], dtype=float)
            - np.asarray(first_surface_arrows[0]["start_position"][:2], dtype=float)
        )
        >= 72.0
    )
    assert (
        np.linalg.norm(
            np.asarray(first_surface_arrows[1]["end_position"][:2], dtype=float)
            - np.asarray(first_surface_arrows[1]["start_position"][:2], dtype=float)
        )
        >= 72.0
    )


def test_trajectory_three_payload_first_surface_arrow_uses_fixed_pad_order() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_records())
    pads = page._ensure_pad_configs(records)
    pad_id = str(pads[0].pad_id)
    page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] = ((1, "WELL-C"),)
    successes = [
        _successful_plan_xy(name="WELL-A", x_offset_m=0.0, y_offset_m=0.0),
        _successful_plan_xy(name="WELL-B", x_offset_m=0.0, y_offset_m=50.0),
        _successful_plan_xy(name="WELL-C", x_offset_m=0.0, y_offset_m=100.0),
    ]

    overrides = page._trajectory_three_payload_overrides(
        records=records,
        successes=successes,
        target_only_wells=[],
        name_to_color={
            "WELL-A": "#22c55e",
            "WELL-B": "#2563eb",
            "WELL-C": "#f59e0b",
        },
    )

    first_surface_arrows = list(overrides["extra_meshes"])
    assert [str(item["well_name"]) for item in first_surface_arrows] == ["WELL-C"]
    arrow = first_surface_arrows[0]
    assert str(arrow["role"]) == "pad_first_surface_arrow"
    start_xy = np.asarray(arrow["start_position"][:2], dtype=float)
    end_xy = np.asarray(arrow["end_position"][:2], dtype=float)
    tip_xy = np.asarray(arrow["vertices"][4][:2], dtype=float)
    expected_surface = records[2].points[0]
    assert np.allclose(start_xy, [expected_surface.x, expected_surface.y])
    assert np.allclose(tip_xy, end_xy)
    assert float(np.linalg.norm(end_xy - start_xy)) >= 50.0
    assert float(arrow["vertices"][4][2]) <= -24.0
    assert float(arrow["vertices"][11][2] - arrow["vertices"][4][2]) >= 5.0


def test_augment_three_payload_hides_flat_well_legend_when_tree_present() -> None:
    page = wt_import_module
    payload = {
        "legend": [
            {"label": "PAD1-A", "color": "#22c55e", "opacity": 1.0},
            {"label": "PAD2-A", "color": "#f59e0b", "opacity": 1.0},
            {"label": "Зоны пересечений", "color": "#fca5a5", "opacity": 0.4},
        ]
    }

    updated = page._augment_three_payload(
        payload=payload,
        legend_tree=[
            {
                "id": "pad::PAD-01",
                "label": "Куст PAD-01",
                "children": [
                    {"id": "well::PAD1-A", "label": "PAD1-A", "color": "#22c55e"}
                ],
            }
        ],
        focus_targets={"pad::PAD-01": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}},
        hidden_flat_legend_labels={"PAD1-A", "PAD2-A"},
    )

    assert updated["legend_tree"]
    assert updated["focus_targets"] == {
        "pad::PAD-01": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    }
    assert list(updated["legend"]) == [
        {"label": "Зоны пересечений", "color": "#fca5a5", "opacity": 0.4}
    ]


def test_well_color_palette_is_large_unique_and_locally_contrasting() -> None:
    page = wt_import_module

    palette = tuple(str(color) for color in page.WELL_COLOR_PALETTE)
    assert len(palette) >= 50
    assert len(set(palette)) == len(palette)
    assert "#8B5E34" in palette[:12]
    assert "#38BDF8" in palette[:12]
    assert "#6B7280" not in palette
    assert "#C62828" not in palette

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

    first_colors = palette[:8]
    for index, color_a in enumerate(first_colors):
        red_a, green_a, blue_a = _rgb_triplet(color_a)
        for color_b in first_colors[index + 1 :]:
            red_b, green_b, blue_b = _rgb_triplet(color_b)
            distance = (
                abs(red_a - red_b) + abs(green_a - green_b) + abs(blue_a - blue_b)
            )
            assert distance >= 115


def test_well_color_map_restarts_palette_for_each_pad() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_multi_pad_records())

    color_map = page._well_color_map(records)
    _, _, well_names_by_pad_id = page._pad_membership(records)

    assert len(well_names_by_pad_id) == 2
    for ordered_names in well_names_by_pad_id.values():
        for index, name in enumerate(ordered_names):
            assert color_map[str(name)] == page._well_color(index)
    first_well_names = [str(names[0]) for names in well_names_by_pad_id.values()]
    assert color_map[first_well_names[0]] == color_map[first_well_names[1]]


def test_well_color_map_uses_parent_color_for_pilot() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = [
        WelltrackRecord(
            name="well_04",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=2.0),
                WelltrackPoint(x=300.0, y=0.0, z=1000.0, md=3.0),
            ),
        ),
        WelltrackRecord(
            name="well_04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=50.0, y=0.0, z=800.0, md=2.0),
            ),
        ),
    ]

    color_map = page._well_color_map(records)

    assert color_map["well_04_PL"] == color_map["well_04"]


def test_reference_trajectory_text_import_populates_reference_wells_state(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    actual_path = tmp_path / "actual_wells.inc"
    actual_path.write_text(
        "\n".join(
            [
                "WELLTRACK 'FACT-1'",
                "0 0 25 0",
                "950 900 25 300",
                "1900 1800 25 400",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    at.session_state["ptc_reference_source_mode::actual"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_actual_welltrack_path"] = str(actual_path)
    approved_path = tmp_path / "approved_wells.inc"
    approved_path.write_text(
        "\n".join(
            [
                "WELLTRACK 'APP-1'",
                "0 0 -35 0",
                "900 850 -35 250",
                "1850 1750 -35 350",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    at.session_state["ptc_reference_source_mode::approved"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_approved_welltrack_path"] = str(approved_path)

    at.run()
    _click_button(at, "Загрузить фактические скважины")
    at.run()
    _click_button(at, "Загрузить проектные утвержденные скважины")
    at.run()

    reference_wells = tuple(at.session_state["wt_reference_wells"])
    assert len(reference_wells) == 2
    assert [str(item.name) for item in reference_wells] == ["FACT-1", "APP-1"]


def test_reference_trajectory_welltrack_path_import_populates_reference_wells_state(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    welltrack_path = tmp_path / "approved_wells.inc"
    welltrack_path.write_text(
        "\n".join(
            [
                "WELLTRACK 'APP-1'",
                "0 0 -35 0",
                "900 850 -35 250",
                "1850 1750 -35 350",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    at.session_state["wt_reference_approved_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_approved_welltrack_path"] = str(welltrack_path)

    at.run()
    _click_button(at, "Загрузить проектные утвержденные скважины")
    at.run(timeout=120)

    approved_wells = tuple(at.session_state["wt_reference_approved_wells"])
    assert len(approved_wells) == 1
    assert str(approved_wells[0].name) == "APP-1"


def test_reference_trajectory_dev_folder_import_is_default_and_uses_file_names(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    folder_a = tmp_path / "dev_fact_a"
    folder_b = tmp_path / "dev_fact_b"
    folder_a.mkdir()
    folder_b.mkdir()
    dev_text = "\n".join(
        [
            "# SURVEY FROM PETRAL",
            "MD X Y Z TVD",
            "0 606207.5 7409801.6 40.9 0",
            "100 606208.5 7409803.6 -59.1 100",
            "250 606220.5 7409810.6 -209.1 250",
        ]
    )
    (folder_a / "well_111.dev").write_text(dev_text, encoding="utf-8")
    (folder_b / "well_222.dev").write_text(dev_text, encoding="utf-8")
    at.session_state["wt_reference_actual_dev_folder_count"] = 2
    at.session_state["wt_reference_actual_dev_folder_path_0"] = str(folder_a)
    at.session_state["wt_reference_actual_dev_folder_path_1"] = str(folder_b)

    at.run(timeout=120)
    _click_button(at, "Загрузить фактические скважины")
    at.run(timeout=120)

    actual_wells = tuple(at.session_state["wt_reference_actual_wells"])
    assert [str(well.name) for well in actual_wells] == ["well_111", "well_222"]
    assert [float(actual_wells[0].stations["X_m"].iloc[0])] == [606207.5]
    assert (
        str(at.session_state["ptc_reference_source_mode::actual"]) == "Загрузить .dev"
    )


def test_reference_trajectory_dev_import_keeps_actual_and_approved_kinds(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    actual_folder = tmp_path / "dev_fact"
    approved_folder = tmp_path / "dev_approved"
    actual_folder.mkdir()
    approved_folder.mkdir()
    dev_text = "\n".join(
        [
            "MD X Y Z",
            "0 606207.5 7409801.6 40.9",
            "100 606208.5 7409803.6 -59.1",
            "250 606220.5 7409810.6 -209.1",
        ]
    )
    (actual_folder / "well_same.dev").write_text(dev_text, encoding="utf-8")
    (approved_folder / "well_same.dev").write_text(dev_text, encoding="utf-8")
    at.session_state["wt_reference_actual_dev_folder_path_0"] = str(actual_folder)
    at.session_state["wt_reference_approved_dev_folder_path_0"] = str(approved_folder)

    at.run(timeout=120)
    _click_button(at, "Загрузить фактические скважины")
    at.run(timeout=120)
    _click_button(at, "Загрузить проектные утвержденные скважины")
    at.run(timeout=120)

    reference_wells = tuple(at.session_state["wt_reference_wells"])
    assert [(str(well.name), str(well.kind)) for well in reference_wells] == [
        ("well_same", "actual"),
        ("well_same", "approved"),
    ]


def test_reference_dev_clear_button_resets_paths_without_widget_state_error(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    folder_a = tmp_path / "dev_fact_a"
    folder_b = tmp_path / "dev_fact_b"
    folder_a.mkdir()
    folder_b.mkdir()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["ptc_reference_source_mode::actual"] = "Загрузить .dev"
    at.session_state["wt_reference_actual_dev_folder_count"] = 2
    at.session_state["wt_reference_actual_dev_folder_path_0"] = str(folder_a)
    at.session_state["wt_reference_actual_dev_folder_path_1"] = str(folder_b)

    at.run(timeout=120)
    _click_button(at, "Очистить фактические скважины")
    at.run(timeout=120)

    assert not at.exception
    assert int(at.session_state["wt_reference_actual_dev_folder_count"]) == 1
    assert str(at.session_state["wt_reference_actual_dev_folder_path_0"]) == ""
    assert str(at.session_state["wt_reference_actual_dev_folder_path_1"]) == ""


def test_reference_welltrack_clear_button_resets_path_without_widget_state_error(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    welltrack_path = tmp_path / "actual.inc"
    welltrack_path.write_text(
        "\n".join(
            [
                "WELLTRACK 'FACT-1'",
                "0 0 25 0",
                "950 900 25 300",
                "1900 1800 25 400",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["ptc_reference_source_mode::actual"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_actual_welltrack_path"] = str(welltrack_path)

    at.run(timeout=120)
    _click_button(at, "Очистить фактические скважины")
    at.run(timeout=120)

    assert not at.exception
    assert str(at.session_state["wt_reference_actual_welltrack_path"]) == ""


def test_welltrack_page_focuses_follow_up_selection_on_unresolved_wells() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
            "Отход t1, м": "—",
            "Длина ГС, м": "—",
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
            "Отход t1, м": "—",
            "Длина ГС, м": "—",
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
    assert _multiselect_value(at, "Скважины для расчёта") == expected
    assert [widget.label for widget in at.metric].count("Без замечаний") == 1
    assert [widget.label for widget in at.metric].count("С предупреждениями") == 1
    checkbox_labels = [widget.label for widget in at.checkbox]
    assert (
        "Использовать отдельные параметры для повторного расчета" not in checkbox_labels
    )


def test_welltrack_successful_batch_run_clears_stale_error_and_updates_selection_on_next_run() -> (
    None
):
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_last_error"] = "Batch-расчет завершился ошибкой"

    at.run()
    _click_button(at, "Рассчитать траектории")
    at.run(timeout=120)

    assert [error.value for error in at.error] == []
    assert at.session_state["wt_last_error"] == ""

    metrics = {widget.label: widget.value for widget in at.metric}
    assert metrics["Без замечаний"] == "3"
    assert metrics["Ошибки"] == "0"

    at.run()
    assert _multiselect_value(at, "Скважины для расчёта") == []


def test_welltrack_page_keeps_single_general_run_form_after_results_exist() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
            "Отход t1, м": "—",
            "Длина ГС, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс ПИ, deg/10m": "—",
            "Макс MD, м": "—",
            "Проблема": "",
        },
    ]

    at.run()

    multiselect_labels = [widget.label for widget in at.multiselect]
    assert multiselect_labels.count("Скважины для расчёта") == 1
    assert "Скважины для повторного расчета" not in multiselect_labels


def test_welltrack_import_auto_applies_pad_layout_for_shared_surface_and_can_reset() -> (
    None
):
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_mode"] = "Файл по пути"
    at.session_state["wt_source_path"] = "tests/test_data/WELLTRACKS2.INC"

    at.run(timeout=120)
    _click_button(at, "Импорт целей")
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


def test_auto_pad_layout_applies_for_each_multi_pad_cluster_with_shared_surface() -> (
    None
):
    page = wt_import_module
    page._init_state()

    records = _multi_pad_records()
    applied = page._auto_apply_pad_layout_if_shared_surface(list(records))

    assert applied is True
    updated_records = page.st.session_state["wt_records"]
    updated_surfaces = _surface_points(updated_records)
    assert len(set(updated_surfaces)) == 4
    assert updated_surfaces[0] != updated_surfaces[1]
    assert updated_surfaces[2] != updated_surfaces[3]
    assert page.st.session_state["wt_pad_auto_applied_on_import"] is True
    assert page.st.session_state["wt_pad_last_applied_at"] != ""


def test_fresh_import_prefills_pilot_parent_as_first_fixed_slot() -> None:
    page = wt_import_module
    page._init_state()
    records = [
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=1000.0),
            ),
        ),
    ]

    page._store_parsed_records(records)
    pads = page._ensure_pad_configs(
        list(page.st.session_state["wt_records_original"])
    )
    pad_id = str(pads[0].pad_id)

    assert page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] == (
        (1, "WELL-A"),
    )
    assert [str(well.name) for well in pads[0].wells] == ["WELL-B", "WELL-A"]


def test_pad_config_defaults_to_center_anchor_mode() -> None:
    page = wt_import_module
    pads = detect_well_pads(_multi_pad_records())

    defaults = page._pad_config_defaults(pads[0])

    assert str(defaults["surface_anchor_mode"]) == page.PAD_SURFACE_ANCHOR_CENTER


def test_project_pads_for_ui_groups_prepositioned_surfaces_into_single_pad() -> None:
    page = wt_import_module
    page.st.session_state["wt_records_original"] = list(_prepositioned_pad_records())

    pads = page._project_pads_for_ui(_prepositioned_pad_records())

    assert [len(pad.wells) for pad in pads] == [3, 2]
    metadata = page.st.session_state["wt_pad_detected_meta"]
    assert bool(metadata[str(pads[0].pad_id)].source_surfaces_defined) is True
    assert bool(metadata[str(pads[1].pad_id)].source_surfaces_defined) is True


def test_project_pads_for_ui_detects_submeter_surface_offsets_as_prepositioned() -> (
    None
):
    page = wt_import_module
    page.st.session_state["wt_records_original"] = list(
        _submeter_prepositioned_pad_records()
    )

    pads = page._project_pads_for_ui(_submeter_prepositioned_pad_records())

    assert [len(pad.wells) for pad in pads] == [2]
    metadata = page.st.session_state["wt_pad_detected_meta"]
    assert bool(metadata[str(pads[0].pad_id)].source_surfaces_defined) is True


def test_build_pad_plan_map_skips_prepositioned_source_surface_pads() -> None:
    page = wt_import_module

    pads = page._ensure_pad_configs(list(_prepositioned_pad_records()))
    plan_map = page._build_pad_plan_map(pads)

    assert plan_map == {}


def test_pad_config_preserves_fixed_slots_for_prepositioned_source_pads() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_prepositioned_pad_records())
    pads = page._ensure_pad_configs(records)
    pad_id = str(pads[0].pad_id)
    page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] = (
        (1, "P1-C"),
        (2, "P1-A"),
    )

    refreshed_pads = page._ensure_pad_configs(records)
    cfg = page._pad_config_for_ui(refreshed_pads[0])
    fixed_names = page._focus_pad_fixed_well_names(
        records=records,
        focus_pad_id=pad_id,
    )

    assert cfg["fixed_slots"] == ((1, "P1-C"), (2, "P1-A"))
    assert fixed_names == ("P1-C", "P1-A")


def test_build_pad_plan_map_preserves_fixed_pad_slots() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_records())
    pads = page._ensure_pad_configs(records)
    pad_id = str(pads[0].pad_id)
    page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] = (
        (1, "WELL-B"),
        (2, "WELL-A"),
    )

    plan_map = page._build_pad_plan_map(pads)

    assert plan_map[pad_id].fixed_slots == (
        (1, "WELL-B"),
        (2, "WELL-A"),
    )


def test_pad_layout_fixed_position_column_uses_slot_selectbox(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    captured: dict[str, object] = {
        "expander_labels": [],
        "number_columns": [],
        "selectbox_columns": [],
    }

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def number_input(self, *args, **kwargs):
            key = kwargs.get("key")
            return page.st.session_state.get(key, 0.0)

        def button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_selectbox(label, options, *args, **kwargs):
        key = kwargs.get("key")
        value = page.st.session_state.get(key, options[0])
        if key is not None:
            page.st.session_state[key] = value
        return value

    def _fake_data_editor(frame, **kwargs):
        captured["column_config"] = dict(kwargs.get("column_config", {}))
        return frame

    def _fake_expander(label, *args, **kwargs):
        captured["expander_labels"].append(str(label))
        return _DummyContext()

    def _fake_number_column(label, *args, **kwargs):
        captured["number_columns"].append(str(label))
        return {"type": "number", "label": str(label), **kwargs}

    def _fake_selectbox_column(label, *args, **kwargs):
        captured["selectbox_columns"].append(
            {"label": str(label), "kwargs": dict(kwargs)}
        )
        return {"type": "selectbox", "label": str(label), **kwargs}

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", _fake_data_editor)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st.column_config, "NumberColumn", _fake_number_column)
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        _fake_selectbox_column,
    )

    page._render_pad_layout_panel(list(_records()))

    position_column = captured["column_config"]["Позиция"]
    position_selectbox = next(
        item for item in captured["selectbox_columns"] if item["label"] == "Позиция"
    )

    assert position_column["type"] == "selectbox"
    assert position_selectbox["kwargs"]["options"] == [1, 2, 3]
    assert "Позиция" not in captured["number_columns"]
    assert captured["expander_labels"] == [
        "Порядок бурения и координаты устьев на кусте"
    ]


def test_pad_layout_clear_fixed_slots_resets_editor_key(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_records())
    pads = page._ensure_pad_configs(records)
    pad_id = str(pads[0].pad_id)
    page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] = ((1, "WELL-B"),)
    captured: dict[str, object] = {
        "editor_keys": [],
        "editor_lengths": [],
    }

    class _Rerun(Exception):
        pass

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def number_input(self, *args, **kwargs):
            key = kwargs.get("key")
            return page.st.session_state.get(key, 0.0)

        def button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_selectbox(label, options, *args, **kwargs):
        key = kwargs.get("key")
        value = page.st.session_state.get(key, options[0])
        if key is not None:
            page.st.session_state[key] = value
        return value

    def _fake_data_editor(frame, **kwargs):
        captured["editor_keys"].append(str(kwargs.get("key")))
        captured["editor_lengths"].append(len(frame))
        return frame

    def _fake_clear_button(label, *args, **kwargs):
        return str(label) == "Очистить фиксацию порядка"

    def _fake_selectbox_column(label, *args, **kwargs):
        return {"type": "selectbox", "label": str(label), **kwargs}

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", _fake_data_editor)
    monkeypatch.setattr(page.st, "button", _fake_clear_button)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda: (_ for _ in ()).throw(_Rerun()))
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        _fake_selectbox_column,
    )

    with pytest.raises(_Rerun):
        page._render_pad_layout_panel(records)

    assert page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] == ()
    assert page.st.session_state[f"wt_pad_fixed_slots_editor_revision_{pad_id}"] == 1
    assert captured["editor_keys"] == [f"wt_pad_fixed_slots_editor_{pad_id}_0"]
    assert captured["editor_lengths"] == [1]

    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    page._render_pad_layout_panel(records)

    assert captured["editor_keys"][-1] == f"wt_pad_fixed_slots_editor_{pad_id}_1"
    assert captured["editor_lengths"][-1] == 0


def test_welltrack_page_marks_prepositioned_surface_pad_as_read_only_reference() -> (
    None
):
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _prepositioned_pad_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run(timeout=120)

    assert any(
        "Положения устьев были заданы в исходных данных" in str(widget.value)
        for widget in at.info
    )


def test_welltrack_import_source_selector_splits_format_and_welltrack_method() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")

    at.run(timeout=120)

    radios_by_label = {str(widget.label): widget for widget in at.radio}
    assert list(radios_by_label["Формат импорта"].options) == [
        "WELLTRACK",
        "Таблица с точками целей",
    ]
    assert radios_by_label["Формат импорта"].value == "WELLTRACK"
    assert list(radios_by_label["Способ загрузки WELLTRACK"].options) == [
        "Файл по пути",
        "Загрузить файл",
        "Вставить текст",
    ]


def test_welltrack_import_target_table_format_hides_welltrack_method() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_format"] = "Таблица с точками целей"

    at.run(timeout=120)

    radio_labels = {str(widget.label) for widget in at.radio}
    assert radio_labels == {"Формат импорта"}
    assert [str(widget.label) for widget in at.expander] == ["Таблица точек целей"]


def test_welltrack_import_accepts_tabular_point_editor_mode() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_mode"] = "Вставить таблицу"
    at.session_state["wt_source_table_df"] = pd.DataFrame(
        [
            {"Wellname": "TAB-01", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "TAB-01", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {
                "Wellname": "TAB-01",
                "Point": "t3",
                "X": 1500.0,
                "Y": 2000.0,
                "Z": 2500.0,
            },
            {"Wellname": "TAB-02", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "TAB-02", "Point": "t1", "X": 650.0, "Y": 780.0, "Z": 2300.0},
            {
                "Wellname": "TAB-02",
                "Point": "t3",
                "X": 1550.0,
                "Y": 1980.0,
                "Z": 2400.0,
            },
        ]
    )

    at.run(timeout=120)
    _click_button(at, "Импорт целей")
    at.run(timeout=120)

    records = at.session_state["wt_records"]
    assert [record.name for record in records] == ["TAB-01", "TAB-02"]
    assert records[0].points[1].x == pytest.approx(600.0)
    assert records[1].points[2].y == pytest.approx(1980.0)


def test_normalize_source_table_df_for_ui_accepts_excel_like_single_column_rows() -> (
    None
):
    page = wt_import_module

    normalized = page._normalize_source_table_df_for_ui(
        pd.DataFrame(
            {
                "Column 1": [
                    "TAB-01\tS\t0\t0\t0",
                    "TAB-01\tt1\t600,5\t800,25\t2400,75",
                    "TAB-01\tt3\t1500\t2000\t2500",
                ]
            }
        )
    )

    assert list(normalized.columns) == ["Wellname", "Point", "X", "Y", "Z"]
    assert normalized.iloc[0].to_dict()["Point"] == "S"
    assert str(normalized.iloc[1]["X"]) == "600,5"


def test_normalize_source_table_df_for_ui_uses_s_for_surface_point() -> None:
    page = wt_import_module
    normalized = page._normalize_source_table_df_for_ui(
        pd.DataFrame(
            [
                {
                    "Wellname": "TAB-01",
                    "Point": "wellhead",
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                },
                {"Wellname": "TAB-01", "Point": "s", "X": 1.0, "Y": 2.0, "Z": 0.0},
                {
                    "Wellname": "TAB-01",
                    "Point": "t1",
                    "X": 600.0,
                    "Y": 800.0,
                    "Z": 2400.0,
                },
            ]
        )
    )

    assert list(normalized["Point"]) == ["S", "S", "t1"]


def test_welltrack_page_renders_anticollision_metrics_for_successful_batch() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
            "Отход t1, м": "—",
            "Длина ГС, м": "—",
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
            "Отход t1, м": "—",
            "Длина ГС, м": "—",
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
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
            "Отход t1, м": "—",
            "Длина ГС, м": "—",
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
            "Отход t1, м": "—",
            "Длина ГС, м": "—",
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


def test_welltrack_page_renders_actual_fund_well_detail_viewer() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()[:2]
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_reference_actual_wells"] = list(_horizontal_reference_well())

    at.run(timeout=120)

    detail_selectboxes = [
        widget
        for widget in at.selectbox
        if widget.label == "Просмотр фактической скважины"
    ]
    assert detail_selectboxes
    assert str(detail_selectboxes[0].value) == "FACT-H"


def test_actual_fund_vertical_profile_uses_explicit_reversed_tvd_range() -> None:
    page = wt_import_module
    detail = page.build_actual_fund_well_analyses(list(_horizontal_reference_well()))[0]

    figure = page._actual_fund_vertical_profile_figure(detail)

    assert figure.layout.yaxis.range is not None
    assert float(figure.layout.yaxis.range[0]) > float(figure.layout.yaxis.range[1])
    assert figure.layout.yaxis.autorange is None
    assert not str(getattr(figure.layout.title, "text", "") or "").strip()
    assert str(figure.layout.xaxis.title.text) == "Координата по разрезу (м)"


def test_actual_fund_plan_figure_uses_equalized_xy_view_without_embedded_title() -> (
    None
):
    page = wt_import_module
    detail = page.build_actual_fund_well_analyses(list(_horizontal_reference_well()))[0]

    figure = page._actual_fund_plan_figure(detail)

    assert not str(getattr(figure.layout.title, "text", "") or "").strip()
    assert str(figure.layout.yaxis.scaleanchor) == "x"


def test_actual_fund_profile_prefers_horizontal_azimuth_over_hold_azimuth() -> None:
    page = wt_import_module
    detail = page.build_actual_fund_well_analyses(
        list(_turned_horizontal_reference_well())
    )[0]

    assert detail.metrics.hold_azi_deg is not None
    assert float(detail.metrics.hold_azi_deg) < 10.0

    profile_azimuth = float(page._reference_profile_azimuth_deg(detail))

    assert 75.0 <= profile_azimuth <= 105.0

    figure = page._actual_fund_vertical_profile_figure(detail)
    horizontal_trace = next(
        trace for trace in figure.data if str(trace.name) == "Горизонтальный"
    )
    horizontal_x = np.asarray(horizontal_trace.x, dtype=float)
    assert float(np.max(horizontal_x) - np.min(horizontal_x)) > 300.0


def test_render_raw_records_table_hides_md_from_file_column(monkeypatch) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}
    page.st.session_state.pop("wt_edit_targets_highlight_names", None)

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_expander(*args, **kwargs):
        return _DummyExpander()

    def _fake_dataframe(frame, **kwargs):
        captured["frame"] = frame.copy()

    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)

    page._render_raw_records_table(_records()[:1])

    frame = captured["frame"]
    assert "MD (из файла), м" not in list(frame.columns)
    assert list(frame.columns) == [
        "Скважина",
        "Точка",
        "X, м",
        "Y, м",
        "Z, м",
    ]


def test_render_raw_records_table_shows_pilot_points(monkeypatch) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}
    page.st.session_state.pop("wt_edit_targets_highlight_names", None)

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_expander(*args, **kwargs):
        return _DummyExpander()

    def _fake_dataframe(frame, **kwargs):
        captured["frame"] = frame.copy()

    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)

    records = [
        *_records()[:1],
        WelltrackRecord(
            name="WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=200.0, z=1800.0, md=1.0),
                WelltrackPoint(x=300.0, y=500.0, z=2400.0, md=2.0),
            ),
        ),
    ]

    page._render_raw_records_table(records)

    frame = captured["frame"]
    pilot_rows = frame.loc[frame["Скважина"] == "WELL-A_PL"]
    assert list(pilot_rows["Точка"]) == ["S", "PL1", "PL2"]


def test_apply_three_edit_targets_preserves_unchanged_results() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_records()[:2])
    page.st.session_state["wt_records"] = list(records)
    page.st.session_state["wt_records_original"] = list(records)
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    page.st.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
    ]
    cached_analysis = {"key": "previous-full-scan"}
    page.st.session_state["wt_anticollision_analysis_cache"] = cached_analysis
    page.st.session_state["wt_results_view_mode"] = "Все скважины"
    page.st.session_state["wt_results_all_view_mode"] = "Anti-collision"

    updated = page._apply_edit_targets_changes(
        [
            {
                "name": "WELL-A",
                "t1": [610.25, 805.5, 2401.0],
                "t3": [1510.75, 2010.25, 2502.0],
            }
        ],
        source="three_viewer",
    )

    assert updated == ["WELL-A"]
    changed_record = page.st.session_state["wt_records"][0]
    changed_original_record = page.st.session_state["wt_records_original"][0]
    assert changed_record.points[1].x == pytest.approx(610.25)
    assert changed_record.points[1].y == pytest.approx(805.5)
    assert changed_record.points[1].z == pytest.approx(2401.0)
    assert changed_record.points[2].x == pytest.approx(1510.75)
    assert changed_record.points[2].y == pytest.approx(2010.25)
    assert changed_record.points[2].z == pytest.approx(2502.0)
    assert changed_original_record.points[1].x == pytest.approx(610.25)
    assert changed_original_record.points[2].x == pytest.approx(1510.75)
    assert [item.name for item in page.st.session_state["wt_successes"]] == ["WELL-B"]
    summary_rows = page.st.session_state["wt_summary_rows"]
    assert [str(row["Скважина"]) for row in summary_rows] == [
        "WELL-A",
        "WELL-B",
    ]
    assert summary_rows[0]["Статус"] == "Не рассчитана"
    assert summary_rows[1]["Статус"] == "OK"
    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-A"]
    assert page.st.session_state["wt_edit_targets_pending_names"] == ["WELL-A"]
    assert page.st.session_state["wt_edit_targets_highlight_names"] == ["WELL-A"]
    assert page.st.session_state["wt_results_view_mode"] == "Все скважины"
    assert page.st.session_state["wt_results_all_view_mode"] == "Anti-collision"
    assert bool(page.st.session_state["wt_pending_all_wells_results_focus"]) is True
    assert page.st.session_state["wt_anticollision_analysis_cache"] is cached_analysis
    target_only = page._failed_target_only_wells(
        records=page.st.session_state["wt_records"],
        summary_rows=summary_rows,
    )
    assert [item.name for item in target_only] == ["WELL-A"]

    page._store_merged_batch_results(
        records=page.st.session_state["wt_records"],
        new_rows=[{"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""}],
        new_successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
    )

    assert [item.name for item in page.st.session_state["wt_successes"]] == [
        "WELL-A",
        "WELL-B",
    ]
    assert page.st.session_state["wt_edit_targets_pending_names"] == []
    assert page.st.session_state["wt_edit_targets_highlight_names"] == []
    assert page.st.session_state["wt_anticollision_analysis_cache"] == {}


def test_apply_three_edit_targets_defers_result_widget_state_update() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_records()[:1])
    page.st.session_state["wt_records"] = list(records)
    page.st.session_state["wt_records_original"] = list(records)
    page.st.session_state["wt_results_view_mode"] = "Отдельная скважина"
    page.st.session_state["wt_results_all_view_mode"] = "Anti-collision"

    updated = page._apply_edit_targets_changes(
        [
            {
                "name": "WELL-A",
                "t1": [610.25, 805.5, 2401.0],
                "t3": [1510.75, 2010.25, 2502.0],
            }
        ],
        source="three_viewer",
    )

    assert updated == ["WELL-A"]
    assert page.st.session_state["wt_results_view_mode"] == "Отдельная скважина"
    assert bool(page.st.session_state["wt_pending_all_wells_results_focus"]) is True

    page._init_state()

    assert page.st.session_state["wt_results_view_mode"] == "Все скважины"
    assert page.st.session_state["wt_results_all_view_mode"] == "Anti-collision"
    assert "wt_3d_backend" not in page.st.session_state
    assert page.st.session_state["wt_pending_all_wells_results_focus"] is False


def test_selected_override_configs_apply_kop_depth_function_per_well_depth() -> None:
    page = wt_import_module
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)
    kop_function = ActualFundKopDepthFunction(
        mode="piecewise_linear",
        cluster_count=3,
        anchor_depths_tvd_m=(1600.0, 2500.0, 3400.0),
        anchor_kop_md_m=(780.0, 1180.0, 1680.0),
        note="test",
    )
    set_kop_min_vertical_function(
        prefix=page.WT_CALC_PARAMS.prefix, kop_function=kop_function
    )
    base_config = TrajectoryConfig()
    records = {record.name: record for record in _records()}

    config_map = page._build_selected_override_configs(
        base_config=base_config,
        selected_names={"WELL-A", "WELL-C"},
        records_by_name=records,
    )

    assert set(config_map) == {"WELL-A", "WELL-C"}
    assert float(config_map["WELL-A"].kop_min_vertical_m) == pytest.approx(
        1135.5555555,
        rel=1e-6,
    )
    assert float(config_map["WELL-C"].kop_min_vertical_m) == pytest.approx(
        1046.6666666,
        rel=1e-6,
    )
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)


def test_focus_all_wells_anticollision_results_sets_result_view_state() -> None:
    page = wt_import_module

    page.st.session_state.clear()
    page._init_state()
    page._focus_all_wells_anticollision_results()

    assert page.st.session_state["wt_results_view_mode"] == "Все скважины"
    assert page.st.session_state["wt_results_all_view_mode"] == "Anti-collision"
    assert page.st.session_state["wt_3d_render_mode"] == page.WT_3D_RENDER_DETAIL
    assert "wt_3d_backend" not in page.st.session_state


def test_init_state_defaults_to_all_wells_detail_three_render_mode() -> None:
    page = wt_import_module

    page.st.session_state.clear()
    page._init_state()

    assert page.st.session_state["wt_results_view_mode"] == "Все скважины"
    assert page.st.session_state["wt_3d_render_mode"] == page.WT_3D_RENDER_DETAIL
    assert "wt_3d_backend" not in page.st.session_state
    assert page.WT_3D_RENDER_AUTO not in page.WT_3D_RENDER_OPTIONS


def test_focus_all_wells_trajectory_results_sets_detail_render_mode() -> None:
    page = wt_import_module

    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_3d_render_mode"] = page.WT_3D_RENDER_FAST
    page.st.session_state["wt_3d_backend"] = "removed-backend"

    page._focus_all_wells_trajectory_results()

    assert page.st.session_state["wt_results_view_mode"] == "Все скважины"
    assert page.st.session_state["wt_results_all_view_mode"] == "Anti-collision"
    assert page.st.session_state["wt_3d_render_mode"] == page.WT_3D_RENDER_DETAIL
    assert "wt_3d_backend" not in page.st.session_state


def test_render_three_payload_uses_local_three_renderer(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    captured: dict[str, object] = {}

    class _DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def plotly_chart(self, *args, **kwargs):
            raise AssertionError("3D chart fallback should not be used in PTC")

    def _fake_three_scene(payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = dict(kwargs)
        return None

    monkeypatch.setattr(page, "render_local_three_scene", _fake_three_scene)
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_THREE_CAMERA,
        "lines": [
            {
                "segments": [[[0.0, 0.0, 0.0], [10.0, 0.0, 5.0]]],
                "color": "#15D562",
                "opacity": 1.0,
                "dash": "solid",
                "role": "line",
            }
        ],
        "points": [],
        "meshes": [],
        "labels": [],
        "legend": [],
    }

    page._render_three_payload(
        container=_DummyContainer(),
        payload=payload,
        height=420,
    )

    assert captured["kwargs"]["height"] == 420
    assert captured["payload"]["lines"]


def test_welltrack_page_respects_anticollision_result_focus_state() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
            "Отход t1, м": "1000.00",
            "KOP MD, м": "700.00",
            "Длина ГС, м": "1000.00",
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
    radio_labels = {str(widget.label) for widget in at.radio}
    assert str(result_mode.value) == "Все скважины"
    assert "Режим отображения всех скважин" not in radio_labels
    assert str(at.session_state["wt_results_all_view_mode"]) == "Anti-collision"


def test_batch_summary_moves_survey_downloads_into_trajectory_export_expander(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    page.st.session_state["wt_survey_download_selected_names"] = ["WELL-B"]
    captured: dict[str, object] = {
        "download_buttons": [],
        "expanders": [],
        "multiselect": None,
        "radio": None,
    }

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def metric(self, *args, **kwargs):
            return None

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyContext() for _ in range(count))

    def _fake_expander(label, *args, **kwargs):
        captured["expanders"].append(str(label))
        return _DummyContext()

    def _fake_multiselect(label, options, *args, **kwargs):
        captured["multiselect"] = {
            "label": str(label),
            "options": [str(option) for option in options],
        }
        return list(page.st.session_state.get(kwargs.get("key"), []))

    def _fake_radio(label, options, *args, **kwargs):
        captured["radio"] = {
            "label": str(label),
            "options": [str(option) for option in options],
            "key": str(kwargs.get("key")),
        }
        return str(page.st.session_state.get(kwargs.get("key"), options[0]))

    def _fake_download_button(label, *args, **kwargs):
        captured["download_buttons"].append(
            {
                "label": str(label),
                "data": kwargs.get("data", b""),
                "disabled": bool(kwargs.get("disabled", False)),
                "file_name": str(kwargs.get("file_name", "")),
                "mime": str(kwargs.get("mime", "")),
            }
        )
        return False

    monkeypatch.setattr(page, "render_small_note", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "radio", _fake_radio)
    monkeypatch.setattr(page.st, "download_button", _fake_download_button)

    page._render_batch_summary(
        [
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
        ]
    )

    downloads = list(captured["download_buttons"])
    download_labels = {str(item["label"]) for item in downloads}
    assert "Скачать сводку (CSV)" not in download_labels
    assert "Выгрузка траекторий" in captured["expanders"]
    assert captured["multiselect"] == {
        "label": "Скважины для выгрузки",
        "options": ["WELL-A", "WELL-B"],
    }
    assert captured["radio"] == {
        "label": "Формат выгрузки",
        "options": ["CSV", "WELLTRACK", ".dev (7z)"],
        "key": "wt_survey_download_format",
    }
    assert download_labels == {
        "Скачать рассчитанные траектории всех скважин",
        "Скачать рассчитанные траектории выбранных скважин",
    }
    selected_download = next(
        item
        for item in downloads
        if item["label"] == "Скачать рассчитанные траектории выбранных скважин"
    )
    selected_csv = bytes(selected_download["data"]).decode("utf-8")
    assert "WELL-B" in selected_csv
    assert "WELL-A" not in selected_csv
    assert selected_download["file_name"] == "welltrack_survey_selected.csv"
    assert selected_download["mime"] == "text/csv"
    assert selected_download["disabled"] is False


def test_batch_summary_dev_export_downloads_7z_or_single_dev(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    page.st.session_state["wt_survey_download_selected_names"] = ["WELL-A"]
    page.st.session_state["wt_survey_download_format"] = ".dev (7z)"
    captured: dict[str, object] = {"download_buttons": []}

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def metric(self, *args, **kwargs):
            return None

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyContext() for _ in range(count))

    def _fake_download_button(label, *args, **kwargs):
        captured["download_buttons"].append(
            {
                "label": str(label),
                "data": kwargs.get("data", b""),
                "disabled": bool(kwargs.get("disabled", False)),
                "file_name": str(kwargs.get("file_name", "")),
                "mime": str(kwargs.get("mime", "")),
            }
        )
        return False

    monkeypatch.setattr(page, "render_small_note", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(
        page.st,
        "multiselect",
        lambda _label, *args, **kwargs: list(
            page.st.session_state.get(kwargs.get("key"), [])
        ),
    )
    monkeypatch.setattr(
        page.st,
        "radio",
        lambda _label, *args, **kwargs: str(
            page.st.session_state.get(kwargs.get("key"))
        ),
    )
    monkeypatch.setattr(page.st, "download_button", _fake_download_button)

    page._render_batch_summary(
        [
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
        ]
    )

    downloads = list(captured["download_buttons"])
    all_download = next(
        item for item in downloads if item["label"] == "Скачать .dev архив всех скважин"
    )
    selected_download = next(
        item for item in downloads if item["label"] == "Скачать .dev выбранной скважины"
    )

    assert bytes(all_download["data"]).startswith(b"7z\xbc\xaf'\x1c")
    assert all_download["file_name"] == "welltrack_survey_all_dev.7z"
    assert all_download["mime"] == "application/x-7z-compressed"
    assert selected_download["file_name"] == "WELL-A.dev"
    assert selected_download["mime"] == "text/plain"
    assert bytes(selected_download["data"]).decode("utf-8").startswith(
        "# SURVEY FROM PYWP"
    )


def test_batch_summary_renders_pilot_sidetrack_details_table(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    pilot = _successful_plan(
        name="well_04_PL",
        y_offset_m=0.0,
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 100.0, 200.0],
                "INC_deg": [0.0, 10.0, 20.0],
                "AZI_deg": [0.0, 90.0, 90.0],
                "X_m": [0.0, 10.0, 20.0],
                "Y_m": [0.0, 0.0, 0.0],
                "Z_m": [0.0, 50.0, 100.0],
                "DLS_deg_per_30m": [0.0, 1.0, 3.0],
                "segment": ["PILOT_1", "PILOT_1", "PILOT_2"],
            }
        ),
    ).validated_copy(
        summary={
            "trajectory_type": "PILOT",
            "pilot_target_count": 2.0,
            "md_total_m": 200.0,
            "max_dls_total_deg_per_30m": 3.0,
        }
    )
    parent = _successful_plan(name="well_04", y_offset_m=0.0).validated_copy(
        summary={
            "trajectory_type": "PILOT_SIDETRACK",
            "pilot_well_name": "well_04_PL",
            "sidetrack_window_md_m": 120.0,
            "sidetrack_window_z_m": 80.0,
            "sidetrack_window_inc_deg": 22.0,
            "sidetrack_window_azi_deg": 135.0,
            "sidetrack_lateral_md_m": 1450.0,
        }
    )
    page.st.session_state["wt_successes"] = [pilot, parent]
    captured: dict[str, object] = {"markdown": [], "frames": []}

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def metric(self, *args, **kwargs):
            return None

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyContext() for _ in range(count))

    def _fake_dataframe(frame, **kwargs):
        payload = frame.data.copy() if hasattr(frame, "data") else frame.copy()
        captured["frames"].append(payload)

    monkeypatch.setattr(page, "render_small_note", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "markdown",
        lambda value, *args, **kwargs: captured["markdown"].append(str(value)),
    )
    monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "multiselect", lambda *args, **kwargs: [])
    monkeypatch.setattr(page.st, "radio", lambda _label, options, **kwargs: options[0])
    monkeypatch.setattr(page.st, "download_button", lambda *args, **kwargs: False)

    page._render_batch_summary(
        [
            {"Скважина": "well_04_PL", "Статус": "OK", "Проблема": "", "Точек": 2},
            {"Скважина": "well_04", "Статус": "OK", "Проблема": "", "Точек": 3},
        ]
    )

    assert "### Скважины с пилотом" in captured["markdown"]
    pilot_frame = captured["frames"][1]
    assert list(pilot_frame["Скважина"]) == ["well_04"]
    assert str(pilot_frame.iloc[0]["Пилот"]) == "well_04_PL"
    assert str(pilot_frame.iloc[0]["BUILD+HOLD до точек пилота"]) == "2"


@pytest.mark.skip(
    reason="PTC page does not render single-event anticollision rerun selectbox present on the removed welltrack_import page"
)
def test_welltrack_page_prepares_vertical_anticollision_rerun_plan() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
            "Отход t1, м": "100.00",
            "KOP MD, м": "820.00",
            "Длина ГС, м": "1000.00",
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
            "Отход t1, м": "100.00",
            "KOP MD, м": "780.00",
            "Длина ГС, м": "1000.00",
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

    assert _multiselect_value(at, "Скважины для расчёта") == ["WELL-A", "WELL-B"]
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
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        all_successes=successes,
    )

    assert context is not None
    assert context.candidate_md_start_m == pytest.approx(1200.0)
    assert context.candidate_md_end_m == pytest.approx(1750.0)
    assert context.references[0].well_name == "WELL-A"
    assert context.references[0].md_start_m == pytest.approx(1100.0)
    assert context.references[0].md_end_m == pytest.approx(1700.0)


def test_build_prepared_trajectory_optimization_context_includes_other_well_cones() -> (
    None
):
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        all_successes=successes,
    )

    assert context is not None
    assert {str(item.well_name) for item in context.references} == {"WELL-A", "WELL-C"}


def test_build_prepared_late_trajectory_context_marks_build2_adjustment() -> None:
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        all_successes=successes,
    )

    assert context is not None
    assert context.prefer_keep_kop is True
    assert context.prefer_keep_build1 is True
    assert context.prefer_adjust_build2 is True


def test_prepared_override_rows_include_sf_before() -> None:
    page = wt_import_module
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
    page = wt_import_module
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

    rows = page._format_prepared_override_scope(
        selected_names=["WELL-A", "WELL-B", "WELL-C"]
    )

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
    page = wt_import_module

    text = page._sf_help_markdown()

    assert "Separation Factor" in text
    assert "SF < 1" in text
    assert "SF > 1" in text


def test_prepared_override_rows_follow_cluster_action_step_order() -> None:
    page = wt_import_module
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
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        uncertainty_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert resolution is not None
    assert float(resolution["after_sf"]) > 1.0
    assert float(resolution["delta_sf"]) > 0.0
    assert str(resolution["status"]) == "Конфликт снят"
    assert str(resolution["uncertainty_preset"]) == DEFAULT_UNCERTAINTY_PRESET


def test_cached_anti_collision_view_model_reuses_analysis_for_identical_inputs(
    monkeypatch,
) -> None:
    page = wt_import_module
    page._init_state()
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
        lambda successes, *, model, name_to_color=None, reference_wells=(), **_kw: (
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        records=[],
    )
    assert first[0] is analysis
    first_run = page.st.session_state["wt_anticollision_last_run"]
    assert bool(first_run["cached"]) is False
    assert int(first_run["pair_count"]) == 0
    assert "Расчёт Anti-collision завершён." in "\n".join(first_run["log_lines"])

    second = page._cached_anti_collision_view_model(
        successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        records=[],
    )
    assert second[0] is analysis
    assert calls["analysis"] == 1
    second_run = page.st.session_state["wt_anticollision_last_run"]
    assert bool(second_run["cached"]) is True
    assert "Использован кэш anti-collision анализа." in "\n".join(
        second_run["log_lines"]
    )


def test_anti_collision_cache_key_includes_reference_mwd_assignment() -> None:
    page = wt_import_module
    reference_wells = _reference_wells()
    successes = [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    poor_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    unknown_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )

    poor_key = page._anti_collision_cache_key(
        successes=successes,
        model=poor_model,
        name_to_color={},
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name={"FACT-1": poor_model},
    )
    unknown_key = page._anti_collision_cache_key(
        successes=successes,
        model=poor_model,
        name_to_color={},
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name={"FACT-1": unknown_model},
    )

    assert poor_key != unknown_key


def test_prepare_rerun_from_cluster_builds_multi_reference_cluster_plan() -> None:
    page = wt_import_module
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
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
                midpoint_xyz=np.array(
                    [[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
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
        str(item.well_name)
        for item in prepared["WELL-B"]["optimization_context"].references
    } == {"WELL-A", "WELL-C"}
    assert {
        str(item.well_name)
        for item in prepared["WELL-A"]["optimization_context"].references
    } == {"WELL-B", "WELL-C"}
    assert str(payload["source"]).startswith("ac-cluster-")
    assert page.st.session_state["wt_pending_selected_names"] == [
        "WELL-C",
        "WELL-B",
        "WELL-A",
    ]
    snapshot = dict(page.st.session_state["wt_prepared_recommendation_snapshot"])
    assert str(snapshot["kind"]) == "cluster"
    assert len(tuple(snapshot["items"])) == 2


def test_prepare_rerun_from_cluster_persists_pad_scoped_target_wells() -> None:
    page = wt_import_module
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
                md_a_start_m=1200.0,
                md_a_end_m=1700.0,
                md_b_start_m=1250.0,
                md_b_end_m=1750.0,
                md_a_values_m=np.array([1200.0, 1700.0], dtype=float),
                md_b_values_m=np.array([1250.0, 1750.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.78, 0.74], dtype=float),
                overlap_depth_values_m=np.array([7.0, 8.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.74,
    )
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    snapshot = page._cluster_snapshot(clusters[0], target_well_names=("WELL-A",))
    assert snapshot["target_well_names"] == ("WELL-A",)
    assert snapshot["focus_well_names"] == ()
    assert page._resolution_snapshot_well_names(snapshot) == ("WELL-A",)


def test_pad_scoped_cluster_target_wells_expand_to_neighbor_cluster_scope() -> None:
    page = wt_import_module
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="PAD1-A", y_offset_m=0.0),
            _successful_plan(name="PAD2-A", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(
            [
                _successful_plan(name="PAD1-A", y_offset_m=0.0),
                _successful_plan(name="PAD2-A", y_offset_m=5.0),
            ]
        ),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    target_names = page._pad_scoped_cluster_target_well_names(
        cluster=clusters[0],
        focus_pad_well_names=("PAD1-A",),
    )
    focus_names = page._pad_scoped_cluster_focus_well_names(
        cluster=clusters[0],
        focus_pad_well_names=("PAD1-A",),
    )

    assert target_names == ("PAD1-A", "PAD2-A")
    assert focus_names == ("PAD1-A",)


def test_prepare_rerun_from_cluster_is_blocked_for_target_spacing_conflicts() -> None:
    page = wt_import_module
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
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
                midpoint_xyz=np.array(
                    [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
    )

    assert page.st.session_state["wt_prepared_well_overrides"] == {}
    assert page.st.session_state["wt_prepared_recommendation_snapshot"] is None
    assert page.st.session_state["wt_pending_selected_names"] is None
    assert "Cluster-level пересчет недоступен" in str(
        page.st.session_state["wt_prepared_override_message"]
    )
    assert "spacing целей" in str(page.st.session_state["wt_prepared_override_message"])


def test_prepare_rerun_from_cluster_merges_vertical_and_trajectory_into_mixed_well_plan() -> (
    None
):
    page = wt_import_module
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
                midpoint_xyz=np.array(
                    [[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
                midpoint_xyz=np.array(
                    [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
        _successful_plan(
            name="WELL-A", y_offset_m=0.0, kop_md_m=700.0, stations=well_a_stations
        ),
        _successful_plan(
            name="WELL-B", y_offset_m=10.0, kop_md_m=900.0, stations=well_b_stations
        ),
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
    )

    prepared = page.st.session_state["wt_prepared_well_overrides"]
    assert "WELL-A" in prepared
    payload = dict(prepared["WELL-A"])
    context = payload["optimization_context"]
    assert context is not None
    assert bool(context.prefer_lower_kop) is True
    assert bool(context.prefer_higher_build1) is True
    assert (
        str(payload["update_fields"]["optimization_mode"]) == "anti_collision_avoidance"
    )
    assert "cone-aware rerun" in str(payload["reason"])
    assert "anti-collision avoidance rerun" in str(payload["reason"])


def test_prepare_rerun_from_cluster_stages_mixed_well_to_early_kop_build1_first() -> (
    None
):
    page = wt_import_module
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
                midpoint_xyz=np.array(
                    [[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
                midpoint_xyz=np.array(
                    [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
    )

    payload = dict(page.st.session_state["wt_prepared_well_overrides"]["WELL-A"])
    context = payload["optimization_context"]
    assert context is not None
    assert float(context.candidate_md_start_m) == pytest.approx(400.0)
    assert float(context.candidate_md_end_m) == pytest.approx(700.0)
    reference_by_name = {str(item.well_name): item for item in context.references}
    assert float(reference_by_name["WELL-C"].md_end_m) == pytest.approx(800.0)
    assert "WELL-C" not in str(payload["reason"])


def test_build_selected_optimization_contexts_rebuilds_prepared_context_from_current_successes() -> (
    None
):
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
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
    stored_context = page.st.session_state["wt_prepared_well_overrides"]["WELL-B"][
        "optimization_context"
    ]
    assert isinstance(stored_context, AntiCollisionOptimizationContext)
    assert float(stored_context.references[0].xyz_m[-1, 0]) == pytest.approx(2000.0)


def test_selected_execution_order_prioritizes_cluster_action_steps() -> None:
    page = wt_import_module
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
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        uncertainty_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert resolution is not None
    assert str(resolution["kind"]) == "cluster"
    assert float(resolution["after_sf"]) >= 1.0
    assert str(resolution["status"]) == "Конфликт снят"
    assert int(resolution["current_cluster_count"]) == 0
    assert tuple(resolution["items"]) == ()


def test_build_last_anticollision_resolution_cluster_rescans_current_conflicts_beyond_snapshot_windows() -> (
    None
):
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
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


def test_prepare_rerun_from_recommendation_skips_trajectory_override_without_context() -> (
    None
):
    page = wt_import_module
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
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
    )

    assert page.st.session_state["wt_prepared_well_overrides"] == {}
    assert (
        "контекст" in str(page.st.session_state["wt_prepared_override_message"]).lower()
    )
    assert page.st.session_state["wt_pending_selected_names"] is None


def test_prepare_rerun_from_recommendation_builds_two_well_pair_plan() -> None:
    page = wt_import_module
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

    page._prepare_rerun_from_recommendation(
        recommendation,
        successes=successes,
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
    )

    prepared = dict(page.st.session_state["wt_prepared_well_overrides"])
    assert set(prepared.keys()) == {"WELL-A", "WELL-B"}
    snapshot = dict(page.st.session_state["wt_prepared_recommendation_snapshot"])
    assert tuple(snapshot["affected_wells"]) == ("WELL-B", "WELL-A")
    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-B", "WELL-A"]


def test_anticollision_three_payload_draws_terminal_cone_boundaries_per_well() -> None:
    page = wt_import_module
    payload = page._all_wells_anticollision_three_payload(
        page._build_anti_collision_analysis(
            [
                _successful_plan(name="WELL-A", y_offset_m=0.0),
                _successful_plan(name="WELL-B", y_offset_m=5.0),
            ],
            model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        )
    )

    boundary_lines = [
        item for item in payload["lines"] if str(item.get("role")) == "cone_tip"
    ]

    assert len(boundary_lines) == 2
    assert {str(item["color"]) for item in boundary_lines} == {
        page._lighten_hex(page._well_color(0), 0.55),
        page._lighten_hex(page._well_color(1), 0.55),
    }
    assert payload["camera"] == DEFAULT_THREE_CAMERA


def test_anticollision_figures_draw_previous_trajectories_as_dashed_overlay() -> None:
    page = wt_import_module
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

    payload_3d = page._all_wells_anticollision_three_payload(
        analysis,
        previous_successes_by_name=previous_successes,
    )
    figure_plan = page._all_wells_anticollision_plan_figure(
        analysis,
        previous_successes_by_name=previous_successes,
    )

    previous_3d = [
        item for item in payload_3d["lines"] if str(item.get("dash")) == "dot"
    ]
    previous_plan = [
        trace
        for trace in figure_plan.data
        if str(trace.name).endswith(": до anti-collision")
    ]
    assert len(previous_3d) == 1
    assert len(previous_plan) == 1
    assert str(previous_3d[0]["dash"]) == "dot"
    assert str(previous_plan[0].line.dash) == "dot"


def test_all_wells_three_payload_uses_default_camera() -> None:
    page = wt_import_module
    payload = page._all_wells_three_payload(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ]
    )

    assert payload["camera"] == DEFAULT_THREE_CAMERA


def test_all_wells_overview_figures_show_targets_for_failed_wells_without_fake_trajectory() -> (
    None
):
    page = wt_import_module
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

    payload_3d = page._all_wells_three_payload(
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

    failed_3d_markers = [
        item for item in payload_3d["points"] if str(item.get("symbol")) == "cross"
    ]
    failed_plan_traces = [
        trace
        for trace in figure_plan.data
        if str(trace.name) == "WELL-C: цели (без траектории)"
    ]
    failed_trajectory_plan = [
        trace for trace in figure_plan.data if str(trace.name) == "WELL-C"
    ]

    assert len(target_only_wells) == 1
    assert failed_3d_markers
    assert failed_plan_traces
    assert not failed_trajectory_plan
    assert failed_3d_markers[0]["hover"][0]["segment"] == "Ошибка расчета"
    assert "Точка: %{customdata[0]}" in str(failed_plan_traces[0].hovertemplate)
    assert str(failed_3d_markers[0]["symbol"]) == "cross"
    assert str(failed_plan_traces[0].marker.symbol) == "x"


def test_all_wells_overview_figures_include_reference_trajectories() -> None:
    page = wt_import_module
    reference_wells = _reference_wells()

    payload_3d = page._all_wells_three_payload(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        reference_wells=reference_wells,
        render_mode=page.WT_3D_RENDER_DETAIL,
    )
    figure_plan = page._all_wells_plan_figure(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        reference_wells=reference_wells,
    )

    trace_names_plan = {str(trace.name) for trace in figure_plan.data}
    hover_names_3d = {
        str(hover.get("name"))
        for item in payload_3d["points"]
        for hover in list(item.get("hover") or [])
    }

    assert "FACT-1" in hover_names_3d
    assert "APP-1" in hover_names_3d
    assert "FACT-1 (Фактическая)" in trace_names_plan
    assert "APP-1 (Проектная утвержденная)" in trace_names_plan

    fact_3d = next(
        item for item in payload_3d["lines"] if str(item["color"]) == "#6B7280"
    )
    approved_3d = next(
        item for item in payload_3d["lines"] if str(item["color"]) == "#C62828"
    )
    assert str(fact_3d["color"]) == "#6B7280"
    assert str(approved_3d["color"]) == "#C62828"
    label_by_text = {str(item["text"]): item for item in payload_3d["labels"]}
    assert "FACT-1" in label_by_text
    assert "APP-1" in label_by_text
    assert str(label_by_text["FACT-1"]["color"]) == "#111111"
    assert str(label_by_text["APP-1"]["color"]) == "#C62828"


def test_actual_reference_mwd_assignment_defaults_to_poor_and_selects_unknown() -> None:
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 100.0,
                    "Y": 0.0,
                    "Z": 500.0,
                    "MD": 510.0,
                },
                {
                    "Wellname": "FACT-2",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 100.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-2",
                    "Type": "actual",
                    "X": 100.0,
                    "Y": 100.0,
                    "Z": 500.0,
                    "MD": 510.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": "approved",
                    "X": 0.0,
                    "Y": 200.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": "approved",
                    "X": 100.0,
                    "Y": 200.0,
                    "Z": 500.0,
                    "MD": 510.0,
                },
            ]
        )
    )

    model_by_name = (
        ptc_anticollision_params.reference_uncertainty_models_for_unknown_actual_names(
            reference_wells,
            unknown_names=("FACT-2", "APP-1", "MISSING"),
        )
    )

    assert set(model_by_name) == {"FACT-1", "FACT-2"}
    assert (
        model_by_name["FACT-1"].iscwsa_tool_code
        == planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
        ).iscwsa_tool_code
    )
    assert (
        model_by_name["FACT-2"].iscwsa_tool_code
        == planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
        ).iscwsa_tool_code
    )

    string_model_by_name = (
        ptc_anticollision_params.reference_uncertainty_models_for_unknown_actual_names(
            reference_wells,
            unknown_names="FACT-2",
        )
    )
    assert (
        string_model_by_name["FACT-2"].iscwsa_tool_code
        == planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
        ).iscwsa_tool_code
    )

    none_model_by_name = (
        ptc_anticollision_params.reference_uncertainty_models_for_unknown_actual_names(
            reference_wells,
            unknown_names=None,
        )
    )
    assert {
        model.iscwsa_tool_code for model in none_model_by_name.values()
    } == {
        planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
        ).iscwsa_tool_code
    }


def test_actual_reference_mwd_models_from_state_does_not_mutate_widget_key() -> None:
    class ReadOnlyWidgetState(dict):
        def __setitem__(self, key: object, value: object) -> None:
            if key in {
                ptc_anticollision_params.ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY,
                ptc_anticollision_params.ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY,
            }:
                raise AssertionError("widget key must not be mutated after render")
            super().__setitem__(key, value)

    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 100.0,
                    "Y": 0.0,
                    "Z": 500.0,
                    "MD": 510.0,
                },
                {
                    "Wellname": "FACT-2",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 100.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-2",
                    "Type": "actual",
                    "X": 100.0,
                    "Y": 100.0,
                    "Z": 500.0,
                    "MD": 510.0,
                },
            ]
        )
    )
    state = ReadOnlyWidgetState(
        {
            ptc_anticollision_params.ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY: [
                "FACT-2",
                "MISSING",
            ]
        }
    )

    model_by_name = ptc_anticollision_params.reference_uncertainty_models_from_state(
        reference_wells,
        state=state,
    )

    assert (
        model_by_name["FACT-1"].iscwsa_tool_code
        == planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
        ).iscwsa_tool_code
    )
    assert (
        model_by_name["FACT-2"].iscwsa_tool_code
        == planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
        ).iscwsa_tool_code
    )
    assert state[ptc_anticollision_params.ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY] == [
        "FACT-2",
        "MISSING",
    ]


def test_actual_reference_mwd_models_from_state_accepts_widget_key_fallback() -> None:
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 100.0,
                    "Y": 0.0,
                    "Z": 500.0,
                    "MD": 510.0,
                },
                {
                    "Wellname": "FACT-2",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 100.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-2",
                    "Type": "actual",
                    "X": 100.0,
                    "Y": 100.0,
                    "Z": 500.0,
                    "MD": 510.0,
                },
            ]
        )
    )
    state = {
        ptc_anticollision_params.ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY: [
            "FACT-2",
        ]
    }

    model_by_name = ptc_anticollision_params.reference_uncertainty_models_from_state(
        reference_wells,
        state=state,
    )

    assert (
        model_by_name["FACT-1"].iscwsa_tool_code
        == planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
        ).iscwsa_tool_code
    )
    assert (
        model_by_name["FACT-2"].iscwsa_tool_code
        == planning_uncertainty_model_for_preset(
            UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
        ).iscwsa_tool_code
    )


def test_reference_well_labels_anchor_at_horizontal_entry_for_horizontal_wells() -> (
    None
):
    page = wt_import_module
    horizontal_well = _horizontal_reference_well()[0]

    anchor = page._reference_label_anchor_point(horizontal_well)

    assert anchor is not None
    assert float(anchor[0]) == pytest.approx(700.0)
    assert float(anchor[1]) == pytest.approx(0.0)
    assert float(anchor[2]) == pytest.approx(1150.0)


def test_reference_pad_labels_group_surface_points_by_chain_and_numeric_prefix() -> (
    None
):
    page = wt_import_module
    reference_wells = parse_reference_trajectory_table(
        [
            {
                "Wellname": "8012",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "8012",
                "Type": "actual",
                "X": 100.0,
                "Y": 0.0,
                "Z": 50.0,
                "MD": 120.0,
            },
            {
                "Wellname": "8013",
                "Type": "actual",
                "X": 250.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "8013",
                "Type": "actual",
                "X": 340.0,
                "Y": 0.0,
                "Z": 60.0,
                "MD": 120.0,
            },
            {
                "Wellname": "8014",
                "Type": "actual",
                "X": 530.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "8014",
                "Type": "actual",
                "X": 620.0,
                "Y": 0.0,
                "Z": 60.0,
                "MD": 120.0,
            },
            {
                "Wellname": "9001",
                "Type": "approved",
                "X": 1200.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9001",
                "Type": "approved",
                "X": 1300.0,
                "Y": 0.0,
                "Z": 60.0,
                "MD": 120.0,
            },
        ]
    )

    pad_labels = page._reference_pad_labels(reference_wells)

    assert [item.label for item in pad_labels] == ["Куст 80", "Куст 90"]
    assert float(pad_labels[0].x) == pytest.approx(0.0)
    assert float(pad_labels[1].x) == pytest.approx(1200.0)

    payload = page._all_wells_three_payload(
        [],
        reference_wells=reference_wells,
        render_mode=page.WT_3D_RENDER_DETAIL,
    )
    assert [
        str(item["text"])
        for item in payload["labels"]
        if str(item.get("role")) == "reference_pad_label"
    ] == ["Куст 80", "Куст 90"]


def test_all_wells_overview_figures_ignore_far_reference_wells_for_axis_zoom() -> None:
    page = wt_import_module
    successes = [_successful_plan(name="WELL-A", y_offset_m=0.0)]

    base_3d = page._all_wells_three_payload(successes)
    base_plan = page._all_wells_plan_figure(successes)
    far_ref_3d = page._all_wells_three_payload(
        successes,
        reference_wells=_far_reference_wells(),
    )
    far_ref_plan = page._all_wells_plan_figure(
        successes,
        reference_wells=_far_reference_wells(),
    )

    assert base_3d["bounds"] == far_ref_3d["bounds"]
    assert tuple(base_plan.layout.xaxis.range) == tuple(far_ref_plan.layout.xaxis.range)
    assert tuple(base_plan.layout.yaxis.range) == tuple(far_ref_plan.layout.yaxis.range)


def test_focus_pad_well_names_return_selected_pad_members_only() -> None:
    page = wt_import_module
    records = _multi_pad_records()

    focus_names = page._focus_pad_well_names(records=records, focus_pad_id="PAD-02")

    assert focus_names == ("PAD2-A", "PAD2-B")


def test_all_wells_figures_focus_camera_on_selected_pad_without_hiding_other_pads() -> (
    None
):
    page = wt_import_module
    successes = [
        _successful_plan_xy(name="PAD1-A", x_offset_m=0.0, y_offset_m=0.0),
        _successful_plan_xy(name="PAD1-B", x_offset_m=0.0, y_offset_m=30.0),
        _successful_plan_xy(name="PAD2-A", x_offset_m=6000.0, y_offset_m=0.0),
        _successful_plan_xy(name="PAD2-B", x_offset_m=6000.0, y_offset_m=30.0),
    ]

    payload_3d = page._all_wells_three_payload(
        successes,
        focus_well_names=("PAD1-A", "PAD1-B"),
    )
    figure_plan = page._all_wells_plan_figure(
        successes,
        focus_well_names=("PAD1-A", "PAD1-B"),
    )

    assert float(payload_3d["bounds"]["max"][0]) < 4000.0
    assert float(figure_plan.layout.xaxis.range[1]) < 4000.0
    assert "PAD2-A" in {str(item["label"]) for item in payload_3d["legend"]}
    assert any(str(trace.name) == "PAD2-B" for trace in figure_plan.data)


def test_anticollision_figures_include_reference_trajectory_wells_without_target_markers() -> (
    None
):
    page = wt_import_module
    reference_wells = _reference_wells()
    analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=reference_wells,
    )

    payload_3d = page._all_wells_anticollision_three_payload(
        analysis,
        render_mode=page.WT_3D_RENDER_DETAIL,
    )
    figure_plan = page._all_wells_anticollision_plan_figure(analysis)

    hover_names = {
        str(hover.get("name"))
        for item in payload_3d["points"]
        for hover in list(item.get("hover") or [])
    }
    assert "FACT-1" in hover_names
    assert "APP-1" in hover_names
    mesh_names = {str(item.get("name")) for item in payload_3d["meshes"]}
    assert "FACT-1 (Фактическая) cone" in mesh_names
    assert "APP-1 (Проектная утвержденная) cone" in mesh_names
    assert not any(
        "FACT-1 (Фактическая): цели" == str(hover.get("name"))
        for item in payload_3d["points"]
        for hover in list(item.get("hover") or [])
    )
    assert not any(
        "APP-1 (Проектная утвержденная): цели" == str(trace.name)
        for trace in figure_plan.data
    )
    assert "FACT-1 (Фактическая) cone" in {str(trace.name) for trace in figure_plan.data}
    assert "APP-1 (Проектная утвержденная) cone" in {
        str(trace.name) for trace in figure_plan.data
    }
    assert any(str(item["text"]) == "FACT-1" for item in payload_3d["labels"])
    assert any(
        str(trace.name) == "Проектная утвержденная: подписи"
        for trace in figure_plan.data
    )


def test_anticollision_figures_ignore_far_reference_wells_for_axis_zoom() -> None:
    page = wt_import_module
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

    base_3d = page._all_wells_anticollision_three_payload(base_analysis)
    base_plan = page._all_wells_anticollision_plan_figure(base_analysis)
    far_ref_3d = page._all_wells_anticollision_three_payload(far_ref_analysis)
    far_ref_plan = page._all_wells_anticollision_plan_figure(far_ref_analysis)

    assert base_3d["bounds"] == far_ref_3d["bounds"]
    assert tuple(base_plan.layout.xaxis.range) == tuple(far_ref_plan.layout.xaxis.range)
    assert tuple(base_plan.layout.yaxis.range) == tuple(far_ref_plan.layout.yaxis.range)


def test_all_wells_three_payload_aggregates_reference_wells_in_fast_mode() -> None:
    page = wt_import_module
    payload = page._all_wells_three_payload(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        reference_wells=_reference_wells(),
        render_mode=page.WT_3D_RENDER_FAST,
    )

    line_colors = {str(item["color"]) for item in payload["lines"]}
    hover_names = {
        str(hover.get("name"))
        for item in payload["points"]
        for hover in list(item.get("hover") or [])
    }
    assert "#6B7280" in line_colors
    assert "#C62828" in line_colors
    assert "FACT-1" in hover_names
    assert "APP-1" in hover_names
    assert "FACT-1 (Фактическая)" not in hover_names
    assert "APP-1 (Проектная утвержденная)" not in hover_names


def test_anticollision_three_payload_aggregates_non_conflicting_reference_wells_in_fast_mode() -> (
    None
):
    page = wt_import_module
    far_reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-FAR",
                    "Type": "actual",
                    "X": 9000.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-FAR",
                    "Type": "actual",
                    "X": 9000.0,
                    "Y": 0.0,
                    "Z": 2000.0,
                    "MD": 2000.0,
                },
                {
                    "Wellname": "APP-FAR",
                    "Type": "approved",
                    "X": 12000.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "APP-FAR",
                    "Type": "approved",
                    "X": 12000.0,
                    "Y": 0.0,
                    "Z": 2000.0,
                    "MD": 2000.0,
                },
            ]
        )
    )
    analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=far_reference_wells,
    )

    payload = page._all_wells_anticollision_three_payload(
        analysis,
        render_mode=page.WT_3D_RENDER_FAST,
    )

    line_colors = {str(item["color"]) for item in payload["lines"]}
    hover_names = {
        str(hover.get("name"))
        for item in payload["points"]
        for hover in list(item.get("hover") or [])
    }
    assert "#6B7280" in line_colors
    assert "#C62828" in line_colors
    assert "FACT-FAR" in hover_names
    assert "APP-FAR" in hover_names
    assert "FACT-FAR (Фактическая)" not in hover_names
    assert "APP-FAR (Проектная утвержденная)" not in hover_names


def test_clusters_touching_focus_pad_expand_focus_to_neighbor_cluster_wells() -> None:
    page = wt_import_module
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="PAD1-A", y_offset_m=0.0),
            _successful_plan(name="PAD2-A", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=page._build_anticollision_well_contexts(
            [
                _successful_plan(name="PAD1-A", y_offset_m=0.0),
                _successful_plan(name="PAD2-A", y_offset_m=5.0),
            ]
        ),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    visible_clusters = page._clusters_touching_focus_pad(
        clusters=clusters,
        focus_pad_well_names=("PAD1-A",),
    )
    focus_names = page._anticollision_focus_well_names(
        clusters=visible_clusters,
        focus_pad_well_names=("PAD1-A",),
    )

    assert visible_clusters
    assert set(focus_names) == {"PAD1-A", "PAD2-A"}
    report_rows = page._report_rows_from_recommendations(
        recommendations,
        analysis,
    )
    assert report_rows
    assert "Рекомендация по устранению" in report_rows[0]


def test_all_wells_and_anticollision_figures_show_well_names_at_t3() -> None:
    page = wt_import_module
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    records = _records()[:2]
    color_map = page._well_color_map(records)

    overview_3d = page._all_wells_three_payload(successes, name_to_color=color_map)
    overview_plan = page._all_wells_plan_figure(successes, name_to_color=color_map)
    analysis = page._build_anti_collision_analysis(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        name_to_color=color_map,
    )
    anticollision_3d = page._all_wells_anticollision_three_payload(analysis)
    anticollision_plan = page._all_wells_anticollision_plan_figure(analysis)

    for payload in (overview_3d, anticollision_3d):
        label_texts = {str(item["text"]) for item in payload["labels"]}
        assert "WELL-A" in label_texts
        assert "WELL-B" in label_texts
    for figure in (overview_plan, anticollision_plan):
        trace_names = {str(trace.name) for trace in figure.data}
        assert "WELL-A: t1 label" in trace_names
        assert "WELL-B: t1 label" in trace_names


def test_all_wells_three_payload_preserves_overview_labels_and_legend() -> (
    None
):
    page = wt_import_module
    success = _successful_plan(name="WELL-A", y_offset_m=0.0)
    payload = page._all_wells_three_payload(
        [success],
        reference_wells=_reference_wells(),
        render_mode=page.WT_3D_RENDER_DETAIL,
    )

    labels = {str(item["text"]) for item in payload["labels"]}
    legend_labels = {str(item["label"]) for item in payload["legend"]}
    assert payload["camera"] == DEFAULT_THREE_CAMERA
    assert "WELL-A" in labels
    assert "FACT-1" in labels
    assert "APP-1" in labels
    assert "WELL-A" in legend_labels
    assert "Фактические скважины" in legend_labels
    assert "Проектные утвержденные скважины" in legend_labels
    assert "Reference pads: кусты" not in legend_labels
    assert "FACT-1 (Фактическая)" not in legend_labels
    well_label = next(
        item for item in payload["labels"] if str(item["text"]) == "WELL-A"
    )
    assert str(well_label["role"]) == "well_label"
    assert well_label["position"] == [
        float(success.t3.x),
        float(success.t3.y),
        float(success.t3.z),
    ]


def test_three_legend_tree_stays_calculated_only_when_reference_pad_labels_exist() -> (
    None
):
    page = wt_import_module
    overrides = page._trajectory_three_payload_overrides(
        records=_multi_pad_records(),
        successes=[
            _successful_plan(name="PAD1-A", y_offset_m=0.0),
            _successful_plan(name="PAD1-B", y_offset_m=5.0),
            _successful_plan(name="PAD2-A", y_offset_m=2500.0),
            _successful_plan(name="PAD2-B", y_offset_m=2505.0),
        ],
        target_only_wells=[],
        name_to_color={
            "PAD1-A": "#00AA55",
            "PAD1-B": "#0066DD",
            "PAD2-A": "#AA00FF",
            "PAD2-B": "#FF8800",
        },
    )

    legend_tree = list(overrides["legend_tree"])

    assert [str(item["label"]) for item in legend_tree] == [
        "Куст PAD-01",
        "Куст PAD-02",
    ]
    flat_child_labels = [
        str(child["label"])
        for group in legend_tree
        for child in list(group["children"])
    ]
    assert "Куст 80" not in flat_child_labels
    assert "Куст 90" not in flat_child_labels


def test_three_payload_preserves_reference_legend_without_flat_geometry_label() -> None:
    page = wt_import_module
    payload = page._all_wells_three_payload(
        [],
        reference_wells=_reference_wells(),
        render_mode=page.WT_3D_RENDER_FAST,
    )

    assert {
        (str(item["label"]), str(item["color"]), float(item["opacity"]))
        for item in payload["legend"]
    } >= {
        ("Фактические скважины", "#6B7280", 1.0),
        ("Проектные утвержденные скважины", "#C62828", 1.0),
    }


def test_anticollision_three_payload_preserves_meshes() -> None:
    page = wt_import_module
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    payload = page._all_wells_anticollision_three_payload(analysis)

    assert payload["meshes"]
    assert any(str(item["label"]) == "WELL-A" for item in payload["legend"])


def test_anticollision_three_payload_overrides_include_edit_wells() -> None:
    page = wt_import_module
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    analysis = page._build_anti_collision_analysis(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    overrides = page._anticollision_three_payload_overrides(
        records=_records(),
        analysis=analysis,
        successes=successes,
    )

    edit_wells = list(overrides["edit_wells"])
    assert [str(item["name"]) for item in edit_wells] == ["WELL-A", "WELL-B"]
    assert edit_wells[0]["t1"] == [1000.0, 0.0, 0.0]
    assert edit_wells[0]["t3"] == [2000.0, 0.0, 0.0]


def test_all_wells_three_payload_preserves_hover_metadata_for_tooltip() -> (
    None
):
    page = wt_import_module
    payload = page._all_wells_three_payload(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    )

    hover_only_points = [
        item for item in payload["points"] if bool(item.get("hover_only"))
    ]
    assert hover_only_points
    first_hover = hover_only_points[0]["hover"][0]
    assert first_hover["name"] == "WELL-A"
    assert "md" in first_hover
    assert "dls" in first_hover
    assert "inc" in first_hover
    assert "segment" in first_hover


def test_optimize_three_payload_merges_same_style_objects() -> None:
    page = wt_import_module
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_THREE_CAMERA,
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
                "role": "overlap",
            },
            {
                "vertices": [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#C62828",
                "opacity": 0.4,
                "role": "overlap",
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
    assert optimized["points"][0]["hover_only"] is False
    assert len(optimized["meshes"]) == 1
    assert len(optimized["meshes"][0]["vertices"]) == 6
    assert len(optimized["meshes"][0]["faces"]) == 2
    assert optimized["meshes"][0]["role"] == "overlap"
    assert optimized["camera"] == DEFAULT_THREE_CAMERA
    assert optimized["labels"] == payload["labels"]
    assert optimized["legend"] == payload["legend"]


def test_optimize_three_payload_keeps_mesh_roles_separate() -> None:
    page = wt_import_module
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_THREE_CAMERA,
        "lines": [],
        "points": [],
        "meshes": [
            {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#15D562",
                "opacity": 0.12,
                "role": "cone",
            },
            {
                "vertices": [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#15D562",
                "opacity": 0.12,
                "role": "overlap",
            },
        ],
        "labels": [],
        "legend": [],
    }

    optimized = page._optimize_three_payload(payload)

    assert len(optimized["meshes"]) == 2
    assert {str(item["role"]) for item in optimized["meshes"]} == {"cone", "overlap"}


def test_three_payload_marks_conflict_segments_as_anticollision_layer() -> None:
    page = wt_import_module
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    payload = page._all_wells_anticollision_three_payload(analysis)

    assert any(str(item.get("role")) == "conflict_segment" for item in payload["lines"])
    assert any(str(item.get("role")) == "conflict_hover" for item in payload["points"])


def test_resolve_3d_render_mode_forces_fast_for_large_reference_scene() -> None:
    page = wt_import_module
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": f"FACT-{index:03d}",
                    "Type": "actual",
                    "X": float(index * 10.0),
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": f"FACT-{index:03d}",
                    "Type": "actual",
                    "X": float(index * 10.0 + 100.0),
                    "Y": 0.0,
                    "Z": 1000.0,
                    "MD": 1000.0,
                },
            ]
        )[0]
        for index in range(page.WT_3D_FAST_REFERENCE_WELL_THRESHOLD + 2)
    )

    resolved = page._resolve_3d_render_mode(
        requested_mode=page.WT_3D_RENDER_DETAIL,
        calculated_well_count=1,
        reference_wells=reference_wells,
    )

    assert resolved == page.WT_3D_RENDER_FAST


def test_three_payload_decimates_reference_hover_points() -> None:
    page = wt_import_module
    stations = pd.DataFrame(
        {
            "MD_m": np.arange(0.0, 3000.0, 10.0, dtype=float),
            "INC_deg": np.linspace(0.0, 90.0, 300, dtype=float),
            "AZI_deg": np.full(300, 90.0, dtype=float),
            "X_m": np.arange(300, dtype=float) * 10.0,
            "Y_m": np.zeros(300, dtype=float),
            "Z_m": np.linspace(0.0, 2500.0, 300, dtype=float),
            "DLS_deg_per_30m": np.full(300, 2.0, dtype=float),
            "segment": np.array(["HOLD"] * 300, dtype=object),
        }
    )
    reference_well = SimpleNamespace(
        name="FACT-001",
        kind=page.REFERENCE_WELL_ACTUAL,
        stations=stations,
        surface=Point3D(0.0, 0.0, 0.0),
    )

    payload = page._all_wells_three_payload(
        [],
        reference_wells=(reference_well,),
        render_mode=page.WT_3D_RENDER_DETAIL,
    )
    hover_only_points = [
        item for item in payload["points"] if bool(item.get("hover_only"))
    ]

    assert len(hover_only_points) == 1
    assert str(hover_only_points[0]["role"]) == "reference_hover"
    assert (
        len(hover_only_points[0]["points"])
        == page.WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
    )
    assert (
        len(hover_only_points[0]["hover"])
        == page.WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
    )
    assert {
        str(item.get("name")) for item in hover_only_points[0]["hover"]
    } == {"FACT-001"}


def test_fast_three_payload_keeps_reference_well_hover_names() -> None:
    page = wt_import_module
    reference_wells = tuple(
        parse_reference_trajectory_table(
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
                    "X": 100.0,
                    "Y": 0.0,
                    "Z": 1000.0,
                    "MD": 1000.0,
                },
                {
                    "Wellname": "APR-007",
                    "Type": "approved",
                    "X": 0.0,
                    "Y": 50.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "APR-007",
                    "Type": "approved",
                    "X": 100.0,
                    "Y": 50.0,
                    "Z": 1000.0,
                    "MD": 1000.0,
                },
            ]
        )
    )

    payload = page._all_wells_three_payload(
        [],
        reference_wells=reference_wells,
        render_mode=page.WT_3D_RENDER_FAST,
    )
    reference_hover = [
        item for item in payload["points"] if str(item.get("role")) == "reference_hover"
    ]
    hover_names = {
        str(hover.get("name"))
        for item in reference_hover
        for hover in item.get("hover", [])
    }

    assert len(reference_hover) <= 2
    assert {"FACT-001", "APR-007"}.issubset(hover_names)


def test_fast_anticollision_3d_keeps_near_reference_cone_by_xy_gap() -> None:
    page = wt_import_module
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-NEAR",
                    "Type": "actual",
                    "X": 2200.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-NEAR",
                    "Type": "actual",
                    "X": 2200.0,
                    "Y": 0.0,
                    "Z": 2000.0,
                    "MD": 2000.0,
                },
                {
                    "Wellname": "FACT-FAR",
                    "Type": "actual",
                    "X": 8000.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-FAR",
                    "Type": "actual",
                    "X": 8000.0,
                    "Y": 0.0,
                    "Z": 2000.0,
                    "MD": 2000.0,
                },
            ]
        )
    )
    analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=reference_wells,
    )

    payload = page._all_wells_anticollision_three_payload(
        analysis,
        render_mode=page.WT_3D_RENDER_FAST,
    )
    near_only_analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=tuple(
            well for well in reference_wells if str(well.name) == "FACT-NEAR"
        ),
    )
    near_only_payload = page._all_wells_anticollision_three_payload(
        near_only_analysis,
        render_mode=page.WT_3D_RENDER_FAST,
    )
    reference_cone = next(
        item for item in payload["meshes"] if str(item.get("color")) == "#6B7280"
    )
    hover_names = {
        str(hover.get("name"))
        for item in payload["points"]
        if str(item.get("role")) == "reference_hover"
        for hover in list(item.get("hover") or [])
    }
    near_only_reference_cone = next(
        item
        for item in near_only_payload["meshes"]
        if str(item.get("color")) == "#6B7280"
    )

    assert len(reference_cone["vertices"]) == len(near_only_reference_cone["vertices"])
    assert "FACT-NEAR" in hover_names


def test_all_wells_three_payload_sets_default_camera() -> None:
    page = wt_import_module
    payload = page._all_wells_three_payload(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    )

    assert payload["camera"] == DEFAULT_THREE_CAMERA


def test_batch_summary_display_df_reorders_and_shortens_summary_columns() -> None:
    page = wt_import_module
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
                "Отход t1, м": "100.00",
                "KOP MD, м": "550.00",
                "Длина ГС, м": "1200.00",
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
        "ГС, м",
        "INC в t1, deg",
    ]
    assert display_df.columns[-1] == "Модель траектории"
    assert "Рестарты" in display_df.columns
    assert "Классификация целей" not in display_df.columns
    assert "Рестарты решателя" not in display_df.columns


def test_anticollision_figures_render_overlap_corridor_and_red_conflict_segments() -> (
    None
):
    page = wt_import_module
    successes = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    analysis = page._build_anti_collision_analysis(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    payload_3d = page._all_wells_anticollision_three_payload(analysis)
    figure_plan = page._all_wells_anticollision_plan_figure(analysis)

    overlap_volumes = [
        item
        for item in payload_3d["meshes"]
        if str(item.get("role")) == "overlap_volume"
    ]
    conflict_3d_lines = [
        item
        for item in payload_3d["lines"]
        if str(item.get("role")) == "conflict_segment"
    ]
    overlap_plan_traces = [
        trace for trace in figure_plan.data if str(trace.name) == "Зоны пересечений"
    ]
    conflict_plan_traces = [
        trace
        for trace in figure_plan.data
        if str(trace.name) == "Конфликтные участки ствола"
    ]
    assert overlap_volumes
    assert all(len(item["rings"]) >= 2 for item in overlap_volumes)
    assert conflict_3d_lines
    assert overlap_plan_traces
    assert conflict_plan_traces
    assert all(str(item["color"]) == "#C62828" for item in conflict_3d_lines)
    assert all("198, 40, 40" in str(trace.line.color) for trace in conflict_plan_traces)
    assert all(
        str(getattr(trace, "hoverinfo", "")) == "skip" for trace in overlap_plan_traces
    )


def test_anticollision_3d_trajectory_hover_is_reserved_for_wells_targets_and_conflict_segments() -> (
    None
):
    page = wt_import_module
    analysis = page._build_anti_collision_analysis(
        [
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=5.0),
        ],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    payload_3d = page._all_wells_anticollision_three_payload(analysis)

    cone_meshes = [
        item for item in payload_3d["meshes"] if str(item.get("role")) == "cone"
    ]
    overlap_meshes = [
        item
        for item in payload_3d["meshes"]
        if str(item.get("role")) == "overlap_volume"
    ]
    well_lines = [
        item for item in payload_3d["lines"] if str(item.get("role")) == "line"
    ]
    target_markers = [
        item for item in payload_3d["points"] if str(item.get("role")) == "marker"
    ]
    conflict_segments = [
        item
        for item in payload_3d["lines"]
        if str(item.get("role")) == "conflict_segment"
    ]

    assert cone_meshes
    assert overlap_meshes
    assert well_lines
    trajectory_hovers = [
        hover
        for item in payload_3d["points"]
        if str(item.get("role")) == "trajectory_hover"
        for hover in list(item.get("hover") or [])
    ]
    assert trajectory_hovers
    assert {"md", "dls", "inc", "segment"}.issubset(trajectory_hovers[0].keys())
    assert target_markers
    assert target_markers[0]["hover"][0]["point"] == "S"
    assert conflict_segments
    conflict_hovers = [
        hover
        for item in payload_3d["points"]
        if str(item.get("role")) == "conflict_hover"
        for hover in list(item.get("hover") or [])
    ]
    assert conflict_hovers
    assert {"md", "dls", "inc"}.issubset(conflict_hovers[0].keys())
