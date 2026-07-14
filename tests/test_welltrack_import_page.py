from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pywp import ptc_core as wt_import_module
from pywp import ptc_anticollision_params
from pywp import ptc_edit_targets
from pywp.actual_fund_analysis import ActualFundKopDepthFunction
from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionCorridor,
    AntiCollisionIncrementalStats,
    AntiCollisionProgress,
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
from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackRecord,
    parse_welltrack_text,
)
from pywp.mcm import compute_positions_min_curv
from pywp.models import Point3D, TrajectoryConfig
from pywp.pilot_wells import sync_pilot_surfaces_to_parents
from pywp.reference_trajectories import ImportedTrajectoryWell, parse_reference_trajectory_table
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
from pywp.well_pad import apply_pad_layout
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner

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


def _degenerate_surface_pad_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="9201",
            points=(
                WelltrackPoint(x=1000.0, y=800.0, z=0.0, md=0.0),
                WelltrackPoint(x=1000.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1900.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="9202",
            points=(
                WelltrackPoint(x=1600.0, y=900.0, z=0.0, md=0.0),
                WelltrackPoint(x=1600.0, y=900.0, z=2350.0, md=2350.0),
                WelltrackPoint(x=2500.0, y=2100.0, z=2450.0, md=3450.0),
            ),
        ),
        WelltrackRecord(
            name="9203",
            points=(
                WelltrackPoint(x=2200.0, y=1000.0, z=0.0, md=0.0),
                WelltrackPoint(x=2200.0, y=1000.0, z=2300.0, md=2300.0),
                WelltrackPoint(x=3100.0, y=2200.0, z=2400.0, md=3400.0),
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


def _prepositioned_depth_order_pad_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="P1-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2100.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2200.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="P1-B",
            points=(
                WelltrackPoint(x=25.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=625.0, y=780.0, z=2600.0, md=2350.0),
                WelltrackPoint(x=1525.0, y=1980.0, z=2700.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="P1-C",
            points=(
                WelltrackPoint(x=50.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=760.0, z=2300.0, md=2300.0),
                WelltrackPoint(x=1550.0, y=1960.0, z=2400.0, md=3350.0),
            ),
        ),
    ]


def _template_pad_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="8001",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="7412",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="9305",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=700.0, y=760.0, z=2200.0, md=2300.0),
                WelltrackPoint(x=1600.0, y=1960.0, z=2350.0, md=3350.0),
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


class _AppTestSessionStateAdapter(MutableMapping[str, object]):
    def __init__(self, state: object) -> None:
        self._state = state

    def __getitem__(self, key: str) -> object:
        return self._state[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._state[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state[key]

    def __iter__(self):
        return iter(dict(self._state.filtered_state))

    def __len__(self) -> int:
        return len(dict(self._state.filtered_state))


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
    expander_calls: list[tuple[str, object]] = []

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
        expander_calls.append((str(label), kwargs.get("expanded")))
        return _DummyExpander()

    def _fake_dataframe(frame, **kwargs):
        captured["frame"] = frame.copy()

    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
        ),
    )
    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyExpander())

    page._render_records_overview([_records()[0]])

    assert ("Статус загрузки целей", False) in expander_calls
    assert ("Изменить длину ГС", False) in expander_calls
    assert captured["metrics"] == [
        ("Скважин", "1"),
        ("Пилотов", "0"),
        ("Боковых стволов", "0"),
        ("Многопластовых скважин", "0"),
        ("Ошибки импорта", "0"),
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
    expander_calls: list[tuple[str, object]] = []
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
        expander_calls.append((str(label), kwargs.get("expanded")))
        return _DummyExpander()

    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
        ),
    )
    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyExpander())

    page._render_records_overview([_records()[0], incomplete])

    assert ("Статус загрузки целей", True) in expander_calls
    assert ("Изменить длину ГС", False) in expander_calls


def test_records_overview_appends_dev_import_failures_to_status_table(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    captured: dict[str, object] = {}
    expander_calls: list[tuple[str, object]] = []
    try:
        page.st.session_state[page.ptc_target_import.TARGET_IMPORT_FAILURES_STATE_KEY] = (
            page.ptc_target_import.TargetImportFailure(
                well_name="BROKEN-DEV",
                problem="Не удалось надежно определить точку t1.",
                source_label="BROKEN-DEV.dev",
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
            expander_calls.append((str(label), kwargs.get("expanded")))
            return _DummyExpander()

        def _fake_dataframe(frame, **kwargs):
            captured["frame"] = frame.copy()

        monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            page.st,
            "columns",
            lambda *args, **kwargs: (
                _DummyColumn(),
                _DummyColumn(),
                _DummyColumn(),
                _DummyColumn(),
                _DummyColumn(),
            ),
        )
        monkeypatch.setattr(page.st, "expander", _fake_expander)
        monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)
        monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
        monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
        monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: None)
        monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
        monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyExpander())

        page._render_records_overview([_records()[0]])

        assert ("Статус загрузки целей", True) in expander_calls
        assert ("Изменить длину ГС", False) in expander_calls
        assert list(captured["frame"]["Скважина"]) == ["WELL-A", "BROKEN-DEV"]
        assert list(captured["frame"]["Статус"]) == ["✅", "❌"]
        assert str(captured["frame"].iloc[-1]["Проблема"]) == (
            "Не удалось надежно определить точку t1."
        )
    finally:
        page.st.session_state.clear()
        page._init_state()


def test_records_overview_marks_three_point_dev_as_plain_target_note() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    record = WelltrackRecord(
        name="THREE-POINT-DEV",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
        ),
    )
    page.st.session_state[page.ptc_target_import.IMPORTED_DEV_TARGET_WELLS_STATE_KEY] = (
        ImportedTrajectoryWell(
            name="THREE-POINT-DEV",
            kind="approved",
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 2400.0, 3500.0],
                    "X_m": [0.0, 600.0, 1500.0],
                    "Y_m": [0.0, 800.0, 2000.0],
                    "Z_m": [0.0, 2400.0, 2500.0],
                }
            ),
            surface=Point3D(x=0.0, y=0.0, z=0.0),
            azimuth_deg=45.0,
        ),
    )

    overview_df = page._records_overview_dataframe([record])

    assert list(overview_df["Скважина"]) == ["THREE-POINT-DEV"]
    assert "3 точки" in str(overview_df.iloc[0]["Примечание"])
    assert "общие параметры расчёта" in str(overview_df.iloc[0]["Примечание"])


def test_records_overview_metrics_count_wells_and_pilots_separately(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
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
        lambda *args, **kwargs: (
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
        ),
    )
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyExpander())

    page._render_records_overview([parent, pilot])

    assert captured["metrics"] == [
        ("Скважин", "1"),
        ("Пилотов", "1"),
        ("Боковых стволов", "0"),
        ("Многопластовых скважин", "0"),
        ("Ошибки импорта", "0"),
    ]


def test_records_overview_metrics_keep_surface_alt_branch_as_regular_well(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    captured: dict[str, object] = {"metrics": []}
    branch = WelltrackRecord(
        name="well_04_2",
        points=(
            WelltrackPoint(x=10.0, y=20.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=1.0),
            WelltrackPoint(x=300.0, y=0.0, z=1200.0, md=2.0),
        ),
    )
    pilot = WelltrackRecord(
        name="well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=50.0, y=0.0, z=800.0, md=1.0),
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
        lambda *args, **kwargs: (
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
        ),
    )
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyExpander())

    page._render_records_overview([branch, pilot])

    assert captured["metrics"] == [
        ("Скважин", "1"),
        ("Пилотов", "1"),
        ("Боковых стволов", "0"),
        ("Многопластовых скважин", "0"),
        ("Ошибки импорта", "0"),
    ]


def test_records_overview_metrics_count_multi_horizontal_wells(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    captured: dict[str, object] = {"metrics": []}
    regular = _records()[0]
    multi_horizontal = WelltrackRecord(
        name="well_multi",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
            WelltrackPoint(x=300.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=400.0, y=0.0, z=1100.0, md=3.0),
            WelltrackPoint(x=600.0, y=0.0, z=1100.0, md=4.0),
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
        lambda *args, **kwargs: (
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
        ),
    )
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyExpander())

    page._render_records_overview([regular, multi_horizontal])

    assert captured["metrics"] == [
        ("Скважин", "2"),
        ("Пилотов", "0"),
        ("Боковых стволов", "0"),
        ("Многопластовых скважин", "1"),
        ("Ошибки импорта", "0"),
    ]


def test_records_overview_metrics_count_multi_horizontal_zbs(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    captured: dict[str, object] = {"metrics": []}
    zbs_multi = WelltrackRecord(
        name="9010_ZBS",
        points=(
            WelltrackPoint(x=650.0, y=0.0, z=1500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=1500.0, md=2.0),
            WelltrackPoint(x=1800.0, y=0.0, z=1520.0, md=3.0),
            WelltrackPoint(x=2300.0, y=0.0, z=1520.0, md=4.0),
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
        lambda *args, **kwargs: (
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
            _DummyColumn(),
        ),
    )
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyExpander())

    page._render_records_overview([zbs_multi])

    assert captured["metrics"] == [
        ("Скважин", "0"),
        ("Пилотов", "0"),
        ("Боковых стволов", "1"),
        ("Многопластовых скважин", "1"),
        ("Ошибки импорта", "0"),
    ]


def test_records_overview_preprocess_uses_fixed_default_and_step(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    captured: dict[str, object] = {}

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def metric(self, *args, **kwargs):
            return None

        def number_input(self, label, **kwargs):
            captured["number_input"] = {
                "label": str(label),
                "step": float(kwargs.get("step", 0.0)),
                "min_value": float(kwargs.get("min_value", 0.0)),
                "key": str(kwargs.get("key", "")),
            }
            return None

        def button(self, *args, **kwargs):
            return False

        def caption(self, *args, **kwargs):
            return None

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "multiselect",
        lambda _label, options, key, **kwargs: page.st.session_state.setdefault(
            key, list(options)
        ),
    )
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)

    page._render_records_overview([_records()[0]])

    assert page.st.session_state["wt_preprocess_horizontal_length_m"] == 1500.0
    assert page.st.session_state["wt_preprocess_selected_names"] == ["WELL-A"]
    assert captured["number_input"] == {
        "label": "Новая длина ГС, м",
        "step": 100.0,
        "min_value": 1.0,
        "key": "wt_preprocess_horizontal_length_m",
    }


def test_records_overview_preprocess_waits_for_apply_button(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    original_records = [_records()[0]]
    page.st.session_state["wt_records"] = list(original_records)
    calls: list[float] = []

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def metric(self, *args, **kwargs):
            return None

        def number_input(self, _label, **kwargs):
            key = str(kwargs.get("key", ""))
            page.st.session_state[key] = 1700.0
            return 1700.0

        def button(self, *args, **kwargs):
            return False

        def caption(self, *args, **kwargs):
            return None

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "multiselect",
        lambda _label, options, key, **kwargs: page.st.session_state.setdefault(
            key, list(options)
        ),
    )
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        page,
        "_bulk_horizontal_length_changes",
        lambda *args, **kwargs: calls.append(float(kwargs["target_length_m"])) or ([], []),
    )

    page._render_records_overview(original_records)

    assert page.st.session_state["wt_preprocess_horizontal_length_m"] == 1700.0
    assert calls == []
    assert page.st.session_state["wt_records"] == list(original_records)


def test_records_overview_preprocess_applies_only_selected_wells(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    original_records = _records()[:2]
    page.st.session_state["wt_records"] = list(original_records)
    page.st.session_state["wt_preprocess_selected_names"] = ["WELL-B"]
    captured: dict[str, object] = {}

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def metric(self, *args, **kwargs):
            return None

        def number_input(self, _label, **kwargs):
            key = str(kwargs.get("key", ""))
            page.st.session_state[key] = 1700.0
            return 1700.0

        def button(self, label, **kwargs):
            return str(label) == "Применить"

        def caption(self, *args, **kwargs):
            return None

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        page.st,
        "multiselect",
        lambda _label, options, key, **kwargs: page.st.session_state.setdefault(
            key, ["WELL-B"]
        ),
    )
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda: None)

    def _fake_bulk_horizontal_length_changes(records, *, target_length_m):
        captured["selected_names"] = [str(record.name) for record in records]
        captured["target_length_m"] = float(target_length_m)
        return ([{"name": "WELL-B", "points": []}], [])

    monkeypatch.setattr(
        page,
        "_bulk_horizontal_length_changes",
        _fake_bulk_horizontal_length_changes,
    )

    def _fake_apply_edit_targets_changes(changes, *, source="3d"):
        captured["source"] = source
        captured["changes"] = changes
        return ["WELL-B"]

    monkeypatch.setattr(page, "_apply_edit_targets_changes", _fake_apply_edit_targets_changes)

    page._render_records_overview(original_records)

    assert captured["selected_names"] == ["WELL-B"]
    assert captured["target_length_m"] == 1700.0
    assert captured["source"] == "bulk_horizontal_length_preprocess"


def test_preprocess_excluded_records_message_lists_pilot_and_multilevel_wells() -> None:
    page = wt_import_module
    parent = WelltrackRecord(
        name="well_04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=2.0),
            WelltrackPoint(x=600.0, y=0.0, z=1200.0, md=3.0),
        ),
    )
    pilot = WelltrackRecord(
        name="well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=2.0),
        ),
    )
    multi_horizontal = WelltrackRecord(
        name="well_multi",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
            WelltrackPoint(x=300.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=400.0, y=0.0, z=1100.0, md=3.0),
            WelltrackPoint(x=600.0, y=0.0, z=1100.0, md=4.0),
        ),
    )

    message = page._preprocess_excluded_records_message(
        [parent, pilot, multi_horizontal]
    )

    assert message == (
        "Изменение длины ГС не применялось к: "
        "well_04 (есть пилот), well_multi (многопластовая)."
    )


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


def test_welltrack_page_renders_and_fixes_zbs_t1_t3_order_from_parent() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    zbs_record = WelltrackRecord(
        name="9010_ZBS",
        points=(
            WelltrackPoint(x=1600.0, y=0.0, z=2500.0, md=1.0),
            WelltrackPoint(x=1100.0, y=0.0, z=2501.0, md=2.0),
        ),
    )
    actual_parent = parse_reference_trajectory_table(
        [
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 1000.0,
                "Y": 0.0,
                "Z": 2500.0,
                "MD": 2500.0,
            },
        ]
    )[0]
    at.session_state["wt_records"] = [zbs_record]
    at.session_state["wt_records_original"] = [zbs_record]
    at.session_state["wt_reference_actual_wells"] = [actual_parent]
    at.session_state["wt_reference_wells"] = [actual_parent]

    at.run(timeout=120)

    assert any("перепутаны местами" in str(widget.value) for widget in at.warning)
    assert any(
        "Подцепили фактическую скважину для бокового ствола" in str(widget.value)
        for widget in at.info
    )
    assert any(
        "После подцепления фактической скважины проверьте порядок целей" in str(widget.value)
        for widget in at.warning
    )
    assert any(
        "Перейти к блоку проверки t1/t3" in str(widget.value)
        for widget in at.warning
    )
    assert any(
        "#wt-t1-t3-order-panel" in str(widget.value)
        for widget in at.warning
    )
    assert any("`9010_ZBS`" in str(widget.value) for widget in at.markdown)
    assert any("родитель→" in str(widget.value) for widget in at.markdown)

    _click_button(at, "Исправить порядок для выбранных скважин")
    at.run(timeout=120)

    fixed = at.session_state["wt_records"][0]
    assert [point.md for point in fixed.points] == pytest.approx([1.0, 2.0])
    assert [point.x for point in fixed.points] == pytest.approx([1100.0, 1600.0])


def test_t1_t3_order_anchor_skips_reference_scan_without_zbs(monkeypatch) -> None:
    page = wt_import_module
    monkeypatch.setattr(
        page,
        "_reference_wells_from_state",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected reference scan")),
    )

    assert page._t1_t3_order_anchor_by_well_name(_bad_order_records()) == {}


def test_t1_t3_order_anchor_uses_nearest_parent_point_for_zbs(monkeypatch) -> None:
    page = wt_import_module
    zbs_record = WelltrackRecord(
        name="9010_ZBS",
        points=(
            WelltrackPoint(x=900.0, y=0.0, z=2500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=2500.0, md=2.0),
        ),
    )
    actual_parent = parse_reference_trajectory_table(
        [
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 1000.0,
                "Y": 0.0,
                "Z": 2500.0,
                "MD": 2500.0,
            },
        ]
    )[0]
    monkeypatch.setattr(page, "_reference_wells_from_state", lambda: (actual_parent,))

    anchors = page._t1_t3_order_anchor_by_well_name([zbs_record])

    assert anchors["9010_ZBS"].x == pytest.approx(1000.0)
    assert anchors["9010_ZBS"].z == pytest.approx(2500.0)


def test_t1_t3_order_anchor_prefers_actual_parent_over_approved_duplicate(
    monkeypatch,
) -> None:
    page = wt_import_module
    zbs_record = WelltrackRecord(
        name="9010_ZBS",
        points=(
            WelltrackPoint(x=900.0, y=0.0, z=2500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=2500.0, md=2.0),
        ),
    )
    actual_parent, approved_parent = parse_reference_trajectory_table(
        [
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 1000.0,
                "Y": 0.0,
                "Z": 2500.0,
                "MD": 2500.0,
            },
            {
                "Wellname": "9010",
                "Type": "approved",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9010",
                "Type": "approved",
                "X": 5000.0,
                "Y": 0.0,
                "Z": 2500.0,
                "MD": 2500.0,
            },
        ]
    )
    monkeypatch.setattr(
        page,
        "_reference_wells_from_state",
        lambda: (actual_parent, approved_parent),
    )

    anchors = page._t1_t3_order_anchor_by_well_name([zbs_record])

    assert anchors["9010_ZBS"].x == pytest.approx(1000.0)


def test_t1_t3_order_anchor_supports_multi_horizontal_zbs(monkeypatch) -> None:
    page = wt_import_module
    zbs_record = WelltrackRecord(
        name="9010_ZBS",
        points=(
            WelltrackPoint(x=900.0, y=0.0, z=2500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=2500.0, md=2.0),
            WelltrackPoint(x=1300.0, y=50.0, z=2520.0, md=3.0),
            WelltrackPoint(x=1700.0, y=50.0, z=2520.0, md=4.0),
        ),
    )
    actual_parent = parse_reference_trajectory_table(
        [
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9010",
                "Type": "actual",
                "X": 1000.0,
                "Y": 0.0,
                "Z": 2500.0,
                "MD": 2500.0,
            },
        ]
    )[0]
    monkeypatch.setattr(page, "_reference_wells_from_state", lambda: (actual_parent,))

    anchors = page._t1_t3_order_anchor_by_well_name([zbs_record])

    assert anchors["9010_ZBS"].x == pytest.approx(1000.0)
    assert anchors["9010_ZBS"].z == pytest.approx(2500.0)


def test_welltrack_page_initial_crs_selectbox_has_no_session_state_default_warning() -> (
    None
):
    at = AppTest.from_file("pages/01_trajectory_constructor.py")

    at.run(timeout=120)

    assert not any(
        "trajectory_crs_selectbox" in str(widget.value)
        or "trajectory_input_crs_selectbox" in str(widget.value)
        for widget in at.warning
    )


def test_welltrack_page_defaults_csv_crs_to_wgs84_utm43() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")

    at.run(timeout=120)

    select_values = {str(widget.label): widget.value for widget in at.selectbox}
    assert select_values["Входная"] == "ГК_13N_42"
    assert select_values["Доп. в выгрузке"] == "WGS84 UTM 43N"


def test_welltrack_page_limits_csv_crs_options() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")

    at.run(timeout=120)

    selectboxes_by_label = {str(widget.label): widget for widget in at.selectbox}
    assert list(selectboxes_by_label["Доп. в выгрузке"].options) == [
        "ГК_13N_42",
        "WGS84 UTM 43N",
        "WGS84 (градусы)",
    ]


def test_welltrack_page_hides_crs_toggle_and_conversion_signature() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")

    at.run(timeout=120)

    toggle_labels = {str(widget.label) for widget in at.toggle}
    caption_values = [str(widget.value) for widget in at.caption]

    assert "Пересчитывать координаты в CSV" not in toggle_labels
    assert not any("**Входная:**" in value and "**CSV:**" in value for value in caption_values)


def test_welltrack_page_keeps_t1_t3_order_panel_visible_when_no_issues() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run(timeout=120)

    assert any(
        "Проверка порядка t1/t3 — без замечаний." in str(widget.value) for widget in at.success
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
        "перепутаны местами" in str(widget.value) for widget in at.warning
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
    assert any("перепутаны местами" in str(widget.value) for widget in at.warning)
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
    md_postcheck_exceeded: bool = False,
    md_postcheck_message: str = "",
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
        md_postcheck_exceeded=md_postcheck_exceeded,
        md_postcheck_message=md_postcheck_message,
    )


def test_all_wells_plan_figure_keeps_message_only_warning_well_solid() -> None:
    page = wt_import_module

    figure = page._all_wells_plan_figure(
        [
            _successful_plan(
                name="WELL-A",
                y_offset_m=0.0,
                md_postcheck_message="MD warning",
            )
        ]
    )

    warning_trace = next(trace for trace in figure.data if str(trace.name) == "WELL-A")

    assert str(warning_trace.line.dash) == "solid"


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


def _ok_summary_row(name: str) -> dict[str, object]:
    return {
        "Скважина": str(name),
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
    }


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
    pad_select.set_value("PAD2")
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
    pad_select.set_value("PAD2")
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
        "Куст PAD1",
        "Куст PAD2",
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
        "pad::PAD1",
        "pad::PAD2",
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


def test_well_label_display_names_follow_pad_slot_order() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = list(_records())
    pads = page._ensure_pad_configs(records)
    pad_id = str(pads[0].pad_id)
    page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] = (
        (1, "WELL-C"),
        (2, "WELL-A"),
    )

    display_names = page._well_label_display_names(records)

    assert display_names["WELL-C"] == "WELL-C (1)"
    assert display_names["WELL-A"] == "WELL-A (2)"
    assert display_names["WELL-B"] == "WELL-B (3)"


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


def test_page_pad_membership_can_auto_order_all_pads_by_target_depth() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_pad_auto_order_by_target_depth"] = True
    records = [
        WelltrackRecord(
            name="A1",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2200.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2300.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="A2",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2600.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2700.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="B1",
            points=(
                WelltrackPoint(x=5000.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=5600.0, y=800.0, z=2100.0, md=2400.0),
                WelltrackPoint(x=6500.0, y=2000.0, z=2200.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="B2",
            points=(
                WelltrackPoint(x=5000.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=5650.0, y=780.0, z=2500.0, md=2350.0),
                WelltrackPoint(x=6550.0, y=1980.0, z=2600.0, md=3400.0),
            ),
        ),
    ]

    _, _, well_names_by_pad_id = page._pad_membership(records)
    ordered_names = list(well_names_by_pad_id.values())

    assert ordered_names == [("A2", "A1"), ("B2", "B1")]


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
    at.session_state["ptc_reference_import_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_import_welltrack_source_count"] = 2
    at.session_state["wt_reference_import_welltrack_source_path_0"] = str(
        actual_path
    )
    at.session_state["wt_reference_import_welltrack_source_kind_0"] = "actual"
    at.session_state["wt_reference_import_welltrack_source_path_1"] = str(
        approved_path
    )
    at.session_state["wt_reference_import_welltrack_source_kind_1"] = "approved"

    at.run()
    _click_button(at, "Загрузить фонд")
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
    at.session_state["ptc_reference_import_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_import_welltrack_source_path_0"] = str(
        welltrack_path
    )
    at.session_state["wt_reference_import_welltrack_source_kind_0"] = "approved"

    at.run()
    _click_button(at, "Загрузить фонд")
    at.run(timeout=120)

    approved_wells = tuple(at.session_state["wt_reference_approved_wells"])
    assert len(approved_wells) == 1
    assert str(approved_wells[0].name) == "APP-1"


def test_reference_import_migrates_mixed_legacy_sources_without_dropping_welltrack(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    actual_folder = tmp_path / "legacy_dev_actual"
    actual_folder.mkdir()
    (actual_folder / "fact_legacy.dev").write_text(
        "\n".join(
            [
                "MD X Y Z",
                "0 606207.5 7409801.6 40.9",
                "100 606208.5 7409803.6 -59.1",
                "250 606220.5 7409810.6 -209.1",
            ]
        ),
        encoding="utf-8",
    )
    approved_path = tmp_path / "legacy_approved.inc"
    approved_path.write_text(
        "\n".join(
            [
                "WELLTRACK 'APP-LEGACY'",
                "0 0 -35 0",
                "900 850 -35 250",
                "1850 1750 -35 350",
                "/",
            ]
        ),
        encoding="utf-8",
    )

    at.session_state["wt_reference_actual_source_mode"] = "Загрузить .dev"
    at.session_state["wt_reference_actual_dev_folder_count"] = 1
    at.session_state["wt_reference_actual_dev_folder_path_0"] = str(actual_folder)
    at.session_state["wt_reference_approved_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_approved_welltrack_path_count"] = 1
    at.session_state["wt_reference_approved_welltrack_path"] = str(approved_path)

    at.run(timeout=120)

    assert str(at.session_state["ptc_reference_import_source_mode"]) == "Загрузить .dev"
    assert str(at.session_state["wt_reference_import_dev_source_path_0"]) == str(
        actual_folder
    )
    assert str(
        at.session_state["wt_reference_import_welltrack_source_path_0"]
    ) == str(approved_path)

    _click_button(at, "Загрузить фонд")
    at.run(timeout=120)

    actual_wells = tuple(at.session_state["wt_reference_actual_wells"])
    approved_wells = tuple(at.session_state["wt_reference_approved_wells"])

    assert [str(item.name) for item in actual_wells] == ["fact_legacy"]
    assert [str(item.name) for item in approved_wells] == ["APP-LEGACY"]
    assert tuple(at.session_state["ptc_reference_pending_legacy_dev_rows"]) == ()
    assert (
        tuple(at.session_state["ptc_reference_pending_legacy_welltrack_rows"])
        == ()
    )


def test_reference_import_migrates_welltrack_only_legacy_mode(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    actual_path = tmp_path / "legacy_actual.inc"
    actual_path.write_text(
        "\n".join(
            [
                "WELLTRACK 'FACT-LEGACY'",
                "0 0 25 0",
                "950 900 25 300",
                "1900 1800 25 400",
                "/",
            ]
        ),
        encoding="utf-8",
    )

    at.session_state["wt_reference_actual_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_actual_welltrack_path_count"] = 1
    at.session_state["wt_reference_actual_welltrack_path"] = str(actual_path)

    at.run(timeout=120)

    assert (
        str(at.session_state["ptc_reference_import_source_mode"])
        == "Путь к WELLTRACK"
    )

    _click_button(at, "Загрузить фонд")
    at.run(timeout=120)

    actual_wells = tuple(at.session_state["wt_reference_actual_wells"])
    assert [str(item.name) for item in actual_wells] == ["FACT-LEGACY"]


def test_reference_trajectory_welltrack_path_import_supports_multiple_paths_and_folder(
    tmp_path,
) -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    direct_file = tmp_path / "actual_direct.inc"
    direct_file.write_text(
        "\n".join(
            [
                "WELLTRACK 'FACT-DIRECT'",
                "0 0 20 0",
                "900 800 20 250",
                "1800 1600 20 350",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    folder = tmp_path / "actual_folder"
    folder.mkdir()
    (folder / "actual_02.InC").write_text(
        "\n".join(
            [
                "WELLTRACK 'FACT-FOLDER-A'",
                "0 0 30 0",
                "850 780 30 240",
                "1700 1560 30 340",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    (folder / "actual_03.INC").write_text(
        "\n".join(
            [
                "WELLTRACK 'FACT-FOLDER-B'",
                "0 0 40 0",
                "870 790 40 245",
                "1740 1580 40 345",
                "/",
            ]
        ),
        encoding="utf-8",
    )
    (folder / "ignore_me.txt").write_text("skip", encoding="utf-8")
    at.session_state["ptc_reference_import_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_import_welltrack_source_count"] = 2
    at.session_state["wt_reference_import_welltrack_source_path_0"] = str(
        direct_file
    )
    at.session_state["wt_reference_import_welltrack_source_kind_0"] = "actual"
    at.session_state["wt_reference_import_welltrack_source_path_1"] = str(folder)
    at.session_state["wt_reference_import_welltrack_source_kind_1"] = "actual"

    at.run(timeout=120)
    _click_button(at, "Загрузить фонд")
    at.run(timeout=120)

    actual_wells = tuple(at.session_state["wt_reference_actual_wells"])
    assert [str(well.name) for well in actual_wells] == [
        "FACT-DIRECT",
        "FACT-FOLDER-A",
        "FACT-FOLDER-B",
    ]


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
    at.session_state["wt_reference_import_dev_source_count"] = 2
    at.session_state["wt_reference_import_dev_source_path_0"] = str(folder_a)
    at.session_state["wt_reference_import_dev_source_kind_0"] = "actual"
    at.session_state["wt_reference_import_dev_source_path_1"] = str(folder_b)
    at.session_state["wt_reference_import_dev_source_kind_1"] = "actual"

    at.run(timeout=120)
    _click_button(at, "Загрузить фонд")
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
    at.session_state["wt_reference_import_dev_source_count"] = 2
    at.session_state["wt_reference_import_dev_source_path_0"] = str(
        actual_folder
    )
    at.session_state["wt_reference_import_dev_source_kind_0"] = "actual"
    at.session_state["wt_reference_import_dev_source_path_1"] = str(
        approved_folder
    )
    at.session_state["wt_reference_import_dev_source_kind_1"] = "approved"

    at.run(timeout=120)
    _click_button(at, "Загрузить фонд")
    at.run(timeout=120)

    reference_wells = tuple(at.session_state["wt_reference_wells"])
    assert [(str(well.name), str(well.kind)) for well in reference_wells] == [
        ("well_same", "actual"),
        ("well_same", "approved"),
    ]


def test_reference_analysis_panels_are_lazy_by_default() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    reference_wells = _reference_wells()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [_ok_summary_row("WELL-A")]
    at.session_state["wt_successes"] = [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    at.session_state["wt_reference_wells"] = list(reference_wells)
    at.session_state["wt_reference_actual_wells"] = [
        item for item in reference_wells if str(item.kind) == "actual"
    ]
    at.session_state["wt_reference_approved_wells"] = [
        item for item in reference_wells if str(item.kind) == "approved"
    ]

    at.run(timeout=120)

    toggle_labels = {str(widget.label) for widget in at.toggle}
    selectbox_labels = {str(widget.label) for widget in at.selectbox}

    assert "Показать анализ фактического фонда" in toggle_labels
    assert "Показать просмотр утверждённого проектного фонда" in toggle_labels
    assert "Просмотр фактической скважины" not in selectbox_labels
    assert "Просмотр утвержденной проектной скважины" not in selectbox_labels


def test_reference_analysis_panels_render_after_explicit_toggle() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    reference_wells = _reference_wells()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [_ok_summary_row("WELL-A")]
    at.session_state["wt_successes"] = [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    at.session_state["wt_reference_wells"] = list(reference_wells)
    at.session_state["wt_reference_actual_wells"] = [
        item for item in reference_wells if str(item.kind) == "actual"
    ]
    at.session_state["wt_reference_approved_wells"] = [
        item for item in reference_wells if str(item.kind) == "approved"
    ]
    at.session_state["wt_show_actual_fund_analysis"] = True
    at.session_state["wt_show_approved_fund_analysis"] = True

    at.run(timeout=120)

    selectbox_labels = {str(widget.label) for widget in at.selectbox}

    assert "Просмотр фактической скважины" in selectbox_labels
    assert "Просмотр утвержденной проектной скважины" in selectbox_labels


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
    at.session_state["wt_reference_import_dev_source_count"] = 2
    at.session_state["wt_reference_import_dev_source_path_0"] = str(folder_a)
    at.session_state["wt_reference_import_dev_source_kind_0"] = "actual"
    at.session_state["wt_reference_import_dev_source_path_1"] = str(folder_b)
    at.session_state["wt_reference_import_dev_source_kind_1"] = "approved"

    at.run(timeout=120)
    _click_button(at, "Очистить фонд")
    at.run(timeout=120)

    assert not at.exception
    assert int(at.session_state["wt_reference_import_dev_source_count"]) == 1
    assert str(at.session_state["wt_reference_import_dev_source_path_0"]) == ""
    assert str(at.session_state["wt_reference_import_dev_source_path_1"]) == ""


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
    at.session_state["ptc_reference_import_source_mode"] = "Путь к WELLTRACK"
    at.session_state["wt_reference_import_welltrack_source_count"] = 2
    at.session_state["wt_reference_import_welltrack_source_path_0"] = str(
        welltrack_path
    )
    at.session_state["wt_reference_import_welltrack_source_kind_0"] = "actual"
    at.session_state["wt_reference_import_welltrack_source_path_1"] = str(tmp_path)
    at.session_state["wt_reference_import_welltrack_source_kind_1"] = "approved"

    at.run(timeout=120)
    _click_button(at, "Очистить фонд")
    at.run(timeout=120)

    assert not at.exception
    assert int(at.session_state["wt_reference_import_welltrack_source_count"]) == 1
    assert str(at.session_state["wt_reference_import_welltrack_source_path_0"]) == ""
    assert str(at.session_state["wt_reference_import_welltrack_source_path_1"]) == ""


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


def test_dev_target_import_stores_read_parameters_and_target_points() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_format"] = ".dev траектория"
    at.session_state["wt_source_mode"] = "Файл по пути"
    at.session_state["wt_source_path"] = "tests/test_data/dev_target_import"

    at.run(timeout=120)
    _click_button(at, "Импорт целей")
    at.run(timeout=120)

    records = at.session_state["wt_records"]
    imported_params = at.session_state["wt_imported_dev_params"]
    imported_dev_wells = at.session_state["wt_imported_dev_target_wells"]

    assert [record.name for record in records] == [
        "build_hold_build_equal_pi_with_horizontal_pi",
        "build_hold_build_split_pi",
        "j_profile_constant_pi",
        "j_profile_variable_pi",
    ]
    assert len(imported_params) == 4
    assert len(imported_dev_wells) == 4
    assert imported_params[0].build1_dls_deg_per_30m == (2.4,)
    assert imported_params[0].build2_dls_deg_per_30m == (2.4, 1.2)
    assert imported_params[0].horizontal_dls_deg_per_30m == ()
    assert imported_params[-1].profile_label == "J-профиль"
    assert at.session_state[wt_import_module.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] is True
    assert at.session_state[wt_import_module.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        name: wt_import_module._manual_well_calc_profile_id_from_dev_summary(name)
        for name in [record.name for record in records]
    }
    assert {
        payload["source"]
        for payload in at.session_state[
            wt_import_module.WT_WELL_CALC_OVERRIDE_STATE_KEY
        ].values()
    } == {"Импорт .dev"}
    markdown_values = [str(widget.value) for widget in at.markdown]
    assert any("Прочитанные параметры .dev" in value for value in markdown_values)


def test_dev_target_import_supports_fixed_inc_t1_selection() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_format"] = ".dev траектория"
    at.session_state["wt_source_mode"] = "Файл по пути"
    at.session_state["wt_source_path"] = (
        "tests/test_data/dev_target_import/build_hold_build_split_pi.dev"
    )
    at.session_state["wt_source_dev_fixed_t1_enabled"] = True
    at.session_state["wt_source_dev_fixed_t1_well_names"] = [
        "build_hold_build_split_pi"
    ]
    at.session_state[
        wt_import_module._dev_fixed_t1_input_key("build_hold_build_split_pi")
    ] = 70.0

    at.run(timeout=120)
    _click_button(at, "Импорт целей")
    at.run(timeout=120)

    records = at.session_state["wt_records"]
    imported_params = at.session_state["wt_imported_dev_params"]

    assert [record.name for record in records] == ["build_hold_build_split_pi"]
    assert float(records[0].points[1].md) == pytest.approx(2860.0)
    assert imported_params[0].t1_md_m == pytest.approx(2860.0)
    assert imported_params[0].entry_inc_deg == pytest.approx(71.7)
    assert imported_params[0].horizontal_dls_deg_per_30m == (2.1,)
    assert "t1 по INC >= 70.0 deg" in str(imported_params[0].note)


def test_dev_fixed_t1_input_keys_do_not_collide_for_similar_names() -> None:
    page = wt_import_module

    first_key = page._dev_fixed_t1_input_key("WELL-1")
    second_key = page._dev_fixed_t1_input_key("WELL_1")

    assert first_key != second_key


def test_build_source_payload_keeps_individual_fixed_t1_thresholds_for_similar_names() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_source_format"] = page.WT_SOURCE_FORMAT_DEV_TRAJECTORY
    page.st.session_state["wt_source_mode"] = page.WT_SOURCE_MODE_FILE_PATH
    page.st.session_state["wt_source_dev_fixed_t1_enabled"] = True
    page.st.session_state["wt_source_dev_fixed_t1_well_names"] = ["WELL-1", "WELL_1"]
    page.st.session_state[page._dev_fixed_t1_input_key("WELL-1")] = 70.0
    page.st.session_state[page._dev_fixed_t1_input_key("WELL_1")] = 82.5

    payload = page._build_source_payload_from_state()

    assert payload.dev_fixed_t1_inc_by_well == (
        ("WELL-1", 70.0),
        ("WELL_1", 82.5),
    )


def test_sync_dev_fixed_t1_selection_uses_hard_default_for_new_wells() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_source_dev_fixed_t1_well_names"] = ["WELL-A", "WELL-B"]
    page.st.session_state["wt_source_dev_fixed_t1_inc_deg"] = 0.5

    normalized_names, selected_names = page._sync_dev_fixed_t1_selection_state(
        ["WELL-A", "WELL-B", "WELL-C"]
    )

    assert normalized_names == ("WELL-A", "WELL-B", "WELL-C")
    assert selected_names == ("WELL-A", "WELL-B")
    assert page.st.session_state[page._dev_fixed_t1_input_key("WELL-A")] == pytest.approx(
        86.0
    )
    assert page.st.session_state[page._dev_fixed_t1_input_key("WELL-B")] == pytest.approx(
        86.0
    )


def test_build_source_payload_ignores_legacy_fixed_t1_default_for_missing_well_keys() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_source_format"] = page.WT_SOURCE_FORMAT_DEV_TRAJECTORY
    page.st.session_state["wt_source_mode"] = page.WT_SOURCE_MODE_FILE_PATH
    page.st.session_state["wt_source_dev_fixed_t1_enabled"] = True
    page.st.session_state["wt_source_dev_fixed_t1_well_names"] = ["WELL-A"]
    page.st.session_state["wt_source_dev_fixed_t1_inc_deg"] = 0.5

    payload = page._build_source_payload_from_state()

    assert payload.dev_fixed_t1_inc_by_well == (("WELL-A", 86.0),)


def test_build_source_payload_uses_common_fixed_t1_threshold_for_selected_wells() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_source_format"] = page.WT_SOURCE_FORMAT_DEV_TRAJECTORY
    page.st.session_state["wt_source_mode"] = page.WT_SOURCE_MODE_FILE_PATH
    page.st.session_state["wt_source_dev_fixed_t1_enabled"] = True
    page.st.session_state["wt_source_dev_fixed_t1_common_enabled"] = True
    page.st.session_state["wt_source_dev_fixed_t1_common_inc_deg"] = 82.5
    page.st.session_state["wt_source_dev_fixed_t1_well_names"] = ["WELL-1", "WELL_1"]
    page.st.session_state[page._dev_fixed_t1_input_key("WELL-1")] = 70.0
    page.st.session_state[page._dev_fixed_t1_input_key("WELL_1")] = 88.0

    payload = page._build_source_payload_from_state()

    assert payload.dev_fixed_t1_inc_by_well == (
        ("WELL-1", 82.5),
        ("WELL_1", 82.5),
    )


def test_build_source_payload_defaults_invalid_source_format_to_target_table() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_source_format"] = "legacy-invalid"
    page.st.session_state["wt_source_mode"] = page.WT_SOURCE_MODE_TARGET_TABLE
    page.st.session_state["wt_source_table_df"] = pd.DataFrame(
        [{"Wellname": "TAB-01", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0}]
    )

    payload = page._build_source_payload_from_state()

    assert payload.source_format == page.WT_SOURCE_FORMAT_TARGET_TABLE
    assert payload.mode == page.WT_SOURCE_MODE_TARGET_TABLE
    assert list(payload.table_rows["Wellname"]) == ["TAB-01"]


def test_build_source_payload_infers_welltrack_for_invalid_format_with_prefilled_path() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_source_format"] = "legacy-invalid"
    page.st.session_state["wt_source_mode"] = page.WT_SOURCE_MODE_FILE_PATH
    page.st.session_state["wt_source_path"] = "tests/test_data/WELLTRACKS2.INC"

    payload = page._build_source_payload_from_state()

    assert payload.source_format == page.WT_SOURCE_FORMAT_WELLTRACK
    assert payload.mode == page.WT_SOURCE_MODE_FILE_PATH
    assert "WELLTRACK" in payload.source_text


def test_sync_dev_fixed_t1_selection_uses_pending_names() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_source_dev_fixed_t1_pending_well_names"] = [
        "WELL-A",
        "WELL-B",
    ]

    normalized_names, selected_names = page._sync_dev_fixed_t1_selection_state(
        ["WELL-A", "WELL-B", "WELL-C"]
    )

    assert normalized_names == ("WELL-A", "WELL-B", "WELL-C")
    assert selected_names == ("WELL-A", "WELL-B")
    assert page.st.session_state["wt_source_dev_fixed_t1_well_names"] == [
        "WELL-A",
        "WELL-B",
    ]
    assert "wt_source_dev_fixed_t1_pending_well_names" not in page.st.session_state


def test_render_dev_fixed_t1_controls_select_all_queues_pending_selection(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    rerun_calls: list[str] = []

    class _DummyColumn:
        def button(self, label, **kwargs):
            return str(label) == "Выбрать все"

    monkeypatch.setattr(
        page.st,
        "columns",
        lambda spec, *args, **kwargs: tuple(
            _DummyColumn()
            for _ in range(int(spec) if isinstance(spec, int) else len(spec))
        ),
    )
    monkeypatch.setattr(
        page.st,
        "toggle",
        lambda _label, key, **kwargs: key == "wt_source_dev_fixed_t1_enabled",
    )
    monkeypatch.setattr(
        page.st,
        "multiselect",
        lambda _label, options, key, **kwargs: page.st.session_state.setdefault(
            key, []
        ),
    )
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "number_input", lambda *args, **kwargs: 86.0)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page, "_rerun_fragment", lambda: rerun_calls.append("rerun"))

    page._render_dev_fixed_t1_controls(available_names=["WELL-A", "WELL-B"])

    assert page.st.session_state["wt_source_dev_fixed_t1_pending_well_names"] == [
        "WELL-A",
        "WELL-B",
    ]
    assert rerun_calls == ["rerun"]


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


def test_auto_pad_layout_applies_for_degenerate_three_point_surface_records() -> None:
    page = wt_import_module
    page._init_state()

    records = _degenerate_surface_pad_records()
    original_surfaces = {
        str(record.name): (
            float(record.points[0].x),
            float(record.points[0].y),
            float(record.points[0].z),
        )
        for record in records
    }

    applied = page._auto_apply_pad_layout_if_shared_surface(list(records))

    assert applied is True
    pads = page._ensure_pad_configs(list(records))
    pad_id = str(pads[0].pad_id)
    metadata = page.st.session_state["wt_pad_detected_meta"]
    assert [len(pad.wells) for pad in pads] == [3]
    assert bool(metadata[pad_id].source_surfaces_defined) is False

    updated_records = page.st.session_state["wt_records"]
    updated_surfaces = {
        str(record.name): (
            float(record.points[0].x),
            float(record.points[0].y),
            float(record.points[0].z),
        )
        for record in updated_records
    }
    assert updated_surfaces != original_surfaces
    assert len(set(updated_surfaces.values())) == 3
    assert {value[2] for value in updated_surfaces.values()} == {0.0}
    assert page.st.session_state["wt_pad_auto_applied_on_import"] is True
    assert page.st.session_state["wt_pad_last_applied_at"] != ""


def test_auto_pad_layout_uses_canonical_pad_plan_map_for_imported_multihorizontal_data() -> (
    None
):
    page = wt_import_module
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS4_MULTIHORIZONTAL.INC").read_text(
            encoding="utf-8"
        )
    )

    page._init_state()
    pads = page._ensure_pad_configs(list(records))
    page._set_import_pilot_fixed_slots(records=list(records), pads=pads)
    expected_records = sync_pilot_surfaces_to_parents(
        apply_pad_layout(
            records=list(records),
            pads=pads,
            plan_by_pad_id=page._build_pad_plan_map(pads),
        )
    )
    expected_surfaces = {
        str(record.name): (
            float(record.points[0].x),
            float(record.points[0].y),
            float(record.points[0].z),
        )
        for record in expected_records
        if record.points
    }

    page._init_state()
    applied = page._auto_apply_pad_layout_if_shared_surface(list(records))

    assert applied is True
    actual_records = page.st.session_state["wt_records"]
    actual_surfaces = {
        str(record.name): (
            float(record.points[0].x),
            float(record.points[0].y),
            float(record.points[0].z),
        )
        for record in actual_records
        if record.points
    }
    assert actual_surfaces == expected_surfaces


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
    pads = page._ensure_pad_configs(list(page.st.session_state["wt_records_original"]))
    pad_id = str(pads[0].pad_id)

    assert page.st.session_state["wt_pad_configs"][pad_id]["fixed_slots"] == (
        (1, "WELL-A"),
    )
    assert [str(well.name) for well in pads[0].wells] == ["WELL-B", "WELL-A"]


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
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
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


def test_pad_layout_preview_hides_midpoint_anchor_and_surface_z_columns(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
    captured: dict[str, object] = {"preview_columns": None}

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

    def _fake_dataframe(data, *args, **kwargs):
        if isinstance(data, pd.DataFrame) and "Порядок" in data.columns:
            captured["preview_columns"] = list(data.columns)

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: frame)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)

    page._render_pad_layout_panel(list(_records()))

    assert captured["preview_columns"] == [
        "Порядок",
        "Скважина",
        "Фиксация",
        "Новое S X, м",
        "Новое S Y, м",
    ]


def test_pad_layout_can_hide_selected_pad_details(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = False
    captured: dict[str, object] = {
        "button_labels": [],
        "selectbox_labels": [],
        "toggle_labels": [],
        "data_editor_calls": 0,
    }

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_selectbox(label, options, *args, **kwargs):
        captured["selectbox_labels"].append(str(label))
        key = kwargs.get("key")
        value = page.st.session_state.get(key, options[0])
        if key is not None:
            page.st.session_state[key] = value
        return value

    def _fake_toggle(label, *args, **kwargs):
        captured["toggle_labels"].append(str(label))
        return False

    def _fake_button(label, *args, **kwargs):
        captured["button_labels"].append(str(label))
        return False

    def _fake_data_editor(*args, **kwargs):
        captured["data_editor_calls"] = int(captured["data_editor_calls"]) + 1
        return None

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "button", _fake_button)
    monkeypatch.setattr(page.st, "data_editor", _fake_data_editor)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)

    page._render_pad_layout_panel(list(_records()))

    assert (
        "Настроить положение куста, НДС и расстояние между устьями."
        in captured["button_labels"]
    )
    assert captured["selectbox_labels"] == []
    assert captured["toggle_labels"] == ["Авто-порядок по глубине целевого пласта"]
    assert captured["data_editor_calls"] == 0


def test_pad_layout_hides_selected_pad_details_by_default(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    captured: dict[str, object] = {
        "button_labels": [],
        "selectbox_labels": [],
        "toggle_labels": [],
        "data_editor_calls": 0,
    }

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_selectbox(label, options, *args, **kwargs):
        captured["selectbox_labels"].append(str(label))
        key = kwargs.get("key")
        value = page.st.session_state.get(key, options[0])
        if key is not None:
            page.st.session_state[key] = value
        return value

    def _fake_toggle(label, *args, **kwargs):
        captured["toggle_labels"].append(str(label))
        return False

    def _fake_button(label, *args, **kwargs):
        captured["button_labels"].append(str(label))
        return False

    def _fake_data_editor(*args, **kwargs):
        captured["data_editor_calls"] = int(captured["data_editor_calls"]) + 1
        return None

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "button", _fake_button)
    monkeypatch.setattr(page.st, "data_editor", _fake_data_editor)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)

    page._render_pad_layout_panel(list(_records()))

    assert (
        "Настроить положение куста, НДС и расстояние между устьями."
        in captured["button_labels"]
    )
    assert captured["selectbox_labels"] == []
    assert captured["toggle_labels"] == ["Авто-порядок по глубине целевого пласта"]
    assert captured["data_editor_calls"] == 0


def test_pad_layout_clear_fixed_slots_resets_editor_key(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
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
        "Устья заданы в исходных данных" in str(widget.value)
        for widget in at.info
    )
    assert any(
        "Исходная привязка скважин к позициям сохраняется" in str(widget.value)
        for widget in at.info
    )


def test_pad_layout_offers_toggle_to_edit_source_defined_positions(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
    records = list(_prepositioned_pad_records())
    captured: dict[str, object] = {"toggle_labels": []}

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

    def _fake_toggle(label, *args, **kwargs):
        captured["toggle_labels"].append(str(label))
        key = kwargs.get("key")
        value = False
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: frame)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        lambda label, *args, **kwargs: {"label": str(label), **kwargs},
    )

    page._render_pad_layout_panel(records)

    assert "Разрешить редактирование позиций куста" in captured["toggle_labels"]
    assert "Применить авто-порядок" in captured["toggle_labels"]


def test_pad_layout_unlocks_source_defined_inputs_when_editing_enabled(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
    records = list(_prepositioned_pad_records())
    captured: dict[str, object] = {
        "number_inputs": [],
        "buttons": [],
    }

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def number_input(self, label, *args, **kwargs):
            captured["number_inputs"].append(
                (str(label), bool(kwargs.get("disabled", False)))
            )
            key = kwargs.get("key")
            return page.st.session_state.get(key, 0.0)

        def button(self, label, *args, **kwargs):
            captured["buttons"].append(
                (str(label), bool(kwargs.get("disabled", False)))
            )
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

    def _fake_toggle(label, *args, **kwargs):
        key = kwargs.get("key")
        value = str(label) == "Разрешить редактирование позиций куста"
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: frame)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        lambda label, *args, **kwargs: {"label": str(label), **kwargs},
    )

    page._render_pad_layout_panel(records)

    disabled_by_label = dict(captured["number_inputs"])
    assert disabled_by_label["Расстояние между устьями, м"] is False
    assert disabled_by_label["НДС (азимут), deg"] is False
    button_disabled = dict(captured["buttons"])
    assert button_disabled["Рассчитать устья скважин"] is False
    assert button_disabled["Вернуть исходные устья"] is False


def test_pad_layout_disables_apply_auto_order_until_source_editing_is_enabled(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
    records = list(_prepositioned_pad_records())
    captured: dict[str, object] = {"toggle_state": {}}

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

    def _fake_toggle(label, *args, **kwargs):
        captured["toggle_state"][str(label)] = bool(kwargs.get("disabled", False))
        key = kwargs.get("key")
        value = False
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: frame)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        lambda label, *args, **kwargs: {"label": str(label), **kwargs},
    )

    page._render_pad_layout_panel(records)

    assert captured["toggle_state"]["Применить авто-порядок"] is True


def test_detect_ui_pads_uses_imported_simple_dev_state_after_t1_edit() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    records = [
        WelltrackRecord(
            name="9201",
            points=(
                WelltrackPoint(x=1000.0, y=800.0, z=0.0, md=0.0),
                WelltrackPoint(x=1120.0, y=880.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=2020.0, y=2080.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="9202",
            points=(
                WelltrackPoint(x=1600.0, y=900.0, z=0.0, md=0.0),
                WelltrackPoint(x=1720.0, y=980.0, z=2350.0, md=2350.0),
                WelltrackPoint(x=2620.0, y=2180.0, z=2450.0, md=3450.0),
            ),
        ),
        WelltrackRecord(
            name="9203",
            points=(
                WelltrackPoint(x=2200.0, y=1000.0, z=0.0, md=0.0),
                WelltrackPoint(x=2320.0, y=1080.0, z=2300.0, md=2300.0),
                WelltrackPoint(x=3220.0, y=2280.0, z=2400.0, md=3400.0),
            ),
        ),
    ]
    page.st.session_state[page.ptc_target_import.IMPORTED_DEV_TARGET_WELLS_STATE_KEY] = (
        ImportedTrajectoryWell(
            name="9201",
            kind="approved",
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 2400.0, 3500.0],
                    "X_m": [1000.0, 1000.0, 1900.0],
                    "Y_m": [800.0, 800.0, 2000.0],
                    "Z_m": [0.0, 2400.0, 2500.0],
                }
            ),
            surface=Point3D(x=1000.0, y=800.0, z=0.0),
            azimuth_deg=45.0,
        ),
        ImportedTrajectoryWell(
            name="9202",
            kind="approved",
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 2350.0, 3450.0],
                    "X_m": [1600.0, 1600.0, 2500.0],
                    "Y_m": [900.0, 900.0, 2100.0],
                    "Z_m": [0.0, 2350.0, 2450.0],
                }
            ),
            surface=Point3D(x=1600.0, y=900.0, z=0.0),
            azimuth_deg=45.0,
        ),
        ImportedTrajectoryWell(
            name="9203",
            kind="approved",
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 2300.0, 3400.0],
                    "X_m": [2200.0, 2200.0, 3100.0],
                    "Y_m": [1000.0, 1000.0, 2200.0],
                    "Z_m": [0.0, 2300.0, 2400.0],
                }
            ),
            surface=Point3D(x=2200.0, y=1000.0, z=0.0),
            azimuth_deg=45.0,
        ),
    )

    pads, metadata = page._detect_ui_pads(records)

    assert [len(pad.wells) for pad in pads] == [3]
    assert [str(pad.pad_id) for pad in pads] == ["Pad 92"]
    assert bool(metadata["Pad 92"].source_surfaces_defined) is False


def test_pad_layout_keeps_user_nds_value_for_source_defined_pad(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
    records = list(_prepositioned_pad_records())
    pads = page._detect_ui_pads(records)[0]
    pad_id = str(pads[0].pad_id)
    nds_widget_key = f"wt_pad_cfg_nds_azimuth_deg_{pad_id}"
    page.st.session_state["wt_pad_selected_id"] = pad_id
    page.st.session_state[nds_widget_key] = 123.0

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

    def _fake_toggle(label, *args, **kwargs):
        key = kwargs.get("key")
        value = str(label) == "Разрешить редактирование позиций куста"
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: frame)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        lambda label, *args, **kwargs: {"label": str(label), **kwargs},
    )

    page._render_pad_layout_panel(records)

    assert page.st.session_state[nds_widget_key] == 123.0


def test_pad_layout_resets_source_defined_widgets_when_editing_is_disabled(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
    records = list(_prepositioned_pad_records())
    pads = page._ensure_pad_configs(records)
    pad_id = str(pads[0].pad_id)
    page.st.session_state["wt_pad_selected_id"] = pad_id
    page.st.session_state["wt_pad_configs"][pad_id].update(
        {
            "spacing_m": 999.0,
            "nds_azimuth_deg": 123.0,
            "first_surface_x": -1.0,
            "first_surface_y": -2.0,
            "first_surface_z": -3.0,
            "surface_anchor_mode": "first",
            page.ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY: True,
            page.ptc_pad_state.WT_PAD_APPLY_AUTO_ORDER_KEY: True,
        }
    )
    page.st.session_state[f"wt_pad_cfg_spacing_m_{pad_id}"] = 999.0
    page.st.session_state[f"wt_pad_cfg_nds_azimuth_deg_{pad_id}"] = 123.0
    page.st.session_state[f"wt_pad_cfg_first_surface_x_{pad_id}"] = -1.0
    page.st.session_state[f"wt_pad_cfg_first_surface_y_{pad_id}"] = -2.0
    page.st.session_state[f"wt_pad_cfg_first_surface_z_{pad_id}"] = -3.0
    page.st.session_state[f"wt_pad_cfg_surface_anchor_center_{pad_id}"] = False
    page.st.session_state[f"wt_pad_cfg_apply_auto_order_{pad_id}"] = True

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

    def _fake_toggle(label, *args, **kwargs):
        key = kwargs.get("key")
        if bool(kwargs.get("disabled", False)) and key is not None:
            return page.st.session_state.get(key, False)
        value = False
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: frame)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        lambda label, *args, **kwargs: {"label": str(label), **kwargs},
    )

    page._render_pad_layout_panel(records)

    cfg = page._pad_config_for_ui(pads[0])
    assert page.st.session_state[f"wt_pad_cfg_spacing_m_{pad_id}"] == float(
        cfg["spacing_m"]
    )
    assert page.st.session_state[f"wt_pad_cfg_nds_azimuth_deg_{pad_id}"] == float(
        cfg["nds_azimuth_deg"]
    )
    assert page.st.session_state[f"wt_pad_cfg_first_surface_x_{pad_id}"] == float(
        cfg["first_surface_x"]
    )
    assert page.st.session_state[f"wt_pad_cfg_first_surface_y_{pad_id}"] == float(
        cfg["first_surface_y"]
    )
    assert page.st.session_state[f"wt_pad_cfg_first_surface_z_{pad_id}"] == float(
        cfg["first_surface_z"]
    )
    assert (
        page.st.session_state[f"wt_pad_cfg_surface_anchor_center_{pad_id}"] is True
    )
    assert page.st.session_state[f"wt_pad_cfg_apply_auto_order_{pad_id}"] is False


def test_pad_layout_preview_applies_auto_order_only_after_explicit_toggle(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state[page.ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = True
    records = list(_prepositioned_depth_order_pad_records())
    page.st.session_state["wt_pad_auto_order_by_target_depth"] = True
    captured_frames: list[pd.DataFrame] = []
    apply_auto_order_enabled = False

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

    def _fake_toggle(label, *args, **kwargs):
        key = kwargs.get("key")
        if str(label) == "Разрешить редактирование позиций куста":
            value = True
        elif str(label) == "Применить авто-порядок":
            value = apply_auto_order_enabled
        elif bool(kwargs.get("disabled", False)) and key is not None:
            value = page.st.session_state.get(key, False)
        else:
            value = page.st.session_state.get(key, False)
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "container", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "dataframe",
        lambda frame, *args, **kwargs: captured_frames.append(frame.copy()),
    )
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: frame)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st.column_config,
        "SelectboxColumn",
        lambda label, *args, **kwargs: {"label": str(label), **kwargs},
    )

    page._render_pad_layout_panel(records)
    preview_without_auto = list(captured_frames[-1]["Скважина"])

    apply_auto_order_enabled = True
    captured_frames.clear()
    page._render_pad_layout_panel(records)
    preview_with_auto = list(captured_frames[-1]["Скважина"])

    assert preview_without_auto == ["P1-A", "P1-B", "P1-C"]
    assert preview_with_auto == ["P1-B", "P1-C", "P1-A"]


def test_welltrack_page_shows_template_pad_name_notice() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _template_pad_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records

    at.run(timeout=120)

    assert any(
        'шаблонное название куста "PAD-01"' in str(widget.value)
        for widget in at.info
    )


def test_welltrack_import_source_selector_splits_format_and_welltrack_method() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")

    at.run(timeout=120)

    radios_by_label = {str(widget.label): widget for widget in at.radio}
    assert list(radios_by_label["Выберите формат импорта:"].options) == [
        "Таблица с точками целей",
        "WELLTRACK",
        ".dev траектория",
    ]
    assert radios_by_label["Выберите формат импорта:"].value == "Таблица с точками целей"
    assert set(radios_by_label) == {"Выберите формат импорта:"}
    assert "**Выберите формат импорта:**" in {
        str(widget.value) for widget in at.markdown
    }


def test_welltrack_import_target_table_format_hides_welltrack_method() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_format"] = "Таблица с точками целей"

    at.run(timeout=120)

    radio_labels = {str(widget.label) for widget in at.radio}
    assert radio_labels == {"Выберите формат импорта:"}
    assert [str(widget.label) for widget in at.expander] == [
        "Таблица точек целей",
        "Как задавать скважины с пилотом, ЗБС и многопластовые скважины",
    ]


def test_dev_target_import_format_shows_dev_loading_methods() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_format"] = ".dev траектория"

    at.run(timeout=120)

    radios_by_label = {str(widget.label): widget for widget in at.radio}
    assert list(radios_by_label["Способ загрузки .dev"].options) == [
        "Файл по пути",
        "Загрузить файл",
        "Вставить текст",
    ]


def test_welltrack_upload_mode_renders_without_widget_state_crash() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_format"] = "WELLTRACK"
    at.session_state["wt_source_mode"] = "Загрузить файл"

    at.run(timeout=120)

    assert not at.exception
    assert "Импорт целей" in {str(widget.label) for widget in at.button}


def test_dev_upload_mode_renders_without_widget_state_crash() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_format"] = ".dev траектория"
    at.session_state["wt_source_mode"] = "Загрузить файл"

    at.run(timeout=120)

    assert not at.exception
    assert "Импорт целей" in {str(widget.label) for widget in at.button}


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


def test_source_table_editor_keeps_canonical_rows_unchanged_until_apply(monkeypatch) -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    original_df = pd.DataFrame(
        [
            {"Wellname": "TAB-01", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "TAB-01", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {"Wellname": "TAB-01", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
        ]
    )
    edited_df = pd.DataFrame(
        [
            {"Wellname": "TAB-01", "Point": "wellhead", "X": 10.0, "Y": 20.0, "Z": 30.0},
            {"Wellname": "TAB-01", "Point": "t1", "X": 610.0, "Y": 810.0, "Z": 2410.0},
            {"Wellname": "TAB-01", "Point": "t3", "X": 1510.0, "Y": 2010.0, "Z": 2510.0},
        ]
    )
    page.st.session_state["wt_source_format"] = page.WT_SOURCE_FORMAT_TARGET_TABLE
    page.st.session_state["wt_source_table_df"] = original_df.copy()

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_radio(_label, options, *args, **kwargs):
        key = kwargs.get("key")
        value = page.st.session_state.get(key, options[0])
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyContext(), _DummyContext()),
    )
    monkeypatch.setattr(page.st, "radio", _fake_radio)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: edited_df.copy())
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        page.st.column_config,
        "TextColumn",
        lambda *args, **kwargs: {"type": "text"},
    )
    monkeypatch.setattr(
        page.st.column_config,
        "NumberColumn",
        lambda *args, **kwargs: {"type": "number"},
    )

    page._render_source_input()

    pd.testing.assert_frame_equal(
        page.st.session_state["wt_source_table_df"].reset_index(drop=True),
        original_df.reset_index(drop=True),
    )
    payload = page._build_source_payload_from_state()
    pd.testing.assert_frame_equal(
        payload.table_rows.reset_index(drop=True),
        original_df.reset_index(drop=True),
    )


def test_source_table_editor_applies_rows_on_first_submit(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    original_df = pd.DataFrame(
        [
            {"Wellname": "TAB-01", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "TAB-01", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {"Wellname": "TAB-01", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
        ]
    )
    edited_df = pd.DataFrame(
        [
            {"Wellname": "TAB-01", "Point": "wellhead", "X": 10.0, "Y": 20.0, "Z": 30.0},
            {"Wellname": "TAB-01", "Point": "t1", "X": 610.0, "Y": 810.0, "Z": 2410.0},
            {"Wellname": "TAB-01", "Point": "t3", "X": 1510.0, "Y": 2010.0, "Z": 2510.0},
        ]
    )
    page.st.session_state["wt_source_format"] = page.WT_SOURCE_FORMAT_TARGET_TABLE
    page.st.session_state["wt_source_table_df"] = original_df.copy()

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_radio(_label, options, *args, **kwargs):
        key = kwargs.get("key")
        value = page.st.session_state.get(key, options[0])
        if key is not None:
            page.st.session_state[key] = value
        return value

    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyContext(), _DummyContext()),
    )
    monkeypatch.setattr(page.st, "radio", _fake_radio)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: edited_df.copy())
    monkeypatch.setattr(page.st, "form_submit_button", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        page.st.column_config,
        "TextColumn",
        lambda *args, **kwargs: {"type": "text"},
    )
    monkeypatch.setattr(
        page.st.column_config,
        "NumberColumn",
        lambda *args, **kwargs: {"type": "number"},
    )

    page._render_source_input()

    applied_df = page.st.session_state["wt_source_table_df"].reset_index(drop=True)
    assert list(applied_df["Point"]) == ["S", "t1", "t3"]
    assert list(applied_df["X"]) == [10.0, 610.0, 1510.0]
    payload = page._build_source_payload_from_state()
    assert payload.mode == page.WT_SOURCE_MODE_TARGET_TABLE
    assert payload.table_rows is not None
    assert list(payload.table_rows.reset_index(drop=True)["X"]) == [10.0, 610.0, 1510.0]


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


def test_welltrack_page_prompts_to_run_anticollision_for_successful_batch() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()[:2]
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        _ok_summary_row("WELL-A"),
        _ok_summary_row("WELL-B"),
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    metric_labels = [widget.label for widget in at.metric]
    assert "Проверено пар" not in metric_labels
    assert "Минимальный SF" not in metric_labels
    button_labels = [str(widget.label) for widget in at.button]
    assert "Расчёт пересечений" in button_labels
    selectboxes_by_label = {str(widget.label): widget for widget in at.selectbox}
    selectbox_labels = list(selectboxes_by_label.keys())
    assert "Пресет неопределенности для anti-collision" not in selectbox_labels
    assert "Параллельный расчёт anti-collision" not in selectbox_labels
    caption_values = [str(widget.value) for widget in at.caption]
    assert any(
        "Multiprocessing для anti-collision отключён автоматически" in value
        for value in caption_values
    )


def test_welltrack_page_normalizes_invalid_anticollision_uncertainty_preset() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()[:2]
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        _ok_summary_row("WELL-A"),
        _ok_summary_row("WELL-B"),
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    at.session_state["wt_anticollision_uncertainty_preset"] = "invalid_preset"
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    assert (
        str(at.session_state["wt_anticollision_uncertainty_preset"])
        == DEFAULT_UNCERTAINTY_PRESET
    )


def test_welltrack_page_keeps_valid_nondefault_anticollision_uncertainty_preset() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()[:2]
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        _ok_summary_row("WELL-A"),
        _ok_summary_row("WELL-B"),
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=5.0),
    ]
    at.session_state["wt_anticollision_uncertainty_preset"] = (
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    assert (
        str(at.session_state["wt_anticollision_uncertainty_preset"])
        == UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )


def test_welltrack_page_renders_actual_fund_well_detail_viewer() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()[:2]
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_reference_actual_wells"] = list(_horizontal_reference_well())
    at.session_state["wt_show_actual_fund_analysis"] = True

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


def test_render_raw_records_table_skips_large_collapsed_table(monkeypatch) -> None:
    page = wt_import_module
    captured: dict[str, object] = {"dataframe": 0}
    page.st.session_state.pop("wt_show_raw_records_table", None)
    page.st.session_state.pop("wt_edit_targets_highlight_names", None)

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_expander(*args, **kwargs):
        captured["expanded"] = bool(kwargs.get("expanded"))
        return _DummyExpander()

    def _fail_raw_dataframe(_records):
        raise AssertionError("large raw point table must wait for explicit user toggle")

    large_record = WelltrackRecord(
        name="WELL-LARGE",
        points=tuple(
            WelltrackPoint(x=float(index), y=0.0, z=float(index), md=float(index))
            for index in range(page.WT_RAW_RECORDS_AUTO_RENDER_POINT_LIMIT + 1)
        ),
    )

    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        page.st,
        "caption",
        lambda message, *args, **kwargs: captured.setdefault("caption", str(message)),
    )
    monkeypatch.setattr(
        page.st,
        "dataframe",
        lambda *args, **kwargs: captured.__setitem__(
            "dataframe", int(captured["dataframe"]) + 1
        ),
    )
    monkeypatch.setattr(
        page.ptc_target_records,
        "raw_records_dataframe",
        _fail_raw_dataframe,
    )

    page._render_raw_records_table([large_record])

    assert captured["expanded"] is False
    assert captured["dataframe"] == 0
    assert "Таблица скрыта для ускорения страницы" in str(captured["caption"])


def test_render_raw_records_table_limits_large_highlighted_view_to_changed_wells(
    monkeypatch,
) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}
    page.st.session_state["wt_edit_targets_highlight_names"] = ["WELL-MH"]
    page.st.session_state["wt_edit_targets_highlight_points"] = {"WELL-MH": [1, 2]}
    page.st.session_state.pop("wt_show_raw_records_table", None)

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_expander(*args, **kwargs):
        captured["expanded"] = bool(kwargs.get("expanded"))
        return _DummyExpander()

    large_record = WelltrackRecord(
        name="WELL-LARGE",
        points=tuple(
            WelltrackPoint(x=float(index), y=0.0, z=float(index), md=float(index))
            for index in range(page.WT_RAW_RECORDS_AUTO_RENDER_POINT_LIMIT + 1)
        ),
    )
    multi_record = WelltrackRecord(
        name="WELL-MH",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=1.0),
            WelltrackPoint(x=200.0, y=0.0, z=2000.0, md=2.0),
            WelltrackPoint(x=300.0, y=0.0, z=2020.0, md=3.0),
            WelltrackPoint(x=400.0, y=0.0, z=2020.0, md=4.0),
        ),
    )

    def _fake_raw_dataframe(records):
        captured["record_names"] = [str(record.name) for record in records]
        return page.ptc_target_records.raw_records_dataframe.__wrapped__(records)  # type: ignore[attr-defined]

    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        page.st,
        "caption",
        lambda message, *args, **kwargs: captured.setdefault("caption", str(message)),
    )
    monkeypatch.setattr(page.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    original_raw_records_dataframe = page.ptc_target_records.raw_records_dataframe
    monkeypatch.setattr(
        page.ptc_target_records,
        "raw_records_dataframe",
        lambda records: (
            captured.setdefault("record_names", [str(record.name) for record in records]),
            original_raw_records_dataframe(records),
        )[1],
    )

    page._render_raw_records_table([large_record, multi_record])

    assert captured["expanded"] is True
    assert captured["record_names"] == ["WELL-MH"]
    assert "показаны только изменённые скважины" in str(captured["caption"]).lower()


def test_render_raw_records_table_resets_large_full_table_toggle_for_new_highlight(
    monkeypatch,
) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}
    page.st.session_state["wt_show_raw_records_table"] = True
    page.st.session_state["wt_show_raw_records_table_highlight_signature"] = (
        ("OLD-WELL",),
        (),
        "3d",
    )
    page.st.session_state["wt_edit_targets_highlight_names"] = ["WELL-MH"]
    page.st.session_state["wt_edit_targets_highlight_points"] = {"WELL-MH": [1, 2]}
    page.st.session_state["wt_edit_targets_last_source"] = "3d"

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    large_record = WelltrackRecord(
        name="WELL-LARGE",
        points=tuple(
            WelltrackPoint(x=float(index), y=0.0, z=float(index), md=float(index))
            for index in range(page.WT_RAW_RECORDS_AUTO_RENDER_POINT_LIMIT + 1)
        ),
    )
    multi_record = WelltrackRecord(
        name="WELL-MH",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=1.0),
            WelltrackPoint(x=200.0, y=0.0, z=2000.0, md=2.0),
            WelltrackPoint(x=300.0, y=0.0, z=2020.0, md=3.0),
            WelltrackPoint(x=400.0, y=0.0, z=2020.0, md=4.0),
        ),
    )

    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(
        page.st,
        "toggle",
        lambda *args, **kwargs: bool(
            page.st.session_state.get(kwargs.get("key", ""), False)
        ),
    )
    monkeypatch.setattr(
        page.st,
        "caption",
        lambda message, *args, **kwargs: captured.setdefault("caption", str(message)),
    )
    monkeypatch.setattr(page.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    original_raw_records_dataframe = page.ptc_target_records.raw_records_dataframe
    monkeypatch.setattr(
        page.ptc_target_records,
        "raw_records_dataframe",
        lambda records: (
            captured.setdefault("record_names", [str(record.name) for record in records]),
            original_raw_records_dataframe(records),
        )[1],
    )

    page._render_raw_records_table([large_record, multi_record])

    assert page.st.session_state["wt_show_raw_records_table"] is False
    assert captured["record_names"] == ["WELL-MH"]


def test_render_raw_records_table_applies_edits_for_filtered_large_highlighted_view(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_raw_records_edit_mode"] = True
    page.st.session_state["wt_raw_records_editor_nonce"] = 0
    page.st.session_state["wt_edit_targets_highlight_names"] = ["WELL-MH"]
    page.st.session_state["wt_edit_targets_highlight_points"] = {"WELL-MH": [1, 2]}
    records = [
        WelltrackRecord(
            name="WELL-LARGE",
            points=tuple(
                WelltrackPoint(x=float(index), y=0.0, z=float(index), md=float(index))
                for index in range(page.WT_RAW_RECORDS_AUTO_RENDER_POINT_LIMIT + 1)
            ),
        ),
        WelltrackRecord(
            name="WELL-MH",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=1.0),
                WelltrackPoint(x=200.0, y=0.0, z=2000.0, md=2.0),
                WelltrackPoint(x=300.0, y=0.0, z=2020.0, md=3.0),
                WelltrackPoint(x=400.0, y=0.0, z=2020.0, md=4.0),
            ),
        ),
    ]
    filtered_records = [records[1]]
    edited_df = page.ptc_target_records.raw_records_dataframe(filtered_records)
    edited_df.loc[2, "Z, м"] = 2050.0
    captured: dict[str, object] = {"rerun_called": False}

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def form_submit_button(self, label, **kwargs):
            return str(label) == "Сохранить изменения"

    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn(), _DummyColumn()),
    )
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: edited_df.copy())
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "rerun",
        lambda *args, **kwargs: captured.__setitem__("rerun_called", True),
    )
    monkeypatch.setattr(
        page.st.column_config,
        "TextColumn",
        lambda *args, **kwargs: {"type": "text"},
    )
    monkeypatch.setattr(
        page.st.column_config,
        "NumberColumn",
        lambda *args, **kwargs: {"type": "number"},
    )

    def _fake_apply_edit_targets_changes(changes, *, source="3d"):
        captured["changes"] = changes
        captured["source"] = source
        return ["WELL-MH"]

    monkeypatch.setattr(page, "_apply_edit_targets_changes", _fake_apply_edit_targets_changes)

    page._render_raw_records_table(records)

    assert captured["source"] == "raw_records_table"
    assert captured["changes"] == [
        {
            "name": "WELL-MH",
            "points": [
                {"index": 2, "position": [200.0, 0.0, 2050.0]},
            ],
        }
    ]
    assert page.st.session_state["wt_raw_records_edit_mode"] is False
    assert page.st.session_state["wt_raw_records_editor_nonce"] == 1
    assert captured["rerun_called"] is True


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


def test_render_raw_records_table_highlights_changed_regular_pilot_and_multi_points(
    monkeypatch,
) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}
    page.st.session_state["wt_edit_targets_highlight_names"] = [
        "WELL-A",
        "WELL-A_PL",
        "WELL-M",
    ]
    page.st.session_state["wt_edit_targets_highlight_points"] = {
        "WELL-A": [1, 2],
        "WELL-A_PL": [0, 2],
        "WELL-M": [0, 3],
    }

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_expander(*args, **kwargs):
        return _DummyExpander()

    def _fake_dataframe(frame, **kwargs):
        captured["frame"] = frame

    monkeypatch.setattr(page.st, "expander", _fake_expander)
    monkeypatch.setattr(page.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", _fake_dataframe)

    records = [
        _records()[0],
        WelltrackRecord(
            name="WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=200.0, z=1800.0, md=1.0),
                WelltrackPoint(x=300.0, y=500.0, z=2400.0, md=2.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-M",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=1.0),
                WelltrackPoint(x=200.0, y=0.0, z=2000.0, md=2.0),
                WelltrackPoint(x=300.0, y=0.0, z=2020.0, md=3.0),
                WelltrackPoint(x=400.0, y=0.0, z=2020.0, md=4.0),
            ),
        ),
    ]

    page._render_raw_records_table(records)

    styler = captured["frame"]
    styler._compute()
    highlighted_rows = {
        row_index
        for (row_index, _col_index), styles in styler.ctx.items()
        if ("background-color", "rgba(34, 197, 94, 0.14)") in styles
    }
    assert highlighted_rows == {1, 2, 3, 5, 6, 9}


def test_render_raw_records_table_uses_preprocess_highlight_message(
    monkeypatch,
) -> None:
    page = wt_import_module
    captured: dict[str, object] = {}
    page.st.session_state["wt_edit_targets_highlight_names"] = ["WELL-A"]
    page.st.session_state["wt_edit_targets_highlight_points"] = {"WELL-A": [2]}
    page.st.session_state["wt_edit_targets_last_source"] = (
        "bulk_horizontal_length_preprocess"
    )

    class _DummyExpander:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyExpander())
    monkeypatch.setattr(
        page.st,
        "success",
        lambda message, *args, **kwargs: captured.setdefault("success", str(message)),
    )
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)

    page._render_raw_records_table(_records()[:1])

    assert captured["success"] == (
        "Скорректированные точки `t3` подсвечены. "
        "Запустите расчёт для обновления траекторий."
    )


def test_render_raw_records_table_keeps_coordinate_edits_pending_until_save(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_raw_records_edit_mode"] = True
    page.st.session_state["wt_raw_records_editor_nonce"] = 0
    records = list(_records()[:1])
    edited_df = page.ptc_target_records.raw_records_dataframe(records)
    edited_df.loc[0, "X, м"] = 10.0
    captured: dict[str, object] = {}

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def form_submit_button(self, label, **kwargs):
            return False

    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn(), _DummyColumn()),
    )

    def _fake_data_editor(frame, **kwargs):
        captured["disabled"] = kwargs.get("disabled")
        captured["num_rows"] = kwargs.get("num_rows")
        return edited_df.copy()

    monkeypatch.setattr(page.st, "data_editor", _fake_data_editor)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st.column_config,
        "TextColumn",
        lambda *args, **kwargs: {"type": "text"},
    )
    monkeypatch.setattr(
        page.st.column_config,
        "NumberColumn",
        lambda *args, **kwargs: {"type": "number"},
    )
    monkeypatch.setattr(
        page,
        "_apply_edit_targets_changes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("changes must not apply before save")
        ),
    )

    page._render_raw_records_table(records)

    assert list(captured["disabled"]) == ["Скважина", "Точка"]
    assert captured["num_rows"] == "fixed"
    assert page.st.session_state["wt_raw_records_edit_mode"] is True
    assert page.st.session_state["wt_raw_records_editor_nonce"] == 0


def test_render_raw_records_table_applies_coordinate_edits_on_first_submit(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_raw_records_edit_mode"] = True
    page.st.session_state["wt_raw_records_editor_nonce"] = 0
    records = list(_records()[:1])
    edited_df = page.ptc_target_records.raw_records_dataframe(records)
    edited_df.loc[0, "X, м"] = 10.0
    edited_df.loc[2, "Z, м"] = 2510.0
    captured: dict[str, object] = {"rerun_called": False}

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def form_submit_button(self, label, **kwargs):
            return str(label) == "Сохранить изменения"

    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "form", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn(), _DummyColumn()),
    )
    monkeypatch.setattr(page.st, "data_editor", lambda frame, **kwargs: edited_df.copy())
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "rerun",
        lambda *args, **kwargs: captured.__setitem__("rerun_called", True),
    )
    monkeypatch.setattr(
        page.st.column_config,
        "TextColumn",
        lambda *args, **kwargs: {"type": "text"},
    )
    monkeypatch.setattr(
        page.st.column_config,
        "NumberColumn",
        lambda *args, **kwargs: {"type": "number"},
    )

    def _fake_apply_edit_targets_changes(changes, *, source="3d"):
        captured["changes"] = changes
        captured["source"] = source
        return ["WELL-A"]

    monkeypatch.setattr(page, "_apply_edit_targets_changes", _fake_apply_edit_targets_changes)

    page._render_raw_records_table(records)

    assert captured["source"] == "raw_records_table"
    assert captured["changes"] == [
        {
            "name": "WELL-A",
            "points": [
                {"index": 0, "position": [10.0, 0.0, 0.0]},
                {"index": 2, "position": [1500.0, 2000.0, 2510.0]},
            ],
        }
    ]
    assert page.st.session_state["wt_raw_records_edit_mode"] is False
    assert page.st.session_state["wt_raw_records_editor_nonce"] == 1
    assert captured["rerun_called"] is True


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
    assert page.st.session_state["wt_edit_targets_highlight_points"] == {
        "WELL-A": [1, 2]
    }
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
    assert page.st.session_state["wt_edit_targets_highlight_points"] == {}
    assert page.st.session_state["wt_anticollision_analysis_cache"] is cached_analysis


def test_multihorizontal_streamlit_flow_recalculates_after_target_edit_with_incremental_ac_cache() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_source_mode"] = "Файл по пути"
    at.session_state["wt_source_path"] = (
        "tests/test_data/WELLTRACKS4_MULTIHORIZONTAL.INC"
    )

    at.run(timeout=120)
    _click_button(at, "Импорт целей")
    at.run(timeout=120)

    assert not at.exception
    assert "well_08" in {str(record.name) for record in at.session_state["wt_records"]}

    selected_names = ["well_08", "well_09", "well_12"]
    at.session_state["wt_cfg_dls_build_max"] = 1.0
    at.session_state["wt_cfg_dls_horizontal_max"] = 1.0
    at.session_state["wt_selected_names"] = selected_names
    _click_button(at, "Рассчитать траектории")
    at.run(timeout=240)

    assert not at.exception
    assert {str(item.name) for item in at.session_state["wt_successes"]} == set(
        selected_names
    )
    assert at.session_state["wt_anticollision_last_run"] is None
    assert at.session_state["wt_anticollision_analysis_cache"] == {}

    _click_button(at, "Расчёт пересечений")
    at.run(timeout=240)

    assert not at.exception
    first_ac_run = at.session_state["wt_anticollision_last_run"]
    assert bool(first_ac_run["cached"]) is False
    assert int(first_ac_run["pair_count"]) == 3
    assert int(first_ac_run["recalculated_pair_count"]) == 3
    cached_analysis = at.session_state["wt_anticollision_analysis_cache"]
    assert cached_analysis["pair_cache"]

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        _AppTestSessionStateAdapter(at.session_state),
        [
            {
                "name": "well_08",
                "points": [
                    {
                        "index": 1,
                        "position": [456018.0, 889281.0, 2339.0],
                    }
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=WelltrackBatchPlanner._base_row,
    )
    assert updated_names == ["well_08"]

    at.run(timeout=120)

    assert not at.exception
    assert at.session_state["wt_edit_targets_pending_names"] == ["well_08"]
    assert {str(item.name) for item in at.session_state["wt_successes"]} == {
        "well_09",
        "well_12",
    }
    edited_status = {
        str(row["Скважина"]): str(row["Статус"])
        for row in at.session_state["wt_summary_rows"]
        if str(row["Скважина"]) in set(selected_names)
    }
    assert edited_status == {
        "well_08": "Не рассчитана",
        "well_09": "OK",
        "well_12": "OK",
    }
    assert at.session_state["wt_anticollision_analysis_cache"] is cached_analysis
    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    assert "Пресет неопределенности для anti-collision" not in selectbox_labels

    at.session_state["wt_selected_names"] = ["well_08"]
    _click_button(at, "Рассчитать траектории")
    at.run(timeout=240)

    assert not at.exception
    assert at.session_state["wt_edit_targets_pending_names"] == []
    assert {str(item.name) for item in at.session_state["wt_successes"]} == set(
        selected_names
    )
    assert at.session_state["wt_anticollision_analysis_cache"] is cached_analysis
    assert at.session_state["wt_anticollision_last_run"] == first_ac_run

    _click_button(at, "Расчёт пересечений")
    at.run(timeout=240)

    assert not at.exception
    incremental_run = at.session_state["wt_anticollision_last_run"]
    assert bool(incremental_run["cached"]) is False
    assert int(incremental_run["reused_well_count"]) == 2
    assert int(incremental_run["rebuilt_well_count"]) == 1
    assert int(incremental_run["reused_pair_count"]) == 1
    assert int(incremental_run["recalculated_pair_count"]) == 2
    assert "Инкрементальный anti-collision" in "\n".join(
        incremental_run["log_lines"]
    )


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


def test_selected_override_configs_apply_manual_per_well_values() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    base_config = TrajectoryConfig()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "WELL-A cfg",
            "values": {
                "dls_build_max": 0.6,
                "entry_inc_target": 84.5,
            },
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-A": "cfg-1"
    }

    config_map = page._build_selected_override_configs(
        base_config=base_config,
        selected_names={"WELL-A", "WELL-B"},
        records_by_name={record.name: record for record in _records()},
    )

    assert set(config_map) == {"WELL-A"}
    assert config_map["WELL-A"].entry_inc_target_deg == pytest.approx(84.5)
    assert config_map["WELL-A"].dls_build_max_deg_per_30m == pytest.approx(1.8)


def test_selected_override_configs_keep_manual_kop_over_global_depth_function() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)
    set_kop_min_vertical_function(
        prefix=page.WT_CALC_PARAMS.prefix,
        kop_function=ActualFundKopDepthFunction(
            mode="piecewise_linear",
            cluster_count=2,
            anchor_depths_tvd_m=(2000.0, 2600.0),
            anchor_kop_md_m=(900.0, 1400.0),
            note="test",
        ),
    )
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "WELL-A cfg",
            "values": {"kop_min_vertical": 1111.0},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-A": "cfg-1"
    }

    config_map = page._build_selected_override_configs(
        base_config=TrajectoryConfig(),
        selected_names={"WELL-A"},
        records_by_name={record.name: record for record in _records()},
    )

    assert config_map["WELL-A"].kop_min_vertical_m == pytest.approx(1111.0)
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)


def test_apply_dev_params_to_manual_well_overrides_sets_local_values() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_imported_dev_params"] = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="J-профиль",
            kop_md_m=980.0,
            t1_md_m=2500.0,
            t3_md_m=4200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            horizontal_dls_deg_per_30m=(1.5,),
        ),
    )

    applied_count, missing_names = page._apply_dev_params_to_manual_well_overrides(
        well_names=["WELL-A"],
    )

    assert applied_count == 1
    assert missing_names == []
    profile_id = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY][
        "WELL-A"
    ]
    payload = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY][profile_id]
    assert payload["name"] == "WELL-A"
    assert payload["source"] == "Импорт .dev"
    assert payload["values"]["kop_min_vertical"] == pytest.approx(980.0)
    assert payload["values"]["use_fixed_kop"] is True
    assert payload["values"]["entry_inc_target"] == pytest.approx(87.0)
    assert payload["values"]["dls_build_max"] == pytest.approx(0.8)
    assert "dls_build2_enabled" not in payload["values"]
    assert "dls_build2_max" not in payload["values"]
    assert payload["values"]["dls_horizontal_max"] == pytest.approx(0.5)
    assert payload["values"]["j_profile_policy"] == "prefer"
    assert payload["values"]["offer_j_profile"] is True
    assert (
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY]
        == profile_id
    )


def test_apply_dev_params_to_manual_well_overrides_clamps_entry_inc_below_90() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_imported_dev_params"] = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="J-профиль",
            kop_md_m=980.0,
            t1_md_m=2500.0,
            t3_md_m=4200.0,
            entry_inc_deg=90.0,
            build1_dls_deg_per_30m=(2.4,),
            horizontal_dls_deg_per_30m=(1.5,),
        ),
    )

    applied_count, missing_names = page._apply_dev_params_to_manual_well_overrides(
        well_names=["WELL-A"],
    )

    assert applied_count == 1
    assert missing_names == []
    profile_id = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY][
        "WELL-A"
    ]
    payload = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY][profile_id]
    assert payload["values"]["entry_inc_target"] == pytest.approx(89.0)


def test_apply_dev_params_to_manual_well_overrides_does_not_toggle_widget_state() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_imported_dev_params"] = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="J-профиль",
            kop_md_m=980.0,
            t1_md_m=2500.0,
            t3_md_m=4200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            horizontal_dls_deg_per_30m=(1.5,),
        ),
    )

    page._apply_dev_params_to_manual_well_overrides(well_names=["WELL-A"])

    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] is False


def test_apply_dev_params_to_manual_well_overrides_sets_optional_build2_when_needed() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_imported_dev_params"] = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="BUILD-HOLD-BUILD",
            kop_md_m=980.0,
            t1_md_m=2500.0,
            t3_md_m=4200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            build2_dls_deg_per_30m=(5.4,),
            horizontal_dls_deg_per_30m=(1.5,),
        ),
    )

    applied_count, missing_names = page._apply_dev_params_to_manual_well_overrides(
        well_names=["WELL-A"],
    )

    assert applied_count == 1
    assert missing_names == []
    profile_id = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY][
        "WELL-A"
    ]
    payload = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY][profile_id]
    assert payload["values"]["use_fixed_kop"] is True
    assert payload["values"]["dls_build_max"] == pytest.approx(0.8)
    assert payload["values"]["dls_build2_enabled"] is True
    assert payload["values"]["dls_build2_max"] == pytest.approx(1.8)


def test_apply_dev_params_to_manual_well_overrides_reports_selected_wells_without_dev(
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_imported_dev_params"] = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="J-профиль",
            kop_md_m=980.0,
            t1_md_m=2500.0,
            t3_md_m=4200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            horizontal_dls_deg_per_30m=(1.5,),
        ),
    )

    applied_count, missing_names = page._apply_dev_params_to_manual_well_overrides(
        well_names=["WELL-B", "WELL-A"],
    )

    assert applied_count == 1
    assert missing_names == ["WELL-B"]
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        "WELL-A": page._manual_well_calc_profile_id_from_dev_summary("WELL-A")
    }


def test_apply_dev_params_to_manual_well_overrides_ignores_simple_target_dev_rows() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_imported_dev_params"] = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="3 точки S / t1 / t3",
            kop_md_m=0.0,
            t1_md_m=2400.0,
            t3_md_m=3500.0,
            entry_inc_deg=float("nan"),
            note="Импортировано как обычные цели из .dev.",
            simple_target_only=True,
        ),
    )

    applied_count, missing_names = page._apply_dev_params_to_manual_well_overrides(
        well_names=["WELL-A"],
    )

    assert applied_count == 0
    assert missing_names == ["WELL-A"]
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {}
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] == {}


def test_apply_dev_params_to_manual_well_overrides_keeps_other_assignments_untouched(
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-b": {
            "name": "Cfg B",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-B": "cfg-b"
    }
    page.st.session_state["wt_imported_dev_params"] = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="J-профиль",
            kop_md_m=980.0,
            t1_md_m=2500.0,
            t3_md_m=4200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            horizontal_dls_deg_per_30m=(1.5,),
        ),
    )

    applied_count, missing_names = page._apply_dev_params_to_manual_well_overrides(
        well_names=["WELL-A"],
    )

    assert applied_count == 1
    assert missing_names == []
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        "WELL-B": "cfg-b",
        "WELL-A": page._manual_well_calc_profile_id_from_dev_summary("WELL-A"),
    }


def test_auto_apply_imported_dev_param_overrides_enables_configs_and_skips_simple_targets() -> (
    None
):
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_selected_names"] = ["WELL-C"]
    summaries = (
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-A",
            profile_label="J-профиль",
            kop_md_m=980.0,
            t1_md_m=2500.0,
            t3_md_m=4200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            horizontal_dls_deg_per_30m=(1.5,),
        ),
        page.ptc_target_import.DevTargetImportSummary(
            well_name="WELL-B",
            profile_label="3 точки S / t1 / t3",
            kop_md_m=0.0,
            t1_md_m=2400.0,
            t3_md_m=3500.0,
            entry_inc_deg=float("nan"),
            note="Импортировано как обычные цели из .dev.",
            simple_target_only=True,
        ),
    )
    page.st.session_state["wt_imported_dev_params"] = summaries

    applied_count = page._auto_apply_imported_dev_param_overrides(
        records=_records(),
        dev_summaries=summaries,
    )

    assert applied_count == 1
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] is True
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        "WELL-A": page._manual_well_calc_profile_id_from_dev_summary("WELL-A")
    }
    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-C", "WELL-A"]
    assert (
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY]
        == "После импорта автоматически созданы или обновлены конфигурации из .dev: 1."
    )


def test_manual_well_calc_profile_export_json_uses_profile_name() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()

    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Cfg / 1",
        values={"dls_build_max": 0.6},
        source="Ручная настройка",
        note="test note",
    )

    payload = json.loads(page._manual_well_calc_profile_export_json("cfg-1"))

    assert payload["kind"] == page.WT_WELL_CALC_PROFILE_JSON_KIND
    assert payload["name"] == "Cfg / 1"
    assert payload["values"]["dls_build_max"] == pytest.approx(0.6)
    assert page._manual_well_calc_profile_export_file_name("cfg-1") == "Cfg _ 1.json"


def test_handle_manual_well_calc_profile_name_change_updates_profile_name() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True

    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values={},
        source="Ручная настройка",
    )
    page.st.session_state[page._manual_well_calc_profile_name_key("cfg-1")] = (
        "Renamed Config"
    )

    page._handle_manual_well_calc_profile_name_change("cfg-1")

    payload = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]["cfg-1"]
    assert payload["name"] == "Renamed Config"
    assert "Renamed Config" in page.st.session_state[
        page.WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY
    ]


def test_import_manual_well_calc_profile_json_bytes_creates_profile_by_name() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()

    profile_id, resolved_name, updated_existing = (
        page._import_manual_well_calc_profile_json_bytes(
            json.dumps(
                {
                    "kind": page.WT_WELL_CALC_PROFILE_JSON_KIND,
                    "schema_version": page.WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION,
                    "name": "Imported Config",
                    "values": {"dls_build_max": 0.7},
                    "source": "Импорт JSON",
                    "note": "from file",
                },
                ensure_ascii=False,
            ).encode("utf-8")
        )
    )

    assert updated_existing is False
    assert resolved_name == "Imported Config"
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY][profile_id][
        "values"
    ]["dls_build_max"] == pytest.approx(0.7)
    assert (
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY]
        == profile_id
    )


def test_import_manual_well_calc_profile_json_bytes_updates_existing_profile() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Imported Config",
        values={"dls_build_max": 0.6},
        source="Ручная настройка",
    )

    profile_id, resolved_name, updated_existing = (
        page._import_manual_well_calc_profile_json_bytes(
            json.dumps(
                {
                    "kind": page.WT_WELL_CALC_PROFILE_JSON_KIND,
                    "schema_version": page.WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION,
                    "name": "Imported Config",
                    "values": {"dls_build_max": 0.9},
                    "source": "Импорт JSON",
                },
                ensure_ascii=False,
            ).encode("utf-8")
        )
    )

    assert updated_existing is True
    assert profile_id == "cfg-1"
    assert resolved_name == "Imported Config"
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]["cfg-1"][
        "values"
    ]["dls_build_max"] == pytest.approx(0.9)


def test_import_manual_well_calc_profile_json_payloads_imports_multiple_files() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()

    imported_count, updated_count, error_messages, last_profile_id, touched_profile_ids = (
        page._import_manual_well_calc_profile_json_payloads(
            [
                (
                    "cfg_a.json",
                    json.dumps(
                        {
                            "kind": page.WT_WELL_CALC_PROFILE_JSON_KIND,
                            "schema_version": page.WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION,
                            "name": "Cfg A",
                            "values": {"dls_build_max": 0.7},
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                ),
                (
                    "cfg_b.json",
                    json.dumps(
                        {
                            "kind": page.WT_WELL_CALC_PROFILE_JSON_KIND,
                            "schema_version": page.WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION,
                            "name": "Cfg B",
                            "values": {"dls_build_max": 0.9},
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                ),
            ]
        )
    )

    assert imported_count == 2
    assert updated_count == 0
    assert error_messages == []
    assert last_profile_id
    assert set(touched_profile_ids) == set(
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY].keys()
    )
    profiles = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]
    assert {payload["name"] for payload in profiles.values()} == {"Cfg A", "Cfg B"}
    assert (
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY]
        == last_profile_id
    )


def test_import_manual_well_calc_profile_json_payloads_collects_errors() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()

    imported_count, updated_count, error_messages, last_profile_id, touched_profile_ids = (
        page._import_manual_well_calc_profile_json_payloads(
            [
                ("bad.json", b"{not-json}"),
                (
                    "good.json",
                    json.dumps(
                        {
                            "kind": page.WT_WELL_CALC_PROFILE_JSON_KIND,
                            "schema_version": page.WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION,
                            "name": "Cfg Good",
                            "values": {"dls_build_max": 0.8},
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                ),
            ]
        )
    )

    assert imported_count == 1
    assert updated_count == 0
    assert len(error_messages) == 1
    assert "bad.json" in error_messages[0]
    assert last_profile_id
    assert touched_profile_ids == (last_profile_id,)


def test_consume_manual_well_override_enabled_applies_pending_state() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()

    page._queue_manual_well_calc_override_enabled(True)
    page._consume_manual_well_calc_override_enabled()

    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] is True


def test_consume_manual_well_calc_active_profile_applies_pending_state() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }

    page._queue_manual_well_calc_active_profile("cfg-1")
    page._consume_manual_well_calc_active_profile()

    assert (
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY]
        == "cfg-1"
    )
    assert (
        page.st.session_state.get(
            page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY
        )
        is None
    )


def test_sync_manual_well_override_editor_selection_tolerates_none_signature() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = None
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }

    page._sync_manual_well_override_editor_selection(
        base_config=TrajectoryConfig(),
        active_profile_id="cfg-1",
    )

    signature = page.st.session_state[
        page.WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY
    ]
    assert signature[0] == "profile"
    assert signature[1] == "cfg-1"


def test_manual_well_calc_profile_assignments_persists_normalized_raw_values() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        " WELL-A ": " cfg-1 "
    }

    assignments = page._manual_well_calc_profile_assignments(
        available_names=["WELL-A"]
    )

    assert assignments == {"WELL-A": "cfg-1"}
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        "WELL-A": "cfg-1"
    }


def test_manual_well_calc_profiles_migrate_legacy_assignments_by_well_name() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "legacy-profile-1": {
            "well_name": "WELL-A",
            "values": {"dls_build_max": 0.6},
            "source": "legacy",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {}

    assignments = page._manual_well_calc_profile_assignments(
        available_names=["WELL-A"]
    )

    assert assignments == {"WELL-A": "legacy-profile-1"}
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        "WELL-A": "legacy-profile-1"
    }


def test_manual_well_calc_profiles_merge_legacy_assignments_into_stale_mapping() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "legacy-profile-1": {
            "well_name": "WELL-A",
            "values": {"dls_build_max": 0.6},
            "source": "legacy",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-Z": "missing-profile"
    }

    assignments = page._manual_well_calc_profile_assignments(
        available_names=["WELL-A"]
    )

    assert assignments == {"WELL-A": "legacy-profile-1"}
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        "WELL-A": "legacy-profile-1"
    }


def test_manual_well_calc_profiles_does_not_rewrite_when_raw_mapping_normalizes_equal() -> None:
    class _RawValues(Mapping[str, object]):
        def __init__(self, payload: Mapping[str, object]) -> None:
            self._payload = dict(payload)

        def __getitem__(self, key: str) -> object:
            return self._payload[key]

        def __iter__(self):
            return iter(self._payload)

        def __len__(self) -> int:
            return len(self._payload)

        def __eq__(self, _other: object) -> bool:
            return False

    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    raw_state = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": _RawValues({"dls_build_max": 0.6}),
            "source": "manual",
            "note": "saved",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = raw_state

    profiles = page._manual_well_calc_profiles()

    assert profiles["cfg-1"]["values"] == {"dls_build_max": pytest.approx(0.6)}
    assert (
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] is raw_state
    )


def test_manual_well_calc_override_signature_distinguishes_enabled_without_values() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()

    assert page._manual_well_calc_override_signature() == (False, ())

    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True

    assert page._manual_well_calc_override_signature() == (True, ())


def test_preserve_manual_well_calc_override_widget_state_keeps_override_keys() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = "cfg-1"
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_KEY] = ["WELL-A"]
    name_key = page._manual_well_calc_profile_name_key("cfg-1")
    page.st.session_state[name_key] = "Cfg 1"

    page._preserve_manual_well_calc_override_widget_state()

    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] is True
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] == "cfg-1"
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_KEY] == ["WELL-A"]
    assert page.st.session_state[name_key] == "Cfg 1"


def test_manual_well_names_for_profile_ids_returns_wells_in_available_order() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {"name": "Cfg 1", "values": {}, "source": "manual"},
        "cfg-2": {"name": "Cfg 2", "values": {}, "source": "manual"},
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-B": "cfg-2",
        "WELL-A": "cfg-1",
        "WELL-C": "cfg-1",
    }

    assigned_names = page._manual_well_names_for_profile_ids(
        profile_ids=["cfg-1"],
        available_names=["WELL-C", "WELL-A", "WELL-B"],
    )

    assert assigned_names == ["WELL-C", "WELL-A"]


def test_queue_batch_selection_additions_merges_with_current_selection() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_selected_names"] = ["WELL-A"]

    queued_names = page._queue_batch_selection_additions(
        well_names=["WELL-C", "WELL-B", "WELL-A"],
        available_names=["WELL-A", "WELL-B", "WELL-C"],
    )

    assert queued_names == ["WELL-A", "WELL-C", "WELL-B"]
    assert page.st.session_state["wt_pending_selected_names"] == [
        "WELL-A",
        "WELL-C",
        "WELL-B",
    ]
    assert page.st.session_state["wt_selected_names"] == ["WELL-A"]


def test_render_manual_well_calc_overrides_preserves_widget_state_before_render(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    calls: list[str] = []

    monkeypatch.setattr(
        page,
        "_preserve_manual_well_calc_override_widget_state",
        lambda: calls.append("preserve"),
    )
    monkeypatch.setattr(page, "_unique_well_names", lambda names: [])

    page._render_manual_well_calc_overrides(records=[])

    assert calls == ["preserve"]


def test_sync_manual_well_override_editor_selection_evaluates_kop_for_assigned_profile(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)
    set_kop_min_vertical_function(
        prefix=page.WT_CALC_PARAMS.prefix,
        kop_function=ActualFundKopDepthFunction(
            mode="piecewise_linear",
            cluster_count=3,
            anchor_depths_tvd_m=(1600.0, 2500.0, 3400.0),
            anchor_kop_md_m=(780.0, 1180.0, 1680.0),
            note="test",
        ),
    )
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-A": "cfg-1"
    }
    captured: dict[str, TrajectoryConfig] = {}

    monkeypatch.setattr(
        page,
        "_set_manual_well_override_editor_from_config",
        lambda config: captured.setdefault("config", config),
    )

    page._sync_manual_well_override_editor_selection(
        base_config=TrajectoryConfig(),
        active_profile_id="cfg-1",
        records_by_name={record.name: record for record in _records()},
    )

    assert float(captured["config"].kop_min_vertical_m) == pytest.approx(
        1135.5555555,
        rel=1e-6,
    )
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)


def test_effective_manual_well_profile_values_tolerates_missing_records_lookup() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)
    set_kop_min_vertical_function(
        prefix=page.WT_CALC_PARAMS.prefix,
        kop_function=ActualFundKopDepthFunction(
            mode="piecewise_linear",
            cluster_count=3,
            anchor_depths_tvd_m=(1600.0, 2500.0, 3400.0),
            anchor_kop_md_m=(780.0, 1180.0, 1680.0),
            note="test",
        ),
    )
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-A": "cfg-1"
    }

    values = page._effective_manual_well_profile_values(
        base_config=TrajectoryConfig(),
        profile_id="cfg-1",
        records_by_name=None,
    )

    assert values["dls_build_max"] == pytest.approx(0.6)
    clear_kop_min_vertical_function(prefix=page.WT_CALC_PARAMS.prefix)


def test_effective_manual_well_profile_values_uses_next_well_when_first_kop_is_none(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-A": "cfg-1",
        "WELL-B": "cfg-1",
    }
    calls: list[str] = []

    monkeypatch.setattr(
        page,
        "kop_min_vertical_function_from_state",
        lambda prefix: object(),
    )

    def _fake_eval(*, record, base_config, kop_function):
        calls.append(str(record.name))
        return None if str(record.name) == "WELL-A" else 987.0

    monkeypatch.setattr(
        page,
        "_evaluated_kop_min_vertical_for_record",
        _fake_eval,
    )

    values = page._effective_manual_well_profile_values(
        base_config=TrajectoryConfig(),
        profile_id="cfg-1",
        records_by_name={
            "WELL-A": SimpleNamespace(name="WELL-A"),
            "WELL-B": SimpleNamespace(name="WELL-B"),
        },
    )

    assert calls == ["WELL-A", "WELL-B"]
    assert values["kop_min_vertical"] == pytest.approx(987.0)


def test_render_manual_well_calc_overrides_disables_editor_when_toggle_is_off(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = False
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-A": "cfg-1"
    }
    captured: dict[str, object] = {
        "buttons": {},
        "captions": [],
        "download_buttons": {},
        "file_uploader_disabled": None,
    }

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            captured["buttons"][str(label)] = bool(kwargs.get("disabled"))
            return False

        def download_button(self, label, *args, **kwargs):
            captured["download_buttons"][str(label)] = bool(kwargs.get("disabled"))
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_multiselect(_label, options, key, **kwargs):
        captured["multiselect_disabled"] = bool(kwargs.get("disabled"))
        page.st.session_state.setdefault(key, [])
        return []

    def _fake_selectbox(_label, options, key, **kwargs):
        captured["selectbox_disabled"] = bool(kwargs.get("disabled"))
        page.st.session_state.setdefault(key, options[0] if options else "")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        captured["text_input_disabled"] = bool(kwargs.get("disabled"))
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_button(label, **kwargs):
        captured["buttons"][str(label)] = bool(kwargs.get("disabled"))
        return False

    def _fake_file_uploader(_label, *args, **kwargs):
        captured["file_uploader_disabled"] = bool(kwargs.get("disabled"))
        captured["file_uploader_accept_multiple"] = bool(
            kwargs.get("accept_multiple_files")
        )
        return None

    def _fake_build_config_form(*args, **kwargs):
        captured["editor_disabled"] = bool(kwargs.get("disabled"))
        return TrajectoryConfig()

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "button", _fake_button)
    monkeypatch.setattr(page.st, "file_uploader", _fake_file_uploader)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page.st,
        "caption",
        lambda message, *args, **kwargs: captured["captions"].append(str(message)),
    )
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page, "_build_config_form", _fake_build_config_form)

    page._render_manual_well_calc_overrides(records=_records())

    assert captured["multiselect_disabled"] is True
    assert captured["selectbox_disabled"] is True
    assert captured["text_input_disabled"] is True
    assert captured["file_uploader_disabled"] is True
    assert captured["file_uploader_accept_multiple"] is True
    assert captured["editor_disabled"] is True
    assert captured["buttons"] == {
        "Новая": True,
        "Удалить": True,
        "Импорт": True,
        "Выбрать все": True,
        "Назначить выбранным": True,
        "Снять назначение": True,
        "Вернуть дефолт": True,
        "Подтянуть из .dev": True,
        "Сохранить конфигурацию": True,
        "Применить для всех и сохранить": True,
    }
    assert captured["download_buttons"] == {"Экспорт": True}
    assert any(
        "Индивидуальные конфигурации сохранены, но сейчас отключены" in item
        for item in captured["captions"]
    )


def test_render_manual_well_calc_overrides_cleans_assignments_when_records_empty() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY] = {
        "cfg-1": {
            "name": "Cfg 1",
            "values": {"dls_build_max": 0.6},
            "source": "manual",
        }
    }
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-A": "cfg-1",
        "WELL-B": "cfg-1",
    }

    page._render_manual_well_calc_overrides(records=[])

    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {}


def test_render_manual_well_calc_overrides_keeps_active_profile_pending_after_dev_apply(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_selected_names"] = ["WELL-C"]
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = "missing"
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_KEY] = ["WELL-A"]
    page.st.session_state["wt_imported_dev_params"] = (
        SimpleNamespace(
            well_name="WELL-A",
            profile_label="BuildHoldBuild",
            kop_md_m=1200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            build2_dls_deg_per_30m=(),
            horizontal_dls_deg_per_30m=(),
        ),
    )

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return str(label) == "Выбрать все"

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, options[0] if options else "")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, ["WELL-A"])
        return page.st.session_state[key]

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(
        page.st,
        "button",
        lambda label, **kwargs: str(label) == "Подтянуть из .dev",
    )
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    assignments = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY]
    profile_id = assignments["WELL-A"]
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] == ""
    assert (
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY]
        == profile_id
    )
    assert (
        page._manual_well_calc_profile_name_key(profile_id)
        not in page.st.session_state
    )
    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-C", "WELL-A"]


def test_render_manual_well_calc_overrides_enables_dev_apply_without_selection(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_KEY] = []
    page.st.session_state["wt_imported_dev_params"] = (
        SimpleNamespace(
            well_name="WELL-A",
            profile_label="BuildHoldBuild",
            kop_md_m=1200.0,
            entry_inc_deg=87.0,
            build1_dls_deg_per_30m=(2.4,),
            build2_dls_deg_per_30m=(),
            horizontal_dls_deg_per_30m=(),
        ),
    )
    captured: dict[str, object] = {"buttons": {}}

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return str(label) == "Выбрать все"

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, options[0] if options else "")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, [])
        return page.st.session_state[key]

    def _fake_button(label, **kwargs):
        captured["buttons"][str(label)] = bool(kwargs.get("disabled"))
        return False

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", _fake_button)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    assert captured["buttons"]["Подтянуть из .dev"] is False


def test_render_manual_well_calc_overrides_assign_adds_wells_to_batch_selection(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_selected_names"] = ["WELL-A"]
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = "cfg-1"
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values={"dls_build_max": 0.7},
        source="Ручная настройка",
    )

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return str(label) == "Назначить выбранным"

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, "cfg-1")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(label, options, key, **kwargs):
        if str(label) == "Скважины для назначения":
            page.st.session_state[key] = ["WELL-B"]
            return page.st.session_state[key]
        page.st.session_state.setdefault(key, [])
        return page.st.session_state[key]

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] == {
        "WELL-B": "cfg-1"
    }
    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-A", "WELL-B"]


def test_render_manual_well_calc_overrides_save_adds_assigned_wells_to_batch_selection(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state["wt_selected_names"] = ["WELL-A"]
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = "cfg-1"
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values={},
        source="Ручная настройка",
    )
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {
        "WELL-B": "cfg-1"
    }
    base_config = page.WT_CALC_PARAMS.build_config()
    editor_values = page.calc_param_state_values_from_config(base_config)
    editor_values["entry_inc_target"] = float(editor_values["entry_inc_target"]) + 1.0

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return str(label) == "Сохранить конфигурацию"

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, "cfg-1")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, [])
        return page.st.session_state[key]

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page,
        "WT_WELL_OVERRIDE_EDITOR",
        SimpleNamespace(
            prefix=page.WT_WELL_OVERRIDE_EDITOR.prefix,
            build_config=lambda: page.build_config_from_values(editor_values),
        ),
    )
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    assert page.st.session_state["wt_pending_selected_names"] == ["WELL-A", "WELL-B"]
    payload = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]["cfg-1"]
    assert payload["values"]["entry_inc_target"] == pytest.approx(
        editor_values["entry_inc_target"]
    )


def test_render_manual_well_calc_overrides_load_default_persists_until_save(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = "cfg-1"
    base_config = page.WT_CALC_PARAMS.build_config()
    base_values = page.calc_param_state_values_from_config(base_config)
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values={"entry_inc_target": float(base_values["entry_inc_target"]) + 1.0},
        source="Ручная настройка",
    )
    stage = {"value": "reset"}

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            if stage["value"] == "reset":
                return str(label) == "Вернуть дефолт"
            if stage["value"] == "save":
                return str(label) == "Сохранить конфигурацию"
            return False

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, "cfg-1")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, [])
        return page.st.session_state[key]

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    stage["value"] = "idle"
    page._render_manual_well_calc_overrides(records=_records())

    editor_config = page.WT_WELL_OVERRIDE_EDITOR.build_config()
    assert float(editor_config.entry_inc_target_deg) == pytest.approx(
        float(base_values["entry_inc_target"])
    )
    profile_values = page._effective_manual_well_profile_values(
        base_config=base_config,
        profile_id="cfg-1",
        records_by_name={str(record.name): record for record in _records()},
    )
    profile_signature = page._manual_well_override_selection_signature(
        active_profile_id="cfg-1",
        values=profile_values,
    )
    base_signature = page._manual_well_override_selection_signature(
        active_profile_id="cfg-1",
        values=base_values,
        source="base",
    )
    assert (
        tuple(
            page.st.session_state[
                page.WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY
            ]
        )
        == base_signature
    )
    assert (
        tuple(
            page.st.session_state[
                page.WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY
            ]
        )
        != profile_signature
    )

    stage["value"] = "save"
    page._render_manual_well_calc_overrides(records=_records())

    payload = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]["cfg-1"]
    assert payload["values"] == {}


def test_render_manual_well_calc_overrides_select_all_queues_full_assignment_selection(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_KEY] = ["WELL-A"]

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return str(label) == "Выбрать все"

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, options[0] if options else "")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, ["WELL-A"])
        return page.st.session_state[key]

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    assert page.st.session_state[
        page.WT_WELL_CALC_OVERRIDE_SELECTION_PENDING_KEY
    ] == ["WELL-A", "WELL-B", "WELL-C"]
    assert page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_KEY] == [
        "WELL-A"
    ]


def test_render_manual_well_calc_overrides_does_not_resync_selection_after_multiselect(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_SELECTION_KEY] = ["WELL-A"]
    rendered_multiselect = False
    sync_calls_after_multiselect = 0
    original_sync = page._sync_manual_well_override_selection

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return False

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, options[0] if options else "")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        nonlocal rendered_multiselect
        rendered_multiselect = True
        page.st.session_state.setdefault(key, ["WELL-A"])
        return page.st.session_state[key]

    def _guarded_sync(*args, **kwargs):
        nonlocal sync_calls_after_multiselect
        if rendered_multiselect:
            sync_calls_after_multiselect += 1
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(page, "_sync_manual_well_override_selection", _guarded_sync)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    assert rendered_multiselect is True
    assert sync_calls_after_multiselect == 0


def test_render_manual_well_calc_overrides_skips_warning_when_json_import_succeeds(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    warning_messages: list[str] = []

    class _DummyUpload:
        name = "cfg.json"

        def getvalue(self) -> bytes:
            return json.dumps(
                {
                    "kind": page.WT_WELL_CALC_PROFILE_JSON_KIND,
                    "schema_version": page.WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION,
                    "name": "Imported Config",
                    "values": {"dls_build_max": 0.7},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return str(label) == "Импорт"

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, options[0] if options else "")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, [])
        return []

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: [_DummyUpload()])
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", warning_messages.append)
    monkeypatch.setattr(page.st, "rerun", lambda: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    assert warning_messages == []
    profile_ids = tuple(
        page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY].keys()
    )
    assert len(profile_ids) == 1
    pending_profile_id = str(profile_ids[0]).strip()
    assert (
        page._manual_well_calc_profile_name_key(pending_profile_id)
        not in page.st.session_state
    )


def test_render_manual_well_calc_overrides_saves_renamed_profile(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values={},
        source="Ручная настройка",
    )
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = "cfg-1"
    page.st.session_state[page.WT_WELL_CALC_OVERRIDE_NAME_INPUT_ACTIVE_KEY] = "cfg-1"
    page.st.session_state[page._manual_well_calc_profile_name_key("cfg-1")] = (
        "Renamed Config"
    )

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def button(self, label, **kwargs):
            return str(label) == "Сохранить конфигурацию"

        def download_button(self, *args, **kwargs):
            return False

    def _fake_columns(spec, *args, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def _fake_toggle(_label, key, **kwargs):
        return page.st.session_state.get(key, False)

    def _fake_selectbox(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, "cfg-1")
        return page.st.session_state[key]

    def _fake_text_input(_label, value="", key=None, **kwargs):
        if key is not None and key not in page.st.session_state:
            page.st.session_state[key] = value
        return page.st.session_state.get(key, value)

    def _fake_multiselect(_label, options, key, **kwargs):
        page.st.session_state.setdefault(key, [])
        return page.st.session_state[key]

    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "toggle", _fake_toggle)
    monkeypatch.setattr(page.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "multiselect", _fake_multiselect)
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        page,
        "_build_config_form",
        lambda *args, **kwargs: TrajectoryConfig(),
    )

    page._render_manual_well_calc_overrides(records=_records())

    payload = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]["cfg-1"]
    assert payload["name"] == "Renamed Config"
    assert "Renamed Config" in page.st.session_state[
        page.WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY
    ]


def test_apply_manual_well_override_editor_to_all_profiles_updates_only_changed_fields(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    base_config = page.WT_CALC_PARAMS.build_config()
    base_values = page.calc_param_state_values_from_config(base_config)
    profile_a_values = {
        "dls_build_max": float(base_values["dls_build_max"]) + 0.1,
        "entry_inc_target": 84.0,
    }
    profile_b_values = {
        "dls_horizontal_max": float(base_values["dls_horizontal_max"]) + 0.2,
    }
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values=profile_a_values,
        source="Ручная настройка",
    )
    page._store_manual_well_calc_profile(
        profile_id="cfg-2",
        profile_name="Config B",
        values=profile_b_values,
        source="Ручная настройка",
    )
    effective_values = page._effective_manual_well_profile_values(
        base_config=base_config,
        profile_id="cfg-1",
    )
    editor_values = dict(effective_values)
    editor_values["entry_inc_target"] = float(effective_values["entry_inc_target"]) + 1.0
    monkeypatch.setattr(
        page,
        "WT_WELL_OVERRIDE_EDITOR",
        SimpleNamespace(
            build_config=lambda: page.build_config_from_values(editor_values),
        ),
    )

    changed, resolved_name, changed_field_count, updated_profile_count = (
        page._apply_manual_well_override_editor_to_all_profiles(
            base_config=base_config,
            active_profile_id="cfg-1",
            profile_name="Config A",
        )
    )

    assert changed is True
    assert resolved_name == "Config A"
    assert changed_field_count == 1
    assert updated_profile_count == 2

    profiles = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]
    assert profiles["cfg-1"]["values"]["dls_build_max"] == pytest.approx(
        profile_a_values["dls_build_max"]
    )
    assert profiles["cfg-1"]["values"]["entry_inc_target"] == pytest.approx(
        editor_values["entry_inc_target"]
    )
    assert profiles["cfg-2"]["values"]["dls_horizontal_max"] == pytest.approx(
        profile_b_values["dls_horizontal_max"]
    )
    assert profiles["cfg-2"]["values"]["entry_inc_target"] == pytest.approx(
        editor_values["entry_inc_target"]
    )
    assert "dls_build_max" not in profiles["cfg-2"]["values"]


def test_apply_manual_well_override_editor_to_all_profiles_counts_shared_build_limit_as_one_field(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    base_config = page.WT_CALC_PARAMS.build_config()
    base_values = page.calc_param_state_values_from_config(base_config)
    profile_a_values = {
        "dls_build_max": float(base_values["dls_build_max"]) + 0.2,
    }
    profile_b_values = {
        "dls_build_max": float(base_values["dls_build_max"]) + 0.1,
        "dls_build2_enabled": True,
        "dls_build2_max": float(base_values["dls_build_max"]) + 0.6,
    }
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values=profile_a_values,
        source="Ручная настройка",
    )
    page._store_manual_well_calc_profile(
        profile_id="cfg-2",
        profile_name="Config B",
        values=profile_b_values,
        source="Ручная настройка",
    )
    effective_values = page._effective_manual_well_profile_values(
        base_config=base_config,
        profile_id="cfg-1",
    )
    editor_values = dict(effective_values)
    editor_values["dls_build_max"] = float(effective_values["dls_build_max"]) + 0.3
    monkeypatch.setattr(
        page,
        "WT_WELL_OVERRIDE_EDITOR",
        SimpleNamespace(
            build_config=lambda: page.build_config_from_values(editor_values),
        ),
    )

    changed, resolved_name, changed_field_count, updated_profile_count = (
        page._apply_manual_well_override_editor_to_all_profiles(
            base_config=base_config,
            active_profile_id="cfg-1",
            profile_name="Config A",
        )
    )

    assert changed is True
    assert resolved_name == "Config A"
    assert changed_field_count == 1
    assert updated_profile_count == 2

    profiles = page.st.session_state[page.WT_WELL_CALC_OVERRIDE_STATE_KEY]
    assert profiles["cfg-1"]["values"]["dls_build_max"] == pytest.approx(
        editor_values["dls_build_max"]
    )
    assert profiles["cfg-2"]["values"]["dls_build_max"] == pytest.approx(
        editor_values["dls_build_max"]
    )
    assert profiles["cfg-2"]["values"]["dls_build2_enabled"] is True
    assert profiles["cfg-2"]["values"]["dls_build2_max"] == pytest.approx(
        profile_b_values["dls_build2_max"]
    )


def test_apply_manual_well_override_editor_to_all_profiles_counts_shared_build_limit_plus_extra_field(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._init_state()
    base_config = page.WT_CALC_PARAMS.build_config()
    base_values = page.calc_param_state_values_from_config(base_config)
    page._store_manual_well_calc_profile(
        profile_id="cfg-1",
        profile_name="Config A",
        values={"dls_build_max": float(base_values["dls_build_max"]) + 0.2},
        source="Ручная настройка",
    )
    page._store_manual_well_calc_profile(
        profile_id="cfg-2",
        profile_name="Config B",
        values={"entry_inc_target": 84.0},
        source="Ручная настройка",
    )
    effective_values = page._effective_manual_well_profile_values(
        base_config=base_config,
        profile_id="cfg-1",
    )
    editor_values = dict(effective_values)
    editor_values["dls_build_max"] = float(effective_values["dls_build_max"]) + 0.3
    editor_values["entry_inc_target"] = float(
        effective_values["entry_inc_target"]
    ) + 1.0
    monkeypatch.setattr(
        page,
        "WT_WELL_OVERRIDE_EDITOR",
        SimpleNamespace(
            build_config=lambda: page.build_config_from_values(editor_values),
        ),
    )

    changed, resolved_name, changed_field_count, updated_profile_count = (
        page._apply_manual_well_override_editor_to_all_profiles(
            base_config=base_config,
            active_profile_id="cfg-1",
            profile_name="Config A",
        )
    )

    assert changed is True
    assert resolved_name == "Config A"
    assert changed_field_count == 2
    assert updated_profile_count == 2


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


def test_render_three_payload_reuses_augmented_payload_for_same_inputs(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page._THREE_AUGMENTED_PAYLOAD_CACHE.clear()
    calls = {"augment": 0}

    class _DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    original_augment = page._augment_three_payload

    def _fake_three_scene(payload, **kwargs):
        return None

    def _counting_augment(*args, **kwargs):
        calls["augment"] += 1
        return original_augment(*args, **kwargs)

    monkeypatch.setattr(page, "render_local_three_scene", _fake_three_scene)
    monkeypatch.setattr(page, "_augment_three_payload", _counting_augment)
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_THREE_CAMERA,
        "lines": [],
        "points": [],
        "meshes": [],
        "labels": [],
        "legend": [],
    }
    overrides = {
        "legend_tree": [],
        "focus_targets": {},
        "hidden_flat_legend_labels": set(),
        "collisions": [],
        "edit_wells": [],
        "extra_labels": [],
        "extra_meshes": [],
        "extra_legend_items": [],
        "component_key": "test",
    }

    page._render_three_payload(
        container=_DummyContainer(),
        payload=payload,
        height=420,
        payload_overrides=overrides,
    )
    page._render_three_payload(
        container=_DummyContainer(),
        payload=payload,
        height=420,
        payload_overrides=overrides,
    )

    assert calls["augment"] == 1


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


def test_batch_summary_keeps_survey_downloads_in_visible_export_block(
    monkeypatch,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_records"] = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=10.0, y=0.0, z=1000.0, md=2.0),
                WelltrackPoint(x=20.0, y=0.0, z=1100.0, md=3.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=20.0, z=0.0, md=1.0),
                WelltrackPoint(x=10.0, y=20.0, z=1000.0, md=2.0),
                WelltrackPoint(x=20.0, y=20.0, z=1100.0, md=3.0),
            ),
        ),
    ]
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    page.st.session_state["wt_survey_download_selected_names"] = ["WELL-B"]
    captured: dict[str, object] = {
        "download_buttons": [],
        "markdown": [],
        "multiselect": None,
        "radios": [],
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

    def _fake_multiselect(label, options, *args, **kwargs):
        captured["multiselect"] = {
            "label": str(label),
            "options": [str(option) for option in options],
        }
        return list(page.st.session_state.get(kwargs.get("key"), []))

    def _fake_radio(label, options, *args, **kwargs):
        captured["radios"].append(
            {
                "label": str(label),
                "options": [str(option) for option in options],
                "key": str(kwargs.get("key")),
            }
        )
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
    monkeypatch.setattr(
        page.st,
        "markdown",
        lambda message, *args, **kwargs: captured["markdown"].append(str(message)),
    )
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
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
    assert "### Выгрузка" in captured["markdown"]
    assert captured["multiselect"] == {
        "label": "Скважины для выгрузки",
        "options": ["WELL-A", "WELL-B"],
    }
    assert captured["radios"] == [
        {
            "label": "Что выгружать",
            "options": ["Траектории", "Цели"],
            "key": "wt_download_kind",
        },
        {
            "label": "Формат выгрузки",
            "options": ["CSV", "WELLTRACK", ".dev (7z)"],
            "key": "wt_survey_download_format",
        },
    ]
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


def test_batch_summary_reuses_prepared_download_payload_cache(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0)
    ]
    build_calls = {"count": 0}

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

    def _fake_builder(*args, **kwargs):
        build_calls["count"] += 1
        return b"payload"

    monkeypatch.setattr(page, "render_small_note", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "multiselect", lambda *args, **kwargs: [])
    monkeypatch.setattr(page.st, "radio", lambda _label, options, **kwargs: options[0])
    monkeypatch.setattr(page.st, "download_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page, "_build_batch_survey_csv", _fake_builder)

    rows = [{"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3}]
    page._render_batch_summary(rows)
    page._render_batch_summary(rows)

    assert build_calls["count"] == 1


def test_batch_summary_skips_large_download_payloads_until_requested(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    row_count = 5001
    stations = pd.DataFrame(
        {
            "MD_m": np.arange(row_count, dtype=float),
            "INC_deg": np.zeros(row_count, dtype=float),
            "AZI_deg": np.zeros(row_count, dtype=float),
            "X_m": np.arange(row_count, dtype=float),
            "Y_m": np.zeros(row_count, dtype=float),
            "Z_m": np.arange(row_count, dtype=float),
            "DLS_deg_per_30m": np.zeros(row_count, dtype=float),
            "segment": ["HORIZONTAL"] * row_count,
        }
    )
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-LARGE", y_offset_m=0.0, stations=stations)
    ]
    captured: dict[str, object] = {"download_buttons": 0, "captions": []}

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

    def _fail_builder(*args, **kwargs):
        raise AssertionError("large download payload must wait for explicit toggle")

    monkeypatch.setattr(page, "render_small_note", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "columns", _fake_columns)
    monkeypatch.setattr(
        page.st,
        "caption",
        lambda value, *args, **kwargs: captured["captions"].append(str(value)),
    )
    monkeypatch.setattr(page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(page.st, "multiselect", lambda *args, **kwargs: [])
    monkeypatch.setattr(page.st, "radio", lambda _label, options, **kwargs: options[0])
    monkeypatch.setattr(page.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        page.st,
        "download_button",
        lambda *args, **kwargs: captured.__setitem__(
            "download_buttons", int(captured["download_buttons"]) + 1
        ),
    )
    monkeypatch.setattr(page, "_build_batch_survey_csv", _fail_builder)

    page._render_batch_summary(
        [{"Скважина": "WELL-LARGE", "Статус": "OK", "Проблема": "", "Точек": 3}]
    )

    assert captured["download_buttons"] == 0
    assert any("Включите подготовку файлов перед скачиванием" in item for item in captured["captions"])


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
    monkeypatch.setattr(
        page.st,
        "text_input",
        lambda _label, *args, **kwargs: page.st.session_state.setdefault(
            kwargs.get("key"), ""
        ),
    )
    monkeypatch.setattr(page.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dialog", lambda *_args, **_kwargs: (lambda func: func))
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
    assert (
        bytes(selected_download["data"])
        .decode("utf-8")
        .startswith("# SURVEY FROM PYWP")
    )


def test_batch_summary_dev_folder_export_writes_selected_files(monkeypatch, tmp_path) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    page.st.session_state["wt_survey_download_selected_names"] = ["WELL-A"]
    page.st.session_state["wt_survey_download_format"] = ".dev (7z)"
    captured: dict[str, object] = {"download_buttons": [], "success": []}

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
        captured["download_buttons"].append(str(label))
        return False

    def _fake_button(label, *args, **kwargs):
        return str(label) == "Сохранить выбранные .dev"

    def _fake_text_input(_label, *args, **kwargs):
        key = kwargs.get("key")
        page.st.session_state[key] = str(tmp_path)
        return str(tmp_path)

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
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "button", _fake_button)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dialog", lambda *_args, **_kwargs: (lambda func: func))
    monkeypatch.setattr(
        page.st,
        "success",
        lambda message, *args, **kwargs: captured["success"].append(str(message)),
    )
    monkeypatch.setattr(page.st, "download_button", _fake_download_button)

    page._render_batch_summary(
        [
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
        ]
    )

    written = tmp_path / "WELL-A.dev"
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith("# SURVEY FROM PYWP")
    assert page.st.session_state.get("wt_dev_export_pending") is None


def test_batch_summary_dev_folder_export_requests_overwrite_confirmation(
    monkeypatch,
    tmp_path,
) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    page.st.session_state["wt_survey_download_selected_names"] = ["WELL-A"]
    page.st.session_state["wt_survey_download_format"] = ".dev (7z)"
    (tmp_path / "WELL-A.dev").write_text("old", encoding="utf-8")
    captured: dict[str, object] = {"dialog_titles": [], "writes": []}

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

    def _fake_button(label, *args, **kwargs):
        label_text = str(label)
        if label_text == "Сохранить выбранные .dev":
            return True
        return False

    def _fake_text_input(_label, *args, **kwargs):
        key = kwargs.get("key")
        page.st.session_state[key] = str(tmp_path)
        return str(tmp_path)

    def _fake_dialog(title, *args, **kwargs):
        captured["dialog_titles"].append(str(title))

        def _decorator(func):
            def _wrapped(*f_args, **f_kwargs):
                return func(*f_args, **f_kwargs)

            return _wrapped

        return _decorator

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
    monkeypatch.setattr(page.st, "text_input", _fake_text_input)
    monkeypatch.setattr(page.st, "button", _fake_button)
    monkeypatch.setattr(page.st, "rerun", lambda *args, **kwargs: None)
    monkeypatch.setattr(page.st, "dialog", _fake_dialog)
    monkeypatch.setattr(
        page.st,
        "write",
        lambda message, *args, **kwargs: captured["writes"].append(str(message)),
    )
    monkeypatch.setattr(page.st, "download_button", lambda *args, **kwargs: False)

    page._render_batch_summary(
        [
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
        ]
    )

    pending = page.st.session_state.get("wt_dev_export_pending")
    assert pending is not None
    assert pending.conflict_file_names == ("WELL-A.dev",)
    assert captured["dialog_titles"] == ["Сохранение .dev в папку"]
    assert any("WELL-A (WELL-A.dev)" in line for line in captured["writes"])


def test_batch_summary_dev_folder_export_rejects_drive_root() -> None:
    panel = wt_import_module.ptc_batch_summary_panel

    path, error_message = panel._validated_dev_export_directory("C:\\")

    assert path is None
    assert error_message == "На диски C: и D: выгрузка .dev файлов запрещена."


def test_batch_summary_dev_folder_export_rejects_drive_d() -> None:
    panel = wt_import_module.ptc_batch_summary_panel

    path, error_message = panel._validated_dev_export_directory(r"D:\Exports\DEV")

    assert path is None
    assert error_message == "На диски C: и D: выгрузка .dev файлов запрещена."


def test_batch_summary_dev_folder_export_accepts_quoted_directory_path(
    tmp_path,
) -> None:
    panel = wt_import_module.ptc_batch_summary_panel

    path, error_message = panel._validated_dev_export_directory(f'"{tmp_path / "dev"}"')

    assert error_message is None
    assert path == Path(tmp_path / "dev")


def test_batch_summary_can_export_target_points(monkeypatch) -> None:
    page = wt_import_module
    page.st.session_state.clear()
    page.st.session_state["wt_download_kind"] = "Цели"
    page.st.session_state["wt_target_download_selected_names"] = ["WELL-A"]
    page.st.session_state["wt_records"] = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=10.0, y=0.0, z=1000.0, md=2.0),
                WelltrackPoint(x=20.0, y=0.0, z=1100.0, md=3.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=20.0, z=0.0, md=1.0),
                WelltrackPoint(x=10.0, y=20.0, z=1000.0, md=2.0),
                WelltrackPoint(x=20.0, y=20.0, z=1100.0, md=3.0),
            ),
        ),
    ]
    page.st.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    captured: dict[str, object] = {"download_buttons": [], "radios": []}

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
        lambda _label, options, *args, **kwargs: list(
            page.st.session_state.get(kwargs.get("key"), [])
        ),
    )
    def _fake_radio(label, options, *args, **kwargs):
        captured["radios"].append(
            {
                "label": str(label),
                "options": [str(option) for option in options],
                "key": str(kwargs.get("key")),
            }
        )
        return str(page.st.session_state.get(kwargs.get("key"), options[0]))

    monkeypatch.setattr(page.st, "radio", _fake_radio)
    monkeypatch.setattr(page.st, "download_button", _fake_download_button)

    page._render_batch_summary(
        [
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
        ]
    )

    downloads = list(captured["download_buttons"])
    labels = {item["label"] for item in downloads}
    assert captured["radios"] == [
        {
            "label": "Что выгружать",
            "options": ["Траектории", "Цели"],
            "key": "wt_download_kind",
        },
        {
            "label": "Формат выгрузки",
            "options": ["CSV", "WELLTRACK", ".dev (7z)"],
            "key": "wt_target_download_format",
        },
    ]
    assert labels == {
        "Скачать цели всех скважин",
        "Скачать цели выбранных скважин",
    }
    selected_download = next(
        item for item in downloads if item["label"] == "Скачать цели выбранных скважин"
    )
    selected_csv = bytes(selected_download["data"]).decode("utf-8")
    assert "WELL-A" in selected_csv
    assert "WELL-B" not in selected_csv
    assert "point_name" in selected_csv
    assert selected_download["file_name"] == "welltrack_targets_selected.csv"
    assert selected_download["mime"] == "text/csv"


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
            "reason": "Подготовлен anti-collision пересчёт для пары.",
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
            "Оптимизация": "Anti-collision",
            "SF до": "0.69",
            "Источник": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "Причина": "Подготовлен anti-collision пересчёт для пары.",
        }
    ]


def test_format_prepared_override_scope_shows_local_mode_for_selected_wells() -> None:
    page = wt_import_module
    page.st.session_state["wt_prepared_well_overrides"] = {
        "WELL-B": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "reason": "Подготовлен anti-collision пересчёт для пары.",
        },
        "WELL-C": {
            "update_fields": {"optimization_mode": "anti_collision_avoidance"},
            "source": "ac-cluster-001 · WELL-A, WELL-B, WELL-C · событий 2 · SF 0.71",
            "reason": "Раннее пересечение: anti-collision пересчёт по KOP/BUILD1.",
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
            "Локальный режим": "Anti-collision",
            "Источник": "WELL-A ↔ WELL-B · Траектория / review · SF 0.69",
            "Маневр": "Сместить post-entry / HORIZONTAL",
        },
        {
            "Скважина": "WELL-C",
            "Локальный режим": "Anti-collision",
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
            "reason": "Раннее пересечение: anti-collision пересчёт по KOP/BUILD1.",
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
            "action_label": "Подготовить anti-collision пересчёт",
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
    page.st.session_state["wt_anticollision_analysis_cache"] = {}
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

    def _fake_incremental_build(
        successes,
        *,
        model,
        name_to_color=None,
        reference_wells=(),
        reference_uncertainty_models_by_name=None,
        well_signature_by_name=None,
        previous_well_cache=None,
        previous_pair_cache=None,
        progress_callback=None,
        parallel_workers=0,
    ):
        calls["analysis"] += 1
        return (
            analysis,
            {str(success.name): ("sig", object()) for success in successes},
            {},
            AntiCollisionIncrementalStats(
                reused_well_count=0,
                rebuilt_well_count=len(successes) + len(reference_wells),
                reused_pair_count=0,
                recalculated_pair_count=0,
            ),
        )

    monkeypatch.setattr(
        page,
        "_build_incremental_anti_collision_analysis",
        _fake_incremental_build,
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


def test_cached_anti_collision_view_model_passes_previous_incremental_cache(
    monkeypatch,
) -> None:
    page = wt_import_module
    page._init_state()
    page.st.session_state["wt_anticollision_analysis_cache"] = {}
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=3,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )
    calls: list[dict[str, object]] = []

    def _fake_incremental_build(
        successes,
        *,
        model,
        name_to_color=None,
        reference_wells=(),
        reference_uncertainty_models_by_name=None,
        well_signature_by_name=None,
        previous_well_cache=None,
        previous_pair_cache=None,
        progress_callback=None,
        parallel_workers=0,
    ):
        call_index = len(calls)
        calls.append(
            {
                "previous_well_cache": previous_well_cache,
                "previous_pair_cache": previous_pair_cache,
                "well_signature_by_name": dict(well_signature_by_name or {}),
            }
        )
        return (
            analysis,
            {
                str(success.name): (
                    str((well_signature_by_name or {}).get(str(success.name), "")),
                    object(),
                )
                for success in successes
            },
            {("WELL-A", "WELL-B"): object()},
            AntiCollisionIncrementalStats(
                reused_well_count=2 if call_index else 0,
                rebuilt_well_count=1 if call_index else len(successes),
                reused_pair_count=1 if call_index else 0,
                recalculated_pair_count=2 if call_index else 3,
            ),
        )

    monkeypatch.setattr(
        page,
        "_build_incremental_anti_collision_analysis",
        _fake_incremental_build,
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

    model = planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET)
    page._cached_anti_collision_view_model(
        successes=[
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=20.0),
            _successful_plan(name="WELL-C", y_offset_m=40.0),
        ],
        uncertainty_model=model,
        records=[],
    )
    page._cached_anti_collision_view_model(
        successes=[
            _successful_plan(name="WELL-A", y_offset_m=0.0),
            _successful_plan(name="WELL-B", y_offset_m=25.0),
            _successful_plan(name="WELL-C", y_offset_m=40.0),
        ],
        uncertainty_model=model,
        records=[],
    )

    assert len(calls) == 2
    assert calls[0]["previous_well_cache"] is None
    assert calls[0]["previous_pair_cache"] is None
    assert isinstance(calls[1]["previous_well_cache"], Mapping)
    assert isinstance(calls[1]["previous_pair_cache"], Mapping)
    assert (
        calls[0]["well_signature_by_name"]["WELL-A"]
        == calls[1]["well_signature_by_name"]["WELL-A"]
    )
    assert (
        calls[0]["well_signature_by_name"]["WELL-B"]
        != calls[1]["well_signature_by_name"]["WELL-B"]
    )
    last_run = page.st.session_state["wt_anticollision_last_run"]
    assert int(last_run["reused_pair_count"]) == 1
    assert "Инкрементальный anti-collision" in "\n".join(last_run["log_lines"])


def test_cached_anti_collision_view_model_reports_pair_progress_with_eta(
    monkeypatch,
) -> None:
    page = wt_import_module
    page._init_state()
    page.st.session_state["wt_anticollision_analysis_cache"] = {}
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=10,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )
    captured_workers: list[int] = []

    def _fake_incremental_build(
        successes,
        *,
        model,
        name_to_color=None,
        reference_wells=(),
        reference_uncertainty_models_by_name=None,
        well_signature_by_name=None,
        previous_well_cache=None,
        previous_pair_cache=None,
        progress_callback=None,
        parallel_workers=0,
    ):
        captured_workers.append(int(parallel_workers))
        assert progress_callback is not None
        progress_callback(
            AntiCollisionProgress(
                pair_count=10,
                completed_pair_count=5,
                reused_pair_count=2,
                recalculated_pair_count=6,
                prefiltered_pair_count=1,
                elapsed_s=30.0,
                parallel_workers=int(parallel_workers),
            )
        )
        return (
            analysis,
            {str(success.name): ("sig", object()) for success in successes},
            {},
            AntiCollisionIncrementalStats(
                reused_well_count=0,
                rebuilt_well_count=len(successes) + len(reference_wells),
                reused_pair_count=2,
                recalculated_pair_count=8,
            ),
        )

    monkeypatch.setattr(
        page,
        "_build_incremental_anti_collision_analysis",
        _fake_incremental_build,
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
    progress_events: list[tuple[int, str]] = []

    page._cached_anti_collision_view_model(
        successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        records=[],
        progress_callback=lambda value, text: progress_events.append(
            (int(value), str(text))
        ),
        parallel_workers=4,
    )

    assert captured_workers == [4]
    assert any(
        value == 48
        and "пары 5/10" in text
        and "осталось оц. 30 с" in text
        and "4 процессов" in text
        for value, text in progress_events
    )
    assert page.st.session_state["wt_anticollision_last_run"]["parallel_workers"] == 4


def test_cached_anti_collision_view_model_reports_cone_progress(
    monkeypatch,
) -> None:
    page = wt_import_module
    page._init_state()
    page.st.session_state["wt_anticollision_analysis_cache"] = {}
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

    def _fake_incremental_build(
        successes,
        *,
        model,
        name_to_color=None,
        reference_wells=(),
        reference_uncertainty_models_by_name=None,
        well_signature_by_name=None,
        previous_well_cache=None,
        previous_pair_cache=None,
        progress_callback=None,
        parallel_workers=0,
    ):
        assert progress_callback is not None
        progress_callback(
            AntiCollisionProgress(
                pair_count=0,
                completed_pair_count=0,
                elapsed_s=12.0,
                parallel_workers=int(parallel_workers),
                stage="wells",
                well_count=2,
                completed_well_count=1,
                reused_well_count=1,
                rebuilt_well_count=0,
            )
        )
        return (
            analysis,
            {str(success.name): ("sig", object()) for success in successes},
            {},
            AntiCollisionIncrementalStats(
                reused_well_count=1,
                rebuilt_well_count=1,
                reused_pair_count=0,
                recalculated_pair_count=0,
            ),
        )

    monkeypatch.setattr(
        page,
        "_build_incremental_anti_collision_analysis",
        _fake_incremental_build,
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
    progress_events: list[tuple[int, str]] = []

    page._cached_anti_collision_view_model(
        successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        records=[],
        progress_callback=lambda value, text: progress_events.append(
            (int(value), str(text))
        ),
        parallel_workers=4,
    )

    assert any(
        "конусы неопределённости 1/2" in text
        and "кэш скважин 1" in text
        for _value, text in progress_events
    )


def test_cached_anti_collision_cone_eta_uses_parallel_rebuild_workers(
    monkeypatch,
) -> None:
    page = wt_import_module
    page._init_state()
    page.st.session_state["wt_anticollision_analysis_cache"] = {}
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

    def _fake_incremental_build(
        successes,
        *,
        model,
        name_to_color=None,
        reference_wells=(),
        reference_uncertainty_models_by_name=None,
        well_signature_by_name=None,
        previous_well_cache=None,
        previous_pair_cache=None,
        progress_callback=None,
        parallel_workers=0,
    ):
        assert progress_callback is not None
        progress_callback(
            AntiCollisionProgress(
                pair_count=0,
                completed_pair_count=0,
                elapsed_s=30.0,
                parallel_workers=int(parallel_workers),
                stage="wells",
                well_count=10,
                completed_well_count=1,
                reused_well_count=0,
                rebuilt_well_count=1,
                rebuild_well_count=10,
                completed_rebuild_well_count=1,
            )
        )
        return (
            analysis,
            {str(success.name): ("sig", object()) for success in successes},
            {},
            AntiCollisionIncrementalStats(
                reused_well_count=0,
                rebuilt_well_count=10,
                reused_pair_count=0,
                recalculated_pair_count=0,
            ),
        )

    monkeypatch.setattr(
        page,
        "_build_incremental_anti_collision_analysis",
        _fake_incremental_build,
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
    progress_events: list[tuple[int, str]] = []

    page._cached_anti_collision_view_model(
        successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        records=[],
        progress_callback=lambda value, text: progress_events.append(
            (int(value), str(text))
        ),
        parallel_workers=4,
    )

    assert any(
        "конусы неопределённости 1/10" in text
        and "осталось оц. 1 мин 08 с" in text
        and "4 процессов" in text
        and "построено 1/10" in text
        for _value, text in progress_events
    )


def test_cached_anti_collision_parallel_eta_uses_worker_count_before_first_wave(
    monkeypatch,
) -> None:
    page = wt_import_module
    page._init_state()
    page.st.session_state["wt_anticollision_analysis_cache"] = {}
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=10,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )

    def _fake_incremental_build(
        successes,
        *,
        model,
        name_to_color=None,
        reference_wells=(),
        reference_uncertainty_models_by_name=None,
        well_signature_by_name=None,
        previous_well_cache=None,
        previous_pair_cache=None,
        progress_callback=None,
        parallel_workers=0,
    ):
        assert progress_callback is not None
        progress_callback(
            AntiCollisionProgress(
                pair_count=10,
                completed_pair_count=1,
                elapsed_s=30.0,
                parallel_workers=int(parallel_workers),
            )
        )
        return (
            analysis,
            {str(success.name): ("sig", object()) for success in successes},
            {},
            AntiCollisionIncrementalStats(
                reused_well_count=0,
                rebuilt_well_count=len(successes) + len(reference_wells),
                reused_pair_count=0,
                recalculated_pair_count=10,
            ),
        )

    monkeypatch.setattr(
        page,
        "_build_incremental_anti_collision_analysis",
        _fake_incremental_build,
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
    progress_events: list[tuple[int, str]] = []

    page._cached_anti_collision_view_model(
        successes=[_successful_plan(name="WELL-A", y_offset_m=0.0)],
        uncertainty_model=planning_uncertainty_model_for_preset(
            DEFAULT_UNCERTAINTY_PRESET
        ),
        records=[],
        progress_callback=lambda value, text: progress_events.append(
            (int(value), str(text))
        ),
        parallel_workers=4,
    )

    assert any(
        "пары 1/10" in text
        and "осталось оц. 1 мин 08 с" in text
        and "4 процессов" in text
        for _value, text in progress_events
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
    assert "Пересчёт по кластеру недоступен" in str(
        page.st.session_state["wt_prepared_override_message"]
    )
    assert "разнести цели" in str(page.st.session_state["wt_prepared_override_message"])


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
    assert "anti-collision пересчёт по KOP/BUILD1" in str(payload["reason"])
    assert "anti-collision пересчёт" in str(payload["reason"])


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
                    "action_label": "Подготовить anti-collision пересчёт",
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
                    "action_label": "Подготовить anti-collision пересчёт",
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
                    "action_label": "Подготовить anti-collision пересчёт",
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
                    "action_label": "Подготовить anti-collision пересчёт",
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
        if str(trace.name).endswith(": до пересчёта")
    ]
    assert len(previous_3d) == 1
    assert len(previous_plan) == 1
    assert str(previous_3d[0]["dash"]) == "dot"
    assert str(previous_plan[0].line.dash) == "dot"


def test_anticollision_figures_show_pending_target_only_points_without_old_trajectory() -> (
    None
):
    page = wt_import_module
    analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-B", y_offset_m=5.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    target_only = page._TargetOnlyWell(
        name="WELL-A",
        surface=Point3D(10.0, 20.0, 0.0),
        t1=Point3D(100.0, 200.0, 1500.0),
        t3=Point3D(300.0, 400.0, 1600.0),
        status="Не рассчитана",
        problem="Цели изменены в 3D",
        target_points=(
            Point3D(10.0, 20.0, 0.0),
            Point3D(100.0, 200.0, 1500.0),
            Point3D(300.0, 400.0, 1600.0),
        ),
        target_labels=("S", "t1", "t3"),
    )

    payload_3d = page._all_wells_anticollision_three_payload(
        analysis,
        target_only_wells=[target_only],
        name_to_color={"WELL-A": "#123456", "WELL-B": "#654321"},
        focus_well_names=("WELL-A", "WELL-B"),
        render_mode=page.WT_3D_RENDER_DETAIL,
    )
    figure_plan = page._all_wells_anticollision_plan_figure(
        analysis,
        target_only_wells=[target_only],
        name_to_color={"WELL-A": "#123456", "WELL-B": "#654321"},
        focus_well_names=("WELL-A", "WELL-B"),
    )

    assert not any(
        str(item.get("name")) == "WELL-A"
        for item in list(payload_3d.get("lines") or [])
    )
    assert not any(
        str(item.get("name")).startswith("WELL-A cone")
        for item in list(payload_3d.get("meshes") or [])
    )
    assert any(
        str(item.get("name")) == "WELL-A: цели (без траектории)"
        and len(list(item.get("points") or [])) == 3
        for item in list(payload_3d.get("points") or [])
    )
    assert any(
        str(trace.name) == "WELL-A: цели (без траектории)"
        and len(trace.x) == 3
        for trace in figure_plan.data
    )


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


def test_all_wells_three_payload_renders_pilot_labels_from_records_mapping() -> None:
    page = wt_import_module
    payload = page._all_wells_three_payload(
        [
            _successful_plan(
                name="WELL-A_PL",
                y_offset_m=0.0,
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 500.0, 1000.0],
                        "X_m": [0.0, 200.0, 500.0],
                        "Y_m": [0.0, 50.0, 100.0],
                        "Z_m": [0.0, 650.0, 1100.0],
                        "INC_deg": [0.0, 10.0, 20.0],
                        "AZI_deg": [0.0, 90.0, 90.0],
                        "DLS_deg_per_30m": [0.0, 1.0, 1.0],
                        "segment": ["PILOT", "PILOT", "PILOT"],
                    }
                ),
            )
        ],
        pilot_study_points_by_name={
            "WELL-A_PL": (
                Point3D(200.0, 50.0, 650.0),
                Point3D(350.0, 75.0, 900.0),
                Point3D(500.0, 100.0, 1100.0),
            )
        },
    )

    pilot_marker = next(
        item for item in payload["points"] if str(item.get("name")) == "WELL-A_PL: цели"
    )
    label_texts = [str(item.get("text")) for item in payload["labels"]]

    assert [hover["point"] for hover in pilot_marker["hover"]] == [
        "WELL-A_PL: 1",
        "WELL-A_PL: 2",
        "WELL-A_PL: 3",
    ]
    assert "WELL-A_PL: 1" in label_texts
    assert "WELL-A_PL: 2" in label_texts
    assert "WELL-A_PL: 3" in label_texts


def test_failed_target_only_wells_include_single_point_pilot_targets() -> None:
    page = wt_import_module
    records = [
        WelltrackRecord(
            name="WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=50.0, z=900.0, md=1.0),
            ),
        )
    ]
    summary_rows = [
        {
            "Скважина": "WELL-A_PL",
            "Статус": "Ошибка расчета",
            "Проблема": "pilot target is incomplete",
        }
    ]

    target_only_wells = page._failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    payload_3d = page._all_wells_three_payload(
        [],
        target_only_wells=target_only_wells,
        name_to_color={"WELL-A_PL": "#123456"},
    )
    figure_plan = page._all_wells_plan_figure(
        [],
        target_only_wells=target_only_wells,
        name_to_color={"WELL-A_PL": "#123456"},
    )

    marker = next(
        item
        for item in payload_3d["points"]
        if str(item.get("name")) == "WELL-A_PL: цели (без траектории)"
    )
    plan_trace = next(
        trace
        for trace in figure_plan.data
        if str(trace.name) == "WELL-A_PL: цели (без траектории)"
    )

    assert len(target_only_wells) == 1
    assert target_only_wells[0].target_labels == ("S", "PL1")
    assert marker["points"] == [[0.0, 0.0, 0.0], [100.0, 50.0, 900.0]]
    assert [hover["point"] for hover in marker["hover"]] == ["S", "PL1"]
    assert list(plan_trace.customdata[:, 0]) == ["S", "PL1"]


def test_failed_target_only_non_zbs_two_point_fallback_keeps_distinct_t1_t3() -> None:
    page = wt_import_module
    records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=50.0, z=900.0, md=1.0),
            ),
        )
    ]
    summary_rows = [
        {
            "Скважина": "WELL-A",
            "Статус": "Ошибка расчета",
            "Проблема": "not enough points for parser",
        }
    ]

    target_only_wells = page._failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )

    assert len(target_only_wells) == 1
    assert target_only_wells[0].surface == Point3D(0.0, 0.0, 0.0)
    assert target_only_wells[0].t1 == Point3D(0.0, 0.0, 0.0)
    assert target_only_wells[0].t3 == Point3D(100.0, 50.0, 900.0)
    assert target_only_wells[0].t1 != target_only_wells[0].t3


def test_failed_target_only_wells_default_to_error_status_when_row_status_is_empty() -> None:
    page = wt_import_module
    records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=0.0, z=900.0, md=900.0),
                WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=1100.0),
            ),
        )
    ]
    summary_rows = [
        {
            "Скважина": "WELL-A",
            "Статус": "",
            "Проблема": "solver failed",
        }
    ]

    target_only_wells = page._failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )

    assert len(target_only_wells) == 1
    assert target_only_wells[0].status == "Ошибка расчета"
    assert target_only_wells[0].problem == "solver failed"


def test_failed_target_only_zbs_uses_two_targets_and_stays_in_focused_3d_bounds() -> None:
    page = wt_import_module
    records = [
        WelltrackRecord(
            name="9010_ZBS",
            points=(
                WelltrackPoint(x=5000.0, y=10.0, z=1500.0, md=1.0),
                WelltrackPoint(x=6200.0, y=20.0, z=1500.0, md=2.0),
            ),
        )
    ]
    summary_rows = [
        {
            "Скважина": "9010_ZBS",
            "Статус": "Ошибка расчета",
            "Проблема": "нет фактической траектории",
        }
    ]

    target_only_wells = page._failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    payload_3d = page._all_wells_three_payload(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        target_only_wells=target_only_wells,
        name_to_color={"9010_ZBS": "#123456"},
        focus_well_names=("WELL-A",),
    )

    assert len(target_only_wells) == 1
    assert target_only_wells[0].target_labels == ("t1", "t3")
    assert target_only_wells[0].t1.x == pytest.approx(5000.0)
    assert target_only_wells[0].t3.x == pytest.approx(6200.0)
    marker = next(
        item
        for item in payload_3d["points"]
        if str(item.get("name")) == "9010_ZBS: цели (без траектории)"
    )
    assert [hover["point"] for hover in marker["hover"]] == ["t1", "t3"]
    assert payload_3d["bounds"]["max"][0] >= 6200.0


def test_failed_target_only_multi_horizontal_zbs_uses_level_labels() -> None:
    page = wt_import_module
    records = [
        WelltrackRecord(
            name="9010_ZBS",
            points=(
                WelltrackPoint(x=5000.0, y=10.0, z=1500.0, md=1.0),
                WelltrackPoint(x=6200.0, y=20.0, z=1500.0, md=2.0),
                WelltrackPoint(x=7000.0, y=30.0, z=1520.0, md=3.0),
                WelltrackPoint(x=7600.0, y=40.0, z=1520.0, md=4.0),
            ),
        )
    ]
    summary_rows = [
        {
            "Скважина": "9010_ZBS",
            "Статус": "Ошибка расчета",
            "Проблема": "нет фактической траектории",
        }
    ]

    target_only_wells = page._failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    edit_wells = page.ptc_three_overrides.build_target_only_edit_wells_payload(
        target_only_wells,
        {"9010_ZBS": "#123456"},
    )

    assert len(target_only_wells) == 1
    assert target_only_wells[0].target_labels == (
        "1_t1",
        "1_t3",
        "2_t1",
        "2_t3",
    )
    assert len(target_only_wells[0].target_pairs) == 2
    assert target_only_wells[0].t1.x == pytest.approx(5000.0)
    assert target_only_wells[0].t3.x == pytest.approx(7600.0)
    assert [point["label"] for point in edit_wells[0]["edit_points"]] == [
        "1_t1",
        "1_t3",
        "2_t1",
        "2_t3",
    ]
    assert [point["index"] for point in edit_wells[0]["edit_points"]] == [0, 1, 2, 3]


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
    assert str(label_by_text["FACT-1"]["color"]) == "#374151"
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
    assert {model.iscwsa_tool_code for model in none_model_by_name.values()} == {
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


def test_reference_well_labels_anchor_at_horizontal_entry_for_horizontal_wells_2d() -> (
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

    focus_names = page._focus_pad_well_names(records=records, focus_pad_id="PAD2")

    assert focus_names == ("PAD2-A", "PAD2-B")


def test_focus_pad_well_names_exclude_zbs_sidetracks_from_optimizer_scope() -> None:
    page = wt_import_module
    page.st.session_state.clear()
    records = [
        *_records()[:2],
        WelltrackRecord(
            name="WELL-A_ZBS",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=200.0, z=2300.0, md=2300.0),
                WelltrackPoint(x=900.0, y=500.0, z=2300.0, md=2900.0),
            ),
        ),
    ]

    focus_names = page._focus_pad_well_names(
        records=records,
        focus_pad_id=page.WT_PAD_FOCUS_ALL,
    )

    assert focus_names == ("WELL-A", "WELL-B")


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
    assert "FACT-1 (Фактическая) cone" in {
        str(trace.name) for trace in figure_plan.data
    }
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


def test_anticollision_payload_ignores_loaded_display_only_reference_wells_for_camera_bounds() -> None:
    page = wt_import_module
    successes = [_successful_plan(name="WELL-A", y_offset_m=0.0)]
    analysis = page._build_anti_collision_analysis(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )

    base_3d = page._all_wells_anticollision_three_payload(analysis)
    far_ref_3d = page._all_wells_anticollision_three_payload(
        analysis,
        reference_wells=_far_reference_wells(),
    )

    assert base_3d["bounds"] == far_ref_3d["bounds"]


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


def test_anticollision_three_payload_omits_far_reference_wells_from_ac_scope() -> (
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

    hover_names = {
        str(hover.get("name"))
        for item in payload["points"]
        for hover in list(item.get("hover") or [])
    }
    analysis_names = {str(well.name) for well in analysis.wells}
    assert analysis_names == {"WELL-A"}
    assert "FACT-FAR" not in hover_names
    assert "APP-FAR" not in hover_names
    assert "FACT-FAR (Фактическая)" not in hover_names
    assert "APP-FAR (Проектная утвержденная)" not in hover_names


def test_anticollision_payload_can_show_loaded_reference_wells_outside_ac_scope() -> (
    None
):
    page = wt_import_module
    far_reference_wells = _far_reference_wells()
    analysis = page._build_anti_collision_analysis(
        [_successful_plan(name="WELL-A", y_offset_m=0.0)],
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        reference_wells=far_reference_wells,
    )

    assert {str(well.name) for well in analysis.wells} == {"WELL-A"}

    payload = page._all_wells_anticollision_three_payload(
        analysis,
        reference_wells=far_reference_wells,
        render_mode=page.WT_3D_RENDER_DETAIL,
    )
    base_payload = page._all_wells_anticollision_three_payload(
        analysis,
        render_mode=page.WT_3D_RENDER_DETAIL,
    )
    figure_plan = page._all_wells_anticollision_plan_figure(
        analysis,
        reference_wells=far_reference_wells,
    )

    hover_names = {
        str(hover.get("name"))
        for item in payload["points"]
        for hover in list(item.get("hover") or [])
    }
    trace_names = {str(trace.name) for trace in figure_plan.data}

    assert {"FACT-FAR", "APP-FAR"}.issubset(hover_names)
    assert "FACT-FAR (Фактическая)" in trace_names
    assert "APP-FAR (Проектная утвержденная)" in trace_names
    assert payload["bounds"] == base_payload["bounds"]


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


def test_all_wells_three_payload_preserves_overview_labels_and_legend() -> None:
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


def test_all_wells_three_payload_keeps_all_calculated_labels_for_large_dev_like_batch() -> None:
    page = wt_import_module
    successes = [
        _successful_plan(name=f"WELL-{index:03d}", y_offset_m=float(index) * 15.0)
        for index in range(96)
    ]

    payload = page._all_wells_three_payload(
        successes,
        reference_wells=_reference_wells(),
        render_mode=page.WT_3D_RENDER_DETAIL,
    )

    label_texts = {str(item["text"]) for item in payload["labels"]}

    for success in successes:
        assert str(success.name) in label_texts


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
        "Куст PAD1",
        "Куст PAD2",
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


def test_all_wells_three_payload_preserves_hover_metadata_for_tooltip() -> None:
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
    assert {str(item.get("name")) for item in hover_only_points[0]["hover"]} == {
        "FACT-001"
    }


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
