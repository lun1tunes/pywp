from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pywp import ptc_core
from pywp import ptc_page_reference
from pywp import ptc_page_state
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
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.run()

    button_labels = {str(widget.label) for widget in at.button}
    assert "Импорт целей" in button_labels
    assert "Очистить импорт" not in button_labels


def test_ptc_page_uses_automatic_parallel_worker_selection() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_records"] = _records()
    at.session_state["wt_records_original"] = _records()

    at.run(timeout=120)

    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    assert "Параллельный расчёт" not in selectbox_labels
    caption_values = [str(widget.value) for widget in at.caption]
    assert not any("Multiprocessing" in value for value in caption_values)


def test_ptc_page_keeps_open_calc_params_panel_after_three_multi_edit_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = False

    def _fake_three_scene(_payload, **_kwargs):
        nonlocal emitted
        if emitted:
            return None
        emitted = True
        return {
            "type": "pywp:editTargets",
            "nonce": "multi-edit-1",
            "changes": [
                {
                    "name": "WELL-A",
                    "points": [
                        {"index": 1, "position": [610.0, 810.0, 2400.0]},
                        {"index": 2, "position": [1510.0, 2010.0, 2500.0]},
                    ],
                },
                {
                    "name": "WELL-B",
                    "points": [
                        {"index": 1, "position": [630.0, 830.0, 2410.0]},
                        {"index": 2, "position": [1530.0, 2030.0, 2510.0]},
                    ],
                },
            ],
        }

    monkeypatch.setattr(ptc_core, "render_local_three_scene", _fake_three_scene)

    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"
    at.session_state[ptc_page_state.PTC_CALC_PARAMS_OPEN_KEY] = True

    at.run(timeout=120)

    assert not at.exception
    button_labels = {str(widget.label) for widget in at.button}
    assert "Скрыть" in button_labels
    assert at.session_state[ptc_page_state.PTC_CALC_PARAMS_OPEN_KEY] is True
    assert at.session_state["wt_edit_targets_pending_names"] == ["WELL-A", "WELL-B"]


def test_ptc_page_applies_three_pad_edit_without_widget_state_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = False
    captured_pad_id = ""

    def _fake_three_scene(payload, **_kwargs):
        nonlocal emitted, captured_pad_id
        if emitted:
            return None
        emitted = True
        edit_pads = list(payload.get("edit_pads") or [])
        assert edit_pads
        captured_pad_id = str(edit_pads[0]["id"])
        return {
            "type": "pywp:editTargets",
            "nonce": "pad-edit-1",
            "pad_changes": [
                {
                    "pad_id": captured_pad_id,
                    "anchor": [100.0, 200.0, 0.0],
                    "nds_azimuth_deg": 35.0,
                }
            ],
        }

    monkeypatch.setattr(ptc_core, "render_local_three_scene", _fake_three_scene)

    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    assert captured_pad_id
    assert not at.exception
    assert at.session_state["wt_last_edit_targets_nonce"] == "pad-edit-1"
    assert at.session_state["wt_edit_targets_pending_names"] == ["WELL-A", "WELL-B"]
    assert at.session_state["wt_pad_configs"][captured_pad_id]["first_surface_x"] == (
        pytest.approx(100.0)
    )
    assert at.session_state["wt_pad_configs"][captured_pad_id]["first_surface_y"] == (
        pytest.approx(200.0)
    )


def test_apply_edit_pad_changes_updates_pad_layout_state() -> None:
    ptc_core.st.session_state.clear()
    records = _records()
    ptc_core.st.session_state["wt_records"] = list(records)
    ptc_core.st.session_state["wt_records_original"] = list(records)
    pads = ptc_core._ensure_pad_configs(base_records=list(records))
    pad_id = str(pads[0].pad_id)
    cfg = ptc_core.st.session_state["wt_pad_configs"][pad_id]
    next_anchor = [
        float(cfg["first_surface_x"]) + 120.0,
        float(cfg["first_surface_y"]) - 45.0,
        float(cfg["first_surface_z"]),
    ]

    updated_names = ptc_core._apply_edit_pad_changes(
        [{"pad_id": pad_id, "anchor": next_anchor}],
        source="three_viewer",
    )

    assert updated_names == ["WELL-A", "WELL-B"]
    assert (
        ptc_core.st.session_state["wt_pad_configs"][pad_id]["first_surface_x"]
        == next_anchor[0]
    )
    assert (
        ptc_core.st.session_state["wt_pad_configs"][pad_id]["first_surface_y"]
        == next_anchor[1]
    )
    assert (
        ptc_core.st.session_state["wt_pad_configs"][pad_id][
            ptc_core.ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY
        ]
        is True
    )
    assert ptc_core.st.session_state["wt_edit_targets_pending_names"] == [
        "WELL-A",
        "WELL-B",
    ]
    assert ptc_core.st.session_state["wt_edit_targets_applied"] == [
        "WELL-A",
        "WELL-B",
    ]
    assert (
        ptc_core.st.session_state["wt_edit_targets_applied_source"]
        == "three_viewer_pad_layout"
    )
    assert ptc_core.st.session_state["wt_edit_targets_last_source"] == (
        "three_viewer_pad_layout"
    )
    assert ptc_core.st.session_state["wt_edit_targets_highlight_points"] == {
        "WELL-A": [0],
        "WELL-B": [0],
    }
    assert any(
        float(updated.points[0].x) != float(original.points[0].x)
        or float(updated.points[0].y) != float(original.points[0].y)
        for updated, original in zip(ptc_core.st.session_state["wt_records"], records)
    )


def test_apply_edit_pad_changes_updates_nds_azimuth() -> None:
    ptc_core.st.session_state.clear()
    records = _records()
    ptc_core.st.session_state["wt_records"] = list(records)
    ptc_core.st.session_state["wt_records_original"] = list(records)
    pads = ptc_core._ensure_pad_configs(base_records=list(records))
    pad_id = str(pads[0].pad_id)
    cfg = ptc_core.st.session_state["wt_pad_configs"][pad_id]
    original_nds = float(cfg["nds_azimuth_deg"])
    next_nds = (original_nds + 37.5) % 360.0
    anchor = [
        float(cfg["first_surface_x"]),
        float(cfg["first_surface_y"]),
        float(cfg["first_surface_z"]),
    ]

    updated_names = ptc_core._apply_edit_pad_changes(
        [
            {
                "pad_id": pad_id,
                "anchor": anchor,
                "nds_azimuth_deg": next_nds,
            }
        ],
        source="three_viewer",
    )

    assert updated_names == ["WELL-A", "WELL-B"]
    assert ptc_core.st.session_state["wt_pad_configs"][pad_id][
        "nds_azimuth_deg"
    ] == pytest.approx(next_nds)
    assert ptc_core.st.session_state[f"wt_pad_cfg_nds_azimuth_deg_{pad_id}"] == (
        pytest.approx(next_nds)
    )


def test_apply_edit_pad_changes_is_noop_when_geometry_is_unchanged() -> None:
    ptc_core.st.session_state.clear()
    records = _records()
    ptc_core.st.session_state["wt_records"] = list(records)
    ptc_core.st.session_state["wt_records_original"] = list(records)
    pads = ptc_core._ensure_pad_configs(base_records=list(records))
    pad_id = str(pads[0].pad_id)
    cfg_before = dict(ptc_core.st.session_state["wt_pad_configs"][pad_id])
    records_before = list(ptc_core.st.session_state["wt_records"])

    anchor = [
        float(cfg_before["first_surface_x"]),
        float(cfg_before["first_surface_y"]),
        float(cfg_before["first_surface_z"]),
    ]
    assert ptc_core._apply_edit_pad_changes(
        [{"pad_id": pad_id, "anchor": anchor}],
        source="three_viewer",
    ) == []

    assert ptc_core.st.session_state["wt_records"] == records_before
    assert ptc_core.st.session_state["wt_pad_configs"][pad_id] == cfg_before
    assert "wt_pad_last_applied_at" not in ptc_core.st.session_state


def test_queue_surface_edit_feedback_merges_tuple_highlight_rows() -> None:
    ptc_core.st.session_state.clear()
    ptc_core.st.session_state["wt_edit_targets_pending_names"] = ["WELL-A"]
    ptc_core.st.session_state["wt_edit_targets_highlight_names"] = ["WELL-A"]
    ptc_core.st.session_state["wt_edit_targets_highlight_points"] = {
        "WELL-A": (2,),
    }

    changed_names = ptc_core._queue_surface_edit_feedback(
        ["WELL-A"],
        source="pad_layout",
    )

    assert changed_names == ["WELL-A"]
    assert ptc_core.st.session_state["wt_edit_targets_highlight_points"] == {
        "WELL-A": [0, 2]
    }
    assert ptc_core.st.session_state["wt_edit_targets_applied_source"] == "pad_layout"


def test_queue_surface_edit_feedback_drops_stale_highlight_names_without_points() -> None:
    ptc_core.st.session_state.clear()
    ptc_core.st.session_state["wt_edit_targets_pending_names"] = ["WELL-OLD"]
    ptc_core.st.session_state["wt_edit_targets_highlight_names"] = ["WELL-OLD"]
    ptc_core.st.session_state["wt_edit_targets_highlight_points"] = {}

    changed_names = ptc_core._queue_surface_edit_feedback(
        ["WELL-NEW"],
        source="pad_layout",
    )

    assert changed_names == ["WELL-NEW"]
    assert ptc_core.st.session_state["wt_edit_targets_pending_names"] == [
        "WELL-OLD",
        "WELL-NEW",
    ]
    assert ptc_core.st.session_state["wt_edit_targets_highlight_names"] == [
        "WELL-NEW"
    ]
    assert ptc_core.st.session_state["wt_edit_targets_highlight_points"] == {
        "WELL-NEW": [0]
    }
    assert ptc_core.st.session_state["wt_pending_selected_names"] == [
        "WELL-OLD",
        "WELL-NEW",
    ]


def test_queue_surface_edit_feedback_invalidates_only_changed_well_and_keeps_cache() -> None:
    ptc_core.st.session_state.clear()
    records = [
        *_records(),
        WelltrackRecord(
            name="WELL-C",
            points=(
                WelltrackPoint(x=40.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=640.0, y=840.0, z=2420.0, md=2420.0),
                WelltrackPoint(x=1540.0, y=2040.0, z=2520.0, md=3520.0),
            ),
        ),
    ]
    cache = {
        "key": "before-pad-edit",
        "well_cache": {"WELL-A": ("a", object()), "WELL-B": ("b", object())},
        "pair_cache": {("WELL-A", "WELL-B"): object()},
    }
    ptc_core.st.session_state["wt_records"] = records
    ptc_core.st.session_state["wt_successes"] = [
        SimpleNamespace(name="WELL-A"),
        SimpleNamespace(name="WELL-B"),
        SimpleNamespace(name="WELL-C"),
    ]
    ptc_core.st.session_state["wt_summary_rows"] = [
        {"Скважина": name, "Статус": "OK", "Проблема": ""}
        for name in ("WELL-A", "WELL-B", "WELL-C")
    ]
    ptc_core.st.session_state["wt_anticollision_analysis_cache"] = cache

    changed_names = ptc_core._queue_surface_edit_feedback(
        ["WELL-B"],
        source="three_viewer_pad_layout",
    )

    assert changed_names == ["WELL-B"]
    assert ptc_core.st.session_state["wt_anticollision_analysis_cache"] is cache
    assert [item.name for item in ptc_core.st.session_state["wt_successes"]] == [
        "WELL-A",
        "WELL-C",
    ]
    rows_by_name = {
        str(row["Скважина"]): row
        for row in ptc_core.st.session_state["wt_summary_rows"]
    }
    assert rows_by_name["WELL-B"]["Статус"] == "Не рассчитана"
    assert rows_by_name["WELL-A"]["Статус"] == "OK"
    assert rows_by_name["WELL-C"]["Статус"] == "OK"
    assert ptc_core.st.session_state["wt_edit_targets_pending_names"] == ["WELL-B"]
    assert ptc_core.st.session_state["wt_pending_all_wells_results_focus"] is True


def test_apply_edit_pad_changes_keeps_other_pads_incremental_cache(monkeypatch) -> None:
    ptc_core.st.session_state.clear()

    def _pad_records(prefix: str, surface_x: float) -> list[WelltrackRecord]:
        return [
            WelltrackRecord(
                name=f"{prefix}-01",
                points=(
                    WelltrackPoint(x=surface_x, y=0.0, z=0.0, md=0.0),
                    WelltrackPoint(x=surface_x + 500.0, y=0.0, z=1000.0, md=1000.0),
                    WelltrackPoint(x=surface_x + 1000.0, y=0.0, z=1000.0, md=1500.0),
                ),
            ),
            WelltrackRecord(
                name=f"{prefix}-02",
                points=(
                    WelltrackPoint(x=surface_x, y=0.0, z=0.0, md=0.0),
                    WelltrackPoint(x=surface_x + 500.0, y=20.0, z=1000.0, md=1000.0),
                    WelltrackPoint(x=surface_x + 1000.0, y=20.0, z=1000.0, md=1500.0),
                ),
            ),
        ]

    records = [
        *_pad_records("PAD-A", 0.0),
        *_pad_records("PAD-B", 10000.0),
        *_pad_records("PAD-C", 20000.0),
        WelltrackRecord(
            name="PAD-C_PL",
            points=(
                WelltrackPoint(x=19000.0, y=25.0, z=0.0, md=0.0),
                WelltrackPoint(x=19500.0, y=25.0, z=1000.0, md=1000.0),
            ),
        ),
    ]
    ptc_core.st.session_state["wt_records"] = list(records)
    ptc_core.st.session_state["wt_records_original"] = list(records)
    ptc_core.st.session_state["wt_successes"] = [
        SimpleNamespace(name=record.name)
        for record in records
        if not str(record.name).endswith("_PL")
    ]
    ptc_core.st.session_state["wt_summary_rows"] = [
        {"Скважина": record.name, "Статус": "OK", "Проблема": ""}
        for record in records
    ]
    cache = {
        "key": "three-pads-before-edit",
        "well_cache": {record.name: (record.name, object()) for record in records},
        "pair_cache": {},
    }
    ptc_core.st.session_state["wt_anticollision_analysis_cache"] = cache
    pads = ptc_core._ensure_pad_configs(base_records=list(records))
    assert [str(pad.pad_id) for pad in pads] == ["PAD-A", "PAD-B", "PAD-C"]
    laid_out_records = ptc_core.sync_pilot_surfaces_to_parents(
        ptc_core.apply_pad_layout(
            records=list(records),
            pads=pads,
            plan_by_pad_id=ptc_core._build_pad_plan_map(pads),
        )
    )
    ptc_core.st.session_state["wt_records"] = laid_out_records
    records_by_name_before = {
        str(record.name): record
        for record in laid_out_records
    }
    pad_id = "PAD-B"
    cfg = ptc_core.st.session_state["wt_pad_configs"][pad_id]
    anchor = [
        float(cfg["first_surface_x"]) + 250.0,
        float(cfg["first_surface_y"]) + 75.0,
        float(cfg["first_surface_z"]),
    ]

    applied_pad_ids: list[str] = []
    original_apply_pad_layout = ptc_core.apply_pad_layout

    def _capture_apply_pad_layout(*, records, pads, plan_by_pad_id):
        applied_pad_ids.extend(str(pad.pad_id) for pad in pads)
        return original_apply_pad_layout(
            records=records,
            pads=pads,
            plan_by_pad_id=plan_by_pad_id,
        )

    monkeypatch.setattr(ptc_core, "apply_pad_layout", _capture_apply_pad_layout)

    updated_names = ptc_core._apply_edit_pad_changes(
        [{"pad_id": pad_id, "anchor": anchor}],
        source="three_viewer",
    )

    assert updated_names == ["PAD-B-01", "PAD-B-02"]
    assert applied_pad_ids == ["PAD-B"]
    assert ptc_core.st.session_state["wt_anticollision_analysis_cache"] is cache
    assert [item.name for item in ptc_core.st.session_state["wt_successes"]] == [
        "PAD-A-01",
        "PAD-A-02",
        "PAD-C-01",
        "PAD-C-02",
    ]
    records_by_name_after = {
        str(record.name): record
        for record in ptc_core.st.session_state["wt_records"]
    }
    for well_name in ("PAD-A-01", "PAD-A-02", "PAD-C-01", "PAD-C-02"):
        assert records_by_name_after[well_name] == records_by_name_before[well_name]
    assert records_by_name_after["PAD-C_PL"] == records_by_name_before["PAD-C_PL"]
    assert ptc_core.st.session_state["wt_edit_targets_pending_names"] == updated_names


def test_ptc_page_hides_engineering_result_controls_and_single_well_debug_sections() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

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
    at.run(timeout=120)

    expander_labels = {str(widget.label) for widget in at.expander}
    assert "Контроль попадания и точность расчета" not in expander_labels
    assert "Технические параметры и диагностика решателя" not in expander_labels


def test_ptc_page_shows_pad_layout_apply_feedback_message() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_records"] = _records()
    at.session_state["wt_records_original"] = _records()
    at.session_state["wt_edit_targets_applied"] = ["WELL-A", "WELL-B"]
    at.session_state["wt_edit_targets_applied_source"] = "pad_layout"

    at.run(timeout=120)

    assert any(
        "Координаты устьев обновлены по параметрам кустов: WELL-A, WELL-B."
        in str(widget.value)
        for widget in at.success
    )


def test_ptc_page_defers_anticollision_when_three_edits_are_pending() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "Не рассчитана", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-C", "Статус": "OK", "Проблема": "", "Точек": 3},
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-B", y_offset_m=25.0),
        _successful_plan(name="WELL-C", y_offset_m=50.0),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"
    at.session_state["wt_edit_targets_pending_names"] = ["WELL-A"]
    at.session_state["wt_pending_selected_names"] = ["WELL-A"]

    at.run(timeout=120)

    radio_labels = {str(widget.label) for widget in at.radio}
    assert "Режим отображения всех скважин" not in radio_labels
    assert str(at.session_state["wt_results_all_view_mode"]) == "Anti-collision"
    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    metric_labels = {str(widget.label) for widget in at.metric}
    assert "Пресет неопределенности для anti-collision" not in selectbox_labels
    assert "Проверено пар" not in metric_labels


def test_ptc_page_wraps_pad_layout_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "@st.fragment" in source
    assert "def _render_pad_layout_section(records: list[object]) -> None:" in source
    assert "_render_pad_layout_section(records=records)" in source


def test_ptc_page_wraps_target_import_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_target_import_section_fragment() -> None:" in source
    assert "render_target_import_section()" in source
    assert "_render_target_import_section_fragment()" in source


def test_ptc_core_contains_bulk_horizontal_length_preprocess_controls() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "Изменить длину ГС" in source
    assert "Скважины для изменения длины ГС" in source
    assert "Новая длина ГС, м" in source
    assert '"wt_preprocess_select_all"' in source
    assert '"wt_preprocess_only_pad"' in source
    assert '"Применить"' in source


def test_ptc_core_keeps_explicit_pilot_and_zbs_name_matching_guidance() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "`well_01` или `well_01_2` и `well_01_PL`" in source
    assert "Обе записи загружаются вместе." in source
    assert "`Пилот от ГС` / `ГС от пилота`" in source
    assert "точку `S` можно задать явно или опустить" in source
    assert "Можно также использовать имя `fact_01_2`." in source


def test_ptc_core_keeps_auto_order_guardrails_for_source_defined_wellheads() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "Разрешить редактирование позиций куста" in source
    assert "Применить авто-порядок" in source
    assert "disabled=not allow_source_surface_edit" in source
    assert "source_surfaces_defined and not allow_source_surface_edit" in source


def test_ptc_core_uses_stateful_pad_layout_details_panel() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert 'key="wt_pad_layout_details_toggle"' in source
    assert "Скрыть настройки выбранного куста" in source
    assert "Настроить положение куста, НДС и расстояние между устьями." in source
    assert "ptc_pad_state.pad_layout_details_open(st.session_state)" in source


def test_ptc_anticollision_params_limit_multiselect_height_via_scoped_container() -> None:
    source = Path("pywp/ptc_anticollision_params.py").read_text(encoding="utf-8")

    assert 'st.container(key=_REFERENCE_UNCERTAINTY_WIDGETS_CONTAINER_KEY)' in source
    assert ".st-key-wt_anticollision_reference_uncertainty_widgets" in source
    assert "max-height: 3.35rem;" in source
    assert "overflow-y: auto;" in source


def test_ptc_page_wraps_reference_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_reference_section_fragment() -> None:" in source
    assert "render_reference_section()" in source
    assert "_render_reference_section_fragment()" in source


def test_ptc_page_wraps_records_overview_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_records_overview_section(records: list[object]) -> None:" in source
    assert "wt._render_records_overview(records=records)" in source
    assert "_render_records_overview_section(records=records)" in source


def test_ptc_page_wraps_raw_records_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_raw_records_section(records: list[object]) -> None:" in source
    assert "wt._render_raw_records_table(records=records)" in source
    assert "_render_raw_records_section(records=records)" in source


def test_ptc_page_extracts_results_section_helper() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_results_section(" in source
    assert "@st.fragment\ndef _render_results_section(" in source
    assert 'st.markdown("## 5. Результаты расчёта")' in source
    assert "render_success_tabs(" in source
    assert "_render_results_section(" in source


def test_ptc_page_run_wraps_run_section_in_fragment() -> None:
    source = Path("pywp/ptc_page_run.py").read_text(encoding="utf-8")

    assert "@st.fragment\ndef render_run_section(*, records: list[object]) -> None:" in source
    assert "def _rerun_app() -> None:" in source
    assert "_rerun_app()" in source


def test_ptc_fragment_sections_use_fragment_scoped_reruns_for_local_ui_updates() -> None:
    core_source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")
    reference_source = Path("pywp/ptc_page_reference.py").read_text(encoding="utf-8")
    run_source = Path("pywp/ptc_page_run.py").read_text(encoding="utf-8")
    results_source = Path("pywp/ptc_page_results.py").read_text(encoding="utf-8")

    assert 'st.rerun(scope="fragment")' in core_source
    assert 'st.rerun(scope="fragment")' in reference_source
    assert 'st.rerun(scope="fragment")' in run_source
    assert 'st.rerun(scope="fragment")' in results_source


def test_parse_reference_sources_requires_explicit_uploaded_welltrack_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    uploaded_file = SimpleNamespace(
        name="uploaded.inc",
        getvalue=lambda: b"WELLTRACK",
    )
    decode_calls: list[str] = []

    monkeypatch.setattr(
        page,
        "_pending_mixed_legacy_reference_sources",
        lambda: None,
    )
    monkeypatch.setattr(
        page,
        "_reference_uploaded_sources_by_kind",
        lambda _uploaded_files: {
            page._REFERENCE_FUND_OPTIONS[0]: [uploaded_file],
            page._REFERENCE_FUND_OPTIONS[1]: [],
        },
    )
    monkeypatch.setattr(
        page.ptc_welltrack_io,
        "decode_welltrack_payload",
        lambda *_args, **_kwargs: decode_calls.append("decode") or "decoded",
    )
    monkeypatch.setattr(
        page,
        "parse_reference_trajectory_welltrack_text",
        lambda *_args, **_kwargs: ("parsed",),
    )

    with pytest.raises(
        ptc_core.WelltrackParseError,
        match="Неподдерживаемый режим импорта фонда",
    ):
        page._parse_reference_sources(
            mode="Неизвестный режим",
            uploaded_files=[uploaded_file],
        )

    assert decode_calls == []


def test_reference_kind_header_aligns_left() -> None:
    captured: dict[str, object] = {}

    class _DummyContainer:
        def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
            captured["body"] = str(body)
            captured["unsafe"] = bool(unsafe_allow_html)

    ptc_page_reference._render_reference_kind_header(_DummyContainer())

    assert "text-align: left" in str(captured["body"])
    assert "text-align: center" not in str(captured["body"])
    assert captured["unsafe"] is True


def test_parse_reference_sources_handles_uploaded_welltrack_only_for_explicit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    uploaded_file = SimpleNamespace(
        name="uploaded.inc",
        getvalue=lambda: b"WELLTRACK",
    )

    monkeypatch.setattr(
        page,
        "_pending_mixed_legacy_reference_sources",
        lambda: None,
    )
    monkeypatch.setattr(
        page,
        "_reference_uploaded_sources_by_kind",
        lambda _uploaded_files: {
            page._REFERENCE_FUND_OPTIONS[0]: [uploaded_file],
            page._REFERENCE_FUND_OPTIONS[1]: [],
        },
    )
    monkeypatch.setattr(
        page.ptc_welltrack_io,
        "decode_welltrack_payload",
        lambda *_args, **_kwargs: "decoded",
    )
    monkeypatch.setattr(
        page,
        "parse_reference_trajectory_welltrack_text",
        lambda payload, *, kind: (f"{kind}:{payload}",),
    )

    parsed = page._parse_reference_sources(
        mode="Загрузить WELLTRACK",
        uploaded_files=[uploaded_file],
    )

    assert parsed[page._REFERENCE_FUND_OPTIONS[0]] == ("actual:decoded",)
    assert parsed[page._REFERENCE_FUND_OPTIONS[1]] == ()


def test_reference_import_migration_preserves_legacy_uploaded_welltrack_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    actual_kind, approved_kind = page._REFERENCE_FUND_OPTIONS
    page.st.session_state[page.reference_state.reference_source_mode_key(actual_kind)] = (
        "Загрузить WELLTRACK"
    )
    page.st.session_state[
        page.reference_state.reference_source_mode_key(approved_kind)
    ] = "Путь к WELLTRACK"
    page.st.session_state[
        page.reference_state.reference_welltrack_path_key(approved_kind)
    ] = "/tmp/approved.inc"

    original_dev_paths = page.reference_state.reference_dev_folder_paths

    def _dev_paths(kind: str) -> tuple[str, ...]:
        if kind == actual_kind:
            raise AssertionError("legacy upload mode should not read .dev paths")
        return original_dev_paths(kind)

    monkeypatch.setattr(
        page.reference_state,
        "reference_dev_folder_paths",
        _dev_paths,
    )

    page._migrate_legacy_reference_import_state()

    assert (
        str(page.st.session_state[page._REFERENCE_IMPORT_MODE_KEY])
        == "Загрузить WELLTRACK"
    )
    assert (
        str(page.st.session_state[page._reference_welltrack_source_path_key(0)])
        == "/tmp/approved.inc"
    )


def test_reference_import_migration_defaults_to_uploaded_welltrack_mode() -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    actual_kind = page._REFERENCE_FUND_OPTIONS[0]
    page.st.session_state[page.reference_state.reference_source_mode_key(actual_kind)] = (
        "Загрузить WELLTRACK"
    )

    page._migrate_legacy_reference_import_state()

    assert (
        str(page.st.session_state[page._REFERENCE_IMPORT_MODE_KEY])
        == "Загрузить WELLTRACK"
    )


def test_ptc_core_keeps_full_rerun_after_successful_target_import() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "label=operation.success_label(elapsed)," in source
    assert "st.rerun()" in source


def test_ptc_page_renders_target_editor_when_all_results_failed() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {
            "Скважина": "WELL-A",
            "Статус": "Ошибка расчета",
            "Проблема": "endpoint miss",
            "Точек": 3,
        },
        {
            "Скважина": "WELL-B",
            "Статус": "Ошибка расчета",
            "Проблема": "endpoint miss",
            "Точек": 3,
        },
    ]
    at.session_state["wt_successes"] = []

    at.run(timeout=120)

    warning_values = [str(widget.value) for widget in at.warning]
    markdown_values = [str(widget.value) for widget in at.markdown]
    assert any(
        "Все выбранные скважины завершились ошибками" in value
        for value in warning_values
    )
    assert any("Исходные точки для правки" in value for value in markdown_values)


def test_ptc_page_shows_recalc_info_when_all_results_are_not_run_after_target_edit() -> (
    None
):
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "Не рассчитана", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-B", "Статус": "Не рассчитана", "Проблема": "", "Точек": 3},
    ]
    at.session_state["wt_successes"] = []
    at.session_state["wt_edit_targets_pending_names"] = ["WELL-A", "WELL-B"]

    at.run(timeout=120)

    info_values = [str(widget.value) for widget in at.info]
    warning_values = [str(widget.value) for widget in at.warning]
    markdown_values = [str(widget.value) for widget in at.markdown]
    assert any("Скважины требуют пересчёта" in value for value in info_values)
    assert not any(
        "Все выбранные скважины завершились ошибками" in value
        for value in warning_values
    )
    assert any("Исходные точки для правки" in value for value in markdown_values)


def test_ptc_page_wraps_reference_well_table_into_expander() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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


def test_ptc_page_renders_approved_reference_well_detail_viewer() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
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
    at.session_state["wt_show_actual_fund_analysis"] = True
    at.session_state["wt_show_approved_fund_analysis"] = True

    at.run(timeout=120)

    expander_labels = {str(widget.label) for widget in at.expander}
    assert "Просмотр загруженных утверждённых проектных скважин" in expander_labels

    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    assert "Просмотр фактической скважины" in selectbox_labels
    assert "Просмотр утвержденной проектной скважины" in selectbox_labels
