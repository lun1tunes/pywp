from __future__ import annotations

import pandas as pd

from pywp import ptc_batch_run
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import Point3D, TrajectoryConfig
from pywp.welltrack_batch import SuccessfulWellPlan


def _records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1000.0),
                WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=1200.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=20.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=20.0, z=1000.0, md=1000.0),
                WelltrackPoint(x=200.0, y=20.0, z=1000.0, md=1200.0),
            ),
        ),
    ]


def _success(name: str) -> SuccessfulWellPlan:
    return SuccessfulWellPlan(
        name=name,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(100.0, 0.0, 1000.0),
        t3=Point3D(200.0, 0.0, 1000.0),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0],
                "X_m": [0.0],
                "Y_m": [0.0],
                "Z_m": [0.0],
            }
        ),
        summary={"md_total_m": 0.0},
        azimuth_deg=0.0,
        md_t1_m=0.0,
        config=TrajectoryConfig(),
    )


def test_sync_selection_state_consumes_pending_names_and_filters_missing() -> None:
    state: dict[str, object] = {
        "wt_selected_names": ["OLD"],
        "wt_pending_selected_names": ["WELL-B", "MISSING"],
    }

    all_names, recommended_names = ptc_batch_run.sync_selection_state(
        state,
        records=_records(),
    )

    assert all_names == ["WELL-A", "WELL-B"]
    assert recommended_names == ["WELL-A", "WELL-B"]
    assert state["wt_selected_names"] == ["WELL-B"]
    assert "wt_pending_selected_names" not in state


def test_batch_selection_status_matches_initial_ui_counts() -> None:
    status = ptc_batch_run.batch_selection_status(
        records=_records(),
        summary_rows=None,
    )

    assert status.has_summary_rows is False
    assert status.ok_count == 0
    assert status.warning_count == 0
    assert status.error_count == 0
    assert status.not_run_count == 2


def test_batch_selection_status_ignores_blank_problem_markers() -> None:
    status = ptc_batch_run.batch_selection_status(
        records=_records(),
        summary_rows=[
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": pd.NA},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "<NA>"},
        ],
    )

    assert status.has_summary_rows is True
    assert status.ok_count == 2
    assert status.warning_count == 0


def test_store_merged_batch_results_clears_recalculated_edit_targets() -> None:
    state: dict[str, object] = {
        "wt_summary_rows": [
            {"Скважина": "WELL-A", "Статус": "Не рассчитана", "Проблема": ""},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
        ],
        "wt_successes": [_success("WELL-B")],
        "wt_edit_targets_pending_names": ["WELL-A"],
        "wt_edit_targets_highlight_names": ["WELL-A"],
        "wt_anticollision_analysis_cache": {"cached": object()},
    }

    def pending_names() -> list[str]:
        return list(state.get("wt_edit_targets_pending_names", []))

    ptc_batch_run.store_merged_batch_results(
        state,
        records=_records(),
        new_rows=[{"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""}],
        new_successes=[_success("WELL-A")],
        pending_edit_target_names=pending_names,
    )

    assert [str(row["Скважина"]) for row in state["wt_summary_rows"]] == [
        "WELL-A",
        "WELL-B",
    ]
    assert [success.name for success in state["wt_successes"]] == [
        "WELL-A",
        "WELL-B",
    ]
    assert state["wt_edit_targets_pending_names"] == []
    assert state["wt_edit_targets_highlight_names"] == []
    assert state["wt_anticollision_analysis_cache"] == {}
