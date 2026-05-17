from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

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


def test_sync_selection_state_hides_pilot_and_maps_old_pilot_selection_to_parent() -> None:
    records = [
        *_records(),
        WelltrackRecord(
            name="WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=80.0, y=0.0, z=800.0, md=800.0),
            ),
        ),
    ]
    state: dict[str, object] = {"wt_selected_names": ["WELL-A_PL"]}

    all_names, recommended_names = ptc_batch_run.sync_selection_state(
        state,
        records=records,
    )

    assert all_names == ["WELL-A", "WELL-B"]
    assert recommended_names == ["WELL-A", "WELL-B"]
    assert state["wt_selected_names"] == ["WELL-A"]


def test_matched_zbs_parent_names_uses_actual_reference_wells() -> None:
    zbs = WelltrackRecord(
        name="9010_ZBS",
        points=(
            WelltrackPoint(x=10.0, y=0.0, z=1200.0, md=1.0),
            WelltrackPoint(x=500.0, y=0.0, z=1200.0, md=2.0),
        ),
    )

    matches = ptc_batch_run._matched_zbs_parent_names(
        records=[*_records(), zbs],
        selected_names={"9010_ZBS"},
        reference_wells=(
            SimpleNamespace(name="9010", kind="actual"),
            SimpleNamespace(name="9010", kind="approved"),
        ),
    )

    assert matches == [("9010_ZBS", "9010")]


def test_batch_selection_status_rolls_pilot_row_into_parent_well() -> None:
    records = [
        *_records(),
        WelltrackRecord(
            name="WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=80.0, y=0.0, z=800.0, md=800.0),
            ),
        ),
    ]

    status = ptc_batch_run.batch_selection_status(
        records=records,
        summary_rows=[
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""},
            {"Скважина": "WELL-A_PL", "Статус": "Ошибка расчета", "Проблема": "bad"},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
        ],
    )

    assert status.error_count == 1
    assert status.ok_count == 1
    assert status.not_run_count == 0


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


def test_store_merged_batch_results_preserves_anticollision_cache_for_incremental_rerun() -> None:
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
    assert "cached" in state["wt_anticollision_analysis_cache"]


def test_run_batch_syncs_pilot_surface_after_active_pad_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = WelltrackRecord(
        name="WELL-A",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1000.0),
            WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=1200.0),
        ),
    )
    pilot = WelltrackRecord(
        name="WELL-A_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=80.0, y=0.0, z=800.0, md=800.0),
        ),
    )
    shifted_parent = WelltrackRecord(
        name="WELL-A",
        points=(
            WelltrackPoint(x=10.0, y=20.0, z=0.0, md=0.0),
            *parent.points[1:],
        ),
    )
    captured_records: list[WelltrackRecord] = []

    def fake_apply_pad_layout(**_kwargs: object) -> list[WelltrackRecord]:
        return [shifted_parent, pilot]

    class FakeBatchPlanner:
        last_evaluation_metadata = SimpleNamespace(
            skipped_selected_names=(),
            cluster_blocked=False,
            cluster_resolved_early=False,
            cluster_blocking_reason=None,
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def evaluate(self, *, records: list[WelltrackRecord], **_kwargs: object):
            captured_records[:] = list(records)
            return ([{"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""}], [])

    state: dict[str, object] = {
        "wt_pad_last_applied_at": "2026-01-01 00:00:00",
        "wt_records_original": [parent, pilot],
        "wt_successes": [],
        "wt_summary_rows": None,
    }
    fake_st = _FakeStreamlit(state)
    hooks = _batch_run_hooks()
    monkeypatch.setattr(ptc_batch_run, "apply_pad_layout", fake_apply_pad_layout)
    monkeypatch.setattr(ptc_batch_run, "WelltrackBatchPlanner", FakeBatchPlanner)

    ptc_batch_run.run_batch_if_clicked(
        requests=[
            ptc_batch_run.BatchRunRequest(
                selected_names=["WELL-A"],
                config=TrajectoryConfig(),
                run_clicked=True,
            )
        ],
        records=[parent, pilot],
        hooks=hooks,
        st_module=fake_st,
    )

    pilot_record = next(
        record for record in captured_records if record.name == "WELL-A_PL"
    )
    assert float(pilot_record.points[0].x) == 10.0
    assert float(pilot_record.points[0].y) == 20.0


def test_run_batch_stores_parallel_worker_count_for_anti_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_parallel_workers: list[int] = []

    class FakeBatchPlanner:
        last_evaluation_metadata = SimpleNamespace(
            skipped_selected_names=(),
            cluster_blocked=False,
            cluster_resolved_early=False,
            cluster_blocking_reason=None,
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def evaluate(self, **kwargs: object):
            captured_parallel_workers.append(int(kwargs["parallel_workers"]))
            return ([{"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""}], [])

    state: dict[str, object] = {
        "wt_successes": [],
        "wt_summary_rows": None,
    }
    fake_st = _FakeStreamlit(state)
    monkeypatch.setattr(ptc_batch_run, "WelltrackBatchPlanner", FakeBatchPlanner)

    ptc_batch_run.run_batch_if_clicked(
        requests=[
            ptc_batch_run.BatchRunRequest(
                selected_names=["WELL-A"],
                config=TrajectoryConfig(),
                run_clicked=True,
                parallel_workers=4,
            )
        ],
        records=_records(),
        hooks=_batch_run_hooks(),
        st_module=fake_st,
    )

    assert captured_parallel_workers == [4]
    assert state["wt_last_parallel_workers"] == 4


def test_run_batch_clears_stale_error_and_recommends_no_followup_after_all_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBatchPlanner:
        last_evaluation_metadata = SimpleNamespace(
            skipped_selected_names=(),
            cluster_blocked=False,
            cluster_resolved_early=False,
            cluster_blocking_reason=None,
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def evaluate(self, **_kwargs: object):
            return (
                [
                    {"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""},
                    {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
                ],
                [_success("WELL-A"), _success("WELL-B")],
            )

    focus_calls: list[str] = []
    state: dict[str, object] = {
        "wt_last_error": "Batch-расчет завершился ошибкой",
        "wt_successes": [],
        "wt_summary_rows": None,
    }
    hooks = ptc_batch_run.BatchRunHooks(
        selected_execution_order=lambda names: list(names),
        pending_edit_target_names=lambda: [],
        ensure_pad_configs=lambda **_kwargs: [object()],
        build_pad_plan_map=lambda _pads: {"pad": object()},
        build_selected_override_configs=lambda **_kwargs: {},
        build_selected_optimization_contexts=lambda **_kwargs: {},
        reference_wells_from_state=lambda: (),
        reference_uncertainty_models_from_state=lambda _reference_wells: {},
        resolution_snapshot_well_names=lambda _snapshot: (),
        format_prepared_override_scope=lambda **_kwargs: [],
        prepared_plan_kind_label=lambda _snapshot: "",
        build_last_anticollision_resolution=lambda **_kwargs: None,
        focus_all_wells_anticollision_results=lambda: focus_calls.append(
            "anticollision"
        ),
        focus_all_wells_trajectory_results=lambda: focus_calls.append("trajectory"),
    )
    monkeypatch.setattr(ptc_batch_run, "WelltrackBatchPlanner", FakeBatchPlanner)

    ptc_batch_run.run_batch_if_clicked(
        requests=[
            ptc_batch_run.BatchRunRequest(
                selected_names=["WELL-A", "WELL-B"],
                config=TrajectoryConfig(),
                run_clicked=True,
            )
        ],
        records=_records(),
        hooks=hooks,
        st_module=_FakeStreamlit(state),
    )

    assert state["wt_last_error"] == ""
    assert state["wt_pending_selected_names"] == []
    assert [str(success.name) for success in state["wt_successes"]] == [
        "WELL-A",
        "WELL-B",
    ]
    assert focus_calls == ["trajectory"]


class _FakePlaceholder:
    def caption(self, *_args: object, **_kwargs: object) -> None:
        pass

    def code(self, *_args: object, **_kwargs: object) -> None:
        pass

    def success(self, *_args: object, **_kwargs: object) -> None:
        pass

    def error(self, *_args: object, **_kwargs: object) -> None:
        pass

    def empty(self) -> None:
        pass


class _FakeProgress(_FakePlaceholder):
    def progress(self, *_args: object, **_kwargs: object) -> None:
        pass


class _FakeSpinner:
    def __enter__(self) -> "_FakeSpinner":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeStreamlit:
    def __init__(self, state: dict[str, object]) -> None:
        self.session_state = state

    def progress(self, *_args: object, **_kwargs: object) -> _FakeProgress:
        return _FakeProgress()

    def empty(self) -> _FakePlaceholder:
        return _FakePlaceholder()

    def spinner(self, *_args: object, **_kwargs: object) -> _FakeSpinner:
        return _FakeSpinner()

    def warning(self, *_args: object, **_kwargs: object) -> None:
        pass

    def error(self, *_args: object, **_kwargs: object) -> None:
        pass


def _batch_run_hooks() -> ptc_batch_run.BatchRunHooks:
    return ptc_batch_run.BatchRunHooks(
        selected_execution_order=lambda names: list(names),
        pending_edit_target_names=lambda: [],
        ensure_pad_configs=lambda **_kwargs: [object()],
        build_pad_plan_map=lambda _pads: {"pad": object()},
        build_selected_override_configs=lambda **_kwargs: {},
        build_selected_optimization_contexts=lambda **_kwargs: {},
        reference_wells_from_state=lambda: (),
        reference_uncertainty_models_from_state=lambda _reference_wells: {},
        resolution_snapshot_well_names=lambda _snapshot: (),
        format_prepared_override_scope=lambda **_kwargs: [],
        prepared_plan_kind_label=lambda _snapshot: "",
        build_last_anticollision_resolution=lambda **_kwargs: None,
        focus_all_wells_anticollision_results=lambda: None,
        focus_all_wells_trajectory_results=lambda: None,
    )
