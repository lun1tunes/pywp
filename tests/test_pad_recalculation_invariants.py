"""Pad preparation regression and real solver/cache integration contracts."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pywp import ptc_batch_run, ptc_core
from pywp.ptc_pad_state import WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import TrajectoryConfig
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL
from pywp.welltrack_batch import WelltrackBatchPlanner


def _records():
    return [
        WelltrackRecord(
            name=f"PAD-{pad}-{slot + 1:02d}",
            points=tuple(
                WelltrackPoint(x=x + offset, y=y + slot * 50, z=z, md=md)
                for x, y, z, md in (
                    (0, 0, 0, 0),
                    (600, 800, 2400, 2400),
                    (1500, 2000, 2500, 3500),
                )
            ),
        )
        for pad, offset in (("A", 0), ("B", 10000), ("C", 20000))
        for slot in range(2)
    ]


def _state(monkeypatch, records):
    state = {
        "wt_records": list(records),
        "wt_records_original": list(records),
        "wt_results_view_mode": "Все скважины",
        "wt_results_all_view_mode": "Anti-collision",
    }
    monkeypatch.setattr(ptc_core.st, "session_state", state)
    ptc_core._ensure_pad_configs(records)
    state["wt_pad_selected_id"] = "PAD-B"
    return state


def _run(state, names, config):
    # The calculation, state hooks, records and cache remain production code.
    ui = MagicMock()
    ui.session_state = state
    ptc_batch_run.run_batch_if_clicked(
        requests=[
            ptc_batch_run.BatchRunRequest(names, config, True, parallel_workers=0)
        ],
        records=state["wt_records"],
        hooks=ptc_core._batch_run_hooks(),
        st_module=ui,
    )
    assert state.get("wt_last_error", "") == "", state.get("wt_last_error")


@pytest.mark.parametrize("with_working_target", [True, False])
def test_batch_preparation_preserves_unselected_working_records_and_pilot(
    monkeypatch, with_working_target
):
    records = _records()
    # A partial/custom project may contain a pilot with its own surface. A
    # recalculation of B must not silently repair the unrelated C construction.
    pilot = WelltrackRecord(
        name="PAD-C-01_PL",
        points=(
            WelltrackPoint(x=20123, y=17, z=0, md=0),
            WelltrackPoint(x=20200, y=50, z=800, md=800),
        ),
    )
    state = _state(monkeypatch, [*records, pilot])
    # Model a previously committed working geometry distinct from import.
    edited_c = records[-1].model_copy(
        update={
            "points": (
                records[-1].points[0],
                records[-1].points[1],
                records[-1]
                .points[2]
                .model_copy(update={"x": records[-1].points[2].x + 15}),
            )
        }
    )
    if with_working_target:
        state["wt_records"][-2] = edited_c
    state["wt_pad_last_applied_at"] = "2026-09-15"
    before = list(state["wt_records"])
    captured = []

    class CaptureBatch:
        last_evaluation_metadata = SimpleNamespace(skipped_selected_names=())

        def __init__(self, **kwargs):
            pass

        def evaluate(self, *, records, **kwargs):
            captured.extend(records)
            return [], []

    monkeypatch.setattr(ptc_batch_run, "WelltrackBatchPlanner", CaptureBatch)
    _run(state, ["PAD-B-01"], TrajectoryConfig())
    assert [r.name for r in captured] == [r.name for r in before]
    after = {r.name: r for r in captured}
    if with_working_target:
        assert after[edited_c.name] == edited_c, (
            "batch preparation replaced working targets with import"
        )
    assert after[pilot.name] == pilot, "batch preparation moved an unrelated pilot"
    for record in before:
        if not record.name.startswith("PAD-B"):
            assert after[record.name] is record


@pytest.mark.integration
def test_three_pad_apply_real_recalculation_and_incremental_anticollision(monkeypatch):
    state = _state(monkeypatch, _records())
    config = TrajectoryConfig(
        dls_build_max_deg_per_30m=6,
        max_total_md_postcheck_m=20000,
        turn_solver_max_restarts=0,
    )
    names = [record.name for record in state["wt_records"]]
    _run(state, names, config)
    assert [success.name for success in state["wt_successes"]] == names
    before_success = {s.name: s for s in state["wt_successes"]}
    before_stations = {
        name: s.stations.copy(deep=True) for name, s in before_success.items()
    }
    before_rows = {row["Скважина"]: deepcopy(row) for row in state["wt_summary_rows"]}
    before_records = {r.name: r for r in state["wt_records"]}
    before_configs = deepcopy(state["wt_pad_configs"])
    original_records = state["wt_records_original"]
    ptc_core._cached_anti_collision_view_model(
        successes=state["wt_successes"],
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        records=state["wt_records"],
        parallel_workers=0,
    )
    cache = state["wt_anticollision_analysis_cache"]
    old_pairs = dict(cache["pair_cache"])
    old_wells = dict(cache["well_cache"])
    assert len(old_wells) == 6 and len(old_pairs) == 15
    cfg = before_configs["PAD-B"]
    event = {
        "type": "pywp:editTargets",
        "nonce": "move-pad-b",
        "changes": [],
        "pad_changes": [
            {
                "pad_id": "PAD-B",
                "anchor": [
                    cfg["first_surface_x"] + 10,
                    cfg["first_surface_y"] + 5,
                    cfg["first_surface_z"],
                ],
            }
        ],
    }
    assert ptc_core._handle_three_edit_event(event)
    assert state["wt_three_edit_ack"]["status"] == "applied"
    assert state["wt_anticollision_analysis_cache"] is cache
    assert state["wt_records_original"] is original_records
    changed = ["PAD-B-01", "PAD-B-02"]
    assert state["wt_edit_targets_pending_names"] == changed
    unaffected = [name for name in names if name not in changed]
    assert [s.name for s in state["wt_successes"]] == unaffected
    assert state["wt_pad_configs"]["PAD-A"] == before_configs["PAD-A"]
    assert state["wt_pad_configs"]["PAD-C"] == before_configs["PAD-C"]
    assert state["wt_pad_selected_id"] == "PAD-B"

    executed = []
    real_evaluate = WelltrackBatchPlanner.evaluate

    def observe_evaluate(self, *args, **kwargs):
        result = real_evaluate(self, *args, **kwargs)
        executed.extend(self.last_evaluation_metadata.executed_well_names)
        return result

    monkeypatch.setattr(WelltrackBatchPlanner, "evaluate", observe_evaluate)
    _run(state, changed, config)
    assert executed == changed
    assert [s.name for s in state["wt_successes"]] == names
    assert [r.name for r in state["wt_records"]] == names
    assert state["wt_edit_targets_pending_names"] == []
    assert state["wt_anticollision_analysis_cache"] is cache
    after_success = {s.name: s for s in state["wt_successes"]}
    after_records = {r.name: r for r in state["wt_records"]}
    after_rows = {r["Скважина"]: r for r in state["wt_summary_rows"]}
    for name in unaffected:
        assert after_records[name] is before_records[name]
        assert after_success[name] is before_success[name]
        assert after_rows[name] == before_rows[name]
        pd.testing.assert_frame_equal(
            after_success[name].stations, before_stations[name], check_exact=True
        )
    for name in changed:
        assert after_success[name] is not before_success[name]
        assert np.all(np.diff(after_success[name].stations.MD_m) > 0)

    analysis, _, _ = ptc_core._cached_anti_collision_view_model(
        successes=state["wt_successes"],
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        records=state["wt_records"],
        parallel_workers=0,
    )
    stats = state["wt_anticollision_last_run"]
    assert stats["reused_well_count"] == 4 and stats["rebuilt_well_count"] == 2
    assert stats["reused_pair_count"] == 6 and stats["recalculated_pair_count"] == 9
    refreshed_cache = state["wt_anticollision_analysis_cache"]
    for name in unaffected:
        assert refreshed_cache["well_cache"][name][1] is old_wells[name][1]
    for pair, entry in old_pairs.items():
        if set(pair).isdisjoint(changed):
            assert refreshed_cache["pair_cache"][pair] is entry
    # Same operation and a fresh no-op must not recalculate or clear results.
    assert not ptc_core._handle_three_edit_event(event)
    successes_before_noop = state["wt_successes"]
    assert ptc_core._handle_three_edit_event({**event, "nonce": "noop-pad-b"})
    assert state["wt_three_edit_ack"]["status"] == "noop"
    assert state["wt_successes"] is successes_before_noop
    assert state["wt_anticollision_analysis_cache"] is refreshed_cache
    repeat, _, _ = ptc_core._cached_anti_collision_view_model(
        successes=state["wt_successes"],
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        records=state["wt_records"],
        parallel_workers=0,
    )
    assert repeat is analysis and state["wt_anticollision_last_run"]["cached"]


@pytest.mark.parametrize("has_import_snapshot", [True, False])
@pytest.mark.parametrize("selected_kind", ["parent", "pilot"])
def test_partial_selection_moves_only_selected_parent_and_its_pilot(
    monkeypatch, has_import_snapshot, selected_kind
):
    records = _records()
    pilot = WelltrackRecord(
        name="PAD-B-01_PL",
        points=(
            WelltrackPoint(x=10000, y=0, z=0, md=0),
            WelltrackPoint(x=10000, y=0, z=800, md=800),
        ),
    )
    state = _state(monkeypatch, [*records, pilot])
    if not has_import_snapshot:
        state.pop("wt_records_original")
    cfg = state["wt_pad_configs"]["PAD-B"]
    cfg[WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY] = True
    cfg["first_surface_x"] += 15
    state["wt_pad_last_applied_at"] = "2026-09-15"
    before = {r.name: r for r in state["wt_records"]}
    captured = []
    planned_pads = []
    real_plan_map = ptc_core._build_pad_plan_map

    def observe_plans(pads):
        planned_pads.extend(pad.pad_id for pad in pads)
        return real_plan_map(pads)

    class CaptureBatch:
        last_evaluation_metadata = SimpleNamespace(skipped_selected_names=())

        def __init__(self, **kwargs):
            pass

        def evaluate(self, *, records, **kwargs):
            captured.extend(records)
            return [], []

    monkeypatch.setattr(ptc_batch_run, "WelltrackBatchPlanner", CaptureBatch)
    monkeypatch.setattr(ptc_core, "_build_pad_plan_map", observe_plans)
    selected = "PAD-B-01" if selected_kind == "parent" else pilot.name
    _run(state, [selected], TrajectoryConfig())
    after = {r.name: r for r in captured}
    expected_shift = 15 if selected_kind == "parent" else 0
    assert planned_pads == (["PAD-B"] if selected_kind == "parent" else [])
    assert (
        after["PAD-B-01"].points[0].x == before["PAD-B-01"].points[0].x + expected_shift
    )
    assert after[pilot.name].points[0].x == after["PAD-B-01"].points[0].x
    assert after[pilot.name].points[1:] == pilot.points[1:]
    for name in set(before) - {"PAD-B-01", pilot.name}:
        assert after[name] is before[name], "unselected sibling or pad was changed"
