from __future__ import annotations

from types import SimpleNamespace

import pytest

from pywp import ptc_edit_targets
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord


def _record(
    name: str = "WELL-A",
    *,
    points: tuple[WelltrackPoint, ...] | None = None,
) -> WelltrackRecord:
    return WelltrackRecord(
        name=name,
        points=points
        or (
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
        ),
    )


def _base_row(record: WelltrackRecord) -> dict[str, object]:
    return {"Скважина": record.name, "Статус": "Не рассчитана", "Проблема": ""}


def test_edit_target_point_accepts_only_finite_xyz_values() -> None:
    assert ptc_edit_targets.edit_target_point(["1.5", 2, 3.25, 99]) == [
        1.5,
        2.0,
        3.25,
    ]
    assert ptc_edit_targets.edit_target_point([1.0, 2.0]) is None
    assert ptc_edit_targets.edit_target_point([1.0, "bad", 3.0]) is None
    assert ptc_edit_targets.edit_target_point([1.0, float("nan"), 3.0]) is None


def test_records_with_edit_targets_updates_only_three_point_records() -> None:
    incomplete = _record(
        "WELL-B",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=10.0, y=20.0, z=30.0, md=40.0),
        ),
    )

    updated_records, updated_names = ptc_edit_targets.records_with_edit_targets(
        [_record(), incomplete],
        {
            "WELL-A": {
                "t1": [610.25, 805.5, 2401.0],
                "t3": [1510.75, 2010.25, 2502.0],
            },
            "WELL-B": {"t1": [1.0, 2.0, 3.0], "t3": [4.0, 5.0, 6.0]},
        },
    )

    assert updated_names == ["WELL-A"]
    assert updated_records[0].points[1].x == pytest.approx(610.25)
    assert updated_records[0].points[1].md == pytest.approx(2400.0)
    assert updated_records[0].points[2].y == pytest.approx(2010.25)
    assert updated_records[0].points[2].md == pytest.approx(3500.0)
    assert updated_records[1] is incomplete


def test_pending_edit_target_names_prefers_pending_and_dedupes() -> None:
    session_state: dict[str, object] = {
        "wt_edit_targets_pending_names": [" WELL-A ", "WELL-A", "", "WELL-B"],
        "wt_edit_targets_highlight_names": ["WELL-C"],
    }

    assert ptc_edit_targets.pending_edit_target_names(session_state) == [
        "WELL-A",
        "WELL-B",
    ]

    session_state["wt_edit_targets_pending_names"] = []

    assert ptc_edit_targets.pending_edit_target_names(session_state) == ["WELL-C"]


def test_apply_edit_targets_changes_invalidates_only_changed_wells() -> None:
    records = [_record("WELL-A"), _record("WELL-B")]
    session_state: dict[str, object] = {
        "wt_records": list(records),
        "wt_records_original": list(records),
        "wt_successes": [
            SimpleNamespace(name="WELL-A"),
            SimpleNamespace(name="WELL-B"),
        ],
        "wt_summary_rows": [
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
        ],
        "wt_edit_targets_pending_names": ["WELL-C"],
        "wt_last_anticollision_resolution": object(),
        "wt_last_anticollision_previous_successes": {"WELL-A": object()},
        "wt_prepared_well_overrides": {"WELL-A": object()},
        "wt_prepared_override_message": "prepared",
        "wt_prepared_recommendation_id": "rec",
        "wt_anticollision_prepared_cluster_id": "cluster",
        "wt_prepared_recommendation_snapshot": object(),
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "WELL-A",
                "t1": [610.25, 805.5, 2401.0],
                "t3": [1510.75, 2010.25, 2502.0],
            },
            {"name": "WELL-X", "t1": [1.0, 2.0, 3.0], "t3": [4.0, 5.0, 6.0]},
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["WELL-A"]
    assert [item.name for item in session_state["wt_successes"]] == ["WELL-B"]
    assert session_state["wt_summary_rows"] == [
        {"Скважина": "WELL-A", "Статус": "Не рассчитана", "Проблема": ""},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
    ]
    assert session_state["wt_edit_targets_pending_names"] == ["WELL-C", "WELL-A"]
    assert session_state["wt_edit_targets_highlight_names"] == ["WELL-C", "WELL-A"]
    assert session_state["wt_pending_selected_names"] == ["WELL-C", "WELL-A"]
    assert session_state["wt_pending_all_wells_results_focus"] is True
    assert session_state["wt_last_anticollision_resolution"] is None
    assert session_state["wt_prepared_well_overrides"] == {}

    changed_record = session_state["wt_records"][0]
    changed_original = session_state["wt_records_original"][0]
    assert changed_record.points[1].x == pytest.approx(610.25)
    assert changed_record.points[2].z == pytest.approx(2502.0)
    assert changed_original.points[1].x == pytest.approx(610.25)


def test_handle_three_edit_event_ignores_duplicate_nonce() -> None:
    session_state: dict[str, object] = {}
    applied: list[tuple[object, str]] = []
    bumped = 0

    def apply_changes(changes: object, source: str) -> list[str]:
        applied.append((changes, source))
        return ["WELL-A"]

    def bump_nonce() -> None:
        nonlocal bumped
        bumped += 1

    event = {
        "type": "pywp:editTargets",
        "nonce": "nonce-1",
        "changes": [{"name": "WELL-A"}],
    }

    assert ptc_edit_targets.handle_three_edit_event(
        session_state,
        event,
        apply_changes=apply_changes,
        bump_three_viewer_nonce=bump_nonce,
    )
    assert session_state["wt_last_edit_targets_nonce"] == "nonce-1"
    assert applied == [([{"name": "WELL-A"}], "three_viewer")]
    assert bumped == 1

    assert not ptc_edit_targets.handle_three_edit_event(
        session_state,
        event,
        apply_changes=apply_changes,
        bump_three_viewer_nonce=bump_nonce,
    )
    assert len(applied) == 1
    assert bumped == 1
