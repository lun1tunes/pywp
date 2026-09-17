from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from pywp import ptc_edit_targets
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import Point3D
from pywp.reference_trajectories import ImportedTrajectoryWell


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


def test_records_with_edit_targets_updates_multi_horizontal_points_by_index() -> None:
    record = _record(
        "MULTI",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=2.0),
            WelltrackPoint(x=200.0, y=0.0, z=2000.0, md=3.0),
            WelltrackPoint(x=300.0, y=0.0, z=2020.0, md=4.0),
            WelltrackPoint(x=400.0, y=0.0, z=2020.0, md=5.0),
        ),
    )

    updated_records, updated_names = ptc_edit_targets.records_with_edit_targets(
        [record],
        {
            "MULTI": {
                "points": [
                    {"index": 0, "position": [10.0, 11.0, -5.0]},
                    {"index": 3, "position": [330.0, 13.0, 2025.0]},
                ],
            },
        },
    )

    assert updated_names == ["MULTI"]
    updated = updated_records[0]
    assert updated.points[0].x == pytest.approx(10.0)
    assert updated.points[0].md == pytest.approx(1.0)
    assert updated.points[3].x == pytest.approx(330.0)
    assert updated.points[3].z == pytest.approx(2025.0)
    assert updated.points[4] == record.points[4]


def test_records_with_edit_targets_updates_pilot_points_by_index() -> None:
    record = _record(
        "WELL-A_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=200.0, z=1800.0, md=2.0),
            WelltrackPoint(x=300.0, y=500.0, z=2400.0, md=3.0),
        ),
    )

    updated_records, updated_names = ptc_edit_targets.records_with_edit_targets(
        [record],
        {
            "WELL-A_PL": {
                "points": [
                    {"index": 0, "position": [10.0, 20.0, -5.0]},
                    {"index": 2, "position": [330.0, 530.0, 2410.0]},
                ],
            },
        },
    )

    assert updated_names == ["WELL-A_PL"]
    updated = updated_records[0]
    assert updated.points[0].x == pytest.approx(10.0)
    assert updated.points[0].md == pytest.approx(1.0)
    assert updated.points[2].y == pytest.approx(530.0)
    assert updated.points[2].md == pytest.approx(3.0)


def test_raw_records_editor_changes_builds_indexed_updates_from_coordinate_edits() -> (
    None
):
    records = [
        _record("WELL-A"),
        _record(
            "WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=100.0, y=200.0, z=1800.0, md=2.0),
                WelltrackPoint(x=300.0, y=500.0, z=2400.0, md=3.0),
            ),
        ),
    ]
    edited_rows = [
        {"Скважина": "WELL-A", "Точка": "S", "X, м": 10.0, "Y, м": 11.0, "Z, м": -5.0},
        {"Скважина": "WELL-A", "Точка": "t1", "X, м": 600.0, "Y, м": 800.0, "Z, м": 2400.0},
        {"Скважина": "WELL-A", "Точка": "t3", "X, м": 1510.0, "Y, м": 2010.0, "Z, м": 2510.0},
        {"Скважина": "WELL-A_PL", "Точка": "S", "X, м": 0.0, "Y, м": 0.0, "Z, м": 0.0},
        {"Скважина": "WELL-A_PL", "Точка": "PL1", "X, м": 110.0, "Y, м": 210.0, "Z, м": 1810.0},
        {"Скважина": "WELL-A_PL", "Точка": "PL2", "X, м": 300.0, "Y, м": 500.0, "Z, м": 2400.0},
    ]

    changes = ptc_edit_targets.raw_records_editor_changes(records, edited_rows)

    assert changes == [
        {
            "name": "WELL-A",
            "points": [
                {"index": 0, "position": [10.0, 11.0, -5.0]},
                {"index": 2, "position": [1510.0, 2010.0, 2510.0]},
            ],
        },
        {
            "name": "WELL-A_PL",
            "points": [
                {"index": 1, "position": [110.0, 210.0, 1810.0]},
            ],
        },
    ]


def test_raw_records_editor_changes_rejects_structure_edits() -> None:
    records = [_record("WELL-A")]
    edited_rows = [
        {"Скважина": "WELL-X", "Точка": "S", "X, м": 0.0, "Y, м": 0.0, "Z, м": 0.0},
        {"Скважина": "WELL-A", "Точка": "t1", "X, м": 600.0, "Y, м": 800.0, "Z, м": 2400.0},
        {"Скважина": "WELL-A", "Точка": "t3", "X, м": 1500.0, "Y, м": 2000.0, "Z, м": 2500.0},
    ]

    with pytest.raises(
        ValueError,
        match="нельзя менять столбцы «Скважина» и «Точка»",
    ):
        ptc_edit_targets.raw_records_editor_changes(records, edited_rows)


def test_bulk_horizontal_length_changes_updates_t3_along_t1_t3_vector() -> None:
    record = _record()

    changes, skipped_names = ptc_edit_targets.bulk_horizontal_length_changes(
        [record],
        target_length_m=2000.0,
    )

    assert skipped_names == []
    assert len(changes) == 1
    change = changes[0]
    assert change["name"] == "WELL-A"
    point_change = change["points"][0]
    assert point_change["index"] == 2
    expected_scale = 2000.0 / math.dist(
        (600.0, 800.0, 2400.0),
        (1500.0, 2000.0, 2500.0),
    )
    assert point_change["position"] == pytest.approx(
        [
            600.0 + 900.0 * expected_scale,
            800.0 + 1200.0 * expected_scale,
            2400.0 + 100.0 * expected_scale,
        ]
    )


def test_queue_all_wells_results_focus_marks_pending_flag() -> None:
    session_state: dict[str, object] = {}

    ptc_edit_targets.queue_all_wells_results_focus(session_state)

    assert session_state["wt_pending_all_wells_results_focus"] is True


def test_bulk_horizontal_length_changes_skips_multi_horizontal_but_updates_single_zbs() -> None:
    multi = _record(
        "MULTI",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=2.0),
            WelltrackPoint(x=250.0, y=0.0, z=2000.0, md=3.0),
            WelltrackPoint(x=300.0, y=0.0, z=2020.0, md=4.0),
            WelltrackPoint(x=500.0, y=0.0, z=2020.0, md=5.0),
        ),
    )
    zbs = _record(
        "9010_ZBS",
        points=(
            WelltrackPoint(x=650.0, y=0.0, z=1500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=1500.0, md=2.0),
        ),
    )

    changes, skipped_names = ptc_edit_targets.bulk_horizontal_length_changes(
        [multi, zbs],
        target_length_m=1000.0,
    )

    assert skipped_names == ["MULTI"]
    assert {item["name"] for item in changes} == {"9010_ZBS"}
    by_name = {item["name"]: item for item in changes}
    assert [item["index"] for item in by_name["9010_ZBS"]["points"]] == [1]


def test_bulk_horizontal_length_changes_skips_parent_well_with_pilot() -> None:
    parent = _record(
        "well_04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=2.0),
            WelltrackPoint(x=600.0, y=0.0, z=1200.0, md=3.0),
        ),
    )
    pilot = _record(
        "well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=2.0),
        ),
    )

    changes, skipped_names = ptc_edit_targets.bulk_horizontal_length_changes(
        [parent, pilot],
        target_length_m=900.0,
    )

    assert changes == []
    assert skipped_names == ["well_04", "well_04_PL"]


def test_bulk_horizontal_length_changes_skips_incomplete_and_degenerate_records() -> None:
    incomplete = _record(
        "BROKEN",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=1.0),
        ),
    )
    degenerate_zbs = _record(
        "9010_ZBS",
        points=(
            WelltrackPoint(x=650.0, y=0.0, z=1500.0, md=1.0),
            WelltrackPoint(x=650.0, y=0.0, z=1500.0, md=2.0),
        ),
    )

    changes, skipped_names = ptc_edit_targets.bulk_horizontal_length_changes(
        [incomplete, degenerate_zbs],
        target_length_m=900.0,
    )

    assert changes == []
    assert skipped_names == ["BROKEN", "9010_ZBS"]


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
    assert session_state["wt_edit_targets_applied_source"] == "three_viewer"
    assert session_state["wt_edit_targets_highlight_names"] == ["WELL-A"]
    assert session_state["wt_edit_targets_highlight_points"] == {"WELL-A": [1, 2]}
    assert session_state["wt_pending_selected_names"] == ["WELL-C", "WELL-A"]
    assert session_state["wt_pending_all_wells_results_focus"] is True
    assert session_state["wt_last_anticollision_resolution"] is None
    assert session_state["wt_prepared_well_overrides"] == {}

    changed_record = session_state["wt_records"][0]
    changed_original = session_state["wt_records_original"][0]
    assert changed_record.points[1].x == pytest.approx(610.25)
    assert changed_record.points[2].z == pytest.approx(2502.0)
    assert changed_original.points[1].x == pytest.approx(610.25)


def test_apply_edit_targets_changes_invalidates_pilot_and_parent_together() -> None:
    records = [
        _record("WELL-A"),
        _record(
            "WELL-A_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=200.0, z=1800.0, md=1800.0),
                WelltrackPoint(x=300.0, y=500.0, z=2400.0, md=2600.0),
            ),
        ),
        _record("WELL-B"),
    ]
    cached_analysis = {"pair_cache": {("WELL-A", "WELL-B"): object()}}
    session_state: dict[str, object] = {
        "wt_records": list(records),
        "wt_records_original": list(records),
        "wt_successes": [
            SimpleNamespace(name="WELL-A"),
            SimpleNamespace(name="WELL-A_PL"),
            SimpleNamespace(name="WELL-B"),
        ],
        "wt_summary_rows": [
            {"Скважина": "WELL-A", "Статус": "OK", "Проблема": ""},
            {"Скважина": "WELL-A_PL", "Статус": "OK", "Проблема": ""},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
        ],
        "wt_anticollision_analysis_cache": cached_analysis,
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "WELL-A_PL",
                "points": [
                    {"index": 1, "position": [120.0, 220.0, 1810.0]},
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["WELL-A_PL"]
    assert [item.name for item in session_state["wt_successes"]] == ["WELL-B"]
    assert session_state["wt_summary_rows"] == [
        {"Скважина": "WELL-A", "Статус": "Не рассчитана", "Проблема": ""},
        {"Скважина": "WELL-A_PL", "Статус": "Не рассчитана", "Проблема": ""},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
    ]
    assert session_state["wt_edit_targets_pending_names"] == [
        "WELL-A_PL",
        "WELL-A",
    ]
    assert session_state["wt_edit_targets_highlight_names"] == ["WELL-A_PL"]
    assert session_state["wt_pending_selected_names"] == [
        "WELL-A_PL",
        "WELL-A",
    ]
    assert session_state["wt_edit_targets_highlight_points"] == {"WELL-A_PL": [1]}
    assert session_state["wt_anticollision_analysis_cache"] is cached_analysis


def test_apply_edit_targets_changes_accepts_two_point_pilot_pl1_index() -> None:
    pilot = _record(
        "WELL-A_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=200.0, z=1800.0, md=1800.0),
        ),
    )
    session_state: dict[str, object] = {
        "wt_records": [pilot],
        "wt_records_original": [pilot],
        "wt_successes": [SimpleNamespace(name="WELL-A_PL")],
        "wt_summary_rows": [
            {"Скважина": "WELL-A_PL", "Статус": "OK", "Проблема": ""}
        ],
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "WELL-A_PL",
                "points": [
                    {"index": 1, "position": [125.0, 225.0, 1810.0]},
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["WELL-A_PL"]
    assert session_state["wt_records"][0].points[1].x == pytest.approx(125.0)
    assert session_state["wt_records_original"][0].points[1].z == pytest.approx(
        1810.0
    )
    assert session_state["wt_edit_targets_highlight_points"] == {
        "WELL-A_PL": [1]
    }


def test_apply_edit_targets_changes_accepts_arbitrary_target_sequence_indices() -> None:
    record = _record(
        "SEQUENCE",
        points=tuple(
            WelltrackPoint(
                x=float(index * 100),
                y=0.0,
                z=float(index * 400),
                md=float(index),
            )
            for index in range(7)
        ),
    )
    session_state: dict[str, object] = {
        "wt_records": [record],
        "wt_records_original": [record],
        "wt_successes": [SimpleNamespace(name="SEQUENCE")],
        "wt_summary_rows": [
            {"Скважина": "SEQUENCE", "Статус": "OK", "Проблема": ""}
        ],
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "SEQUENCE",
                "points": [
                    {"index": 2, "position": [225.0, 15.0, 805.0]},
                    {"index": 5, "position": [530.0, 25.0, 2010.0]},
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["SEQUENCE"]
    assert session_state["wt_records"][0].points[2].x == pytest.approx(225.0)
    assert session_state["wt_records"][0].points[5].y == pytest.approx(25.0)
    assert session_state["wt_records_original"][0].points[5].z == pytest.approx(
        2010.0
    )


def test_apply_edit_targets_changes_accepts_multi_horizontal_point_payload() -> None:
    record = _record(
        "MULTI",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=2000.0, md=2.0),
            WelltrackPoint(x=200.0, y=0.0, z=2000.0, md=3.0),
            WelltrackPoint(x=300.0, y=0.0, z=2020.0, md=4.0),
            WelltrackPoint(x=400.0, y=0.0, z=2020.0, md=5.0),
        ),
    )
    session_state: dict[str, object] = {
        "wt_records": [record],
        "wt_records_original": [record],
        "wt_successes": [SimpleNamespace(name="MULTI")],
        "wt_summary_rows": [{"Скважина": "MULTI", "Статус": "OK", "Проблема": ""}],
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "MULTI",
                "points": [
                    {"index": 0, "position": [1.0, 2.0, -10.0]},
                    {"index": 4, "position": [410.0, 20.0, 2022.0]},
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["MULTI"]
    assert session_state["wt_edit_targets_highlight_points"] == {"MULTI": [0, 4]}
    assert session_state["wt_successes"] == []
    assert session_state["wt_summary_rows"] == [
        {"Скважина": "MULTI", "Статус": "Не рассчитана", "Проблема": ""}
    ]
    updated_record = session_state["wt_records"][0]
    updated_original = session_state["wt_records_original"][0]
    assert updated_record.points[0].z == pytest.approx(-10.0)
    assert updated_record.points[4].x == pytest.approx(410.0)
    assert updated_original.points[4].y == pytest.approx(20.0)


def test_apply_edit_targets_changes_preserves_laid_out_dev_surface_in_records_but_keeps_original_stick() -> (
    None
):
    current_record = WelltrackRecord(
        name="9201",
        points=(
            WelltrackPoint(x=1400.0, y=950.0, z=0.0, md=0.0),
            WelltrackPoint(x=1000.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1900.0, y=2000.0, z=2500.0, md=3500.0),
        ),
    )
    original_record = WelltrackRecord(
        name="9201",
        points=(
            WelltrackPoint(x=1000.0, y=800.0, z=0.0, md=0.0),
            WelltrackPoint(x=1000.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1900.0, y=2000.0, z=2500.0, md=3500.0),
        ),
    )
    session_state: dict[str, object] = {
        "wt_records": [current_record],
        "wt_records_original": [original_record],
        "wt_successes": [SimpleNamespace(name="9201")],
        "wt_summary_rows": [{"Скважина": "9201", "Статус": "OK", "Проблема": ""}],
        "wt_imported_dev_target_wells": (
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
        ),
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "9201",
                "t1": [1110.0, 910.0, 2405.0],
                "t3": [2010.0, 2110.0, 2505.0],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["9201"]
    updated_record = session_state["wt_records"][0]
    updated_original = session_state["wt_records_original"][0]
    assert updated_record.points[0].x == pytest.approx(1400.0)
    assert updated_record.points[0].y == pytest.approx(950.0)
    assert updated_record.points[1].x == pytest.approx(1110.0)
    assert updated_record.points[1].y == pytest.approx(910.0)
    assert updated_original.points[0].x == pytest.approx(1110.0)
    assert updated_original.points[0].y == pytest.approx(910.0)
    assert updated_original.points[0].z == pytest.approx(0.0)
    assert updated_original.points[1].x == pytest.approx(1110.0)
    assert updated_original.points[1].y == pytest.approx(910.0)


def test_apply_edit_targets_changes_accepts_sidetrack_multi_horizontal_indices() -> None:
    record = _record(
        "WELL-04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=800.0, y=0.0, z=2200.0, md=2.0),
            WelltrackPoint(x=1800.0, y=0.0, z=2200.0, md=3.0),
            WelltrackPoint(x=2800.0, y=0.0, z=2220.0, md=4.0),
            WelltrackPoint(x=3400.0, y=0.0, z=2220.0, md=5.0),
        ),
    )
    session_state: dict[str, object] = {
        "wt_records": [record],
        "wt_records_original": [record],
        "wt_successes": [SimpleNamespace(name="WELL-04")],
        "wt_summary_rows": [{"Скважина": "WELL-04", "Статус": "OK", "Проблема": ""}],
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "WELL-04",
                "points": [
                    {"index": 1, "position": [810.0, 5.0, 2201.0]},
                    {"index": 4, "position": [3410.0, 8.0, 2222.0]},
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["WELL-04"]
    assert session_state["wt_edit_targets_highlight_points"] == {"WELL-04": [1, 4]}
    updated_record = session_state["wt_records"][0]
    assert updated_record.points[0] == record.points[0]
    assert updated_record.points[1].x == pytest.approx(810.0)
    assert updated_record.points[1].md == pytest.approx(2.0)
    assert updated_record.points[4].z == pytest.approx(2222.0)


def test_apply_edit_targets_changes_accepts_multi_horizontal_zbs_indices() -> None:
    record = _record(
        "9010_ZBS",
        points=(
            WelltrackPoint(x=650.0, y=0.0, z=1500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=1500.0, md=2.0),
            WelltrackPoint(x=2200.0, y=0.0, z=1520.0, md=3.0),
            WelltrackPoint(x=2800.0, y=0.0, z=1520.0, md=4.0),
        ),
    )
    session_state: dict[str, object] = {
        "wt_records": [record],
        "wt_records_original": [record],
        "wt_successes": [SimpleNamespace(name="9010_ZBS")],
        "wt_summary_rows": [{"Скважина": "9010_ZBS", "Статус": "OK", "Проблема": ""}],
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "9010_ZBS",
                "points": [
                    {"index": 0, "position": [660.0, 1.0, 1501.0]},
                    {"index": 3, "position": [2810.0, 3.0, 1522.0]},
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["9010_ZBS"]
    assert session_state["wt_edit_targets_highlight_points"] == {"9010_ZBS": [0, 3]}
    updated_record = session_state["wt_records"][0]
    assert updated_record.points[0].x == pytest.approx(660.0)
    assert updated_record.points[0].md == pytest.approx(1.0)
    assert updated_record.points[3].z == pytest.approx(1522.0)


def test_apply_edit_targets_changes_accepts_two_point_zbs_indices() -> None:
    record = _record(
        "9010_ZBS",
        points=(
            WelltrackPoint(x=650.0, y=0.0, z=1500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=1500.0, md=2.0),
        ),
    )
    session_state: dict[str, object] = {
        "wt_records": [record],
        "wt_records_original": [record],
        "wt_successes": [SimpleNamespace(name="9010_ZBS")],
        "wt_summary_rows": [{"Скважина": "9010_ZBS", "Статус": "OK", "Проблема": ""}],
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "9010_ZBS",
                "points": [
                    {"index": 0, "position": [660.0, 1.0, 1501.0]},
                    {"index": 1, "position": [1210.0, 3.0, 1502.0]},
                ],
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["9010_ZBS"]
    assert session_state["wt_edit_targets_highlight_points"] == {"9010_ZBS": [0, 1]}
    updated_record = session_state["wt_records"][0]
    assert updated_record.points[0].x == pytest.approx(660.0)
    assert updated_record.points[0].md == pytest.approx(1.0)
    assert updated_record.points[1].x == pytest.approx(1210.0)
    assert updated_record.points[1].md == pytest.approx(2.0)


def test_apply_edit_targets_changes_queues_sidetrack_window_override() -> None:
    records = [_record("9010_ZBS"), _record("WELL-B")]
    session_state: dict[str, object] = {
        "wt_records": list(records),
        "wt_records_original": list(records),
        "wt_successes": [
            SimpleNamespace(name="9010_ZBS"),
            SimpleNamespace(name="WELL-B"),
        ],
        "wt_summary_rows": [
            {"Скважина": "9010_ZBS", "Статус": "OK", "Проблема": ""},
            {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
        ],
    }

    updated_names = ptc_edit_targets.apply_edit_targets_changes(
        session_state,
        [
            {
                "name": "9010_ZBS",
                "sidetrack_window": {
                    "kind": "md",
                    "value_m": 1240.5,
                    "position": [10.0, 20.0, 1200.0],
                },
            }
        ],
        source="three_viewer",
        base_row_factory=_base_row,
    )

    assert updated_names == ["9010_ZBS"]
    assert session_state["wt_records"] == records
    assert session_state["wt_records_original"] == records
    assert [item.name for item in session_state["wt_successes"]] == ["WELL-B"]
    assert session_state["wt_summary_rows"] == [
        {"Скважина": "9010_ZBS", "Статус": "Не рассчитана", "Проблема": ""},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": ""},
    ]
    assert session_state["wt_sidetrack_window_editor_overrides"] == {
        "9010_ZBS": {"kind": "MD", "value_m": 1240.5}
    }
    assert session_state["wt_edit_targets_pending_names"] == ["9010_ZBS"]
    assert session_state["wt_edit_targets_highlight_names"] == []
    assert session_state["wt_edit_targets_highlight_points"] == {}


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
    assert session_state["wt_three_edit_ack"] == {
        "nonce": "nonce-1", "status": "applied"
    }
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


def test_three_edit_noop_is_acknowledged_once_without_geometry_refresh() -> None:
    state = {}
    applied = []
    event = {"type": "pywp:editTargets", "nonce": "noop", "changes": []}

    def apply(changes, source):
        applied.append(changes)
        return []

    def bump():
        raise AssertionError("no-op must not invalidate the scene")

    assert ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=apply, bump_three_viewer_nonce=bump
    )
    assert state["wt_three_edit_ack"] == {"nonce": "noop", "status": "noop"}
    assert not ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=apply, bump_three_viewer_nonce=bump
    )
    assert applied == [[]]


def test_three_edit_failure_is_acknowledged_without_retrying_same_operation() -> None:
    state = {}
    attempts = []
    event = {"type": "pywp:editTargets", "nonce": "failed", "changes": []}

    def apply(changes, source):
        attempts.append(changes)
        raise ValueError("invalid edit")

    assert ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=apply, bump_three_viewer_nonce=lambda: None
    )
    assert state["wt_three_edit_ack"]["nonce"] == "failed"
    assert state["wt_three_edit_ack"]["status"] == "error"
    assert not ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=apply, bump_three_viewer_nonce=lambda: None
    )
    assert attempts == [[]]


def test_three_edit_replays_ack_but_never_reapplies_an_older_operation() -> None:
    state = {}
    applied = []

    def apply(changes, source):
        applied.append(changes)
        return ["WELL-A"]

    def deliver(nonce):
        return ptc_edit_targets.handle_three_edit_event(
            state, {"type": "pywp:editTargets", "nonce": nonce, "changes": [nonce]},
            apply_changes=apply, bump_three_viewer_nonce=lambda: None,
        )

    assert deliver("first")
    assert deliver("second")
    assert deliver("first")
    assert not deliver("first")
    assert applied == [["first"], ["second"]]
    assert state["wt_three_edit_ack"] == {"nonce": "first", "status": "applied"}


def test_three_edit_acknowledgements_are_scoped_to_their_viewers() -> None:
    state = {}

    def deliver(viewer):
        return ptc_edit_targets.handle_three_edit_event(
            state, {"type": "pywp:editTargets", "nonce": viewer, "changes": []},
            apply_changes=lambda *_args: [], bump_three_viewer_nonce=lambda: None,
            ack_key=f"ack:{viewer}",
        )

    assert deliver("first-viewer")
    assert deliver("second-viewer")
    for _ in range(3):
        assert not deliver("first-viewer")
        assert not deliver("second-viewer")
    assert state["ack:first-viewer"]["nonce"] == "first-viewer"
    assert state["ack:second-viewer"]["nonce"] == "second-viewer"


def test_handle_three_edit_event_accepts_pad_changes() -> None:
    session_state: dict[str, object] = {}
    applied: list[tuple[object, str]] = []
    pad_applied: list[tuple[object, str]] = []
    bumped = 0

    def apply_changes(changes: object, source: str) -> list[str]:
        applied.append((changes, source))
        return []

    def apply_pad_changes(changes: object, source: str) -> list[str]:
        pad_applied.append((changes, source))
        return ["PAD1-A", "PAD1-B"]

    def bump_nonce() -> None:
        nonlocal bumped
        bumped += 1

    event = {
        "type": "pywp:editTargets",
        "nonce": "pad-1",
        "pad_changes": [{"pad_id": "PAD-1", "anchor": [10.0, 20.0, 0.0]}],
    }

    assert ptc_edit_targets.handle_three_edit_event(
        session_state,
        event,
        apply_changes=apply_changes,
        apply_pad_changes=apply_pad_changes,
        bump_three_viewer_nonce=bump_nonce,
    )
    assert session_state["wt_last_edit_targets_nonce"] == "pad-1"
    assert applied == [(None, "three_viewer")]
    assert pad_applied == [
        ([{"pad_id": "PAD-1", "anchor": [10.0, 20.0, 0.0]}], "three_viewer")
    ]
    assert bumped == 1


def test_three_edit_transaction_does_not_publish_targets_when_pad_stage_fails() -> None:
    records = [_record(), _record("WELL-B")]
    untouched_cache = {"expensive": object()}
    state = {
        "wt_records": records,
        "wt_records_original": records,
        "wt_anticollision_analysis_cache": untouched_cache,
    }
    before = dict(state)

    def fail_pad(staged, changes, source):
        assert staged["wt_records"][0].points[2].x == 1600
        assert state["wt_records"] is records
        raise ValueError("pad planning failed")

    with pytest.raises(ValueError, match="pad planning failed"):
        ptc_edit_targets.apply_three_edit_transaction(
            state,
            [{"name": "WELL-A", "points": [{"index": 2, "position": [1600, 2000, 2500]}]}],
            [], source="three_viewer", base_row_factory=_base_row,
            apply_pad_changes=fail_pad,
        )
    assert state.keys() == before.keys()
    assert all(state[key] is value for key, value in before.items())


def test_three_edit_transaction_commit_preserves_untouched_records_and_cache() -> None:
    records = [_record(), _record("WELL-B")]
    cache = {"result": object()}
    state = {"wt_records": records, "wt_records_original": records,
             "wt_anticollision_analysis_cache": cache}
    updated = ptc_edit_targets.apply_three_edit_transaction(
        state,
        [{"name": "WELL-A", "points": [{"index": 2, "position": [1600, 2000, 2500]}]}],
        [], source="three_viewer", base_row_factory=_base_row,
        apply_pad_changes=lambda *_args: [],
    )
    assert updated == ["WELL-A"]
    assert state["wt_records"][0].points[2].x == 1600
    assert state["wt_records"][1] is records[1]
    assert state["wt_anticollision_analysis_cache"] is cache
    assert records[0].points[2].x == 1500


@pytest.mark.parametrize("bad_change", [
    {"name": "missing", "t1": [0, 0, 0], "t3": [1, 1, 1]},
    {"name": "WELL-A", "points": [{"index": 99, "position": [1, 1, 1]}]},
    {"name": "WELL-A", "points": [{"index": 2, "position": [1, float("nan"), 1]}]},
    {"name": "WELL-A", "sidetrack_window": {"kind": "md", "value_m": "bad"}},
    {"name": "WELL-A"},
])
def test_three_edit_transaction_rejects_partial_invalid_changes(bad_change) -> None:
    records = [_record()]
    state = {"wt_records": records}
    with pytest.raises(ValueError):
        ptc_edit_targets.apply_three_edit_transaction(
            state, [bad_change], [], source="three_viewer",
            base_row_factory=_base_row, apply_pad_changes=lambda *_args: [],
        )
    assert state == {"wt_records": records}
    assert state["wt_records"] is records


def test_three_edit_transaction_rolls_back_if_session_commit_rejects_a_key() -> None:
    class RejectOnce(dict):
        rejected = False

        def __setitem__(self, key, value):
            if key == "wt_edit_targets_pending_names" and not self.rejected:
                self.rejected = True
                raise RuntimeError("session write rejected")
            super().__setitem__(key, value)

    records = [_record()]
    state = RejectOnce(wt_records=records, wt_records_original=records)
    before = dict(state)
    with pytest.raises(RuntimeError, match="session write rejected"):
        ptc_edit_targets.apply_three_edit_transaction(
            state,
            [{"name": "WELL-A", "points": [{"index": 2, "position": [1600, 2000, 2500]}]}],
            [], source="three_viewer", base_row_factory=_base_row,
            apply_pad_changes=lambda *_args: [],
        )
    assert state.rejected
    assert state.keys() == before.keys()
    assert all(state[key] is value for key, value in before.items())
