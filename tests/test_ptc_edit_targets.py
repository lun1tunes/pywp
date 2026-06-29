from __future__ import annotations

import math
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
