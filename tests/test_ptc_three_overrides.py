from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionWell,
    anti_collision_report_events,
)
from pywp.anticollision_rerun import build_anti_collision_analysis_for_successes
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord, parse_welltrack_text
from pywp.models import Point3D, TrajectoryConfig
from pywp import ptc_pad_state
from pywp import ptc_three_overrides
from pywp.uncertainty import DEFAULT_UNCERTAINTY_PRESET, planning_uncertainty_model_for_preset
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner


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


def _multi_pad_records() -> list[WelltrackRecord]:
    return [
        *_records()[:2],
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


def _successful_plan_xy(
    *,
    name: str,
    x_offset_m: float,
    y_offset_m: float,
    station_count: int = 3,
) -> SuccessfulWellPlan:
    x_values = np.linspace(x_offset_m, x_offset_m + 2000.0, station_count)
    y_values = np.full(station_count, y_offset_m, dtype=float)
    z_values = np.zeros(station_count, dtype=float)
    stations = pd.DataFrame(
        {
            "MD_m": np.linspace(0.0, 2000.0, station_count),
            "INC_deg": np.linspace(0.0, 90.0, station_count),
            "AZI_deg": np.full(station_count, 90.0, dtype=float),
            "X_m": x_values,
            "Y_m": y_values,
            "Z_m": z_values,
            "DLS_deg_per_30m": np.zeros(station_count, dtype=float),
            "segment": np.array(["HOLD"] * station_count, dtype=object),
        }
    )
    return SuccessfulWellPlan(
        name=name,
        surface={"x": x_offset_m, "y": y_offset_m, "z": 0.0},
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


def test_trajectory_overrides_build_tree_focus_targets_for_multi_pad() -> None:
    session_state: dict[str, object] = {}
    records = _multi_pad_records()
    successes = [
        _successful_plan_xy(name="WELL-A", x_offset_m=0.0, y_offset_m=0.0),
        _successful_plan_xy(name="WELL-B", x_offset_m=0.0, y_offset_m=50.0),
        _successful_plan_xy(name="PAD2-A", x_offset_m=5000.0, y_offset_m=0.0),
        _successful_plan_xy(name="PAD2-B", x_offset_m=5000.0, y_offset_m=50.0),
    ]

    overrides = ptc_three_overrides.trajectory_three_payload_overrides(
        session_state,
        records=records,
        successes=successes,
        target_only_wells=[],
        name_to_color={
            "WELL-A": "#22c55e",
            "WELL-B": "#2563eb",
            "PAD2-A": "#f59e0b",
            "PAD2-B": "#c026d3",
        },
    )

    legend_tree = list(overrides["legend_tree"])
    focus_targets = dict(overrides["focus_targets"])
    assert [str(item["label"]) for item in legend_tree] == [
        "Куст PAD-01",
        "Куст PAD-02",
    ]
    assert set(focus_targets) == {
        "pad::PAD-01",
        "pad::PAD-02",
        "well::WELL-A",
        "well::WELL-B",
        "well::PAD2-A",
        "well::PAD2-B",
    }
    assert set(overrides["hidden_flat_legend_labels"]) == {
        "WELL-A",
        "WELL-B",
        "PAD2-A",
        "PAD2-B",
    }


def test_first_surface_arrow_honors_fixed_pad_order() -> None:
    session_state: dict[str, object] = {}
    records = _records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id]["fixed_slots"] = ((1, "WELL-C"),)
    successes = [
        _successful_plan_xy(name="WELL-A", x_offset_m=0.0, y_offset_m=0.0),
        _successful_plan_xy(name="WELL-B", x_offset_m=0.0, y_offset_m=50.0),
        _successful_plan_xy(name="WELL-C", x_offset_m=0.0, y_offset_m=100.0),
    ]

    overrides = ptc_three_overrides.trajectory_three_payload_overrides(
        session_state,
        records=records,
        successes=successes,
        target_only_wells=[],
        name_to_color={},
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


def test_first_surface_arrow_uses_record_wellhead_for_sidetrack_surface() -> None:
    record_surface = WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0)
    records = [
        WelltrackRecord(
            name="well_04",
            points=(
                record_surface,
                WelltrackPoint(x=456978.0, y=890541.0, z=1852.0, md=2.0),
                WelltrackPoint(x=459130.0, y=887003.0, z=2554.0, md=3.0),
            ),
        )
    ]
    sidetrack_success = _successful_plan_xy(
        name="well_04",
        x_offset_m=456950.0,
        y_offset_m=890500.0,
    )

    overrides = ptc_three_overrides.trajectory_three_payload_overrides(
        {},
        records=records,
        successes=[sidetrack_success],
        target_only_wells=[],
        name_to_color={},
    )

    arrow = list(overrides["extra_meshes"])[0]
    assert arrow["start_position"] == [
        float(record_surface.x),
        float(record_surface.y),
        float(record_surface.z),
    ]
    assert arrow["start_position"] != [
        float(sidetrack_success.surface.x),
        float(sidetrack_success.surface.y),
        float(sidetrack_success.surface.z),
    ]


def test_anticollision_arrow_uses_record_wellhead_for_sidetrack_surface() -> None:
    record_surface = WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0)
    records = [
        WelltrackRecord(
            name="well_04",
            points=(
                record_surface,
                WelltrackPoint(x=456978.0, y=890541.0, z=1852.0, md=2.0),
                WelltrackPoint(x=459130.0, y=887003.0, z=2554.0, md=3.0),
            ),
        )
    ]
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0],
            "X_m": [456950.0, 457050.0],
            "Y_m": [890500.0, 890520.0],
            "Z_m": [1800.0, 1850.0],
        }
    )
    analysis = AntiCollisionAnalysis(
        wells=(
            AntiCollisionWell(
                name="well_04",
                color="#2563eb",
                overlay=SimpleNamespace(),
                samples=(),
                stations=stations,
                surface=Point3D(x=456950.0, y=890500.0, z=1800.0),
                t1=None,
                t3=None,
                md_t1_m=None,
                md_t3_m=None,
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

    overrides = ptc_three_overrides.anticollision_three_payload_overrides(
        {},
        records=records,
        analysis=analysis,
    )

    assert list(overrides["extra_meshes"])[0]["start_position"] == [
        float(record_surface.x),
        float(record_surface.y),
        float(record_surface.z),
    ]


def test_first_surface_arrow_mesh_spans_first_to_seventh_surface() -> None:
    arrow = ptc_three_overrides._pad_first_surface_arrow_payload(
        surface=Point3D(x=0.0, y=0.0, z=0.0),
        end_surface=Point3D(x=0.0, y=300.0, z=0.0),
        nds_azimuth_deg=0.0,
        spacing_m=50.0,
        well_name="WELL-01",
        end_well_name="WELL-07",
        pad_id="PAD-01",
    )

    assert arrow["start_position"] == [0.0, 0.0, 0.0]
    assert arrow["end_position"] == [0.0, 300.0, 0.0]
    assert str(arrow["end_well_name"]) == "WELL-07"
    assert len(arrow["vertices"]) == 14
    assert len(arrow["faces"]) == 20
    assert arrow["vertices"][4][:2] == [0.0, 300.0]
    assert float(arrow["vertices"][4][2]) < float(arrow["vertices"][11][2])


def test_first_surface_arrow_payloads_end_at_seventh_visible_surface() -> None:
    records = [
        WelltrackRecord(
            name=f"WELL-{index:02d}",
            points=(
                WelltrackPoint(x=0.0, y=float((index - 1) * 50), z=0.0, md=0.0),
                WelltrackPoint(x=1000.0, y=float((index - 1) * 50), z=0.0, md=1000.0),
            ),
        )
        for index in range(1, 9)
    ]
    surface_by_name = {
        str(record.name): Point3D(
            x=float(record.points[0].x),
            y=float(record.points[0].y),
            z=float(record.points[0].z),
        )
        for record in records
    }

    session_state: dict[str, object] = {}
    arrows = ptc_three_overrides.pad_first_surface_arrow_payloads(
        session_state,
        records=records,
        visible_well_names=[record.name for record in records],
        surface_by_name=surface_by_name,
    )
    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(session_state, records)
    ordered_names = next(iter(well_names_by_pad_id.values()))

    assert len(arrows) == 1
    arrow = arrows[0]
    expected_start = surface_by_name[str(ordered_names[0])]
    expected_end = surface_by_name[str(ordered_names[6])]
    assert str(arrow["well_name"]) == str(ordered_names[0])
    assert str(arrow["end_well_name"]) == str(ordered_names[6])
    assert arrow["start_position"] == [
        float(expected_start.x),
        float(expected_start.y),
        float(expected_start.z),
    ]
    assert arrow["end_position"] == [
        float(expected_end.x),
        float(expected_end.y),
        float(expected_end.z),
    ]
    assert arrow["vertices"][4][:2] == [float(expected_end.x), float(expected_end.y)]


def test_build_edit_wells_payload_decimates_large_station_arrays() -> None:
    success = _successful_plan_xy(
        name="WELL-A",
        x_offset_m=0.0,
        y_offset_m=0.0,
        station_count=900,
    )

    edit_wells = ptc_three_overrides.build_edit_wells_payload(
        [success],
        {"WELL-A": "#123456"},
    )

    assert edit_wells[0]["name"] == "WELL-A"
    assert edit_wells[0]["color"] == "#123456"
    assert len(edit_wells[0]["base_points"]) == (
        ptc_three_overrides.MAX_EDIT_BASE_POINTS
    )


def test_build_edit_wells_payload_includes_multi_horizontal_edit_points() -> None:
    regular = _successful_plan_xy(
        name="WELL-A",
        x_offset_m=0.0,
        y_offset_m=0.0,
    )
    multi = _successful_plan_xy(
        name="MULTI",
        x_offset_m=5000.0,
        y_offset_m=0.0,
    ).model_copy(
        update={
            "target_pairs": (
                (Point3D(6000.0, 0.0, 0.0), Point3D(6500.0, 0.0, 0.0)),
                (Point3D(6700.0, 0.0, 30.0), Point3D(7200.0, 0.0, 30.0)),
            ),
        }
    )

    edit_wells = ptc_three_overrides.build_edit_wells_payload(
        [regular, multi],
        {"WELL-A": "#123456", "MULTI": "#abcdef"},
    )

    assert [item["name"] for item in edit_wells] == ["WELL-A", "MULTI"]
    multi_payload = edit_wells[1]
    assert multi_payload["color"] == "#abcdef"
    assert multi_payload["edit_points"] == [
        {
            "index": 0,
            "label": "S",
            "point_type": "surface",
            "position": [5000.0, 0.0, 0.0],
        },
        {
            "index": 1,
            "label": "1_t1",
            "point_type": "t1",
            "position": [6000.0, 0.0, 0.0],
        },
        {
            "index": 2,
            "label": "1_t3",
            "point_type": "t3",
            "position": [6500.0, 0.0, 0.0],
        },
        {
            "index": 3,
            "label": "2_t1",
            "point_type": "t1",
            "position": [6700.0, 0.0, 30.0],
        },
        {
            "index": 4,
            "label": "2_t3",
            "point_type": "t3",
            "position": [7200.0, 0.0, 30.0],
        },
    ]


def test_build_target_only_edit_wells_payload_preserves_failed_record_points() -> None:
    surface = Point3D(0.0, 0.0, 0.0)
    pilot_point = Point3D(100.0, 5.0, 900.0)
    multi_pairs = (
        (Point3D(1000.0, 0.0, 2100.0), Point3D(1500.0, 0.0, 2100.0)),
        (Point3D(1700.0, 0.0, 2140.0), Point3D(2200.0, 0.0, 2140.0)),
    )
    target_only_wells = [
        SimpleNamespace(
            name="ORDINARY",
            surface=surface,
            t1=Point3D(500.0, 0.0, 2000.0),
            t3=Point3D(1200.0, 0.0, 2000.0),
            target_pairs=(),
            target_points=(
                surface,
                Point3D(500.0, 0.0, 2000.0),
                Point3D(1200.0, 0.0, 2000.0),
            ),
            target_labels=("S", "t1", "t3"),
        ),
        SimpleNamespace(
            name="ORDINARY_PL",
            surface=surface,
            t1=pilot_point,
            t3=pilot_point,
            target_pairs=(),
            target_points=(surface, pilot_point),
            target_labels=("S", "PL1"),
        ),
        SimpleNamespace(
            name="MULTI",
            surface=surface,
            t1=multi_pairs[0][0],
            t3=multi_pairs[-1][1],
            target_pairs=multi_pairs,
            target_points=(surface, *(point for pair in multi_pairs for point in pair)),
            target_labels=("S", "1_t1", "1_t3", "2_t1", "2_t3"),
        ),
    ]

    edit_wells = ptc_three_overrides.build_target_only_edit_wells_payload(
        target_only_wells,
        {"ORDINARY": "#111111", "ORDINARY_PL": "#222222", "MULTI": "#333333"},
    )

    assert [item["name"] for item in edit_wells] == [
        "ORDINARY",
        "ORDINARY_PL",
        "MULTI",
    ]
    assert [point["label"] for point in edit_wells[0]["edit_points"]] == [
        "S",
        "t1",
        "t3",
    ]
    assert [point["label"] for point in edit_wells[1]["edit_points"]] == ["S", "PL1"]
    assert edit_wells[1]["edit_points"][1]["point_type"] == "pilot"
    assert edit_wells[1]["t1"] == [100.0, 5.0, 900.0]
    assert edit_wells[1]["t3"] == [100.0, 5.0, 900.0]
    assert [point["label"] for point in edit_wells[2]["edit_points"]] == [
        "S",
        "1_t1",
        "1_t3",
        "2_t1",
        "2_t3",
    ]


def test_trajectory_overrides_make_failed_target_only_wells_editable() -> None:
    records = _records()
    target_only = SimpleNamespace(
        name="WELL-C",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(700.0, 760.0, 2200.0),
        t3=Point3D(1600.0, 1960.0, 2350.0),
        target_pairs=(),
        target_points=(
            Point3D(0.0, 0.0, 0.0),
            Point3D(700.0, 760.0, 2200.0),
            Point3D(1600.0, 1960.0, 2350.0),
        ),
        target_labels=("S", "t1", "t3"),
    )

    overrides = ptc_three_overrides.trajectory_three_payload_overrides(
        {},
        records=records,
        successes=[],
        target_only_wells=[target_only],
        name_to_color={"WELL-C": "#abcdef"},
    )

    edit_wells = list(overrides["edit_wells"])
    assert len(edit_wells) == 1
    assert edit_wells[0]["name"] == "WELL-C"
    assert edit_wells[0]["color"] == "#abcdef"
    assert [point["label"] for point in edit_wells[0]["edit_points"]] == [
        "S",
        "t1",
        "t3",
    ]


def test_augment_three_payload_hides_flat_well_legend_when_tree_present() -> None:
    payload = {
        "legend": [
            {"label": "WELL-A", "color": "#22c55e", "opacity": 1.0},
            {"label": "WELL-B", "color": "#2563eb", "opacity": 1.0},
            {"label": "Зоны пересечений", "color": "#fca5a5", "opacity": 0.4},
        ]
    }

    updated = ptc_three_overrides.augment_three_payload(
        payload=payload,
        legend_tree=[{"id": "pad::PAD-01", "label": "Куст PAD-01"}],
        hidden_flat_legend_labels={"WELL-A", "WELL-B"},
    )

    assert [str(item["label"]) for item in updated["legend"]] == ["Зоны пересечений"]
    assert updated["legend_tree"] == [{"id": "pad::PAD-01", "label": "Куст PAD-01"}]


def test_anticollision_overlap_volume_payload_uses_aligned_rings() -> None:
    first_ring = np.array(
        [
            [0.0, 0.0, 100.0],
            [10.0, 0.0, 100.0],
            [10.0, 10.0, 100.0],
            [0.0, 10.0, 100.0],
        ],
        dtype=float,
    )
    shifted_reversed_ring = np.array(
        [
            [1.0, 11.0, 120.0],
            [11.0, 11.0, 120.0],
            [11.0, 1.0, 120.0],
            [1.0, 1.0, 120.0],
        ],
        dtype=float,
    )
    analysis = SimpleNamespace(
        corridors=[
            SimpleNamespace(
                well_a="WELL-A",
                well_b="WELL-B",
                overlap_rings_xyz=(first_ring, shifted_reversed_ring),
            )
        ]
    )

    volumes = ptc_three_overrides.overlap_volume_payloads(analysis)

    assert len(volumes) == 1
    assert volumes[0]["role"] == "overlap_volume"
    assert volumes[0]["name"] == "Зоны пересечений"
    assert len(volumes[0]["rings"]) == 2
    assert volumes[0]["rings"][0][0] == [0.0, 0.0, 100.0]
    assert volumes[0]["rings"][1][0] == [1.0, 1.0, 120.0]


def test_anticollision_overlap_volume_payload_keeps_single_ring_corridor() -> None:
    single_ring = np.array(
        [
            [0.0, 0.0, 100.0],
            [10.0, 0.0, 100.0],
            [10.0, 10.0, 100.0],
            [0.0, 10.0, 100.0],
        ],
        dtype=float,
    )
    analysis = SimpleNamespace(
        corridors=[
            SimpleNamespace(
                well_a="well_09",
                well_b="well_03",
                midpoint_xyz=np.array([[5.0, 5.0, 100.0]], dtype=float),
                overlap_core_radius_m=np.array([5.0], dtype=float),
                overlap_depth_values_m=np.array([8.0], dtype=float),
                overlap_rings_xyz=(single_ring,),
            )
        ]
    )

    volumes = ptc_three_overrides.overlap_volume_payloads(analysis)

    assert len(volumes) == 1
    assert volumes[0]["well_a"] == "well_09"
    assert volumes[0]["well_b"] == "well_03"
    assert len(volumes[0]["rings"]) == 2
    assert len(volumes[0]["rings"][0]) == 4
    assert volumes[0]["rings"][0] != volumes[0]["rings"][1]


def test_anticollision_overlap_volume_payload_does_not_invent_fallback_volume() -> None:
    analysis = SimpleNamespace(
        corridors=[
            SimpleNamespace(
                well_a="WELL-A",
                well_b="WELL-B",
                midpoint_xyz=np.array([[0.0, 0.0, 100.0]], dtype=float),
                overlap_core_radius_m=np.array([6.0], dtype=float),
                overlap_depth_values_m=np.array([12.0], dtype=float),
                overlap_rings_xyz=(),
            )
        ]
    )

    volumes = ptc_three_overrides.overlap_volume_payloads(analysis)

    assert volumes == []


def test_welltracks4_overlap_volumes_match_report_zones() -> None:
    records = [
        record
        for record in parse_welltrack_text(
            Path("tests/test_data/WELLTRACKS4.INC").read_text(encoding="utf-8")
        )
        if str(record.name) in {"well_01", "well_03", "well_09", "well_11"}
    ]
    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={str(record.name) for record in records},
        config=TrajectoryConfig(turn_solver_max_restarts=0),
    )
    assert {row["Скважина"]: row["Статус"] for row in rows} == {
        "well_01": "OK",
        "well_03": "OK",
        "well_09": "OK",
        "well_11": "OK",
    }

    analysis = build_anti_collision_analysis_for_successes(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    events = anti_collision_report_events(analysis)
    volumes = ptc_three_overrides.overlap_volume_payloads(analysis)

    assert len(volumes) == len(events)
    pair_events = [
        event
        for event in events
        if {str(event.well_a), str(event.well_b)} == {"well_01", "well_11"}
    ]
    pair_volumes = [
        volume
        for volume in volumes
        if {str(volume["well_a"]), str(volume["well_b"])} == {"well_01", "well_11"}
    ]
    assert len(pair_events) == len(pair_volumes)
    assert len(pair_events) >= 2
    assert any(
        str(event.classification) == "target-trajectory"
        and int(event.merged_corridor_count) >= 3
        for event in pair_events
    )
    target_pair_events = [
        event
        for event in pair_events
        if str(event.classification) == "target-trajectory"
    ]
    target_pair_volumes = [
        volume
        for volume in pair_volumes
        if str(volume["classification"]) == "target-trajectory"
    ]
    assert len(target_pair_events) == len(target_pair_volumes)
    assert target_pair_events
    for event, volume in zip(
        sorted(target_pair_events, key=lambda item: float(item.md_a_start_m)),
        sorted(target_pair_volumes, key=lambda item: float(item["md_a_start_m"])),
    ):
        assert len(volume["rings"]) >= int(event.merged_corridor_count)

    well_09_03_volumes = [
        volume
        for volume in volumes
        if {str(volume["well_a"]), str(volume["well_b"])} == {"well_09", "well_03"}
    ]
    assert any(
        str(volume["classification"]) == "trajectory"
        and len(volume["rings"]) >= 3
        for volume in well_09_03_volumes
    )


def test_welltracks4_well_06_07_overlap_volumes_are_continuous_3d_zones() -> None:
    records = [
        record
        for record in parse_welltrack_text(
            Path("tests/test_data/WELLTRACKS4.INC").read_text(encoding="utf-8")
        )
        if str(record.name) in {"well_06", "well_07"}
    ]
    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={str(record.name) for record in records},
        config=TrajectoryConfig(turn_solver_max_restarts=0),
    )
    assert {row["Скважина"]: row["Статус"] for row in rows} == {
        "well_06": "OK",
        "well_07": "OK",
    }

    analysis = build_anti_collision_analysis_for_successes(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
    )
    events = [
        event
        for event in anti_collision_report_events(analysis)
        if {str(event.well_a), str(event.well_b)} == {"well_06", "well_07"}
    ]
    volumes = [
        volume
        for volume in ptc_three_overrides.overlap_volume_payloads(analysis)
        if {str(volume["well_a"]), str(volume["well_b"])} == {"well_06", "well_07"}
    ]

    assert events
    assert [
        event
        for event in events
        if min(float(event.md_a_start_m), float(event.md_b_start_m)) > 1000.0
    ] == []
    assert len(volumes) == len(events)
    assert all(len(volume["rings"]) >= 2 for volume in volumes)
    assert any(len(volume["rings"]) >= 8 for volume in volumes)
    assert all(
        len({tuple(point) for ring in volume["rings"] for point in ring}) > len(volume["rings"])
        for volume in volumes
    )


def test_augment_three_payload_appends_overlap_volume_and_legend_once() -> None:
    payload = {
        "meshes": [],
        "legend": [{"label": "Зоны пересечений", "color": "#C62828"}],
    }
    volume = {
        "name": "Зоны пересечений",
        "role": "overlap_volume",
        "rings": [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        ],
    }

    updated = ptc_three_overrides.augment_three_payload(
        payload=payload,
        extra_meshes=[volume],
        extra_legend_items=[
            {"label": "Зоны пересечений", "color": "#C62828", "opacity": 0.34}
        ],
    )

    assert updated["meshes"] == [volume]
    assert [str(item["label"]) for item in updated["legend"]] == ["Зоны пересечений"]
