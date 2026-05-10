from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import TrajectoryConfig
from pywp import ptc_pad_state
from pywp import ptc_three_overrides
from pywp.welltrack_batch import SuccessfulWellPlan


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
    angle_rad = np.deg2rad(float(arrow["nds_azimuth_deg"]))
    direction = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=float)
    tip_xy = np.asarray(arrow["vertices"][5][:2], dtype=float)
    center_xy = np.array([0.0, 100.0], dtype=float)
    delta_xy = tip_xy - center_xy
    assert float(np.dot(delta_xy, direction)) > 0.0
    cross_z = float(direction[0] * delta_xy[1] - direction[1] * delta_xy[0])
    assert abs(cross_z) < 1e-9
    assert float(arrow["vertices"][5][2]) < 0.0


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
