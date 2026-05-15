from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from pywp import ptc_three_payload
from pywp.models import Point3D, TrajectoryConfig
from pywp.ptc_three_builders import (
    all_wells_three_payload,
    anticollision_three_payload,
    single_well_three_payload,
)
from pywp.three_config import DEFAULT_THREE_CAMERA
from pywp.welltrack_batch import SuccessfulWellPlan


def test_optimize_three_payload_keeps_mesh_roles_separate() -> None:
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_THREE_CAMERA,
        "lines": [],
        "points": [],
        "meshes": [
            {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#15D562",
                "opacity": 0.12,
                "role": "cone",
            },
            {
                "vertices": [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#15D562",
                "opacity": 0.12,
                "role": "overlap",
            },
        ],
        "labels": [],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert len(optimized["meshes"]) == 2
    assert {str(item["role"]) for item in optimized["meshes"]} == {
        "cone",
        "overlap",
    }


def test_optimize_three_payload_preserves_overlap_volume_meshes() -> None:
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_THREE_CAMERA,
        "lines": [],
        "points": [],
        "meshes": [
            {
                "name": "Зоны пересечений",
                "role": "overlap_volume",
                "color": "#C62828",
                "opacity": 0.34,
                "rings": [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                ],
            }
        ],
        "labels": [],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert optimized["meshes"] == payload["meshes"]


def test_optimize_three_payload_preserves_zero_opacity() -> None:
    payload = {
        "lines": [
            {
                "segments": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                "color": "#111111",
                "opacity": 0.0,
                "role": "line",
            }
        ],
        "points": [
            {
                "points": [[0.0, 0.0, 0.0]],
                "hover": [],
                "color": "#222222",
                "opacity": 0.0,
                "size": 4.0,
                "role": "marker",
            }
        ],
        "meshes": [
            {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "faces": [[0, 1, 2]],
                "color": "#333333",
                "opacity": 0.0,
                "role": "mesh",
            }
        ],
        "labels": [],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert optimized["lines"][0]["opacity"] == 0.0
    assert optimized["points"][0]["opacity"] == 0.0
    assert optimized["meshes"][0]["opacity"] == 0.0


def test_optimize_three_payload_keeps_named_wells_separate() -> None:
    payload = {
        "lines": [
            {
                "name": "WELL-1",
                "segments": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                "color": "#22C55E",
                "opacity": 1.0,
                "dash": "solid",
                "role": "line",
            },
            {
                "name": "WELL-2",
                "segments": [[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
                "color": "#22C55E",
                "opacity": 1.0,
                "dash": "solid",
                "role": "line",
            },
        ],
        "points": [
            {
                "name": "WELL-1: цели",
                "points": [[0.0, 0.0, 0.0]],
                "hover": [{"name": "WELL-1"}],
                "color": "#22C55E",
                "opacity": 1.0,
                "size": 5.0,
                "symbol": "circle",
                "role": "marker",
            },
            {
                "name": "WELL-2: цели",
                "points": [[0.0, 1.0, 0.0]],
                "hover": [{"name": "WELL-2"}],
                "color": "#22C55E",
                "opacity": 1.0,
                "size": 5.0,
                "symbol": "circle",
                "role": "marker",
            },
        ],
        "meshes": [],
        "labels": [],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert [item["name"] for item in optimized["lines"]] == ["WELL-1", "WELL-2"]
    assert [item["name"] for item in optimized["points"]] == [
        "WELL-1: цели",
        "WELL-2: цели",
    ]


def test_optimize_three_payload_preserves_hover_alignment_with_sparse_hover() -> None:
    payload = {
        "lines": [],
        "points": [
            {
                "points": [[0.0, 0.0, 0.0]],
                "hover": [],
                "color": "#111111",
                "opacity": 1.0,
                "size": 6.0,
                "symbol": "circle",
                "role": "marker",
            },
            {
                "points": [[1.0, 0.0, 0.0]],
                "hover": [{"name": "second"}],
                "color": "#111111",
                "opacity": 1.0,
                "size": 6.0,
                "symbol": "circle",
                "role": "marker",
            },
        ],
        "meshes": [],
        "labels": [],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert len(optimized["points"]) == 1
    assert optimized["points"][0]["hover"] == [{}, {"name": "second"}]


def test_single_well_three_payload_renders_pilot_family_labels() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "X_m": [0.0, 1000.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 1000.0],
        }
    )
    pilot_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 700.0, 1200.0],
            "X_m": [0.0, 200.0, 500.0],
            "Y_m": [0.0, 50.0, 100.0],
            "Z_m": [0.0, 650.0, 1100.0],
        }
    )

    payload = single_well_three_payload(
        stations,
        well_name="well_04",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(800.0, 0.0, 900.0),
        t3=Point3D(1000.0, 0.0, 1000.0),
        pilot_name="well_04_PL",
        pilot_stations=pilot_stations,
        pilot_study_points=(
            Point3D(200.0, 50.0, 650.0),
            Point3D(500.0, 100.0, 1100.0),
        ),
    )

    line_names = [str(item.get("name")) for item in payload["lines"]]
    label_texts = [str(item.get("text")) for item in payload["labels"]]
    pilot_marker = next(
        item
        for item in payload["points"]
        if str(item.get("name")) == "well_04_PL: точки пилота"
    )

    assert "well_04_PL" in line_names
    assert "well_04_PL: 1" in label_texts
    assert "well_04_PL: 2" in label_texts
    assert len(pilot_marker["points"]) == 2


def test_all_wells_three_payload_renders_pilot_point_labels() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 600.0, 1200.0],
            "X_m": [0.0, 200.0, 500.0],
            "Y_m": [0.0, 50.0, 100.0],
            "Z_m": [0.0, 650.0, 1100.0],
        }
    )
    pilot_success = SuccessfulWellPlan(
        name="well_04_PL",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(200.0, 50.0, 650.0),
        t3=Point3D(500.0, 100.0, 1100.0),
        stations=stations,
        summary={"trajectory_type": "PILOT"},
        azimuth_deg=0.0,
        md_t1_m=600.0,
        config=TrajectoryConfig(),
    )

    payload = all_wells_three_payload(
        [pilot_success],
        pilot_study_points_by_name={
            "well_04_PL": (
                Point3D(200.0, 50.0, 650.0),
                Point3D(350.0, 75.0, 900.0),
                Point3D(500.0, 100.0, 1100.0),
            )
        },
    )

    label_texts = [str(item.get("text")) for item in payload["labels"]]
    pilot_marker = next(
        item
        for item in payload["points"]
        if str(item.get("name")) == "well_04_PL: цели"
    )

    assert "well_04_PL: 1" in label_texts
    assert "well_04_PL: 2" in label_texts
    assert "well_04_PL: 3" in label_texts
    assert [hover["point"] for hover in pilot_marker["hover"]] == [
        "well_04_PL: 1",
        "well_04_PL: 2",
        "well_04_PL: 3",
    ]
    assert pilot_marker["points"] == [
        [200.0, 50.0, 650.0],
        [350.0, 75.0, 900.0],
        [500.0, 100.0, 1100.0],
    ]


def test_anticollision_three_payload_renders_pilot_point_labels() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 600.0, 1200.0],
            "X_m": [0.0, 200.0, 500.0],
            "Y_m": [0.0, 50.0, 100.0],
            "Z_m": [0.0, 650.0, 1100.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0],
        }
    )
    analysis = SimpleNamespace(
        wells=(
            SimpleNamespace(
                name="well_04_PL",
                color="#22C55E",
                overlay=SimpleNamespace(samples=()),
                stations=stations,
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(200.0, 50.0, 650.0),
                t3=Point3D(500.0, 100.0, 1100.0),
                md_t1_m=600.0,
                is_reference_only=False,
                target_pairs=(),
            ),
        ),
        corridors=(),
        well_segments=(),
        zones=(),
    )

    payload = anticollision_three_payload(
        analysis,
        pilot_study_points_by_name={
            "well_04_PL": (
                Point3D(200.0, 50.0, 650.0),
                Point3D(350.0, 75.0, 900.0),
                Point3D(500.0, 100.0, 1100.0),
            )
        },
    )

    label_texts = [str(item.get("text")) for item in payload["labels"]]
    pilot_marker = next(
        item
        for item in payload["points"]
        if str(item.get("name")) == "well_04_PL: цели"
    )

    assert "well_04_PL: 1" in label_texts
    assert "well_04_PL: 2" in label_texts
    assert "well_04_PL: 3" in label_texts
    assert [hover["point"] for hover in pilot_marker["hover"]] == [
        "well_04_PL: 1",
        "well_04_PL: 2",
        "well_04_PL: 3",
    ]


def test_all_wells_three_payload_keeps_original_target_markers() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "X_m": [0.0, 990.0, 1990.0],
            "Y_m": [0.0, 25.0, 25.0],
            "Z_m": [0.0, 1500.0, 1500.0],
        }
    )
    success = SuccessfulWellPlan(
        name="well_12",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 50.0, 1500.0),
        t3=Point3D(2000.0, 50.0, 1500.0),
        stations=stations,
        summary={
            "t1_exact_x_m": 990.0,
            "t1_exact_y_m": 25.0,
            "t1_exact_z_m": 1500.0,
            "t3_exact_x_m": 1990.0,
            "t3_exact_y_m": 25.0,
            "t3_exact_z_m": 1500.0,
        },
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )

    payload = all_wells_three_payload([success])
    marker = next(
        item for item in payload["points"] if str(item.get("name")) == "well_12: цели"
    )
    label = next(item for item in payload["labels"] if str(item.get("text")) == "well_12")

    assert marker["points"][1] == [1000.0, 50.0, 1500.0]
    assert marker["points"][2] == [2000.0, 50.0, 1500.0]
    assert label["position"] == [2000.0, 50.0, 1500.0]


def test_all_wells_three_payload_includes_multi_horizontal_target_pairs() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 1600.0, 1900.0, 2500.0],
            "X_m": [0.0, 100.0, 500.0, 650.0, 1050.0],
            "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Z_m": [0.0, 1000.0, 1000.0, 1020.0, 1020.0],
        }
    )
    success = SuccessfulWellPlan(
        name="multi",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(100.0, 0.0, 1000.0),
        t3=Point3D(1050.0, 0.0, 1020.0),
        target_pairs=(
            (Point3D(100.0, 0.0, 1000.0), Point3D(500.0, 0.0, 1000.0)),
            (Point3D(650.0, 0.0, 1020.0), Point3D(1050.0, 0.0, 1020.0)),
        ),
        stations=stations,
        summary={},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )

    payload = all_wells_three_payload([success])
    marker = next(
        item for item in payload["points"] if str(item.get("name")) == "multi: цели"
    )

    assert marker["points"] == [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 1000.0],
        [500.0, 0.0, 1000.0],
        [650.0, 0.0, 1020.0],
        [1050.0, 0.0, 1020.0],
    ]
    assert [item["point"] for item in marker["hover"]] == [
        "S",
        "1_t1",
        "1_t3",
        "2_t1",
        "2_t3",
    ]


def test_all_wells_three_payload_includes_multi_horizontal_target_only_pairs() -> None:
    target_only = SimpleNamespace(
        name="multi_failed",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(100.0, 0.0, 1000.0),
        t3=Point3D(1050.0, 0.0, 1020.0),
        target_pairs=(
            (Point3D(100.0, 0.0, 1000.0), Point3D(500.0, 0.0, 1000.0)),
            (Point3D(650.0, 0.0, 1020.0), Point3D(1050.0, 0.0, 1020.0)),
        ),
        status="Ошибка расчета",
        problem="short transition",
    )

    payload = all_wells_three_payload([], target_only_wells=[target_only])
    marker = next(
        item
        for item in payload["points"]
        if str(item.get("name")) == "multi_failed: цели (без траектории)"
    )

    assert marker["points"] == [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 1000.0],
        [500.0, 0.0, 1000.0],
        [650.0, 0.0, 1020.0],
        [1050.0, 0.0, 1020.0],
    ]
