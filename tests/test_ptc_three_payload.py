from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from pywp import ptc_three_payload
from pywp.anticollision_rerun import build_anti_collision_analysis_for_successes
from pywp.models import Point3D, TrajectoryConfig
from pywp.ptc_three_builders import (
    all_wells_three_payload,
    anticollision_three_payload,
    single_well_three_payload,
    single_well_target_only_three_payload,
)
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    parse_reference_trajectory_table,
)
from pywp.three_config import DEFAULT_THREE_CAMERA
from pywp.uncertainty import PlanningUncertaintyModel
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


def test_optimize_three_payload_keeps_all_primary_labels_even_with_many_well_labels() -> None:
    well_labels = [
        {
            "text": f"WELL-{index}",
            "position": [float(index), 0.0, 0.0],
            "color": "#111111",
            "role": "well_label",
        }
        for index in range(ptc_three_payload.WT_THREE_MAX_LABELS + 10)
    ]
    reference_labels = [
        {
            "text": "FACT-1",
            "position": [1000.0, 0.0, 0.0],
            "color": "#374151",
            "role": "reference_label",
        },
        {
            "text": "APP-1",
            "position": [1100.0, 0.0, 0.0],
            "color": "#F87171",
            "role": "reference_label",
        },
    ]
    payload = {
        "lines": [],
        "points": [],
        "meshes": [],
        "labels": [*well_labels, *reference_labels],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert len(optimized["labels"]) == len(well_labels) + len(reference_labels)
    assert sum(
        1 for item in optimized["labels"] if item.get("role") == "reference_label"
    ) == len(reference_labels)
    assert sum(
        1 for item in optimized["labels"] if item.get("role") == "well_label"
    ) == len(well_labels)


def test_optimize_three_payload_limits_only_auxiliary_labels() -> None:
    payload = {
        "lines": [],
        "points": [],
        "meshes": [],
        "labels": [
            {
                "text": "WELL-A",
                "position": [0.0, 0.0, 0.0],
                "color": "#111111",
                "role": "well_label",
            },
            *[
                {
                    "text": f"aux-{index}",
                    "position": [float(index), 0.0, 0.0],
                    "color": "#222222",
                    "role": "pilot_point_label",
                }
                for index in range(ptc_three_payload.WT_THREE_MAX_LABELS + 10)
            ],
            {
                "text": "FACT-1",
                "position": [1000.0, 0.0, 0.0],
                "color": "#374151",
                "role": "reference_label",
            },
        ],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert len(optimized["labels"]) == ptc_three_payload.WT_THREE_MAX_LABELS + 2
    label_texts = [str(item.get("text")) for item in optimized["labels"]]
    assert "WELL-A" in label_texts
    assert "FACT-1" in label_texts
    assert f"aux-{ptc_three_payload.WT_THREE_MAX_LABELS - 1}" in label_texts
    assert f"aux-{ptc_three_payload.WT_THREE_MAX_LABELS}" not in label_texts


def test_optimize_three_payload_keeps_all_reference_labels() -> None:
    payload = {
        "lines": [],
        "points": [],
        "meshes": [],
        "labels": [
            *[
                {
                    "text": f"WELL-{index}",
                    "position": [float(index), 0.0, 0.0],
                    "color": "#111111",
                    "role": "well_label",
                }
                for index in range(5)
            ],
            *[
                {
                    "text": f"REF-{index}",
                    "position": [1000.0 + float(index), 0.0, 0.0],
                    "color": "#374151",
                    "role": "reference_label",
                }
                for index in range(ptc_three_payload.WT_THREE_MAX_REFERENCE_LABELS + 8)
            ],
            *[
                {
                    "text": f"aux-{index}",
                    "position": [2000.0 + float(index), 0.0, 0.0],
                    "color": "#222222",
                    "role": "pilot_point_label",
                }
                for index in range(50)
            ],
        ],
        "legend": [],
    }

    optimized = ptc_three_payload.optimize_three_payload(payload)

    assert len(optimized["labels"]) == (
        5
        + (ptc_three_payload.WT_THREE_MAX_REFERENCE_LABELS + 8)
        + ptc_three_payload.WT_THREE_MAX_LABELS
    )
    assert sum(
        1 for item in optimized["labels"] if item.get("role") == "reference_label"
    ) == (ptc_three_payload.WT_THREE_MAX_REFERENCE_LABELS + 8)
    assert sum(
        1 for item in optimized["labels"] if item.get("role") == "well_label"
    ) == 5
    assert sum(
        1 for item in optimized["labels"] if item.get("role") == "pilot_point_label"
    ) == ptc_three_payload.WT_THREE_MAX_LABELS


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
    assert {
        str(item.get("role"))
        for item in payload["labels"]
        if str(item.get("text")).startswith("well_04_PL: ")
    } == {"pilot_point_label"}
    assert len(pilot_marker["points"]) == 2


def test_single_well_three_payload_labels_targets_and_kop() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0],
            "X_m": [0.0, 100.0, 500.0, 1000.0],
            "Y_m": [0.0, 0.0, 50.0, 100.0],
            "Z_m": [0.0, 500.0, 900.0, 1000.0],
        }
    )

    payload = single_well_three_payload(
        stations,
        well_name="well_01",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(500.0, 50.0, 900.0),
        t3=Point3D(1000.0, 100.0, 1000.0),
        md_t1_m=1000.0,
        kop_md_m=500.0,
    )

    label_by_text = {str(item.get("text")): item for item in payload["labels"]}
    assert {"t1", "t3", "KOP"}.issubset(label_by_text)
    assert "well_01" not in label_by_text
    assert label_by_text["t1"]["role"] == "target_label"
    assert label_by_text["t3"]["role"] == "target_label"
    assert label_by_text["KOP"]["role"] == "control_point_label"
    kop_marker = next(item for item in payload["points"] if item.get("name") == "KOP")
    assert kop_marker["points"] == [[100.0, 0.0, 500.0]]


def test_single_well_three_payload_uses_pilot_kop_and_sidetrack_window() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0],
            "X_m": [0.0, 100.0, 200.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 500.0, 1000.0],
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
        kop_md_m=500.0,
        pilot_name="well_04_PL",
        pilot_stations=pilot_stations,
        pilot_kop_md_m=700.0,
        sidetrack_window_point=Point3D(150.0, 25.0, 500.0),
    )

    label_by_text = {str(item.get("text")): item for item in payload["labels"]}
    kop_marker = next(item for item in payload["points"] if item.get("name") == "KOP")
    window_marker = next(
        item for item in payload["points"] if item.get("name") == "Окно зарезки"
    )

    assert "well_04" not in label_by_text
    assert label_by_text["KOP"]["position"] == [200.0, 50.0, 650.0]
    assert label_by_text["Окно"]["position"] == [150.0, 25.0, 500.0]
    assert kop_marker["points"] == [[200.0, 50.0, 650.0]]
    assert window_marker["points"] == [[150.0, 25.0, 500.0]]


def test_single_well_target_only_three_payload_shows_editable_targets_context() -> None:
    payload = single_well_target_only_three_payload(
        well_name="single_well",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
    )

    point_trace = next(
        item
        for item in payload["points"]
        if str(item.get("name")) == "single_well: цели (без траектории)"
    )
    label_texts = {str(item.get("text")) for item in payload["labels"]}
    legend_item = next(
        item
        for item in payload["legend"]
        if str(item.get("label")) == "single_well: цели (без траектории)"
    )

    assert point_trace["points"] == [
        [0.0, 0.0, 0.0],
        [600.0, 800.0, 2400.0],
        [1500.0, 2000.0, 2500.0],
    ]
    assert {"S", "t1", "t3"}.issubset(label_texts)
    assert "single_well" not in label_texts
    assert legend_item["symbol"] == "point"


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
    assert {
        str(item.get("role"))
        for item in payload["labels"]
        if str(item.get("text")).startswith("well_04_PL: ")
    } == {"pilot_point_label"}
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


def test_all_wells_three_payload_keeps_pilot_labels_without_pad_position() -> None:
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
        display_name_by_well_name={"well_04_PL": "well_04_PL (2)"},
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

    assert "well_04_PL" in label_texts
    assert "well_04_PL (2)" not in label_texts
    assert all(
        str(hover.get("name")) == "well_04_PL: цели"
        for hover in pilot_marker["hover"]
    )
    assert all(
        "(2)" not in str(hover.get("point"))
        for hover in pilot_marker["hover"]
    )


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
    assert {
        str(item.get("role"))
        for item in payload["labels"]
        if str(item.get("text")).startswith("well_04_PL: ")
    } == {"pilot_point_label"}
    assert [hover["point"] for hover in pilot_marker["hover"]] == [
        "well_04_PL: 1",
        "well_04_PL: 2",
        "well_04_PL: 3",
    ]


def test_anticollision_three_payload_keeps_pilot_labels_without_pad_position() -> None:
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
        display_name_by_well_name={"well_04_PL": "well_04_PL (2)"},
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

    assert "well_04_PL" in label_texts
    assert "well_04_PL (2)" not in label_texts
    assert all(
        str(hover.get("name")) == "well_04_PL: цели"
        for hover in pilot_marker["hover"]
    )
    assert all(
        "(2)" not in str(hover.get("point"))
        for hover in pilot_marker["hover"]
    )


def test_anticollision_three_payload_can_show_pad_position_in_well_label() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "X_m": [0.0, 1000.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 1000.0],
            "DLS_deg_per_30m": [0.0, 0.0],
        }
    )
    analysis = SimpleNamespace(
        wells=(
            SimpleNamespace(
                name="well_01",
                color="#22C55E",
                overlay=SimpleNamespace(samples=()),
                stations=stations,
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(300.0, 0.0, 300.0),
                t3=Point3D(1000.0, 0.0, 1000.0),
                md_t1_m=300.0,
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
        display_name_by_well_name={"well_01": "well_01 (2)"},
    )

    assert any(
        str(item.get("text")) == "well_01 (2)" and str(item.get("role")) == "well_label"
        for item in payload["labels"]
    )


def test_anticollision_three_payload_labels_reference_wells_in_fast_mode() -> None:
    calc_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "X_m": [0.0, 1000.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 1000.0],
            "DLS_deg_per_30m": [0.0, 0.0],
        }
    )
    actual_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1200.0],
            "X_m": [100.0, 1800.0],
            "Y_m": [25.0, 25.0],
            "Z_m": [0.0, 420.0],
        }
    )
    approved_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1100.0],
            "X_m": [150.0, 1700.0],
            "Y_m": [-35.0, -35.0],
            "Z_m": [0.0, 360.0],
        }
    )
    analysis = SimpleNamespace(
        wells=(
            SimpleNamespace(
                name="well_01",
                color="#22C55E",
                overlay=SimpleNamespace(samples=()),
                stations=calc_stations,
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(300.0, 0.0, 300.0),
                t3=Point3D(1000.0, 0.0, 1000.0),
                md_t1_m=300.0,
                is_reference_only=False,
                target_pairs=(),
            ),
            SimpleNamespace(
                name="FACT-1",
                well_kind=REFERENCE_WELL_ACTUAL,
                color="#6B7280",
                overlay=SimpleNamespace(samples=()),
                stations=actual_stations,
                surface=Point3D(100.0, 25.0, 0.0),
                t1=None,
                t3=None,
                is_reference_only=True,
            ),
            SimpleNamespace(
                name="APP-1",
                well_kind=REFERENCE_WELL_APPROVED,
                color="#C62828",
                overlay=SimpleNamespace(samples=()),
                stations=approved_stations,
                surface=Point3D(150.0, -35.0, 0.0),
                t1=None,
                t3=None,
                is_reference_only=True,
            ),
        ),
        corridors=(),
        well_segments=(),
        zones=(),
    )

    payload = anticollision_three_payload(analysis)

    reference_labels = {
        str(item.get("text")): item
        for item in payload["labels"]
        if str(item.get("role")) == "reference_label"
    }
    assert reference_labels["FACT-1"]["position"] == [1800.0, 25.0, 420.0]
    assert reference_labels["FACT-1"]["color"] == "#374151"
    assert reference_labels["APP-1"]["position"] == [1700.0, -35.0, 360.0]
    assert reference_labels["APP-1"]["color"] == "#C62828"


def test_anticollision_three_payload_does_not_label_display_only_reference_wells_fast() -> None:
    calc_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "X_m": [0.0, 1000.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 1000.0],
            "DLS_deg_per_30m": [0.0, 0.0],
        }
    )
    display_only_reference = parse_reference_trajectory_table(
        [
            {
                "Wellname": "FACT-FAR",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 5000.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "FACT-FAR",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 6000.0,
                "Y": 0.0,
                "Z": 500.0,
                "MD": 1200.0,
            },
        ]
    )
    analysis = SimpleNamespace(
        wells=(
            SimpleNamespace(
                name="well_01",
                color="#22C55E",
                overlay=SimpleNamespace(samples=()),
                stations=calc_stations,
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(300.0, 0.0, 300.0),
                t3=Point3D(1000.0, 0.0, 1000.0),
                md_t1_m=300.0,
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
        reference_wells=tuple(display_only_reference),
    )

    assert not any(
        str(item.get("text")) == "FACT-FAR"
        and str(item.get("role")) == "reference_label"
        for item in payload["labels"]
    )
    assert payload["optional_reference_labels"] == [
        {
            "text": "FACT-FAR",
            "position": [6000.0, 0.0, 500.0],
            "color": "#374151",
            "role": "reference_label_optional",
        }
    ]


def test_anticollision_three_payload_ignores_display_only_reference_wells_in_default_bounds() -> None:
    calc_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "X_m": [0.0, 1000.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 1000.0],
            "DLS_deg_per_30m": [0.0, 0.0],
        }
    )
    display_only_reference = parse_reference_trajectory_table(
        [
            {
                "Wellname": "FACT-FAR",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 5000.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "FACT-FAR",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 6000.0,
                "Y": 0.0,
                "Z": 500.0,
                "MD": 1200.0,
            },
        ]
    )
    analysis = SimpleNamespace(
        wells=(
            SimpleNamespace(
                name="well_01",
                color="#22C55E",
                overlay=SimpleNamespace(samples=()),
                stations=calc_stations,
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(300.0, 0.0, 300.0),
                t3=Point3D(1000.0, 0.0, 1000.0),
                md_t1_m=300.0,
                is_reference_only=False,
                target_pairs=(),
            ),
        ),
        corridors=(),
        well_segments=(),
        zones=(),
    )

    base_payload = anticollision_three_payload(analysis)
    payload = anticollision_three_payload(
        analysis,
        reference_wells=tuple(display_only_reference),
    )

    assert payload["bounds"] == base_payload["bounds"]


def _sidetrack_parent_payload_analysis():
    stations = pd.DataFrame(
        {
            "MD_m": [1000.0, 1200.0, 1600.0, 2000.0],
            "INC_deg": [90.0, 90.0, 90.0, 90.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0],
            "X_m": [1000.0, 1200.0, 1600.0, 2000.0],
            "Y_m": [0.0, 0.0, 0.0, 0.0],
            "Z_m": [0.0, 0.0, 0.0, 0.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    zbs = SuccessfulWellPlan(
        name="9010_ZBS",
        surface=Point3D(1000.0, 0.0, 0.0),
        t1=Point3D(1200.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        stations=stations,
        summary={
            "trajectory_type": "FACT_SIDETRACK",
            "actual_parent_well_name": "9010",
            "sidetrack_parent_well_name": "9010",
            "sidetrack_parent_kind": REFERENCE_WELL_ACTUAL,
            "sidetrack_window_md_m": 1000.0,
        },
        azimuth_deg=90.0,
        md_t1_m=1200.0,
        config=TrajectoryConfig(),
    )
    parent = parse_reference_trajectory_table(
        [
            {
                "Wellname": "9010",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9010",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 1000.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 1000.0,
            },
            {
                "Wellname": "9010",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 2000.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 2000.0,
            },
        ],
        default_kind=REFERENCE_WELL_ACTUAL,
    )[0]
    return build_anti_collision_analysis_for_successes(
        [zbs],
        model=PlanningUncertaintyModel(sample_step_m=50.0, min_refined_step_m=10.0),
        reference_wells=(parent,),
        include_display_geometry=False,
        build_overlap_geometry=False,
        analysis_sample_step_m=50.0,
    )


def test_anticollision_three_payload_includes_sidetrack_relative_cones_with_hidden_default() -> None:
    payload = anticollision_three_payload(_sidetrack_parent_payload_analysis())

    assert any(
        str(item.get("role")) == "sidetrack_relative_cone"
        for item in payload["meshes"]
    )
    assert payload["anti_collision_layer_state"]["sidetrack_relative_cones"] is False


def test_anticollision_three_payload_can_enable_sidetrack_relative_cones_layer() -> None:
    payload = anticollision_three_payload(
        _sidetrack_parent_payload_analysis(),
        show_sidetrack_relative_cones=True,
    )

    assert any(
        str(item.get("role")) == "sidetrack_relative_cone"
        for item in payload["meshes"]
    )
    assert any(
        str(item.get("role")) == "sidetrack_relative_cone_tip"
        for item in payload["lines"]
    )
    assert any(
        str(item.get("label")) == "Относительные конуса боковых стволов"
        for item in payload["legend"]
    )
    assert payload["anti_collision_layer_state"]["sidetrack_relative_cones"] is True


def test_all_wells_three_payload_places_reference_labels_at_well_end() -> None:
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-1",
                    "Type": REFERENCE_WELL_ACTUAL,
                    "X": 0.0,
                    "Y": 25.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": REFERENCE_WELL_ACTUAL,
                    "X": 1800.0,
                    "Y": 25.0,
                    "Z": 400.0,
                    "MD": 1900.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": REFERENCE_WELL_APPROVED,
                    "X": 0.0,
                    "Y": -35.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": REFERENCE_WELL_APPROVED,
                    "X": 1700.0,
                    "Y": -35.0,
                    "Z": 320.0,
                    "MD": 1800.0,
                },
            ]
        )
    )

    payload = all_wells_three_payload(
        [],
        reference_wells=reference_wells,
        render_mode="detail",
    )

    reference_labels = {
        str(item.get("text")): list(item.get("position") or [])
        for item in payload["labels"]
        if str(item.get("role")) == "reference_label"
    }

    assert reference_labels["FACT-1"] == [1800.0, 25.0, 400.0]
    assert reference_labels["APP-1"] == [1700.0, -35.0, 320.0]


def test_all_wells_three_payload_exposes_optional_reference_labels_for_fast_mode() -> None:
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-1",
                    "Type": REFERENCE_WELL_ACTUAL,
                    "X": 0.0,
                    "Y": 25.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": REFERENCE_WELL_ACTUAL,
                    "X": 1800.0,
                    "Y": 25.0,
                    "Z": 400.0,
                    "MD": 1900.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": REFERENCE_WELL_APPROVED,
                    "X": 0.0,
                    "Y": -35.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "APP-1",
                    "Type": REFERENCE_WELL_APPROVED,
                    "X": 1700.0,
                    "Y": -35.0,
                    "Z": 320.0,
                    "MD": 1800.0,
                },
            ]
        )
    )

    payload = all_wells_three_payload(
        [],
        reference_wells=reference_wells,
        render_mode="Быстро",
    )

    optional_labels = {
        str(item.get("text")): item
        for item in payload["optional_reference_labels"]
    }

    assert not [
        item for item in payload["labels"] if str(item.get("role")) == "reference_label"
    ]
    assert optional_labels["FACT-1"]["position"] == [1800.0, 25.0, 400.0]
    assert optional_labels["FACT-1"]["color"] == "#374151"
    assert optional_labels["FACT-1"]["role"] == "reference_label_optional"
    assert optional_labels["APP-1"]["position"] == [1700.0, -35.0, 320.0]
    assert optional_labels["APP-1"]["color"] == "#C62828"
    assert optional_labels["APP-1"]["role"] == "reference_label_optional"


def test_all_wells_three_payload_highlights_actual_sidetrack_parent_in_fast_mode() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [1000.0, 1200.0, 1600.0, 2000.0],
            "INC_deg": [90.0, 90.0, 90.0, 90.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0],
            "X_m": [1000.0, 1200.0, 1600.0, 2000.0],
            "Y_m": [0.0, 0.0, 0.0, 0.0],
            "Z_m": [0.0, 0.0, 0.0, 0.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    zbs = SuccessfulWellPlan(
        name="9010_ZBS",
        surface=Point3D(1000.0, 0.0, 0.0),
        t1=Point3D(1200.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        stations=stations,
        summary={
            "trajectory_type": "FACT_SIDETRACK",
            "actual_parent_well_name": "9010",
            "sidetrack_parent_well_name": "9010",
            "sidetrack_parent_kind": REFERENCE_WELL_ACTUAL,
            "sidetrack_window_md_m": 1000.0,
        },
        azimuth_deg=90.0,
        md_t1_m=1200.0,
        config=TrajectoryConfig(),
    )
    parent = parse_reference_trajectory_table(
        [
            {
                "Wellname": "9010",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "9010",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 1000.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 1000.0,
            },
            {
                "Wellname": "9010",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": 2000.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 2000.0,
            },
        ],
        default_kind=REFERENCE_WELL_ACTUAL,
    )[0]

    payload = all_wells_three_payload(
        [zbs],
        reference_wells=(parent,),
        render_mode="Быстро",
    )

    reference_labels = {
        str(item.get("text")): list(item.get("position") or [])
        for item in payload["labels"]
        if str(item.get("role")) == "reference_label"
    }
    hover_names = {
        str(hover.get("name"))
        for item in payload["points"]
        if str(item.get("role")) == "reference_hover"
        for hover in list(item.get("hover") or [])
    }

    assert reference_labels["9010"] == [2000.0, 0.0, 0.0]
    assert "9010" in hover_names


def test_all_wells_three_payload_can_show_pad_position_in_well_label() -> None:
    success = SuccessfulWellPlan(
        name="well_01",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 1000.0, 2000.0],
                "X_m": [0.0, 1000.0, 2000.0],
                "Y_m": [0.0, 0.0, 0.0],
                "Z_m": [0.0, 0.0, 0.0],
                "DLS_deg_per_30m": [0.0, 0.0, 0.0],
            }
        ),
        summary={},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )

    payload = all_wells_three_payload(
        [success],
        display_name_by_well_name={"well_01": "well_01 (2)"},
    )

    assert any(
        str(item.get("text")) == "well_01 (2)" and str(item.get("role")) == "well_label"
        for item in payload["labels"]
    )


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


def test_all_wells_three_payload_keeps_warning_wells_solid() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "X_m": [0.0, 1000.0, 2000.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 1500.0, 1500.0],
        }
    )
    success = SuccessfulWellPlan(
        name="warning_well",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 1500.0),
        t3=Point3D(2000.0, 0.0, 1500.0),
        stations=stations,
        summary={},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
        md_postcheck_exceeded=True,
    )

    payload = all_wells_three_payload([success])
    line = next(
        item for item in payload["lines"] if str(item.get("name")) == "warning_well"
    )

    assert line["dash"] == "solid"


def test_all_wells_three_payload_keeps_message_only_warning_wells_solid() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "X_m": [0.0, 1000.0, 2000.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 1200.0, 1200.0],
        }
    )
    success = SuccessfulWellPlan(
        name="warning_message_well",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 1200.0),
        t3=Point3D(2000.0, 0.0, 1200.0),
        stations=stations,
        summary={},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
        md_postcheck_exceeded=False,
        md_postcheck_message="MD warning",
    )

    payload = all_wells_three_payload([success])
    line = next(
        item
        for item in payload["lines"]
        if str(item.get("name")) == "warning_message_well"
    )

    assert line["dash"] == "solid"


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
    legend_item = next(
        item
        for item in payload["legend"]
        if str(item.get("label")) == "multi_failed: цели (без траектории)"
    )

    assert marker["points"] == [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 1000.0],
        [500.0, 0.0, 1000.0],
        [650.0, 0.0, 1020.0],
        [1050.0, 0.0, 1020.0],
    ]
    assert legend_item["symbol"] == "point"
