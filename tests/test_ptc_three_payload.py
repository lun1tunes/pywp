from __future__ import annotations

from pywp import ptc_three_payload
from pywp.three_config import DEFAULT_THREE_CAMERA


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
