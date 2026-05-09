from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from pywp import ptc_three_payload
from pywp.plotly_config import DEFAULT_3D_CAMERA


def test_scatter3d_legendonly_trace_preserves_legend_without_geometry() -> None:
    trace = go.Scatter3d(
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        mode="lines",
        name="Фактические скважины",
        line={"width": 4, "color": "#6B7280"},
        visible="legendonly",
    )

    payload = ptc_three_payload.scatter3d_trace_to_three_payload(trace)

    assert payload["lines"] == []
    assert payload["points"] == []
    assert payload["labels"] == []
    assert payload["legend_items"] == [
        {
            "label": "Фактические скважины",
            "color": "#6B7280",
            "opacity": 1.0,
        }
    ]


def test_scatter3d_conflict_trace_marks_line_and_hover_roles() -> None:
    trace = go.Scatter3d(
        x=[0.0, 10.0],
        y=[0.0, 0.0],
        z=[0.0, -10.0],
        mode="lines+markers",
        name="Конфликтный участок ствола",
        line={"color": "rgb(198, 40, 40)", "width": 8},
        marker={"color": "rgba(0, 0, 0, 0.001)", "size": 7.5},
        customdata=np.array(
            [
                [0.0, 0.0, 0.0, "Конфликтный участок"],
                [10.0, 1.0, 3.0, "Конфликтный участок"],
            ],
            dtype=object,
        ),
    )

    payload = ptc_three_payload.scatter3d_trace_to_three_payload(trace)

    assert payload["lines"][0]["role"] == "conflict_segment"
    assert payload["points"][0]["role"] == "conflict_hover"


def test_plotly_3d_payload_preserves_custom_camera() -> None:
    custom_camera = {
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.15, "y": -0.05, "z": 0.0},
        "eye": {"x": 1.6, "y": 0.9, "z": 1.1},
    }
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=[0.0, 100.0],
                y=[0.0, 0.0],
                z=[0.0, 100.0],
                mode="lines",
                name="WELL-A",
            )
        ]
    )
    figure.update_layout(scene={"camera": custom_camera})

    payload = ptc_three_payload.plotly_3d_figure_to_three_payload(figure)

    assert payload["camera"] == custom_camera


def test_plotly_3d_payload_uses_default_camera_when_not_set() -> None:
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=[0.0, 100.0],
                y=[0.0, 0.0],
                z=[0.0, 100.0],
                mode="lines",
                name="WELL-A",
            )
        ]
    )

    payload = ptc_three_payload.plotly_3d_figure_to_three_payload(figure)

    assert payload["camera"] == DEFAULT_3D_CAMERA


def test_optimize_three_payload_keeps_mesh_roles_separate() -> None:
    payload = {
        "background": "#FFFFFF",
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
        "camera": DEFAULT_3D_CAMERA,
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


def test_reference_hover_points_are_decimated() -> None:
    trace = go.Scatter3d(
        x=np.arange(300, dtype=float),
        y=np.zeros(300, dtype=float),
        z=np.arange(300, dtype=float),
        mode="lines",
        name="FACT-001 (Фактическая)",
        customdata=np.array(
            [[float(index), 2.0, 80.0, "HOLD"] for index in range(300)],
            dtype=object,
        ),
    )

    payload = ptc_three_payload.scatter3d_trace_to_three_payload(trace)
    hover_only_points = [
        item for item in payload["points"] if bool(item.get("hover_only"))
    ]

    assert len(hover_only_points) == 1
    assert str(hover_only_points[0]["role"]) == "reference_hover"
    assert len(hover_only_points[0]["points"]) == (
        ptc_three_payload.WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
    )
