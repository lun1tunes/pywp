from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pywp.models import Point3D

DEG2RAD = np.pi / 180.0


def _section_coordinate(df: pd.DataFrame, surface: Point3D, azimuth_deg: float) -> np.ndarray:
    az = azimuth_deg * DEG2RAD
    dn = df["Y_m"].to_numpy() - surface.y
    de = df["X_m"].to_numpy() - surface.x
    return dn * np.cos(az) + de * np.sin(az)


def trajectory_3d_figure(
    df: pd.DataFrame, surface: Point3D, t1: Point3D, t3: Point3D, height: int = 560
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["X_m"],
            y=df["Y_m"],
            z=df["Z_m"],
            mode="lines",
            name="Trajectory",
            line={"width": 6, "color": "#006D77"},
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[surface.x, t1.x, t3.x],
            y=[surface.y, t1.y, t3.y],
            z=[surface.z, t1.z, t3.z],
            mode="markers+text",
            text=["S", "t1", "t3"],
            textposition="top center",
            name="Targets",
            marker={"size": 6, "color": ["#EF476F", "#FFD166", "#118AB2"]},
        )
    )

    fig.update_layout(
        title="3D trajectory",
        scene={
            "xaxis_title": "X / East (m)",
            "yaxis_title": "Y / North (m)",
            "zaxis_title": "Z / TVD (m)",
            "zaxis": {"autorange": "reversed"},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=height,
    )
    return fig


def plan_view_figure(
    df: pd.DataFrame, surface: Point3D, t1: Point3D, t3: Point3D, height: int = 460
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["X_m"],
            y=df["Y_m"],
            mode="lines",
            name="Trajectory",
            line={"width": 4, "color": "#0B6E4F"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[surface.x, t1.x, t3.x],
            y=[surface.y, t1.y, t3.y],
            mode="markers+text",
            text=["S", "t1", "t3"],
            textposition="top center",
            name="Targets",
            marker={"size": 9, "color": ["#FF006E", "#FB5607", "#3A86FF"]},
        )
    )

    fig.update_layout(
        title="Plan view (E-N)",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def section_view_figure(
    df: pd.DataFrame,
    surface: Point3D,
    azimuth_deg: float,
    t1: Point3D,
    t3: Point3D,
    height: int = 460,
) -> go.Figure:
    vs = _section_coordinate(df=df, surface=surface, azimuth_deg=azimuth_deg)

    t_points = pd.DataFrame(
        {
            "X_m": [surface.x, t1.x, t3.x],
            "Y_m": [surface.y, t1.y, t3.y],
            "Z_m": [surface.z, t1.z, t3.z],
            "name": ["S", "t1", "t3"],
        }
    )
    t_points["VS_m"] = _section_coordinate(t_points, surface=surface, azimuth_deg=azimuth_deg)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vs,
            y=df["Z_m"],
            mode="lines",
            name="Trajectory",
            line={"width": 4, "color": "#1D3557"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_points["VS_m"],
            y=t_points["Z_m"],
            mode="markers+text",
            text=t_points["name"],
            textposition="top center",
            name="Targets",
            marker={"size": 9, "color": "#E76F51"},
        )
    )

    fig.update_layout(
        title="Vertical section",
        xaxis_title="Section coordinate (m)",
        yaxis_title="TVD (m)",
        yaxis={"autorange": "reversed"},
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def dls_figure(df: pd.DataFrame, dls_limits: dict[str, float], height: int = 560) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["MD_m"],
            y=df["DLS_deg_per_30m"],
            mode="lines+markers",
            name="DLS",
            line={"width": 2, "color": "#6A4C93"},
            marker={"size": 4},
        )
    )

    for segment, limit in dls_limits.items():
        fig.add_hline(
            y=float(limit),
            line_dash="dot",
            line_color="#C1121F",
            annotation_text=f"{segment} limit {limit:.1f}",
            annotation_position="top left",
        )

    fig.update_layout(
        title="DLS vs MD",
        xaxis_title="MD (m)",
        yaxis_title="DLS (deg/30m)",
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        showlegend=True,
    )
    return fig
