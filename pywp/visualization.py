from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pywp.models import Point3D

DEG2RAD = np.pi / 180.0
SEGMENT_COLORS = {
    "BUILD1": "#F77F00",
    "HOLD": "#2A9D8F",
    "BUILD2": "#D00000",
    "BUILD_REV": "#8E44AD",
    "HOLD_REV": "#7D8597",
    "DROP_REV": "#A06CD5",
    "HORIZONTAL": "#3A86FF",
    "VERTICAL": "#5C677D",
    "BUILD": "#F77F00",
}
FALLBACK_SEGMENT_COLORS = ("#1D3557", "#8D99AE", "#EF476F", "#06D6A0", "#FFBE0B", "#8338EC")
HOVER_TEMPLATE_XYZ_MD_DLS = (
    "X: %{customdata[0]} m<br>"
    "Y: %{customdata[1]} m<br>"
    "Z/TVD: %{customdata[2]} m<br>"
    "MD: %{customdata[3]} m<br>"
    "DLS: %{customdata[4]} deg/30m"
    "<extra>%{fullData.name}</extra>"
)


def _section_coordinate(df: pd.DataFrame, surface: Point3D, azimuth_deg: float) -> np.ndarray:
    az = azimuth_deg * DEG2RAD
    dn = df["Y_m"].to_numpy() - surface.y
    de = df["X_m"].to_numpy() - surface.x
    return dn * np.cos(az) + de * np.sin(az)


def _equalized_axis_ranges(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    pad_fraction: float = 0.08,
) -> tuple[list[float], list[float], list[float]]:
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))

    xy_min = min(x_min, y_min)
    xy_max = max(x_max, y_max)
    xy_span = xy_max - xy_min

    span = max(xy_span, z_max - z_min, 1.0)
    span = span * (1.0 + pad_fraction)

    xy_center = (xy_min + xy_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    half = span / 2.0
    # Keep X and Y on the exact same range so both axes have identical scale and ticks.
    x_range = [xy_center - half, xy_center + half]
    y_range = [xy_center - half, xy_center + half]
    # Reverse Z to keep depth increasing downward while maintaining equal scale.
    z_range = [z_center + half, z_center - half]
    return x_range, y_range, z_range


def _nice_tick_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0.0:
        return 1.0
    raw = span / max(target_ticks, 1)
    exponent = np.floor(np.log10(raw))
    base = 10.0**exponent
    fraction = raw / base
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * base)


def _linear_tick_values(axis_range: list[float], step: float) -> list[float]:
    lo = min(axis_range[0], axis_range[1])
    hi = max(axis_range[0], axis_range[1])
    if step <= 0.0:
        return [float(lo), float(hi)]
    first = np.floor(lo / step) * step
    values = np.arange(first, hi + step * 0.5, step, dtype=float)
    if values.size == 0:
        values = np.array([lo, hi], dtype=float)
    return np.round(values, 6).tolist()


def _segment_color(segment_name: str) -> str:
    normalized = segment_name.strip().upper()
    if normalized in SEGMENT_COLORS:
        return SEGMENT_COLORS[normalized]
    key = sum(ord(char) for char in normalized)
    return FALLBACK_SEGMENT_COLORS[key % len(FALLBACK_SEGMENT_COLORS)]


def _hover_value_strings(values: np.ndarray, digits: int = 2) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    formatted = np.char.mod(f"%.{digits}f", safe)
    return np.where(np.isfinite(arr), formatted, "n/a")


def _build_hover_customdata(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    md_values: np.ndarray | None = None,
    dls_values: np.ndarray | None = None,
) -> np.ndarray:
    count = len(x_values)
    md = np.full(count, np.nan, dtype=float) if md_values is None else np.asarray(md_values, dtype=float)
    dls = np.full(count, np.nan, dtype=float) if dls_values is None else np.asarray(dls_values, dtype=float)
    return np.column_stack(
        [
            _hover_value_strings(np.asarray(x_values, dtype=float)),
            _hover_value_strings(np.asarray(y_values, dtype=float)),
            _hover_value_strings(np.asarray(z_values, dtype=float)),
            _hover_value_strings(md),
            _hover_value_strings(dls),
        ]
    )


def _station_hover_customdata(df: pd.DataFrame) -> np.ndarray:
    dls_values = (
        df["DLS_deg_per_30m"].to_numpy(dtype=float)
        if "DLS_deg_per_30m" in df.columns
        else np.full(len(df), np.nan, dtype=float)
    )
    return _build_hover_customdata(
        x_values=df["X_m"].to_numpy(dtype=float),
        y_values=df["Y_m"].to_numpy(dtype=float),
        z_values=df["Z_m"].to_numpy(dtype=float),
        md_values=df["MD_m"].to_numpy(dtype=float),
        dls_values=dls_values,
    )


def trajectory_3d_figure(
    df: pd.DataFrame,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    height: int = 560,
    md_t1_m: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    x_values = np.concatenate([df["X_m"].to_numpy(), np.array([surface.x, t1.x, t3.x])])
    y_values = np.concatenate([df["Y_m"].to_numpy(), np.array([surface.y, t1.y, t3.y])])
    z_values = np.concatenate([df["Z_m"].to_numpy(), np.array([surface.z, t1.z, t3.z])])
    x_range, y_range, z_range = _equalized_axis_ranges(x_values=x_values, y_values=y_values, z_values=z_values)
    xy_span = x_range[1] - x_range[0]
    xy_dtick = _nice_tick_step(xy_span, target_ticks=6)
    xy_tick0 = float(np.floor(min(x_range[0], y_range[0]) / xy_dtick) * xy_dtick)
    xy_tickvals = _linear_tick_values(axis_range=x_range, step=xy_dtick)
    xy_axis_style = {
        "tickmode": "array",
        "tickvals": xy_tickvals,
        "dtick": xy_dtick,
        "tick0": xy_tick0,
        "tickformat": ".0f",
        "showexponent": "none",
        "exponentformat": "none",
        "showgrid": True,
        "gridcolor": "rgba(0, 0, 0, 0.15)",
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": "rgba(0, 0, 0, 0.65)",
        "zerolinewidth": 2,
        "showline": True,
        "linecolor": "rgba(0, 0, 0, 0.65)",
        "linewidth": 1.5,
    }

    fig.add_trace(
        go.Scatter3d(
            x=df["X_m"],
            y=df["Y_m"],
            z=df["Z_m"],
            mode="lines",
            name="Траектория",
            line={"width": 6, "color": "#006D77"},
            customdata=_station_hover_customdata(df),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )

    targets_df = pd.DataFrame(
        {
            "X_m": [surface.x, t1.x, t3.x],
            "Y_m": [surface.y, t1.y, t3.y],
            "Z_m": [surface.z, t1.z, t3.z],
        }
    )
    fig.add_trace(
        go.Scatter3d(
            x=[surface.x, t1.x, t3.x],
            y=[surface.y, t1.y, t3.y],
            z=[surface.z, t1.z, t3.z],
            mode="markers+text",
            text=["S", "t1", "t3"],
            textposition="top center",
            name="Цели",
            marker={"size": 6, "color": ["#EF476F", "#FFD166", "#118AB2"]},
            customdata=_build_hover_customdata(
                x_values=targets_df["X_m"].to_numpy(dtype=float),
                y_values=targets_df["Y_m"].to_numpy(dtype=float),
                z_values=targets_df["Z_m"].to_numpy(dtype=float),
            ),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )

    if md_t1_m is not None:
        t1_idx = int((df["MD_m"] - float(md_t1_m)).abs().idxmin())
        t1_calc = df.loc[t1_idx]
        t3_calc = df.iloc[-1]
        calc_df = pd.DataFrame([t1_calc, t3_calc]).reset_index(drop=True)
        fig.add_trace(
            go.Scatter3d(
                x=[float(t1_calc["X_m"]), float(t3_calc["X_m"])],
                y=[float(t1_calc["Y_m"]), float(t3_calc["Y_m"])],
                z=[float(t1_calc["Z_m"]), float(t3_calc["Z_m"])],
                mode="markers",
                name="Расчетные в t1/t3",
                marker={"size": 5, "color": "#073B4C", "symbol": "diamond"},
                customdata=_station_hover_customdata(calc_df),
                hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
            )
        )

    zero_axis_color = "rgba(0, 0, 0, 0.70)"
    z_zero_ref = float(surface.z)
    if y_range[0] <= 0.0 <= y_range[1]:
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, 0.0],
                y=[float(y_range[0]), float(y_range[1])],
                z=[z_zero_ref, z_zero_ref],
                mode="lines",
                name="Y=0 axis",
                showlegend=False,
                hoverinfo="skip",
                line={"width": 4, "color": zero_axis_color},
            )
        )
    if x_range[0] <= 0.0 <= x_range[1]:
        fig.add_trace(
            go.Scatter3d(
                x=[float(x_range[0]), float(x_range[1])],
                y=[0.0, 0.0],
                z=[z_zero_ref, z_zero_ref],
                mode="lines",
                name="X=0 axis",
                showlegend=False,
                hoverinfo="skip",
                line={"width": 4, "color": zero_axis_color},
            )
        )

    fig.update_layout(
        title="3D траектория",
        scene={
            "xaxis_title": "X / Восток (м)",
            "yaxis_title": "Y / Север (м)",
            "zaxis_title": "Z / TVD (m)",
            "xaxis": {"range": x_range, **xy_axis_style},
            "yaxis": {"range": y_range, **xy_axis_style},
            "zaxis": {
                "range": z_range,
                "tickformat": ".0f",
                "showexponent": "none",
                "exponentformat": "none",
                "showgrid": True,
                "gridcolor": "rgba(0, 0, 0, 0.12)",
                "gridwidth": 1,
                "zeroline": True,
                "zerolinecolor": "rgba(0, 0, 0, 0.45)",
                "zerolinewidth": 1,
            },
            "aspectmode": "cube",
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
            name="Траектория",
            line={"width": 4, "color": "#0B6E4F"},
            customdata=_station_hover_customdata(df),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    targets_df = pd.DataFrame(
        {
            "X_m": [surface.x, t1.x, t3.x],
            "Y_m": [surface.y, t1.y, t3.y],
            "Z_m": [surface.z, t1.z, t3.z],
        }
    )
    fig.add_trace(
        go.Scatter(
            x=[surface.x, t1.x, t3.x],
            y=[surface.y, t1.y, t3.y],
            mode="markers+text",
            text=["S", "t1", "t3"],
            textposition="top center",
            name="Цели",
            marker={"size": 9, "color": ["#FF006E", "#FB5607", "#3A86FF"]},
            customdata=_build_hover_customdata(
                x_values=targets_df["X_m"].to_numpy(dtype=float),
                y_values=targets_df["Y_m"].to_numpy(dtype=float),
                z_values=targets_df["Z_m"].to_numpy(dtype=float),
            ),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )

    fig.update_layout(
        title="План (E-N)",
        xaxis_title="Восток (м)",
        yaxis_title="Север (м)",
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
            name="Траектория",
            line={"width": 4, "color": "#1D3557"},
            customdata=_station_hover_customdata(df),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_points["VS_m"],
            y=t_points["Z_m"],
            mode="markers+text",
            text=t_points["name"],
            textposition="top center",
            name="Цели",
            marker={"size": 9, "color": "#E76F51"},
            customdata=_build_hover_customdata(
                x_values=t_points["X_m"].to_numpy(dtype=float),
                y_values=t_points["Y_m"].to_numpy(dtype=float),
                z_values=t_points["Z_m"].to_numpy(dtype=float),
            ),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )

    fig.update_layout(
        title="Вертикальный разрез",
        xaxis_title="Координата по разрезу (м)",
        yaxis_title="TVD (м)",
        yaxis={"autorange": "reversed"},
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def dls_figure(df: pd.DataFrame, dls_limits: dict[str, float], height: int = 560) -> go.Figure:
    fig = go.Figure()
    if "segment" not in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["MD_m"],
                y=df["DLS_deg_per_30m"],
                mode="lines+markers",
                name="DLS",
                line={"width": 2, "color": "#6A4C93"},
                marker={"size": 4},
                customdata=_station_hover_customdata(df),
                hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
            )
        )
    else:
        segments = df["segment"].fillna("UNKNOWN").astype(str).str.upper().to_numpy()
        legend_shown: set[str] = set()
        start_idx = 0
        for idx in range(1, len(df) + 1):
            is_boundary = idx == len(df) or segments[idx] != segments[start_idx]
            if not is_boundary:
                continue

            segment_name = segments[start_idx]
            block = df.iloc[start_idx:idx]
            color = _segment_color(segment_name)
            fig.add_trace(
                go.Scatter(
                    x=block["MD_m"],
                    y=block["DLS_deg_per_30m"],
                    mode="lines+markers",
                    name=segment_name,
                    legendgroup=f"segment_{segment_name}",
                    showlegend=segment_name not in legend_shown,
                    line={"width": 3, "color": color},
                    marker={"size": 5, "color": color},
                    customdata=_station_hover_customdata(block),
                    hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
                )
            )
            legend_shown.add(segment_name)
            start_idx = idx

        # Draw dashed links where segment traces break to keep DLS transitions readable.
        md_values = df["MD_m"].to_numpy(dtype=float)
        dls_values = df["DLS_deg_per_30m"].to_numpy(dtype=float)
        for idx in range(1, len(df)):
            if segments[idx] == segments[idx - 1]:
                continue
            y0 = dls_values[idx - 1]
            y1 = dls_values[idx]
            if np.isnan(y0) or np.isnan(y1):
                continue
            transition_df = df.iloc[idx - 1 : idx + 1]
            fig.add_trace(
                go.Scatter(
                    x=[md_values[idx - 1], md_values[idx]],
                    y=[y0, y1],
                    mode="lines",
                    name="Переход",
                    legendgroup="transition",
                    showlegend=False,
                    line={"width": 2, "color": "rgba(120, 120, 120, 0.9)", "dash": "dash"},
                    customdata=_station_hover_customdata(transition_df),
                    hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
                )
            )

    # Avoid overlapping labels for equal DLS limits (e.g., BUILD1/BUILD2 or HOLD/HORIZONTAL).
    grouped_limits: dict[float, list[str]] = {}
    for segment, limit in dls_limits.items():
        key = round(float(limit), 6)
        grouped_limits.setdefault(key, []).append(str(segment))

    for limit_value in sorted(grouped_limits.keys(), reverse=True):
        segment_label = "/".join(grouped_limits[limit_value])
        fig.add_hline(
            y=limit_value,
            line_dash="dot",
            line_color="#C1121F",
            annotation_text=f"{segment_label} лимит {limit_value:.1f}",
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
