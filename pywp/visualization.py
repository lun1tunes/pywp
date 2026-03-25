from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pywp.models import Point3D
from pywp.plotly_config import DEFAULT_3D_CAMERA
from pywp.plot_axes import (
    equalized_axis_ranges,
    equalized_xy_ranges,
    linear_tick_values,
    nice_tick_step,
)
from pywp.uncertainty import (
    WellUncertaintyOverlay,
    build_uncertainty_tube_mesh,
    uncertainty_ribbon_polygon,
)
from pywp.ui_utils import dls_to_pi

DEG2RAD = np.pi / 180.0
TRAJECTORY_COLOR_PRIMARY = "#C1121F"
PLAN_CSB_COLOR = "#0B6E4F"
ACTUAL_PROFILE_COLOR = "#111111"
TARGET_COLOR_PRIMARY = "#C1121F"
UNCERTAINTY_FILL_COLOR = "rgba(33, 33, 33, 0.16)"
UNCERTAINTY_SURFACE_COLOR = "#5A5A5A"
INC_LABEL_TEXT_COLOR = TARGET_COLOR_PRIMARY
INC_LABEL_TICK_COLOR = "#000000"
SEGMENT_COLORS = {
    "BUILD1": "#F77F00",
    "HOLD": "#2A9D8F",
    "BUILD2": "#D00000",
    "HORIZONTAL": "#3A86FF",
    "VERTICAL": "#5C677D",
    "BUILD": "#F77F00",
}
FALLBACK_SEGMENT_COLORS = (
    "#1D3557",
    "#8D99AE",
    "#EF476F",
    "#06D6A0",
    "#FFBE0B",
    "#8338EC",
)
HOVER_TEMPLATE_XYZ_MD_DLS = (
    "X: %{customdata[0]} m<br>"
    "Y: %{customdata[1]} m<br>"
    "Z/TVD: %{customdata[2]} m<br>"
    "MD: %{customdata[3]} m<br>"
    "ПИ: %{customdata[4]} deg/10m"
    "<extra>%{fullData.name}</extra>"
)


def _t1_label_trace_3d(
    *,
    well_name: str | None,
    t1: Point3D,
    color: str,
) -> go.Scatter3d | None:
    if not well_name:
        return None
    return go.Scatter3d(
        x=[float(t1.x)],
        y=[float(t1.y)],
        z=[float(t1.z)],
        mode="text",
        text=[str(well_name)],
        textposition="top center",
        name=f"{well_name}: t1 label",
        showlegend=False,
        textfont={"color": str(color), "size": 12},
        hoverinfo="skip",
    )


def _t1_label_trace_2d(
    *,
    well_name: str | None,
    x_value: float,
    y_value: float,
    color: str,
) -> go.Scatter | None:
    if not well_name:
        return None
    return go.Scatter(
        x=[float(x_value)],
        y=[float(y_value)],
        mode="text",
        text=[str(well_name)],
        textposition="top center",
        name=f"{well_name}: t1 label",
        showlegend=False,
        textfont={"color": str(color), "size": 12},
        hoverinfo="skip",
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(90, 90, 90, {float(alpha):.3f})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {float(alpha):.3f})"


def _lighten_hex(hex_color: str, blend_with_white: float = 0.38) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return "#A0A0A0"
    blend = float(np.clip(blend_with_white, 0.0, 1.0))
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    red = int(round(red + (255 - red) * blend))
    green = int(round(green + (255 - green) * blend))
    blue = int(round(blue + (255 - blue) * blend))
    return f"#{red:02X}{green:02X}{blue:02X}"


def _segment_blocks(segment_names: np.ndarray) -> list[tuple[int, int, str]]:
    if len(segment_names) == 0:
        return []
    blocks: list[tuple[int, int, str]] = []
    start = 0
    for idx in range(1, len(segment_names) + 1):
        if idx == len(segment_names) or segment_names[idx] != segment_names[start]:
            blocks.append((start, idx, str(segment_names[start])))
            start = idx
    return blocks


def _inc_label_candidate_indices(df: pd.DataFrame) -> list[int]:
    count = len(df)
    if count == 0:
        return []

    inc_values = df["INC_deg"].to_numpy(dtype=float)
    if "segment" in df.columns:
        segment_names = (
            df["segment"].fillna("UNKNOWN").astype(str).str.upper().to_numpy()
        )
    else:
        segment_names = np.full(count, "TRAJECTORY", dtype=object)

    candidates: list[int] = [0, count - 1]
    for start, end, segment_name in _segment_blocks(segment_names):
        if end <= start:
            continue
        block_indices = np.arange(start, end, dtype=int)
        if block_indices.size == 0:
            continue

        midpoint = int(block_indices[len(block_indices) // 2])
        candidates.append(midpoint)

        inc_delta = abs(float(inc_values[end - 1]) - float(inc_values[start]))
        is_curved_segment = (
            inc_delta >= 3.0 or "BUILD" in segment_name or "DROP" in segment_name
        )
        if is_curved_segment and block_indices.size >= 3:
            sample_count = 2 if inc_delta < 25.0 else 3
            for t_value in np.linspace(0.2, 0.8, sample_count):
                pos = int(round(t_value * (block_indices.size - 1)))
                candidates.append(int(block_indices[pos]))

    unique_sorted = sorted({int(np.clip(idx, 0, count - 1)) for idx in candidates})
    return unique_sorted


def _select_inc_label_indices(df: pd.DataFrame, section_x: np.ndarray) -> list[int]:
    count = len(df)
    if count == 0:
        return []

    z_values = df["Z_m"].to_numpy(dtype=float)
    inc_values = df["INC_deg"].to_numpy(dtype=float)
    candidates = _inc_label_candidate_indices(df=df)
    if not candidates:
        return []

    x_span = max(float(np.max(section_x) - np.min(section_x)), 1.0)
    z_span = max(float(np.max(z_values) - np.min(z_values)), 1.0)
    min_spacing = max(120.0, 0.08 * max(x_span, z_span))
    min_angle_step = 5
    max_labels = 8

    selected: list[int] = []
    seen_angles: set[int] = set()
    for idx in candidates:
        angle_int = int(np.rint(float(inc_values[idx])))
        is_forced = idx in (0, count - 1)

        if angle_int in seen_angles:
            continue

        if selected:
            prev = selected[-1]
            prev_angle = int(np.rint(float(inc_values[prev])))
            spacing = float(
                np.hypot(
                    section_x[idx] - section_x[prev], z_values[idx] - z_values[prev]
                )
            )
            if spacing < min_spacing and abs(angle_int - prev_angle) < min_angle_step:
                if not is_forced:
                    continue

        crowding = False
        for chosen in selected:
            spacing = float(
                np.hypot(
                    section_x[idx] - section_x[chosen], z_values[idx] - z_values[chosen]
                )
            )
            chosen_angle = int(np.rint(float(inc_values[chosen])))
            if spacing < 0.7 * min_spacing and abs(angle_int - chosen_angle) <= 2:
                crowding = True
                break
        if crowding and not is_forced:
            continue

        selected.append(idx)
        seen_angles.add(angle_int)

    if len(selected) <= max_labels:
        return selected

    keep: list[int] = [selected[0]]
    middle = selected[1:-1]
    take = max_labels - 2
    if middle and take > 0:
        pick = np.unique(np.linspace(0, len(middle) - 1, take, dtype=int))
        keep.extend(middle[int(pos)] for pos in pick.tolist())
    keep.append(selected[-1])
    return sorted(set(keep))


def _add_section_inc_labels(
    fig: go.Figure, df: pd.DataFrame, section_x: np.ndarray
) -> None:
    label_indices = _select_inc_label_indices(df=df, section_x=section_x)
    if not label_indices:
        return

    z_values = df["Z_m"].to_numpy(dtype=float)
    inc_values = df["INC_deg"].to_numpy(dtype=float)

    x_span = max(float(np.max(section_x) - np.min(section_x)), 1.0)
    z_span = max(float(np.max(z_values) - np.min(z_values)), 1.0)
    x_min = float(np.min(section_x))
    x_max = float(np.max(section_x))
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    tick_length_norm = 0.032
    text_gap_norm = 0.018
    text_drift_norm = 0.012
    pad_x = max(18.0, x_span * 0.02)
    pad_z = max(14.0, z_span * 0.02)
    x_low = float(min(x_min + pad_x, x_max - pad_x))
    x_high = float(max(x_min + pad_x, x_max - pad_x))
    z_low = float(min(z_min + pad_z, z_max - pad_z))
    z_high = float(max(z_min + pad_z, z_max - pad_z))

    text_x: list[float] = []
    text_y: list[float] = []
    text_labels: list[str] = []
    text_positions: list[str] = []
    occupied_text_points: list[tuple[float, float]] = []

    for rank, idx in enumerate(label_indices):
        left = max(idx - 1, 0)
        right = min(idx + 1, len(section_x) - 1)
        tangent_x = float(section_x[right] - section_x[left])
        tangent_y = float(z_values[right] - z_values[left])
        tangent_x_norm = tangent_x / x_span
        tangent_y_norm = tangent_y / z_span
        tangent_norm = float(np.hypot(tangent_x_norm, tangent_y_norm))
        if tangent_norm <= 1e-9:
            tangent_x_norm, tangent_y_norm = 1.0, 0.0
            tangent_norm = 1.0

        # Compute a normal in normalized axes space to keep the mark visually
        # perpendicular even when X/Z spans differ.
        tangent_unit_x = tangent_x_norm / tangent_norm
        tangent_unit_y = tangent_y_norm / tangent_norm
        normal_x_norm = tangent_unit_y
        normal_y_norm = -tangent_unit_x

        center_x = float(section_x[idx])
        center_y = float(z_values[idx])

        tick_dx = normal_x_norm * tick_length_norm * x_span
        tick_dy = normal_y_norm * tick_length_norm * z_span
        if float(np.hypot(tick_dx, tick_dy)) <= 1e-9:
            tick_dx = 0.0
            tick_dy = -max(24.0, 0.015 * z_span)

        def _direction_score(direction_sign: float) -> float:
            end_x = center_x + direction_sign * tick_dx
            end_y = center_y + direction_sign * tick_dy
            margin_x = min(end_x - x_min, x_max - end_x)
            margin_y = min(end_y - z_min, z_max - end_y)
            return float(margin_x + 0.4 * margin_y)

        direction = 1.0 if _direction_score(1.0) >= _direction_score(-1.0) else -1.0

        x_start = center_x
        y_start = center_y
        x_end = float(center_x + direction * tick_dx)
        y_end = float(center_y + direction * tick_dy)

        fig.add_shape(
            type="line",
            x0=x_start,
            y0=y_start,
            x1=x_end,
            y1=y_end,
            line={"color": INC_LABEL_TICK_COLOR, "width": 3},
        )

        label_dx = normal_x_norm * (tick_length_norm + text_gap_norm) * x_span
        label_dy = normal_y_norm * (tick_length_norm + text_gap_norm) * z_span
        drift_dy = text_drift_norm * z_span * (1.0 if rank % 2 == 0 else -1.0)
        label_x = float(center_x + direction * label_dx)
        label_y = float(center_y + direction * label_dy + drift_dy)
        for attempt in range(6):
            conflict = False
            for used_x, used_y in occupied_text_points:
                if np.hypot(label_x - used_x, label_y - used_y) < max(
                    24.0, 0.02 * max(x_span, z_span)
                ):
                    conflict = True
                    shift_sign = 1.0 if (rank + attempt) % 2 == 0 else -1.0
                    label_y += shift_sign * text_drift_norm * z_span
                    break
            if not conflict:
                break
        label_x = float(np.clip(label_x, x_low, x_high))
        label_y = float(np.clip(label_y, z_low, z_high))
        occupied_text_points.append((label_x, label_y))

        text_x.append(float(label_x))
        text_y.append(float(label_y))
        text_labels.append(f"{int(np.rint(float(inc_values[idx])))}°")
        text_positions.append("middle left" if direction >= 0.0 else "middle right")
    fig.add_trace(
        go.Scatter(
            x=text_x,
            y=text_y,
            mode="text",
            text=text_labels,
            textposition=text_positions,
            textfont={"size": 13, "color": INC_LABEL_TEXT_COLOR},
            cliponaxis=False,
            name="INC метки",
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _section_coordinate(
    df: pd.DataFrame, surface: Point3D, azimuth_deg: float
) -> np.ndarray:
    az = azimuth_deg * DEG2RAD
    dn = df["Y_m"].to_numpy() - surface.y
    de = df["X_m"].to_numpy() - surface.x
    return dn * np.cos(az) + de * np.sin(az)


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
    md = (
        np.full(count, np.nan, dtype=float)
        if md_values is None
        else np.asarray(md_values, dtype=float)
    )
    dls = (
        np.full(count, np.nan, dtype=float)
        if dls_values is None
        else np.asarray(dls_values, dtype=float)
    )
    pi = dls_to_pi(dls)
    return np.column_stack(
        [
            _hover_value_strings(np.asarray(x_values, dtype=float)),
            _hover_value_strings(np.asarray(y_values, dtype=float)),
            _hover_value_strings(np.asarray(z_values, dtype=float)),
            _hover_value_strings(md),
            _hover_value_strings(pi),
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


def _add_uncertainty_plan_or_section_traces(
    fig: go.Figure,
    *,
    overlay: WellUncertaintyOverlay,
    projection: str,
) -> None:
    ribbon = uncertainty_ribbon_polygon(overlay, projection=projection)
    if len(ribbon) >= 3:
        fig.add_trace(
            go.Scatter(
                x=ribbon[:, 0],
                y=ribbon[:, 1],
                mode="lines",
                name="Конус неопределенности (2σ)",
                legendgroup="uncertainty_overlay",
                showlegend=True,
                line={"width": 0.0, "color": "rgba(0, 0, 0, 0)"},
                fill="toself",
                fillcolor=UNCERTAINTY_FILL_COLOR,
                hoverinfo="skip",
            )
        )


def _add_uncertainty_3d_traces(
    fig: go.Figure,
    *,
    overlay: WellUncertaintyOverlay,
    terminal_boundary_color: str | None = None,
) -> None:
    tube_mesh = build_uncertainty_tube_mesh(overlay)
    if tube_mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=tube_mesh.vertices_xyz[:, 0],
                y=tube_mesh.vertices_xyz[:, 1],
                z=tube_mesh.vertices_xyz[:, 2],
                i=tube_mesh.i,
                j=tube_mesh.j,
                k=tube_mesh.k,
                name="Конус неопределенности (2σ)",
                legendgroup="uncertainty_overlay",
                showlegend=True,
                color=UNCERTAINTY_SURFACE_COLOR,
                opacity=0.16,
                flatshading=True,
                hoverinfo="skip",
            )
        )
    if overlay.samples and terminal_boundary_color:
        terminal_ring = np.asarray(overlay.samples[-1].ring_xyz, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=terminal_ring[:, 0],
                y=terminal_ring[:, 1],
                z=terminal_ring[:, 2],
                mode="lines",
                name="Граница конуса неопределенности",
                showlegend=False,
                line={
                    "width": 1.5,
                    "color": _lighten_hex(str(terminal_boundary_color)),
                },
                hoverinfo="skip",
            )
        )


def trajectory_3d_figure(
    df: pd.DataFrame,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    well_name: str | None = None,
    height: int = 560,
    md_t1_m: float | None = None,
    trajectory_line_dash: str = "solid",
    plan_csb_df: pd.DataFrame | None = None,
    actual_df: pd.DataFrame | None = None,
    uncertainty_overlay: WellUncertaintyOverlay | None = None,
) -> go.Figure:
    fig = go.Figure()
    x_arrays = [
        df["X_m"].to_numpy(dtype=float),
        np.array([surface.x, t1.x, t3.x], dtype=float),
    ]
    y_arrays = [
        df["Y_m"].to_numpy(dtype=float),
        np.array([surface.y, t1.y, t3.y], dtype=float),
    ]
    z_arrays = [
        df["Z_m"].to_numpy(dtype=float),
        np.array([surface.z, t1.z, t3.z], dtype=float),
    ]
    if plan_csb_df is not None and len(plan_csb_df) > 0:
        x_arrays.append(plan_csb_df["X_m"].to_numpy(dtype=float))
        y_arrays.append(plan_csb_df["Y_m"].to_numpy(dtype=float))
        z_arrays.append(plan_csb_df["Z_m"].to_numpy(dtype=float))
    if actual_df is not None and len(actual_df) > 0:
        x_arrays.append(actual_df["X_m"].to_numpy(dtype=float))
        y_arrays.append(actual_df["Y_m"].to_numpy(dtype=float))
        z_arrays.append(actual_df["Z_m"].to_numpy(dtype=float))
    if uncertainty_overlay is not None:
        for sample in uncertainty_overlay.samples:
            x_arrays.append(sample.ring_xyz[:, 0])
            y_arrays.append(sample.ring_xyz[:, 1])
            z_arrays.append(sample.ring_xyz[:, 2])
    x_values = np.concatenate(x_arrays)
    y_values = np.concatenate(y_arrays)
    z_values = np.concatenate(z_arrays)
    x_range, y_range, z_range = equalized_axis_ranges(
        x_values=x_values, y_values=y_values, z_values=z_values
    )
    xy_span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    xy_dtick = nice_tick_step(xy_span, target_ticks=6)
    x_tickvals = linear_tick_values(axis_range=x_range, step=xy_dtick)
    y_tickvals = linear_tick_values(axis_range=y_range, step=xy_dtick)
    xy_axis_style = {
        "tickmode": "array",
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
            line={
                "width": 6,
                "color": TRAJECTORY_COLOR_PRIMARY,
                "dash": str(trajectory_line_dash),
            },
            customdata=_station_hover_customdata(df),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    if uncertainty_overlay is not None and uncertainty_overlay.samples:
        _add_uncertainty_3d_traces(
            fig,
            overlay=uncertainty_overlay,
            terminal_boundary_color=TRAJECTORY_COLOR_PRIMARY,
        )
    if plan_csb_df is not None and len(plan_csb_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=plan_csb_df["X_m"],
                y=plan_csb_df["Y_m"],
                z=plan_csb_df["Z_m"],
                mode="lines",
                name="План ЦСБ",
                line={"width": 5, "color": PLAN_CSB_COLOR},
                customdata=_build_hover_customdata(
                    x_values=plan_csb_df["X_m"].to_numpy(dtype=float),
                    y_values=plan_csb_df["Y_m"].to_numpy(dtype=float),
                    z_values=plan_csb_df["Z_m"].to_numpy(dtype=float),
                ),
                hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
            )
        )
    if actual_df is not None and len(actual_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=actual_df["X_m"],
                y=actual_df["Y_m"],
                z=actual_df["Z_m"],
                mode="lines",
                name="Фактический профиль",
                line={"width": 5, "color": ACTUAL_PROFILE_COLOR},
                customdata=_build_hover_customdata(
                    x_values=actual_df["X_m"].to_numpy(dtype=float),
                    y_values=actual_df["Y_m"].to_numpy(dtype=float),
                    z_values=actual_df["Z_m"].to_numpy(dtype=float),
                ),
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
    t1_label_trace = _t1_label_trace_3d(
        well_name=well_name,
        t1=t1,
        color=TARGET_COLOR_PRIMARY,
    )
    if t1_label_trace is not None:
        fig.add_trace(t1_label_trace)

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
            "camera": DEFAULT_3D_CAMERA,
            "xaxis": {
                "range": x_range,
                "tickvals": x_tickvals,
                **xy_axis_style,
            },
            "yaxis": {
                "range": y_range,
                "tickvals": y_tickvals,
                **xy_axis_style,
            },
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
    df: pd.DataFrame,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    well_name: str | None = None,
    height: int = 460,
    trajectory_line_dash: str = "solid",
    plan_csb_df: pd.DataFrame | None = None,
    actual_df: pd.DataFrame | None = None,
    uncertainty_overlay: WellUncertaintyOverlay | None = None,
) -> go.Figure:
    x_arrays = [
        df["X_m"].to_numpy(dtype=float),
        np.array([surface.x, t1.x, t3.x], dtype=float),
    ]
    y_arrays = [
        df["Y_m"].to_numpy(dtype=float),
        np.array([surface.y, t1.y, t3.y], dtype=float),
    ]
    if plan_csb_df is not None and len(plan_csb_df) > 0:
        x_arrays.append(plan_csb_df["X_m"].to_numpy(dtype=float))
        y_arrays.append(plan_csb_df["Y_m"].to_numpy(dtype=float))
    if actual_df is not None and len(actual_df) > 0:
        x_arrays.append(actual_df["X_m"].to_numpy(dtype=float))
        y_arrays.append(actual_df["Y_m"].to_numpy(dtype=float))
    if uncertainty_overlay is not None:
        for sample in uncertainty_overlay.samples:
            x_arrays.append(sample.ring_plan_xy[:, 0])
            y_arrays.append(sample.ring_plan_xy[:, 1])
    x_values = np.concatenate(x_arrays)
    y_values = np.concatenate(y_arrays)
    x_range, y_range = equalized_xy_ranges(x_values=x_values, y_values=y_values)
    xy_dtick = nice_tick_step(
        max(x_range[1] - x_range[0], y_range[1] - y_range[0]), target_ticks=6
    )
    x_tickvals = linear_tick_values(axis_range=x_range, step=xy_dtick)
    y_tickvals = linear_tick_values(axis_range=y_range, step=xy_dtick)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["X_m"],
            y=df["Y_m"],
            mode="lines",
            name="Траектория",
            line={
                "width": 4,
                "color": TRAJECTORY_COLOR_PRIMARY,
                "dash": str(trajectory_line_dash),
            },
            customdata=_station_hover_customdata(df),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    if uncertainty_overlay is not None and uncertainty_overlay.samples:
        _add_uncertainty_plan_or_section_traces(
            fig,
            overlay=uncertainty_overlay,
            projection="plan",
        )
    if plan_csb_df is not None and len(plan_csb_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=plan_csb_df["X_m"],
                y=plan_csb_df["Y_m"],
                mode="lines",
                name="План ЦСБ",
                line={"width": 3, "color": PLAN_CSB_COLOR},
                customdata=_build_hover_customdata(
                    x_values=plan_csb_df["X_m"].to_numpy(dtype=float),
                    y_values=plan_csb_df["Y_m"].to_numpy(dtype=float),
                    z_values=plan_csb_df["Z_m"].to_numpy(dtype=float),
                ),
                hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
            )
        )
    if actual_df is not None and len(actual_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=actual_df["X_m"],
                y=actual_df["Y_m"],
                mode="lines",
                name="Фактический профиль",
                line={"width": 3, "color": ACTUAL_PROFILE_COLOR},
                customdata=_build_hover_customdata(
                    x_values=actual_df["X_m"].to_numpy(dtype=float),
                    y_values=actual_df["Y_m"].to_numpy(dtype=float),
                    z_values=actual_df["Z_m"].to_numpy(dtype=float),
                ),
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
            marker={"size": 9, "color": TARGET_COLOR_PRIMARY},
            customdata=_build_hover_customdata(
                x_values=targets_df["X_m"].to_numpy(dtype=float),
                y_values=targets_df["Y_m"].to_numpy(dtype=float),
                z_values=targets_df["Z_m"].to_numpy(dtype=float),
            ),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    t1_label_trace = _t1_label_trace_2d(
        well_name=well_name,
        x_value=float(t1.x),
        y_value=float(t1.y),
        color=TARGET_COLOR_PRIMARY,
    )
    if t1_label_trace is not None:
        fig.add_trace(t1_label_trace)

    fig.update_layout(
        title="План (E-N)",
        xaxis_title="Восток (м)",
        yaxis_title="Север (м)",
        xaxis={
            "range": x_range,
            "tickmode": "array",
            "tickvals": x_tickvals,
            "tickformat": ".0f",
            "showexponent": "none",
            "exponentformat": "none",
        },
        yaxis={
            "range": y_range,
            "tickmode": "array",
            "tickvals": y_tickvals,
            "tickformat": ".0f",
            "showexponent": "none",
            "exponentformat": "none",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
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
    well_name: str | None = None,
    height: int = 460,
    trajectory_line_dash: str = "solid",
    plan_csb_df: pd.DataFrame | None = None,
    actual_df: pd.DataFrame | None = None,
    uncertainty_overlay: WellUncertaintyOverlay | None = None,
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
    t_points["VS_m"] = _section_coordinate(
        t_points, surface=surface, azimuth_deg=azimuth_deg
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vs,
            y=df["Z_m"],
            mode="lines",
            name="Траектория",
            line={
                "width": 4,
                "color": TRAJECTORY_COLOR_PRIMARY,
                "dash": str(trajectory_line_dash),
            },
            customdata=_station_hover_customdata(df),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    if plan_csb_df is not None and len(plan_csb_df) > 0:
        plan_csb_vs = _section_coordinate(
            df=plan_csb_df,
            surface=surface,
            azimuth_deg=azimuth_deg,
        )
        fig.add_trace(
            go.Scatter(
                x=plan_csb_vs,
                y=plan_csb_df["Z_m"],
                mode="lines",
                name="План ЦСБ",
                line={"width": 3, "color": PLAN_CSB_COLOR},
                customdata=_build_hover_customdata(
                    x_values=plan_csb_df["X_m"].to_numpy(dtype=float),
                    y_values=plan_csb_df["Y_m"].to_numpy(dtype=float),
                    z_values=plan_csb_df["Z_m"].to_numpy(dtype=float),
                ),
                hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
            )
        )
    if actual_df is not None and len(actual_df) > 0:
        actual_vs = _section_coordinate(
            df=actual_df,
            surface=surface,
            azimuth_deg=azimuth_deg,
        )
        fig.add_trace(
            go.Scatter(
                x=actual_vs,
                y=actual_df["Z_m"],
                mode="lines",
                name="Фактический профиль",
                line={"width": 3, "color": ACTUAL_PROFILE_COLOR},
                customdata=_build_hover_customdata(
                    x_values=actual_df["X_m"].to_numpy(dtype=float),
                    y_values=actual_df["Y_m"].to_numpy(dtype=float),
                    z_values=actual_df["Z_m"].to_numpy(dtype=float),
                ),
                hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    if uncertainty_overlay is not None and uncertainty_overlay.samples:
        _add_uncertainty_plan_or_section_traces(
            fig,
            overlay=uncertainty_overlay,
            projection="section",
        )
    fig.add_trace(
        go.Scatter(
            x=t_points["VS_m"],
            y=t_points["Z_m"],
            mode="markers+text",
            text=t_points["name"],
            textposition="top center",
            name="Цели",
            marker={"size": 9, "color": TARGET_COLOR_PRIMARY},
            customdata=_build_hover_customdata(
                x_values=t_points["X_m"].to_numpy(dtype=float),
                y_values=t_points["Y_m"].to_numpy(dtype=float),
                z_values=t_points["Z_m"].to_numpy(dtype=float),
            ),
            hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
        )
    )
    t1_label_trace = _t1_label_trace_2d(
        well_name=well_name,
        x_value=float(t_points.loc[1, "VS_m"]),
        y_value=float(t_points.loc[1, "Z_m"]),
        color=TARGET_COLOR_PRIMARY,
    )
    if t1_label_trace is not None:
        fig.add_trace(t1_label_trace)
    _add_section_inc_labels(fig=fig, df=df, section_x=vs)

    fig.update_layout(
        title="Вертикальный разрез",
        xaxis_title="Координата по разрезу (м)",
        yaxis_title="TVD (м)",
        yaxis={"autorange": "reversed"},
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def dls_figure(
    df: pd.DataFrame, dls_limits: dict[str, float], height: int = 560
) -> go.Figure:
    fig = go.Figure()
    active_segments: set[str] | None = None
    if "segment" not in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["MD_m"],
                y=dls_to_pi(df["DLS_deg_per_30m"].to_numpy(dtype=float)),
                mode="lines+markers",
                name="ПИ",
                line={"width": 2, "color": "#6A4C93"},
                marker={"size": 4},
                customdata=_station_hover_customdata(df),
                hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
            )
        )
    else:
        segments = df["segment"].fillna("UNKNOWN").astype(str).str.upper().to_numpy()
        active_segments = set(segments.tolist())
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
                    y=dls_to_pi(block["DLS_deg_per_30m"].to_numpy(dtype=float)),
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

        # Draw dashed links where segment traces break to keep PI transitions readable.
        md_values = df["MD_m"].to_numpy(dtype=float)
        dls_values = dls_to_pi(df["DLS_deg_per_30m"].to_numpy(dtype=float))
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
                    line={
                        "width": 2,
                        "color": "rgba(120, 120, 120, 0.9)",
                        "dash": "dash",
                    },
                    customdata=_station_hover_customdata(transition_df),
                    hovertemplate=HOVER_TEMPLATE_XYZ_MD_DLS,
                )
            )

    # Avoid overlapping labels for equal PI limits (e.g., BUILD1/BUILD2 or HOLD/HORIZONTAL).
    grouped_limits: dict[float, list[str]] = {}
    for segment, limit in dls_limits.items():
        segment_name = str(segment).upper()
        if active_segments is not None and segment_name not in active_segments:
            continue
        key = round(float(dls_to_pi(limit)), 6)
        grouped_limits.setdefault(key, []).append(segment_name)

    for limit_value in sorted(grouped_limits.keys(), reverse=True):
        segment_label = "/".join(grouped_limits[limit_value])
        fig.add_hline(
            y=limit_value,
            line_dash="dot",
            line_color="#C1121F",
            annotation_text=f"{segment_label} лимит ПИ {limit_value:.2f}",
            annotation_position="top left",
        )

    fig.update_layout(
        title="ПИ vs MD",
        xaxis_title="MD (m)",
        yaxis_title="ПИ (deg/10m)",
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        showlegend=True,
    )
    return fig
