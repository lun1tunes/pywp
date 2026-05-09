from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go

from pywp.plotly_config import DEFAULT_3D_CAMERA

__all__ = [
    "WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE",
    "WT_THREE_MAX_HOVER_POINTS_PER_TRACE",
    "WT_THREE_MAX_LABELS",
    "WT_THREE_MAX_REFERENCE_LABELS",
    "customdata_row_to_hover_item",
    "decimate_hover_payload",
    "is_reference_trace_name",
    "merge_raw_bounds",
    "merge_three_line_payloads",
    "merge_three_mesh_payloads",
    "merge_three_point_payloads",
    "mesh3d_trace_to_three_payload",
    "optimize_three_payload",
    "plotly_3d_figure_to_three_payload",
    "plotly_color_and_opacity",
    "raw_bounds_from_xyz_arrays",
    "scatter3d_trace_to_three_payload",
    "split_nan_separated_xyz_segments",
    "trace_extra_name",
    "trace_showlegend",
    "trace_visibility_state",
]

WT_THREE_MAX_HOVER_POINTS_PER_TRACE = 96
WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE = 24
WT_THREE_MAX_LABELS = 48
WT_THREE_MAX_REFERENCE_LABELS = 12


def is_reference_trace_name(trace_name: str) -> bool:
    normalized = str(trace_name).strip()
    if not normalized:
        return False
    return bool(
        "(Фактическая)" in normalized
        or "(Проектная утвержденная)" in normalized
        or "Фактические скважины" in normalized
        or "Проектные утвержденные скважины" in normalized
    )


def decimate_hover_payload(
    *,
    points: list[list[float]],
    hover_items: list[dict[str, object]],
    max_points: int,
) -> tuple[list[list[float]], list[dict[str, object]]]:
    if max_points <= 1 or len(points) <= max_points:
        return points, hover_items
    indices = np.unique(
        np.linspace(0, len(points) - 1, num=int(max_points), dtype=int)
    ).tolist()
    return (
        [points[index] for index in indices],
        [hover_items[index] for index in indices],
    )


def trace_showlegend(trace: object) -> bool:
    showlegend = getattr(trace, "showlegend", None)
    if showlegend is None:
        return bool(str(getattr(trace, "name", "") or "").strip())
    return bool(showlegend)


def trace_visibility_state(trace: object) -> str:
    visible = getattr(trace, "visible", True)
    if visible is False:
        return "hidden"
    if str(visible) == "legendonly":
        return "legendonly"
    return "visible"


def plotly_color_and_opacity(
    color_value: object,
    *,
    fallback_opacity: float = 1.0,
) -> tuple[str, float]:
    color_text = str(color_value or "").strip()
    if not color_text:
        return "#94A3B8", float(np.clip(fallback_opacity, 0.0, 1.0))
    if color_text.startswith("#"):
        return color_text, float(np.clip(fallback_opacity, 0.0, 1.0))
    if color_text.startswith("rgba(") and color_text.endswith(")"):
        raw = color_text[5:-1]
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 4:
            try:
                red = int(float(parts[0]))
                green = int(float(parts[1]))
                blue = int(float(parts[2]))
                alpha = float(parts[3])
                return (
                    f"#{red:02X}{green:02X}{blue:02X}",
                    float(np.clip(alpha, 0.0, 1.0)),
                )
            except ValueError:
                pass
    if color_text.startswith("rgb(") and color_text.endswith(")"):
        raw = color_text[4:-1]
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 3:
            try:
                red = int(float(parts[0]))
                green = int(float(parts[1]))
                blue = int(float(parts[2]))
                return (
                    f"#{red:02X}{green:02X}{blue:02X}",
                    float(np.clip(fallback_opacity, 0.0, 1.0)),
                )
            except ValueError:
                pass
    return color_text, float(np.clip(fallback_opacity, 0.0, 1.0))


def trace_extra_name(trace: object) -> str:
    hovertemplate = str(getattr(trace, "hovertemplate", "") or "")
    start = hovertemplate.find("<extra>")
    end = hovertemplate.find("</extra>")
    if start >= 0 and end > start:
        return hovertemplate[start + len("<extra>") : end].strip()
    return ""


def customdata_row_to_hover_item(
    customdata_row: object,
    *,
    fallback_name: str,
) -> dict[str, object]:
    if isinstance(customdata_row, np.ndarray):
        values = customdata_row.tolist()
    elif isinstance(customdata_row, (list, tuple)):
        values = list(customdata_row)
    elif customdata_row is None:
        values = []
    else:
        values = [customdata_row]
    item: dict[str, object] = {"name": str(fallback_name).strip()}
    if values:
        first = values[0]
        try:
            first_float = float(first)
        except (TypeError, ValueError):
            first_float = None
        if first_float is not None and np.isfinite(first_float):
            item["md"] = float(first_float)
        elif str(first).strip():
            item["point"] = str(first).strip()
    if len(values) >= 2:
        try:
            second_float = float(values[1])
        except (TypeError, ValueError):
            second_float = None
        if second_float is not None and np.isfinite(second_float):
            item["dls"] = float(second_float)
    if len(values) >= 3:
        try:
            third_float = float(values[2])
        except (TypeError, ValueError):
            third_float = None
        if third_float is not None and np.isfinite(third_float):
            item["inc"] = float(third_float)
    if len(values) >= 4 and str(values[3]).strip():
        item["segment"] = str(values[3]).strip()
    return item


def split_nan_separated_xyz_segments(
    *,
    x_values: Iterable[object],
    y_values: Iterable[object],
    z_values: Iterable[object],
) -> list[list[list[float]]]:
    x_array = np.asarray(list(x_values), dtype=float)
    y_array = np.asarray(list(y_values), dtype=float)
    z_array = np.asarray(list(z_values), dtype=float)
    if not (len(x_array) == len(y_array) == len(z_array)):
        return []
    segments: list[list[list[float]]] = []
    current_segment: list[list[float]] = []
    for x_value, y_value, z_value in zip(
        x_array, y_array, z_array, strict=False
    ):
        if not (
            np.isfinite(float(x_value))
            and np.isfinite(float(y_value))
            and np.isfinite(float(z_value))
        ):
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
            continue
        current_segment.append(
            [float(x_value), float(y_value), float(z_value)]
        )
    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def scatter3d_trace_to_three_payload(
    trace: go.Scatter3d,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "lines": [],
        "points": [],
        "labels": [],
        "legend_items": [],
    }
    mode = str(trace.mode or "")
    trace_name = str(trace.name or "").strip()
    extra_name = trace_extra_name(trace)
    hover_name = trace_name or extra_name
    trace_opacity = (
        1.0
        if getattr(trace, "opacity", None) is None
        else float(trace.opacity)
    )
    visibility_state = trace_visibility_state(trace)
    is_reference_trace = is_reference_trace_name(trace_name)
    legend_added = False
    if "lines" in mode:
        line = trace.line or {}
        color, opacity = plotly_color_and_opacity(
            getattr(line, "color", None),
            fallback_opacity=trace_opacity,
        )
        if visibility_state == "visible":
            x_array = np.asarray(
                list(() if trace.x is None else trace.x), dtype=float
            )
            y_array = np.asarray(
                list(() if trace.y is None else trace.y), dtype=float
            )
            z_array = np.asarray(
                list(() if trace.z is None else trace.z), dtype=float
            )
            segments = split_nan_separated_xyz_segments(
                x_values=x_array,
                y_values=y_array,
                z_values=z_array,
            )
            if segments:
                payload["lines"].append(
                    {
                        "name": trace_name,
                        "segments": segments,
                        "color": color,
                        "opacity": opacity,
                        "dash": str(getattr(line, "dash", "solid") or "solid"),
                        "role": (
                            "cone_tip"
                            if "граница конуса" in trace_name.lower()
                            else (
                                "conflict_segment"
                                if "конфликтный участок ствола"
                                in trace_name.lower()
                                else "line"
                            )
                        ),
                    }
                )
            customdata_rows = list(
                ()
                if getattr(trace, "customdata", None) is None
                else np.asarray(trace.customdata, dtype=object)
            )
            if customdata_rows:
                hover_points: list[list[float]] = []
                hover_items: list[dict[str, object]] = []
                for index, (x_value, y_value, z_value) in enumerate(
                    zip(x_array, y_array, z_array, strict=False)
                ):
                    if not (
                        np.isfinite(float(x_value))
                        and np.isfinite(float(y_value))
                        and np.isfinite(float(z_value))
                    ):
                        continue
                    hover_points.append(
                        [float(x_value), float(y_value), float(z_value)]
                    )
                    row = (
                        customdata_rows[index]
                        if index < len(customdata_rows)
                        else None
                    )
                    hover_items.append(
                        customdata_row_to_hover_item(
                            row,
                            fallback_name=hover_name,
                        )
                    )
                if hover_points:
                    hover_points, hover_items = decimate_hover_payload(
                        points=hover_points,
                        hover_items=hover_items,
                        max_points=(
                            WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
                            if is_reference_trace
                            else WT_THREE_MAX_HOVER_POINTS_PER_TRACE
                        ),
                    )
                    payload["points"].append(
                        {
                            "name": hover_name,
                            "points": hover_points,
                            "color": color,
                            "opacity": 0.001,
                            "size": 8.5,
                            "symbol": "circle",
                            "hover": hover_items,
                            "hover_only": True,
                            "role": (
                                "reference_hover"
                                if is_reference_trace
                                else (
                                    "conflict_hover"
                                    if "конфликтный участок ствола"
                                    in hover_name.lower()
                                    else "trajectory_hover"
                                )
                            ),
                        }
                    )
        if trace_showlegend(trace) and trace_name:
            payload["legend_items"].append(
                {
                    "label": trace_name,
                    "color": color,
                    "opacity": opacity,
                }
            )
            legend_added = True
    if "markers" in mode:
        marker = trace.marker or {}
        color, opacity = plotly_color_and_opacity(
            getattr(marker, "color", None),
            fallback_opacity=trace_opacity,
        )
        if not (trace_name or opacity > 0.01):
            return payload
        if visibility_state == "visible":
            x_array = np.asarray(
                list(() if trace.x is None else trace.x), dtype=float
            )
            y_array = np.asarray(
                list(() if trace.y is None else trace.y), dtype=float
            )
            z_array = np.asarray(
                list(() if trace.z is None else trace.z), dtype=float
            )
            customdata_rows = list(
                ()
                if getattr(trace, "customdata", None) is None
                else np.asarray(trace.customdata, dtype=object)
            )
            points: list[list[float]] = []
            hover_items: list[dict[str, object]] = []
            for index, (x_value, y_value, z_value) in enumerate(
                zip(x_array, y_array, z_array, strict=False)
            ):
                if not (
                    np.isfinite(float(x_value))
                    and np.isfinite(float(y_value))
                    and np.isfinite(float(z_value))
                ):
                    continue
                points.append([float(x_value), float(y_value), float(z_value)])
                row = (
                    customdata_rows[index]
                    if index < len(customdata_rows)
                    else None
                )
                hover_items.append(
                    customdata_row_to_hover_item(
                        row,
                        fallback_name=hover_name,
                    )
                )
            if points:
                marker_size = getattr(marker, "size", 6)
                if isinstance(marker_size, (list, tuple, np.ndarray)):
                    marker_size = marker_size[0] if len(marker_size) else 6
                hover_only = opacity <= 0.01
                payload["points"].append(
                    {
                        "name": hover_name,
                        "points": points,
                        "color": color,
                        "opacity": opacity,
                        "size": (
                            max(float(marker_size), 8.5)
                            if hover_only
                            else float(marker_size)
                        ),
                        "symbol": str(
                            getattr(marker, "symbol", "circle") or "circle"
                        ),
                        "hover": hover_items,
                        "hover_only": hover_only,
                        "role": (
                            "reference_marker"
                            if is_reference_trace
                            else "marker"
                        ),
                    }
                )
        if trace_showlegend(trace) and trace_name and not legend_added:
            payload["legend_items"].append(
                {
                    "label": trace_name,
                    "color": color,
                    "opacity": opacity,
                }
            )
    if "text" in mode and "lines" not in mode and "markers" not in mode:
        text_font = trace.textfont or {}
        color = str(getattr(text_font, "color", "#0F172A"))
        if visibility_state == "visible":
            x_array = np.asarray(
                list(() if trace.x is None else trace.x), dtype=float
            )
            y_array = np.asarray(
                list(() if trace.y is None else trace.y), dtype=float
            )
            z_array = np.asarray(
                list(() if trace.z is None else trace.z), dtype=float
            )
            text_values = list(() if trace.text is None else trace.text)
            for x_value, y_value, z_value, text_value in zip(
                x_array,
                y_array,
                z_array,
                text_values,
                strict=False,
            ):
                if not (
                    np.isfinite(float(x_value))
                    and np.isfinite(float(y_value))
                    and np.isfinite(float(z_value))
                ):
                    continue
                payload["labels"].append(
                    {
                        "text": str(text_value),
                        "position": [
                            float(x_value),
                            float(y_value),
                            float(z_value),
                        ],
                        "color": color,
                        "role": (
                            "reference_pad_label"
                            if "кусты" in trace_name.lower()
                            else (
                                "reference_label"
                                if "подписи" in trace_name.lower()
                                else (
                                    "well_label"
                                    if trace_name.endswith(": t1 label")
                                    else "label"
                                )
                            )
                        ),
                    }
                )
    return payload


def mesh3d_trace_to_three_payload(
    trace: go.Mesh3d,
) -> dict[str, object] | None:
    if trace_visibility_state(trace) != "visible":
        return None
    x_array = np.asarray(list(() if trace.x is None else trace.x), dtype=float)
    y_array = np.asarray(list(() if trace.y is None else trace.y), dtype=float)
    z_array = np.asarray(list(() if trace.z is None else trace.z), dtype=float)
    i_array = np.asarray(list(() if trace.i is None else trace.i), dtype=int)
    j_array = np.asarray(list(() if trace.j is None else trace.j), dtype=int)
    k_array = np.asarray(list(() if trace.k is None else trace.k), dtype=int)
    if (
        len(x_array) == 0
        or len(y_array) != len(x_array)
        or len(z_array) != len(x_array)
        or len(i_array) == 0
        or len(i_array) != len(j_array)
        or len(i_array) != len(k_array)
    ):
        return None
    color, opacity = plotly_color_and_opacity(
        getattr(trace, "color", None),
        fallback_opacity=float(
            1.0 if getattr(trace, "opacity", None) is None else trace.opacity
        ),
    )
    return {
        "name": str(trace.name or "").strip(),
        "vertices": [
            [float(x_value), float(y_value), float(z_value)]
            for x_value, y_value, z_value in zip(
                x_array, y_array, z_array, strict=False
            )
        ],
        "faces": [
            [int(i_value), int(j_value), int(k_value)]
            for i_value, j_value, k_value in zip(
                i_array, j_array, k_array, strict=False
            )
        ],
        "color": color,
        "opacity": opacity,
        "role": (
            "cone"
            if "cone" in str(trace.name or "").lower()
            else (
                "overlap"
                if "overlap" in str(trace.name or "").lower()
                else "mesh"
            )
        ),
    }


def optimize_three_payload(payload: dict[str, object]) -> dict[str, object]:
    optimized = dict(payload)
    optimized["lines"] = merge_three_line_payloads(payload.get("lines") or [])
    optimized["points"] = merge_three_point_payloads(
        payload.get("points") or []
    )
    optimized["meshes"] = merge_three_mesh_payloads(
        payload.get("meshes") or []
    )
    labels = list(payload.get("labels") or [])
    reference_labels = [
        item
        for item in labels
        if str(item.get("role") or "") == "reference_label"
    ]
    other_labels = [
        item
        for item in labels
        if str(item.get("role") or "") != "reference_label"
    ]
    if len(reference_labels) > WT_THREE_MAX_REFERENCE_LABELS:
        indices = np.unique(
            np.linspace(
                0,
                len(reference_labels) - 1,
                num=WT_THREE_MAX_REFERENCE_LABELS,
                dtype=int,
            )
        ).tolist()
        reference_labels = [reference_labels[index] for index in indices]
    labels = other_labels + reference_labels
    if len(labels) > WT_THREE_MAX_LABELS:
        labels = labels[:WT_THREE_MAX_LABELS]
    optimized["labels"] = labels
    optimized["legend"] = list(payload.get("legend") or [])
    return optimized


def merge_three_line_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str, str], list[list[list[float]]]] = {}
    ordered_keys: list[tuple[str, float, str, str]] = []
    for item in items:
        color = str(item.get("color") or "#0F172A")
        opacity = float(item.get("opacity") or 1.0)
        dash = str(item.get("dash") or "solid")
        role = str(item.get("role") or "line")
        key = (color, opacity, dash, role)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].extend(
            [
                segment
                for segment in (item.get("segments") or [])
                if isinstance(segment, list) and len(segment) >= 2
            ]
        )
    merged: list[dict[str, object]] = []
    for color, opacity, dash, role in ordered_keys:
        segments = grouped[(color, opacity, dash, role)]
        if not segments:
            continue
        merged.append(
            {
                "segments": segments,
                "color": color,
                "opacity": float(opacity),
                "dash": dash,
                "role": role,
            }
        )
    return merged


def merge_three_point_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, float, float, str, bool, str], dict[str, object]
    ] = {}
    ordered_keys: list[tuple[str, float, float, str, bool, str]] = []
    for item in items:
        color = str(item.get("color") or "#0F172A")
        opacity = float(item.get("opacity") or 1.0)
        size = float(item.get("size") or 6.0)
        symbol = str(item.get("symbol") or "circle")
        hover_only = bool(item.get("hover_only"))
        role = str(item.get("role") or "point")
        key = (color, opacity, size, symbol, hover_only, role)
        if key not in grouped:
            grouped[key] = {
                "points": [],
                "hover": [],
                "color": color,
                "opacity": float(opacity),
                "size": float(size),
                "symbol": symbol,
                "hover_only": hover_only,
                "role": role,
            }
            ordered_keys.append(key)
        valid_points = [
            point
            for point in (item.get("points") or [])
            if isinstance(point, list) and len(point) == 3
        ]
        grouped[key]["points"].extend(valid_points)
        raw_hover = list(item.get("hover") or [])
        grouped[key]["hover"].extend(raw_hover[: len(valid_points)])
    merged: list[dict[str, object]] = []
    for key in ordered_keys:
        entry = grouped[key]
        points = list(entry["points"])
        if not points:
            continue
        merged.append(
            {
                "points": points,
                "hover": list(entry["hover"]),
                "color": str(entry["color"]),
                "opacity": float(entry["opacity"]),
                "size": float(entry["size"]),
                "symbol": str(entry["symbol"]),
                "hover_only": bool(entry["hover_only"]),
                "role": str(entry["role"]),
            }
        )
    return merged


def merge_three_mesh_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], dict[str, object]] = {}
    ordered_keys: list[tuple[str, float, str]] = []
    for item in items:
        color = str(item.get("color") or "#94A3B8")
        opacity = float(item.get("opacity") or 1.0)
        role = str(item.get("role") or "mesh")
        key = (color, opacity, role)
        if key not in grouped:
            grouped[key] = {
                "vertices": [],
                "faces": [],
                "color": color,
                "opacity": float(opacity),
                "role": role,
            }
            ordered_keys.append(key)
        merged = grouped[key]
        vertices = [
            vertex
            for vertex in (item.get("vertices") or [])
            if isinstance(vertex, list) and len(vertex) == 3
        ]
        faces = [
            face
            for face in (item.get("faces") or [])
            if isinstance(face, list) and len(face) == 3
        ]
        if not vertices or not faces:
            continue
        vertex_offset = len(merged["vertices"])
        merged["vertices"].extend(vertices)
        merged["faces"].extend(
            [
                [
                    int(face[0]) + vertex_offset,
                    int(face[1]) + vertex_offset,
                    int(face[2]) + vertex_offset,
                ]
                for face in faces
            ]
        )
    return [
        grouped[key]
        for key in ordered_keys
        if grouped[key]["vertices"] and grouped[key]["faces"]
    ]


def raw_bounds_from_xyz_arrays(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> dict[str, list[float]] | None:
    if not (len(x_values) == len(y_values) == len(z_values)):
        return None
    finite_mask = (
        np.isfinite(x_values.astype(float, copy=False))
        & np.isfinite(y_values.astype(float, copy=False))
        & np.isfinite(z_values.astype(float, copy=False))
    )
    if not finite_mask.any():
        return None
    filtered_x = x_values[finite_mask].astype(float, copy=False)
    filtered_y = y_values[finite_mask].astype(float, copy=False)
    filtered_z = z_values[finite_mask].astype(float, copy=False)
    return {
        "min": [
            float(np.min(filtered_x)),
            float(np.min(filtered_y)),
            float(np.min(filtered_z)),
        ],
        "max": [
            float(np.max(filtered_x)),
            float(np.max(filtered_y)),
            float(np.max(filtered_z)),
        ],
    }


def merge_raw_bounds(
    bounds_items: Iterable[dict[str, list[float]] | None],
) -> dict[str, list[float]] | None:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for item in bounds_items:
        if not item:
            continue
        mins.append(np.asarray(item["min"], dtype=float))
        maxs.append(np.asarray(item["max"], dtype=float))
    if not mins or not maxs:
        return None
    min_stack = np.vstack(mins)
    max_stack = np.vstack(maxs)
    return {
        "min": np.min(min_stack, axis=0).astype(float).tolist(),
        "max": np.max(max_stack, axis=0).astype(float).tolist(),
    }


def plotly_3d_figure_to_three_payload(fig: go.Figure) -> dict[str, object]:
    scene = fig.layout.scene
    x_range = (
        [float(scene.xaxis.range[0]), float(scene.xaxis.range[1])]
        if scene.xaxis.range is not None
        else [0.0, 1000.0]
    )
    y_range = (
        [float(scene.yaxis.range[0]), float(scene.yaxis.range[1])]
        if scene.yaxis.range is not None
        else [0.0, 1000.0]
    )
    z_range = (
        [float(scene.zaxis.range[0]), float(scene.zaxis.range[1])]
        if scene.zaxis.range is not None
        else [0.0, 1000.0]
    )
    camera = (
        scene.camera.to_plotly_json()
        if getattr(scene, "camera", None) is not None
        else {}
    )
    payload: dict[str, object] = {
        "background": "#FFFFFF",
        "title": str(
            getattr(getattr(fig.layout, "title", None), "text", "") or ""
        ),
        "bounds": {
            "min": [x_range[0], y_range[0], z_range[0]],
            "max": [x_range[1], y_range[1], z_range[1]],
        },
        "camera": camera or DEFAULT_3D_CAMERA,
        "lines": [],
        "meshes": [],
        "points": [],
        "labels": [],
        "legend": [],
    }
    seen_legend_labels: set[str] = set()
    for trace in fig.data:
        if isinstance(trace, go.Scatter3d):
            trace_payload = scatter3d_trace_to_three_payload(trace)
            payload["lines"].extend(trace_payload["lines"])
            payload["points"].extend(trace_payload["points"])
            payload["labels"].extend(trace_payload["labels"])
            for item in trace_payload["legend_items"]:
                label = str(item["label"])
                if not label or label in seen_legend_labels:
                    continue
                payload["legend"].append(item)
                seen_legend_labels.add(label)
            continue
        if isinstance(trace, go.Mesh3d):
            mesh_payload = mesh3d_trace_to_three_payload(trace)
            if mesh_payload is None:
                continue
            payload["meshes"].append(mesh_payload)
            if trace_showlegend(trace):
                label = str(mesh_payload["name"])
                if label and label not in seen_legend_labels:
                    payload["legend"].append(
                        {
                            "label": label,
                            "color": str(mesh_payload["color"]),
                            "opacity": float(mesh_payload["opacity"]),
                        }
                    )
                    seen_legend_labels.add(label)
    return optimize_three_payload(payload)
