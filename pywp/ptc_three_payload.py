from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

__all__ = [
    "WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE",
    "WT_THREE_MAX_HOVER_POINTS_PER_TRACE",
    "WT_THREE_MAX_LABELS",
    "WT_THREE_MAX_REFERENCE_LABELS",
    "decimate_hover_payload",
    "merge_raw_bounds",
    "merge_three_line_payloads",
    "merge_three_mesh_payloads",
    "merge_three_point_payloads",
    "optimize_three_payload",
    "raw_bounds_from_xyz_arrays",
]

WT_THREE_MAX_HOVER_POINTS_PER_TRACE = 96
WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE = 24
WT_THREE_MAX_LABELS = 48
WT_THREE_MAX_REFERENCE_LABELS = 12


def _float_or_default(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


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


def optimize_three_payload(payload: dict[str, object]) -> dict[str, object]:
    optimized = dict(payload)
    optimized["lines"] = merge_three_line_payloads(payload.get("lines") or [])
    optimized["points"] = merge_three_point_payloads(payload.get("points") or [])
    optimized["meshes"] = merge_three_mesh_payloads(payload.get("meshes") or [])

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
    grouped: dict[tuple[str, str, float, str, str], list[list[list[float]]]] = {}
    ordered_keys: list[tuple[str, str, float, str, str]] = []
    for item in items:
        name = str(item.get("name") or "")
        color = str(item.get("color") or "#0F172A")
        opacity = _float_or_default(item.get("opacity"), 1.0)
        dash = str(item.get("dash") or "solid")
        role = str(item.get("role") or "line")
        key = (name, color, opacity, dash, role)
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
    for name, color, opacity, dash, role in ordered_keys:
        segments = grouped[(name, color, opacity, dash, role)]
        if not segments:
            continue
        merged.append(
            {
                "name": name,
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
        tuple[str, str, float, float, str, bool, str], dict[str, object]
    ] = {}
    ordered_keys: list[tuple[str, str, float, float, str, bool, str]] = []
    for item in items:
        name = str(item.get("name") or "")
        color = str(item.get("color") or "#0F172A")
        opacity = _float_or_default(item.get("opacity"), 1.0)
        size = float(item.get("size") or 6.0)
        symbol = str(item.get("symbol") or "circle")
        hover_only = bool(item.get("hover_only"))
        role = str(item.get("role") or "point")
        key = (name, color, opacity, size, symbol, hover_only, role)
        if key not in grouped:
            grouped[key] = {
                "name": name,
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

        raw_hover = list(item.get("hover") or [])
        for point_index, point in enumerate(item.get("points") or []):
            if not isinstance(point, list) or len(point) != 3:
                continue
            grouped[key]["points"].append(point)
            hover_item = raw_hover[point_index] if point_index < len(raw_hover) else {}
            grouped[key]["hover"].append(
                dict(hover_item) if isinstance(hover_item, Mapping) else {}
            )

    merged: list[dict[str, object]] = []
    for key in ordered_keys:
        entry = grouped[key]
        points = list(entry["points"])
        if not points:
            continue
        merged.append(
            {
                "name": str(entry["name"]),
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
    grouped: dict[tuple[str, str, float, str], dict[str, object]] = {}
    ordered_keys: list[tuple[str, str, float, str]] = []
    passthrough_items: list[dict[str, object]] = []
    for item in items:
        name = str(item.get("name") or "")
        color = str(item.get("color") or "#94A3B8")
        opacity = _float_or_default(item.get("opacity"), 1.0)
        role = str(item.get("role") or "mesh")
        if role == "overlap_volume":
            passthrough_items.append(dict(item))
            continue

        key = (name, color, opacity, role)
        if key not in grouped:
            grouped[key] = {
                "name": name,
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

    merged_items = [
        grouped[key]
        for key in ordered_keys
        if grouped[key]["vertices"] and grouped[key]["faces"]
    ]
    return [*merged_items, *passthrough_items]


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
