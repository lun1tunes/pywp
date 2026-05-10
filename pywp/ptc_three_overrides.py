from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping

import numpy as np

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionZone,
    _segment_types_for_interval,
    anti_collision_report_events,
)
from pywp.eclipse_welltrack import WelltrackRecord
from pywp.models import Point3D
from pywp import ptc_pad_state
from pywp import ptc_three_payload
from pywp.well_pad import WellPad
from pywp.welltrack_batch import SuccessfulWellPlan

__all__ = [
    "anticollision_three_payload_overrides",
    "augment_three_payload",
    "build_edit_wells_payload",
    "legend_pad_label",
    "overlap_volume_payloads",
    "pad_first_surface_arrow_payloads",
    "successful_plan_raw_bounds",
    "target_only_raw_bounds",
    "three_legend_tree_payload",
    "trajectory_three_payload_overrides",
]

MAX_EDIT_BASE_POINTS = 700


def successful_plan_raw_bounds(
    success: SuccessfulWellPlan,
) -> dict[str, list[float]] | None:
    stations = success.stations
    point_bounds = _point_triplet_bounds(success.surface, success.t1, success.t3)
    if stations.empty or not {"X_m", "Y_m", "Z_m"}.issubset(stations.columns):
        return point_bounds
    station_bounds = ptc_three_payload.raw_bounds_from_xyz_arrays(
        x_values=stations["X_m"].to_numpy(dtype=float),
        y_values=stations["Y_m"].to_numpy(dtype=float),
        z_values=stations["Z_m"].to_numpy(dtype=float),
    )
    return ptc_three_payload.merge_raw_bounds((station_bounds, point_bounds))


def target_only_raw_bounds(target_only: object) -> dict[str, list[float]]:
    return _point_triplet_bounds(
        getattr(target_only, "surface"),
        getattr(target_only, "t1"),
        getattr(target_only, "t3"),
    ) or {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}


def legend_pad_label(pad: WellPad) -> str:
    return f"Куст {str(pad.pad_id)}"


def three_legend_tree_payload(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    visible_well_names: Iterable[str],
    well_bounds_by_name: Mapping[str, dict[str, list[float]]],
    name_to_color: Mapping[str, str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, list[float]]], set[str]]:
    visible_set = {str(name) for name in visible_well_names if str(name).strip()}
    if not visible_set:
        return [], {}, set()
    pads, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )
    tree: list[dict[str, object]] = []
    focus_targets: dict[str, dict[str, list[float]]] = {}
    hidden_flat_legend_labels: set[str] = set()
    for pad in pads:
        pad_id = str(pad.pad_id)
        ordered_names = [
            str(name)
            for name in well_names_by_pad_id.get(pad_id, ())
            if str(name) in visible_set and str(name) in well_bounds_by_name
        ]
        if not ordered_names:
            continue
        child_nodes = []
        child_bounds = []
        for well_name in ordered_names:
            focus_id = f"well::{well_name}"
            child_nodes.append(
                {
                    "id": focus_id,
                    "label": str(well_name),
                    "color": str(name_to_color.get(str(well_name), "#64748b")),
                }
            )
            focus_targets[focus_id] = dict(well_bounds_by_name[well_name])
            child_bounds.append(well_bounds_by_name[well_name])
            hidden_flat_legend_labels.add(str(well_name))
        pad_focus_id = f"pad::{pad_id}"
        merged_pad_bounds = ptc_three_payload.merge_raw_bounds(child_bounds)
        if merged_pad_bounds is not None:
            focus_targets[pad_focus_id] = merged_pad_bounds
        tree.append(
            {
                "id": pad_focus_id,
                "label": legend_pad_label(pad),
                "children": child_nodes,
            }
        )
    return tree, focus_targets, hidden_flat_legend_labels


def pad_first_surface_arrow_payloads(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    visible_well_names: Iterable[str],
    surface_by_name: Mapping[str, Point3D],
) -> list[dict[str, object]]:
    visible_set = {str(name) for name in visible_well_names if str(name).strip()}
    if not visible_set:
        return []
    pads, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )
    arrows: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for pad in pads:
        cfg = ptc_pad_state.pad_config_for_ui(session_state, pad)
        ordered_visible_names = [
            str(name)
            for name in well_names_by_pad_id.get(str(pad.pad_id), ())
            if str(name) in visible_set and str(name) in surface_by_name
        ]
        if not ordered_visible_names:
            continue
        first_name = ordered_visible_names[0]
        if first_name in seen_names:
            continue
        seen_names.add(first_name)
        surface = surface_by_name[first_name]
        arrows.append(
            _pad_first_surface_arrow_payload(
                surface=surface,
                nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
                spacing_m=float(cfg["spacing_m"]),
                well_name=first_name,
                pad_id=str(pad.pad_id),
            )
        )
    return arrows


def _pad_first_surface_arrow_payload(
    *,
    surface: Point3D,
    nds_azimuth_deg: float,
    spacing_m: float,
    well_name: str,
    pad_id: str,
) -> dict[str, object]:
    length_m = float(np.clip(float(spacing_m) * 0.55, 14.0, 34.0))
    shaft_width_m = max(length_m * 0.12, 2.0)
    head_length_m = length_m * 0.32
    head_width_m = length_m * 0.34
    vertical_lift_m = float(np.clip(length_m * 0.28, 4.0, 10.0))

    angle_rad = np.deg2rad(float(nds_azimuth_deg) % 360.0)
    direction = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=float)
    if float(np.linalg.norm(direction)) <= 1e-9:
        direction = np.array([0.0, 1.0], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    center = np.array([float(surface.x), float(surface.y)], dtype=float)

    tail = center - direction * (length_m * 0.5)
    tip = center + direction * (length_m * 0.5)
    head_base = tip - direction * head_length_m
    z_value = float(surface.z) - vertical_lift_m

    shaft_half = normal * (shaft_width_m * 0.5)
    head_half = normal * (head_width_m * 0.5)
    vertices_xy = [
        tail - shaft_half,
        tail + shaft_half,
        head_base + shaft_half,
        head_base - shaft_half,
        head_base + head_half,
        tip,
        head_base - head_half,
    ]
    return {
        "name": "Начало порядка скважин",
        "vertices": [
            [float(point[0]), float(point[1]), z_value] for point in vertices_xy
        ],
        "faces": [[0, 1, 2], [0, 2, 3], [4, 5, 6]],
        "color": "#64748B",
        "opacity": 0.86,
        "role": "pad_first_surface_arrow",
        "well_name": str(well_name),
        "pad_id": str(pad_id),
        "nds_azimuth_deg": float(nds_azimuth_deg) % 360.0,
    }


def augment_three_payload(
    *,
    payload: dict[str, object],
    legend_tree: list[dict[str, object]] | None = None,
    focus_targets: Mapping[str, dict[str, list[float]]] | None = None,
    hidden_flat_legend_labels: set[str] | None = None,
    collisions: list[dict[str, object]] | None = None,
    edit_wells: list[dict[str, object]] | None = None,
    extra_labels: list[dict[str, object]] | None = None,
    extra_meshes: list[dict[str, object]] | None = None,
    extra_legend_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    updated = dict(payload)
    if legend_tree:
        updated["legend_tree"] = list(legend_tree)
    if focus_targets:
        updated["focus_targets"] = {
            str(key): dict(value) for key, value in focus_targets.items()
        }
    hidden_labels = {str(label) for label in (hidden_flat_legend_labels or set())}
    if hidden_labels:
        updated["legend"] = [
            item
            for item in list(updated.get("legend") or [])
            if str(dict(item).get("label", "")).strip() not in hidden_labels
        ]
    if collisions is not None:
        updated["collisions"] = list(collisions)
    if edit_wells is not None:
        updated["edit_wells"] = list(edit_wells)
    if extra_labels:
        updated["labels"] = [
            *list(updated.get("labels") or []),
            *list(extra_labels),
        ]
    if extra_meshes:
        updated["meshes"] = [
            *list(updated.get("meshes") or []),
            *list(extra_meshes),
        ]
    if extra_legend_items:
        seen_labels = {
            str(dict(item).get("label", "")).strip()
            for item in list(updated.get("legend") or [])
        }
        legend_items = list(updated.get("legend") or [])
        for item in extra_legend_items:
            label = str(dict(item).get("label", "")).strip()
            if not label or label in seen_labels:
                continue
            legend_items.append(dict(item))
            seen_labels.add(label)
        updated["legend"] = legend_items
    return updated


def build_edit_wells_payload(
    successes: list[SuccessfulWellPlan],
    name_to_color: Mapping[str, str],
) -> list[dict[str, object]]:
    edit_wells: list[dict[str, object]] = []
    for success in successes:
        config = success.config
        base_points = _decimated_base_points(success)
        edit_wells.append(
            {
                "name": str(success.name),
                "surface": _point3d_payload(success.surface),
                "t1": _point3d_payload(success.t1),
                "t3": _point3d_payload(success.t3),
                "color": str(name_to_color.get(str(success.name), "#2563eb")),
                "base_points": base_points,
                "config": {
                    "entry_inc_target_deg": float(config.entry_inc_target_deg),
                    "max_inc_deg": float(config.max_inc_deg),
                    "dls_build_max_deg_per_30m": float(
                        config.dls_build_max_deg_per_30m
                    ),
                    "kop_min_vertical_m": float(config.kop_min_vertical_m),
                },
            }
        )
    return edit_wells


def trajectory_three_payload_overrides(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    successes: list[SuccessfulWellPlan],
    target_only_wells: list[object],
    name_to_color: Mapping[str, str],
) -> dict[str, object]:
    well_bounds_by_name: dict[str, dict[str, list[float]]] = {}
    surface_by_name: dict[str, Point3D] = {}
    for success in successes:
        bounds = successful_plan_raw_bounds(success)
        if bounds is not None:
            well_bounds_by_name[str(success.name)] = bounds
            surface_by_name[str(success.name)] = success.surface
    for target_only in target_only_wells:
        well_name = str(getattr(target_only, "name"))
        well_bounds_by_name[well_name] = target_only_raw_bounds(target_only)
        surface_by_name[well_name] = getattr(target_only, "surface")
    legend_tree, focus_targets, hidden_labels = three_legend_tree_payload(
        session_state,
        records=records,
        visible_well_names=tuple(well_bounds_by_name.keys()),
        well_bounds_by_name=well_bounds_by_name,
        name_to_color=name_to_color,
    )
    return {
        "legend_tree": legend_tree,
        "focus_targets": focus_targets,
        "hidden_flat_legend_labels": hidden_labels,
        "edit_wells": build_edit_wells_payload(successes, name_to_color),
        "extra_meshes": pad_first_surface_arrow_payloads(
            session_state,
            records=records,
            visible_well_names=tuple(well_bounds_by_name.keys()),
            surface_by_name=surface_by_name,
        ),
        "component_key": "trajectory-overview",
    }


def anticollision_three_payload_overrides(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    analysis: AntiCollisionAnalysis,
    successes: list[SuccessfulWellPlan] | None = None,
) -> dict[str, object]:
    visible_names: list[str] = []
    well_bounds_by_name: dict[str, dict[str, list[float]]] = {}
    name_to_color: dict[str, str] = {}
    surface_by_name: dict[str, Point3D] = {}
    for well in analysis.wells:
        if bool(well.is_reference_only):
            continue
        visible_names.append(str(well.name))
        name_to_color[str(well.name)] = str(well.color)
        surface_by_name[str(well.name)] = well.surface
        bounds = ptc_three_payload.raw_bounds_from_xyz_arrays(
            x_values=well.stations["X_m"].to_numpy(dtype=float),
            y_values=well.stations["Y_m"].to_numpy(dtype=float),
            z_values=well.stations["Z_m"].to_numpy(dtype=float),
        )
        extra_bounds = None
        if well.t1 is not None and well.t3 is not None:
            extra_bounds = _point_triplet_bounds(well.surface, well.t1, well.t3)
        merged_bounds = ptc_three_payload.merge_raw_bounds((bounds, extra_bounds))
        if merged_bounds is not None:
            well_bounds_by_name[str(well.name)] = merged_bounds
    legend_tree, focus_targets, hidden_labels = three_legend_tree_payload(
        session_state,
        records=records,
        visible_well_names=visible_names,
        well_bounds_by_name=well_bounds_by_name,
        name_to_color=name_to_color,
    )
    result: dict[str, object] = {
        "legend_tree": legend_tree,
        "focus_targets": focus_targets,
        "hidden_flat_legend_labels": hidden_labels,
        "collisions": _collision_payloads(analysis),
        "extra_meshes": pad_first_surface_arrow_payloads(
            session_state,
            records=records,
            visible_well_names=tuple(visible_names),
            surface_by_name=surface_by_name,
        ),
        "component_key": "anticollision-overview",
    }
    if successes:
        result["edit_wells"] = build_edit_wells_payload(successes, name_to_color)
    return result


def _point_triplet_bounds(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
) -> dict[str, list[float]]:
    return {
        "min": [
            float(min(surface.x, t1.x, t3.x)),
            float(min(surface.y, t1.y, t3.y)),
            float(min(surface.z, t1.z, t3.z)),
        ],
        "max": [
            float(max(surface.x, t1.x, t3.x)),
            float(max(surface.y, t1.y, t3.y)),
            float(max(surface.z, t1.z, t3.z)),
        ],
    }


def _point3d_payload(point: Point3D) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def _decimated_base_points(success: SuccessfulWellPlan) -> list[list[float]]:
    station_columns = ("X_m", "Y_m", "Z_m")
    if not all(column in success.stations.columns for column in station_columns):
        return []
    station_values = (
        success.stations.loc[:, list(station_columns)].dropna().to_numpy(dtype=float)
    )
    if not station_values.size:
        return []
    if len(station_values) > MAX_EDIT_BASE_POINTS:
        sample_indices = np.unique(
            np.linspace(
                0,
                len(station_values) - 1,
                num=MAX_EDIT_BASE_POINTS,
                dtype=int,
            )
        )
        station_values = station_values[sample_indices]
    return [
        [float(row[0]), float(row[1]), float(row[2])]
        for row in station_values
        if np.all(np.isfinite(row))
    ]


def _collision_payloads(
    analysis: AntiCollisionAnalysis,
) -> list[dict[str, object]]:
    events = anti_collision_report_events(analysis)
    zone_lookup: dict[tuple[str, str], list[AntiCollisionZone]] = {}
    for zone in analysis.zones:
        key = (str(zone.well_a), str(zone.well_b))
        zone_lookup.setdefault(key, []).append(zone)

    collisions: list[dict[str, object]] = []
    for index, event in enumerate(events):
        segment_a = _segment_types_for_interval(
            analysis,
            str(event.well_a),
            float(event.md_a_start_m),
            float(event.md_a_end_m),
        )
        segment_b = _segment_types_for_interval(
            analysis,
            str(event.well_b),
            float(event.md_b_start_m),
            float(event.md_b_end_m),
        )
        best_zone = _best_event_zone(
            zones=zone_lookup.get((str(event.well_a), str(event.well_b)), []),
            md_a_start_m=float(event.md_a_start_m),
            md_a_end_m=float(event.md_a_end_m),
            md_b_start_m=float(event.md_b_start_m),
            md_b_end_m=float(event.md_b_end_m),
        )
        hotspot = (
            list(best_zone.hotspot_xyz) if best_zone is not None else [0.0, 0.0, 0.0]
        )
        collisions.append(
            {
                "id": f"collision::{event.well_a}::{event.well_b}::{index}",
                "well_a": str(event.well_a),
                "well_b": str(event.well_b),
                "label": f"{event.well_a} ↔ {event.well_b}",
                "classification": str(event.classification),
                "priority_rank": int(event.priority_rank),
                "hotspot": hotspot,
                "separation_factor": float(event.min_separation_factor),
                "center_distance_m": float(event.min_center_distance_m),
                "segment_a": segment_a,
                "segment_b": segment_b,
            }
        )
    return collisions


def overlap_volume_payloads(
    analysis: AntiCollisionAnalysis,
) -> list[dict[str, object]]:
    volumes: list[dict[str, object]] = []
    for index, corridor in enumerate(analysis.corridors):
        rings = _aligned_overlap_rings(corridor.overlap_rings_xyz)
        if len(rings) < 2:
            continue
        volumes.append(
            {
                "name": "Зоны пересечений",
                "role": "overlap_volume",
                "well_a": str(corridor.well_a),
                "well_b": str(corridor.well_b),
                "id": f"overlap-volume::{corridor.well_a}::{corridor.well_b}::{index}",
                "color": "#C62828",
                "opacity": 0.34,
                "rings": [
                    [
                        [float(x_value), float(y_value), float(z_value)]
                        for x_value, y_value, z_value in ring.tolist()
                    ]
                    for ring in rings
                ],
            }
        )
    return volumes


def _aligned_overlap_rings(
    raw_rings: Iterable[np.ndarray],
) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for raw_ring in raw_rings:
        ring = _open_finite_ring(raw_ring)
        if ring is None:
            continue
        if previous is not None and len(previous) == len(ring):
            ring = _align_ring_to_previous(previous, ring)
        rings.append(ring)
        previous = ring
    return rings


def _open_finite_ring(raw_ring: object) -> np.ndarray | None:
    try:
        ring = np.asarray(raw_ring, dtype=float)
    except (TypeError, ValueError):
        return None
    if ring.ndim != 2 or ring.shape[1] != 3 or len(ring) < 3:
        return None
    finite_mask = np.all(np.isfinite(ring), axis=1)
    ring = ring[finite_mask]
    if len(ring) < 3:
        return None
    if np.linalg.norm(ring[0] - ring[-1]) <= 1e-6:
        ring = ring[:-1]
    if len(ring) < 3:
        return None
    return np.asarray(ring, dtype=float)


def _align_ring_to_previous(previous: np.ndarray, ring: np.ndarray) -> np.ndarray:
    candidates = [ring, ring[::-1]]
    best_ring = ring
    best_score = float("inf")
    for candidate in candidates:
        for shift in range(len(candidate)):
            shifted = np.roll(candidate, -shift, axis=0)
            score = float(np.linalg.norm(previous - shifted, axis=1).sum())
            if score < best_score:
                best_score = score
                best_ring = shifted
    return np.asarray(best_ring, dtype=float)


def _best_event_zone(
    *,
    zones: list[AntiCollisionZone],
    md_a_start_m: float,
    md_a_end_m: float,
    md_b_start_m: float,
    md_b_end_m: float,
) -> AntiCollisionZone | None:
    best_zone: AntiCollisionZone | None = None
    for zone in zones:
        if (
            float(zone.md_a_m) >= md_a_start_m - 1.0
            and float(zone.md_a_m) <= md_a_end_m + 1.0
            and float(zone.md_b_m) >= md_b_start_m - 1.0
            and float(zone.md_b_m) <= md_b_end_m + 1.0
        ):
            if best_zone is None or float(zone.separation_factor) < float(
                best_zone.separation_factor
            ):
                best_zone = zone
    if best_zone is not None:
        return best_zone
    if zones:
        return min(zones, key=lambda item: float(item.separation_factor))
    return None
