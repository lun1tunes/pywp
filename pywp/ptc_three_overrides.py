from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping

import numpy as np

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionCorridor,
    AntiCollisionZone,
    _segment_types_for_interval,
    anti_collision_report_event_groups,
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


def _record_surface_by_name(records: Iterable[WelltrackRecord]) -> dict[str, Point3D]:
    surface_by_name: dict[str, Point3D] = {}
    for record in records:
        points = tuple(record.points)
        if not points:
            continue
        surface = points[0]
        surface_by_name[str(record.name)] = Point3D(
            x=float(surface.x),
            y=float(surface.y),
            z=float(surface.z),
        )
    return surface_by_name


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
        end_name = (
            ordered_visible_names[6]
            if len(ordered_visible_names) >= 7
            else ordered_visible_names[-1]
        )
        surface = surface_by_name[first_name]
        arrows.append(
            _pad_first_surface_arrow_payload(
                surface=surface,
                end_surface=surface_by_name.get(end_name),
                nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
                spacing_m=float(cfg["spacing_m"]),
                well_name=first_name,
                end_well_name=end_name,
                pad_id=str(pad.pad_id),
            )
        )
    return arrows


def _pad_first_surface_arrow_payload(
    *,
    surface: Point3D,
    end_surface: Point3D | None,
    nds_azimuth_deg: float,
    spacing_m: float,
    well_name: str,
    end_well_name: str,
    pad_id: str,
) -> dict[str, object]:
    spacing = max(float(spacing_m), 1.0)
    vertical_lift_m = float(np.clip(spacing * 1.2, 24.0, 110.0))
    thickness_m = float(np.clip(spacing * 0.22, 5.0, 20.0))

    angle_rad = np.deg2rad(float(nds_azimuth_deg) % 360.0)
    direction = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=float)
    if float(np.linalg.norm(direction)) <= 1e-9:
        direction = np.array([0.0, 1.0], dtype=float)
    start_xy = np.array([float(surface.x), float(surface.y)], dtype=float)
    if end_surface is not None:
        end_xy = np.array([float(end_surface.x), float(end_surface.y)], dtype=float)
        measured_direction = end_xy - start_xy
        measured_length = float(np.linalg.norm(measured_direction))
        if measured_length > max(spacing * 0.35, 1.0):
            direction = measured_direction / measured_length
        else:
            end_xy = start_xy + direction * float(np.clip(spacing * 6.0, 72.0, 320.0))
    else:
        end_xy = start_xy + direction * float(np.clip(spacing * 6.0, 72.0, 320.0))
    length_m = max(float(np.linalg.norm(end_xy - start_xy)), 1.0)
    normal = np.array([-direction[1], direction[0]], dtype=float)

    tail = start_xy
    tip = end_xy
    shaft_width_m = max(min(length_m * 0.10, spacing * 0.55), 6.0)
    head_length_m = float(np.clip(length_m * 0.22, 18.0, max(length_m * 0.38, 18.0)))
    head_width_m = max(shaft_width_m * 2.6, min(length_m * 0.22, spacing * 1.35))
    head_base = tip - direction * head_length_m
    top_z = float(surface.z) - vertical_lift_m - thickness_m * 0.5
    bottom_z = float(surface.z) - vertical_lift_m + thickness_m * 0.5

    shaft_half = normal * (shaft_width_m * 0.5)
    head_half = normal * (head_width_m * 0.5)
    vertices_xy = [
        tail - shaft_half,
        tail + shaft_half,
        head_base + shaft_half,
        head_base + head_half,
        tip,
        head_base - head_half,
        head_base - shaft_half,
    ]
    vertices_top = [
        [float(point[0]), float(point[1]), top_z] for point in vertices_xy
    ]
    vertices_bottom = [
        [float(point[0]), float(point[1]), bottom_z] for point in vertices_xy
    ]
    faces = [
        [0, 1, 2],
        [0, 2, 6],
        [3, 4, 5],
        [9, 8, 7],
        [13, 9, 7],
        [12, 11, 10],
    ]
    outline = (0, 1, 2, 3, 4, 5, 6)
    for left, right in zip(outline, (*outline[1:], outline[0]), strict=False):
        faces.append([left, right, right + 7])
        faces.append([left, right + 7, left + 7])
    return {
        "name": "Начало порядка скважин",
        "vertices": [*vertices_top, *vertices_bottom],
        "faces": faces,
        "color": "#475569",
        "opacity": 0.94,
        "role": "pad_first_surface_arrow",
        "well_name": str(well_name),
        "end_well_name": str(end_well_name),
        "pad_id": str(pad_id),
        "nds_azimuth_deg": float(nds_azimuth_deg) % 360.0,
        "start_position": [float(surface.x), float(surface.y), float(surface.z)],
        "end_position": [
            float(end_xy[0]),
            float(end_xy[1]),
            float(surface.z if end_surface is None else end_surface.z),
        ],
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
    surface_by_name.update(_record_surface_by_name(records))
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
    surface_by_name.update(_record_surface_by_name(records))
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
    if not hasattr(analysis, "wells"):
        return _legacy_overlap_volume_payloads(getattr(analysis, "corridors", ()))

    volumes: list[dict[str, object]] = []
    for index, group in enumerate(anti_collision_report_event_groups(analysis)):
        event = group.event
        rings = _overlap_volume_rings_for_corridors(group.corridors)
        if len(rings) < 2:
            continue
        volumes.append(
            {
                "name": "Зоны пересечений",
                "role": "overlap_volume",
                "well_a": str(event.well_a),
                "well_b": str(event.well_b),
                "id": f"overlap-volume::{event.well_a}::{event.well_b}::{index}",
                "classification": str(event.classification),
                "priority_rank": int(event.priority_rank),
                "md_a_start_m": float(event.md_a_start_m),
                "md_a_end_m": float(event.md_a_end_m),
                "md_b_start_m": float(event.md_b_start_m),
                "md_b_end_m": float(event.md_b_end_m),
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


def _legacy_overlap_volume_payloads(corridors: Iterable[object]) -> list[dict[str, object]]:
    volumes: list[dict[str, object]] = []
    for index, corridor in enumerate(corridors):
        rings = _overlap_volume_rings(corridor)
        if len(rings) < 2:
            continue
        well_a = str(getattr(corridor, "well_a", ""))
        well_b = str(getattr(corridor, "well_b", ""))
        volumes.append(
            {
                "name": "Зоны пересечений",
                "role": "overlap_volume",
                "well_a": well_a,
                "well_b": well_b,
                "id": f"overlap-volume::{well_a}::{well_b}::{index}",
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


def _overlap_volume_rings_for_corridors(
    corridors: Iterable[AntiCollisionCorridor],
) -> list[np.ndarray]:
    corridor_list = sorted(
        tuple(corridors),
        key=lambda corridor: (
            float(corridor.md_a_start_m),
            float(corridor.md_b_start_m),
            int(corridor.priority_rank),
        ),
    )
    raw_rings = [
        ring
        for corridor in corridor_list
        for ring in getattr(corridor, "overlap_rings_xyz", ())
    ]
    rings = _aligned_overlap_rings(raw_rings)
    if len(rings) >= 2:
        return rings
    if len(rings) == 1:
        return _thin_volume_from_single_ring(rings[0])

    fallback_rings = [
        ring
        for corridor in corridor_list
        for ring in _fallback_overlap_rings_from_corridor(corridor)
    ]
    rings = _aligned_overlap_rings(fallback_rings)
    if len(rings) >= 2:
        return rings
    if len(rings) == 1:
        return _thin_volume_from_single_ring(rings[0])
    return []


def _overlap_volume_rings(corridor: object) -> list[np.ndarray]:
    rings = _aligned_overlap_rings(getattr(corridor, "overlap_rings_xyz", ()))
    if len(rings) >= 2:
        return rings
    if len(rings) == 1:
        return _thin_volume_from_single_ring(rings[0])
    return _fallback_overlap_rings_from_corridor(corridor)


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


def _thin_volume_from_single_ring(ring: np.ndarray) -> list[np.ndarray]:
    ring = np.asarray(ring, dtype=float)
    center = np.mean(ring, axis=0)
    distances = np.linalg.norm(ring - center[None, :], axis=1)
    radius = float(np.nanmax(distances)) if distances.size else 1.0
    normal = _ring_normal_vector(ring)
    thickness = _overlap_fallback_thickness(radius)
    return [
        np.asarray(ring - normal[None, :] * thickness * 0.5, dtype=float),
        np.asarray(ring + normal[None, :] * thickness * 0.5, dtype=float),
    ]


def _fallback_overlap_rings_from_corridor(corridor: object) -> list[np.ndarray]:
    centers = _finite_xyz_rows(getattr(corridor, "midpoint_xyz", ()))
    if centers.size == 0:
        return []
    radii = _positive_values(getattr(corridor, "overlap_core_radius_m", ()))
    if radii.size == 0:
        radii = _positive_values(getattr(corridor, "overlap_depth_values_m", ()))
    if radii.size == 0:
        radii = np.full(len(centers), 2.0, dtype=float)
    if len(radii) < len(centers):
        radii = np.pad(radii, (0, len(centers) - len(radii)), mode="edge")
    radii = radii[: len(centers)]
    rings: list[np.ndarray] = []
    if len(centers) == 1:
        radius = float(max(radii[0], 1.0))
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        thickness = _overlap_fallback_thickness(radius)
        for offset in (-0.5, 0.5):
            rings.append(
                _circle_ring(
                    center=centers[0] + normal * thickness * offset,
                    normal=normal,
                    radius=radius,
                )
            )
        return rings
    for index, center in enumerate(centers):
        tangent = _centerline_tangent(centers, index)
        rings.append(
            _circle_ring(
                center=center,
                normal=tangent,
                radius=float(max(radii[index], 1.0)),
            )
        )
    return _aligned_overlap_rings(rings)


def _finite_xyz_rows(values: object) -> np.ndarray:
    try:
        rows = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 3), dtype=float)
    if rows.ndim == 1 and rows.shape[0] == 3:
        rows = rows.reshape(1, 3)
    if rows.ndim != 2 or rows.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    return np.asarray(rows[np.all(np.isfinite(rows), axis=1)], dtype=float)


def _positive_values(values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    return np.asarray(array, dtype=float)


def _overlap_fallback_thickness(radius: float) -> float:
    return float(np.clip(max(float(radius), 1.0) * 0.12, 0.6, 8.0))


def _ring_normal_vector(ring: np.ndarray) -> np.ndarray:
    center = np.mean(np.asarray(ring, dtype=float), axis=0)
    centered = np.asarray(ring, dtype=float) - center[None, :]
    if len(centered) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=float)
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / norm


def _centerline_tangent(centers: np.ndarray, index: int) -> np.ndarray:
    if len(centers) <= 1:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if index <= 0:
        tangent = centers[1] - centers[0]
    elif index >= len(centers) - 1:
        tangent = centers[-1] - centers[-2]
    else:
        tangent = centers[index + 1] - centers[index - 1]
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(norm) or norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return np.asarray(tangent / norm, dtype=float)


def _circle_ring(
    *,
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    point_count: int = 24,
) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 1e-9:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / normal_norm
    seed = (
        np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(normal[2])) < 0.92
        else np.array([0.0, 1.0, 0.0], dtype=float)
    )
    basis_u = np.cross(normal, seed)
    basis_u_norm = float(np.linalg.norm(basis_u))
    if not np.isfinite(basis_u_norm) or basis_u_norm <= 1e-9:
        basis_u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        basis_u = basis_u / basis_u_norm
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / max(float(np.linalg.norm(basis_v)), 1e-9)
    angles = np.linspace(0.0, 2.0 * np.pi, int(max(point_count, 8)), endpoint=False)
    safe_radius = float(max(radius, 1.0))
    return np.asarray(
        [
            center + safe_radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
            for angle in angles
        ],
        dtype=float,
    )


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
