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
    "pad_first_surface_label_payloads",
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
) -> tuple[
    list[dict[str, object]], dict[str, dict[str, list[float]]], set[str]
]:
    visible_set = {
        str(name) for name in visible_well_names if str(name).strip()
    }
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


def pad_first_surface_label_payloads(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    visible_well_names: Iterable[str],
    surface_by_name: Mapping[str, Point3D],
) -> list[dict[str, object]]:
    visible_set = {
        str(name) for name in visible_well_names if str(name).strip()
    }
    if not visible_set:
        return []
    pads, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )
    labels: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for pad in pads:
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
        labels.append(
            {
                "text": "1",
                "position": [
                    float(surface.x),
                    float(surface.y),
                    float(surface.z),
                ],
                "color": "#0f172a",
                "role": "pad_first_surface_label",
                "well_name": first_name,
                "pad_id": str(pad.pad_id),
            }
        )
    return labels


def augment_three_payload(
    *,
    payload: dict[str, object],
    legend_tree: list[dict[str, object]] | None = None,
    focus_targets: Mapping[str, dict[str, list[float]]] | None = None,
    hidden_flat_legend_labels: set[str] | None = None,
    collisions: list[dict[str, object]] | None = None,
    edit_wells: list[dict[str, object]] | None = None,
    extra_labels: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    updated = dict(payload)
    if legend_tree:
        updated["legend_tree"] = list(legend_tree)
    if focus_targets:
        updated["focus_targets"] = {
            str(key): dict(value) for key, value in focus_targets.items()
        }
    hidden_labels = {
        str(label) for label in (hidden_flat_legend_labels or set())
    }
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
                "color": str(
                    name_to_color.get(str(success.name), "#2563eb")
                ),
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
        "extra_labels": pad_first_surface_label_payloads(
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
        "extra_labels": pad_first_surface_label_payloads(
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
        success.stations.loc[:, list(station_columns)]
        .dropna()
        .to_numpy(dtype=float)
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
