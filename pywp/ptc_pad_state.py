from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from pywp.constants import SMALL
from pywp import ptc_target_import
from pywp.eclipse_welltrack import WelltrackRecord
from pywp.models import Point3D
from pywp.pilot_wells import well_name_key
from pywp.welltrack_targets import ordinary_record_target_layout
from pywp.well_pad import (
    DEFAULT_PAD_WELL_AUTO_ORDER_MODE,
    PAD_WELL_AUTO_ORDER_NAME,
    PAD_WELL_AUTO_ORDER_PROJECTION,
    PAD_WELL_AUTO_ORDER_TARGET_DEPTH_DESC,
    PAD_SURFACE_ANCHOR_CENTER,
    PAD_SURFACE_ANCHOR_FIRST,
    PadLayoutPlan,
    PadWell,
    WellPad,
    aligned_pad_nds_azimuth_deg,
    estimate_pad_nds_azimuth_deg,
    ordered_pad_wells,
    well_name_natural_sort_key,
)

__all__ = [
    "DEFAULT_PAD_SPACING_M",
    "DEFAULT_PAD_SURFACE_ANCHOR_MODE",
    "PAD_SURFACE_ANCHOR_CENTER",
    "PAD_SURFACE_ANCHOR_FIRST",
    "WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M",
    "WT_PAD_FOCUS_ALL",
    "WT_PAD_LAYOUT_DETAILS_OPEN_KEY",
    "DetectedPadUiMeta",
    "PadSurfaceAssignment",
    "WT_PAD_APPLY_AUTO_ORDER_KEY",
    "build_pad_plan_map",
    "detect_ui_pads",
    "ensure_pad_configs",
    "estimate_surface_pad_axis_deg",
    "focus_pad_fixed_well_names",
    "focus_pad_well_names",
    "inferred_surface_spacing_m",
    "normalize_focus_pad_id",
    "pad_anchor_defaults",
    "pad_auto_order_mode",
    "pad_auto_order_mode_label",
    "pad_anchor_mode_label",
    "pad_config_defaults",
    "pad_config_for_ui",
    "pad_display_label",
    "pad_layout_details_open",
    "pad_fixed_slots_editor_rows",
    "pad_fixed_slots_from_config",
    "pad_fixed_slots_from_editor",
    "pad_membership",
    "pad_surface_assignments",
    "project_pads_for_ui",
    "record_midpoint_xyz",
    "resolved_pad_nds_azimuth_deg",
    "source_surface_xyz",
    "toggle_pad_layout_details_open",
]

DEFAULT_PAD_SPACING_M = 40.0
DEFAULT_PAD_SURFACE_ANCHOR_MODE = PAD_SURFACE_ANCHOR_CENTER
WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M = 400.0
WT_PAD_FOCUS_ALL = "__all_pads__"
WT_PAD_AUTO_ORDER_BY_TARGET_DEPTH_KEY = "wt_pad_auto_order_by_target_depth"
WT_PAD_LAYOUT_DETAILS_OPEN_KEY = "wt_pad_layout_details_open"
WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY = "allow_source_surface_edit"
WT_PAD_APPLY_AUTO_ORDER_KEY = "apply_auto_order"


@dataclass(frozen=True)
class DetectedPadUiMeta:
    source_surfaces_defined: bool
    inferred_spacing_m: float
    source_surface_x_m: float
    source_surface_y_m: float
    source_surface_z_m: float
    source_surface_count: int
    source_surface_slots: tuple[tuple[str, float, float, float], ...] = ()
    auto_name_notice: str = ""


@dataclass(frozen=True)
class PadSurfaceAssignment:
    slot_index: int
    well_name: str
    surface_x_m: float
    surface_y_m: float
    surface_z_m: float
    source_well_name: str = ""


@dataclass(frozen=True)
class _PadDetectionRecord:
    record_index: int
    record: WelltrackRecord
    cluster_surface_xyz: tuple[float, float, float]
    source_surface_xyz: tuple[float, float, float]
    source_surface_defined: bool
    inferred_pad_group_key: str = ""


_PAD_COMPONENT_EDGE_RE = re.compile(r"^[\s_\-]+|[\s_\-]+$")
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(\d+)$")
_TRAILING_GROUP_TOKEN_RE = re.compile(r"^(.*?)[_\-]([A-Za-z0-9]+)$")


def pad_layout_details_open(session_state: Mapping[str, object]) -> bool:
    raw_value = session_state.get(WT_PAD_LAYOUT_DETAILS_OPEN_KEY)
    return False if raw_value is None else bool(raw_value)


def toggle_pad_layout_details_open(session_state: MutableMapping[str, object]) -> bool:
    next_open = not pad_layout_details_open(session_state)
    session_state[WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = next_open
    return next_open


def pad_config_defaults(pad: WellPad) -> dict[str, object]:
    return {
        "spacing_m": float(DEFAULT_PAD_SPACING_M),
        "nds_azimuth_deg": float(
            estimate_pad_nds_azimuth_deg(
                wells=pad.wells,
                surface_x=float(pad.surface.x),
                surface_y=float(pad.surface.y),
                surface_anchor_mode=DEFAULT_PAD_SURFACE_ANCHOR_MODE,
            )
        ),
        "first_surface_x": float(pad.surface.x),
        "first_surface_y": float(pad.surface.y),
        "first_surface_z": float(pad.surface.z),
        "surface_anchor_mode": DEFAULT_PAD_SURFACE_ANCHOR_MODE,
        WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY: False,
        WT_PAD_APPLY_AUTO_ORDER_KEY: False,
        "fixed_slots": (),
    }


def pad_fixed_slots_from_config(
    *,
    pad: WellPad,
    config: Mapping[str, object],
) -> tuple[tuple[int, str], ...]:
    raw_value = config.get("fixed_slots", ())
    if not isinstance(raw_value, (list, tuple)):
        return ()
    well_names = {str(well.name) for well in pad.wells}
    max_slot = len(pad.wells)
    used_slots: set[int] = set()
    used_names: set[str] = set()
    normalized: list[tuple[int, str]] = []
    for item in raw_value:
        if isinstance(item, Mapping):
            raw_slot = item.get("slot", item.get("position"))
            raw_name = item.get("name", item.get("well_name"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_slot = item[0]
            raw_name = item[1]
        else:
            continue
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            continue
        name = str(raw_name or "").strip()
        if (
            slot < 1
            or slot > max_slot
            or name not in well_names
            or slot in used_slots
            or name in used_names
        ):
            continue
        used_slots.add(slot)
        used_names.add(name)
        normalized.append((slot, name))
    return tuple(sorted(normalized, key=lambda value: value[0]))


def _apply_fixed_slots_to_ordered_names(
    ordered_names: list[str],
    *,
    fixed_slots: tuple[tuple[int, str], ...],
) -> list[str]:
    if not ordered_names or not fixed_slots:
        return list(ordered_names)
    normalized_names = [str(name) for name in ordered_names]
    max_slot = len(normalized_names)
    fixed_map = {
        int(slot): str(name)
        for slot, name in fixed_slots
        if 1 <= int(slot) <= max_slot and str(name) in normalized_names
    }
    if not fixed_map:
        return normalized_names

    placed_names = set(fixed_map.values())
    remaining_names = [
        str(name) for name in normalized_names if str(name) not in placed_names
    ]
    remaining_iter = iter(remaining_names)
    resolved_names: list[str] = []
    for slot_index in range(1, max_slot + 1):
        fixed_name = fixed_map.get(int(slot_index))
        if fixed_name is not None:
            resolved_names.append(str(fixed_name))
            continue
        resolved_names.append(next(remaining_iter))
    return resolved_names


def pad_fixed_slots_editor_rows(
    *,
    pad: WellPad,
    config: Mapping[str, object],
) -> pd.DataFrame:
    slots = pad_fixed_slots_from_config(pad=pad, config=config)
    return pd.DataFrame(
        [
            {"Позиция": int(slot), "Скважина": str(name)}
            for slot, name in slots
        ],
        columns=["Позиция", "Скважина"],
    )


def pad_fixed_slots_from_editor(
    *,
    pad: WellPad,
    editor_value: object,
) -> tuple[tuple[tuple[int, str], ...], list[str]]:
    if isinstance(editor_value, pd.DataFrame):
        rows = editor_value.to_dict("records")
    elif isinstance(editor_value, list):
        rows = [item for item in editor_value if isinstance(item, Mapping)]
    else:
        return (), []

    well_names = {str(well.name) for well in pad.wells}
    max_slot = len(pad.wells)
    used_slots: set[int] = set()
    used_names: set[str] = set()
    fixed_slots: list[tuple[int, str]] = []
    warnings: list[str] = []
    for row in rows:
        raw_slot = row.get("Позиция")
        raw_name = row.get("Скважина")
        slot_blank = (
            raw_slot is None
            or (isinstance(raw_slot, float) and pd.isna(raw_slot))
            or str(raw_slot).strip() == ""
        )
        name_blank = (
            raw_name is None
            or (isinstance(raw_name, float) and pd.isna(raw_name))
            or str(raw_name).strip() == ""
        )
        if slot_blank and name_blank:
            continue
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            warnings.append("Строки без корректной позиции пропущены.")
            continue
        name = str(raw_name or "").strip()
        if not name or name.lower() == "nan":
            warnings.append(f"Позиция {slot}: выберите скважину.")
            continue
        if slot < 1 or slot > max_slot:
            warnings.append(
                f"Позиция {slot}: допустимый диапазон 1–{max_slot}."
            )
            continue
        if name not in well_names:
            warnings.append(f"{name}: скважина не входит в выбранный куст.")
            continue
        if slot in used_slots:
            warnings.append(f"Позиция {slot}: дубль, оставлена первая строка.")
            continue
        if name in used_names:
            warnings.append(f"{name}: дубль, оставлена первая строка.")
            continue
        used_slots.add(slot)
        used_names.add(name)
        fixed_slots.append((slot, name))
    return tuple(sorted(fixed_slots, key=lambda value: value[0])), warnings


def source_surface_xyz(
    record: WelltrackRecord,
) -> tuple[float, float, float] | None:
    points = tuple(record.points)
    if not points:
        return None
    surface = points[0]
    return float(surface.x), float(surface.y), float(surface.z)


def _degenerate_three_point_surface_cluster_xyz(
    record: WelltrackRecord,
    *,
    allow_surface_offset: bool = False,
) -> tuple[float, float, float] | None:
    points = tuple(record.points)
    if len(points) != 3:
        return None
    surface_xyz = source_surface_xyz(record)
    if surface_xyz is None:
        return None
    try:
        layout = ordinary_record_target_layout(record)
    except (TypeError, ValueError):
        return None
    if not allow_surface_offset and (
        abs(float(surface_xyz[0]) - float(layout.t1.x)) > SMALL
        or abs(float(surface_xyz[1]) - float(layout.t1.y)) > SMALL
    ):
        return None
    if (
        abs(float(layout.final_target.x) - float(layout.t1.x)) <= SMALL
        and abs(float(layout.final_target.y) - float(layout.t1.y)) <= SMALL
    ):
        return None
    return float(layout.t1.x), float(layout.t1.y), float(surface_xyz[2])


def _inferred_pad_group_key(name: str) -> str:
    trimmed = _trim_pad_component(str(name))
    if not trimmed:
        return ""
    reduced = _reduced_well_name_for_pad(trimmed)
    if reduced != trimmed:
        return str(reduced).casefold()
    match = _TRAILING_GROUP_TOKEN_RE.fullmatch(trimmed)
    if match is None:
        return ""
    prefix, suffix = match.groups()
    normalized_prefix = _trim_pad_component(prefix)
    if not normalized_prefix:
        return ""
    suffix_text = str(suffix)
    if (
        suffix_text.isdigit()
        or len(suffix_text) == 1
        or normalized_prefix[-1].isdigit()
    ):
        return str(normalized_prefix).casefold()
    return ""


def _pad_detection_records(
    records: list[WelltrackRecord],
    *,
    remembered_degenerate_well_names: tuple[str, ...] = (),
) -> list[_PadDetectionRecord]:
    remembered_name_keys = {
        well_name_key(str(name))
        for name in remembered_degenerate_well_names
        if str(name).strip()
    }
    prepared: list[_PadDetectionRecord] = []
    for record_index, record in enumerate(records):
        surface_xyz = source_surface_xyz(record)
        if surface_xyz is None:
            continue
        degenerate_cluster_xyz = _degenerate_three_point_surface_cluster_xyz(record)
        if (
            degenerate_cluster_xyz is None
            and well_name_key(str(record.name)) in remembered_name_keys
        ):
            degenerate_cluster_xyz = _degenerate_three_point_surface_cluster_xyz(
                record,
                allow_surface_offset=True,
            )
        if degenerate_cluster_xyz is not None:
            prepared.append(
                _PadDetectionRecord(
                    record_index=int(record_index),
                    record=record,
                    cluster_surface_xyz=degenerate_cluster_xyz,
                    source_surface_xyz=surface_xyz,
                    source_surface_defined=False,
                    inferred_pad_group_key=_inferred_pad_group_key(str(record.name)),
                )
            )
            continue
        prepared.append(
            _PadDetectionRecord(
                record_index=int(record_index),
                record=record,
                cluster_surface_xyz=surface_xyz,
                source_surface_xyz=surface_xyz,
                source_surface_defined=True,
                inferred_pad_group_key=_inferred_pad_group_key(str(record.name)),
            )
        )
    return prepared


def record_midpoint_xyz(record: WelltrackRecord) -> tuple[float, float, float]:
    points = tuple(record.points)
    if len(points) >= 3:
        try:
            layout = ordinary_record_target_layout(record)
            target_points = (
                layout.target_pairs[0]
                if layout.target_pairs
                else (layout.t1, layout.final_target)
            )
            return (
                float(np.mean([point.x for point in target_points])),
                float(np.mean([point.y for point in target_points])),
                float(np.mean([point.z for point in target_points])),
            )
        except (TypeError, ValueError):
            pass
    surface_xyz = source_surface_xyz(record)
    return surface_xyz or (0.0, 0.0, 0.0)


def record_target_axis_xy(record: WelltrackRecord) -> tuple[float, float]:
    points = tuple(record.points)
    if len(points) >= 3:
        try:
            layout = ordinary_record_target_layout(record)
            dx = float(layout.final_target.x) - float(layout.t1.x)
            dy = float(layout.final_target.y) - float(layout.t1.y)
            norm = float(np.hypot(dx, dy))
            if norm > SMALL:
                return float(dx / norm), float(dy / norm)
        except (TypeError, ValueError):
            pass
    return 0.0, 0.0


def estimate_surface_pad_axis_deg(
    surface_xyzs: list[tuple[float, float, float]],
) -> float:
    if len(surface_xyzs) <= 1:
        return 0.0
    points = np.asarray(
        [(item[0], item[1]) for item in surface_xyzs], dtype=float
    )
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_index = int(np.argmax(np.asarray(eigenvalues, dtype=float)))
        direction = np.asarray(eigenvectors[:, principal_index], dtype=float)
    except np.linalg.LinAlgError:
        direction = np.asarray([1.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= SMALL:
        return 0.0
    unit = direction / norm
    return float(
        np.degrees(np.arctan2(float(unit[0]), float(unit[1]))) % 360.0
    )


def inferred_surface_spacing_m(
    *,
    surface_xyzs: list[tuple[float, float, float]],
    nds_azimuth_deg: float,
) -> float:
    if len(surface_xyzs) <= 1:
        return 0.0
    angle_rad = np.deg2rad(float(nds_azimuth_deg) % 360.0)
    ux = float(np.sin(angle_rad))
    uy = float(np.cos(angle_rad))
    projections = sorted(
        float(x) * ux + float(y) * uy for x, y, _ in surface_xyzs
    )
    diffs = [
        float(right - left)
        for left, right in zip(projections, projections[1:], strict=False)
        if float(right - left) > 1e-6
    ]
    if not diffs:
        return 0.0
    return float(np.median(np.asarray(diffs, dtype=float)))


def detect_ui_pads(
    records: list[WelltrackRecord],
    *,
    remembered_degenerate_well_names: tuple[str, ...] = (),
) -> tuple[list[WellPad], dict[str, DetectedPadUiMeta]]:
    indexed_records = _pad_detection_records(
        records,
        remembered_degenerate_well_names=remembered_degenerate_well_names,
    )
    if not indexed_records:
        return [], {}

    adjacency: dict[int, set[int]] = {
        entry.record_index: set() for entry in indexed_records
    }
    for left_pos, left_entry in enumerate(indexed_records):
        for right_entry in indexed_records[left_pos + 1 :]:
            if (
                (
                    not left_entry.source_surface_defined
                    or not right_entry.source_surface_defined
                )
                and left_entry.inferred_pad_group_key
                and left_entry.inferred_pad_group_key
                == right_entry.inferred_pad_group_key
            ):
                adjacency[left_entry.record_index].add(right_entry.record_index)
                adjacency[right_entry.record_index].add(left_entry.record_index)
                continue
            distance_xy = float(
                np.hypot(
                    float(left_entry.cluster_surface_xyz[0])
                    - float(right_entry.cluster_surface_xyz[0]),
                    float(left_entry.cluster_surface_xyz[1])
                    - float(right_entry.cluster_surface_xyz[1]),
                )
            )
            if distance_xy <= WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M + SMALL:
                adjacency[left_entry.record_index].add(right_entry.record_index)
                adjacency[right_entry.record_index].add(left_entry.record_index)

    by_index = {
        entry.record_index: entry for entry in indexed_records
    }
    clusters: list[list[_PadDetectionRecord]] = []
    visited: set[int] = set()
    for entry in indexed_records:
        index = int(entry.record_index)
        if index in visited:
            continue
        queue = [index]
        visited.add(index)
        cluster: list[_PadDetectionRecord] = []
        while queue:
            current = queue.pop()
            cluster.append(by_index[current])
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        clusters.append(cluster)

    prepared: list[
        tuple[
            float,
            float,
            float,
            list[_PadDetectionRecord],
        ]
    ] = []
    for cluster in clusters:
        surface_xyzs = [entry.cluster_surface_xyz for entry in cluster]
        center_x = float(np.mean([item[0] for item in surface_xyzs]))
        center_y = float(np.mean([item[1] for item in surface_xyzs]))
        center_z = float(np.mean([item[2] for item in surface_xyzs]))
        prepared.append((center_x, center_y, center_z, cluster))
    prepared.sort(key=lambda item: (item[0], item[1], item[2]))
    resolved_pad_names = _resolved_auto_pad_names(prepared)

    pads: list[WellPad] = []
    metadata: dict[str, DetectedPadUiMeta] = {}
    for index, (center_x, center_y, center_z, cluster) in enumerate(
        prepared, start=1
    ):
        wells: list[PadWell] = []
        surface_xyzs: list[tuple[float, float, float]] = []
        source_defined_surface_xyzs: list[tuple[float, float, float]] = []
        surface_slot_rows: list[tuple[str, float, float, float]] = []
        unique_surface_keys: set[tuple[int, int, int]] = set()
        for entry in cluster:
            midpoint_x, midpoint_y, midpoint_z = record_midpoint_xyz(entry.record)
            target_axis_x, target_axis_y = record_target_axis_xy(entry.record)
            wells.append(
                PadWell(
                    name=str(entry.record.name),
                    record_index=int(entry.record_index),
                    midpoint_x=float(midpoint_x),
                    midpoint_y=float(midpoint_y),
                    midpoint_z=float(midpoint_z),
                    target_axis_x=float(target_axis_x),
                    target_axis_y=float(target_axis_y),
                )
            )
            surface_xyzs.append(entry.cluster_surface_xyz)
            if entry.source_surface_defined:
                source_surface = entry.source_surface_xyz
                source_defined_surface_xyzs.append(source_surface)
                surface_slot_rows.append(
                    (
                        str(entry.record.name),
                        float(source_surface[0]),
                        float(source_surface[1]),
                        float(source_surface[2]),
                    )
                )
                unique_surface_keys.add(
                    (
                        round(float(source_surface[0]), 6),
                        round(float(source_surface[1]), 6),
                        round(float(source_surface[2]), 6),
                    )
                )
        pad_id, auto_name_notice = resolved_pad_names[index - 1]
        source_surfaces_defined = (
            len(surface_slot_rows) == len(cluster) and len(unique_surface_keys) > 1
        )
        layout_surface_xyzs = (
            source_defined_surface_xyzs if source_defined_surface_xyzs else surface_xyzs
        )
        if layout_surface_xyzs:
            center_x = float(np.mean([item[0] for item in layout_surface_xyzs]))
            center_y = float(np.mean([item[1] for item in layout_surface_xyzs]))
            center_z = float(np.mean([item[2] for item in layout_surface_xyzs]))
        auto_nds = estimate_surface_pad_axis_deg(layout_surface_xyzs)
        if abs(float(auto_nds)) <= SMALL:
            auto_nds = estimate_pad_nds_azimuth_deg(
                wells=tuple(wells),
                surface_x=float(center_x),
                surface_y=float(center_y),
                surface_anchor_mode=DEFAULT_PAD_SURFACE_ANCHOR_MODE,
            )
        pads.append(
            WellPad(
                pad_id=pad_id,
                surface=Point3D(
                    x=float(center_x), y=float(center_y), z=float(center_z)
                ),
                wells=tuple(wells),
                auto_nds_azimuth_deg=float(auto_nds),
            )
        )
        metadata[pad_id] = DetectedPadUiMeta(
            source_surfaces_defined=bool(source_surfaces_defined),
            inferred_spacing_m=inferred_surface_spacing_m(
                surface_xyzs=layout_surface_xyzs,
                nds_azimuth_deg=float(auto_nds),
            ),
            source_surface_x_m=float(center_x),
            source_surface_y_m=float(center_y),
            source_surface_z_m=float(center_z),
            source_surface_count=len(surface_slot_rows),
            source_surface_slots=tuple(
                sorted(
                    surface_slot_rows,
                    key=lambda item: well_name_natural_sort_key(str(item[0])),
                )
            ),
            auto_name_notice=str(auto_name_notice),
        )
    return pads, metadata


def _source_defined_surface_map(
    pad_meta: object,
) -> dict[str, tuple[float, float, float]]:
    if not isinstance(pad_meta, DetectedPadUiMeta):
        return {}
    return {
        str(name): (float(x), float(y), float(z))
        for name, x, y, z in pad_meta.source_surface_slots
    }


def _source_defined_pad_basis(
    pad: WellPad,
    pad_meta: object,
) -> dict[str, object] | None:
    surfaces_by_name = _source_defined_surface_map(pad_meta)
    if not surfaces_by_name:
        return None
    natural_names = [
        str(well.name)
        for well in sorted(
            pad.wells,
            key=lambda well: well_name_natural_sort_key(str(well.name)),
        )
        if str(well.name) in surfaces_by_name
    ]
    if not natural_names:
        return None

    axis_deg = float(pad.auto_nds_azimuth_deg) % 360.0
    angle_rad = np.deg2rad(axis_deg)
    ux = float(np.sin(angle_rad))
    uy = float(np.cos(angle_rad))
    first_name = str(natural_names[0])
    last_name = str(natural_names[-1])
    first_xyz = surfaces_by_name.get(first_name, next(iter(surfaces_by_name.values())))
    last_xyz = surfaces_by_name.get(last_name, first_xyz)
    first_projection = float(first_xyz[0]) * ux + float(first_xyz[1]) * uy
    last_projection = float(last_xyz[0]) * ux + float(last_xyz[1]) * uy
    if last_projection + SMALL < first_projection:
        axis_deg = float((axis_deg + 180.0) % 360.0)
        angle_rad = np.deg2rad(axis_deg)
        ux = float(np.sin(angle_rad))
        uy = float(np.cos(angle_rad))
        first_projection = float(first_xyz[0]) * ux + float(first_xyz[1]) * uy
        last_projection = float(last_xyz[0]) * ux + float(last_xyz[1]) * uy

    vx = float(-uy)
    vy = float(ux)
    slot_rows: list[tuple[str, float, float, float, float, float]] = []
    for well_name, (x_value, y_value, z_value) in surfaces_by_name.items():
        projection = float(x_value) * ux + float(y_value) * uy
        cross_projection = float(x_value) * vx + float(y_value) * vy
        slot_rows.append(
            (
                str(well_name),
                float(x_value),
                float(y_value),
                float(z_value),
                float(projection),
                float(cross_projection),
            )
        )
    slot_rows.sort(
        key=lambda item: (
            float(item[4]),
            well_name_natural_sort_key(str(item[0])),
            str(item[0]),
        )
    )
    slot_index_by_name = {
        str(name): int(index)
        for index, name in enumerate(
            sorted(
                (str(well.name) for well in pad.wells),
                key=well_name_natural_sort_key,
            )
        )
    }
    known_rows_by_slot = sorted(
        (
            (
                int(slot_index_by_name[str(well_name)]),
                float(projection),
            )
            for well_name, _x_value, _y_value, _z_value, projection, _cross_projection in slot_rows
            if str(well_name) in slot_index_by_name
        ),
        key=lambda item: int(item[0]),
    )
    spacing_candidates = [
        float((right_projection - left_projection) / (right_index - left_index))
        for (left_index, left_projection), (right_index, right_projection) in zip(
            known_rows_by_slot,
            known_rows_by_slot[1:],
            strict=False,
        )
        if int(right_index - left_index) > 0
        and float(right_projection - left_projection) > 1e-6
    ]
    spacing_ref_m = (
        float(np.median(np.asarray(spacing_candidates, dtype=float)))
        if spacing_candidates
        else float(
            max(
                getattr(pad_meta, "inferred_spacing_m", 0.0),
                0.0,
            )
        )
    )
    center_xyz = (
        0.5 * (float(first_xyz[0]) + float(last_xyz[0])),
        0.5 * (float(first_xyz[1]) + float(last_xyz[1])),
        0.5 * (float(first_xyz[2]) + float(last_xyz[2])),
    )
    center_projection = float(center_xyz[0]) * ux + float(center_xyz[1]) * uy
    first_cross_projection = float(first_xyz[0]) * vx + float(first_xyz[1]) * vy
    center_cross_projection = float(center_xyz[0]) * vx + float(center_xyz[1]) * vy
    return {
        "axis_deg": float(axis_deg),
        "slots": tuple(slot_rows),
        "spacing_ref_m": float(max(spacing_ref_m, 0.0)),
        "first_anchor_xyz": (
            float(first_xyz[0]),
            float(first_xyz[1]),
            float(first_xyz[2]),
        ),
        "first_anchor_projection": float(first_projection),
        "first_anchor_cross_projection": float(first_cross_projection),
        "center_anchor_xyz": center_xyz,
        "center_anchor_projection": float(center_projection),
        "center_anchor_cross_projection": float(center_cross_projection),
    }


def _uses_source_surface_basis(
    pad: WellPad,
    pad_meta: object,
    basis: Mapping[str, object] | None,
) -> bool:
    if basis is None or not isinstance(pad_meta, DetectedPadUiMeta):
        return False
    if bool(pad_meta.source_surfaces_defined):
        return True
    source_surface_count = int(getattr(pad_meta, "source_surface_count", 0) or 0)
    return 0 < source_surface_count < int(len(pad.wells))


def _preserves_source_surface_slots(pad_meta: object) -> bool:
    return isinstance(pad_meta, DetectedPadUiMeta) and bool(
        pad_meta.source_surfaces_defined
    )


def _partial_source_surface_layout(
    *,
    pad: WellPad,
    basis: Mapping[str, object],
) -> dict[str, object] | None:
    raw_slot_rows = tuple(basis.get("slots", ()))
    if not raw_slot_rows:
        return None

    ordered_slot_names = tuple(
        str(well.name)
        for well in sorted(
            pad.wells,
            key=lambda well: well_name_natural_sort_key(str(well.name)),
        )
    )
    if not ordered_slot_names:
        return None

    slot_row_by_name = {
        str(slot_name): (
            float(x_value),
            float(y_value),
            float(z_value),
            float(projection),
            float(cross_projection),
        )
        for slot_name, x_value, y_value, z_value, projection, cross_projection in raw_slot_rows
    }
    known_rows: list[tuple[int, float, float, float]] = []
    for slot_index, well_name in enumerate(ordered_slot_names):
        row = slot_row_by_name.get(str(well_name))
        if row is None:
            continue
        known_rows.append(
            (
                int(slot_index),
                float(row[2]),
                float(row[3]),
                float(row[4]),
            )
        )
    if not known_rows:
        return None

    axis_deg = float(basis.get("axis_deg", pad.auto_nds_azimuth_deg)) % 360.0
    spacing_ref_m = float(max(float(basis.get("spacing_ref_m", 0.0)), 0.0))
    if spacing_ref_m <= SMALL:
        spacing_ref_m = float(DEFAULT_PAD_SPACING_M)

    z_step_candidates = [
        float((right_z - left_z) / (right_index - left_index))
        for (left_index, left_z, _left_projection, _left_cross), (
            right_index,
            right_z,
            _right_projection,
            _right_cross,
        ) in zip(known_rows, known_rows[1:], strict=False)
        if int(right_index - left_index) > 0
    ]
    z_step_m = (
        float(np.median(np.asarray(z_step_candidates, dtype=float)))
        if z_step_candidates
        else 0.0
    )
    projection_origin_candidates = [
        float(projection - slot_index * spacing_ref_m)
        for slot_index, _z_value, projection, _cross_projection in known_rows
    ]
    projection_origin = (
        float(np.median(np.asarray(projection_origin_candidates, dtype=float)))
        if projection_origin_candidates
        else float(known_rows[0][2])
    )
    cross_projection_values = [
        float(cross_projection)
        for _slot_index, _z_value, _projection, cross_projection in known_rows
    ]
    cross_projection_ref = (
        float(np.median(np.asarray(cross_projection_values, dtype=float)))
        if cross_projection_values
        else 0.0
    )
    z_origin_candidates = [
        float(z_value - slot_index * z_step_m)
        for slot_index, z_value, _projection, _cross_projection in known_rows
    ]
    z_origin = (
        float(np.median(np.asarray(z_origin_candidates, dtype=float)))
        if z_origin_candidates
        else float(known_rows[0][1])
    )

    angle_rad = np.deg2rad(axis_deg)
    ux = float(np.sin(angle_rad))
    uy = float(np.cos(angle_rad))
    vx = float(-uy)
    vy = float(ux)

    def _slot_xyz(slot_position: float) -> tuple[float, float, float]:
        projection = float(projection_origin + slot_position * spacing_ref_m)
        z_value = float(z_origin + slot_position * z_step_m)
        return (
            float(projection * ux + cross_projection_ref * vx),
            float(projection * uy + cross_projection_ref * vy),
            float(z_value),
        )

    center_slot_position = 0.5 * float(max(len(ordered_slot_names) - 1, 0))
    return {
        "axis_deg": float(axis_deg),
        "spacing_ref_m": float(spacing_ref_m),
        "z_step_m": float(z_step_m),
        "ordered_slot_names": ordered_slot_names,
        "first_anchor_xyz": _slot_xyz(0.0),
        "center_anchor_xyz": _slot_xyz(center_slot_position),
    }


def _uniform_pad_surface_assignments(
    *,
    base_order_names: tuple[str, ...],
    ordered_names: list[str],
    cfg: Mapping[str, object],
    z_step_m: float = 0.0,
) -> tuple[PadSurfaceAssignment, ...]:
    angle_rad = np.deg2rad(float(cfg["nds_azimuth_deg"]) % 360.0)
    ux = float(np.sin(angle_rad))
    uy = float(np.cos(angle_rad))
    center_slot_index = 0.5 * float(max(len(base_order_names) - 1, 0))
    anchor_mode = str(
        cfg.get("surface_anchor_mode", DEFAULT_PAD_SURFACE_ANCHOR_MODE)
    )
    assignments: list[PadSurfaceAssignment] = []
    for slot_index, (source_well_name, well_name) in enumerate(
        zip(base_order_names, ordered_names, strict=True),
        start=1,
    ):
        if anchor_mode == PAD_SURFACE_ANCHOR_CENTER:
            slot_offset = float(slot_index - 1) - center_slot_index
        else:
            slot_offset = float(slot_index - 1)
        assignments.append(
            PadSurfaceAssignment(
                slot_index=int(slot_index),
                well_name=str(well_name),
                surface_x_m=float(
                    float(cfg["first_surface_x"]) + slot_offset * float(cfg["spacing_m"]) * ux
                ),
                surface_y_m=float(
                    float(cfg["first_surface_y"]) + slot_offset * float(cfg["spacing_m"]) * uy
                ),
                surface_z_m=float(
                    float(cfg["first_surface_z"]) + slot_offset * float(z_step_m)
                ),
                source_well_name=str(source_well_name),
            )
        )
    return tuple(assignments)


def _expanded_source_slot_rows(
    *,
    pad: WellPad,
    basis: Mapping[str, object],
    spacing_fallback_m: float,
) -> tuple[tuple[str, float, float, float, float, float], ...]:
    raw_slot_rows = tuple(basis.get("slots", ()))
    if not raw_slot_rows:
        return ()

    ordered_names = [
        str(well.name)
        for well in sorted(
            pad.wells,
            key=lambda well: well_name_natural_sort_key(str(well.name)),
        )
    ]
    if not ordered_names:
        return ()

    slot_row_by_name = {
        str(slot_name): (
            float(x_value),
            float(y_value),
            float(z_value),
            float(projection),
            float(cross_projection),
        )
        for slot_name, x_value, y_value, z_value, projection, cross_projection in raw_slot_rows
    }
    known_rows: list[tuple[int, float, float, float]] = []
    for slot_index, well_name in enumerate(ordered_names):
        row = slot_row_by_name.get(str(well_name))
        if row is None:
            continue
        known_rows.append(
            (
                int(slot_index),
                float(row[2]),
                float(row[3]),
                float(row[4]),
            )
        )
    if not known_rows:
        return ()

    axis_deg = float(basis.get("axis_deg", 0.0)) % 360.0
    angle_rad = np.deg2rad(axis_deg)
    ux = float(np.sin(angle_rad))
    uy = float(np.cos(angle_rad))
    vx = float(-uy)
    vy = float(ux)

    spacing_m = float(max(float(spacing_fallback_m), 0.0))
    if spacing_m <= SMALL:
        spacing_m = float(max(float(basis.get("spacing_ref_m", 0.0)), 0.0))
    if spacing_m <= SMALL:
        spacing_m = float(DEFAULT_PAD_SPACING_M)

    def _step_between(
        left: tuple[int, float, float, float],
        right: tuple[int, float, float, float],
    ) -> tuple[float, float, float]:
        left_index, left_z, left_projection, left_cross = left
        right_index, right_z, right_projection, right_cross = right
        span = int(right_index - left_index)
        if span <= 0:
            return (float(spacing_m), 0.0, 0.0)
        return (
            float((right_z - left_z) / span),
            float((right_projection - left_projection) / span),
            float((right_cross - left_cross) / span),
        )

    first_known = known_rows[0]
    last_known = known_rows[-1]
    first_step = (
        _step_between(known_rows[0], known_rows[1])
        if len(known_rows) > 1
        else (0.0, float(spacing_m), 0.0)
    )
    last_step = (
        _step_between(known_rows[-2], known_rows[-1])
        if len(known_rows) > 1
        else (0.0, float(spacing_m), 0.0)
    )

    expanded: list[tuple[str, float, float, float, float, float]] = []
    for slot_index, well_name in enumerate(ordered_names):
        exact_row = slot_row_by_name.get(str(well_name))
        if exact_row is not None:
            x_value, y_value, z_value, projection, cross_projection = exact_row
            expanded.append(
                (
                    str(well_name),
                    float(x_value),
                    float(y_value),
                    float(z_value),
                    float(projection),
                    float(cross_projection),
                )
            )
            continue

        left_known = next(
            (
                item
                for item in reversed(known_rows)
                if int(item[0]) < int(slot_index)
            ),
            None,
        )
        right_known = next(
            (
                item
                for item in known_rows
                if int(item[0]) > int(slot_index)
            ),
            None,
        )
        if left_known is not None and right_known is not None:
            left_index, left_z, left_projection, left_cross = left_known
            right_index, right_z, right_projection, right_cross = right_known
            span = float(right_index - left_index)
            ratio = float((slot_index - left_index) / span) if span > 0.0 else 0.0
            z_value = float(left_z + ratio * (right_z - left_z))
            projection = float(
                left_projection + ratio * (right_projection - left_projection)
            )
            cross_projection = float(
                left_cross + ratio * (right_cross - left_cross)
            )
        elif left_known is not None:
            left_index, left_z, left_projection, left_cross = left_known
            delta = int(slot_index - left_index)
            z_value = float(left_z + delta * float(last_step[0]))
            projection = float(left_projection + delta * float(last_step[1]))
            cross_projection = float(left_cross + delta * float(last_step[2]))
        else:
            right_index, right_z, right_projection, right_cross = (
                right_known if right_known is not None else first_known
            )
            delta = int(right_index - slot_index)
            z_value = float(right_z - delta * float(first_step[0]))
            projection = float(right_projection - delta * float(first_step[1]))
            cross_projection = float(right_cross - delta * float(first_step[2]))

        expanded.append(
            (
                str(well_name),
                float(projection * ux + cross_projection * vx),
                float(projection * uy + cross_projection * vy),
                float(z_value),
                float(projection),
                float(cross_projection),
            )
        )
    return tuple(expanded)


def pad_anchor_defaults(
    session_state: Mapping[str, object],
    *,
    pad: WellPad,
    anchor_mode: str,
) -> tuple[float, float, float]:
    metadata_raw = session_state.get("wt_pad_detected_meta", {})
    metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
    pad_meta = metadata.get(str(pad.pad_id))
    basis = _source_defined_pad_basis(pad, pad_meta)
    if basis is None:
        defaults = pad_config_defaults(pad)
        return (
            float(defaults["first_surface_x"]),
            float(defaults["first_surface_y"]),
            float(defaults["first_surface_z"]),
        )
    if not _preserves_source_surface_slots(pad_meta):
        partial_layout = _partial_source_surface_layout(pad=pad, basis=basis)
        if partial_layout is not None:
            anchor_xyz = (
                partial_layout["center_anchor_xyz"]
                if str(anchor_mode) == PAD_SURFACE_ANCHOR_CENTER
                else partial_layout["first_anchor_xyz"]
            )
            return (
                float(anchor_xyz[0]),
                float(anchor_xyz[1]),
                float(anchor_xyz[2]),
            )
    if str(anchor_mode) == PAD_SURFACE_ANCHOR_CENTER:
        center_xyz = basis["center_anchor_xyz"]
        return (
            float(center_xyz[0]),
            float(center_xyz[1]),
            float(center_xyz[2]),
        )
    first_xyz = basis["first_anchor_xyz"]
    return (float(first_xyz[0]), float(first_xyz[1]), float(first_xyz[2]))


def _pad_effective_auto_order_mode(
    session_state: Mapping[str, object],
    *,
    pad: WellPad,
    config: Mapping[str, object],
    pad_meta: object,
) -> str:
    if _uses_source_surface_basis(
        pad,
        pad_meta,
        _source_defined_pad_basis(pad, pad_meta),
    ) and (
        not bool(config.get(WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY, False))
        or not bool(config.get(WT_PAD_APPLY_AUTO_ORDER_KEY, False))
    ):
        return PAD_WELL_AUTO_ORDER_PROJECTION
    return pad_auto_order_mode(session_state)


def pad_surface_assignments(
    session_state: Mapping[str, object],
    *,
    pad: WellPad,
    config: Mapping[str, object] | None = None,
) -> tuple[PadSurfaceAssignment, ...]:
    metadata_raw = session_state.get("wt_pad_detected_meta", {})
    metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
    config_raw = session_state.get("wt_pad_configs", {})
    config_map = config_raw if isinstance(config_raw, Mapping) else {}
    pad_meta = metadata.get(str(pad.pad_id))
    defaults = _pad_defaults_for_metadata(pad, pad_meta)
    current = (
        config
        if isinstance(config, Mapping)
        else (
            config_map.get(str(pad.pad_id), {})
            if isinstance(config_map.get(str(pad.pad_id), {}), Mapping)
            else {}
        )
    )
    cfg = _merge_pad_config(
        current=current,
        defaults=defaults,
        fixed_slots=pad_fixed_slots_from_config(pad=pad, config=current),
    )
    fixed_slots = pad_fixed_slots_from_config(pad=pad, config=cfg)
    basis = _source_defined_pad_basis(pad, pad_meta)
    partial_layout = (
        _partial_source_surface_layout(pad=pad, basis=basis)
        if basis is not None and not _preserves_source_surface_slots(pad_meta)
        else None
    )

    if _uses_source_surface_basis(pad, pad_meta, basis):
        if partial_layout is not None:
            base_order_names = list(partial_layout["ordered_slot_names"])
            if bool(cfg.get(WT_PAD_APPLY_AUTO_ORDER_KEY, False)):
                ordered_names = [
                    str(well.name)
                    for well in ordered_pad_wells(
                        pad=pad,
                        nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
                        fixed_slots=fixed_slots,
                        auto_order_mode=pad_auto_order_mode(session_state),
                    )
                ]
            else:
                ordered_names = _apply_fixed_slots_to_ordered_names(
                    base_order_names,
                    fixed_slots=fixed_slots,
                )
            return _uniform_pad_surface_assignments(
                base_order_names=tuple(base_order_names),
                ordered_names=ordered_names,
                cfg=cfg,
                z_step_m=float(partial_layout["z_step_m"]),
            )

        slot_rows = _expanded_source_slot_rows(
            pad=pad,
            basis=basis,
            spacing_fallback_m=float(cfg["spacing_m"]),
        )
        if not slot_rows:
            slot_rows = tuple(basis.get("slots", ()))
        base_order_names = [
            str(slot_name)
            for slot_name, *_rest in slot_rows
        ]
        if _preserves_source_surface_slots(pad_meta) and not bool(
            cfg.get(WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY, False)
        ):
            ordered_names = _apply_fixed_slots_to_ordered_names(
                base_order_names,
                fixed_slots=fixed_slots,
            )
            slot_row_by_name = {
                str(slot_name): (
                    float(x_value),
                    float(y_value),
                    float(z_value),
                )
                for slot_name, x_value, y_value, z_value, *_rest in slot_rows
            }
            return tuple(
                PadSurfaceAssignment(
                    slot_index=int(index),
                    well_name=str(well_name),
                    surface_x_m=float(slot_row_by_name[str(source_well_name)][0]),
                    surface_y_m=float(slot_row_by_name[str(source_well_name)][1]),
                    surface_z_m=float(slot_row_by_name[str(source_well_name)][2]),
                    source_well_name=str(source_well_name),
                )
                for index, (source_well_name, well_name) in enumerate(
                    zip(base_order_names, ordered_names, strict=True),
                    start=1,
                )
            )

        anchor_mode = str(
            cfg.get("surface_anchor_mode", DEFAULT_PAD_SURFACE_ANCHOR_MODE)
        )
        if anchor_mode == PAD_SURFACE_ANCHOR_CENTER:
            source_anchor_xyz = basis["center_anchor_xyz"]
            source_anchor_projection = float(basis["center_anchor_projection"])
            source_anchor_cross_projection = float(
                basis["center_anchor_cross_projection"]
            )
        else:
            source_anchor_xyz = basis["first_anchor_xyz"]
            source_anchor_projection = float(basis["first_anchor_projection"])
            source_anchor_cross_projection = float(
                basis["first_anchor_cross_projection"]
            )
        current_anchor_x = float(cfg["first_surface_x"])
        current_anchor_y = float(cfg["first_surface_y"])
        current_anchor_z = float(cfg["first_surface_z"])
        current_angle_rad = np.deg2rad(float(cfg["nds_azimuth_deg"]) % 360.0)
        current_ux = float(np.sin(current_angle_rad))
        current_uy = float(np.cos(current_angle_rad))
        current_vx = float(-current_uy)
        current_vy = float(current_ux)
        spacing_ref_m = float(max(basis["spacing_ref_m"], 0.0))
        spacing_m = float(max(cfg["spacing_m"], 0.0))
        scale = (
            float(spacing_m / spacing_ref_m)
            if spacing_ref_m > SMALL
            else 0.0
        )
        transformed_slots: list[tuple[str, float, float, float]] = []
        for slot_name, x_value, y_value, z_value, projection, cross_projection in slot_rows:
            relative_projection = float(projection) - source_anchor_projection
            relative_cross_projection = (
                float(cross_projection) - source_anchor_cross_projection
            )
            relative_z = float(z_value) - float(source_anchor_xyz[2])
            transformed_slots.append(
                (
                    str(slot_name),
                    float(
                        current_anchor_x
                        + relative_projection * scale * current_ux
                        + relative_cross_projection * current_vx
                    ),
                    float(
                        current_anchor_y
                        + relative_projection * scale * current_uy
                        + relative_cross_projection * current_vy
                    ),
                    float(current_anchor_z + relative_z),
                )
            )

        if bool(cfg.get(WT_PAD_APPLY_AUTO_ORDER_KEY, False)):
            ordered_names = [
                str(well.name)
                for well in ordered_pad_wells(
                    pad=pad,
                    nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
                    fixed_slots=fixed_slots,
                    auto_order_mode=pad_auto_order_mode(session_state),
                )
            ]
        else:
            ordered_names = _apply_fixed_slots_to_ordered_names(
                [str(slot_name) for slot_name, _x, _y, _z in transformed_slots],
                fixed_slots=fixed_slots,
            )

        return tuple(
            PadSurfaceAssignment(
                slot_index=int(index),
                well_name=str(well_name),
                surface_x_m=float(surface_x),
                surface_y_m=float(surface_y),
                surface_z_m=float(surface_z),
                source_well_name=str(source_well_name),
            )
            for index, (
                (source_well_name, surface_x, surface_y, surface_z),
                well_name,
            ) in enumerate(
                zip(transformed_slots, ordered_names, strict=True),
                start=1,
            )
        )

    auto_order_mode = _pad_effective_auto_order_mode(
        session_state,
        pad=pad,
        config=cfg,
        pad_meta=pad_meta,
    )
    ordered_wells = ordered_pad_wells(
        pad=pad,
        nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
        fixed_slots=fixed_slots,
        auto_order_mode=auto_order_mode,
    )
    ordered_names = [str(well.name) for well in ordered_wells]
    return _uniform_pad_surface_assignments(
        base_order_names=tuple(ordered_names),
        ordered_names=ordered_names,
        cfg=cfg,
        z_step_m=0.0,
    )


def ensure_pad_configs(
    session_state: MutableMapping[str, object],
    *,
    base_records: list[WelltrackRecord],
) -> list[WellPad]:
    pads, metadata = detect_ui_pads(
        base_records,
        remembered_degenerate_well_names=ptc_target_import.simple_dev_target_well_names_from_state(
            session_state
        ),
    )
    existing_raw = session_state.get("wt_pad_configs", {})
    existing = existing_raw if isinstance(existing_raw, Mapping) else {}
    merged: dict[str, dict[str, object]] = {}
    for pad in pads:
        defaults = _pad_defaults_for_metadata(pad, metadata.get(str(pad.pad_id)))
        current_raw = existing.get(str(pad.pad_id), {})
        current = current_raw if isinstance(current_raw, Mapping) else {}
        current_fixed_slots = pad_fixed_slots_from_config(
            pad=pad,
            config=current,
        )
        merged[str(pad.pad_id)] = _merge_pad_config(
            current=current,
            defaults=defaults,
            fixed_slots=current_fixed_slots,
        )
        pad_meta = metadata.get(str(pad.pad_id))
        if isinstance(pad_meta, DetectedPadUiMeta) and bool(
            pad_meta.source_surfaces_defined
        ) and not bool(current.get(WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY, False)):
            merged[str(pad.pad_id)] = {
                **dict(defaults),
                WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY: False,
                "fixed_slots": current_fixed_slots,
            }
    session_state["wt_pad_configs"] = merged
    session_state["wt_pad_detected_meta"] = metadata

    pad_ids = [str(pad.pad_id) for pad in pads]
    if not pad_ids:
        session_state["wt_pad_selected_id"] = ""
        return pads
    if str(session_state.get("wt_pad_selected_id", "")) not in pad_ids:
        session_state["wt_pad_selected_id"] = pad_ids[0]
    return pads


def build_pad_plan_map(
    session_state: Mapping[str, object],
    pads: list[WellPad],
) -> dict[str, PadLayoutPlan]:
    config_raw = session_state.get("wt_pad_configs", {})
    config_map = config_raw if isinstance(config_raw, Mapping) else {}
    metadata_raw = session_state.get("wt_pad_detected_meta", {})
    metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
    plan_map: dict[str, PadLayoutPlan] = {}
    for pad in pads:
        pad_id = str(pad.pad_id)
        pad_meta = metadata.get(pad_id)
        default_cfg = _pad_defaults_for_metadata(pad, pad_meta)
        cfg_raw = config_map.get(pad_id, default_cfg)
        cfg = cfg_raw if isinstance(cfg_raw, Mapping) else default_cfg
        if isinstance(pad_meta, DetectedPadUiMeta) and bool(
            pad_meta.source_surfaces_defined
        ) and not bool(cfg.get(WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY, False)):
            continue
        auto_order_mode = _pad_effective_auto_order_mode(
            session_state,
            pad=pad,
            config=cfg,
            pad_meta=pad_meta,
        )
        resolved_nds_azimuth_deg = resolved_pad_nds_azimuth_deg(
            session_state,
            pad=pad,
            nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
        )
        resolved_cfg = {
            **dict(cfg),
            "nds_azimuth_deg": float(resolved_nds_azimuth_deg),
        }
        assignments = pad_surface_assignments(
            session_state,
            pad=pad,
            config=resolved_cfg,
        )
        plan_map[pad_id] = PadLayoutPlan(
            pad_id=pad_id,
            first_surface_x=float(resolved_cfg["first_surface_x"]),
            first_surface_y=float(resolved_cfg["first_surface_y"]),
            first_surface_z=float(resolved_cfg["first_surface_z"]),
            spacing_m=float(max(resolved_cfg["spacing_m"], 0.0)),
            nds_azimuth_deg=resolved_nds_azimuth_deg,
            surface_anchor_mode=str(
                resolved_cfg.get("surface_anchor_mode", DEFAULT_PAD_SURFACE_ANCHOR_MODE)
            ),
            fixed_slots=pad_fixed_slots_from_config(
                pad=pad,
                config=resolved_cfg,
            ),
            auto_order_mode=auto_order_mode,
            surface_positions_by_well_name=tuple(
                (
                    str(item.well_name),
                    float(item.surface_x_m),
                    float(item.surface_y_m),
                    float(item.surface_z_m),
                )
                for item in assignments
            ),
        )
    return plan_map


def project_pads_for_ui(
    session_state: MutableMapping[str, object],
    records: list[WelltrackRecord],
) -> list[WellPad]:
    base_records = session_state.get("wt_records_original")
    source_records = (
        list(base_records)
        if isinstance(base_records, list) and base_records
        else list(records)
    )
    pads, metadata = detect_ui_pads(
        source_records,
        remembered_degenerate_well_names=ptc_target_import.simple_dev_target_well_names_from_state(
            session_state
        ),
    )
    session_state["wt_pad_detected_meta"] = metadata
    return pads


def pad_display_label(pad: WellPad) -> str:
    return f"{str(pad.pad_id)} · {int(len(pad.wells))} скв."


def _resolved_auto_pad_names(
    prepared: list[
        tuple[
            float,
            float,
            float,
            list[_PadDetectionRecord],
        ]
    ],
) -> list[tuple[str, str]]:
    proposed: list[tuple[str, bool, int]] = []
    for cluster_index, (_x, _y, _z, cluster) in enumerate(prepared, start=1):
        well_names = [str(entry.record.name) for entry in cluster]
        component = _pad_name_component_for_cluster(well_names)
        if component:
            proposed.append((_format_auto_pad_name(component), False, cluster_index))
        else:
            proposed.append((f"PAD-{cluster_index:02d}", True, cluster_index))

    seen_counts: dict[str, int] = {}
    resolved: list[tuple[str, str]] = []
    for base_label, used_template, cluster_index in proposed:
        duplicate_index = seen_counts.get(base_label, 0)
        seen_counts[base_label] = duplicate_index + 1
        final_label = (
            str(base_label)
            if duplicate_index == 0
            else f"{base_label}{_alpha_suffix(duplicate_index)}"
        )
        notice = ""
        if used_template:
            notice = (
                "Все номера скважин на кусте отличаются, было принято "
                f'шаблонное название куста "{final_label}".'
            )
        resolved.append((final_label, notice))
    return resolved


def _pad_name_component_for_cluster(well_names: list[str]) -> str:
    normalized = [str(name).strip() for name in well_names if str(name).strip()]
    if not normalized:
        return ""
    reduced = [_reduced_well_name_for_pad(name) for name in normalized]
    common_component = _trim_pad_component(_longest_common_prefix(reduced))
    if common_component:
        return common_component
    repeated = Counter(component for component in reduced if component)
    if not repeated:
        return ""
    most_common = repeated.most_common()
    top_count = int(most_common[0][1])
    if top_count <= 1:
        return ""
    leaders = [component for component, count in most_common if int(count) == top_count]
    if len(leaders) != 1:
        return ""
    return str(leaders[0])


def _reduced_well_name_for_pad(name: str) -> str:
    trimmed = _trim_pad_component(str(name))
    match = _TRAILING_DIGITS_RE.fullmatch(trimmed)
    if match is None:
        return trimmed
    prefix, digits = match.groups()
    if len(digits) < 2:
        return trimmed
    reduced = _trim_pad_component(f"{prefix}{digits[:-2]}")
    return reduced or trimmed


def _trim_pad_component(value: str) -> str:
    return _PAD_COMPONENT_EDGE_RE.sub("", str(value).strip())


def _longest_common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while prefix and not str(value).startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix


def _format_auto_pad_name(component: str) -> str:
    trimmed = _trim_pad_component(component)
    if not trimmed:
        return ""
    if trimmed.isdigit():
        return f"Pad {trimmed}"
    return trimmed


def _alpha_suffix(index: int) -> str:
    value = int(index)
    if value <= 0:
        return ""
    chars: list[str] = []
    while value > 0:
        value -= 1
        chars.append(chr(ord("A") + (value % 26)))
        value //= 26
    return "".join(reversed(chars))


def pad_config_for_ui(
    session_state: Mapping[str, object],
    pad: WellPad,
) -> dict[str, object]:
    metadata_raw = session_state.get("wt_pad_detected_meta", {})
    metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
    pad_meta = metadata.get(str(pad.pad_id))
    defaults = _pad_defaults_for_metadata(pad, pad_meta)
    config_state = session_state.get("wt_pad_configs", {})
    current_map = config_state if isinstance(config_state, Mapping) else {}
    current = current_map.get(str(pad.pad_id), {})
    current_mapping = current if isinstance(current, Mapping) else {}
    fixed_config = current_mapping
    if isinstance(pad_meta, DetectedPadUiMeta) and bool(
        pad_meta.source_surfaces_defined
    ) and not bool(current_mapping.get(WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY, False)):
        current_mapping = {}
    return _merge_pad_config(
        current=current_mapping,
        defaults=defaults,
        fixed_slots=pad_fixed_slots_from_config(
            pad=pad,
            config=fixed_config,
        ),
    )


def pad_anchor_mode_label(mode: object) -> str:
    if str(mode) == PAD_SURFACE_ANCHOR_CENTER:
        return "Центр куста"
    return "S первой скважины"


def pad_auto_order_mode(session_state: Mapping[str, object]) -> str:
    if bool(session_state.get(WT_PAD_AUTO_ORDER_BY_TARGET_DEPTH_KEY, False)):
        return PAD_WELL_AUTO_ORDER_TARGET_DEPTH_DESC
    return DEFAULT_PAD_WELL_AUTO_ORDER_MODE


def pad_auto_order_mode_label(session_state: Mapping[str, object]) -> str:
    if pad_auto_order_mode(session_state) == PAD_WELL_AUTO_ORDER_TARGET_DEPTH_DESC:
        return "по глубине цели: от более глубоких к менее глубоким"
    if pad_auto_order_mode(session_state) == PAD_WELL_AUTO_ORDER_NAME:
        return "по имени скважины: A->Z, 1->99"
    return "по авто-порядку"


def resolved_pad_nds_azimuth_deg(
    session_state: Mapping[str, object],
    *,
    pad: WellPad,
    nds_azimuth_deg: float,
) -> float:
    metadata_raw = session_state.get("wt_pad_detected_meta", {})
    metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
    config_raw = session_state.get("wt_pad_configs", {})
    config_map = config_raw if isinstance(config_raw, Mapping) else {}
    pad_id = str(pad.pad_id)
    pad_meta = metadata.get(pad_id)
    default_cfg = _pad_defaults_for_metadata(pad, pad_meta)
    current_raw = config_map.get(pad_id, {})
    current = current_raw if isinstance(current_raw, Mapping) else {}
    cfg = _merge_pad_config(
        current={
            **dict(current),
            "nds_azimuth_deg": float(nds_azimuth_deg),
        },
        defaults=default_cfg,
        fixed_slots=pad_fixed_slots_from_config(
            pad=pad,
            config=current,
        ),
    )
    return aligned_pad_nds_azimuth_deg(
        pad,
        nds_azimuth_deg=float(nds_azimuth_deg),
        auto_order_mode=_pad_effective_auto_order_mode(
            session_state,
            pad=pad,
            config=cfg,
            pad_meta=pad_meta,
        ),
    )


def pad_membership(
    session_state: MutableMapping[str, object],
    records: list[WelltrackRecord],
) -> tuple[list[WellPad], dict[str, str], dict[str, tuple[str, ...]]]:
    pads = project_pads_for_ui(session_state, records)
    name_to_pad_id: dict[str, str] = {}
    well_names_by_pad_id: dict[str, tuple[str, ...]] = {}
    for pad in pads:
        pad_id = str(pad.pad_id)
        assignments = pad_surface_assignments(session_state, pad=pad)
        ordered_names = tuple(str(item.well_name) for item in assignments)
        well_names_by_pad_id[pad_id] = ordered_names
        for well_name in ordered_names:
            name_to_pad_id[well_name] = pad_id
    return pads, name_to_pad_id, well_names_by_pad_id


def normalize_focus_pad_id(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    requested_pad_id: str | None,
) -> str:
    pads, _, _ = pad_membership(session_state, records)
    valid_options = {WT_PAD_FOCUS_ALL, *(str(pad.pad_id) for pad in pads)}
    selected = str(requested_pad_id or "").strip()
    if not selected or selected not in valid_options:
        if len(pads) == 1:
            return str(pads[0].pad_id)
        return WT_PAD_FOCUS_ALL
    if selected == WT_PAD_FOCUS_ALL and len(pads) == 1:
        return str(pads[0].pad_id)
    return selected


def focus_pad_well_names(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str | None,
) -> tuple[str, ...]:
    normalized = normalize_focus_pad_id(
        session_state,
        records=records,
        requested_pad_id=focus_pad_id,
    )
    if normalized == WT_PAD_FOCUS_ALL:
        return ()
    _, _, well_names_by_pad_id = pad_membership(session_state, records)
    return tuple(well_names_by_pad_id.get(str(normalized), ()))


def focus_pad_fixed_well_names(
    session_state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str | None,
) -> tuple[str, ...]:
    normalized = normalize_focus_pad_id(
        session_state,
        records=records,
        requested_pad_id=focus_pad_id,
    )
    if normalized == WT_PAD_FOCUS_ALL:
        return ()
    pads = project_pads_for_ui(session_state, records)
    pad = next(
        (item for item in pads if str(item.pad_id) == str(normalized)),
        None,
    )
    if pad is None:
        return ()
    cfg = pad_config_for_ui(session_state, pad)
    return tuple(
        str(name)
        for _, name in pad_fixed_slots_from_config(pad=pad, config=cfg)
    )


def _pad_defaults_for_metadata(
    pad: WellPad,
    pad_meta: object,
) -> dict[str, object]:
    basis = _source_defined_pad_basis(pad, pad_meta)
    if _uses_source_surface_basis(pad, pad_meta, basis):
        defaults = pad_config_defaults(pad)
        partial_layout = (
            _partial_source_surface_layout(pad=pad, basis=basis)
            if basis is not None and not _preserves_source_surface_slots(pad_meta)
            else None
        )
        if partial_layout is not None:
            center_xyz = partial_layout["center_anchor_xyz"]
            return {
                "spacing_m": float(
                    max(float(partial_layout["spacing_ref_m"]), 0.0)
                ),
                "nds_azimuth_deg": float(partial_layout["axis_deg"]) % 360.0,
                "first_surface_x": float(center_xyz[0]),
                "first_surface_y": float(center_xyz[1]),
                "first_surface_z": float(center_xyz[2]),
                "surface_anchor_mode": DEFAULT_PAD_SURFACE_ANCHOR_MODE,
                WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY: False,
                WT_PAD_APPLY_AUTO_ORDER_KEY: False,
                "fixed_slots": (),
            }
        center_xyz = (
            basis["center_anchor_xyz"]
            if isinstance(basis.get("center_anchor_xyz"), tuple)
            else (
                float(getattr(pad_meta, "source_surface_x_m", pad.surface.x)),
                float(getattr(pad_meta, "source_surface_y_m", pad.surface.y)),
                float(getattr(pad_meta, "source_surface_z_m", pad.surface.z)),
            )
        )
        return {
            "spacing_m": float(
                max(
                    float(basis.get("spacing_ref_m", 0.0))
                    if float(basis.get("spacing_ref_m", 0.0)) > SMALL
                    else float(defaults["spacing_m"]),
                    0.0,
                )
            ),
            "nds_azimuth_deg": float(
                float(basis.get("axis_deg", defaults["nds_azimuth_deg"]))
            )
            % 360.0,
            "first_surface_x": float(center_xyz[0]),
            "first_surface_y": float(center_xyz[1]),
            "first_surface_z": float(center_xyz[2]),
            "surface_anchor_mode": DEFAULT_PAD_SURFACE_ANCHOR_MODE,
            WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY: False,
            WT_PAD_APPLY_AUTO_ORDER_KEY: False,
            "fixed_slots": (),
        }
    return pad_config_defaults(pad)


def _merge_pad_config(
    *,
    current: Mapping[str, object],
    defaults: Mapping[str, object],
    fixed_slots: tuple[tuple[int, str], ...],
) -> dict[str, object]:
    return {
        "spacing_m": float(current.get("spacing_m", defaults["spacing_m"])),
        "nds_azimuth_deg": float(
            current.get("nds_azimuth_deg", defaults["nds_azimuth_deg"])
        )
        % 360.0,
        "first_surface_x": float(
            current.get("first_surface_x", defaults["first_surface_x"])
        ),
        "first_surface_y": float(
            current.get("first_surface_y", defaults["first_surface_y"])
        ),
        "first_surface_z": float(
            current.get("first_surface_z", defaults["first_surface_z"])
        ),
        "surface_anchor_mode": str(
            current.get("surface_anchor_mode", defaults["surface_anchor_mode"])
        ),
        WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY: bool(
            current.get(
                WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY,
                defaults.get(WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY, False),
            )
        ),
        WT_PAD_APPLY_AUTO_ORDER_KEY: bool(
            current.get(
                WT_PAD_APPLY_AUTO_ORDER_KEY,
                defaults.get(WT_PAD_APPLY_AUTO_ORDER_KEY, False),
            )
        ),
        "fixed_slots": fixed_slots,
    }
