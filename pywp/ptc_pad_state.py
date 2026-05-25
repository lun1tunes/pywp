from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from pywp.constants import SMALL
from pywp.eclipse_welltrack import WelltrackRecord, welltrack_points_to_target_pairs
from pywp.models import Point3D
from pywp.well_pad import (
    DEFAULT_PAD_WELL_AUTO_ORDER_MODE,
    PAD_WELL_AUTO_ORDER_NAME,
    PAD_WELL_AUTO_ORDER_TARGET_DEPTH_DESC,
    PAD_SURFACE_ANCHOR_CENTER,
    PadLayoutPlan,
    PadWell,
    WellPad,
    aligned_pad_nds_azimuth_deg,
    estimate_pad_nds_azimuth_deg,
    ordered_pad_wells,
)

__all__ = [
    "DEFAULT_PAD_SPACING_M",
    "DEFAULT_PAD_SURFACE_ANCHOR_MODE",
    "WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M",
    "WT_PAD_FOCUS_ALL",
    "DetectedPadUiMeta",
    "build_pad_plan_map",
    "detect_ui_pads",
    "ensure_pad_configs",
    "estimate_surface_pad_axis_deg",
    "focus_pad_fixed_well_names",
    "focus_pad_well_names",
    "inferred_surface_spacing_m",
    "normalize_focus_pad_id",
    "pad_auto_order_mode",
    "pad_auto_order_mode_label",
    "pad_anchor_mode_label",
    "pad_config_defaults",
    "pad_config_for_ui",
    "pad_display_label",
    "pad_fixed_slots_editor_rows",
    "pad_fixed_slots_from_config",
    "pad_fixed_slots_from_editor",
    "pad_membership",
    "project_pads_for_ui",
    "record_midpoint_xyz",
    "resolved_pad_nds_azimuth_deg",
    "source_surface_xyz",
]

DEFAULT_PAD_SPACING_M = 20.0
DEFAULT_PAD_SURFACE_ANCHOR_MODE = PAD_SURFACE_ANCHOR_CENTER
WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M = 400.0
WT_PAD_FOCUS_ALL = "__all_pads__"
WT_PAD_AUTO_ORDER_BY_TARGET_DEPTH_KEY = "wt_pad_auto_order_by_target_depth"
WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY = "allow_source_surface_edit"


@dataclass(frozen=True)
class DetectedPadUiMeta:
    source_surfaces_defined: bool
    inferred_spacing_m: float
    source_surface_x_m: float
    source_surface_y_m: float
    source_surface_z_m: float
    source_surface_count: int
    auto_name_notice: str = ""


_PAD_COMPONENT_EDGE_RE = re.compile(r"^[\s_\-]+|[\s_\-]+$")
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(\d+)$")


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


def record_midpoint_xyz(record: WelltrackRecord) -> tuple[float, float, float]:
    points = tuple(record.points)
    if len(points) >= 3:
        try:
            _, target_pairs = welltrack_points_to_target_pairs(points)
            pair_t1, pair_t3 = target_pairs[0]
            target_points = (pair_t1, pair_t3)
            return (
                float(np.mean([point.x for point in target_points])),
                float(np.mean([point.y for point in target_points])),
                float(np.mean([point.z for point in target_points])),
            )
        except (TypeError, ValueError):
            pass
    surface_xyz = source_surface_xyz(record)
    return surface_xyz or (0.0, 0.0, 0.0)


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
) -> tuple[list[WellPad], dict[str, DetectedPadUiMeta]]:
    indexed_records = [
        (index, record, source_surface_xyz(record))
        for index, record in enumerate(records)
    ]
    indexed_records = [
        (index, record, surface_xyz)
        for index, record, surface_xyz in indexed_records
        if surface_xyz is not None
    ]
    if not indexed_records:
        return [], {}

    adjacency: dict[int, set[int]] = {
        index: set() for index, _, _ in indexed_records
    }
    for left_pos, (left_index, _, left_surface) in enumerate(indexed_records):
        for right_index, _, right_surface in indexed_records[left_pos + 1 :]:
            distance_xy = float(
                np.hypot(
                    float(left_surface[0]) - float(right_surface[0]),
                    float(left_surface[1]) - float(right_surface[1]),
                )
            )
            if distance_xy <= WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M + SMALL:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    by_index = {
        index: (index, record, surface_xyz)
        for index, record, surface_xyz in indexed_records
    }
    clusters: list[
        list[tuple[int, WelltrackRecord, tuple[float, float, float]]]
    ] = []
    visited: set[int] = set()
    for index, _, _ in indexed_records:
        if index in visited:
            continue
        queue = [index]
        visited.add(index)
        cluster: list[
            tuple[int, WelltrackRecord, tuple[float, float, float]]
        ] = []
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
            list[tuple[int, WelltrackRecord, tuple[float, float, float]]],
        ]
    ] = []
    for cluster in clusters:
        surface_xyzs = [surface_xyz for _, _, surface_xyz in cluster]
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
        unique_surface_keys: set[tuple[int, int, int]] = set()
        for record_index, record, surface_xyz in cluster:
            midpoint_x, midpoint_y, midpoint_z = record_midpoint_xyz(record)
            wells.append(
                PadWell(
                    name=str(record.name),
                    record_index=int(record_index),
                    midpoint_x=float(midpoint_x),
                    midpoint_y=float(midpoint_y),
                    midpoint_z=float(midpoint_z),
                )
            )
            surface_xyzs.append(surface_xyz)
            unique_surface_keys.add(
                (
                    round(float(surface_xyz[0]), 6),
                    round(float(surface_xyz[1]), 6),
                    round(float(surface_xyz[2]), 6),
                )
            )
        pad_id, auto_name_notice = resolved_pad_names[index - 1]
        source_surfaces_defined = len(unique_surface_keys) > 1
        auto_nds = estimate_surface_pad_axis_deg(surface_xyzs)
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
                surface_xyzs=surface_xyzs,
                nds_azimuth_deg=float(auto_nds),
            ),
            source_surface_x_m=float(center_x),
            source_surface_y_m=float(center_y),
            source_surface_z_m=float(center_z),
            source_surface_count=len(surface_xyzs),
            auto_name_notice=str(auto_name_notice),
        )
    return pads, metadata


def ensure_pad_configs(
    session_state: MutableMapping[str, object],
    *,
    base_records: list[WelltrackRecord],
) -> list[WellPad]:
    pads, metadata = detect_ui_pads(base_records)
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
    auto_order_mode = pad_auto_order_mode(session_state)
    plan_map: dict[str, PadLayoutPlan] = {}
    for pad in pads:
        pad_id = str(pad.pad_id)
        pad_meta = metadata.get(pad_id)
        default_cfg = pad_config_defaults(pad)
        cfg_raw = config_map.get(pad_id, default_cfg)
        cfg = cfg_raw if isinstance(cfg_raw, Mapping) else default_cfg
        if isinstance(pad_meta, DetectedPadUiMeta) and bool(
            pad_meta.source_surfaces_defined
        ) and not bool(cfg.get(WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY, False)):
            continue
        resolved_nds_azimuth_deg = resolved_pad_nds_azimuth_deg(
            session_state,
            pad=pad,
            nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
        )
        plan_map[pad_id] = PadLayoutPlan(
            pad_id=pad_id,
            first_surface_x=float(cfg["first_surface_x"]),
            first_surface_y=float(cfg["first_surface_y"]),
            first_surface_z=float(cfg["first_surface_z"]),
            spacing_m=float(max(cfg["spacing_m"], 0.0)),
            nds_azimuth_deg=resolved_nds_azimuth_deg,
            surface_anchor_mode=str(
                cfg.get("surface_anchor_mode", DEFAULT_PAD_SURFACE_ANCHOR_MODE)
            ),
            fixed_slots=pad_fixed_slots_from_config(
                pad=pad,
                config=cfg,
            ),
            auto_order_mode=auto_order_mode,
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
    pads, metadata = detect_ui_pads(source_records)
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
            list[tuple[int, WelltrackRecord, tuple[float, float, float]]],
        ]
    ],
) -> list[tuple[str, str]]:
    proposed: list[tuple[str, bool, int]] = []
    for cluster_index, (_x, _y, _z, cluster) in enumerate(prepared, start=1):
        well_names = [str(record.name) for _, record, _surface_xyz in cluster]
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
    return aligned_pad_nds_azimuth_deg(
        pad,
        nds_azimuth_deg=float(nds_azimuth_deg),
        auto_order_mode=pad_auto_order_mode(session_state),
    )


def pad_membership(
    session_state: MutableMapping[str, object],
    records: list[WelltrackRecord],
) -> tuple[list[WellPad], dict[str, str], dict[str, tuple[str, ...]]]:
    pads = project_pads_for_ui(session_state, records)
    auto_order_mode = pad_auto_order_mode(session_state)
    name_to_pad_id: dict[str, str] = {}
    well_names_by_pad_id: dict[str, tuple[str, ...]] = {}
    for pad in pads:
        pad_id = str(pad.pad_id)
        cfg = pad_config_for_ui(session_state, pad)
        ordered = ordered_pad_wells(
            pad=pad,
            nds_azimuth_deg=float(cfg["nds_azimuth_deg"]),
            fixed_slots=pad_fixed_slots_from_config(pad=pad, config=cfg),
            auto_order_mode=auto_order_mode,
        )
        ordered_names = tuple(str(item.name) for item in ordered)
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
    if isinstance(pad_meta, DetectedPadUiMeta) and bool(
        pad_meta.source_surfaces_defined
    ):
        return {
            "spacing_m": float(max(pad_meta.inferred_spacing_m, 0.0)),
            "nds_azimuth_deg": float(pad.auto_nds_azimuth_deg) % 360.0,
            "first_surface_x": float(pad_meta.source_surface_x_m),
            "first_surface_y": float(pad_meta.source_surface_y_m),
            "first_surface_z": float(pad_meta.source_surface_z_m),
            "surface_anchor_mode": DEFAULT_PAD_SURFACE_ANCHOR_MODE,
            WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY: False,
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
        "fixed_slots": fixed_slots,
    }
