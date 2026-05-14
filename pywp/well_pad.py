from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from pywp.constants import SMALL
from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackRecord,
    welltrack_points_to_target_pairs,
)
from pywp.models import Point3D
from pywp.pydantic_base import FrozenModel

_EPS = SMALL
_PCA_AXIS_DOMINANCE_RATIO_MIN = 1.05
PAD_SURFACE_ANCHOR_FIRST = "first"
PAD_SURFACE_ANCHOR_CENTER = "center"


class PadWell(FrozenModel):
    name: str
    record_index: int
    midpoint_x: float
    midpoint_y: float
    midpoint_z: float


class WellPad(FrozenModel):
    pad_id: str
    surface: Point3D
    wells: tuple[PadWell, ...]
    auto_nds_azimuth_deg: float


class PadLayoutPlan(FrozenModel):
    pad_id: str
    first_surface_x: float
    first_surface_y: float
    first_surface_z: float
    spacing_m: float
    nds_azimuth_deg: float
    surface_anchor_mode: str = PAD_SURFACE_ANCHOR_FIRST
    fixed_slots: tuple[tuple[int, str], ...] = ()


def detect_well_pads(
    records: Iterable[WelltrackRecord],
    *,
    surface_tolerance_m: float = 1e-6,
) -> list[WellPad]:
    grouped: dict[tuple[int, int, int], list[PadWell]] = {}
    surface_anchor: dict[tuple[int, int, int], Point3D] = {}
    ordered_keys: list[tuple[int, int, int]] = []
    record_list = list(records)

    for record_index, record in enumerate(record_list):
        if not record.points:
            continue
        surface_point = record.points[0]
        key = _surface_key(
            x=float(surface_point.x),
            y=float(surface_point.y),
            z=float(surface_point.z),
            tolerance_m=surface_tolerance_m,
        )
        if key not in grouped:
            grouped[key] = []
            surface_anchor[key] = Point3D(
                x=float(surface_point.x),
                y=float(surface_point.y),
                z=float(surface_point.z),
            )
            ordered_keys.append(key)

        midpoint_x, midpoint_y, midpoint_z = _well_midpoint_xyz(record=record)
        grouped[key].append(
            PadWell(
                name=str(record.name),
                record_index=int(record_index),
                midpoint_x=float(midpoint_x),
                midpoint_y=float(midpoint_y),
                midpoint_z=float(midpoint_z),
            )
        )

    pads: list[WellPad] = []
    sorted_keys = sorted(
        ordered_keys,
        key=lambda key: (
            float(surface_anchor[key].x),
            float(surface_anchor[key].y),
            float(surface_anchor[key].z),
        ),
    )
    for index, key in enumerate(sorted_keys, start=1):
        surface = surface_anchor[key]
        wells = tuple(grouped[key])
        auto_azimuth = estimate_pad_nds_azimuth_deg(
            wells=wells,
            surface_x=float(surface.x),
            surface_y=float(surface.y),
        )
        pads.append(
            WellPad(
                pad_id=f"PAD-{index:02d}",
                surface=surface,
                wells=wells,
                auto_nds_azimuth_deg=float(auto_azimuth),
            )
        )
    return pads


def estimate_pad_nds_azimuth_deg(
    *,
    wells: tuple[PadWell, ...],
    surface_x: float,
    surface_y: float,
    surface_anchor_mode: str = PAD_SURFACE_ANCHOR_FIRST,
) -> float:
    """Estimate rig walking/skidding azimuth for a pad from target geometry.

    The primary heuristic is the dominant principal axis of the midpoint cloud
    ((t1 + t3) / 2 in XY). This is a reasonable geometric seed when wells have a
    clear elongated spread and the goal is to order them along the dominant pad
    direction. If the cloud is close to isotropic, the principal axis becomes
    numerically weak, so we fall back to the centroid direction from the pad
    surface. If that is also degenerate, no geometry-driven NDS exists and the
    function returns 0° as a stable default.
    """
    if not wells:
        return 0.0
    if len(wells) == 1:
        dx = float(wells[0].midpoint_x - surface_x)
        dy = float(wells[0].midpoint_y - surface_y)
        return _vector_to_azimuth_deg(dx=dx, dy=dy)

    points = np.array(
        [(well.midpoint_x, well.midpoint_y) for well in wells], dtype=float
    )
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        principal_index = int(np.argmax(eigenvalues))
        principal_value = float(eigenvalues[principal_index])
        minor_value = float(np.min(eigenvalues))
        if principal_value <= _EPS:
            direction = centroid - np.array([surface_x, surface_y], dtype=float)
        else:
            dominance_ratio = principal_value / max(minor_value, _EPS)
            if dominance_ratio < _PCA_AXIS_DOMINANCE_RATIO_MIN:
                direction = centroid - np.array([surface_x, surface_y], dtype=float)
            else:
                direction = eigenvectors[:, principal_index]
    except np.linalg.LinAlgError:
        direction = np.array([1.0, 0.0], dtype=float)

    norm = float(np.linalg.norm(direction))
    if norm <= _EPS:
        direction = centroid - np.array([surface_x, surface_y], dtype=float)
        norm = float(np.linalg.norm(direction))
    if norm <= _EPS:
        return 0.0

    unit = direction / norm
    reference = centroid - np.array([surface_x, surface_y], dtype=float)
    if str(surface_anchor_mode) == PAD_SURFACE_ANCHOR_CENTER:
        source_order = np.asarray(
            [float(index) for index, _ in enumerate(wells)],
            dtype=float,
        )
        source_order = source_order - float(np.mean(source_order))
        centered_projection = points @ unit
        centered_projection = centered_projection - float(np.mean(centered_projection))
        if float(np.dot(centered_projection, source_order)) < 0.0:
            unit = -unit
    elif float(np.dot(unit, reference)) < 0.0:
        unit = -unit
    return _vector_to_azimuth_deg(dx=float(unit[0]), dy=float(unit[1]))


def _normalized_fixed_slots(
    *,
    wells: tuple[PadWell, ...],
    fixed_slots: Iterable[tuple[int, str]] | None,
) -> tuple[tuple[int, str], ...]:
    well_names = {str(well.name) for well in wells}
    max_slot = len(wells)
    used_slots: set[int] = set()
    used_names: set[str] = set()
    normalized: list[tuple[int, str]] = []
    for raw_slot, raw_name in fixed_slots or ():
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            continue
        name = str(raw_name).strip()
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
    return tuple(sorted(normalized, key=lambda item: item[0]))


def ordered_pad_wells(
    pad: WellPad,
    *,
    nds_azimuth_deg: float,
    fixed_slots: Iterable[tuple[int, str]] | None = None,
) -> list[PadWell]:
    ux, uy = _azimuth_to_unit_xy(azimuth_deg=nds_azimuth_deg)
    sortable: list[tuple[float, str, PadWell]] = []
    for well in pad.wells:
        projection = float(well.midpoint_x * ux + well.midpoint_y * uy)
        sortable.append((projection, str(well.name), well))
    sortable.sort(key=lambda item: (item[0], item[1]))
    base_order = [item[2] for item in sortable]
    normalized_fixed_slots = _normalized_fixed_slots(
        wells=pad.wells,
        fixed_slots=fixed_slots,
    )
    if not normalized_fixed_slots:
        return base_order

    well_by_name = {str(well.name): well for well in pad.wells}
    fixed_names = {name for _, name in normalized_fixed_slots}
    ordered: list[PadWell | None] = [None] * len(base_order)
    for slot, name in normalized_fixed_slots:
        ordered[int(slot) - 1] = well_by_name[name]

    remaining = [well for well in base_order if str(well.name) not in fixed_names]
    remaining_iter = iter(remaining)
    for index, item in enumerate(ordered):
        if item is None:
            ordered[index] = next(remaining_iter)
    return [well for well in ordered if well is not None]


def apply_pad_layout(
    *,
    records: Iterable[WelltrackRecord],
    pads: Iterable[WellPad],
    plan_by_pad_id: dict[str, PadLayoutPlan],
) -> list[WelltrackRecord]:
    updated_records = list(records)
    name_to_index = {
        str(record.name): index for index, record in enumerate(updated_records)
    }

    for pad in pads:
        plan = plan_by_pad_id.get(str(pad.pad_id))
        if plan is None:
            continue

        spacing_m = float(max(plan.spacing_m, 0.0))
        ux, uy = _azimuth_to_unit_xy(azimuth_deg=float(plan.nds_azimuth_deg))
        ordered = ordered_pad_wells(
            pad=pad,
            nds_azimuth_deg=float(plan.nds_azimuth_deg),
            fixed_slots=tuple(plan.fixed_slots),
        )
        anchor_mode = str(
            getattr(plan, "surface_anchor_mode", PAD_SURFACE_ANCHOR_FIRST)
        )
        center_slot_index = 0.5 * float(max(len(ordered) - 1, 0))

        for slot_index, well in enumerate(ordered):
            record_index = name_to_index.get(str(well.name))
            if record_index is None:
                continue
            source_record = updated_records[record_index]
            if not source_record.points:
                continue

            source_surface = source_record.points[0]
            if anchor_mode == PAD_SURFACE_ANCHOR_CENTER:
                shift_m = (float(slot_index) - center_slot_index) * spacing_m
            else:
                shift_m = float(slot_index) * spacing_m
            new_surface = WelltrackPoint(
                x=float(plan.first_surface_x + shift_m * ux),
                y=float(plan.first_surface_y + shift_m * uy),
                z=float(plan.first_surface_z),
                md=float(source_surface.md),
            )
            updated_records[record_index] = WelltrackRecord(
                name=str(source_record.name),
                points=(new_surface, *source_record.points[1:]),
            )

    return updated_records


def _surface_key(
    *, x: float, y: float, z: float, tolerance_m: float
) -> tuple[int, int, int]:
    tol = float(max(tolerance_m, _EPS))
    return (
        int(round(x / tol)),
        int(round(y / tol)),
        int(round(z / tol)),
    )


def _well_midpoint_xyz(record: WelltrackRecord) -> tuple[float, float, float]:
    if len(record.points) >= 3:
        try:
            _, target_pairs = welltrack_points_to_target_pairs(tuple(record.points))
            target_points = [
                point
                for pair_t1, pair_t3 in target_pairs
                for point in (pair_t1, pair_t3)
            ]
            return (
                float(np.mean([point.x for point in target_points])),
                float(np.mean([point.y for point in target_points])),
                float(np.mean([point.z for point in target_points])),
            )
        except (TypeError, ValueError):
            pass
    first = record.points[0]
    return float(first.x), float(first.y), float(first.z)


def _azimuth_to_unit_xy(*, azimuth_deg: float) -> tuple[float, float]:
    angle_rad = math.radians(float(azimuth_deg) % 360.0)
    return float(math.sin(angle_rad)), float(math.cos(angle_rad))


def _vector_to_azimuth_deg(*, dx: float, dy: float) -> float:
    if abs(float(dx)) <= _EPS and abs(float(dy)) <= _EPS:
        return 0.0
    return float(math.degrees(math.atan2(float(dx), float(dy))) % 360.0)
