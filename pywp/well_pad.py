from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackRecord,
    welltrack_points_to_targets,
)
from pywp.models import Point3D

_EPS = 1e-9


@dataclass(frozen=True)
class PadWell:
    name: str
    record_index: int
    midpoint_x: float
    midpoint_y: float
    midpoint_z: float


@dataclass(frozen=True)
class WellPad:
    pad_id: str
    surface: Point3D
    wells: tuple[PadWell, ...]
    auto_nds_azimuth_deg: float


@dataclass(frozen=True)
class PadLayoutPlan:
    pad_id: str
    first_surface_x: float
    first_surface_y: float
    first_surface_z: float
    spacing_m: float
    nds_azimuth_deg: float


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
) -> float:
    if not wells:
        return 0.0
    if len(wells) == 1:
        dx = float(wells[0].midpoint_x - surface_x)
        dy = float(wells[0].midpoint_y - surface_y)
        return _vector_to_azimuth_deg(dx=dx, dy=dy)

    points = np.array([(well.midpoint_x, well.midpoint_y) for well in wells], dtype=float)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
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
    if float(np.dot(unit, reference)) < 0.0:
        unit = -unit
    return _vector_to_azimuth_deg(dx=float(unit[0]), dy=float(unit[1]))


def ordered_pad_wells(pad: WellPad, *, nds_azimuth_deg: float) -> list[PadWell]:
    ux, uy = _azimuth_to_unit_xy(azimuth_deg=nds_azimuth_deg)
    sortable: list[tuple[float, str, PadWell]] = []
    for well in pad.wells:
        projection = float(well.midpoint_x * ux + well.midpoint_y * uy)
        sortable.append((projection, str(well.name), well))
    sortable.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in sortable]


def apply_pad_layout(
    *,
    records: Iterable[WelltrackRecord],
    pads: Iterable[WellPad],
    plan_by_pad_id: dict[str, PadLayoutPlan],
) -> list[WelltrackRecord]:
    updated_records = list(records)
    name_to_index = {str(record.name): index for index, record in enumerate(updated_records)}

    for pad in pads:
        plan = plan_by_pad_id.get(str(pad.pad_id))
        if plan is None:
            continue

        spacing_m = float(max(plan.spacing_m, 0.0))
        ux, uy = _azimuth_to_unit_xy(azimuth_deg=float(plan.nds_azimuth_deg))
        ordered = ordered_pad_wells(pad=pad, nds_azimuth_deg=float(plan.nds_azimuth_deg))

        for slot_index, well in enumerate(ordered):
            record_index = name_to_index.get(str(well.name))
            if record_index is None:
                continue
            source_record = updated_records[record_index]
            if not source_record.points:
                continue

            source_surface = source_record.points[0]
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


def _surface_key(*, x: float, y: float, z: float, tolerance_m: float) -> tuple[int, int, int]:
    tol = float(max(tolerance_m, _EPS))
    return (
        int(round(x / tol)),
        int(round(y / tol)),
        int(round(z / tol)),
    )


def _well_midpoint_xyz(record: WelltrackRecord) -> tuple[float, float, float]:
    if len(record.points) >= 3:
        try:
            _, t1, t3 = welltrack_points_to_targets(tuple(record.points[:3]))
            return (
                float(0.5 * (t1.x + t3.x)),
                float(0.5 * (t1.y + t3.y)),
                float(0.5 * (t1.z + t3.z)),
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

