from __future__ import annotations

import math
from typing import Iterable

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.pydantic_base import FrozenModel


class T1T3OrderIssue(FrozenModel):
    well_name: str
    t1_offset_m: float
    t3_offset_m: float
    delta_m: float


def detect_t1_t3_order_issues(
    records: Iterable[WelltrackRecord],
    *,
    min_delta_m: float = 0.5,
) -> list[T1T3OrderIssue]:
    issues: list[T1T3OrderIssue] = []
    min_delta = float(max(min_delta_m, 0.0))

    for record in records:
        points = tuple(record.points)
        if len(points) < 3:
            continue
        surface = points[0]
        t1 = points[1]
        t3 = points[2]
        t1_offset_m = _offset_xy(surface=surface, target=t1)
        t3_offset_m = _offset_xy(surface=surface, target=t3)
        delta_m = float(t1_offset_m - t3_offset_m)
        if delta_m > min_delta:
            issues.append(
                T1T3OrderIssue(
                    well_name=str(record.name),
                    t1_offset_m=float(t1_offset_m),
                    t3_offset_m=float(t3_offset_m),
                    delta_m=float(delta_m),
                )
            )

    issues.sort(key=lambda item: (float(item.delta_m), str(item.well_name)), reverse=True)
    return issues


def swap_t1_t3_for_wells(
    records: Iterable[WelltrackRecord],
    *,
    well_names: set[str],
) -> list[WelltrackRecord]:
    if not well_names:
        return list(records)

    updated: list[WelltrackRecord] = []
    for record in records:
        points = tuple(record.points)
        name = str(record.name)
        if len(points) < 3 or name not in well_names:
            updated.append(record)
            continue

        p0, p1, p2 = points[0], points[1], points[2]
        swapped_t1 = WelltrackPoint(
            x=float(p2.x),
            y=float(p2.y),
            z=float(p2.z),
            md=float(p1.md),
        )
        swapped_t3 = WelltrackPoint(
            x=float(p1.x),
            y=float(p1.y),
            z=float(p1.z),
            md=float(p2.md),
        )
        updated_points = (p0, swapped_t1, swapped_t3, *points[3:])
        updated.append(WelltrackRecord(name=name, points=updated_points))

    return updated


def _offset_xy(*, surface: WelltrackPoint, target: WelltrackPoint) -> float:
    dx = float(target.x - surface.x)
    dy = float(target.y - surface.y)
    return float(math.hypot(dx, dy))
