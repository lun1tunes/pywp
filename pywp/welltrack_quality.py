from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Iterable

from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackRecord,
    welltrack_multi_horizontal_level_count,
)
from pywp.models import Point3D
from pywp.pilot_wells import is_zbs_name, parent_name_for_zbs, well_name_key
from pywp.pydantic_base import FrozenModel


class T1T3OrderIssue(FrozenModel):
    well_name: str
    t1_offset_m: float
    t3_offset_m: float
    delta_m: float
    anchor_label: str = "S→"


def detect_t1_t3_order_issues(
    records: Iterable[WelltrackRecord],
    *,
    min_delta_m: float = 0.5,
    anchor_by_well_name: Mapping[str, WelltrackPoint | Point3D] | None = None,
) -> list[T1T3OrderIssue]:
    issues: list[T1T3OrderIssue] = []
    min_delta = float(max(min_delta_m, 0.0))
    anchor_by_key = {
        well_name_key(name): point
        for name, point in (anchor_by_well_name or {}).items()
        if str(name).strip()
    }

    for record in records:
        points = tuple(record.points)
        is_zbs = is_zbs_name(record.name)
        if is_zbs:
            if len(points) < 2 or len(points) % 2 != 0:
                continue
            anchor = anchor_by_key.get(well_name_key(record.name))
            if anchor is None:
                anchor = anchor_by_key.get(
                    well_name_key(parent_name_for_zbs(record.name))
                )
            if anchor is None:
                continue
            issue = _worst_t1_t3_order_issue_for_pairs(
                well_name=str(record.name),
                pairs=tuple(
                    (points[index], points[index + 1])
                    for index in range(0, len(points), 2)
                ),
                anchor=anchor,
                min_delta_m=min_delta,
                anchor_label="родитель→",
            )
            if issue is not None:
                issues.append(issue)
            continue
        else:
            if len(points) < 3:
                continue
            if welltrack_multi_horizontal_level_count(points) > 1:
                continue
            surface = points[0]
            t1 = points[1]
            t3 = points[2]
            anchor_label = "S→"

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
                    anchor_label=anchor_label,
                )
            )

    issues.sort(
        key=lambda item: (float(item.delta_m), str(item.well_name)),
        reverse=True,
    )
    return issues


def swap_t1_t3_for_wells(
    records: Iterable[WelltrackRecord],
    *,
    well_names: set[str],
    anchor_by_well_name: Mapping[str, WelltrackPoint | Point3D] | None = None,
    min_delta_m: float = 0.5,
) -> list[WelltrackRecord]:
    if not well_names:
        return list(records)

    updated: list[WelltrackRecord] = []
    anchor_by_key = {
        well_name_key(name): point
        for name, point in (anchor_by_well_name or {}).items()
        if str(name).strip()
    }
    min_delta = float(max(min_delta_m, 0.0))
    for record in records:
        points = tuple(record.points)
        name = str(record.name)
        if is_zbs_name(record.name):
            if name in well_names and len(points) >= 2 and len(points) % 2 == 0:
                anchor = anchor_by_key.get(well_name_key(name))
                if anchor is None:
                    anchor = anchor_by_key.get(
                        well_name_key(parent_name_for_zbs(name))
                    )
                updated_points = _swap_zbs_t1_t3_pairs(
                    points=points,
                    anchor=anchor,
                    min_delta_m=min_delta,
                )
                updated.append(WelltrackRecord(name=name, points=updated_points))
                continue
            updated.append(record)
            continue
        if len(points) < 3 or name not in well_names:
            updated.append(record)
            continue
        if welltrack_multi_horizontal_level_count(points) > 1:
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


def _worst_t1_t3_order_issue_for_pairs(
    *,
    well_name: str,
    pairs: tuple[tuple[WelltrackPoint, WelltrackPoint], ...],
    anchor: WelltrackPoint | Point3D,
    min_delta_m: float,
    anchor_label: str,
) -> T1T3OrderIssue | None:
    worst: T1T3OrderIssue | None = None
    for t1, t3 in pairs:
        t1_offset_m = _offset_xy(surface=anchor, target=t1)
        t3_offset_m = _offset_xy(surface=anchor, target=t3)
        delta_m = float(t1_offset_m - t3_offset_m)
        if delta_m <= float(min_delta_m):
            continue
        candidate = T1T3OrderIssue(
            well_name=str(well_name),
            t1_offset_m=float(t1_offset_m),
            t3_offset_m=float(t3_offset_m),
            delta_m=float(delta_m),
            anchor_label=str(anchor_label),
        )
        if worst is None or float(candidate.delta_m) > float(worst.delta_m):
            worst = candidate
    return worst


def _swap_zbs_t1_t3_pairs(
    *,
    points: tuple[WelltrackPoint, ...],
    anchor: WelltrackPoint | Point3D | None,
    min_delta_m: float,
) -> tuple[WelltrackPoint, ...]:
    updated: list[WelltrackPoint] = []
    for index in range(0, len(points), 2):
        t1, t3 = points[index], points[index + 1]
        should_swap = False
        if anchor is None:
            should_swap = len(points) == 2
        else:
            delta_m = _offset_xy(surface=anchor, target=t1) - _offset_xy(
                surface=anchor,
                target=t3,
            )
            should_swap = bool(delta_m > float(min_delta_m))
        if should_swap:
            updated.extend(
                (_copy_point_with_md(t3, t1.md), _copy_point_with_md(t1, t3.md))
            )
        else:
            updated.extend((t1, t3))
    return tuple(updated)


def _copy_point_with_md(point: WelltrackPoint, md: float) -> WelltrackPoint:
    return WelltrackPoint(
        x=float(point.x),
        y=float(point.y),
        z=float(point.z),
        md=float(md),
    )


def _offset_xy(*, surface: WelltrackPoint | Point3D, target: WelltrackPoint) -> float:
    dx = float(target.x - surface.x)
    dy = float(target.y - surface.y)
    return float(math.hypot(dx, dy))
