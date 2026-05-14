from __future__ import annotations

import math

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.welltrack_quality import detect_t1_t3_order_issues, swap_t1_t3_for_wells


def _record(
    name: str,
    *,
    surface_xy: tuple[float, float],
    t1_xy: tuple[float, float],
    t3_xy: tuple[float, float],
    t1_md: float = 1000.0,
    t3_md: float = 2000.0,
) -> WelltrackRecord:
    sx, sy = surface_xy
    t1x, t1y = t1_xy
    t3x, t3y = t3_xy
    return WelltrackRecord(
        name=name,
        points=(
            WelltrackPoint(x=sx, y=sy, z=0.0, md=0.0),
            WelltrackPoint(x=t1x, y=t1y, z=2500.0, md=t1_md),
            WelltrackPoint(x=t3x, y=t3y, z=2500.0, md=t3_md),
        ),
    )


def _multi_horizontal_record(name: str = "MH-1") -> WelltrackRecord:
    return WelltrackRecord(
        name=name,
        points=(
            WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0),
            WelltrackPoint(x=456008.0, y=889281.0, z=2339.0, md=2.0),
            WelltrackPoint(x=456601.0, y=889139.0, z=2339.0, md=3.0),
            WelltrackPoint(x=456699.0, y=889116.0, z=2365.0, md=4.0),
            WelltrackPoint(x=457282.0, y=888977.0, z=2365.0, md=5.0),
            WelltrackPoint(x=457379.0, y=888953.0, z=2390.0, md=6.0),
            WelltrackPoint(x=457954.0, y=888817.0, z=2390.0, md=7.0),
        ),
    )


def test_detect_t1_t3_order_issues_by_horizontal_offset() -> None:
    records = [
        _record(
            "BAD-1",
            surface_xy=(0.0, 0.0),
            t1_xy=(1200.0, 0.0),
            t3_xy=(500.0, 0.0),
        ),
        _record(
            "OK-1",
            surface_xy=(0.0, 0.0),
            t1_xy=(500.0, 0.0),
            t3_xy=(1200.0, 0.0),
        ),
    ]
    issues = detect_t1_t3_order_issues(records, min_delta_m=0.1)
    assert [item.well_name for item in issues] == ["BAD-1"]
    assert issues[0].delta_m > 0.0


def test_detect_t1_t3_order_issues_skips_multi_horizontal_records() -> None:
    source = _multi_horizontal_record()

    issues = detect_t1_t3_order_issues([source], min_delta_m=0.1)

    assert issues == []


def test_swap_t1_t3_for_wells_preserves_md_positions() -> None:
    source = _record(
        "BAD-1",
        surface_xy=(100.0, 200.0),
        t1_xy=(1200.0, 0.0),
        t3_xy=(500.0, 0.0),
        t1_md=1400.0,
        t3_md=2800.0,
    )
    updated = swap_t1_t3_for_wells([source], well_names={"BAD-1"})
    fixed = updated[0]
    assert fixed.name == "BAD-1"
    assert math.isclose(float(fixed.points[1].md), 1400.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(fixed.points[2].md), 2800.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(fixed.points[1].x), 500.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(fixed.points[2].x), 1200.0, rel_tol=0.0, abs_tol=1e-9)

    issues_after = detect_t1_t3_order_issues([fixed], min_delta_m=0.1)
    assert issues_after == []


def test_swap_t1_t3_for_wells_leaves_multi_horizontal_records_unchanged() -> None:
    source = _multi_horizontal_record()

    updated = swap_t1_t3_for_wells([source], well_names={str(source.name)})

    assert tuple(updated[0].points) == tuple(source.points)
