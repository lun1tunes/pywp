from __future__ import annotations

import math

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.well_pad import (
    PadLayoutPlan,
    apply_pad_layout,
    detect_well_pads,
    ordered_pad_wells,
)


def _record(
    name: str,
    *,
    sx: float,
    sy: float,
    sz: float,
    t1x: float,
    t1y: float,
    t1z: float,
    t3x: float,
    t3y: float,
    t3z: float,
) -> WelltrackRecord:
    return WelltrackRecord(
        name=name,
        points=(
            WelltrackPoint(x=sx, y=sy, z=sz, md=0.0),
            WelltrackPoint(x=t1x, y=t1y, z=t1z, md=1000.0),
            WelltrackPoint(x=t3x, y=t3y, z=t3z, md=2000.0),
        ),
    )


def test_detect_well_pads_groups_by_surface() -> None:
    records = [
        _record(
            "W1",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=100.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=200.0,
            t3y=0.0,
            t3z=2100.0,
        ),
        _record(
            "W2",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=120.0,
            t1y=10.0,
            t1z=2000.0,
            t3x=220.0,
            t3y=10.0,
            t3z=2100.0,
        ),
        _record(
            "W3",
            sx=500.0,
            sy=0.0,
            sz=0.0,
            t1x=600.0,
            t1y=30.0,
            t1z=2000.0,
            t3x=700.0,
            t3y=30.0,
            t3z=2100.0,
        ),
    ]
    pads = detect_well_pads(records)
    assert len(pads) == 2
    counts = sorted(len(pad.wells) for pad in pads)
    assert counts == [1, 2]


def test_ordered_pad_wells_uses_projection_along_nds() -> None:
    records = [
        _record(
            "W1",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=100.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=200.0,
            t3y=0.0,
            t3z=2100.0,
        ),
        _record(
            "W2",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=300.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=400.0,
            t3y=0.0,
            t3z=2100.0,
        ),
        _record(
            "W3",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=200.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=300.0,
            t3y=0.0,
            t3z=2100.0,
        ),
    ]
    pads = detect_well_pads(records)
    assert len(pads) == 1
    ordered = ordered_pad_wells(pad=pads[0], nds_azimuth_deg=90.0)
    assert [well.name for well in ordered] == ["W1", "W3", "W2"]


def test_apply_pad_layout_rewrites_surface_points() -> None:
    records = [
        _record(
            "W1",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=100.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=200.0,
            t3y=0.0,
            t3z=2100.0,
        ),
        _record(
            "W2",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=300.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=400.0,
            t3y=0.0,
            t3z=2100.0,
        ),
    ]
    pads = detect_well_pads(records)
    assert len(pads) == 1
    plan = PadLayoutPlan(
        pad_id=str(pads[0].pad_id),
        first_surface_x=1000.0,
        first_surface_y=500.0,
        first_surface_z=5.0,
        spacing_m=20.0,
        nds_azimuth_deg=90.0,
    )
    updated = apply_pad_layout(
        records=records,
        pads=pads,
        plan_by_pad_id={str(pads[0].pad_id): plan},
    )
    s1 = updated[0].points[0]
    s2 = updated[1].points[0]
    assert math.isclose(float(s1.x), 1000.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(s1.y), 500.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(s2.x), 1020.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(s2.y), 500.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(s1.z), 5.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(s2.z), 5.0, rel_tol=0.0, abs_tol=1e-9)

