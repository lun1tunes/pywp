from __future__ import annotations

import math

from pydantic import BaseModel

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.well_pad import (
    PAD_SURFACE_ANCHOR_CENTER,
    PadLayoutPlan,
    apply_pad_layout,
    detect_well_pads,
    estimate_pad_nds_azimuth_deg,
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


def test_detect_well_pads_uses_all_multi_horizontal_targets_for_midpoint() -> None:
    records = [
        WelltrackRecord(
            name="MULTI",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
                WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=2.0),
                WelltrackPoint(x=700.0, y=0.0, z=1040.0, md=3.0),
                WelltrackPoint(x=1100.0, y=0.0, z=1040.0, md=4.0),
            ),
        )
    ]

    pad = detect_well_pads(records)[0]
    well = pad.wells[0]

    assert well.midpoint_x == 600.0
    assert well.midpoint_y == 0.0
    assert well.midpoint_z == 1020.0


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


def test_ordered_pad_wells_honors_fixed_slots_and_fills_remaining_by_projection() -> None:
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

    ordered = ordered_pad_wells(
        pad=pads[0],
        nds_azimuth_deg=90.0,
        fixed_slots=((1, "W2"), (3, "W1")),
    )

    assert [well.name for well in ordered] == ["W2", "W3", "W1"]


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


def test_apply_pad_layout_uses_fixed_slots_for_surface_positions() -> None:
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
    plan = PadLayoutPlan(
        pad_id=str(pads[0].pad_id),
        first_surface_x=1000.0,
        first_surface_y=500.0,
        first_surface_z=5.0,
        spacing_m=20.0,
        nds_azimuth_deg=90.0,
        fixed_slots=((1, "W2"), (2, "W1")),
    )

    updated = apply_pad_layout(
        records=records,
        pads=pads,
        plan_by_pad_id={str(pads[0].pad_id): plan},
    )
    updated_by_name = {str(record.name): record for record in updated}

    assert math.isclose(
        float(updated_by_name["W2"].points[0].x),
        1000.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        float(updated_by_name["W1"].points[0].x),
        1020.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        float(updated_by_name["W3"].points[0].x),
        1040.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_apply_pad_layout_can_center_surfaces_about_anchor() -> None:
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
            t1x=200.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=300.0,
            t3y=0.0,
            t3z=2100.0,
        ),
        _record(
            "W3",
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
    ordered = ordered_pad_wells(pad=pads[0], nds_azimuth_deg=90.0)
    plan = PadLayoutPlan(
        pad_id=str(pads[0].pad_id),
        first_surface_x=1000.0,
        first_surface_y=500.0,
        first_surface_z=5.0,
        spacing_m=20.0,
        nds_azimuth_deg=90.0,
        surface_anchor_mode=PAD_SURFACE_ANCHOR_CENTER,
    )

    updated = apply_pad_layout(
        records=records,
        pads=pads,
        plan_by_pad_id={str(pads[0].pad_id): plan},
    )
    updated_by_name = {record.name: record for record in updated}

    center_surface = updated_by_name[ordered[1].name].points[0]
    left_surface = updated_by_name[ordered[0].name].points[0]
    right_surface = updated_by_name[ordered[2].name].points[0]

    assert math.isclose(float(center_surface.x), 1000.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(center_surface.y), 500.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(left_surface.x), 980.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(right_surface.x), 1020.0, rel_tol=0.0, abs_tol=1e-9)


def test_apply_pad_layout_accepts_model_like_points_from_stale_session_state() -> None:
    class LegacyPoint(BaseModel):
        x: float
        y: float
        z: float
        md: float

    class LegacyRecord(BaseModel):
        name: str
        points: tuple[LegacyPoint, ...]

    records = [
        LegacyRecord(
            name="W1",
            points=(
                LegacyPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                LegacyPoint(x=100.0, y=0.0, z=2000.0, md=1000.0),
                LegacyPoint(x=200.0, y=0.0, z=2100.0, md=2000.0),
            ),
        ),
        LegacyRecord(
            name="W2",
            points=(
                LegacyPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                LegacyPoint(x=300.0, y=0.0, z=2000.0, md=1000.0),
                LegacyPoint(x=400.0, y=0.0, z=2100.0, md=2000.0),
            ),
        ),
    ]
    pads = detect_well_pads(records)
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

    assert isinstance(updated[0], WelltrackRecord)
    assert all(isinstance(point, WelltrackPoint) for point in updated[0].points)
    assert math.isclose(float(updated[1].points[0].x), 1020.0, rel_tol=0.0, abs_tol=1e-9)


def test_estimate_pad_nds_azimuth_deg_uses_stable_fallback_for_isotropic_cloud() -> None:
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
            t1x=-100.0,
            t1y=0.0,
            t1z=2000.0,
            t3x=-200.0,
            t3y=0.0,
            t3z=2100.0,
        ),
        _record(
            "W3",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=0.0,
            t1y=100.0,
            t1z=2000.0,
            t3x=0.0,
            t3y=200.0,
            t3z=2100.0,
        ),
        _record(
            "W4",
            sx=0.0,
            sy=0.0,
            sz=0.0,
            t1x=0.0,
            t1y=-100.0,
            t1z=2000.0,
            t3x=0.0,
            t3y=-200.0,
            t3z=2100.0,
        ),
    ]
    pad = detect_well_pads(records)[0]

    azimuth_deg = estimate_pad_nds_azimuth_deg(
        wells=pad.wells,
        surface_x=0.0,
        surface_y=0.0,
    )

    assert math.isclose(float(azimuth_deg), 0.0, rel_tol=0.0, abs_tol=1e-9)
