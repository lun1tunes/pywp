from __future__ import annotations

import math
from pathlib import Path

from pywp.eclipse_welltrack import parse_welltrack_text
from pywp.well_pad import (
    PadLayoutPlan,
    apply_pad_layout,
    detect_well_pads,
    ordered_pad_wells,
)


def test_welltracks2_pad_detection_and_layout() -> None:
    text = Path("tests/test_data/WELLTRACKS2.INC").read_text(encoding="utf-8")
    records = parse_welltrack_text(text)
    assert len(records) == 6

    pads = detect_well_pads(records)
    assert len(pads) == 1
    pad = pads[0]
    assert len(pad.wells) == 6

    nds_azimuth_deg = float(pad.auto_nds_azimuth_deg)
    ordered = ordered_pad_wells(pad=pad, nds_azimuth_deg=nds_azimuth_deg)
    assert len({well.name for well in ordered}) == len(ordered)

    plan = PadLayoutPlan(
        pad_id=str(pad.pad_id),
        first_surface_x=float(pad.surface.x),
        first_surface_y=float(pad.surface.y),
        first_surface_z=float(pad.surface.z),
        spacing_m=20.0,
        nds_azimuth_deg=nds_azimuth_deg,
    )
    updated = apply_pad_layout(
        records=records,
        pads=pads,
        plan_by_pad_id={str(pad.pad_id): plan},
    )

    updated_by_name = {record.name: record for record in updated}
    ux = math.sin(math.radians(nds_azimuth_deg))
    uy = math.cos(math.radians(nds_azimuth_deg))
    for slot_index, well in enumerate(ordered):
        surface = updated_by_name[well.name].points[0]
        dx = float(surface.x - plan.first_surface_x)
        dy = float(surface.y - plan.first_surface_y)
        projection_m = dx * ux + dy * uy
        assert math.isclose(
            projection_m,
            float(slot_index) * plan.spacing_m,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        assert math.isclose(
            float(surface.z), plan.first_surface_z, rel_tol=0.0, abs_tol=1e-9
        )

    for before, after in zip(records, updated):
        assert before.name == after.name
        assert math.isclose(
            float(before.points[0].md), float(after.points[0].md), rel_tol=0.0, abs_tol=1e-12
        )
        assert before.points[1:] == after.points[1:]
