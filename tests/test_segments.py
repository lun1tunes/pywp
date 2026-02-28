from __future__ import annotations

import numpy as np
import pytest

from pywp.mcm import add_dls, compute_positions_min_curv, dogleg_angle_rad
from pywp.models import Point3D
from pywp.segments import BuildSegment


def test_build_segment_length_uses_spatial_dogleg() -> None:
    segment = BuildSegment(
        inc_from_deg=40.0,
        inc_to_deg=40.0,
        dls_deg_per_30m=3.0,
        azi_deg=20.0,
        azi_to_deg=50.0,
        name="BUILD2",
    )

    dogleg_deg = float(dogleg_angle_rad(40.0, 20.0, 40.0, 50.0) * (180.0 / np.pi))
    expected_length_m = dogleg_deg * 30.0 / 3.0

    assert segment.length_m == pytest.approx(expected_length_m, rel=1e-9, abs=1e-9)


def test_build_segment_generates_azimuth_turn_and_target_dls() -> None:
    segment = BuildSegment(
        inc_from_deg=25.0,
        inc_to_deg=70.0,
        dls_deg_per_30m=4.0,
        azi_deg=15.0,
        azi_to_deg=55.0,
        name="BUILD2",
    )

    stations = segment.generate(md_start=0.0, md_step_m=1.0)
    stations = add_dls(compute_positions_min_curv(stations=stations, start=Point3D(0.0, 0.0, 0.0)))

    assert stations["INC_deg"].iloc[0] == pytest.approx(25.0, abs=1e-6)
    assert stations["INC_deg"].iloc[-1] == pytest.approx(70.0, abs=1e-6)
    assert stations["AZI_deg"].iloc[0] == pytest.approx(15.0, abs=1e-6)
    assert stations["AZI_deg"].iloc[-1] == pytest.approx(55.0, abs=1e-6)

    build_dls = stations["DLS_deg_per_30m"].dropna()
    assert len(build_dls) > 10
    assert float(build_dls.mean()) == pytest.approx(4.0, rel=0.03)
