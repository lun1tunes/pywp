from __future__ import annotations

import numpy as np
import pytest

from pywp.mcm import add_dls, compute_positions_min_curv, dogleg_angle_rad
from pywp.models import Point3D
from pywp import models as _models
from pywp.segments import (
    BuildSegment,
    DEFAULT_INTERPOLATION_METHOD,
    HoldSegment,
    VerticalSegment,
    INTERPOLATION_RODRIGUES,
    INTERPOLATION_SLERP,
    _direction_vector,
    _make_md_grid,
    _rodrigues_directions,
    _slerp_directions,
)
from pywp.trajectory import MIN_STATION_MD_INTERVAL_M, WellTrajectory


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


def test_make_md_grid_preserves_large_md_segment_endpoints() -> None:
    md_start = 3998.684149268526
    md_end = 4008.718149268526
    md = _make_md_grid(
        md_start=md_start,
        length_m=md_end - md_start,
        md_step_m=10.0,
    )

    assert np.isclose(md[-2], md_end)
    assert md[-2] != pytest.approx(md_end, abs=1e-9)
    assert md[-1] == pytest.approx(md_end, abs=1e-12)


def test_welltrajectory_skips_submillimeter_segment_boundary_station() -> None:
    trajectory = WellTrajectory(
        [
            BuildSegment(
                inc_from_deg=0.0,
                inc_to_deg=30.0,
                dls_deg_per_30m=3.0,
                azi_deg=0.0,
                name="BUILD1",
            ),
            HoldSegment(
                length_m=MIN_STATION_MD_INTERVAL_M * 0.5,
                inc_deg=30.0,
                azi_deg=0.0,
                name="HOLD",
            ),
            BuildSegment(
                inc_from_deg=30.0,
                inc_to_deg=60.0,
                dls_deg_per_30m=3.0,
                azi_deg=0.0,
                name="BUILD2",
            ),
        ]
    )

    stations = trajectory.stations(md_step_m=10.0)
    md_values = stations["MD_m"].to_numpy(dtype=float)

    assert "HOLD" not in set(stations["segment"])
    assert np.min(np.diff(md_values)) > MIN_STATION_MD_INTERVAL_M
    compute_positions_min_curv(stations=stations, start=Point3D(0.0, 0.0, 0.0))


def test_welltrajectory_preserves_terminal_md_when_final_segment_is_submillimeter() -> None:
    trajectory = WellTrajectory(
        [
            BuildSegment(
                inc_from_deg=0.0,
                inc_to_deg=30.0,
                dls_deg_per_30m=3.0,
                azi_deg=0.0,
                name="BUILD1",
            ),
            HoldSegment(
                length_m=MIN_STATION_MD_INTERVAL_M * 0.5,
                inc_deg=30.0,
                azi_deg=0.0,
                name="HOLD",
            ),
        ]
    )
    expected_terminal_md = sum(segment.length_m for segment in trajectory.segments)

    stations = trajectory.stations(md_step_m=10.0)
    md_values = stations["MD_m"].to_numpy(dtype=float)

    assert "HOLD" not in set(stations["segment"])
    assert float(md_values[-1]) == pytest.approx(expected_terminal_md, abs=1e-12)
    assert np.min(np.diff(md_values)) > MIN_STATION_MD_INTERVAL_M


def test_welltrajectory_collapses_near_duplicate_station_inside_segment() -> None:
    trajectory = WellTrajectory(
        [
            VerticalSegment(length_m=400.000000134, azi_deg=0.0),
            BuildSegment(
                inc_from_deg=0.0,
                inc_to_deg=10.0,
                dls_deg_per_30m=3.0,
                azi_deg=0.0,
                name="BUILD1",
            ),
        ]
    )

    stations = trajectory.stations(md_step_m=10.0)
    md_values = stations["MD_m"].to_numpy(dtype=float)

    assert float(md_values[40]) == pytest.approx(400.000000134, abs=1e-12)
    assert np.min(np.diff(md_values)) > MIN_STATION_MD_INTERVAL_M
    compute_positions_min_curv(stations=stations, start=Point3D(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Rodrigues vs SLERP equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inc_from,azi_from,inc_to,azi_to",
    [
        (0.1, 0.0, 60.0, 0.0),       # simple inclination build
        (25.0, 15.0, 70.0, 55.0),     # combined inc + azi turn
        (40.0, 20.0, 40.0, 50.0),     # pure azimuth turn
        (10.0, 350.0, 80.0, 20.0),    # azimuth wrap around north
        (85.0, 120.0, 86.0, 121.0),   # small dogleg
    ],
)
def test_rodrigues_matches_slerp(inc_from: float, azi_from: float, inc_to: float, azi_to: float) -> None:
    """Rodrigues and SLERP must produce identical directions for normal doglegs."""
    d_from = _direction_vector(inc_from, azi_from)
    d_to = _direction_vector(inc_to, azi_to)
    t = np.linspace(0.0, 1.0, 50)

    rod = _rodrigues_directions(d_from, d_to, t)
    slp = _slerp_directions(d_from, d_to, t)

    np.testing.assert_allclose(rod, slp, atol=1e-10)


def test_rodrigues_matches_slerp_build_segment_stations() -> None:
    """Full MCM pipeline must give identical positions for both methods."""
    kwargs = dict(
        inc_from_deg=25.0, inc_to_deg=70.0, dls_deg_per_30m=4.0,
        azi_deg=15.0, azi_to_deg=55.0, name="BUILD2",
    )
    seg_rod = BuildSegment(**kwargs, interpolation_method=INTERPOLATION_RODRIGUES)
    seg_slp = BuildSegment(**kwargs, interpolation_method=INTERPOLATION_SLERP)
    origin = Point3D(0.0, 0.0, 0.0)

    st_rod = compute_positions_min_curv(seg_rod.generate(0.0, 1.0), start=origin)
    st_slp = compute_positions_min_curv(seg_slp.generate(0.0, 1.0), start=origin)

    np.testing.assert_allclose(
        st_rod[["N_m", "E_m", "TVD_m"]].to_numpy(),
        st_slp[["N_m", "E_m", "TVD_m"]].to_numpy(),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Rodrigues near-pi stability
# ---------------------------------------------------------------------------


def test_rodrigues_near_pi_dogleg_is_stable() -> None:
    """Rodrigues must not produce NaN/inf for dogleg close to 180 degrees.

    SLERP divides by sin(theta) which blows up near pi; Rodrigues does not.
    """
    d_from = _direction_vector(inc_deg=1.0, azi_deg=0.0)
    d_to = _direction_vector(inc_deg=179.0, azi_deg=0.0)  # ~178 deg dogleg
    t = np.linspace(0.0, 1.0, 100)

    result = _rodrigues_directions(d_from, d_to, t)

    assert np.all(np.isfinite(result)), "Rodrigues produced non-finite values near pi"
    # All vectors must be unit length.
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)
    # Endpoints must match.
    np.testing.assert_allclose(result[0], d_from / np.linalg.norm(d_from), atol=1e-10)
    np.testing.assert_allclose(result[-1], d_to / np.linalg.norm(d_to), atol=1e-10)


def test_rodrigues_zero_dogleg() -> None:
    """When start == end, all interpolated directions must equal start."""
    d = _direction_vector(inc_deg=45.0, azi_deg=120.0)
    t = np.linspace(0.0, 1.0, 20)
    result = _rodrigues_directions(d, d, t)
    for row in result:
        np.testing.assert_allclose(row, d / np.linalg.norm(d), atol=1e-12)


# ---------------------------------------------------------------------------
# Default method and parameter dispatch
# ---------------------------------------------------------------------------


def test_default_interpolation_method_is_rodrigues() -> None:
    assert DEFAULT_INTERPOLATION_METHOD == INTERPOLATION_RODRIGUES


def test_build_segment_default_uses_rodrigues() -> None:
    seg = BuildSegment(inc_from_deg=10.0, inc_to_deg=60.0, dls_deg_per_30m=3.0, azi_deg=0.0)
    assert seg.interpolation_method == INTERPOLATION_RODRIGUES


def test_build_segment_slerp_override() -> None:
    seg = BuildSegment(
        inc_from_deg=10.0, inc_to_deg=60.0, dls_deg_per_30m=3.0,
        azi_deg=0.0, interpolation_method=INTERPOLATION_SLERP,
    )
    assert seg.interpolation_method == INTERPOLATION_SLERP


def test_build_segment_rodrigues_constant_dls() -> None:
    """Rodrigues-interpolated BUILD must yield constant DLS (same as SLERP test above)."""
    seg = BuildSegment(
        inc_from_deg=25.0, inc_to_deg=70.0, dls_deg_per_30m=4.0,
        azi_deg=15.0, azi_to_deg=55.0, name="BUILD2",
        interpolation_method=INTERPOLATION_RODRIGUES,
    )
    stations = seg.generate(md_start=0.0, md_step_m=1.0)
    stations = add_dls(compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0)))

    assert stations["INC_deg"].iloc[0] == pytest.approx(25.0, abs=1e-6)
    assert stations["INC_deg"].iloc[-1] == pytest.approx(70.0, abs=1e-6)
    build_dls = stations["DLS_deg_per_30m"].dropna()
    assert len(build_dls) > 10
    assert float(build_dls.mean()) == pytest.approx(4.0, rel=0.03)


def test_interpolation_constants_match_models() -> None:
    """Ensure segments.py and models.py interpolation constants stay in sync."""
    assert INTERPOLATION_SLERP == _models.INTERPOLATION_SLERP
    assert INTERPOLATION_RODRIGUES == _models.INTERPOLATION_RODRIGUES


def test_build_segment_rejects_unknown_interpolation_method() -> None:
    seg = BuildSegment(
        inc_from_deg=0.0, inc_to_deg=30.0, dls_deg_per_30m=3.0,
        azi_deg=90.0, interpolation_method="unknown_method",
    )
    with pytest.raises(ValueError, match="Unknown interpolation_method"):
        seg.generate(md_start=0.0, md_step_m=10.0)
