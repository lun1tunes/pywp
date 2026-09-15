from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pywp.mcm import (
    add_dls,
    compute_positions_min_curv,
    dogleg_angle_rad,
    minimum_curvature_increment,
    wrap_azimuth_deg,
)
from pywp.models import Point3D


def test_vertical_interval_has_only_tvd_increment() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0],
            "INC_deg": [0.0, 0.0],
            "AZI_deg": [0.0, 0.0],
            "segment": ["VERTICAL", "VERTICAL"],
        }
    )
    out = compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))

    assert np.isclose(out.loc[1, "X_m"], 0.0, atol=1e-8)
    assert np.isclose(out.loc[1, "Y_m"], 0.0, atol=1e-8)
    assert np.isclose(out.loc[1, "Z_m"], 100.0, atol=1e-8)


def test_constant_tangent_matches_straight_line_projection() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0],
            "INC_deg": [30.0, 30.0],
            "AZI_deg": [90.0, 90.0],
            "segment": ["HOLD", "HOLD"],
        }
    )
    out = compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))

    assert np.isclose(out.loc[1, "X_m"], 50.0, atol=1e-6)
    assert np.isclose(out.loc[1, "Y_m"], 0.0, atol=1e-6)
    assert np.isclose(out.loc[1, "Z_m"], 86.6025403784, atol=1e-6)


def test_dls_is_zero_for_parallel_stations() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 30.0],
            "INC_deg": [45.0, 45.0],
            "AZI_deg": [120.0, 120.0],
            "segment": ["HOLD", "HOLD"],
        }
    )
    out = add_dls(stations)
    assert np.isclose(out.loc[1, "DLS_deg_per_30m"], 0.0, atol=1e-9)


def test_dogleg_angle_zero_for_same_direction() -> None:
    beta = dogleg_angle_rad(10.0, 20.0, 10.0, 20.0)
    assert np.isclose(beta, 0.0, atol=1e-7)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_dogleg_angle_rejects_non_finite_angles(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite INC/AZI"):
        dogleg_angle_rad(10.0, bad_value, 10.0, 20.0)


@pytest.mark.parametrize("inc_deg", [-0.1, 180.1])
def test_dogleg_angle_rejects_physically_invalid_inclination(inc_deg: float) -> None:
    with pytest.raises(ValueError, match=r"INC values in \[0, 180\]"):
        dogleg_angle_rad(inc_deg, 0.0, 10.0, 20.0)


def test_compute_positions_min_curv_rejects_empty_stations() -> None:
    stations = pd.DataFrame(columns=["MD_m", "INC_deg", "AZI_deg"])

    with pytest.raises(ValueError, match="at least one station"):
        compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))


@pytest.mark.parametrize("inc_deg", [-0.1, 180.1])
def test_compute_positions_min_curv_rejects_invalid_inc_even_for_single_station(
    inc_deg: float,
) -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0],
            "INC_deg": [inc_deg],
            "AZI_deg": [0.0],
        }
    )

    with pytest.raises(ValueError, match=r"INC values in \[0, 180\]"):
        compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))


def test_compute_positions_min_curv_is_deterministic_and_does_not_mutate_input() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 30.0, 75.0],
            "INC_deg": [0.0, 35.0, 82.0],
            "AZI_deg": [-10.0, 370.0, 45.0],
            "segment": ["VERTICAL", "BUILD", "HOLD"],
        }
    )
    before = stations.copy(deep=True)

    first = compute_positions_min_curv(stations, start=Point3D(10.0, 20.0, 30.0))
    second = compute_positions_min_curv(stations, start=Point3D(10.0, 20.0, 30.0))

    pd.testing.assert_frame_equal(stations, before)
    pd.testing.assert_frame_equal(first, second)


def test_add_dls_returns_nan_for_zero_md_interval() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [100.0, 100.0],
            "INC_deg": [10.0, 12.0],
            "AZI_deg": [20.0, 20.0],
            "segment": ["HOLD", "HOLD"],
        }
    )
    out = add_dls(stations)
    assert np.isnan(out.loc[1, "DLS_deg_per_30m"])


def test_add_dls_rejects_non_finite_md() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, float("inf")],
            "INC_deg": [0.0, 0.0],
            "AZI_deg": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="finite MD values"):
        add_dls(stations)


def test_wrap_azimuth_deg_normalizes_negative_and_large_values() -> None:
    wrapped = wrap_azimuth_deg(np.array([-721.0, -1.0, 0.0, 361.0, 721.0]))
    assert np.allclose(wrapped, np.array([359.0, 359.0, 0.0, 1.0, 1.0]), atol=1e-12)


def test_minimum_curvature_increment_is_invariant_to_360_degree_azimuth_shift() -> None:
    base = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=82.0,
        azi1_deg=359.0,
        md2_m=30.0,
        inc2_deg=86.0,
        azi2_deg=1.0,
    )
    shifted = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=82.0,
        azi1_deg=-1.0,
        md2_m=30.0,
        inc2_deg=86.0,
        azi2_deg=361.0,
    )
    assert np.allclose(base, shifted, atol=1e-10, rtol=0.0)


def test_compute_positions_min_curv_is_wrap_invariant_for_station_series() -> None:
    base_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 30.0, 60.0, 90.0],
            "INC_deg": [0.0, 40.0, 65.0, 87.0],
            "AZI_deg": [359.0, 1.0, 2.0, 3.0],
            "segment": ["BUILD1", "BUILD1", "BUILD2", "HORIZONTAL"],
        }
    )
    shifted_stations = base_stations.copy()
    shifted_stations["AZI_deg"] = np.array([-1.0, 361.0, 362.0, 363.0])

    base = compute_positions_min_curv(base_stations, start=Point3D(0.0, 0.0, 0.0))
    shifted = compute_positions_min_curv(shifted_stations, start=Point3D(0.0, 0.0, 0.0))

    assert np.allclose(base[["X_m", "Y_m", "Z_m"]], shifted[["X_m", "Y_m", "Z_m"]], atol=1e-9, rtol=0.0)


def test_minimum_curvature_increment_remains_stable_for_very_small_dogleg() -> None:
    reference = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=87.0,
        azi1_deg=15.0,
        md2_m=30.0,
        inc2_deg=87.0,
        azi2_deg=15.0,
    )
    tiny_turn = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=87.0,
        azi1_deg=15.0,
        md2_m=30.0,
        inc2_deg=87.000001,
        azi2_deg=15.000001,
    )

    assert np.all(np.isfinite(tiny_turn))
    assert np.allclose(tiny_turn, reference, atol=5e-6, rtol=0.0)


def test_minimum_curvature_increment_wrap_invariance_holds_for_randomized_cases() -> None:
    rng = np.random.default_rng(20260314)
    for _ in range(100):
        inc1_deg, inc2_deg = rng.uniform(0.0, 95.0, size=2)
        azi1_deg, azi2_deg = rng.uniform(-720.0, 720.0, size=2)
        md2_m = float(rng.uniform(1.0, 120.0))
        base = minimum_curvature_increment(
            md1_m=0.0,
            inc1_deg=float(inc1_deg),
            azi1_deg=float(azi1_deg),
            md2_m=md2_m,
            inc2_deg=float(inc2_deg),
            azi2_deg=float(azi2_deg),
        )
        shifted = minimum_curvature_increment(
            md1_m=0.0,
            inc1_deg=float(inc1_deg),
            azi1_deg=float(azi1_deg + 360.0 * rng.integers(-2, 3)),
            md2_m=md2_m,
            inc2_deg=float(inc2_deg),
            azi2_deg=float(azi2_deg + 360.0 * rng.integers(-2, 3)),
        )
        assert np.allclose(base, shifted, atol=1e-9, rtol=0.0)


def test_minimum_curvature_increment_rejects_non_positive_md_interval() -> None:
    with pytest.raises(ValueError, match="md2_m > md1_m"):
        minimum_curvature_increment(
            md1_m=100.0,
            inc1_deg=10.0,
            azi1_deg=20.0,
            md2_m=100.0,
            inc2_deg=12.0,
            azi2_deg=30.0,
        )


def test_minimum_curvature_increment_rejects_non_numeric_md() -> None:
    with pytest.raises(ValueError, match="numeric MD values"):
        minimum_curvature_increment(
            md1_m="bad",  # type: ignore[arg-type]
            inc1_deg=10.0,
            azi1_deg=20.0,
            md2_m=100.0,
            inc2_deg=12.0,
            azi2_deg=30.0,
        )


def test_compute_positions_min_curv_rejects_non_increasing_md() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 100.0],
            "INC_deg": [0.0, 10.0, 20.0],
            "AZI_deg": [0.0, 20.0, 40.0],
        }
    )

    with pytest.raises(ValueError, match="strictly increasing MD"):
        compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))


def test_compute_positions_min_curv_rejects_descending_md_without_reordering() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [100.0, 0.0],
            "INC_deg": [0.0, 0.0],
            "AZI_deg": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="strictly increasing MD"):
        compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))


def test_compute_positions_min_curv_rejects_non_finite_coordinates() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [-1.0e308, 1.0e308],
            "INC_deg": [0.0, 0.0],
            "AZI_deg": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="finite MD values"):
        compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))


def test_minimum_curvature_increment_rejects_near_180_degree_dogleg() -> None:
    with pytest.raises(ValueError, match="too close to 180 degrees"):
        minimum_curvature_increment(
            md1_m=0.0,
            inc1_deg=90.0,
            azi1_deg=0.0,
            md2_m=30.0,
            inc2_deg=90.0,
            azi2_deg=180.0,
        )
