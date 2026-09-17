"""Numeric regressions found by the functional P0 audit (no UI mocks)."""

import numpy as np
import pandas as pd
import pytest

from pywp.mcm import compute_positions_min_curv
from pywp.models import Point3D
from pywp.segments import BuildSegment, HoldSegment, VerticalSegment, _make_md_grid
from pywp.trajectory import WellTrajectory


@pytest.mark.parametrize("method", ["rodrigues", "slerp"])
def test_positive_subnanometer_build_keeps_both_station_endpoints(method):
    segment = BuildSegment(0, 30, 1e12, 0, interpolation_method=method)
    assert 0 < segment.length_m < 1e-9
    stations = segment.generate(md_start=0, md_step_m=1)
    assert len(stations) == 2
    assert stations.MD_m.to_list() == [0, segment.length_m]
    np.testing.assert_allclose(stations.INC_deg, [0, 30], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "lengths", [[0.0005], [0.0005, 10], [0, 0.0005, 10], [0, 0.0005]]
)
def test_tiny_initial_segments_keep_origin_and_full_displacement(lengths):
    trajectory = WellTrajectory([VerticalSegment(length) for length in lengths])
    stations = trajectory.stations(md_step_m=10)
    assert stations.MD_m.iloc[0] == 0
    assert stations.MD_m.iloc[-1] == pytest.approx(sum(lengths), abs=1e-12)
    assert np.all(np.diff(stations.MD_m) > 0)
    positioned = compute_positions_min_curv(stations, start=Point3D(0, 0, 0))
    assert positioned.Z_m.iloc[-1] == pytest.approx(sum(lengths), rel=0, abs=1e-12)


@pytest.mark.parametrize("step", [1e-3, 5e-4, 1e-4])
def test_requested_submillimeter_sampling_does_not_collapse_to_one_station(step):
    trajectory = WellTrajectory([HoldSegment(0.01, 60, 90)])
    stations = trajectory.stations(md_step_m=step)
    expected_md = _make_md_grid(0, 0.01, step)
    np.testing.assert_array_equal(stations.MD_m, expected_md)
    positioned = compute_positions_min_curv(stations, start=Point3D(0, 0, 0))
    assert positioned.X_m.iloc[-1] == pytest.approx(0.01 * np.sin(np.pi / 3), abs=1e-12)
    assert positioned.Z_m.iloc[-1] == pytest.approx(0.005, abs=1e-12)


def test_station_generation_is_deterministic_and_does_not_modify_segments():
    segments = [VerticalSegment(0.0005), HoldSegment(0.01, 60, 90)]
    before = [dict(vars(segment)) for segment in segments]
    trajectory = WellTrajectory(segments)
    first = trajectory.stations(md_step_m=0.0005)
    second = trajectory.stations(md_step_m=0.0005)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert [dict(vars(segment)) for segment in segments] == before


@pytest.mark.parametrize("md_start", [-1.0, -1e-12])
def test_segment_grid_rejects_negative_absolute_md(md_start):
    with pytest.raises(ValueError, match="md_start must be non-negative"):
        _make_md_grid(md_start, 10, 1)


def test_zero_length_build_rejects_unknown_interpolation():
    segment = BuildSegment(0, 0, 0, 0, interpolation_method="invalid")
    with pytest.raises(ValueError, match="Unknown interpolation_method"):
        segment.generate(md_start=0, md_step_m=10)


@pytest.mark.parametrize("seed", range(6))
def test_random_small_segment_chains_preserve_total_md_and_origin(seed):
    rng = np.random.default_rng(seed)
    lengths = rng.uniform(1e-5, 8e-4, size=16)
    trajectory = WellTrajectory([VerticalSegment(length) for length in lengths])
    stations = trajectory.stations(md_step_m=0.005)
    assert stations.MD_m.iloc[0] == 0
    assert np.all(np.diff(stations.MD_m) > 0)
    expected = sum(float(length) for length in lengths)
    assert stations.MD_m.iloc[-1] == pytest.approx(expected, rel=0, abs=1e-15)
    positioned = compute_positions_min_curv(stations, start=Point3D(0, 0, 0))
    assert positioned.Z_m.iloc[-1] == pytest.approx(expected, rel=0, abs=1e-15)
