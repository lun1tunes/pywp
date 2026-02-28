from __future__ import annotations

import pytest

from pywp.classification import (
    COMPLEXITY_COMPLEX,
    COMPLEXITY_VERY_COMPLEX,
    TRAJECTORY_REVERSE_DIRECTION,
    TRAJECTORY_SAME_DIRECTION,
    classify_trajectory_type,
    classify_well,
    interpolate_limits,
)


def test_trajectory_type_for_gv_1000_is_only_same_direction() -> None:
    assert classify_trajectory_type(gv_m=1000.0, horizontal_offset_t1_m=300.0) == TRAJECTORY_SAME_DIRECTION


def test_trajectory_type_for_gv_2000_uses_reverse_window() -> None:
    assert classify_trajectory_type(gv_m=2000.0, horizontal_offset_t1_m=300.0) == TRAJECTORY_REVERSE_DIRECTION
    assert classify_trajectory_type(gv_m=2000.0, horizontal_offset_t1_m=900.0) == TRAJECTORY_SAME_DIRECTION


def test_interpolated_limits_for_mid_depth() -> None:
    limits = interpolate_limits(gv_m=3300.0)
    assert limits.reverse_min_m == pytest.approx(0.0, abs=1e-6)
    assert limits.reverse_max_m == pytest.approx(700.0, abs=1e-6)
    assert limits.ordinary_offset_max_m == pytest.approx(2150.0, abs=1e-6)
    assert limits.complex_offset_max_m == pytest.approx(3350.0, abs=1e-6)
    assert limits.hold_ordinary_max_deg == pytest.approx(36.0, abs=1e-6)
    assert limits.hold_complex_max_deg == pytest.approx(52.5, abs=1e-6)


def test_complexity_is_max_of_offset_and_hold_criteria() -> None:
    by_offset = classify_well(gv_m=2000.0, horizontal_offset_t1_m=1800.0, hold_inc_deg=20.0)
    assert by_offset.complexity_by_offset == COMPLEXITY_COMPLEX
    assert by_offset.complexity == COMPLEXITY_COMPLEX

    by_hold = classify_well(gv_m=2000.0, horizontal_offset_t1_m=600.0, hold_inc_deg=60.0)
    assert by_hold.complexity_by_hold == COMPLEXITY_VERY_COMPLEX
    assert by_hold.complexity == COMPLEXITY_VERY_COMPLEX
