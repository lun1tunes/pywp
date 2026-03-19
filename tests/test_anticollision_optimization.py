from __future__ import annotations

import pandas as pd
import pytest

from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
    evaluate_stations_anti_collision_clearance,
    sample_profile_stations_in_md_window,
    sample_stations_in_md_window,
)
from pywp.mcm import compute_positions_min_curv
from pywp.models import Point3D
from pywp.planner_types import ProfileParameters
from pywp.planner_validation import _build_trajectory
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL


def _straight_stations(*, y_offset_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 90.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [0.0, 1000.0, 2000.0],
            "Y_m": [y_offset_m, y_offset_m, y_offset_m],
            "Z_m": [0.0, 0.0, 0.0],
        }
    )


def _line_stations(*, x_values: list[float], y_offset_m: float, azimuth_deg: list[float] | None = None) -> pd.DataFrame:
    azi = azimuth_deg if azimuth_deg is not None else [90.0] * len(x_values)
    return pd.DataFrame(
        {
            "MD_m": [float(index * 1000.0) for index in range(len(x_values))],
            "INC_deg": [90.0] * len(x_values),
            "AZI_deg": azi,
            "X_m": x_values,
            "Y_m": [y_offset_m] * len(x_values),
            "Z_m": [0.0] * len(x_values),
        }
    )


def test_clearance_evaluation_improves_for_more_separated_candidate_path() -> None:
    reference_stations = _straight_stations(y_offset_m=0.0)
    close_candidate_stations = _straight_stations(y_offset_m=30.0)
    far_candidate_stations = _straight_stations(y_offset_m=140.0)

    reference_path = build_anti_collision_reference_path(
        well_name="REF",
        stations=reference_stations,
        md_start_m=250.0,
        md_end_m=2000.0,
        sample_step_m=100.0,
        model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=250.0,
        candidate_md_end_m=2000.0,
        sf_target=1.0,
        sample_step_m=100.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(reference_path,),
    )

    close_eval = evaluate_stations_anti_collision_clearance(
        stations=close_candidate_stations,
        context=context,
    )
    far_eval = evaluate_stations_anti_collision_clearance(
        stations=far_candidate_stations,
        context=context,
    )

    assert close_eval.min_separation_factor >= 0.0
    assert far_eval.min_separation_factor > close_eval.min_separation_factor
    assert far_eval.max_overlap_depth_m < close_eval.max_overlap_depth_m


def test_reference_path_sampling_preserves_requested_md_window_bounds() -> None:
    reference_path = build_anti_collision_reference_path(
        well_name="REF",
        stations=_straight_stations(y_offset_m=0.0),
        md_start_m=275.0,
        md_end_m=1825.0,
        sample_step_m=200.0,
        model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    )

    assert reference_path.md_start_m == pytest.approx(275.0)
    assert reference_path.md_end_m == pytest.approx(1825.0)
    assert reference_path.sample_md_m[0] == pytest.approx(275.0)
    assert reference_path.sample_md_m[-1] == pytest.approx(1825.0)


def test_reference_path_sampling_interpolates_azimuth_across_north_without_180_degree_flip() -> None:
    sampled = sample_stations_in_md_window(
        stations=_line_stations(
            x_values=[0.0, 1000.0],
            y_offset_m=0.0,
            azimuth_deg=[359.0, 1.0],
        ),
        md_start_m=0.0,
        md_end_m=1000.0,
        sample_step_m=500.0,
    )
    assert sampled["MD_m"].tolist() == pytest.approx([0.0, 500.0, 1000.0])
    assert float(sampled["AZI_deg"].iloc[1]) == pytest.approx(0.0, abs=1.0)


def test_clearance_detects_skewed_window_crossing_via_continuous_closest_approach() -> None:
    candidate_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "INC_deg": [90.0, 90.0],
            "AZI_deg": [90.0, 90.0],
            "X_m": [0.0, 1000.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 0.0],
        }
    )
    reference_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "INC_deg": [90.0, 90.0],
            "AZI_deg": [180.0, 180.0],
            "X_m": [250.0, 250.0],
            "Y_m": [1000.0, -1000.0],
            "Z_m": [0.0, 0.0],
        }
    )

    reference_path = build_anti_collision_reference_path(
        well_name="REF",
        stations=reference_stations,
        md_start_m=0.0,
        md_end_m=1000.0,
        sample_step_m=250.0,
        model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=0.0,
        candidate_md_end_m=1000.0,
        sf_target=1.0,
        sample_step_m=250.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(reference_path,),
    )

    evaluation = evaluate_stations_anti_collision_clearance(
        stations=candidate_stations,
        context=context,
    )

    assert evaluation.min_separation_factor == pytest.approx(0.0, abs=1e-9)
    assert evaluation.max_overlap_depth_m > 0.0


def test_sample_profile_stations_in_md_window_matches_full_trajectory_reference() -> None:
    candidate = ProfileParameters(
        kop_vertical_m=550.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=90.0,
        build1_length_m=1600.0,
        hold_length_m=300.0,
        build2_length_m=900.0,
        horizontal_length_m=720.0,
        horizontal_adjust_length_m=120.0,
        horizontal_hold_length_m=600.0,
        inc_hold_deg=48.0,
        horizontal_inc_deg=90.0,
        azimuth_hold_deg=22.0,
        azimuth_entry_deg=41.0,
        dls_build1_deg_per_30m=2.5,
        dls_build2_deg_per_30m=1.7,
        horizontal_dls_deg_per_30m=2.0,
    )
    surface = Point3D(10.0, -15.0, 5.0)
    md_start_m = 900.0
    md_end_m = 3150.0
    sample_step_m = 75.0

    sampled = sample_profile_stations_in_md_window(
        candidate=candidate,
        surface=surface,
        md_start_m=md_start_m,
        md_end_m=md_end_m,
        sample_step_m=sample_step_m,
    )

    full_stations = compute_positions_min_curv(
        _build_trajectory(params=candidate).stations(md_step_m=sample_step_m),
        start=surface,
    )
    reference = sample_stations_in_md_window(
        stations=full_stations,
        md_start_m=md_start_m,
        md_end_m=md_end_m,
        sample_step_m=sample_step_m,
    )

    pd.testing.assert_series_equal(sampled["MD_m"], reference["MD_m"], check_names=False)
    for column in ("INC_deg", "AZI_deg"):
        assert sampled[column].tolist() == pytest.approx(reference[column].tolist(), abs=0.5)
    for column in ("X_m", "Y_m", "Z_m"):
        assert sampled[column].tolist() == pytest.approx(reference[column].tolist(), abs=1.0)
