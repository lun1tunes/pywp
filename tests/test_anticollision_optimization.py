from __future__ import annotations

import pandas as pd
import pytest

from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
    evaluate_stations_anti_collision_clearance,
    sample_stations_in_md_window,
)
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


def test_clearance_uses_local_progress_alignment_instead_of_full_cross_product() -> None:
    candidate_stations = _line_stations(x_values=[0.0, 1000.0], y_offset_m=0.0)
    reference_stations = _line_stations(x_values=[1000.0, 2000.0], y_offset_m=0.0)

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

    assert evaluation.min_separation_factor > 1.0
    assert evaluation.max_overlap_depth_m == pytest.approx(0.0, abs=1e-9)
