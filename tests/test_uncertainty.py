from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pywp.models import Point3D, TrajectoryConfig
from pywp.planner import TrajectoryPlanner
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    UNCERTAINTY_PRESET_CONSERVATIVE,
    UNCERTAINTY_PRESET_OPTIMISTIC,
    planning_uncertainty_model_for_preset,
    PlanningUncertaintyModel,
    build_uncertainty_overlay,
    build_uncertainty_tube_mesh,
    local_uncertainty_axes_xyz,
    normalize_uncertainty_preset,
    station_uncertainty_axes_m,
    station_uncertainty_covariance_xyz,
    uncertainty_preset_label,
    uncertainty_ribbon_polygon,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD_m": [0.0, 120.0, 260.0, 520.0, 900.0],
            "INC_deg": [0.0, 18.0, 42.0, 74.0, 86.0],
            "AZI_deg": [0.0, 35.0, 55.0, 75.0, 90.0],
            "X_m": [0.0, 20.0, 80.0, 240.0, 610.0],
            "Y_m": [0.0, 28.0, 92.0, 280.0, 620.0],
            "Z_m": [0.0, 110.0, 230.0, 430.0, 520.0],
        }
    )


def test_local_uncertainty_axes_are_orthonormal() -> None:
    tangent, inc_axis, azi_axis = local_uncertainty_axes_xyz(inc_deg=45.0, azi_deg=90.0)

    assert np.isclose(np.linalg.norm(tangent), 1.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(inc_axis), 1.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(azi_axis), 1.0, atol=1e-9)
    assert np.isclose(np.dot(tangent, inc_axis), 0.0, atol=1e-9)
    assert np.isclose(np.dot(tangent, azi_axis), 0.0, atol=1e-9)
    assert np.isclose(np.dot(inc_axis, azi_axis), 0.0, atol=1e-9)


def test_station_uncertainty_covariance_is_positive_semidefinite() -> None:
    cov = station_uncertainty_covariance_xyz(
        md_m=1800.0,
        inc_deg=78.0,
        azi_deg=125.0,
    )
    eigenvalues = np.linalg.eigvalsh(cov)

    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T, atol=1e-12)
    assert float(np.min(eigenvalues)) >= -1e-9


def test_azimuth_uncertainty_axis_vanishes_for_vertical_station() -> None:
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=1200.0,
        inc_deg=0.0,
    )

    assert semi_inc_m > 0.0
    assert semi_azi_m == semi_inc_m


def test_build_uncertainty_overlay_returns_projected_rings() -> None:
    model = PlanningUncertaintyModel(sample_step_m=200.0, max_display_ellipses=8)
    overlay = build_uncertainty_overlay(
        stations=_sample_df(),
        surface=Point3D(0.0, 0.0, 0.0),
        azimuth_deg=90.0,
        model=model,
    )

    assert overlay.samples
    md_values = [sample.md_m for sample in overlay.samples]
    assert md_values == sorted(md_values)
    assert md_values[0] > 0.0

    sample = overlay.samples[-1]
    assert sample.ring_xyz.shape[1] == 3
    assert sample.ring_plan_xy.shape[1] == 2
    assert sample.ring_section_xz.shape[1] == 2
    assert sample.semi_axis_inc_m > 0.0
    assert sample.semi_axis_azi_m > 0.0
    assert len(sample.center_plan_xy) == 2
    assert len(sample.center_section_xz) == 2


def test_overlay_uses_dense_interpolated_md_samples_and_preserves_required_markers() -> None:
    model = PlanningUncertaintyModel(sample_step_m=100.0, max_display_ellipses=16)
    overlay = build_uncertainty_overlay(
        stations=_sample_df(),
        surface=Point3D(0.0, 0.0, 0.0),
        azimuth_deg=90.0,
        model=model,
        required_md_m=(135.0, 350.0, 900.0),
    )

    sample_md_values = [sample.md_m for sample in overlay.samples]

    assert len(sample_md_values) > len(_sample_df()) - 1
    assert any(np.isclose(md_value, 135.0, atol=1e-6) for md_value in sample_md_values)
    assert any(np.isclose(md_value, 350.0, atol=1e-6) for md_value in sample_md_values)
    required_sample = next(
        sample for sample in overlay.samples if np.isclose(sample.md_m, 350.0, atol=1e-6)
    )
    assert required_sample.center_xyz[0] == pytest.approx(135.38461538, abs=1e-6)
    assert required_sample.center_xyz[1] == pytest.approx(157.07692308, abs=1e-6)
    assert required_sample.center_xyz[2] == pytest.approx(299.23076923, abs=1e-6)


def test_default_mwd_proxy_radii_are_in_engineering_range_for_5_to_6km_md() -> None:
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=5500.0,
        inc_deg=86.0,
    )

    assert 130.0 <= semi_inc_m <= 170.0
    assert 150.0 <= semi_azi_m <= 190.0


def test_vertical_uncertainty_before_kop_is_not_overinflated_by_lateral_drift() -> None:
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=550.0,
        inc_deg=0.0,
    )

    assert 5.0 <= semi_inc_m <= 7.0
    assert semi_azi_m == pytest.approx(semi_inc_m, abs=1e-9)


def test_lateral_drift_contribution_grows_with_inclination() -> None:
    vertical_inc_m, vertical_azi_m = station_uncertainty_axes_m(
        md_m=2000.0,
        inc_deg=0.0,
    )
    build_inc_m, build_azi_m = station_uncertainty_axes_m(
        md_m=2000.0,
        inc_deg=60.0,
    )

    assert build_inc_m > vertical_inc_m
    assert build_azi_m > vertical_azi_m


def test_uncertainty_presets_are_normalized_and_monotonic() -> None:
    assert normalize_uncertainty_preset("unknown") == DEFAULT_UNCERTAINTY_PRESET
    assert uncertainty_preset_label(UNCERTAINTY_PRESET_OPTIMISTIC) == "Оптимистичный"

    optimistic_model = planning_uncertainty_model_for_preset(UNCERTAINTY_PRESET_OPTIMISTIC)
    conservative_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_CONSERVATIVE
    )

    optimistic_inc_m, optimistic_azi_m = station_uncertainty_axes_m(
        md_m=4000.0,
        inc_deg=86.0,
        model=optimistic_model,
    )
    conservative_inc_m, conservative_azi_m = station_uncertainty_axes_m(
        md_m=4000.0,
        inc_deg=86.0,
        model=conservative_model,
    )

    assert conservative_inc_m > optimistic_inc_m
    assert conservative_azi_m > optimistic_azi_m


def test_uncertainty_ribbon_and_tube_mesh_are_continuous() -> None:
    overlay = build_uncertainty_overlay(
        stations=_sample_df(),
        surface=Point3D(0.0, 0.0, 0.0),
        azimuth_deg=90.0,
    )
    ribbon = uncertainty_ribbon_polygon(overlay, projection="plan")
    tube = build_uncertainty_tube_mesh(overlay)

    assert len(ribbon) >= 5
    assert ribbon.shape[1] == 2
    assert tube is not None
    assert tube.vertices_xyz.shape[1] == 3
    assert len(tube.i) == len(tube.j) == len(tube.k)
    assert len(tube.i) > 0


def test_regression_overlay_ring_alignment_avoids_twist_for_build_to_hold_case() -> None:
    result = TrajectoryPlanner().plan(
        Point3D(0.0, 0.0, 0.0),
        Point3D(334.0, 46.0, 3769.0),
        Point3D(1464.0, 649.0, 3806.0),
        TrajectoryConfig(),
    )
    overlay = build_uncertainty_overlay(
        stations=result.stations,
        surface=Point3D(0.0, 0.0, 0.0),
        azimuth_deg=float(result.azimuth_deg),
        required_md_m=(
            float(result.summary["kop_md_m"]),
            float(result.md_t1_m),
            float(result.summary["md_total_m"]),
        ),
    )

    for prev_sample, current_sample in zip(overlay.samples, overlay.samples[1:]):
        previous_ring = prev_sample.ring_xyz[:-1]
        current_ring = current_sample.ring_xyz[:-1]
        current_cost = float(np.mean(np.linalg.norm(current_ring - previous_ring, axis=1)))
        best_cost = current_cost
        for candidate_base in (current_ring, current_ring[::-1]):
            for shift in range(current_ring.shape[0]):
                candidate = np.roll(candidate_base, shift=shift, axis=0)
                candidate_cost = float(
                    np.mean(np.linalg.norm(candidate - previous_ring, axis=1))
                )
                best_cost = min(best_cost, candidate_cost)
        assert current_cost == pytest.approx(best_cost, abs=1e-9)


def test_adaptive_refinement_densifies_curved_build_intervals() -> None:
    result = TrajectoryPlanner().plan(
        Point3D(0.0, 0.0, 0.0),
        Point3D(334.0, 46.0, 3769.0),
        Point3D(1464.0, 649.0, 3806.0),
        TrajectoryConfig(),
    )
    overlay = build_uncertainty_overlay(
        stations=result.stations,
        surface=Point3D(0.0, 0.0, 0.0),
        azimuth_deg=float(result.azimuth_deg),
        model=PlanningUncertaintyModel(
            sample_step_m=200.0,
            min_refined_step_m=50.0,
            directional_refine_threshold_deg=5.0,
            max_display_ellipses=200,
        ),
        required_md_m=(
            float(result.summary["kop_md_m"]),
            float(result.md_t1_m),
            float(result.summary["md_total_m"]),
        ),
    )

    sample_md_values = [sample.md_m for sample in overlay.samples]
    curved_window = [value for value in sample_md_values if 3200.0 <= value <= 4000.0]
    curved_gaps = [right - left for left, right in zip(curved_window, curved_window[1:])]

    assert curved_window
    assert max(curved_gaps) <= 50.0 + 1e-6
