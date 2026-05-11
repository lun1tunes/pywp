from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pywp.eclipse_welltrack import parse_welltrack_text
from pywp.iscwsa_mwd import IscwsaMwdEnvironment
from pywp.mcm import compute_positions_min_curv
from pywp.models import Point3D, TrajectoryConfig
from pywp.planner import TrajectoryPlanner
from pywp.uncertainty import (
    _continuous_extreme_index,
    _open_closed_ring,
    DEFAULT_UNCERTAINTY_PRESET,
    UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC,
    UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC,
    planning_uncertainty_model_for_preset,
    PlanningUncertaintyModel,
    build_uncertainty_overlay,
    build_uncertainty_station_samples,
    build_uncertainty_tube_mesh,
    local_uncertainty_axes_xyz,
    normalize_uncertainty_preset,
    station_uncertainty_axes_m,
    station_uncertainty_covariance_samples_for_stations,
    station_uncertainty_covariance_xyz,
    station_uncertainty_covariance_xyz_for_stations,
    station_uncertainty_covariance_xyz_many,
    uncertainty_model_caption,
    uncertainty_preset_label,
    uncertainty_ribbon_polygon,
)
from pywp.welltrack_batch import WelltrackBatchPlanner


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


def _long_horizontal_stations(*, td_md_m: float = 5000.0) -> pd.DataFrame:
    md_values = np.asarray(
        [0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, td_md_m],
        dtype=float,
    )
    md_values = np.unique(md_values)
    inc_values = np.asarray(
        [
            0.0 if md <= 600.0 else min(90.0, (md - 600.0) / 900.0 * 90.0)
            for md in md_values
        ],
        dtype=float,
    )
    return compute_positions_min_curv(
        pd.DataFrame(
            {
                "MD_m": md_values,
                "INC_deg": inc_values,
                "AZI_deg": np.full_like(md_values, 90.0),
            }
        ),
        Point3D(0.0, 0.0, 0.0),
    )


def _plan_major_diameter_m(
    covariance_xyz: np.ndarray, *, confidence_scale: float
) -> float:
    covariance_xy = np.asarray(covariance_xyz, dtype=float)[:2, :2]
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance_xy + covariance_xy.T))
    semi_major_m = float(confidence_scale) * float(np.sqrt(max(eigenvalues[-1], 0.0)))
    return 2.0 * semi_major_m


def _normal_plane_axis_diameters_m(
    covariance_xyz: np.ndarray,
    *,
    inc_deg: float,
    azi_deg: float,
    confidence_scale: float,
) -> tuple[float, float]:
    _, inc_axis, azi_axis = local_uncertainty_axes_xyz(
        inc_deg=inc_deg,
        azi_deg=azi_deg,
    )
    vertical_like_diameter_m = (
        2.0
        * float(confidence_scale)
        * float(np.sqrt(max(float(inc_axis @ covariance_xyz @ inc_axis), 0.0)))
    )
    lateral_like_diameter_m = (
        2.0
        * float(confidence_scale)
        * float(np.sqrt(max(float(azi_axis @ covariance_xyz @ azi_axis), 0.0)))
    )
    return vertical_like_diameter_m, lateral_like_diameter_m


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


def test_vectorized_station_uncertainty_covariance_matches_scalar_formulation() -> None:
    model = PlanningUncertaintyModel(confidence_scale=1.0)
    md_values = np.array([550.0, 1800.0, 4200.0], dtype=float)
    inc_values = np.array([0.0, 42.0, 86.0], dtype=float)
    azi_values = np.array([5.0, 125.0, 270.0], dtype=float)

    vectorized = station_uncertainty_covariance_xyz_many(
        md_m=md_values,
        inc_deg=inc_values,
        azi_deg=azi_values,
        model=model,
    )
    scalar = np.stack(
        [
            station_uncertainty_covariance_xyz(
                md_m=float(md_m),
                inc_deg=float(inc_deg),
                azi_deg=float(azi_deg),
                model=model,
            )
            for md_m, inc_deg, azi_deg in zip(md_values, inc_values, azi_values)
        ],
        axis=0,
    )

    assert vectorized.shape == (3, 3, 3)
    assert np.allclose(vectorized, scalar, atol=1e-12)


def test_azimuth_uncertainty_axis_vanishes_for_vertical_station() -> None:
    model = PlanningUncertaintyModel(confidence_scale=1.0)
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=1200.0,
        inc_deg=0.0,
        model=model,
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


def test_overlay_uses_dense_interpolated_md_samples_and_preserves_required_markers() -> (
    None
):
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
        sample
        for sample in overlay.samples
        if np.isclose(sample.md_m, 350.0, atol=1e-6)
    )
    assert required_sample.center_xyz[0] == pytest.approx(135.38461538, abs=1e-6)
    assert required_sample.center_xyz[1] == pytest.approx(157.07692308, abs=1e-6)
    assert required_sample.center_xyz[2] == pytest.approx(299.23076923, abs=1e-6)


def test_default_iscwsa_mwd_poor_radii_are_in_engineering_range_for_5_to_6km_md() -> (
    None
):
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=5500.0,
        inc_deg=86.0,
    )

    assert 95.0 <= semi_inc_m <= 125.0
    assert 15.0 <= semi_azi_m <= 35.0


def test_mwd_poor_erd_reference_horizontal_diameter_matches_reference() -> None:
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACK_ERD.INC").read_text(encoding="utf-8")
    )
    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={"well_ERD_01"},
        selected_order=["well_ERD_01"],
        config=TrajectoryConfig(
            max_total_md_postcheck_m=9000.0,
            turn_solver_max_restarts=0,
        ),
    )

    assert {str(row["Скважина"]): str(row["Статус"]) for row in rows} == {
        "well_ERD_01": "OK"
    }
    well = successes[0]
    assert 67.0 <= float(well.summary["hold_inc_deg"]) <= 73.0
    assert float(well.summary["t1_horizontal_offset_m"]) == pytest.approx(5600.0)
    assert float(well.summary["horizontal_length_m"]) == pytest.approx(1000.0, abs=1.0)

    poor_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    unknown_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )
    old_dip_model = replace(
        poor_model,
        iscwsa_environment=IscwsaMwdEnvironment(dip_deg=70.0),
    )
    terminal_md_m = float(well.stations["MD_m"].iloc[-1])
    poor_covariance = station_uncertainty_covariance_samples_for_stations(
        stations=well.stations,
        sample_md_m=np.asarray([terminal_md_m], dtype=float),
        model=poor_model,
    ).covariance_xyz[0]
    unknown_covariance = station_uncertainty_covariance_samples_for_stations(
        stations=well.stations,
        sample_md_m=np.asarray([terminal_md_m], dtype=float),
        model=unknown_model,
    ).covariance_xyz[0]
    old_dip_covariance = station_uncertainty_covariance_samples_for_stations(
        stations=well.stations,
        sample_md_m=np.asarray([terminal_md_m], dtype=float),
        model=old_dip_model,
    ).covariance_xyz[0]
    horizontal_diameter_m = _plan_major_diameter_m(
        poor_covariance,
        confidence_scale=float(poor_model.confidence_scale),
    )
    unknown_horizontal_diameter_m = _plan_major_diameter_m(
        unknown_covariance,
        confidence_scale=float(unknown_model.confidence_scale),
    )
    old_dip_horizontal_diameter_m = _plan_major_diameter_m(
        old_dip_covariance,
        confidence_scale=float(old_dip_model.confidence_scale),
    )

    assert horizontal_diameter_m == pytest.approx(1227.0, abs=2.0)
    assert old_dip_horizontal_diameter_m == pytest.approx(503.0, abs=2.0)
    assert horizontal_diameter_m > old_dip_horizontal_diameter_m * 2.0
    assert 1450.0 <= unknown_horizontal_diameter_m <= 1560.0
    assert unknown_horizontal_diameter_m > horizontal_diameter_m


def test_mwd_poor_erd_reference_normal_plane_axes_match_toolcode() -> None:
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACK_ERD.INC").read_text(encoding="utf-8")
    )
    _, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={"well_ERD_01"},
        selected_order=["well_ERD_01"],
        config=TrajectoryConfig(
            max_total_md_postcheck_m=9000.0,
            turn_solver_max_restarts=0,
        ),
    )
    well = successes[0]
    model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    terminal_md_m = float(well.stations["MD_m"].iloc[-1])
    covariance = station_uncertainty_covariance_samples_for_stations(
        stations=well.stations,
        sample_md_m=np.asarray([terminal_md_m], dtype=float),
        model=model,
    ).covariance_xyz[0]
    vertical_diameter_m, lateral_diameter_m = _normal_plane_axis_diameters_m(
        covariance,
        inc_deg=float(well.stations["INC_deg"].iloc[-1]),
        azi_deg=float(well.stations["AZI_deg"].iloc[-1]),
        confidence_scale=float(model.confidence_scale),
    )

    assert vertical_diameter_m == pytest.approx(100.0, abs=3.0)
    assert lateral_diameter_m == pytest.approx(1227.0, abs=2.0)
    assert vertical_diameter_m / lateral_diameter_m == pytest.approx(0.082, abs=0.01)


def test_vertical_uncertainty_before_kop_is_not_overinflated_by_lateral_drift() -> None:
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=550.0,
        inc_deg=0.0,
    )

    assert 0.75 <= semi_inc_m <= 1.5
    assert 0.75 <= semi_azi_m <= 1.5
    assert semi_azi_m == pytest.approx(semi_inc_m, rel=0.15)


def test_lateral_drift_contribution_grows_with_inclination() -> None:
    model = PlanningUncertaintyModel(confidence_scale=1.0)
    vertical_inc_m, vertical_azi_m = station_uncertainty_axes_m(
        md_m=2000.0,
        inc_deg=0.0,
        model=model,
    )
    build_inc_m, build_azi_m = station_uncertainty_axes_m(
        md_m=2000.0,
        inc_deg=60.0,
        model=model,
    )

    assert build_inc_m > vertical_inc_m
    assert build_azi_m > vertical_azi_m


def test_iscwsa_mwd_poor_covariance_uses_full_station_history_for_md_window() -> None:
    model = planning_uncertainty_model_for_preset(UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC)
    samples = station_uncertainty_covariance_xyz_for_stations(
        stations=_sample_df(),
        sample_md_m=np.asarray([520.0, 900.0], dtype=float),
        model=model,
    )

    assert samples.shape == (2, 3, 3)
    assert float(np.trace(samples[0])) > 0.0
    assert float(np.trace(samples[1])) > float(np.trace(samples[0]))


def test_iscwsa_mwd_covariance_does_not_depend_on_display_sample_density() -> None:
    model = planning_uncertainty_model_for_preset(UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC)
    stations = _sample_df()
    terminal_md_m = float(stations["MD_m"].iloc[-1])

    terminal_only = station_uncertainty_covariance_samples_for_stations(
        stations=stations,
        sample_md_m=np.asarray([terminal_md_m], dtype=float),
        model=model,
    )
    dense_samples = np.asarray(
        [
            0.0,
            80.0,
            160.0,
            240.0,
            320.0,
            400.0,
            480.0,
            560.0,
            640.0,
            720.0,
            800.0,
            terminal_md_m,
        ],
        dtype=float,
    )
    dense = station_uncertainty_covariance_samples_for_stations(
        stations=stations,
        sample_md_m=dense_samples,
        model=model,
    )

    np.testing.assert_allclose(
        dense.covariance_xyz[-1],
        terminal_only.covariance_xyz[0],
        atol=1e-12,
    )
    assert [name for name, _ in dense.global_source_vectors_xyz] == [
        name for name, _ in terminal_only.global_source_vectors_xyz
    ]
    for (_, dense_vectors), (_, terminal_vectors) in zip(
        dense.global_source_vectors_xyz,
        terminal_only.global_source_vectors_xyz,
    ):
        np.testing.assert_allclose(
            dense_vectors[-1],
            terminal_vectors[0],
            atol=1e-12,
        )


def test_iscwsa_mwd_poor_covariance_samples_preserve_global_source_vectors() -> None:
    model = planning_uncertainty_model_for_preset(UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC)
    samples = station_uncertainty_covariance_samples_for_stations(
        stations=_sample_df(),
        sample_md_m=np.asarray([520.0, 900.0], dtype=float),
        model=model,
    )

    global_from_sources = np.zeros_like(samples.covariance_xyz_global)
    for _, source_vectors in samples.global_source_vectors_xyz:
        global_from_sources += np.einsum("ni,nj->nij", source_vectors, source_vectors)

    assert samples.covariance_xyz_random.shape == (2, 3, 3)
    assert samples.covariance_xyz_systematic.shape == (2, 3, 3)
    assert samples.covariance_xyz_global.shape == (2, 3, 3)
    assert {name for name, _ in samples.global_source_vectors_xyz} == {
        "dbhg",
        "decg",
        "dstg",
    }
    assert np.allclose(samples.covariance_xyz_global, global_from_sources, atol=1e-12)


def test_iscwsa_mwd_station_covariance_uses_tvd_from_first_station() -> None:
    model = planning_uncertainty_model_for_preset(UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC)
    shifted_stations = _sample_df().copy()
    shifted_stations["Z_m"] = shifted_stations["Z_m"] + 1250.0
    sample_md = np.asarray([520.0, 900.0], dtype=float)

    baseline = station_uncertainty_covariance_samples_for_stations(
        stations=_sample_df(),
        sample_md_m=sample_md,
        model=model,
    )
    shifted = station_uncertainty_covariance_samples_for_stations(
        stations=shifted_stations,
        sample_md_m=sample_md,
        model=model,
    )

    assert np.allclose(shifted.covariance_xyz, baseline.covariance_xyz, atol=1e-12)
    assert np.allclose(
        shifted.covariance_xyz_global,
        baseline.covariance_xyz_global,
        atol=1e-12,
    )


def test_mwd_poor_long_horizontal_smoke_check_has_elongated_plan_ellipse() -> None:
    stations = _long_horizontal_stations(td_md_m=5000.0)
    poor_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    unknown_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )

    sample_md = np.asarray([float(stations["MD_m"].iloc[-1])], dtype=float)
    poor_samples = station_uncertainty_covariance_samples_for_stations(
        stations=stations,
        sample_md_m=sample_md,
        model=poor_model,
    )
    unknown_samples = station_uncertainty_covariance_samples_for_stations(
        stations=stations,
        sample_md_m=sample_md,
        model=unknown_model,
    )
    poor_major_diameter_m = _plan_major_diameter_m(
        poor_samples.covariance_xyz[0],
        confidence_scale=float(poor_model.confidence_scale),
    )
    unknown_major_diameter_m = _plan_major_diameter_m(
        unknown_samples.covariance_xyz[0],
        confidence_scale=float(unknown_model.confidence_scale),
    )

    assert float(stations["X_m"].iloc[-1]) > 4000.0
    assert 700.0 <= poor_major_diameter_m <= 850.0
    assert unknown_major_diameter_m > poor_major_diameter_m


def test_iscwsa_mwd_poor_default_model_is_not_proxy() -> None:
    model = planning_uncertainty_model_for_preset(UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC)

    assert model.iscwsa_tool_code is not None
    assert "ISCWSA MWD POOR Magnetic" in uncertainty_model_caption(model)


def test_uncertainty_presets_are_normalized_and_monotonic() -> None:
    assert normalize_uncertainty_preset("unknown") == DEFAULT_UNCERTAINTY_PRESET
    assert (
        uncertainty_preset_label(UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC)
        == "MWD POOR magnetic (ISCWSA)"
    )

    poor_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    unknown_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )

    poor_inc_m, poor_azi_m = station_uncertainty_axes_m(
        md_m=4000.0,
        inc_deg=86.0,
        model=poor_model,
    )
    unknown_inc_m, unknown_azi_m = station_uncertainty_axes_m(
        md_m=4000.0,
        inc_deg=86.0,
        model=unknown_model,
    )

    assert unknown_inc_m > poor_inc_m
    assert unknown_azi_m > poor_azi_m


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


def test_regression_overlay_ring_alignment_avoids_twist_for_build_to_hold_case() -> (
    None
):
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
        current_cost = float(
            np.mean(np.linalg.norm(current_ring - previous_ring, axis=1))
        )
        best_cost = current_cost
        for candidate_base in (current_ring, current_ring[::-1]):
            for shift in range(current_ring.shape[0]):
                candidate = np.roll(candidate_base, shift=shift, axis=0)
                candidate_cost = float(
                    np.mean(np.linalg.norm(candidate - previous_ring, axis=1))
                )
                best_cost = min(best_cost, candidate_cost)
        assert current_cost == pytest.approx(best_cost, abs=1e-9)


def test_open_closed_ring_drops_duplicate_endpoint() -> None:
    ring = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    opened = _open_closed_ring(ring)

    assert opened.shape == (3, 2)
    assert np.allclose(opened[0], ring[0])
    assert np.allclose(opened[-1], ring[2])


def test_continuous_extreme_index_prefers_nearby_candidate_on_closed_ring() -> None:
    offsets = np.array([9.8, 10.0, 9.9, -2.0, -3.0, 9.85], dtype=float)

    index = _continuous_extreme_index(
        offsets=offsets,
        maximize=True,
        previous_index=5,
    )

    assert int(index) == 5


def test_uncertainty_ribbon_regression_stays_locally_continuous_on_turn_case() -> None:
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

    ribbon = uncertainty_ribbon_polygon(overlay, projection="plan")
    sample_count = len(overlay.samples)
    positive_side = np.asarray(ribbon[:sample_count], dtype=float)
    negative_side = np.asarray(ribbon[sample_count : sample_count * 2], dtype=float)[
        ::-1
    ]
    centers = np.asarray(
        [sample.center_plan_xy for sample in overlay.samples], dtype=float
    )

    center_step_max = float(np.max(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    positive_step_max = float(
        np.max(np.linalg.norm(np.diff(positive_side, axis=0), axis=1))
    )
    negative_step_max = float(
        np.max(np.linalg.norm(np.diff(negative_side, axis=0), axis=1))
    )

    assert positive_step_max <= center_step_max * 1.25
    assert negative_step_max <= center_step_max * 1.25


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
    curved_gaps = [
        right - left for left, right in zip(curved_window, curved_window[1:])
    ]

    assert curved_window
    assert max(curved_gaps) <= 50.0 + 1e-6


def test_build_uncertainty_overlay_rejects_non_increasing_md() -> None:
    df = _sample_df()
    df.loc[2, "MD_m"] = df.loc[1, "MD_m"]

    with pytest.raises(ValueError, match="strictly increasing"):
        build_uncertainty_overlay(
            stations=df,
            surface=Point3D(0.0, 0.0, 0.0),
            azimuth_deg=90.0,
        )


def test_build_uncertainty_overlay_rejects_out_of_range_inclination() -> None:
    df = _sample_df()
    df.loc[1, "INC_deg"] = 181.0

    with pytest.raises(ValueError, match="within \\[0, 180\\]"):
        build_uncertainty_overlay(
            stations=df,
            surface=Point3D(0.0, 0.0, 0.0),
            azimuth_deg=90.0,
        )


def test_runtime_uncertainty_station_samples_keep_small_radius_sections() -> None:
    samples = build_uncertainty_station_samples(
        stations=_sample_df(),
        model=PlanningUncertaintyModel(
            sample_step_m=120.0,
            max_display_ellipses=12,
            min_display_radius_m=1000.0,
        ),
    )

    assert samples
    assert samples[0].md_m == pytest.approx(0.0, abs=1e-9)
