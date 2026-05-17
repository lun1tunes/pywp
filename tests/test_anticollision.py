from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pywp.anticollision as anticollision_module
import pywp.anticollision_rerun as anticollision_rerun_module
from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionCorridor,
    AntiCollisionSample,
    AntiCollisionWell,
    TARGET_T1,
    TARGET_T3,
    anti_collision_report_events,
    anti_collision_report_rows,
    analyze_anti_collision,
    analyze_anti_collision_incremental,
    build_anti_collision_well,
    collision_corridor_plan_polygon,
    collision_corridor_tube_mesh,
    collision_zone_plan_polygon,
    collision_zone_sphere_mesh,
)
from pywp.anticollision_rerun import (
    build_anti_collision_analysis_for_successes,
    build_anti_collision_wells_for_successes,
)
from pywp.anticollision_recommendations import (
    MANEUVER_BUILD2_ENTRY,
    MANEUVER_POSTENTRY_TURN,
    RECOMMENDATION_REDUCE_KOP,
    RECOMMENDATION_TARGET_SPACING,
    RECOMMENDATION_TRAJECTORY_REVIEW,
    build_anti_collision_recommendation_clusters,
    anti_collision_recommendation_rows,
    build_anti_collision_recommendations,
    AntiCollisionWellContext,
)
from pywp.eclipse_welltrack import parse_welltrack_text
from pywp.models import (
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    Point3D,
    TrajectoryConfig,
)
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    parse_reference_trajectory_dev_directories,
    parse_reference_trajectory_table,
)
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    PlanningUncertaintyModel,
    UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC,
    UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC,
    planning_uncertainty_model_for_preset,
    station_uncertainty_covariance_samples_for_stations,
)
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner


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


def test_anti_collision_scan_samples_are_decoupled_from_display_ellipses() -> None:
    model = PlanningUncertaintyModel(sample_step_m=250.0, max_display_ellipses=5)

    well = build_anti_collision_well(
        name="WELL-A",
        color="#123456",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        model=model,
        include_display_geometry=True,
        analysis_sample_step_m=10.0,
    )

    sample_md = np.asarray([sample.md_m for sample in well.samples], dtype=float)

    assert len(well.overlay.samples) <= 5
    assert len(well.samples) > len(well.overlay.samples)
    assert float(np.max(np.diff(sample_md))) <= 10.0 + 1e-6


def test_overlap_geometry_uses_dense_scan_samples_not_display_indices() -> None:
    model = PlanningUncertaintyModel(sample_step_m=250.0, max_display_ellipses=4)
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#123456",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        model=model,
        include_display_geometry=True,
        analysis_sample_step_m=10.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#654321",
        stations=_straight_stations(y_offset_m=6.0),
        surface=Point3D(0.0, 6.0, 0.0),
        t1=Point3D(1000.0, 6.0, 0.0),
        t3=Point3D(2000.0, 6.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        model=model,
        include_display_geometry=True,
        analysis_sample_step_m=10.0,
    )

    analysis = analyze_anti_collision([well_a, well_b], build_overlap_geometry=True)

    assert analysis.corridors
    assert any(len(corridor.overlap_rings_xyz) > 0 for corridor in analysis.corridors)


def test_anti_collision_progress_reports_pair_counts() -> None:
    wells = [
        build_anti_collision_well(
            name=name,
            color="#123456",
            stations=_straight_stations(y_offset_m=y_offset_m),
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            include_display_geometry=False,
        )
        for name, y_offset_m in (
            ("WELL-A", 0.0),
            ("WELL-B", 10.0),
            ("WELL-C", 20.0),
        )
    ]
    events: list[anticollision_module.AntiCollisionProgress] = []

    analyze_anti_collision_incremental(
        wells,
        build_overlap_geometry=False,
        progress_callback=events.append,
    )

    assert events
    assert events[0].pair_count == 3
    assert events[0].completed_pair_count == 0
    assert events[-1].pair_count == 3
    assert events[-1].completed_pair_count == 3
    assert events[-1].recalculated_pair_count == 3


def test_parallel_anti_collision_matches_serial_result() -> None:
    model = PlanningUncertaintyModel(sample_step_m=250.0, max_display_ellipses=4)
    wells = [
        build_anti_collision_well(
            name=name,
            color="#123456",
            stations=_straight_stations(y_offset_m=y_offset_m),
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            model=model,
            include_display_geometry=False,
            analysis_sample_step_m=10.0,
        )
        for name, y_offset_m in (
            ("WELL-A", 0.0),
            ("WELL-B", 6.0),
            ("WELL-C", 80.0),
        )
    ]
    parallel_events: list[anticollision_module.AntiCollisionProgress] = []

    serial = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=0,
    )
    parallel = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=2,
        progress_callback=parallel_events.append,
    )

    assert parallel_events[-1].parallel_workers == 2
    assert parallel_events[-1].completed_pair_count == serial.pair_count
    assert parallel.pair_count == serial.pair_count
    assert parallel.overlapping_pair_count == serial.overlapping_pair_count
    assert parallel.target_overlap_pair_count == serial.target_overlap_pair_count
    assert parallel.worst_separation_factor == pytest.approx(
        serial.worst_separation_factor
    )
    assert anti_collision_report_rows(parallel) == anti_collision_report_rows(serial)


@pytest.mark.integration
def test_parallel_anti_collision_runs_under_spawn_context(monkeypatch) -> None:
    ctx = multiprocessing.get_context("spawn")
    monkeypatch.setattr(
        anticollision_module,
        "process_pool_context",
        lambda **_kwargs: ctx,
    )
    model = PlanningUncertaintyModel(sample_step_m=250.0, max_display_ellipses=4)
    wells = [
        build_anti_collision_well(
            name=name,
            color="#123456",
            stations=_straight_stations(y_offset_m=y_offset_m),
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            model=model,
            include_display_geometry=False,
            analysis_sample_step_m=10.0,
        )
        for name, y_offset_m in (
            ("WELL-A", 0.0),
            ("WELL-B", 6.0),
            ("WELL-C", 80.0),
        )
    ]

    serial = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=0,
    )
    parallel = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=2,
    )

    assert parallel.pair_count == serial.pair_count
    assert parallel.overlapping_pair_count == serial.overlapping_pair_count
    assert anti_collision_report_rows(parallel) == anti_collision_report_rows(serial)


def test_parallel_anti_collision_falls_back_when_process_pool_breaks(
    monkeypatch,
) -> None:
    model = PlanningUncertaintyModel(sample_step_m=250.0, max_display_ellipses=4)
    wells = [
        build_anti_collision_well(
            name=name,
            color="#123456",
            stations=_straight_stations(y_offset_m=y_offset_m),
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            model=model,
            include_display_geometry=False,
            analysis_sample_step_m=10.0,
        )
        for name, y_offset_m in (
            ("WELL-A", 0.0),
            ("WELL-B", 6.0),
            ("WELL-C", 80.0),
        )
    ]

    class BrokenExecutor:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def submit(self, *args, **kwargs):
            raise anticollision_module.BrokenProcessPool("spawn pool unavailable")

    monkeypatch.setattr(anticollision_module, "ProcessPoolExecutor", BrokenExecutor)

    serial = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=0,
    )
    fallback = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=2,
    )

    assert fallback.pair_count == serial.pair_count
    assert fallback.overlapping_pair_count == serial.overlapping_pair_count
    assert anti_collision_report_rows(fallback) == anti_collision_report_rows(serial)


def test_parallel_anti_collision_falls_back_when_payload_cannot_pickle(
    monkeypatch,
) -> None:
    model = PlanningUncertaintyModel(sample_step_m=250.0, max_display_ellipses=4)
    wells = [
        build_anti_collision_well(
            name=name,
            color="#123456",
            stations=_straight_stations(y_offset_m=y_offset_m),
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            model=model,
            include_display_geometry=False,
            analysis_sample_step_m=10.0,
        )
        for name, y_offset_m in (
            ("WELL-A", 0.0),
            ("WELL-B", 6.0),
            ("WELL-C", 80.0),
        )
    ]

    class PicklingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def submit(self, *args, **kwargs):
            raise anticollision_module.PicklingError(
                "Can't pickle <class 'pywp.models.Point3D'>"
            )

    monkeypatch.setattr(anticollision_module, "ProcessPoolExecutor", PicklingExecutor)

    serial = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=0,
    )
    fallback = analyze_anti_collision(
        wells,
        build_overlap_geometry=False,
        parallel_workers=2,
    )

    assert fallback.pair_count == serial.pair_count
    assert fallback.overlapping_pair_count == serial.overlapping_pair_count
    assert anti_collision_report_rows(fallback) == anti_collision_report_rows(serial)


def test_local_refine_finds_overlap_between_coarse_scan_stations() -> None:
    covariance = np.diag([36.0, 36.0, 36.0])
    zero = np.zeros((3, 3), dtype=float)
    model = PlanningUncertaintyModel(confidence_scale=1.0)

    def sample(md_m: float, center: tuple[float, float, float]) -> AntiCollisionSample:
        return AntiCollisionSample(
            md_m=md_m,
            center_xyz=center,
            covariance_xyz=covariance,
            covariance_xyz_random=covariance,
            covariance_xyz_systematic=zero,
            covariance_xyz_global=zero,
            global_source_vectors_xyz=(),
            inc_deg=90.0,
            azi_deg=90.0,
        )

    well_a = AntiCollisionWell(
        name="WELL-A",
        color="#123456",
        overlay=anticollision_module.WellUncertaintyOverlay(samples=(), model=model),
        samples=(sample(0.0, (0.0, 0.0, 0.0)), sample(10.0, (10.0, 0.0, 0.0))),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 10.0],
                "INC_deg": [90.0, 90.0],
                "AZI_deg": [90.0, 90.0],
                "X_m": [0.0, 10.0],
                "Y_m": [0.0, 0.0],
                "Z_m": [0.0, 0.0],
            }
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=None,
        t3=None,
        md_t1_m=None,
        md_t3_m=None,
    )
    well_b = AntiCollisionWell(
        name="WELL-B",
        color="#654321",
        overlay=anticollision_module.WellUncertaintyOverlay(samples=(), model=model),
        samples=(sample(0.0, (0.0, 20.0, 0.0)), sample(10.0, (10.0, 20.0, 0.0))),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 5.0, 10.0],
                "INC_deg": [90.0, 90.0, 90.0],
                "AZI_deg": [90.0, 90.0, 90.0],
                "X_m": [0.0, 5.0, 10.0],
                "Y_m": [20.0, 0.0, 20.0],
                "Z_m": [0.0, 0.0, 0.0],
            }
        ),
        surface=Point3D(0.0, 20.0, 0.0),
        t1=None,
        t3=None,
        md_t1_m=None,
        md_t3_m=None,
    )

    for build_overlap_geometry in (True, False):
        analysis = analyze_anti_collision(
            [well_a, well_b],
            build_overlap_geometry=build_overlap_geometry,
        )

        assert analysis.corridors
        assert min(
            float(np.min(corridor.separation_factor_values))
            for corridor in analysis.corridors
        ) == pytest.approx(0.0)


def test_local_refine_still_runs_when_a_coarse_overlap_already_exists() -> None:
    covariance = np.diag([36.0, 36.0, 36.0])
    zero = np.zeros((3, 3), dtype=float)
    model = PlanningUncertaintyModel(confidence_scale=1.0)

    def sample(md_m: float, center: tuple[float, float, float]) -> AntiCollisionSample:
        return AntiCollisionSample(
            md_m=md_m,
            center_xyz=center,
            covariance_xyz=covariance,
            covariance_xyz_random=covariance,
            covariance_xyz_systematic=zero,
            covariance_xyz_global=zero,
            global_source_vectors_xyz=(),
            inc_deg=90.0,
            azi_deg=90.0,
        )

    well_a = AntiCollisionWell(
        name="WELL-A",
        color="#123456",
        overlay=anticollision_module.WellUncertaintyOverlay(samples=(), model=model),
        samples=tuple(
            sample(float(md_m), (float(md_m), 0.0, 0.0))
            for md_m in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
        ),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                "INC_deg": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
                "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
                "X_m": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "Z_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=None,
        t3=None,
        md_t1_m=None,
        md_t3_m=None,
    )
    well_b = AntiCollisionWell(
        name="WELL-B",
        color="#654321",
        overlay=anticollision_module.WellUncertaintyOverlay(samples=(), model=model),
        samples=tuple(
            sample(
                float(md_m),
                (
                    float(md_m),
                    0.0 if md_m == 0.0 else 20.0 if md_m >= 40.0 else 60.0,
                    0.0,
                ),
            )
            for md_m in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
        ),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0],
                "INC_deg": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
                "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
                "X_m": [0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0],
                "Y_m": [0.0, 60.0, 60.0, 60.0, 20.0, 0.0, 20.0],
                "Z_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=None,
        t3=None,
        md_t1_m=None,
        md_t3_m=None,
    )

    analysis = analyze_anti_collision(
        [well_a, well_b],
        build_overlap_geometry=False,
    )

    assert analysis.corridors
    assert any(float(corridor.md_a_end_m) >= 45.0 for corridor in analysis.corridors)
    assert min(
        float(np.min(corridor.separation_factor_values))
        for corridor in analysis.corridors
    ) == pytest.approx(0.0)


def test_success_analysis_uses_definitive_scan_step_for_report_geometry() -> None:
    model = PlanningUncertaintyModel(sample_step_m=250.0, max_display_ellipses=5)
    successes = [
        SuccessfulWellPlan(
            name="WELL-A",
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(1000.0, 0.0, 0.0),
            t3=Point3D(2000.0, 0.0, 0.0),
            stations=_straight_stations(y_offset_m=0.0),
            summary={},
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            config=TrajectoryConfig(),
        ),
        SuccessfulWellPlan(
            name="WELL-B",
            surface=Point3D(0.0, 6.0, 0.0),
            t1=Point3D(1000.0, 6.0, 0.0),
            t3=Point3D(2000.0, 6.0, 0.0),
            stations=_straight_stations(y_offset_m=6.0),
            summary={},
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            config=TrajectoryConfig(),
        ),
    ]

    analysis = build_anti_collision_analysis_for_successes(
        successes,
        model=model,
        include_display_geometry=True,
        build_overlap_geometry=True,
    )

    sample_md = np.asarray(
        [sample.md_m for sample in analysis.wells[0].samples], dtype=float
    )

    assert float(np.max(np.diff(sample_md))) <= 10.0 + 1e-6
    assert len(analysis.wells[0].overlay.samples) <= 5


def test_reference_actual_and_approved_wells_use_station_history_iscwsa_ellipses() -> (
    None
):
    model = planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET)
    success = SuccessfulWellPlan(
        name="WELL-P",
        surface=Point3D(0.0, -600.0, 0.0),
        t1=Point3D(200.0, -600.0, 700.0),
        t3=Point3D(1300.0, -600.0, 900.0),
        stations=_straight_stations(y_offset_m=-600.0),
        summary={"kop_md_m": 700.0},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )
    reference_rows = []
    for well_name, well_kind, y_m in (
        ("REF-ACT", "actual", 0.0),
        ("REF-APP", "approved", 100.0),
    ):
        for x_m, z_m, md_m in (
            (0.0, 0.0, 0.0),
            (200.0, 700.0, 760.0),
            (1300.0, 900.0, 1900.0),
        ):
            reference_rows.append(
                {
                    "Wellname": well_name,
                    "Type": well_kind,
                    "X": x_m,
                    "Y": y_m,
                    "Z": z_m,
                    "MD": md_m,
                }
            )
    reference_wells = tuple(parse_reference_trajectory_table(reference_rows))

    analysis = build_anti_collision_analysis_for_successes(
        [success],
        model=model,
        reference_wells=reference_wells,
        include_display_geometry=True,
        build_overlap_geometry=False,
    )
    reference_by_kind = {
        str(well.well_kind): well
        for well in analysis.wells
        if bool(well.is_reference_only)
    }

    assert set(reference_by_kind) == {REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED}
    for imported_reference in reference_wells:
        reference_well = reference_by_kind[str(imported_reference.kind)]
        terminal_md_m = float(imported_reference.stations["MD_m"].iloc[-1])
        terminal_sample = next(
            sample
            for sample in reference_well.samples
            if float(sample.md_m) == pytest.approx(terminal_md_m)
        )
        expected = station_uncertainty_covariance_samples_for_stations(
            stations=imported_reference.stations,
            sample_md_m=np.asarray([terminal_md_m], dtype=float),
            model=model,
        )
        expected_covariance = np.asarray(expected.covariance_xyz[0], dtype=float)

        assert reference_well.overlay.samples
        assert reference_well.overlay.samples[-1].md_m == pytest.approx(terminal_md_m)
        np.testing.assert_allclose(terminal_sample.covariance_xyz, expected_covariance)
        np.testing.assert_allclose(
            reference_well.overlay.samples[-1].covariance_xyz,
            expected_covariance,
        )
        assert float(np.trace(terminal_sample.covariance_xyz)) > 0.0
        assert {
            source_name for source_name, _ in terminal_sample.global_source_vectors_xyz
        } == {"dbhg", "decg", "dstg"}

    np.testing.assert_allclose(
        reference_by_kind[REFERENCE_WELL_ACTUAL].samples[-1].covariance_xyz,
        reference_by_kind[REFERENCE_WELL_APPROVED].samples[-1].covariance_xyz,
    )


def test_reference_actual_well_can_use_selected_unknown_mwd_model() -> None:
    poor_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    unknown_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )
    success = SuccessfulWellPlan(
        name="WELL-P",
        surface=Point3D(0.0, -600.0, 0.0),
        t1=Point3D(200.0, -600.0, 700.0),
        t3=Point3D(1300.0, -600.0, 900.0),
        stations=_straight_stations(y_offset_m=-600.0),
        summary={"kop_md_m": 700.0},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )
    reference_rows = []
    for well_name, well_kind, y_m in (
        ("REF-ACT", "actual", 0.0),
        ("REF-APP", "approved", 100.0),
    ):
        for x_m, z_m, md_m in (
            (0.0, 0.0, 0.0),
            (200.0, 700.0, 760.0),
            (1300.0, 900.0, 1900.0),
        ):
            reference_rows.append(
                {
                    "Wellname": well_name,
                    "Type": well_kind,
                    "X": x_m,
                    "Y": y_m,
                    "Z": z_m,
                    "MD": md_m,
                }
            )
    reference_wells = tuple(parse_reference_trajectory_table(reference_rows))

    analysis = build_anti_collision_analysis_for_successes(
        [success],
        model=poor_model,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name={"REF-ACT": unknown_model},
        include_display_geometry=True,
        build_overlap_geometry=False,
    )
    reference_by_kind = {
        str(well.well_kind): well
        for well in analysis.wells
        if bool(well.is_reference_only)
    }
    actual_reference = reference_by_kind[REFERENCE_WELL_ACTUAL]
    approved_reference = reference_by_kind[REFERENCE_WELL_APPROVED]

    assert (
        actual_reference.overlay.model.iscwsa_tool_code
        == unknown_model.iscwsa_tool_code
    )
    assert (
        approved_reference.overlay.model.iscwsa_tool_code == poor_model.iscwsa_tool_code
    )
    assert float(np.trace(actual_reference.samples[-1].covariance_xyz)) > float(
        np.trace(approved_reference.samples[-1].covariance_xyz)
    )


def test_reference_mwd_assignment_does_not_leak_to_approved_duplicate_name() -> None:
    poor_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    unknown_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )
    success = SuccessfulWellPlan(
        name="WELL-P",
        surface=Point3D(0.0, -600.0, 0.0),
        t1=Point3D(200.0, -600.0, 700.0),
        t3=Point3D(1300.0, -600.0, 900.0),
        stations=_straight_stations(y_offset_m=-600.0),
        summary={"kop_md_m": 700.0},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )
    reference_rows = []
    for well_kind, y_m in (
        ("actual", 0.0),
        ("approved", 100.0),
    ):
        for x_m, z_m, md_m in (
            (0.0, 0.0, 0.0),
            (200.0, 700.0, 760.0),
            (1300.0, 900.0, 1900.0),
        ):
            reference_rows.append(
                {
                    "Wellname": "REF-SAME",
                    "Type": well_kind,
                    "X": x_m,
                    "Y": y_m,
                    "Z": z_m,
                    "MD": md_m,
                }
            )
    reference_wells = tuple(parse_reference_trajectory_table(reference_rows))

    analysis = build_anti_collision_analysis_for_successes(
        [success],
        model=poor_model,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name={"REF-SAME": unknown_model},
        include_display_geometry=True,
        build_overlap_geometry=False,
    )
    reference_by_kind = {
        str(well.well_kind): well
        for well in analysis.wells
        if bool(well.is_reference_only)
    }

    assert (
        reference_by_kind[REFERENCE_WELL_ACTUAL].overlay.model.iscwsa_tool_code
        == unknown_model.iscwsa_tool_code
    )
    assert (
        reference_by_kind[REFERENCE_WELL_APPROVED].overlay.model.iscwsa_tool_code
        == poor_model.iscwsa_tool_code
    )


def test_global_source_vectors_are_correlated_in_relative_clearance() -> None:
    covariance_zero = np.zeros((3, 3), dtype=float)
    sample_a = AntiCollisionSample(
        md_m=100.0,
        center_xyz=(0.0, 0.0, 0.0),
        covariance_xyz=np.diag([25.0, 0.0, 0.0]),
        covariance_xyz_random=covariance_zero,
        covariance_xyz_systematic=covariance_zero,
        covariance_xyz_global=np.diag([25.0, 0.0, 0.0]),
        global_source_vectors_xyz=(("decg", np.asarray([5.0, 0.0, 0.0])),),
    )
    sample_b_same_global = AntiCollisionSample(
        md_m=100.0,
        center_xyz=(10.0, 0.0, 0.0),
        covariance_xyz=np.diag([25.0, 0.0, 0.0]),
        covariance_xyz_random=covariance_zero,
        covariance_xyz_systematic=covariance_zero,
        covariance_xyz_global=np.diag([25.0, 0.0, 0.0]),
        global_source_vectors_xyz=(("decg", np.asarray([5.0, 0.0, 0.0])),),
    )
    sample_b_different_global = AntiCollisionSample(
        md_m=100.0,
        center_xyz=(10.0, 0.0, 0.0),
        covariance_xyz=np.diag([25.0, 0.0, 0.0]),
        covariance_xyz_random=covariance_zero,
        covariance_xyz_systematic=covariance_zero,
        covariance_xyz_global=np.diag([25.0, 0.0, 0.0]),
        global_source_vectors_xyz=(("decg", np.asarray([-5.0, 0.0, 0.0])),),
    )
    direction = np.asarray([[1.0, 0.0, 0.0]], dtype=float)

    same_sigma2 = anticollision_module._directional_global_relative_sigma2_for_pairs(
        samples_a=(sample_a,),
        samples_b=(sample_b_same_global,),
        direction=direction,
    )
    different_sigma2 = (
        anticollision_module._directional_global_relative_sigma2_for_pairs(
            samples_a=(sample_a,),
            samples_b=(sample_b_different_global,),
            direction=direction,
        )
    )

    assert same_sigma2[0] == pytest.approx(0.0)
    assert different_sigma2[0] == pytest.approx(100.0)


def _vertical_build_stations(
    *,
    y_offset_m: float,
    lateral_y_t1_m: float,
    lateral_y_end_m: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "INC_deg": [0.0, 0.0, 0.0, 20.0, 50.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "X_m": [0.0, 0.0, 0.0, 50.0, 320.0],
            "Y_m": [
                y_offset_m,
                y_offset_m,
                y_offset_m,
                lateral_y_t1_m,
                lateral_y_end_m,
            ],
            "Z_m": [0.0, 500.0, 1000.0, 1450.0, 1775.0],
            "segment": ["VERTICAL", "VERTICAL", "VERTICAL", "BUILD1", "BUILD1"],
        }
    )


def test_anti_collision_analysis_prioritizes_target_overlaps() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )

    assert any(sample.target_label == TARGET_T1 for sample in well_a.samples)
    assert any(sample.target_label == TARGET_T3 for sample in well_a.samples)

    analysis = analyze_anti_collision([well_a, well_b])

    assert analysis.pair_count == 1
    assert analysis.overlapping_pair_count == 1
    assert analysis.target_overlap_pair_count == 1
    assert analysis.corridors
    assert analysis.well_segments
    assert analysis.zones
    assert analysis.zones[0].classification == "target-target"
    assert analysis.zones[0].separation_factor < 1.0

    rows = anti_collision_report_rows(analysis)
    assert rows
    assert rows[0]["Приоритет"] == "Цели ↔ цели"
    assert len(analysis.zones) <= 5
    target_mds = {
        (float(zone.md_a_m), float(zone.md_b_m))
        for zone in analysis.zones
        if zone.classification == "target-target"
    }
    assert (1000.0, 1000.0) in target_mds
    assert (2000.0, 2000.0) in target_mds


def test_collision_zone_geometry_helpers_return_closed_plan_polygon_and_sphere_grid() -> (
    None
):
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    zone = analyze_anti_collision([well_a, well_b]).zones[0]

    polygon = collision_zone_plan_polygon(zone)
    sphere_x, sphere_y, sphere_z = collision_zone_sphere_mesh(zone)

    assert polygon.shape[1] == 2
    assert tuple(polygon[0]) == tuple(polygon[-1])
    assert sphere_x.shape == sphere_y.shape == sphere_z.shape
    assert sphere_x.ndim == 2


def test_collision_corridor_geometry_and_well_segments_are_built() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])

    corridor = analysis.corridors[0]
    polygon = collision_corridor_plan_polygon(corridor)
    tube_mesh = collision_corridor_tube_mesh(corridor)

    assert tuple(polygon[0]) == tuple(polygon[-1])
    assert tube_mesh is not None
    assert tube_mesh.vertices_xyz.shape[1] == 3
    assert {segment.well_name for segment in analysis.well_segments} == {
        "WELL-A",
        "WELL-B",
    }


def test_collision_corridor_overlap_mesh_uses_independent_lenses() -> None:
    corridor = AntiCollisionCorridor(
        well_a="WELL-A",
        well_b="WELL-B",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=1000.0,
        md_a_end_m=1030.0,
        md_b_start_m=1000.0,
        md_b_end_m=1030.0,
        md_a_values_m=np.array([1000.0, 1030.0], dtype=float),
        md_b_values_m=np.array([1000.0, 1030.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [8.0, -1.0, 0.0],
                    [12.0, -1.0, 0.0],
                    [12.0, 1.0, 0.0],
                    [8.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        ),
        overlap_core_radius_m=np.array([2.0, 2.0], dtype=float),
        separation_factor_values=np.array([0.6, 0.7], dtype=float),
        overlap_depth_values_m=np.array([4.0, 3.0], dtype=float),
    )

    mesh = collision_corridor_tube_mesh(corridor)

    assert mesh is not None
    assert mesh.vertices_xyz.shape == (10, 3)
    assert len(mesh.i) == 16
    assert set(mesh.i[-8:-4]) == {8}
    assert set(mesh.i[-4:]) == {9}
    side_vertices = np.concatenate([mesh.i[:8], mesh.j[:8], mesh.k[:8]])
    assert np.any(side_vertices < 4)
    assert np.any((side_vertices >= 4) & (side_vertices < 8))
    assert np.all(mesh.j[-8:-4] < 4)
    assert np.all((mesh.j[-4:] >= 4) & (mesh.j[-4:] < 8))


def test_analyze_anti_collision_skips_distant_pair_before_corridor_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=1500.0),
        surface=Point3D(0.0, 1500.0, 0.0),
        t1=Point3D(1000.0, 1500.0, 0.0),
        t3=Point3D(2000.0, 1500.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )

    def _unexpected_pair_scan(**_: object) -> list[AntiCollisionCorridor]:
        raise AssertionError("distant pair should be skipped by XY prefilter")

    monkeypatch.setattr(
        anticollision_module,
        "_pair_overlap_corridors",
        _unexpected_pair_scan,
    )

    analysis = analyze_anti_collision(
        [well_a, well_b],
        build_overlap_geometry=False,
    )

    assert analysis.pair_count == 1
    assert analysis.overlapping_pair_count == 0
    assert not analysis.corridors


def test_analyze_anti_collision_keeps_near_pair_after_xy_prefilter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=25.0),
        surface=Point3D(0.0, 25.0, 0.0),
        t1=Point3D(1000.0, 25.0, 0.0),
        t3=Point3D(2000.0, 25.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )
    seen_pairs: list[tuple[str, str]] = []

    def _record_pair_scan(
        *,
        well_a: object,
        well_b: object,
        build_overlap_geometry: bool,
    ) -> list[AntiCollisionCorridor]:
        assert build_overlap_geometry is False
        seen_pairs.append((str(getattr(well_a, "name")), str(getattr(well_b, "name"))))
        return []

    monkeypatch.setattr(
        anticollision_module,
        "_pair_overlap_corridors",
        _record_pair_scan,
    )

    analysis = analyze_anti_collision(
        [well_a, well_b],
        build_overlap_geometry=False,
    )

    assert analysis.pair_count == 1
    assert seen_pairs == [("WELL-A", "WELL-B")]


def test_analyze_anti_collision_incremental_reuses_unchanged_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wells = [
        build_anti_collision_well(
            name=name,
            color="#0B6E4F",
            stations=_straight_stations(y_offset_m=y_offset_m),
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            include_display_geometry=False,
        )
        for name, y_offset_m in (
            ("WELL-A", 0.0),
            ("WELL-B", 10.0),
            ("WELL-C", 20.0),
        )
    ]
    scanned_pairs: list[tuple[str, str]] = []

    def _record_pair_scan(
        *,
        well_a: AntiCollisionWell,
        well_b: AntiCollisionWell,
        build_overlap_geometry: bool,
    ) -> list[AntiCollisionCorridor]:
        assert build_overlap_geometry is False
        scanned_pairs.append((str(well_a.name), str(well_b.name)))
        return []

    monkeypatch.setattr(
        anticollision_module,
        "_pair_overlap_corridors",
        _record_pair_scan,
    )

    _, pair_cache, first_stats = analyze_anti_collision_incremental(
        wells,
        build_overlap_geometry=False,
        well_signature_by_name={
            "WELL-A": "a-v1",
            "WELL-B": "b-v1",
            "WELL-C": "c-v1",
        },
    )

    assert first_stats.reused_pair_count == 0
    assert first_stats.recalculated_pair_count == 3
    assert scanned_pairs == [
        ("WELL-A", "WELL-B"),
        ("WELL-A", "WELL-C"),
        ("WELL-B", "WELL-C"),
    ]

    scanned_pairs.clear()
    _, _, second_stats = analyze_anti_collision_incremental(
        wells,
        build_overlap_geometry=False,
        well_signature_by_name={
            "WELL-A": "a-v1",
            "WELL-B": "b-v2",
            "WELL-C": "c-v1",
        },
        previous_pair_cache=pair_cache,
    )

    assert second_stats.reused_pair_count == 1
    assert second_stats.recalculated_pair_count == 2
    assert scanned_pairs == [("WELL-A", "WELL-B"), ("WELL-B", "WELL-C")]


def test_incremental_lightweight_scan_reuses_full_geometry_pair_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wells = [
        build_anti_collision_well(
            name=name,
            color="#0B6E4F",
            stations=_straight_stations(y_offset_m=y_offset_m),
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            include_display_geometry=False,
        )
        for name, y_offset_m in (("WELL-A", 0.0), ("WELL-B", 10.0))
    ]
    scanned_flags: list[bool] = []

    def _record_pair_scan(
        *,
        well_a: AntiCollisionWell,
        well_b: AntiCollisionWell,
        build_overlap_geometry: bool,
    ) -> list[AntiCollisionCorridor]:
        scanned_flags.append(bool(build_overlap_geometry))
        return []

    monkeypatch.setattr(
        anticollision_module,
        "_pair_overlap_corridors",
        _record_pair_scan,
    )

    _, pair_cache, first_stats = analyze_anti_collision_incremental(
        wells,
        build_overlap_geometry=True,
        well_signature_by_name={"WELL-A": "a-v1", "WELL-B": "b-v1"},
    )
    _, _, second_stats = analyze_anti_collision_incremental(
        wells,
        build_overlap_geometry=False,
        well_signature_by_name={"WELL-A": "a-v1", "WELL-B": "b-v1"},
        previous_pair_cache=pair_cache,
    )

    assert first_stats.recalculated_pair_count == 1
    assert second_stats.reused_pair_count == 1
    assert second_stats.recalculated_pair_count == 0
    assert scanned_flags == [True]


def test_build_anti_collision_wells_for_successes_reuses_unchanged_wells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successes = [
        SuccessfulWellPlan(
            name=name,
            surface=Point3D(0.0, y_offset_m, 0.0),
            t1=Point3D(1000.0, y_offset_m, 0.0),
            t3=Point3D(2000.0, y_offset_m, 0.0),
            stations=_straight_stations(y_offset_m=y_offset_m),
            summary={"kop_md_m": 0.0},
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            config={"optimization_mode": "none"},
        )
        for name, y_offset_m in (("WELL-A", 0.0), ("WELL-B", 20.0))
    ]
    built_names: list[str] = []
    original_builder = anticollision_rerun_module.build_anti_collision_well

    def _counting_builder(**kwargs: object) -> AntiCollisionWell:
        built_names.append(str(kwargs["name"]))
        return original_builder(**kwargs)

    monkeypatch.setattr(
        anticollision_rerun_module,
        "build_anti_collision_well",
        _counting_builder,
    )

    wells, well_cache, reused_count, rebuilt_count = (
        build_anti_collision_wells_for_successes(
            successes,
            model=PlanningUncertaintyModel(),
            well_signature_by_name={"WELL-A": "a-v1", "WELL-B": "b-v1"},
            include_display_geometry=False,
            build_overlap_geometry=False,
        )
    )

    assert built_names == ["WELL-A", "WELL-B"]
    assert reused_count == 0
    assert rebuilt_count == 2

    built_names.clear()
    next_wells, next_cache, reused_count, rebuilt_count = (
        build_anti_collision_wells_for_successes(
            successes,
            model=PlanningUncertaintyModel(),
            well_signature_by_name={"WELL-A": "a-v1", "WELL-B": "b-v2"},
            previous_well_cache=well_cache,
            include_display_geometry=False,
            build_overlap_geometry=False,
        )
    )

    assert built_names == ["WELL-B"]
    assert reused_count == 1
    assert rebuilt_count == 1
    assert next_wells[0] is wells[0]
    assert next_wells[1] is not wells[1]
    assert set(next_cache) == {"WELL-A", "WELL-B"}


def test_analyze_anti_collision_does_not_skip_pair_by_terminal_geometry_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(1000.0, 0.0, 0.0),
        t1=Point3D(2000.0, 0.0, 0.0),
        t3=Point3D(3000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )

    envelope_by_name = {
        "WELL-A": anticollision_module._AntiCollisionLateralEnvelope(
            min_x_m=0.0,
            max_x_m=1000.0,
            min_y_m=0.0,
            max_y_m=50.0,
            max_lateral_radius_m=0.0,
            surface_x_m=0.0,
            surface_y_m=0.0,
            terminal_x_m=0.0,
            terminal_y_m=0.0,
            terminal_z_m=0.0,
            terminal_lateral_radius_m=100.0,
            terminal_spatial_radius_m=100.0,
        ),
        "WELL-B": anticollision_module._AntiCollisionLateralEnvelope(
            min_x_m=500.0,
            max_x_m=1500.0,
            min_y_m=0.0,
            max_y_m=50.0,
            max_lateral_radius_m=0.0,
            surface_x_m=1000.0,
            surface_y_m=0.0,
            terminal_x_m=1000.0,
            terminal_y_m=0.0,
            terminal_z_m=0.0,
            terminal_lateral_radius_m=100.0,
            terminal_spatial_radius_m=100.0,
        ),
    }

    def _fake_envelope(
        well: anticollision_module.AntiCollisionWell,
    ) -> anticollision_module._AntiCollisionLateralEnvelope:
        return envelope_by_name[str(well.name)]

    seen_pairs: list[tuple[str, str]] = []

    def _record_pair_scan(
        *,
        well_a: object,
        well_b: object,
        build_overlap_geometry: bool,
    ) -> list[AntiCollisionCorridor]:
        assert build_overlap_geometry is False
        seen_pairs.append((str(getattr(well_a, "name")), str(getattr(well_b, "name"))))
        return []

    monkeypatch.setattr(
        anticollision_module,
        "_lateral_envelope_for_prefilter",
        _fake_envelope,
    )
    monkeypatch.setattr(
        anticollision_module,
        "_pair_overlap_corridors",
        _record_pair_scan,
    )

    analysis = analyze_anti_collision(
        [well_a, well_b],
        build_overlap_geometry=False,
    )

    assert analysis.pair_count == 1
    assert analysis.overlapping_pair_count == 0
    assert not analysis.corridors
    assert seen_pairs == [("WELL-A", "WELL-B")]


def test_analyze_anti_collision_builds_overlap_geometry_without_display_overlay() -> (
    None
):
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )

    analysis = analyze_anti_collision([well_a, well_b], build_overlap_geometry=True)

    assert analysis.corridors
    assert any(len(corridor.overlap_rings_xyz) > 0 for corridor in analysis.corridors)


def test_lightweight_runtime_analysis_matches_full_report_events_and_recommendations() -> (
    None
):
    stations_a = _vertical_build_stations(
        y_offset_m=0.0,
        lateral_y_t1_m=60.0,
        lateral_y_end_m=280.0,
    )
    stations_b = _vertical_build_stations(
        y_offset_m=5.0,
        lateral_y_t1_m=140.0,
        lateral_y_end_m=340.0,
    )
    success_a = SuccessfulWellPlan(
        name="WELL-A",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(50.0, 60.0, 1450.0),
        t3=Point3D(320.0, 280.0, 1775.0),
        stations=stations_a,
        summary={"kop_md_m": 820.0},
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    success_b = SuccessfulWellPlan(
        name="WELL-B",
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(50.0, 140.0, 1450.0),
        t3=Point3D(320.0, 340.0, 1775.0),
        stations=stations_b,
        summary={"kop_md_m": 780.0},
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    full_analysis = analyze_anti_collision(
        [
            build_anti_collision_well(
                name="WELL-A",
                color="#0B6E4F",
                stations=stations_a,
                surface=success_a.surface,
                t1=success_a.t1,
                t3=success_a.t3,
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="WELL-B",
                color="#D1495B",
                stations=stations_b,
                surface=success_b.surface,
                t1=success_b.t1,
                t3=success_b.t3,
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
        ],
        build_overlap_geometry=False,
    )
    runtime_analysis = build_anti_collision_analysis_for_successes(
        [success_a, success_b],
        model=full_analysis.wells[0].overlay.model,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=820.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=780.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    assert runtime_analysis.zones
    assert all(
        len(corridor.overlap_rings_xyz) == 0 for corridor in runtime_analysis.corridors
    )
    runtime_events = anti_collision_report_events(runtime_analysis)
    full_events = anti_collision_report_events(full_analysis)
    assert len(runtime_events) == len(full_events)
    for runtime_event, full_event in zip(runtime_events, full_events):
        assert runtime_event.well_a == full_event.well_a
        assert runtime_event.well_b == full_event.well_b
        assert runtime_event.classification == full_event.classification
        assert runtime_event.priority_rank == full_event.priority_rank
        assert runtime_event.label_a == full_event.label_a
        assert runtime_event.label_b == full_event.label_b
        assert runtime_event.md_a_start_m == full_event.md_a_start_m
        assert runtime_event.md_a_end_m == full_event.md_a_end_m
        assert runtime_event.md_b_start_m == full_event.md_b_start_m
        assert runtime_event.md_b_end_m == full_event.md_b_end_m
        assert runtime_event.min_separation_factor == pytest.approx(
            full_event.min_separation_factor
        )
        assert runtime_event.max_overlap_depth_m == pytest.approx(
            full_event.max_overlap_depth_m
        )

    runtime_recommendations = build_anti_collision_recommendations(
        runtime_analysis,
        well_context_by_name=contexts,
    )
    full_recommendations = build_anti_collision_recommendations(
        full_analysis,
        well_context_by_name=contexts,
    )
    assert len(runtime_recommendations) == len(full_recommendations)
    for runtime_item, full_item in zip(runtime_recommendations, full_recommendations):
        assert runtime_item.category == full_item.category
        assert runtime_item.summary == full_item.summary
        assert runtime_item.detail == full_item.detail
        assert runtime_item.expected_maneuver == full_item.expected_maneuver
        assert runtime_item.action_label == full_item.action_label
        assert runtime_item.can_prepare_rerun == full_item.can_prepare_rerun
        assert runtime_item.affected_wells == full_item.affected_wells
        assert runtime_item.min_separation_factor == pytest.approx(
            full_item.min_separation_factor
        )
        assert runtime_item.max_overlap_depth_m == pytest.approx(
            full_item.max_overlap_depth_m
        )


def test_welltracks4_well_03_well_09_shared_surface_overlap_is_reported() -> None:
    records = [
        record
        for record in parse_welltrack_text(
            Path("tests/test_data/WELLTRACKS4.INC").read_text(encoding="utf-8")
        )
        if str(record.name) in {"well_03", "well_09"}
    ]
    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={str(record.name) for record in records},
        config=TrajectoryConfig(turn_solver_max_restarts=0),
    )
    assert {str(row["Скважина"]): str(row["Статус"]) for row in rows} == {
        "well_03": "OK",
        "well_09": "OK",
    }

    analysis = build_anti_collision_analysis_for_successes(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        include_display_geometry=False,
        build_overlap_geometry=False,
    )

    pair_events = [
        event
        for event in anti_collision_report_events(analysis)
        if {str(event.well_a), str(event.well_b)} == {"well_03", "well_09"}
    ]
    assert any(
        str(event.classification) == "trajectory"
        and float(event.min_center_distance_m) == pytest.approx(0.0)
        and float(event.max_overlap_depth_m) > 0.0
        for event in pair_events
    )


def test_welltracks4_well_06_well_07_visual_cones_use_dense_iscwsa_samples() -> None:
    records = [
        record
        for record in parse_welltrack_text(
            Path("tests/test_data/WELLTRACKS4.INC").read_text(encoding="utf-8")
        )
        if str(record.name) in {"well_06", "well_07"}
    ]
    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={str(record.name) for record in records},
        config=TrajectoryConfig(turn_solver_max_restarts=0),
    )
    assert {str(row["Скважина"]): str(row["Статус"]) for row in rows} == {
        "well_06": "OK",
        "well_07": "OK",
    }

    analysis = build_anti_collision_analysis_for_successes(
        successes,
        model=planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET),
        include_display_geometry=True,
        build_overlap_geometry=True,
    )
    wells_by_name = {str(well.name): well for well in analysis.wells}
    pair_events = [
        event
        for event in anti_collision_report_events(analysis)
        if {str(event.well_a), str(event.well_b)} == {"well_06", "well_07"}
    ]
    late_pair_events = [
        event
        for event in pair_events
        if min(float(event.md_a_start_m), float(event.md_b_start_m)) > 1000.0
    ]

    assert pair_events
    assert late_pair_events == []
    assert len(wells_by_name["well_06"].overlay.samples) >= 150
    assert len(wells_by_name["well_07"].overlay.samples) >= 150


def test_overlap_ring_is_not_reduced_to_uniform_circle_for_offset_wells() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])

    radial_spreads = []
    for corridor in analysis.corridors:
        for ring in corridor.overlap_rings_xyz:
            ring_xyz = np.asarray(ring, dtype=float)
            center = np.mean(ring_xyz, axis=0)
            radial_distances = np.linalg.norm(ring_xyz - center[None, :], axis=1)
            radial_spreads.append(
                float(np.max(radial_distances) - np.min(radial_distances))
            )

    assert max(radial_spreads) > 0.25


def test_runtime_analysis_supports_reference_trajectory_wells_without_target_overlap_pollution() -> (
    None
):
    success = SuccessfulWellPlan(
        name="WELL-A",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        stations=_straight_stations(y_offset_m=0.0),
        summary={"kop_md_m": 700.0},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 1000.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 1000.0,
                },
                {
                    "Wellname": "FACT-1",
                    "Type": "actual",
                    "X": 2000.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 2000.0,
                },
            ]
        )
    )
    reference_model = build_anti_collision_well(
        name="TMP",
        color="#000000",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    ).overlay.model

    analysis = build_anti_collision_analysis_for_successes(
        [success],
        model=reference_model,
        reference_wells=reference_wells,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )

    assert len(analysis.wells) == 2
    assert any(bool(well.is_reference_only) for well in analysis.wells)
    assert analysis.pair_count == 1
    assert analysis.target_overlap_pair_count == 0


def test_runtime_analysis_disambiguates_reference_name_matching_planned_well() -> None:
    success = SuccessfulWellPlan(
        name="WELL-A",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        stations=_straight_stations(y_offset_m=0.0),
        summary={"kop_md_m": 700.0},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "WELL-A",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "WELL-A",
                    "Type": "actual",
                    "X": 1000.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 1000.0,
                },
                {
                    "Wellname": "WELL-A",
                    "Type": "actual",
                    "X": 2000.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 2000.0,
                },
            ]
        )
    )
    reference_model = build_anti_collision_well(
        name="TMP",
        color="#000000",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    ).overlay.model

    analysis = build_anti_collision_analysis_for_successes(
        [success],
        model=reference_model,
        name_to_color={"WELL-A": "#123456"},
        reference_wells=reference_wells,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )

    assert [(well.name, well.well_kind) for well in analysis.wells] == [
        ("WELL-A", "project"),
        ("WELL-A (Фактическая)", "actual"),
    ]
    assert [well.color for well in analysis.wells] == ["#123456", "#6B7280"]
    assert analysis.pair_count == 1
    assert all(corridor.well_a != corridor.well_b for corridor in analysis.corridors)


def test_runtime_analysis_disambiguates_actual_and_approved_same_stem() -> None:
    success = SuccessfulWellPlan(
        name="WELL-P",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        stations=_straight_stations(y_offset_m=0.0),
        summary={"kop_md_m": 700.0},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    reference_wells = tuple(
        parse_reference_trajectory_table(
            [
                {
                    "Wellname": "DUP",
                    "Type": "actual",
                    "X": 0.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "DUP",
                    "Type": "actual",
                    "X": 1000.0,
                    "Y": 5.0,
                    "Z": 0.0,
                    "MD": 1000.0,
                },
                {
                    "Wellname": "DUP",
                    "Type": "approved",
                    "X": 0.0,
                    "Y": 6.0,
                    "Z": 0.0,
                    "MD": 0.0,
                },
                {
                    "Wellname": "DUP",
                    "Type": "approved",
                    "X": 1000.0,
                    "Y": 6.0,
                    "Z": 0.0,
                    "MD": 1000.0,
                },
            ]
        )
    )
    reference_model = build_anti_collision_well(
        name="TMP",
        color="#000000",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    ).overlay.model

    analysis = build_anti_collision_analysis_for_successes(
        [success],
        model=reference_model,
        reference_wells=reference_wells,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )

    assert [well.name for well in analysis.wells] == [
        "WELL-P",
        "DUP (Фактическая)",
        "DUP (Проектная утвержденная)",
    ]
    assert analysis.pair_count == 2
    compared_pairs = {
        tuple(sorted((corridor.well_a, corridor.well_b)))
        for corridor in analysis.corridors
    }
    assert ("DUP (Фактическая)", "WELL-P") in compared_pairs
    assert ("DUP (Проектная утвержденная)", "WELL-P") in compared_pairs


def test_runtime_analysis_skips_reference_to_reference_pairs() -> None:
    calculated_well = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        include_display_geometry=False,
    )
    reference_actual = build_anti_collision_well(
        name="FACT-1",
        color="#6B7280",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=None,
        t3=None,
        azimuth_deg=90.0,
        md_t1_m=None,
        include_display_geometry=False,
        well_kind="actual",
        is_reference_only=True,
    )
    reference_approved = build_anti_collision_well(
        name="APP-1",
        color="#C62828",
        stations=_straight_stations(y_offset_m=6.0),
        surface=Point3D(0.0, 6.0, 0.0),
        t1=None,
        t3=None,
        azimuth_deg=90.0,
        md_t1_m=None,
        include_display_geometry=False,
        well_kind="approved",
        is_reference_only=True,
    )

    analysis = analyze_anti_collision(
        [calculated_well, reference_actual, reference_approved],
        build_overlap_geometry=False,
    )

    assert analysis.pair_count == 2
    compared_pairs = {
        tuple(sorted((corridor.well_a, corridor.well_b)))
        for corridor in analysis.corridors
    }
    assert ("APP-1", "FACT-1") not in compared_pairs


def test_runtime_analysis_includes_dev_references_from_multiple_folders(
    tmp_path,
) -> None:
    success = SuccessfulWellPlan(
        name="WELL-P",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        stations=_straight_stations(y_offset_m=0.0),
        summary={"kop_md_m": 700.0},
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    actual_a = tmp_path / "actual_a"
    actual_b = tmp_path / "actual_b"
    approved = tmp_path / "approved"
    actual_a.mkdir()
    actual_b.mkdir()
    approved.mkdir()

    def _write_dev(path, *, y_offset_m: float) -> None:
        path.write_text(
            "\n".join(
                [
                    "MD X Y Z",
                    f"0 0 {y_offset_m} 0",
                    f"1000 1000 {y_offset_m} -10",
                    f"2000 2000 {y_offset_m} -20",
                ]
            ),
            encoding="utf-8",
        )

    _write_dev(actual_a / "well_2.dev", y_offset_m=5.0)
    _write_dev(actual_b / "well_10.dev", y_offset_m=6.0)
    _write_dev(approved / "approved_1.dev", y_offset_m=7.0)
    reference_wells = tuple(
        [
            *parse_reference_trajectory_dev_directories(
                [actual_a, actual_b],
                kind="actual",
            ),
            *parse_reference_trajectory_dev_directories(
                [approved],
                kind="approved",
            ),
        ]
    )
    reference_model = build_anti_collision_well(
        name="TMP",
        color="#000000",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    ).overlay.model

    analysis = build_anti_collision_analysis_for_successes(
        [success],
        model=reference_model,
        reference_wells=reference_wells,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )

    assert [(well.name, well.well_kind) for well in analysis.wells] == [
        ("WELL-P", "project"),
        ("well_2", "actual"),
        ("well_10", "actual"),
        ("approved_1", "approved"),
    ]
    assert analysis.pair_count == 3


def test_report_merges_adjacent_corridors_into_single_event() -> None:
    corridor_a = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4200.0,
        md_a_end_m=4250.0,
        md_b_start_m=4100.0,
        md_b_end_m=4150.0,
        md_a_values_m=np.array([4200.0, 4250.0], dtype=float),
        md_b_values_m=np.array([4100.0, 4150.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.zeros((16, 3), dtype=float),
            np.ones((16, 3), dtype=float),
        ),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.72, 0.69], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    corridor_b = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4250.0,
        md_a_end_m=4300.0,
        md_b_start_m=4150.0,
        md_b_end_m=4200.0,
        md_a_values_m=np.array([4250.0, 4300.0], dtype=float),
        md_b_values_m=np.array([4150.0, 4200.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.zeros((16, 3), dtype=float),
            np.ones((16, 3), dtype=float),
        ),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.68, 0.66], dtype=float),
        overlap_depth_values_m=np.array([10.0, 11.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor_a, corridor_b),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.66,
    )

    events = anti_collision_report_events(analysis)
    rows = anti_collision_report_rows(analysis)

    assert len(events) == 1
    assert events[0].md_a_start_m == 4200.0
    assert events[0].md_a_end_m == 4300.0
    assert events[0].md_b_start_m == 4100.0
    assert events[0].md_b_end_m == 4200.0
    assert events[0].merged_corridor_count == 2
    assert rows[0]["Интервал A, м"] == "4200 - 4300"
    assert rows[0]["Интервал B, м"] == "4100 - 4200"
    assert "Мин. расстояние, м" in rows[0]
    assert rows[0]["Мин. расстояние, м"] >= 0.0


def test_report_keeps_distinct_target_labels_in_adjacent_events() -> None:
    def _corridor(
        *,
        start_m: float,
        end_m: float,
        classification: str,
        priority_rank: int,
        label_a: str,
        label_b: str,
        sf: float,
    ) -> AntiCollisionCorridor:
        return AntiCollisionCorridor(
            well_a="well_01",
            well_b="well_11",
            classification=classification,
            priority_rank=priority_rank,
            label_a=label_a,
            label_b=label_b,
            md_a_start_m=start_m,
            md_a_end_m=end_m,
            md_b_start_m=start_m,
            md_b_end_m=end_m,
            md_a_values_m=np.array([start_m, end_m], dtype=float),
            md_b_values_m=np.array([start_m, end_m], dtype=float),
            label_a_values=(label_a, label_a),
            label_b_values=(label_b, label_b),
            midpoint_xyz=np.array(
                [[start_m, 0.0, 0.0], [end_m, 0.0, 0.0]], dtype=float
            ),
            overlap_rings_xyz=(
                np.zeros((16, 3), dtype=float),
                np.ones((16, 3), dtype=float),
            ),
            overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
            separation_factor_values=np.array([sf, sf], dtype=float),
            overlap_depth_values_m=np.array([8.0, 8.0], dtype=float),
        )

    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            _corridor(
                start_m=1000.0,
                end_m=1050.0,
                classification="target-trajectory",
                priority_rank=1,
                label_a=TARGET_T1,
                label_b="",
                sf=0.6,
            ),
            _corridor(
                start_m=1050.0,
                end_m=1100.0,
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                sf=0.5,
            ),
            _corridor(
                start_m=1100.0,
                end_m=1150.0,
                classification="target-trajectory",
                priority_rank=1,
                label_a=TARGET_T3,
                label_b="",
                sf=0.4,
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=1,
        worst_separation_factor=0.4,
    )

    events = anti_collision_report_events(analysis)

    assert [(event.label_a, event.label_b) for event in events] == [
        (TARGET_T1, ""),
        (TARGET_T3, ""),
    ]
    assert events[0].merged_corridor_count == 2
    assert events[0].md_a_start_m == 1000.0
    assert events[0].md_a_end_m == 1100.0
    assert events[1].merged_corridor_count == 1


def test_recommendations_prioritize_target_spacing_for_target_overlap() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])

    recommendations = build_anti_collision_recommendations(analysis)
    rows = anti_collision_recommendation_rows(recommendations)

    assert recommendations
    assert recommendations[0].category == RECOMMENDATION_TARGET_SPACING
    assert recommendations[0].required_spacing_t1_m is not None
    assert recommendations[0].can_prepare_rerun is False
    assert "2σ overlap" not in recommendations[0].summary
    assert (
        "увеличьте расстояние между конфликтными участками"
        in recommendations[0].summary
    )
    assert rows[0]["Тип действия"] == "Цели / spacing"
    assert "Рекомендация по устранению" in rows[0]
    assert rows[0]["Подготовка пересчета"] == "Только рекомендация по целям"


def test_recommendations_prepare_anticollision_kop_build1_for_vertical_collision() -> (
    None
):
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_vertical_build_stations(
            y_offset_m=0.0,
            lateral_y_t1_m=60.0,
            lateral_y_end_m=280.0,
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(50.0, 60.0, 1450.0),
        t3=Point3D(320.0, 280.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_vertical_build_stations(
            y_offset_m=5.0,
            lateral_y_t1_m=140.0,
            lateral_y_end_m=340.0,
        ),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(50.0, 140.0, 1450.0),
        t3=Point3D(320.0, 340.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=820.0,
            kop_min_vertical_m=550.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=780.0,
            kop_min_vertical_m=550.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    vertical_recommendations = [
        item for item in recommendations if item.category == RECOMMENDATION_REDUCE_KOP
    ]

    assert vertical_recommendations
    recommendation = vertical_recommendations[0]
    assert recommendation.can_prepare_rerun is True
    assert recommendation.affected_wells == ("WELL-A", "WELL-B")
    assert all(
        suggestion.config_updates.get("optimization_mode") == "anti_collision_avoidance"
        for suggestion in recommendation.override_suggestions
    )
    assert recommendation.expected_maneuver == "Сместить ранний уход: KOP / BUILD1"
    assert (
        recommendation.action_label
        == "Подготовить anti-collision пересчет (KOP/BUILD1)"
    )


def test_pre_kop_trajectory_conflict_is_reclassified_to_kop_build1_anticollision() -> (
    None
):
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_vertical_build_stations(
            y_offset_m=0.0,
            lateral_y_t1_m=60.0,
            lateral_y_end_m=280.0,
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(50.0, 60.0, 1450.0),
        t3=Point3D(320.0, 280.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_vertical_build_stations(
            y_offset_m=5.0,
            lateral_y_t1_m=140.0,
            lateral_y_end_m=340.0,
        ),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(50.0, 140.0, 1450.0),
        t3=Point3D(320.0, 340.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        include_display_geometry=False,
    )
    corridor = AntiCollisionCorridor(
        well_a="WELL-A",
        well_b="WELL-B",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=100.0,
        md_a_end_m=1450.0,
        md_b_start_m=100.0,
        md_b_end_m=1450.0,
        md_a_values_m=np.array([100.0, 1450.0], dtype=float),
        md_b_values_m=np.array([100.0, 1450.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.6, 0.55], dtype=float),
        overlap_depth_values_m=np.array([10.0, 12.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(well_a, well_b),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.55,
    )
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=820.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=780.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    assert recommendations[0].category == RECOMMENDATION_REDUCE_KOP
    assert (
        recommendations[0].action_label
        == "Подготовить anti-collision пересчет (KOP/BUILD1)"
    )
    assert all(
        suggestion.config_updates.get("optimization_mode") == "anti_collision_avoidance"
        for suggestion in recommendations[0].override_suggestions
    )


def test_build2_trajectory_conflict_reports_build2_maneuver() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [3600.0, 4000.0, 4300.0, 4580.0, 4680.0],
            "INC_deg": [40.0, 50.0, 70.0, 86.0, 90.0],
            "AZI_deg": [150.0, 180.0, 220.0, 238.3, 238.3],
            "X_m": [0.0, 200.0, 600.0, 950.0, 1050.0],
            "Y_m": [0.0, 120.0, 360.0, 520.0, 580.0],
            "Z_m": [2200.0, 2500.0, 3000.0, 3400.0, 3420.0],
            "segment": ["BUILD2", "BUILD2", "BUILD2", "BUILD2", "HORIZONTAL"],
        }
    )
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=stations,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(950.0, 520.0, 3400.0),
        t3=Point3D(1050.0, 580.0, 3420.0),
        azimuth_deg=90.0,
        md_t1_m=4580.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=stations.assign(Y_m=stations["Y_m"] + 20.0),
        surface=Point3D(0.0, 20.0, 0.0),
        t1=Point3D(950.0, 540.0, 3400.0),
        t3=Point3D(1050.0, 600.0, 3420.0),
        azimuth_deg=90.0,
        md_t1_m=4580.0,
        include_display_geometry=False,
    )
    corridor = AntiCollisionCorridor(
        well_a="WELL-A",
        well_b="WELL-B",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4075.0,
        md_a_end_m=4275.0,
        md_b_start_m=3975.0,
        md_b_end_m=4225.0,
        md_a_values_m=np.array([4075.0, 4275.0], dtype=float),
        md_b_values_m=np.array([3975.0, 4225.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.7, 0.65], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(well_a, well_b),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.65,
    )
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=2000.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4580.0,
            md_total_m=5000.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=1900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4580.0,
            md_total_m=5000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    assert recommendations[0].category == RECOMMENDATION_TRAJECTORY_REVIEW
    assert recommendations[0].expected_maneuver == MANEUVER_BUILD2_ENTRY


def test_recommendations_prepare_pairwise_anti_collision_rerun_for_trajectory_collision() -> (
    None
):
    corridor = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4200.0,
        md_a_end_m=4300.0,
        md_b_start_m=4100.0,
        md_b_end_m=4200.0,
        md_a_values_m=np.array([4200.0, 4300.0], dtype=float),
        md_b_values_m=np.array([4100.0, 4200.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.zeros((16, 3), dtype=float),
            np.ones((16, 3), dtype=float),
        ),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.72, 0.69], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    contexts = {
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3500.0,
            md_total_m=5600.0,
            optimization_mode="none",
        ),
        "well_05": AntiCollisionWellContext(
            well_name="well_05",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.category == RECOMMENDATION_TRAJECTORY_REVIEW
    assert recommendation.can_prepare_rerun is True
    assert set(recommendation.affected_wells) == {"well_02", "well_05"}
    assert len(recommendation.override_suggestions) == 2
    assert {str(item.well_name) for item in recommendation.override_suggestions} == {
        "well_02",
        "well_05",
    }
    assert all(
        item.config_updates["optimization_mode"]
        == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
        for item in recommendation.override_suggestions
    )
    assert (
        recommendation.expected_maneuver == "Pre-entry azimuth turn / сдвиг HOLD до t1"
    )
    rows = anti_collision_recommendation_rows(recommendations)
    assert rows[0]["Ожидаемый маневр"] == recommendation.expected_maneuver


def test_recommendations_switch_movable_well_after_one_side_already_used_anticollision() -> (
    None
):
    corridor = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4200.0,
        md_a_end_m=4300.0,
        md_b_start_m=4100.0,
        md_b_end_m=4200.0,
        md_a_values_m=np.array([4200.0, 4300.0], dtype=float),
        md_b_values_m=np.array([4100.0, 4200.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.zeros((16, 3), dtype=float),
            np.ones((16, 3), dtype=float),
        ),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.72, 0.69], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    contexts = {
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=2200.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4500.0,
            md_total_m=5600.0,
            optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
        ),
        "well_05": AntiCollisionWellContext(
            well_name="well_05",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="minimize_kop",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.category == RECOMMENDATION_TRAJECTORY_REVIEW
    assert set(recommendation.affected_wells) == {"well_02", "well_05"}
    assert recommendation.override_suggestions[0].well_name == "well_05"


def test_recommendations_disable_repeated_late_trajectory_rerun_after_both_wells_already_used_anticollision() -> (
    None
):
    corridor = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4200.0,
        md_a_end_m=4300.0,
        md_b_start_m=4100.0,
        md_b_end_m=4200.0,
        md_a_values_m=np.array([4200.0, 4300.0], dtype=float),
        md_b_values_m=np.array([4100.0, 4200.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.zeros((16, 3), dtype=float),
            np.ones((16, 3), dtype=float),
        ),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.72, 0.69], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    contexts = {
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=550.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4500.0,
            md_total_m=6800.0,
            optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
        ),
        "well_05": AntiCollisionWellContext(
            well_name="well_05",
            kop_md_m=550.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6700.0,
            optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.category == RECOMMENDATION_TRAJECTORY_REVIEW
    assert recommendation.can_prepare_rerun is False
    assert recommendation.override_suggestions == ()
    assert recommendation.action_label == "Только рекомендация"
    assert "исчерпан" in recommendation.summary


def test_recommendation_clusters_merge_connected_pairs_into_single_cluster() -> None:
    corridor_ab = AntiCollisionCorridor(
        well_a="well_01",
        well_b="well_03",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=2200.0,
        md_a_end_m=2400.0,
        md_b_start_m=2100.0,
        md_b_end_m=2350.0,
        md_a_values_m=np.array([2200.0, 2400.0], dtype=float),
        md_b_values_m=np.array([2100.0, 2350.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.zeros((16, 3), dtype=float),
            np.ones((16, 3), dtype=float),
        ),
        overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
        separation_factor_values=np.array([0.82, 0.78], dtype=float),
        overlap_depth_values_m=np.array([4.0, 5.0], dtype=float),
    )
    corridor_bc = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_03",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=2600.0,
        md_a_end_m=2900.0,
        md_b_start_m=2550.0,
        md_b_end_m=2850.0,
        md_a_values_m=np.array([2600.0, 2900.0], dtype=float),
        md_b_values_m=np.array([2550.0, 2850.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(
            np.zeros((16, 3), dtype=float),
            np.ones((16, 3), dtype=float),
        ),
        overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
        separation_factor_values=np.array([0.76, 0.71], dtype=float),
        overlap_depth_values_m=np.array([6.0, 7.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor_ab, corridor_bc),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.71,
    )
    contexts = {
        "well_01": AntiCollisionWellContext(
            well_name="well_01",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3500.0,
            md_total_m=5200.0,
            optimization_mode="none",
        ),
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3600.0,
            md_total_m=5400.0,
            optimization_mode="none",
        ),
        "well_03": AntiCollisionWellContext(
            well_name="well_03",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.well_names == ("well_01", "well_02", "well_03")
    assert cluster.recommendation_count == 2
    assert cluster.trajectory_conflict_count == 2
    assert cluster.can_prepare_rerun is True
    assert cluster.affected_wells == ("well_01", "well_02", "well_03")
    assert cluster.first_rerun_well == "well_03"
    assert cluster.rerun_order_label == "well_03 → well_02 → well_01"
    assert cluster.action_steps[0].well_name == "well_03"
    assert (
        cluster.action_steps[0].optimization_mode
        == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    )
    rows = cluster.recommendations
    assert rows[0].expected_maneuver


def test_cluster_with_target_spacing_conflict_is_not_actionable_for_rerun() -> None:
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="well_01",
                well_b="well_02",
                classification="target-target",
                priority_rank=1,
                label_a=TARGET_T1,
                label_b=TARGET_T1,
                md_a_start_m=3500.0,
                md_a_end_m=3500.0,
                md_b_start_m=3600.0,
                md_b_end_m=3600.0,
                md_a_values_m=np.array([3500.0], dtype=float),
                md_b_values_m=np.array([3600.0], dtype=float),
                label_a_values=(TARGET_T1,),
                label_b_values=(TARGET_T1,),
                midpoint_xyz=np.array([[1000.0, 0.0, 2400.0]], dtype=float),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([12.0], dtype=float),
                separation_factor_values=np.array([0.62], dtype=float),
                overlap_depth_values_m=np.array([18.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="well_02",
                well_b="well_03",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=4100.0,
                md_a_end_m=4550.0,
                md_b_start_m=4050.0,
                md_b_end_m=4500.0,
                md_a_values_m=np.array([4100.0, 4550.0], dtype=float),
                md_b_values_m=np.array([4050.0, 4500.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array(
                    [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([6.0, 6.0], dtype=float),
                separation_factor_values=np.array([0.81, 0.76], dtype=float),
                overlap_depth_values_m=np.array([7.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=1,
        worst_separation_factor=0.62,
    )
    contexts = {
        "well_01": AntiCollisionWellContext(
            well_name="well_01",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3500.0,
            md_total_m=5200.0,
            optimization_mode="none",
        ),
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3600.0,
            md_total_m=5400.0,
            optimization_mode="none",
        ),
        "well_03": AntiCollisionWellContext(
            well_name="well_03",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.target_conflict_count == 1
    assert cluster.trajectory_conflict_count == 1
    assert cluster.can_prepare_rerun is False
    assert cluster.action_label == "Только advisory"
    assert cluster.blocking_advisory == "Сначала решить spacing целей."


def test_cluster_action_steps_normalize_mixed_vertical_and_trajectory_for_same_well() -> (
    None
):
    analysis = AntiCollisionAnalysis(
        wells=(
            build_anti_collision_well(
                name="well_a",
                color="#0B6E4F",
                stations=_vertical_build_stations(
                    y_offset_m=0.0,
                    lateral_y_t1_m=30.0,
                    lateral_y_end_m=120.0,
                ),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(50.0, 30.0, 1450.0),
                t3=Point3D(320.0, 120.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_b",
                color="#3A86FF",
                stations=_vertical_build_stations(
                    y_offset_m=10.0,
                    lateral_y_t1_m=40.0,
                    lateral_y_end_m=150.0,
                ),
                surface=Point3D(0.0, 10.0, 0.0),
                t1=Point3D(50.0, 40.0, 1450.0),
                t3=Point3D(320.0, 150.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_c",
                color="#00798C",
                stations=_straight_stations(y_offset_m=0.0),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(1000.0, 0.0, 0.0),
                t3=Point3D(2000.0, 0.0, 0.0),
                azimuth_deg=90.0,
                md_t1_m=1000.0,
            ),
        ),
        corridors=(
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_b",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=400.0,
                md_a_end_m=700.0,
                md_b_start_m=410.0,
                md_b_end_m=710.0,
                md_a_values_m=np.array([400.0, 700.0], dtype=float),
                md_b_values_m=np.array([410.0, 710.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array(
                    [[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                separation_factor_values=np.array([0.82, 0.78], dtype=float),
                overlap_depth_values_m=np.array([5.0, 7.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_c",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1750.0,
                md_a_end_m=2050.0,
                md_b_start_m=1700.0,
                md_b_end_m=2000.0,
                md_a_values_m=np.array([1750.0, 2050.0], dtype=float),
                md_b_values_m=np.array([1700.0, 2000.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array(
                    [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.76, 0.73], dtype=float),
                overlap_depth_values_m=np.array([7.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.73,
    )
    contexts = {
        "well_a": AntiCollisionWellContext(
            well_name="well_a",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2600.0,
            optimization_mode="none",
        ),
        "well_b": AntiCollisionWellContext(
            well_name="well_b",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2700.0,
            optimization_mode="none",
        ),
        "well_c": AntiCollisionWellContext(
            well_name="well_c",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1000.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    cluster = clusters[0]
    step = next(item for item in cluster.action_steps if item.well_name == "well_a")
    assert step.optimization_mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    assert step.category == "mixed"
    assert step.expected_maneuver == MANEUVER_POSTENTRY_TURN


def test_saturated_early_kop_build1_conflict_becomes_advisory_and_shifts_cluster_to_trajectory() -> (
    None
):
    analysis = AntiCollisionAnalysis(
        wells=(
            build_anti_collision_well(
                name="well_a",
                color="#0B6E4F",
                stations=_vertical_build_stations(
                    y_offset_m=0.0,
                    lateral_y_t1_m=30.0,
                    lateral_y_end_m=120.0,
                ),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(50.0, 30.0, 1450.0),
                t3=Point3D(320.0, 120.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_b",
                color="#3A86FF",
                stations=_vertical_build_stations(
                    y_offset_m=10.0,
                    lateral_y_t1_m=40.0,
                    lateral_y_end_m=150.0,
                ),
                surface=Point3D(0.0, 10.0, 0.0),
                t1=Point3D(50.0, 40.0, 1450.0),
                t3=Point3D(320.0, 150.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_c",
                color="#00798C",
                stations=_straight_stations(y_offset_m=0.0),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(1000.0, 0.0, 0.0),
                t3=Point3D(2000.0, 0.0, 0.0),
                azimuth_deg=90.0,
                md_t1_m=1000.0,
            ),
        ),
        corridors=(
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_b",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=400.0,
                md_a_end_m=700.0,
                md_b_start_m=410.0,
                md_b_end_m=710.0,
                md_a_values_m=np.array([400.0, 700.0], dtype=float),
                md_b_values_m=np.array([410.0, 710.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array(
                    [[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([6.0, 7.0], dtype=float),
                separation_factor_values=np.array([0.55, 0.52], dtype=float),
                overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_c",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1750.0,
                md_a_end_m=2050.0,
                md_b_start_m=1700.0,
                md_b_end_m=2000.0,
                md_a_values_m=np.array([1750.0, 2050.0], dtype=float),
                md_b_values_m=np.array([1700.0, 2000.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array(
                    [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float
                ),
                overlap_rings_xyz=(
                    np.zeros((16, 3), dtype=float),
                    np.ones((16, 3), dtype=float),
                ),
                overlap_core_radius_m=np.array([3.0, 3.0], dtype=float),
                separation_factor_values=np.array([0.82, 0.80], dtype=float),
                overlap_depth_values_m=np.array([4.0, 4.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.52,
    )
    contexts = {
        "well_a": AntiCollisionWellContext(
            well_name="well_a",
            kop_md_m=550.0,
            kop_min_vertical_m=550.0,
            build1_dls_deg_per_30m=3.0,
            build_dls_max_deg_per_30m=3.0,
            md_t1_m=1500.0,
            md_total_m=2600.0,
            optimization_mode="anti_collision_avoidance",
        ),
        "well_b": AntiCollisionWellContext(
            well_name="well_b",
            kop_md_m=550.0,
            kop_min_vertical_m=550.0,
            build1_dls_deg_per_30m=3.0,
            build_dls_max_deg_per_30m=3.0,
            md_t1_m=1500.0,
            md_total_m=2700.0,
            optimization_mode="anti_collision_avoidance",
        ),
        "well_c": AntiCollisionWellContext(
            well_name="well_c",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            build1_dls_deg_per_30m=2.0,
            build_dls_max_deg_per_30m=3.0,
            md_t1_m=1000.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    early = next(
        item
        for item in recommendations
        if {item.well_a, item.well_b} == {"well_a", "well_b"}
    )
    assert early.category == RECOMMENDATION_REDUCE_KOP
    assert early.can_prepare_rerun is False

    clusters = build_anti_collision_recommendation_clusters(recommendations)
    cluster = clusters[0]
    assert cluster.action_steps
    step = cluster.action_steps[0]
    assert step.category == RECOMMENDATION_TRAJECTORY_REVIEW


def test_cluster_action_steps_keep_worst_sf_first_and_preserve_order_label() -> None:
    recommendations = build_anti_collision_recommendations(
        AntiCollisionAnalysis(
            wells=(),
            corridors=(
                AntiCollisionCorridor(
                    well_a="well_a",
                    well_b="well_c",
                    classification="trajectory",
                    priority_rank=2,
                    label_a="",
                    label_b="",
                    md_a_start_m=2500.0,
                    md_a_end_m=2700.0,
                    md_b_start_m=2600.0,
                    md_b_end_m=2800.0,
                    md_a_values_m=np.array([2500.0, 2700.0], dtype=float),
                    md_b_values_m=np.array([2600.0, 2800.0], dtype=float),
                    label_a_values=("", ""),
                    label_b_values=("", ""),
                    midpoint_xyz=np.array(
                        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float
                    ),
                    overlap_rings_xyz=(
                        np.zeros((16, 3), dtype=float),
                        np.ones((16, 3), dtype=float),
                    ),
                    overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                    separation_factor_values=np.array([0.82, 0.75], dtype=float),
                    overlap_depth_values_m=np.array([5.0, 7.0], dtype=float),
                ),
                AntiCollisionCorridor(
                    well_a="well_a",
                    well_b="well_b",
                    classification="trajectory",
                    priority_rank=2,
                    label_a="",
                    label_b="",
                    md_a_start_m=600.0,
                    md_a_end_m=900.0,
                    md_b_start_m=620.0,
                    md_b_end_m=910.0,
                    md_a_values_m=np.array([600.0, 900.0], dtype=float),
                    md_b_values_m=np.array([620.0, 910.0], dtype=float),
                    label_a_values=("", ""),
                    label_b_values=("", ""),
                    midpoint_xyz=np.array(
                        [[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float
                    ),
                    overlap_rings_xyz=(
                        np.zeros((16, 3), dtype=float),
                        np.ones((16, 3), dtype=float),
                    ),
                    overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                    separation_factor_values=np.array([0.79, 0.77], dtype=float),
                    overlap_depth_values_m=np.array([4.0, 5.0], dtype=float),
                ),
            ),
            well_segments=(),
            zones=(),
            pair_count=2,
            overlapping_pair_count=2,
            target_overlap_pair_count=0,
            worst_separation_factor=0.75,
        ),
        well_context_by_name={
            "well_a": AntiCollisionWellContext(
                well_name="well_a",
                kop_md_m=900.0,
                kop_min_vertical_m=550.0,
                md_t1_m=3400.0,
                md_total_m=6000.0,
                optimization_mode="none",
            ),
            "well_b": AntiCollisionWellContext(
                well_name="well_b",
                kop_md_m=840.0,
                kop_min_vertical_m=550.0,
                md_t1_m=1800.0,
                md_total_m=5300.0,
                optimization_mode="none",
            ),
            "well_c": AntiCollisionWellContext(
                well_name="well_c",
                kop_md_m=950.0,
                kop_min_vertical_m=550.0,
                md_t1_m=4200.0,
                md_total_m=6200.0,
                optimization_mode="none",
            ),
        },
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert tuple(step.well_name for step in cluster.action_steps) == (
        "well_a",
        "well_c",
        "well_b",
    )
    assert cluster.first_rerun_well == "well_a"
    assert cluster.rerun_order_label == "well_a → well_c → well_b"
