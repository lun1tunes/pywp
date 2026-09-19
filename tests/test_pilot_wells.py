from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.mcm import compute_positions_min_curv
from pywp.models import (
    PILOT_PLANNING_PILOT_FROM_MAIN_BORE,
    PlannerResult,
    Point3D,
    TrajectoryConfig,
)
from pywp.pilot_wells import (
    PilotWindow,
    SidetrackWindowOverride,
    build_pilot_trajectory,
    combine_pilot_and_sidetrack,
    is_zbs_name,
    is_zbs_record,
    is_pilot_name,
    order_records_with_pilots_first,
    parent_name_for_pilot,
    parent_name_for_zbs,
    paired_pilot_parent_names,
    plan_pilot_from_main_bore,
    pilot_parent_key_for_record,
    plan_reoriented_pilot_sidetrack_fallback,
    select_sidetrack_window,
    sync_pilot_surfaces_to_parents,
    well_name_key,
)
from pywp import pilot_wells
from pywp.planner_types import PlanningError
from pywp.sidetrack_solver import SidetrackPlanner, SidetrackStart
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    planning_uncertainty_model_for_preset,
    station_uncertainty_covariance_samples_for_stations,
)


def test_pilot_name_helpers() -> None:
    assert is_pilot_name("well_04_PL")
    assert is_pilot_name("well_04_pl")
    assert is_pilot_name("201PL")
    assert not is_pilot_name("PL")
    assert parent_name_for_pilot("well_04_PL") == "well_04"
    assert parent_name_for_pilot("201pL") == "201"
    assert well_name_key("201PL") == well_name_key("201_PL") == "201_pl"
    assert is_zbs_name("201ZBS")
    assert parent_name_for_zbs("201zbs") == "201"
    assert well_name_key("201ZBS") == well_name_key("201_ZBS") == "201_zbs"
    assert paired_pilot_parent_names("well_04", "well_04_PL")
    assert paired_pilot_parent_names("well_04", "WELL_04_pl")
    assert paired_pilot_parent_names("201", "201PL")
    assert paired_pilot_parent_names("well_04_2", "well_04_PL")
    assert not paired_pilot_parent_names("well_04", "well_05_PL")


def test_alt_branch_name_can_pair_with_pilot_without_being_zbs() -> None:
    branch = WelltrackRecord(
        name="well_04_2",
        points=(
            WelltrackPoint(x=10.0, y=20.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=1.0),
            WelltrackPoint(x=600.0, y=0.0, z=1200.0, md=2.0),
        ),
    )
    pilot = WelltrackRecord(
        name="well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=1.0),
        ),
    )

    ordered = order_records_with_pilots_first([branch, pilot])
    synced = sync_pilot_surfaces_to_parents([branch, pilot])

    assert is_zbs_record(branch) is False
    assert [record.name for record in ordered] == ["well_04_PL", "well_04_2"]
    assert float(synced[1].points[0].x) == pytest.approx(10.0)
    assert float(synced[1].points[0].y) == pytest.approx(20.0)


def test_sync_pilot_surfaces_can_be_scoped_to_changed_pad_parent_keys() -> None:
    parent_a = WelltrackRecord(
        name="well_a",
        points=(
            WelltrackPoint(x=100.0, y=200.0, z=0.0, md=0.0),
            WelltrackPoint(x=200.0, y=200.0, z=1000.0, md=1000.0),
        ),
    )
    pilot_a = WelltrackRecord(
        name="well_a_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=700.0),
        ),
    )
    parent_b = WelltrackRecord(
        name="well_b",
        points=(
            WelltrackPoint(x=300.0, y=400.0, z=0.0, md=0.0),
            WelltrackPoint(x=400.0, y=400.0, z=1000.0, md=1000.0),
        ),
    )
    pilot_b = WelltrackRecord(
        name="well_b_PL",
        points=(
            WelltrackPoint(x=1.0, y=2.0, z=0.0, md=0.0),
            WelltrackPoint(x=60.0, y=0.0, z=700.0, md=700.0),
        ),
    )

    synced = sync_pilot_surfaces_to_parents(
        [parent_a, pilot_a, parent_b, pilot_b],
        only_parent_keys={pilot_parent_key_for_record(parent_a)},
    )

    assert synced[1].points[0] == parent_a.points[0]
    assert synced[3].points[0] == pilot_b.points[0]


def test_pilot_parent_key_for_record_accepts_plain_name_inputs() -> None:
    branch = WelltrackRecord(
        name="well_04_2",
        points=(
            WelltrackPoint(x=10.0, y=20.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=1.0),
            WelltrackPoint(x=600.0, y=0.0, z=1200.0, md=2.0),
        ),
    )
    pilot = WelltrackRecord(
        name="well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=1.0),
        ),
    )

    assert pilot_parent_key_for_record(branch) == "well_04"
    assert pilot_parent_key_for_record(branch.name) == "well_04"
    assert pilot_parent_key_for_record(pilot) == "well_04"
    assert pilot_parent_key_for_record(pilot.name) == "well_04"


def test_alt_branch_name_without_surface_is_treated_as_fact_sidetrack() -> None:
    sidetrack = WelltrackRecord(
        name="9010_2",
        points=(
            WelltrackPoint(x=650.0, y=0.0, z=1500.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=1500.0, md=2.0),
        ),
    )

    assert is_zbs_record(sidetrack) is True
    assert parent_name_for_zbs("9010_2") == "9010"


def test_shallow_alt_branch_without_surface_is_still_treated_as_fact_sidetrack() -> (
    None
):
    sidetrack = WelltrackRecord(
        name="9010_2",
        points=(
            WelltrackPoint(x=650.0, y=0.0, z=25.0, md=1.0),
            WelltrackPoint(x=1200.0, y=0.0, z=30.0, md=2.0),
        ),
    )

    assert is_zbs_record(sidetrack) is True


def test_alt_branch_with_explicit_surface_label_is_not_zbs_even_with_even_point_count() -> (
    None
):
    branch = WelltrackRecord(
        name="well_04_2",
        points=(
            WelltrackPoint(x=10.0, y=20.0, z=180.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=1.0),
            WelltrackPoint(x=350.0, y=0.0, z=1600.0, md=2.0),
            WelltrackPoint(x=600.0, y=0.0, z=1800.0, md=3.0),
        ),
        point_labels=("S", "t1", "t2", "t3"),
    )

    assert is_zbs_record(branch) is False


def test_order_records_with_pilots_first_uses_case_insensitive_suffix() -> None:
    parent = WelltrackRecord(
        name="well_04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=3.0),
        ),
    )
    pilot = WelltrackRecord(
        name="well_04_pl",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=2.0),
        ),
    )

    ordered = order_records_with_pilots_first([parent, pilot])

    assert [record.name for record in ordered] == ["well_04_pl", "well_04"]


def test_build_pilot_trajectory_starts_vertical_before_building_to_targets() -> None:
    record = WelltrackRecord(
        name="well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=350.0, y=0.0, z=1500.0, md=3.0),
        ),
    )

    pilot = build_pilot_trajectory(
        record,
        config=TrajectoryConfig(md_step_m=100.0),
    )

    assert pilot.summary["kop_md_m"] == pytest.approx(400.0)
    vertical_rows = pilot.stations.loc[pilot.stations["segment"] == "VERTICAL"]
    assert not vertical_rows.empty
    assert set(vertical_rows["X_m"]) == {0.0}
    assert set(vertical_rows["Y_m"]) == {0.0}
    assert set(vertical_rows["INC_deg"]) == {0.0}
    assert pilot.md_first_target_m > float(pilot.summary["kop_md_m"])
    assert float(pilot.stations["X_m"].iloc[-1]) == pytest.approx(350.0)
    assert float(pilot.stations["Y_m"].iloc[-1]) == pytest.approx(0.0)
    assert float(pilot.stations["Z_m"].iloc[-1]) == pytest.approx(1500.0)
    assert pilot.summary["trajectory_type"] == "PILOT"
    assert float(pilot.summary["max_dls_total_deg_per_30m"]) <= 3.0 + 1e-6
    assert {"VERTICAL", "PILOT_BUILD_1", "PILOT_HOLD_1", "PILOT_BUILD_2"}.issubset(
        set(pilot.stations["segment"])
    )


def test_pilot_from_main_bore_optimizes_total_drilled_md() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        md_step_control_m=5.0,
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=12.0,
        dls_horizontal_max_deg_per_30m=3.0,
        max_inc_deg=110.0,
        turn_solver_max_restarts=0,
    )
    surface = Point3D(0.0, 0.0, 0.0)
    main = pilot_wells.TrajectoryPlanner().plan(
        surface=surface,
        t1=Point3D(800.0, 0.0, 2200.0),
        t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
    )
    pilot_targets = (
        surface,
        Point3D(260.0, 100.0, 1200.0),
        Point3D(300.0, 120.0, 1450.0),
    )

    planned = plan_pilot_from_main_bore(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_target_points=pilot_targets,
        main_bore=main,
        pilot_config=config,
        main_config=config,
    )

    assert planned.pilot.summary["pilot_planning_mode"] == (
        PILOT_PLANNING_PILOT_FROM_MAIN_BORE
    )
    assert float(planned.window.point.z) < float(pilot_targets[1].z)
    assert planned.total_drilled_md_m == pytest.approx(
        float(main.stations["MD_m"].iloc[-1])
        + float(planned.pilot.md_total_m)
        - float(planned.window.md_m)
    )
    assert planned.pilot_tail_md_m == pytest.approx(
        float(planned.pilot.md_total_m) - float(planned.window.md_m)
    )
    assert planned.window_to_first_pl_m == pytest.approx(
        np.linalg.norm(
            np.asarray(
                [
                    planned.window.point.x - pilot_targets[1].x,
                    planned.window.point.y - pilot_targets[1].y,
                    planned.window.point.z - pilot_targets[1].z,
                ]
            )
        )
    )
    assert planned.window_search_resolution_m == pytest.approx(0.5)
    main_md = float(main.stations["MD_m"].iloc[-1])
    main_md_values = main.stations["MD_m"].to_numpy(dtype=float)
    grid_scores: list[float] = []
    min_window_md = float(config.min_structural_segment_m)
    max_window_md = float(main.md_t1_m) - float(config.min_structural_segment_m)
    for candidate_md in np.arange(
        min_window_md,
        max_window_md + 1e-9,
        float(config.md_step_control_m),
    ):
        row = pilot_wells._interpolate_main_bore_window_by_md(
            main.stations,
            float(candidate_md),
            main_md_values,
        )
        candidate_point = Point3D(
            x=float(row["X_m"]),
            y=float(row["Y_m"]),
            z=float(row["Z_m"]),
        )
        if candidate_point.z >= pilot_targets[1].z:
            continue
        try:
            tail = pilot_wells._exact_pilot_tail_geometry(
                start=candidate_point,
                start_inc_deg=float(row["INC_deg"]),
                start_azi_deg=float(row["AZI_deg"]),
                study_points=pilot_targets[1:],
                config=config,
            )
        except ValueError:
            continue
        if tail.first_leg_md_m < float(config.min_structural_segment_m) - 1e-9:
            continue
        grid_scores.append(main_md + float(tail.extra_md_m))
    assert grid_scores
    assert planned.total_drilled_md_m <= min(grid_scores) + 1e-6
    for candidate_md in (300.0, 500.0, 700.0, 900.0, 1000.0):
        candidate = plan_pilot_from_main_bore(
            pilot_name="WELL-04_PL",
            parent_name="WELL-04",
            pilot_target_points=pilot_targets,
            main_bore=main,
            pilot_config=config,
            main_config=config,
            window_override=SidetrackWindowOverride(kind="md", value_m=candidate_md),
        )
        assert planned.total_drilled_md_m <= candidate.total_drilled_md_m + 1e-6
    selected_by_z = plan_pilot_from_main_bore(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_target_points=pilot_targets,
        main_bore=main,
        pilot_config=config,
        main_config=config,
        window_override=SidetrackWindowOverride(
            kind="z", value_m=float(planned.window.point.z)
        ),
    )
    assert selected_by_z.window.md_m == pytest.approx(planned.window.md_m, abs=1e-5)
    assert float(planned.pilot.stations["X_m"].iloc[-1]) == pytest.approx(
        pilot_targets[-1].x, abs=1e-6
    )
    assert float(planned.pilot.stations["Y_m"].iloc[-1]) == pytest.approx(
        pilot_targets[-1].y, abs=1e-6
    )
    assert float(planned.pilot.stations["Z_m"].iloc[-1]) == pytest.approx(
        pilot_targets[-1].z, abs=1e-6
    )


def test_pilot_window_refinement_keeps_each_disconnected_local_minimum() -> None:
    candidates = (
        (10.0, 105.0, 80.0),
        (12.0, 100.0, 75.0),
        (14.0, 103.0, 70.0),
        (30.0, 99.0, 60.0),
        (32.0, 101.0, 55.0),
    )

    assert pilot_wells._pilot_window_local_minimum_mds(
        candidates,
        control_step_m=2.0,
    ) == {12.0, 30.0}


def test_joint_pilot_tail_optimization_can_trade_first_leg_md_for_shorter_total() -> (
    None
):
    pilot_config = TrajectoryConfig(
        dls_build_max_deg_per_30m=8.0,
        max_inc_deg=110.0,
        min_structural_segment_m=30.0,
    )
    main_config = pilot_config.validated_copy(dls_build_max_deg_per_30m=30.0)
    surface = Point3D(0.0, 0.0, 0.0)
    window_inc_deg = 75.00669426993768
    window_azi_deg = 342.4511630309409
    main_stations = compute_positions_min_curv(
        pd.DataFrame(
            {
                "MD_m": [0.0, 100.0, 200.0, 500.0, 800.0],
                "INC_deg": [
                    0.0,
                    window_inc_deg,
                    window_inc_deg,
                    window_inc_deg,
                    window_inc_deg,
                ],
                "AZI_deg": [window_azi_deg] * 5,
                "segment": ["VERTICAL", "BUILD1", "HOLD", "HOLD", "HOLD"],
            }
        ),
        start=surface,
    )
    window_row = main_stations.iloc[2]
    window_point = Point3D(
        float(window_row["X_m"]),
        float(window_row["Y_m"]),
        float(window_row["Z_m"]),
    )
    targets = (
        Point3D(
            window_point.x - 188.77751367,
            window_point.y + 123.50831807,
            window_point.z + 335.33908228,
        ),
        Point3D(
            window_point.x - 55.19669031,
            window_point.y - 285.14800870,
            window_point.z + 508.49533983,
        ),
    )

    locally_shortest = pilot_wells._exact_pilot_tail_geometry(
        start=window_point,
        start_inc_deg=window_inc_deg,
        start_azi_deg=window_azi_deg,
        study_points=targets,
        config=pilot_config,
    )
    planned = plan_pilot_from_main_bore(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_target_points=(surface, *targets),
        main_bore=PlannerResult(
            stations=main_stations,
            summary={},
            azimuth_deg=window_azi_deg,
            md_t1_m=500.0,
        ),
        pilot_config=pilot_config,
        main_config=main_config,
        window_override=SidetrackWindowOverride(kind="md", value_m=200.0),
    )

    assert planned.pilot_tail_md_m < locally_shortest.extra_md_m - 70.0
    assert planned.pilot.summary["pilot_tail_optimization"] == "joint_min_total_md"
    selected_dls = tuple(
        float(value)
        for value in str(planned.pilot.summary["pilot_leg_dls_deg_per_30m"]).split("|")
    )
    assert selected_dls[0] < 8.0
    assert selected_dls[-1] == pytest.approx(8.0)
    assert float(planned.pilot.stations["X_m"].iloc[-1]) == pytest.approx(targets[-1].x)
    assert float(planned.pilot.stations["Y_m"].iloc[-1]) == pytest.approx(targets[-1].y)
    assert float(planned.pilot.stations["Z_m"].iloc[-1]) == pytest.approx(targets[-1].z)


def test_joint_pilot_tail_optimizer_finds_narrow_feasible_dls_interval() -> None:
    config = TrajectoryConfig(
        dls_build_max_deg_per_30m=8.0,
        max_inc_deg=110.0,
        min_structural_segment_m=30.0,
    )
    targets = (
        Point3D(-303.57411193452833, 427.3734121642668, 874.1279410834087),
        Point3D(356.19494471139217, 177.120694277422, 985.8026094430365),
    )

    geometry = pilot_wells._optimized_pilot_tail_geometry(
        start=Point3D(0.0, 0.0, 0.0),
        start_inc_deg=11.164968644009798,
        start_azi_deg=264.37435438048266,
        study_points=targets,
        config=config,
    )

    assert 1.5 < geometry.dls_deg_per_30m[0] < 1.75
    assert geometry.dls_deg_per_30m[1] == pytest.approx(8.0)
    assert geometry.extra_md_m > 0.0


def test_joint_pilot_tail_optimizer_honors_dls_limits_below_point_one(
    monkeypatch,
) -> None:
    observed_dls: list[tuple[float, ...]] = []

    def fake_exact_tail_geometry(*, dls_values_deg_per_30m, **_kwargs):
        values = tuple(float(value) for value in dls_values_deg_per_30m)
        observed_dls.append(values)
        return pilot_wells._PilotTailGeometry(
            extra_md_m=float(sum(values)),
            first_leg_md_m=100.0,
            dls_deg_per_30m=values,
        )

    monkeypatch.setattr(
        pilot_wells,
        "_exact_pilot_tail_geometry",
        fake_exact_tail_geometry,
    )
    config = TrajectoryConfig(
        dls_build_min_deg_per_30m=0.02,
        dls_build_max_deg_per_30m=0.2,
        max_inc_deg=110.0,
        min_structural_segment_m=30.0,
    )

    geometry = pilot_wells._optimized_pilot_tail_geometry(
        start=Point3D(0.0, 0.0, 0.0),
        start_inc_deg=0.0,
        start_azi_deg=0.0,
        study_points=(Point3D(0.0, 0.0, 100.0), Point3D(0.0, 0.0, 200.0)),
        config=config,
    )

    assert observed_dls
    assert any(values[0] < 0.1 for values in observed_dls)
    assert config.dls_build_min_deg_per_30m <= geometry.dls_deg_per_30m[0] < 0.1
    assert geometry.dls_deg_per_30m[-1] == pytest.approx(
        config.dls_build_max_deg_per_30m
    )


def test_exact_pilot_geometry_rejects_mcm_singular_dogleg() -> None:
    target_vector = np.asarray(
        [131.44400146, -195.49457057, 525.83784028],
        dtype=float,
    )

    geometry = pilot_wells._exact_build_hold_geometry(
        target_vector=target_vector,
        start_inc_deg=103.29697547505805,
        start_azi_deg=264.17956396210695,
        dls_deg_per_30m=6.502631882288366,
        max_inc_deg=120.0,
    )

    assert geometry is None


def test_pilot_dls_check_does_not_apply_classical_build2_limit() -> None:
    stations = pd.DataFrame(
        {
            "DLS_deg_per_30m": [6.0, 6.0, 1.5],
            "segment": ["PILOT_BUILD_1", "PILOT_BUILD_2", "PILOT_HOLD_2"],
        }
    )
    config = TrajectoryConfig(
        dls_build_max_deg_per_30m=6.0,
        dls_build2_max_deg_per_30m=2.0,
    )

    assert pilot_wells._max_dls_limit_excess(stations, config) == pytest.approx(0.0)


def test_joint_window_optimizer_counts_tail_from_window_only(monkeypatch) -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0],
            "INC_deg": [90.0, 90.0],
            "AZI_deg": [90.0, 90.0],
            "X_m": [0.0, 1000.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 0.0],
        }
    )

    def fake_tail_geometry(*, start, **_kwargs):
        # The tail is already measured from the window; subtracting window MD
        # again would incorrectly prefer a longer tail at the deeper window.
        window_md = float(start.x)
        return pilot_wells._PilotTailGeometry(
            extra_md_m=1000.0 + 0.2 * window_md,
            first_leg_md_m=100.0,
            dls_deg_per_30m=(2.0, 2.0),
        )

    monkeypatch.setattr(
        pilot_wells,
        "_exact_pilot_tail_geometry",
        fake_tail_geometry,
    )
    config = TrajectoryConfig(
        md_step_control_m=100.0,
        md_step_m=100.0,
        dls_build_max_deg_per_30m=2.0,
        min_structural_segment_m=100.0,
    )

    candidate = pilot_wells._optimized_pilot_window_tail_geometry(
        main_stations=stations,
        md_values=stations["MD_m"].to_numpy(dtype=float),
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        min_window_md_m=100.0,
        max_window_md_m=900.0,
        study_points=(Point3D(0.0, 0.0, 1000.0), Point3D(0.0, 0.0, 2000.0)),
        config=config,
        seed_window_mds=(100.0, 900.0),
    )

    assert float(candidate.window.md_m) == pytest.approx(100.0, abs=1.0)
    assert 1000.0 + candidate.tail.extra_md_m == pytest.approx(
        2000.0 + 0.2 * float(candidate.window.md_m)
    )


def test_joint_window_optimizer_handles_three_pl_points_in_one_global_search() -> None:
    surface = Point3D(0.0, 0.0, 0.0)
    main_stations = compute_positions_min_curv(
        pd.DataFrame(
            {
                "MD_m": [0.0, 100.0, 200.0, 500.0, 800.0],
                "INC_deg": [0.0, 30.0, 30.0, 30.0, 30.0],
                "AZI_deg": [90.0] * 5,
                "segment": ["VERTICAL", "BUILD1", "HOLD", "HOLD", "HOLD"],
            }
        ),
        start=surface,
    )
    reference = main_stations.iloc[2]
    targets = (
        Point3D(
            float(reference["X_m"]) + 100.0,
            float(reference["Y_m"]),
            float(reference["Z_m"]) + 400.0,
        ),
        Point3D(
            float(reference["X_m"]) + 250.0,
            float(reference["Y_m"]) + 50.0,
            float(reference["Z_m"]) + 800.0,
        ),
        Point3D(
            float(reference["X_m"]) + 450.0,
            float(reference["Y_m"]),
            float(reference["Z_m"]) + 1200.0,
        ),
    )
    pilot_config = TrajectoryConfig(
        dls_build_max_deg_per_30m=8.0,
        max_inc_deg=110.0,
        min_structural_segment_m=30.0,
        md_step_control_m=5.0,
    )

    planned = plan_pilot_from_main_bore(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_target_points=(surface, *targets),
        main_bore=PlannerResult(
            stations=main_stations,
            summary={},
            azimuth_deg=90.0,
            md_t1_m=500.0,
        ),
        pilot_config=pilot_config,
        main_config=pilot_config.validated_copy(dls_build_max_deg_per_30m=30.0),
    )

    assert float(planned.window.md_m) < 470.0
    assert planned.pilot.summary["pilot_target_count"] == pytest.approx(3.0)
    assert len(str(planned.pilot.summary["pilot_leg_dls_deg_per_30m"]).split("|")) == 3
    assert planned.total_drilled_md_m == pytest.approx(800.0 + planned.pilot_tail_md_m)
    assert float(planned.pilot.stations["X_m"].iloc[-1]) == pytest.approx(targets[-1].x)
    assert float(planned.pilot.stations["Y_m"].iloc[-1]) == pytest.approx(targets[-1].y)
    assert float(planned.pilot.stations["Z_m"].iloc[-1]) == pytest.approx(targets[-1].z)


def test_pilot_from_main_bore_rejects_infeasible_manual_window() -> None:
    config = TrajectoryConfig(
        dls_build_max_deg_per_30m=3.0,
        dls_horizontal_max_deg_per_30m=3.0,
        turn_solver_max_restarts=0,
    )
    surface = Point3D(0.0, 0.0, 0.0)
    main = pilot_wells.TrajectoryPlanner().plan(
        surface=surface,
        t1=Point3D(800.0, 0.0, 2200.0),
        t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
    )

    with pytest.raises(ValueError, match="не дало буримую траекторию"):
        plan_pilot_from_main_bore(
            pilot_name="WELL-04_PL",
            parent_name="WELL-04",
            pilot_target_points=(
                surface,
                Point3D(0.0, 0.0, 800.0),
                Point3D(200.0, 0.0, 1300.0),
            ),
            main_bore=main,
            pilot_config=config,
            main_config=config,
            window_override=SidetrackWindowOverride(kind="md", value_m=760.0),
        )


def test_main_bore_window_md_interpolation_uses_minimum_curvature() -> None:
    survey = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 200.0],
            "INC_deg": [0.0, 30.0, 60.0],
            "AZI_deg": [0.0, 0.0, 0.0],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 0.0, 0.0],
            "segment": ["VERTICAL", "BUILD1", "BUILD1"],
        }
    )
    survey = compute_positions_min_curv(survey, start=Point3D(0.0, 0.0, 0.0))
    row = pilot_wells._interpolate_main_bore_window_by_md(
        survey,
        150.0,
        survey["MD_m"].to_numpy(dtype=float),
    )
    expected = compute_positions_min_curv(
        pd.DataFrame(
            {
                "MD_m": [100.0, 150.0],
                "INC_deg": [30.0, 45.0],
                "AZI_deg": [0.0, 0.0],
            }
        ),
        start=Point3D(
            x=float(survey.loc[1, "X_m"]),
            y=float(survey.loc[1, "Y_m"]),
            z=float(survey.loc[1, "Z_m"]),
        ),
    )
    assert float(row["INC_deg"]) == pytest.approx(45.0)
    assert float(row["X_m"]) == pytest.approx(float(expected["X_m"].iloc[-1]))
    assert float(row["Y_m"]) == pytest.approx(float(expected["Y_m"].iloc[-1]))
    assert float(row["Z_m"]) == pytest.approx(float(expected["Z_m"].iloc[-1]))


def test_main_bore_window_z_interpolation_finds_interior_depth_crossing() -> None:
    survey = compute_positions_min_curv(
        pd.DataFrame(
            {
                "MD_m": [0.0, 200.0],
                "INC_deg": [80.0, 100.0],
                "AZI_deg": [0.0, 0.0],
                "segment": ["BUILD1", "BUILD1"],
            }
        ),
        start=Point3D(0.0, 0.0, 0.0),
    )
    md_values = survey["MD_m"].to_numpy(dtype=float)

    row = pilot_wells._interpolate_main_bore_window_by_z(
        survey,
        5.0,
        md_values,
    )

    assert 0.0 < float(row["MD_m"]) < 100.0
    assert float(row["Z_m"]) == pytest.approx(5.0, abs=1e-6)


def test_sidetrack_from_pilot_window_preserves_pose_and_first_dogleg_limit() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=3.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )
    window_row = pilot.stations.loc[
        (pilot.stations["segment"] == "PILOT_BUILD_2")
        & (pilot.stations["INC_deg"] >= 7.0)
    ].iloc[0]
    window = PilotWindow.from_station(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        row=window_row,
    )
    t1 = Point3D(
        float(window.point.x) + 800.0,
        float(window.point.y),
        float(window.point.z) + 900.0,
    )
    t3 = Point3D(float(t1.x) + 1000.0, float(t1.y), float(t1.z))

    result = SidetrackPlanner().plan(
        start=SidetrackStart(
            point=window.point,
            inc_deg=window.inc_deg,
            azi_deg=window.azi_deg,
        ),
        t1=t1,
        t3=t3,
        config=config,
    )

    first_station = result.stations.iloc[0]
    assert float(first_station["X_m"]) == pytest.approx(float(window.point.x))
    assert float(first_station["Y_m"]) == pytest.approx(float(window.point.y))
    assert float(first_station["Z_m"]) == pytest.approx(float(window.point.z))
    assert float(first_station["INC_deg"]) == pytest.approx(float(window.inc_deg))
    assert float(first_station["AZI_deg"]) == pytest.approx(float(window.azi_deg))
    first_dls = float(result.stations["DLS_deg_per_30m"].dropna().iloc[0])
    assert first_dls <= float(config.dls_build_max_deg_per_30m) + 1e-6


def test_sidetrack_window_is_selected_50_to_100m_above_first_pilot_target() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=3.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )

    window, _result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(800.0, 0.0, 2200.0),
        parent_t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
        planner=object(),
    )

    offset_m = float(pilot.md_first_target_m) - float(window.md_m)
    assert 50.0 <= offset_m <= 100.0


def test_sidetrack_window_search_expands_after_preferred_group_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=900.0,
        point=Point3D(0.0, 0.0, 900.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    expanded = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=600.0,
        point=Point3D(0.0, 0.0, 600.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    calls: list[float] = []

    monkeypatch.setattr(
        pilot_wells,
        "_sidetrack_window_candidate_groups",
        lambda **_kwargs: [[preferred], [expanded]],
    )
    real_plan = SidetrackPlanner().plan

    def fake_plan(self: SidetrackPlanner, **kwargs: object):
        start = kwargs["start"]
        calls.append(float(start.point.z))
        if float(start.point.z) == pytest.approx(900.0):
            raise PlanningError("preferred failed")
        return real_plan(**kwargs)

    monkeypatch.setattr(SidetrackPlanner, "plan", fake_plan)

    selected, result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pd.DataFrame({"MD_m": [0.0]}),
        parent_t1=Point3D(500.0, 0.0, 1200.0),
        parent_t3=Point3D(1500.0, 0.0, 1200.0),
        config=TrajectoryConfig(
            md_step_m=25.0,
            dls_build_max_deg_per_30m=6.0,
            max_inc_deg=120.0,
        ),
        planner=object(),
    )

    assert selected == expanded
    assert calls == [900.0, 600.0]
    assert float(result.summary["distance_t1_m"]) == pytest.approx(0.0)


def test_sidetrack_window_search_rejects_candidate_after_full_sequence_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=900.0,
        point=Point3D(0.0, 0.0, 900.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    expanded = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=600.0,
        point=Point3D(0.0, 0.0, 600.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    callback_calls: list[float] = []

    def local_result(point: Point3D) -> PlannerResult:
        return PlannerResult(
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 100.0],
                    "INC_deg": [0.0, 0.0],
                    "AZI_deg": [0.0, 0.0],
                    "X_m": [float(point.x), float(point.x)],
                    "Y_m": [float(point.y), float(point.y)],
                    "Z_m": [float(point.z), float(point.z) + 100.0],
                    "DLS_deg_per_30m": [0.0, 0.0],
                }
            ),
            summary={
                "md_total_m": 100.0,
                "max_dls_total_deg_per_30m": 0.0,
                "build_dls_max_config_deg_per_30m": 10.0,
            },
            azimuth_deg=0.0,
            md_t1_m=50.0,
        )

    def complete_result(window: PilotWindow) -> PlannerResult:
        return PlannerResult(
            stations=pd.DataFrame(
                {
                    "MD_m": [float(window.md_m), float(window.md_m) + 200.0],
                    "INC_deg": [0.0, 0.0],
                    "AZI_deg": [0.0, 0.0],
                    "X_m": [float(window.point.x), float(window.point.x)],
                    "Y_m": [float(window.point.y), float(window.point.y)],
                    "Z_m": [float(window.point.z), float(window.point.z) + 200.0],
                    "DLS_deg_per_30m": [0.0, 0.0],
                }
            ),
            summary={
                "md_total_m": float(window.md_m) + 200.0,
                "max_dls_total_deg_per_30m": 0.0,
                "build_dls_max_config_deg_per_30m": 10.0,
            },
            azimuth_deg=0.0,
            md_t1_m=float(window.md_m) + 100.0,
        )

    monkeypatch.setattr(
        pilot_wells,
        "_sidetrack_window_candidate_groups",
        lambda **_kwargs: [[preferred], [expanded]],
    )

    def fake_plan(self: SidetrackPlanner, **kwargs: object) -> PlannerResult:
        start = kwargs["start"]
        return local_result(start.point)

    monkeypatch.setattr(SidetrackPlanner, "plan", fake_plan)

    def validate_full_candidate(
        _pilot_stations: pd.DataFrame,
        window: PilotWindow,
        _sidetrack_result: PlannerResult,
    ) -> PlannerResult:
        callback_calls.append(float(window.md_m))
        if window == preferred:
            raise PlanningError("полная последовательность не проходит")
        return complete_result(window)

    selected, result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pd.DataFrame({"MD_m": [0.0]}),
        parent_t1=Point3D(500.0, 0.0, 1200.0),
        parent_t3=Point3D(1500.0, 0.0, 1200.0),
        config=TrajectoryConfig(
            md_step_m=25.0,
            dls_build_max_deg_per_30m=6.0,
            max_inc_deg=120.0,
        ),
        planner=object(),
        candidate_validator=validate_full_candidate,
    )

    assert selected == expanded
    assert callback_calls == [900.0, 600.0]
    assert float(result.summary["md_total_m"]) == pytest.approx(100.0)


def test_sidetrack_window_score_uses_complete_lateral_md(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=600.0,
        point=Point3D(0.0, 0.0, 600.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    second = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=700.0,
        point=Point3D(0.0, 0.0, 700.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )

    def fake_plan(self: SidetrackPlanner, **kwargs: object) -> PlannerResult:
        start = kwargs["start"]
        window = start.point
        return PlannerResult(
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 100.0],
                    "INC_deg": [0.0, 0.0],
                    "AZI_deg": [0.0, 0.0],
                    "X_m": [float(window.x), float(window.x)],
                    "Y_m": [float(window.y), float(window.y)],
                    "Z_m": [float(window.z), float(window.z) + 100.0],
                    "DLS_deg_per_30m": [0.0, 0.0],
                }
            ),
            summary={
                "md_total_m": 100.0,
                "max_dls_total_deg_per_30m": 0.0,
                "build_dls_max_config_deg_per_30m": 10.0,
            },
            azimuth_deg=0.0,
            md_t1_m=50.0,
        )

    def complete_result(window: PilotWindow, lateral_md_m: float) -> PlannerResult:
        return PlannerResult(
            stations=pd.DataFrame(
                {
                    "MD_m": [
                        float(window.md_m),
                        float(window.md_m) + float(lateral_md_m),
                    ],
                    "INC_deg": [0.0, 0.0],
                    "AZI_deg": [0.0, 0.0],
                    "X_m": [float(window.point.x), float(window.point.x)],
                    "Y_m": [float(window.point.y), float(window.point.y)],
                    "Z_m": [
                        float(window.point.z),
                        float(window.point.z) + float(lateral_md_m),
                    ],
                    "DLS_deg_per_30m": [0.0, 0.0],
                }
            ),
            summary={
                "md_total_m": float(window.md_m) + float(lateral_md_m),
                "max_dls_total_deg_per_30m": 0.0,
                "build_dls_max_config_deg_per_30m": 10.0,
            },
            azimuth_deg=0.0,
            md_t1_m=float(window.md_m) + float(lateral_md_m) / 2.0,
        )

    monkeypatch.setattr(
        pilot_wells,
        "_sidetrack_window_candidate_groups",
        lambda **_kwargs: [[first, second]],
    )
    monkeypatch.setattr(SidetrackPlanner, "plan", fake_plan)

    selected, _result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pd.DataFrame({"MD_m": [0.0]}),
        parent_t1=Point3D(500.0, 0.0, 1200.0),
        parent_t3=Point3D(1500.0, 0.0, 1200.0),
        config=TrajectoryConfig(
            md_step_m=25.0,
            dls_build_max_deg_per_30m=6.0,
            max_inc_deg=120.0,
        ),
        planner=object(),
        candidate_validator=lambda _pilot, window, _local: complete_result(
            window,
            500.0 if window == first else 100.0,
        ),
    )

    assert selected == second


def test_sidetrack_window_search_optimizes_across_preferred_and_expanded_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=900.0,
        point=Point3D(0.0, 0.0, 900.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    expanded = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=600.0,
        point=Point3D(0.0, 0.0, 600.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )

    def result_for(window: PilotWindow, *, local: bool) -> PlannerResult:
        lateral_md_m = 100.0 if local or window == expanded else 500.0
        start_md_m = 0.0 if local else float(window.md_m)
        return PlannerResult(
            stations=pd.DataFrame(
                {
                    "MD_m": [start_md_m, start_md_m + lateral_md_m],
                    "INC_deg": [0.0, 0.0],
                    "AZI_deg": [0.0, 0.0],
                    "X_m": [float(window.point.x), float(window.point.x)],
                    "Y_m": [float(window.point.y), float(window.point.y)],
                    "Z_m": [
                        float(window.point.z),
                        float(window.point.z) + lateral_md_m,
                    ],
                    "DLS_deg_per_30m": [0.0, 0.0],
                }
            ),
            summary={
                "md_total_m": start_md_m + lateral_md_m,
                "max_dls_total_deg_per_30m": 0.0,
                "build_dls_max_config_deg_per_30m": 10.0,
            },
            azimuth_deg=0.0,
            md_t1_m=start_md_m + lateral_md_m / 2.0,
        )

    monkeypatch.setattr(
        pilot_wells,
        "_sidetrack_window_candidate_groups",
        lambda **_kwargs: [[preferred], [expanded]],
    )

    def fake_plan(self: SidetrackPlanner, **kwargs: object) -> PlannerResult:
        start = kwargs["start"]
        window = preferred if float(start.point.z) == 900.0 else expanded
        return result_for(window, local=True)

    monkeypatch.setattr(SidetrackPlanner, "plan", fake_plan)

    selected, _ = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pd.DataFrame({"MD_m": [0.0]}),
        parent_t1=Point3D(0.0, 0.0, 1000.0),
        parent_t3=Point3D(0.0, 0.0, 1100.0),
        config=TrajectoryConfig(),
        planner=object(),
        candidate_validator=lambda _pilot, window, _local: result_for(
            window,
            local=False,
        ),
    )

    assert selected == expanded


def test_sidetrack_window_search_skips_nonfinite_candidate_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=900.0,
        point=Point3D(0.0, 0.0, 900.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    expanded = PilotWindow(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        md_m=600.0,
        point=Point3D(0.0, 0.0, 600.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    malformed = PlannerResult(
        stations=pd.DataFrame(),
        summary={"md_total_m": 100.0},
        azimuth_deg=0.0,
        md_t1_m=50.0,
    )
    valid = PlannerResult(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 100.0],
                "INC_deg": [0.0, 0.0],
                "AZI_deg": [0.0, 0.0],
                "X_m": [0.0, 0.0],
                "Y_m": [0.0, 0.0],
                "Z_m": [600.0, 700.0],
                "DLS_deg_per_30m": [np.nan, 0.0],
            }
        ),
        summary={"md_total_m": 100.0},
        azimuth_deg=0.0,
        md_t1_m=50.0,
    )
    calls: list[float] = []

    monkeypatch.setattr(
        pilot_wells,
        "_sidetrack_window_candidate_groups",
        lambda **_kwargs: [[preferred], [expanded]],
    )

    def fake_plan(self: SidetrackPlanner, **kwargs: object) -> PlannerResult:
        start = kwargs["start"]
        calls.append(float(start.point.z))
        return malformed if float(start.point.z) == 900.0 else valid

    monkeypatch.setattr(SidetrackPlanner, "plan", fake_plan)

    selected, result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pd.DataFrame({"MD_m": [0.0]}),
        parent_t1=Point3D(0.0, 0.0, 800.0),
        parent_t3=Point3D(0.0, 0.0, 900.0),
        config=TrajectoryConfig(),
        planner=object(),
    )

    assert selected == expanded
    assert result is valid
    assert calls == [900.0, 600.0]


def test_single_point_pilot_window_minimizes_sidetrack_md() -> None:
    config = TrajectoryConfig()
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="well_04_PL",
            points=(
                WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0),
                WelltrackPoint(x=457667.0, y=889821.0, z=2554.0, md=2.0),
            ),
        ),
        config=config,
    )

    window, result = select_sidetrack_window(
        pilot_name="well_04_PL",
        parent_name="well_04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(458200.0, 888775.0, 2452.0),
        parent_t3=Point3D(459130.0, 887003.0, 2554.0),
        config=config,
        planner=object(),
    )

    assert float(window.md_m) > 2000.0
    assert float(window.point.z) > 1700.0
    assert float(result.summary["md_total_m"]) < 5000.0
    assert float(result.summary["max_dls_total_deg_per_30m"]) <= 1.5


def test_two_point_welltracks4_pilot_keeps_window_near_first_study_point() -> None:
    config = TrajectoryConfig()
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="well_04_PL",
            points=(
                WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0),
                WelltrackPoint(x=457653.0, y=890180.0, z=1821.0, md=2.0),
                WelltrackPoint(x=457667.0, y=889821.0, z=2554.0, md=3.0),
            ),
        ),
        config=config,
    )

    window, result = select_sidetrack_window(
        pilot_name="well_04_PL",
        parent_name="well_04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(458200.0, 888775.0, 2452.0),
        parent_t3=Point3D(459130.0, 887003.0, 2554.0),
        config=config,
        planner=object(),
    )

    offset_m = float(pilot.md_first_target_m) - float(window.md_m)
    assert 50.0 <= offset_m <= 100.0
    assert 1750.0 <= float(window.point.z) <= 1850.0
    assert float(result.summary["md_total_m"]) < 3800.0
    assert float(result.summary["max_dls_total_deg_per_30m"]) <= 1.1


def test_three_point_pilot_window_keeps_sidetrack_near_first_study_point() -> None:
    config = TrajectoryConfig()
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="well_04_PL",
            points=(
                WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0),
                WelltrackPoint(x=457500.0, y=890600.0, z=1200.0, md=2.0),
                WelltrackPoint(x=457653.0, y=890180.0, z=1821.0, md=3.0),
                WelltrackPoint(x=457667.0, y=889821.0, z=2554.0, md=4.0),
            ),
        ),
        config=config,
    )

    window, result = select_sidetrack_window(
        pilot_name="well_04_PL",
        parent_name="well_04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(458200.0, 888775.0, 2452.0),
        parent_t3=Point3D(459130.0, 887003.0, 2554.0),
        config=config,
        planner=object(),
    )

    offset_m = float(pilot.md_first_target_m) - float(window.md_m)
    assert 50.0 <= offset_m <= 100.0
    assert 1100.0 <= float(window.point.z) <= 1200.0
    assert float(result.summary["md_total_m"]) < 4600.0
    assert float(result.summary["max_dls_total_deg_per_30m"]) <= 1.5


def test_manual_sidetrack_window_md_override_interpolates_pilot_pose() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=3.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )
    manual_md = float(pilot.md_first_target_m) - 63.5

    window, result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(800.0, 0.0, 2200.0),
        parent_t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
        planner=object(),
        window_override=SidetrackWindowOverride(kind="md", value_m=manual_md),
    )

    first_station = result.stations.iloc[0]
    assert float(window.md_m) == pytest.approx(manual_md)
    assert float(first_station["X_m"]) == pytest.approx(float(window.point.x))
    assert float(first_station["Y_m"]) == pytest.approx(float(window.point.y))
    assert float(first_station["Z_m"]) == pytest.approx(float(window.point.z))
    assert float(first_station["INC_deg"]) == pytest.approx(float(window.inc_deg))
    assert float(first_station["AZI_deg"]) == pytest.approx(float(window.azi_deg))


def test_manual_sidetrack_window_z_override_interpolates_pilot_pose() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=3.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )
    manual_z = 760.0

    window, _result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(800.0, 0.0, 2200.0),
        parent_t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
        planner=object(),
        window_override=SidetrackWindowOverride(kind="z", value_m=manual_z),
    )

    assert float(window.point.z) == pytest.approx(manual_z)
    assert 0.0 < float(window.md_m) < float(pilot.md_total_m)


def test_manual_sidetrack_window_rejects_out_of_range_value() -> None:
    config = TrajectoryConfig(md_step_m=25.0, kop_min_vertical_m=200.0)
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
            ),
        ),
        config=config,
    )

    with pytest.raises(ValueError, match="вне диапазона пилота"):
        select_sidetrack_window(
            pilot_name="WELL-04_PL",
            parent_name="WELL-04",
            pilot_stations=pilot.stations,
            parent_t1=Point3D(800.0, 0.0, 2200.0),
            parent_t3=Point3D(1800.0, 0.0, 2200.0),
            config=config,
            planner=object(),
            window_override=SidetrackWindowOverride(kind="md", value_m=-1.0),
        )


@pytest.mark.parametrize(
    "md_values",
    ([0.0, 100.0, 50.0], [0.0, 100.0, 100.0]),
)
def test_sidetrack_window_search_rejects_nonmonotonic_pilot_md(
    md_values: list[float],
) -> None:
    stations = pd.DataFrame(
        {
            "MD_m": md_values,
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 100.0, 200.0],
            "INC_deg": [0.0, 0.0, 0.0],
            "AZI_deg": [0.0, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="не возрастающий MD"):
        pilot_wells._sidetrack_window_candidate_groups(
            pilot_name="WELL-04_PL",
            parent_name="WELL-04",
            pilot_stations=stations,
            parent_t1=Point3D(0.0, 0.0, 1000.0),
            config=TrajectoryConfig(),
        )


def test_manual_sidetrack_window_rejects_nonmonotonic_pilot_md() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 100.0],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 100.0, 200.0],
            "INC_deg": [0.0, 0.0, 0.0],
            "AZI_deg": [0.0, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="строго возрастающим"):
        pilot_wells._manual_sidetrack_window(
            pilot_name="WELL-04_PL",
            parent_name="WELL-04",
            pilot_stations=stations,
            override=SidetrackWindowOverride(kind="md", value_m=50.0),
        )


def test_sidetrack_uncertainty_at_window_inherits_pilot_covariance() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=3.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )
    window, sidetrack_result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(800.0, 0.0, 2200.0),
        parent_t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
        planner=object(),
    )
    sidetrack = combine_pilot_and_sidetrack(
        pilot_stations=pilot.stations,
        sidetrack_result=sidetrack_result,
        window=window,
        config=config,
    )
    model = planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET)
    sample_md = np.asarray([float(window.md_m)], dtype=float)

    pilot_covariance = station_uncertainty_covariance_samples_for_stations(
        stations=pilot.stations,
        sample_md_m=sample_md,
        model=model,
    ).covariance_xyz[0]
    sidetrack_covariance = station_uncertainty_covariance_samples_for_stations(
        stations=sidetrack.stations,
        sample_md_m=sample_md,
        model=model,
    ).covariance_xyz[0]

    np.testing.assert_allclose(
        sidetrack_covariance,
        pilot_covariance,
        atol=1e-9,
        rtol=1e-9,
    )


def test_sidetrack_solver_preserves_window_pose_and_hits_t1_t3() -> None:
    result = SidetrackPlanner().plan(
        start=SidetrackStart(
            point=Point3D(0.0, 0.0, 1000.0),
            inc_deg=30.0,
            azi_deg=90.0,
        ),
        t1=Point3D(500.0, 100.0, 1500.0),
        t3=Point3D(1500.0, 100.0, 1500.0),
        config=TrajectoryConfig(
            md_step_m=25.0,
            dls_build_max_deg_per_30m=6.0,
            max_inc_deg=100.0,
        ),
    )

    assert float(result.stations.loc[0, "INC_deg"]) == pytest.approx(30.0)
    assert float(result.stations.loc[0, "AZI_deg"]) == pytest.approx(90.0)
    assert float(result.summary["distance_t1_m"]) == pytest.approx(0.0)
    assert float(result.summary["distance_t3_m"]) == pytest.approx(0.0)
    assert result.summary["solver_strategy"] == "pilot_sidetrack_bezier"


def test_sidetrack_solver_uses_bend_fallback_when_single_bezier_misses_dls() -> None:
    start = SidetrackStart(
        point=Point3D(0.0, 0.0, 0.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )
    t1 = Point3D(500.0, 0.0, 500.0)
    t3 = Point3D(1500.0, 0.0, 500.0)
    config = TrajectoryConfig(
        md_step_m=25.0,
        dls_build_max_deg_per_30m=4.0,
        max_inc_deg=120.0,
    )

    result = SidetrackPlanner().plan(
        start=start,
        t1=t1,
        t3=t3,
        config=config,
    )

    assert result.summary["solver_strategy"] == "pilot_sidetrack_bend_fallback"
    assert result.summary["sidetrack_fallback_used"] == "yes"
    assert result.summary["build_dls_split_selected"] == "yes"
    assert float(result.summary["max_dls_total_deg_per_30m"]) <= 4.0 + 1e-6
    assert float(result.summary["max_inc_actual_deg"]) <= 120.0 + 1e-6
    assert float(result.summary["distance_t1_m"]) == pytest.approx(0.0)
    assert float(result.summary["distance_t3_m"]) == pytest.approx(0.0)
    first = result.stations.iloc[0]
    assert float(first["INC_deg"]) == pytest.approx(start.inc_deg)
    assert float(first["AZI_deg"]) == pytest.approx(start.azi_deg)
    assert "BUILD2" in set(result.stations["segment"])


def test_sidetrack_solver_rejects_over_limit_dls() -> None:
    with pytest.raises(PlanningError, match="ПИ бокового ствола превышает лимит"):
        SidetrackPlanner().plan(
            start=SidetrackStart(
                point=Point3D(0.0, 0.0, 0.0),
                inc_deg=0.0,
                azi_deg=0.0,
            ),
            t1=Point3D(50.0, 0.0, 50.0),
            t3=Point3D(500.0, 0.0, 50.0),
            config=TrajectoryConfig(
                md_step_m=25.0,
                dls_build_max_deg_per_30m=0.2,
                max_inc_deg=100.0,
            ),
        )


def test_reoriented_pilot_fallback_uses_standalone_geometry_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=100.0,
        dls_build_max_deg_per_30m=6.0,
        max_inc_deg=100.0,
    )
    surface = Point3D(0.0, 0.0, 0.0)
    pl1 = Point3D(0.0, 0.0, 800.0)
    parent_t1 = Point3D(400.0, 400.0, 1500.0)
    parent_t3 = Point3D(1200.0, 800.0, 1500.0)
    productive_target = Point3D(900.0, 900.0, 1500.0)
    standalone_result = PlannerResult(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 100.0, 200.0],
                "X_m": [0.0, 200.0, 400.0],
                "Y_m": [0.0, 200.0, 400.0],
                "Z_m": [0.0, 800.0, 1500.0],
                "INC_deg": [5.0, 25.0, 45.0],
                "AZI_deg": [135.0, 135.0, 135.0],
                "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
            }
        ),
        summary={},
        azimuth_deg=135.0,
        md_t1_m=150.0,
    )
    pilot_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 250.0, 880.0],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 800.0, 1200.0],
            "INC_deg": [0.0, 35.0, 40.0],
            "AZI_deg": [135.0, 135.0, 135.0],
            "DLS_deg_per_30m": [0.0, 1.0, 1.0],
            "segment": ["VERTICAL", "BUILD1", "HOLD"],
        }
    )
    pilot_result = pilot_wells.PilotBuildResult(
        stations=pilot_stations,
        surface=surface,
        first_target=pl1,
        final_target=pl1,
        md_first_target_m=250.0,
        md_total_m=880.0,
        azimuth_deg=135.0,
        summary={"max_dls_total_deg_per_30m": 1.0},
    )
    sidetrack_result = PlannerResult(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 320.0],
                "X_m": [0.0, 800.0],
                "Y_m": [0.0, 400.0],
                "Z_m": [0.0, 0.0],
                "INC_deg": [35.0, 90.0],
                "AZI_deg": [135.0, 135.0],
                "segment": ["BUILD1", "HORIZONTAL"],
            }
        ),
        summary={"md_total_m": 320.0, "max_dls_total_deg_per_30m": 1.5},
        azimuth_deg=135.0,
        md_t1_m=160.0,
    )
    window = PilotWindow(
        pilot_name="WELL_PL",
        parent_name="WELL",
        md_m=220.0,
        point=Point3D(0.0, 0.0, 760.0),
        inc_deg=35.0,
        azi_deg=135.0,
    )
    captured_hints: list[tuple[float, float]] = []

    class FakeTrajectoryPlanner:
        def plan(self, **kwargs: object) -> PlannerResult:
            return standalone_result

    class FakeSidetrackPlanner:
        def plan(self, **kwargs: object) -> PlannerResult:
            return sidetrack_result

    def fake_reoriented_pilot_candidates(**kwargs: object) -> list[object]:
        captured_hints.append(
            (
                float(kwargs["target_azimuth_deg"]),
                float(kwargs["terminal_inc_hint_deg"]),
            )
        )
        return [pilot_result]

    monkeypatch.setattr(pilot_wells, "TrajectoryPlanner", FakeTrajectoryPlanner)
    monkeypatch.setattr(pilot_wells, "SidetrackPlanner", FakeSidetrackPlanner)
    monkeypatch.setattr(
        pilot_wells,
        "_reoriented_pilot_candidates",
        fake_reoriented_pilot_candidates,
    )
    monkeypatch.setattr(
        pilot_wells,
        "_sidetrack_window_candidate_groups",
        lambda **kwargs: [[window]],
    )

    def validate_complete_candidate(
        _pilot_stations: pd.DataFrame,
        candidate_window: PilotWindow,
        _sidetrack_result: PlannerResult,
    ) -> PlannerResult:
        return PlannerResult(
            stations=pd.DataFrame(
                {
                    "MD_m": [
                        float(candidate_window.md_m),
                        float(candidate_window.md_m) + 320.0,
                    ],
                    "X_m": [
                        float(candidate_window.point.x),
                        float(candidate_window.point.x) + 800.0,
                    ],
                    "Y_m": [
                        float(candidate_window.point.y),
                        float(candidate_window.point.y) + 400.0,
                    ],
                    "Z_m": [
                        float(candidate_window.point.z),
                        float(candidate_window.point.z),
                    ],
                    "INC_deg": [
                        float(candidate_window.inc_deg),
                        90.0,
                    ],
                    "AZI_deg": [
                        float(candidate_window.azi_deg),
                        float(candidate_window.azi_deg),
                    ],
                }
            ),
            summary={"md_total_m": float(candidate_window.md_m) + 320.0},
            azimuth_deg=float(candidate_window.azi_deg),
            md_t1_m=float(candidate_window.md_m) + 160.0,
        )

    fallback = plan_reoriented_pilot_sidetrack_fallback(
        pilot_name="WELL_PL",
        parent_name="WELL",
        pilot_target_points=(surface, pl1),
        parent_t1=parent_t1,
        parent_t3=parent_t3,
        productive_direction_target=productive_target,
        pilot_config=config,
        sidetrack_config=config,
        candidate_validator=validate_complete_candidate,
    )

    assert captured_hints[0] == pytest.approx((135.0, 35.0))
    assert fallback.geometry_seed_source == "standalone_sidetrack"
    assert fallback.geometry_seed_azimuth_deg == pytest.approx(135.0)
    assert fallback.geometry_seed_inc_deg == pytest.approx(35.0)
    assert fallback.geometry_seed_md_total_m == pytest.approx(200.0)
    assert fallback.target_azimuth_deg == pytest.approx(135.0)
    assert fallback.total_drilled_md_m == pytest.approx(1200.0)
    assert fallback.sidetrack_lateral_md_m == pytest.approx(320.0)


def test_reoriented_pilot_candidates_remain_minimum_curvature_reconstructable() -> None:
    surface = Point3D(0.0, 0.0, 0.0)
    candidates = pilot_wells._reoriented_pilot_candidates(
        target_points=(
            surface,
            Point3D(0.0, 0.0, 800.0),
            Point3D(200.0, 0.0, 1300.0),
        ),
        target_azimuth_deg=90.0,
        terminal_inc_hint_deg=90.0,
        productive_direction_target=Point3D(1800.0, 0.0, 2200.0),
        parent_t1=Point3D(800.0, 0.0, 2200.0),
        config=TrajectoryConfig(
            md_step_m=25.0,
            kop_min_vertical_m=200.0,
            dls_build_max_deg_per_30m=12.0,
            max_inc_deg=110.0,
        ),
    )

    assert candidates
    for candidate in candidates:
        stations = candidate.stations
        reconstructed = compute_positions_min_curv(
            stations[["MD_m", "INC_deg", "AZI_deg"]],
            start=surface,
        )
        mismatch_m = np.linalg.norm(
            reconstructed[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=float)
            - stations[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=float),
            axis=1,
        )
        assert float(np.max(mismatch_m)) < 0.25


def test_reoriented_pilot_fallback_uses_direct_seed_when_standalone_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=100.0,
        dls_build_max_deg_per_30m=6.0,
        max_inc_deg=100.0,
    )
    surface = Point3D(0.0, 0.0, 0.0)
    pl1 = Point3D(0.0, 0.0, 800.0)
    parent_t1 = Point3D(400.0, 0.0, 1500.0)
    parent_t3 = Point3D(1200.0, 0.0, 1500.0)
    productive_target = Point3D(900.0, 0.0, 1500.0)
    pilot_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 800.0],
            "X_m": [0.0, 0.0],
            "Y_m": [0.0, 0.0],
            "Z_m": [0.0, 800.0],
            "INC_deg": [0.0, 90.0],
            "AZI_deg": [90.0, 90.0],
            "DLS_deg_per_30m": [0.0, 1.0],
        }
    )
    pilot_result = pilot_wells.PilotBuildResult(
        stations=pilot_stations,
        surface=surface,
        first_target=pl1,
        final_target=pl1,
        md_first_target_m=800.0,
        md_total_m=900.0,
        azimuth_deg=90.0,
        summary={"max_dls_total_deg_per_30m": 1.0},
    )
    sidetrack_result = PlannerResult(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 300.0],
                "X_m": [0.0, 800.0],
                "Y_m": [0.0, 0.0],
                "Z_m": [0.0, 0.0],
                "INC_deg": [90.0, 90.0],
                "AZI_deg": [90.0, 90.0],
            }
        ),
        summary={"md_total_m": 300.0, "max_dls_total_deg_per_30m": 1.0},
        azimuth_deg=90.0,
        md_t1_m=100.0,
    )
    window = PilotWindow(
        pilot_name="WELL_PL",
        parent_name="WELL",
        md_m=700.0,
        point=Point3D(0.0, 0.0, 700.0),
        inc_deg=90.0,
        azi_deg=90.0,
    )
    captured_hints: list[tuple[float, float]] = []

    class FailingTrajectoryPlanner:
        def plan(self, **kwargs: object) -> PlannerResult:
            raise KeyError("no standalone seed")

    class FakeSidetrackPlanner:
        def plan(self, **kwargs: object) -> PlannerResult:
            return sidetrack_result

    def fake_reoriented_pilot_candidates(**kwargs: object) -> list[object]:
        captured_hints.append(
            (
                float(kwargs["target_azimuth_deg"]),
                float(kwargs["terminal_inc_hint_deg"]),
            )
        )
        return [pilot_result]

    monkeypatch.setattr(pilot_wells, "TrajectoryPlanner", FailingTrajectoryPlanner)
    monkeypatch.setattr(pilot_wells, "SidetrackPlanner", FakeSidetrackPlanner)
    monkeypatch.setattr(
        pilot_wells,
        "_reoriented_pilot_candidates",
        fake_reoriented_pilot_candidates,
    )
    monkeypatch.setattr(
        pilot_wells,
        "_sidetrack_window_candidate_groups",
        lambda **kwargs: [[window]],
    )

    fallback = plan_reoriented_pilot_sidetrack_fallback(
        pilot_name="WELL_PL",
        parent_name="WELL",
        pilot_target_points=(surface, pl1),
        parent_t1=parent_t1,
        parent_t3=parent_t3,
        productive_direction_target=productive_target,
        pilot_config=config,
        sidetrack_config=config,
    )

    assert captured_hints == pytest.approx([(90.0, 90.0)])
    assert fallback.geometry_seed_source == "productive_direction"
    assert fallback.geometry_seed_azimuth_deg == pytest.approx(90.0)
    assert fallback.geometry_seed_inc_deg == pytest.approx(90.0)
    assert fallback.geometry_seed_md_total_m == pytest.approx(0.0)
    assert fallback.total_drilled_md_m == pytest.approx(1200.0)


def test_sidetrack_geometry_seed_falls_back_when_standalone_seed_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedTrajectoryPlanner:
        def plan(self, **kwargs: object) -> PlannerResult:
            return PlannerResult(
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0],
                        "INC_deg": [float("nan"), float("nan")],
                    }
                ),
                summary={"md_total_m": "bad", "entry_inc_deg": float("nan")},
                azimuth_deg=float("nan"),
                md_t1_m=None,
            )

    monkeypatch.setattr(pilot_wells, "TrajectoryPlanner", MalformedTrajectoryPlanner)

    seed = pilot_wells._sidetrack_geometry_seed(
        pilot_target_points=(Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 100.0)),
        parent_t1=Point3D(100.0, 0.0, 100.0),
        productive_direction_target=Point3D(300.0, 0.0, 100.0),
        sidetrack_config=TrajectoryConfig(md_step_m=25.0),
    )

    assert seed.source == "productive_direction"
    assert seed.azimuth_deg == pytest.approx(90.0)
    assert seed.inc_deg == pytest.approx(90.0)
    assert seed.md_total_m == pytest.approx(0.0)


def test_combine_pilot_and_sidetrack_uses_station_md_for_lateral_length() -> None:
    config = TrajectoryConfig(max_total_md_postcheck_m=500.0)
    pilot_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 500.0],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 100.0, 500.0],
            "INC_deg": [0.0, 20.0, 30.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "DLS_deg_per_30m": [0.0, 1.0, 1.0],
        }
    )
    sidetrack_result = PlannerResult(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 50.0, 240.0],
                "X_m": [0.0, 40.0, 230.0],
                "Y_m": [0.0, 0.0, 0.0],
                "Z_m": [100.0, 130.0, 130.0],
                "INC_deg": [20.0, 40.0, 90.0],
                "AZI_deg": [90.0, 90.0, 90.0],
                "DLS_deg_per_30m": [0.0, 1.0, 1.0],
            }
        ),
        summary={
            "md_total_m": 1.0,
            "max_dls_total_deg_per_30m": float("nan"),
            "kop_md_m": float("nan"),
        },
        azimuth_deg=90.0,
        md_t1_m=50.0,
    )
    window = PilotWindow(
        pilot_name="WELL_PL",
        parent_name="WELL",
        md_m=100.0,
        point=Point3D(0.0, 0.0, 100.0),
        inc_deg=20.0,
        azi_deg=90.0,
    )

    sidetrack = combine_pilot_and_sidetrack(
        pilot_stations=pilot_stations,
        sidetrack_result=sidetrack_result,
        window=window,
        config=config,
    )

    assert sidetrack.summary["sidetrack_lateral_md_m"] == pytest.approx(240.0)
    assert sidetrack.summary["pilot_total_md_m"] == pytest.approx(500.0)
    assert sidetrack.summary["total_drilled_md_m"] == pytest.approx(740.0)
    assert sidetrack.summary["total_drilled_footage_m"] == pytest.approx(740.0)
    assert sidetrack.summary["sidetrack_window_optimization_objective_m"] == (
        pytest.approx(740.0)
    )
    assert sidetrack.summary["md_total_m"] == pytest.approx(340.0)
    assert sidetrack.summary["sidetrack_total_md_m"] == pytest.approx(340.0)
    assert sidetrack.summary["md_postcheck_excess_m"] == pytest.approx(0.0)
    assert sidetrack.summary["md_postcheck_exceeded"] == "no"
    assert sidetrack.summary["kop_md_m"] == pytest.approx(100.0)
    assert sidetrack.summary["max_dls_total_deg_per_30m"] >= 0.0


def test_combine_pilot_and_sidetrack_sorts_local_sidetrack_md() -> None:
    pilot_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 200.0],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 100.0, 200.0],
            "INC_deg": [0.0, 0.0, 0.0],
            "AZI_deg": [0.0, 0.0, 0.0],
        }
    )
    local_stations = pd.DataFrame(
        {
            "MD_m": [80.0, 0.0, 30.0],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [180.0, 100.0, 130.0],
            "INC_deg": [0.0, 0.0, 0.0],
            "AZI_deg": [0.0, 0.0, 0.0],
            "segment": ["HOLD", "BUILD", "BUILD"],
        }
    )
    sidetrack_result = PlannerResult(
        stations=local_stations,
        summary={"md_total_m": 80.0},
        azimuth_deg=0.0,
        md_t1_m=30.0,
    )
    window = PilotWindow(
        pilot_name="WELL_PL",
        parent_name="WELL",
        md_m=100.0,
        point=Point3D(0.0, 0.0, 100.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )

    combined = combine_pilot_and_sidetrack(
        pilot_stations=pilot_stations,
        sidetrack_result=sidetrack_result,
        window=window,
        config=TrajectoryConfig(),
    )

    assert combined.stations["MD_m"].tolist() == pytest.approx([100.0, 130.0, 180.0])
    assert combined.stations["Z_m"].tolist() == pytest.approx([100.0, 130.0, 180.0])
    assert np.all(np.diff(combined.stations["MD_m"].to_numpy(dtype=float)) > 0.0)


def test_combine_pilot_and_sidetrack_rejects_invalid_surveys() -> None:
    valid_pilot = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 200.0],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 100.0, 200.0],
            "INC_deg": [0.0, 0.0, 0.0],
            "AZI_deg": [0.0, 0.0, 0.0],
        }
    )
    window = PilotWindow(
        pilot_name="WELL_PL",
        parent_name="WELL",
        md_m=100.0,
        point=Point3D(0.0, 0.0, 100.0),
        inc_deg=0.0,
        azi_deg=0.0,
    )

    valid_sidetrack = PlannerResult(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 50.0],
                "X_m": [0.0, 0.0],
                "Y_m": [0.0, 0.0],
                "Z_m": [100.0, 150.0],
                "INC_deg": [0.0, 0.0],
                "AZI_deg": [0.0, 0.0],
            }
        ),
        summary={"md_total_m": 50.0},
        azimuth_deg=0.0,
        md_t1_m=25.0,
    )
    nonmonotonic_pilot = valid_pilot.iloc[[0, 2, 1]].reset_index(drop=True)

    with pytest.raises(ValueError, match="не возрастающий MD"):
        combine_pilot_and_sidetrack(
            pilot_stations=nonmonotonic_pilot,
            sidetrack_result=valid_sidetrack,
            window=window,
            config=TrajectoryConfig(),
        )

    duplicate_sidetrack = valid_sidetrack.model_copy(
        update={
            "stations": pd.DataFrame(
                {
                    "MD_m": [0.0, 50.0, 50.0],
                    "X_m": [0.0, 0.0, 0.0],
                    "Y_m": [0.0, 0.0, 0.0],
                    "Z_m": [100.0, 150.0, 150.0],
                    "INC_deg": [0.0, 0.0, 0.0],
                    "AZI_deg": [0.0, 0.0, 0.0],
                }
            )
        }
    )
    with pytest.raises(ValueError, match="не возрастающий MD"):
        combine_pilot_and_sidetrack(
            pilot_stations=valid_pilot,
            sidetrack_result=duplicate_sidetrack,
            window=window,
            config=TrajectoryConfig(),
        )

    with pytest.raises(ValueError, match="MD t1 бокового ствола находится вне"):
        combine_pilot_and_sidetrack(
            pilot_stations=valid_pilot,
            sidetrack_result=valid_sidetrack.model_copy(update={"md_t1_m": 75.0}),
            window=window,
            config=TrajectoryConfig(),
        )
