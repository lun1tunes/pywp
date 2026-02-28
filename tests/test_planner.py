from __future__ import annotations

import numpy as np
import pytest

from pywp.models import Point3D, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner


@pytest.mark.parametrize(
    "surface,t1,t3",
    [
        (Point3D(0.0, 0.0, 0.0), Point3D(600.0, 800.0, 2400.0), Point3D(1500.0, 2000.0, 2500.0)),
        (Point3D(0.0, 0.0, 0.0), Point3D(800.0, 1066.6667, 2400.0), Point3D(2600.0, 3466.6667, 2600.0)),
        (Point3D(0.0, 0.0, 0.0), Point3D(900.0, 1200.0, 2500.0), Point3D(2800.0, 3733.3333, 2650.0)),
    ],
)
def test_planner_finds_solution_for_reference_scenarios(
    surface: Point3D, t1: Point3D, t3: Point3D
) -> None:
    config = TrajectoryConfig(
        md_step_m=10.0,
        md_step_control_m=2.0,
        pos_tolerance_m=2.0,
        entry_inc_target_deg=86.0,
        entry_inc_tolerance_deg=2.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=10.0,
        dls_limits_deg_per_30m={
            "BUILD1": 10.0,
            "HOLD": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        },
    )

    result = TrajectoryPlanner().plan(surface=surface, t1=t1, t3=t3, config=config)

    assert result.summary["distance_t1_m"] <= config.pos_tolerance_m
    assert result.summary["distance_t3_m"] <= config.pos_tolerance_m
    assert abs(result.summary["entry_inc_deg"] - config.entry_inc_target_deg) <= config.entry_inc_tolerance_deg
    assert result.summary["max_dls_build1_deg_per_30m"] <= config.dls_limits_deg_per_30m["BUILD1"] + 1e-6
    assert result.summary["max_dls_build2_deg_per_30m"] <= config.dls_limits_deg_per_30m["BUILD2"] + 1e-6
    assert result.summary["max_dls_horizontal_deg_per_30m"] <= config.dls_limits_deg_per_30m["HORIZONTAL"] + 1e-6
    assert len(result.stations) > 2
    assert set(result.stations["segment"]) == {"VERTICAL", "BUILD1", "HOLD", "BUILD2", "HORIZONTAL"}


def test_planner_raises_for_non_planar_geometry_without_turn() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(pos_tolerance_m=1.0)

    with pytest.raises(PlanningError):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 40.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=config,
        )


def test_planner_validates_invalid_dls_bounds() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        dls_build_min_deg_per_30m=3.0,
        dls_build_max_deg_per_30m=2.0,
    )

    with pytest.raises(PlanningError):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=config,
        )


def test_planner_validates_invalid_kop_config() -> None:
    planner = TrajectoryPlanner()

    with pytest.raises(PlanningError, match="kop_min_vertical_m must be non-negative"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=TrajectoryConfig(kop_min_vertical_m=-1.0),
        )

    with pytest.raises(PlanningError, match="kop_search_grid_size must be >= 2"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=TrajectoryConfig(kop_search_grid_size=1),
        )

    with pytest.raises(PlanningError, match="reverse_inc_min_deg must be < reverse_inc_max_deg"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=TrajectoryConfig(reverse_inc_min_deg=45.0, reverse_inc_max_deg=45.0),
        )


def test_planner_raises_if_entry_angle_from_t1_t3_not_86pm2() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(entry_inc_target_deg=86.0, entry_inc_tolerance_deg=2.0)

    with pytest.raises(PlanningError):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(700.0, 0.0, 3200.0),  # steeply downward: INC much lower than 84 deg
            config=config,
        )


def test_planner_rejects_unknown_objective_mode() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(objective_mode="unsupported_mode")  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="objective_mode must be one of"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_objective_mode_minimize_build_dls_not_higher_than_maximize_hold() -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        md_step_m=10.0,
        md_step_control_m=2.0,
        pos_tolerance_m=2.0,
        entry_inc_target_deg=86.0,
        entry_inc_tolerance_deg=2.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=10.0,
        dls_limits_deg_per_30m={
            "BUILD1": 10.0,
            "HOLD": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        },
    )
    config_hold = TrajectoryConfig(**base_kwargs, objective_mode="maximize_hold")
    config_min_dls = TrajectoryConfig(**base_kwargs, objective_mode="minimize_build_dls")

    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(600.0, 800.0, 2400.0)
    t3 = Point3D(1500.0, 2000.0, 2500.0)
    result_hold = planner.plan(surface=surface, t1=t1, t3=t3, config=config_hold)
    result_min_dls = planner.plan(surface=surface, t1=t1, t3=t3, config=config_min_dls)

    assert (
        result_min_dls.summary["max_dls_build1_deg_per_30m"]
        <= result_hold.summary["max_dls_build1_deg_per_30m"] + 1e-6
    )
    assert (
        result_min_dls.summary["max_dls_build2_deg_per_30m"]
        <= result_hold.summary["max_dls_build2_deg_per_30m"] + 1e-6
    )


def test_horizontal_segment_has_zero_dls_when_inc_and_azi_are_constant() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        md_step_m=5.0,
        md_step_control_m=1.0,
        dls_limits_deg_per_30m={
            "BUILD1": 10.0,
            "HOLD": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        },
    )
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    horizontal = result.stations[result.stations["segment"] == "HORIZONTAL"]
    assert len(horizontal) > 2
    assert float(horizontal["DLS_deg_per_30m"].max(skipna=True)) <= 1e-6


def test_profile_has_all_segments_in_expected_order() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        dls_limits_deg_per_30m={
            "BUILD1": 10.0,
            "HOLD": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        }
    )
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(800.0, 1066.6667, 2400.0),
        t3=Point3D(2600.0, 3466.6667, 2600.0),
        config=config,
    )

    sequence = result.stations["segment"].drop_duplicates().tolist()
    assert sequence == ["VERTICAL", "BUILD1", "HOLD", "BUILD2", "HORIZONTAL"]


def test_planner_raises_when_build_dls_limit_is_too_low_for_geometry() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        dls_build_min_deg_per_30m=0.1,
        dls_build_max_deg_per_30m=0.2,
        dls_limits_deg_per_30m={
            "BUILD1": 0.2,
            "HOLD": 2.0,
            "BUILD2": 0.2,
            "HORIZONTAL": 2.0,
        },
    )

    with pytest.raises(PlanningError, match="BUILD max DLS"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_planner_respects_max_total_md_limit() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        max_total_md_m=1000.0,
        dls_limits_deg_per_30m={
            "BUILD1": 10.0,
            "HOLD": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        },
    )

    with pytest.raises(PlanningError, match="configured limits"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_md_t1_is_boundary_between_build2_and_horizontal() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        md_step_m=2.0,
        md_step_control_m=1.0,
        dls_limits_deg_per_30m={
            "BUILD1": 10.0,
            "HOLD": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        },
    )
    t1 = Point3D(900.0, 1200.0, 2500.0)
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=t1,
        t3=Point3D(2800.0, 3733.3333, 2650.0),
        config=config,
    )

    stations = result.stations
    idx_t1 = int((stations["MD_m"] - result.md_t1_m).abs().idxmin())
    row_t1 = stations.loc[idx_t1]
    assert np.isclose(row_t1["X_m"], t1.x, atol=config.pos_tolerance_m)
    assert np.isclose(row_t1["Y_m"], t1.y, atol=config.pos_tolerance_m)
    assert np.isclose(row_t1["Z_m"], t1.z, atol=config.pos_tolerance_m)

    md_build2_max = float(stations.loc[stations["segment"] == "BUILD2", "MD_m"].max())
    md_horizontal_min = float(stations.loc[stations["segment"] == "HORIZONTAL", "MD_m"].min())
    assert md_build2_max <= result.md_t1_m + config.md_step_m + 1e-6
    assert md_horizontal_min >= result.md_t1_m - config.md_step_m - 1e-6


def test_default_min_vertical_before_kop_is_present() -> None:
    planner = TrajectoryPlanner()
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=TrajectoryConfig(),
    )

    vertical = result.stations[result.stations["segment"] == "VERTICAL"]
    assert not vertical.empty
    assert float(vertical["MD_m"].max()) >= 300.0 - 1e-6
    assert result.summary["kop_md_m"] >= 300.0 - 1e-6


def test_planner_raises_when_kop_min_vertical_exceeds_t1_depth() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(kop_min_vertical_m=2500.0)
    with pytest.raises(PlanningError, match="minimum vertical before KOP is too deep"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_planner_builds_reverse_profile_when_classification_requires_it() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        md_step_m=5.0,
        md_step_control_m=1.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=3.0,
    )

    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 0.0, 2000.0),
        t3=Point3D(1300.0, 0.0, 2070.0),
        config=config,
    )

    segments = set(result.stations["segment"])
    assert {"BUILD_REV", "DROP_REV"}.issubset(segments)
    assert str(result.summary["trajectory_type"]) == "Цели в обратном направлении"
    assert "hold_inc_deg" in result.summary
    assert "well_complexity" in result.summary


def test_planner_honors_segment_build_limits_even_if_global_build_max_is_higher() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=10.0,
        dls_limits_deg_per_30m={
            "BUILD1": 3.0,
            "HOLD": 2.0,
            "BUILD2": 3.0,
            "HORIZONTAL": 2.0,
        },
    )

    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert result.summary["distance_t1_m"] <= config.pos_tolerance_m
    assert result.summary["distance_t3_m"] <= config.pos_tolerance_m
    assert result.summary["max_dls_build1_deg_per_30m"] <= 3.0 + 1e-6
    assert result.summary["max_dls_build2_deg_per_30m"] <= 3.0 + 1e-6


def test_planner_raises_when_build_segment_limits_conflict_with_global_min() -> None:
    planner = TrajectoryPlanner()
    config = TrajectoryConfig(
        dls_build_min_deg_per_30m=4.0,
        dls_build_max_deg_per_30m=10.0,
        dls_limits_deg_per_30m={
            "BUILD1": 3.0,
            "HOLD": 2.0,
            "BUILD2": 3.0,
            "HORIZONTAL": 2.0,
        },
    )

    with pytest.raises(PlanningError, match="No feasible BUILD DLS interval"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )
