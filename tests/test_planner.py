from __future__ import annotations

import pytest

from pywp.models import Point3D, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner


@pytest.mark.parametrize(
    "surface,t1,t3",
    [
        (Point3D(0.0, 0.0, 0.0), Point3D(300.0, 0.0, 2500.0), Point3D(1500.0, 0.0, 2600.0)),
        (Point3D(0.0, 0.0, 0.0), Point3D(600.0, 0.0, 2400.0), Point3D(2600.0, 0.0, 2600.0)),
        (Point3D(0.0, 0.0, 0.0), Point3D(800.0, 0.0, 2500.0), Point3D(2800.0, 0.0, 2650.0)),
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
            "VERTICAL": 1.0,
            "BUILD": 10.0,
            "HOLD": 2.0,
        },
    )

    result = TrajectoryPlanner().plan(surface=surface, t1=t1, t3=t3, config=config)

    assert result.summary["distance_t1_m"] <= config.pos_tolerance_m
    assert result.summary["distance_t3_m"] <= config.pos_tolerance_m
    assert abs(result.summary["entry_inc_deg"] - config.entry_inc_target_deg) <= config.entry_inc_tolerance_deg
    assert result.summary["max_dls_build_deg_per_30m"] <= config.dls_limits_deg_per_30m["BUILD"] + 1e-6
    assert len(result.stations) > 2


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
