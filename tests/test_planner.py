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
        inc_entry_min_deg=55.0,
        inc_entry_max_deg=86.0,
        dls_build1_min_deg_per_30m=2.0,
        dls_build1_max_deg_per_30m=8.0,
        dls_build2_min_deg_per_30m=2.0,
        dls_build2_max_deg_per_30m=10.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD1": 8.0,
            "HOLD1": 2.0,
            "HOLD2": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        },
    )

    result = TrajectoryPlanner().plan(surface=surface, t1=t1, t3=t3, config=config)

    assert result.summary["distance_t1_m"] <= config.pos_tolerance_m
    assert result.summary["distance_t3_m"] <= config.pos_tolerance_m
    assert result.summary["entry_inc_deg"] <= config.inc_entry_max_deg + config.angle_tolerance_deg
    assert result.summary["max_dls_build1_deg_per_30m"] <= config.dls_limits_deg_per_30m["BUILD1"] + 1e-6
    assert result.summary["max_dls_build2_deg_per_30m"] <= config.dls_limits_deg_per_30m["BUILD2"] + 1e-6
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
        dls_build1_min_deg_per_30m=3.0,
        dls_build1_max_deg_per_30m=2.0,
    )

    with pytest.raises(PlanningError):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=config,
        )
