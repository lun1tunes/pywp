from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from pywp.models import (
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    Point3D,
    TrajectoryConfig,
)
from pywp.planner import PlanningError, TrajectoryPlanner

pytestmark = pytest.mark.integration


def _fast_config(**overrides: object) -> TrajectoryConfig:
    base = {
        "md_step_m": 10.0,
        "md_step_control_m": 2.0,
        "pos_tolerance_m": 2.0,
        "entry_inc_target_deg": 86.0,
        "entry_inc_tolerance_deg": 2.0,
        "dls_build_max_deg_per_30m": 6.0,
        "max_total_md_m": 20000.0,
        "max_total_md_postcheck_m": 20000.0,
        "turn_solver_mode": TURN_SOLVER_LEAST_SQUARES,
    }
    base.update(overrides)
    return TrajectoryConfig(**base)


def test_same_direction_reference_case_solves_with_minimum_kop() -> None:
    config = _fast_config(kop_min_vertical_m=550.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert abs(float(result.summary["entry_inc_deg"]) - config.entry_inc_target_deg) <= config.entry_inc_tolerance_deg
    assert float(result.summary["kop_md_m"]) == pytest.approx(550.0, abs=1e-6)
    assert float(result.summary["azimuth_turn_deg"]) == pytest.approx(0.0, abs=1e-6)
    assert str(result.summary["trajectory_type"]) == "Unified J Profile + Build + Azimuth Turn"
    assert float(result.summary["max_dls_build1_deg_per_30m"]) <= 6.0 + 1e-6
    assert float(result.summary["max_dls_build2_deg_per_30m"]) <= 6.0 + 1e-6


def test_higher_min_vertical_pushes_kop_up_deterministically() -> None:
    config_low = _fast_config(kop_min_vertical_m=550.0)
    config_high = _fast_config(kop_min_vertical_m=900.0)
    planner = TrajectoryPlanner()

    result_low = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_low,
    )
    result_high = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_high,
    )

    assert float(result_low.summary["kop_md_m"]) == pytest.approx(550.0, abs=1e-6)
    assert float(result_high.summary["kop_md_m"]) >= 900.0 - 1e-6
    assert float(result_high.summary["md_total_m"]) >= float(result_low.summary["md_total_m"])


def test_higher_build_dls_reduces_total_md() -> None:
    planner = TrajectoryPlanner()
    config_soft = _fast_config(dls_build_max_deg_per_30m=3.0)
    config_hard = _fast_config(dls_build_max_deg_per_30m=6.0)

    result_soft = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_soft,
    )
    result_hard = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_hard,
    )

    assert float(result_hard.summary["md_total_m"]) <= float(result_soft.summary["md_total_m"])


def test_non_planar_turn_solver_least_squares_hits_targets() -> None:
    config = _fast_config(turn_solver_mode=TURN_SOLVER_LEAST_SQUARES, pos_tolerance_m=2.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["azimuth_turn_deg"]) > 1.0
    assert str(result.summary["solver_turn_mode"]) == TURN_SOLVER_LEAST_SQUARES
    assert int(result.summary["solver_turn_max_restarts"]) == int(config.turn_solver_max_restarts)
    assert int(result.summary["solver_turn_restarts_used"]) >= 0


def test_non_planar_solver_handles_post_entry_inc_drop_with_default_limits() -> None:
    config = TrajectoryConfig(turn_solver_max_restarts=0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(900.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["horizontal_inc_deg"]) < float(result.summary["entry_inc_deg"])
    assert int(result.summary["solver_turn_restarts_used"]) == 0
    assert float(result.summary["azimuth_turn_deg"]) > 1.0


@pytest.mark.slow
def test_non_planar_turn_solver_de_hybrid_hits_targets() -> None:
    config = _fast_config(turn_solver_mode=TURN_SOLVER_DE_HYBRID, pos_tolerance_m=2.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["azimuth_turn_deg"]) > 1.0
    assert str(result.summary["solver_turn_mode"]) == TURN_SOLVER_DE_HYBRID


def test_reverse_direction_geometry_uses_turn_branch_and_solves() -> None:
    config = _fast_config(pos_tolerance_m=2.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 400.0, 3000.0),
        t3=Point3D(1020.0, 1360.0, 3083.9122),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert str(result.summary["trajectory_target_direction"]) == "Цели в обратном направлении"
    assert np.isfinite(float(result.summary["azimuth_turn_deg"]))


def test_turn_solver_retries_with_deeper_search_when_first_attempt_fails(monkeypatch) -> None:
    import pywp.planner as planner_module

    original = planner_module._solve_turn_profile
    seen_restarts: list[int] = []

    def flaky_turn_solver(*args, **kwargs):
        search_settings = kwargs["search_settings"]
        seen_restarts.append(int(search_settings.restart_index))
        if len(seen_restarts) == 1:
            raise PlanningError(
                "No valid trajectory solution found within configured limits. "
                "Closest miss to t1 is 7.87 m.\n"
                "Reasons and actions:\n"
                "- Solver endpoint miss to t1 after optimization is 7.87 m (tolerance 2.00 m)."
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(planner_module, "_solve_turn_profile", flaky_turn_solver)
    config = _fast_config(turn_solver_max_restarts=1, pos_tolerance_m=2.0)

    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )

    assert seen_restarts == [0, 1]
    assert int(result.summary["solver_turn_restarts_used"]) == 1
    assert float(result.summary["solver_turn_search_depth_scale"]) > 1.0


def test_planner_rejects_negative_turn_restart_budget() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(turn_solver_max_restarts=-1)

    with pytest.raises(PlanningError, match="turn_solver_max_restarts must be non-negative"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_planner_keeps_solution_when_total_md_exceeds_soft_limit() -> None:
    config = _fast_config(max_total_md_postcheck_m=100.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["md_total_m"]) > float(config.max_total_md_postcheck_m)
    assert str(result.summary["md_postcheck_exceeded"]) == "yes"
    assert float(result.summary["md_postcheck_excess_m"]) > 0.0


def test_planner_validates_negative_build_dls_bounds() -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 0.0, 2500.0),
        t3=Point3D(1500.0, 0.0, 2600.0),
    )

    with pytest.raises(
        PlanningError, match="dls_build_min_deg_per_30m cannot be negative"
    ):
        planner.plan(
            **base_kwargs,
            config=_fast_config(dls_build_min_deg_per_30m=-0.1),
        )

    with pytest.raises(
        PlanningError, match="dls_build_max_deg_per_30m cannot be negative"
    ):
        planner.plan(
            **base_kwargs,
            config=_fast_config(dls_build_max_deg_per_30m=-0.1),
        )


def test_planner_validates_supported_entry_inc_target_range() -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
    )

    with pytest.raises(PlanningError, match="entry_inc_target_deg must be in \\(0, 90\\)"):
        planner.plan(
            **base_kwargs,
            config=_fast_config(entry_inc_target_deg=-5.0),
        )

    with pytest.raises(PlanningError, match="entry_inc_target_deg must be in \\(0, 90\\)"):
        planner.plan(
            **base_kwargs,
            config=_fast_config(entry_inc_target_deg=95.0, max_inc_deg=110.0),
        )


def test_planner_rejects_unknown_turn_solver_mode() -> None:
    with pytest.raises(ValidationError, match="least_squares|de_hybrid"):
        _fast_config(turn_solver_mode="unsupported_turn_solver")  # type: ignore[arg-type]
