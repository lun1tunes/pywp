from __future__ import annotations

import numpy as np
import pytest

import pywp.planner as planner_module
from pywp.models import (
    Point3D,
    SAME_DIRECTION_PROFILE_AUTO,
    SAME_DIRECTION_PROFILE_CLASSIC,
    SAME_DIRECTION_PROFILE_J_CURVE,
    TrajectoryConfig,
)
from pywp.planner import PlanningError, TrajectoryPlanner


def _fast_config(**overrides: object) -> TrajectoryConfig:
    # Keep planner tests deterministic and fast: narrow search controls and skip
    # final dense validation unless a test explicitly needs it.
    base = {
        "kop_search_grid_size": 21,
        "reverse_inc_grid_size": 21,
        "adaptive_grid_enabled": True,
        "adaptive_dense_check_enabled": False,
        "adaptive_grid_initial_size": 5,
        "adaptive_grid_refine_levels": 1,
        "adaptive_grid_top_k": 2,
        "parallel_jobs": 1,
        "turn_solver_qmc_samples": 8,
        "turn_solver_local_starts": 4,
        "max_total_md_postcheck_m": 20000.0,
        "same_direction_profile_mode": SAME_DIRECTION_PROFILE_CLASSIC,
    }
    base.update(overrides)
    return TrajectoryConfig(**base)


def _profile_stub(*, hold_length_m: float, turn_deg: float, dls_build: float) -> planner_module.ProfileParameters:
    return planner_module.ProfileParameters(
        profile_kind="classic_s",
        trajectory_type="same_direction",
        kop_vertical_m=1000.0,
        reverse_inc_deg=0.0,
        reverse_hold_length_m=0.0,
        reverse_dls_deg_per_30m=0.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=86.0,
        inc_hold_deg=80.0,
        dls_build1_deg_per_30m=float(dls_build),
        dls_build2_deg_per_30m=float(dls_build),
        build1_length_m=300.0,
        hold_length_m=float(hold_length_m),
        build2_length_m=200.0,
        horizontal_length_m=1200.0,
        horizontal_adjust_length_m=0.0,
        horizontal_hold_length_m=1200.0,
        horizontal_inc_deg=86.0,
        horizontal_dls_deg_per_30m=0.5,
        azimuth_hold_deg=0.0,
        azimuth_entry_deg=float(turn_deg),
    )


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
    config = _fast_config(
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


def test_planner_supports_turn_for_non_planar_reverse_geometry() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(pos_tolerance_m=1.0)

    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )
    assert result.summary["distance_t1_m"] <= config.pos_tolerance_m
    assert result.summary["distance_t3_m"] <= config.pos_tolerance_m
    assert str(result.summary["trajectory_type"]) == "Цели в обратном направлении"
    assert float(result.summary["azimuth_turn_deg"]) > 1.0
    build2 = result.stations[result.stations["segment"] == "BUILD2"]
    assert float(build2["AZI_deg"].max() - build2["AZI_deg"].min()) > 1.0


def test_same_direction_j_profile_forced_uses_single_build_before_t1() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        same_direction_profile_mode=SAME_DIRECTION_PROFILE_J_CURVE,
        dls_build_min_deg_per_30m=0.1,
        dls_build_max_deg_per_30m=6.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD1": 6.0,
            "HOLD": 2.0,
            "BUILD2": 6.0,
            "HORIZONTAL": 2.0,
        },
    )
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(750.0, 0.0, 3000.0),
        t3=Point3D(2300.0, 0.0, 3070.0),
        config=config,
    )

    segments = set(result.stations["segment"])
    assert "BUILD1" in segments
    assert "HOLD" not in segments
    assert "BUILD2" not in segments
    assert str(result.summary["same_direction_profile_mode_applied"]) == "j_curve"


def test_same_direction_auto_mode_selects_j_profile_for_near_wellhead_targets() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        same_direction_profile_mode=SAME_DIRECTION_PROFILE_AUTO,
        dls_build_min_deg_per_30m=0.1,
        dls_build_max_deg_per_30m=6.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD1": 6.0,
            "HOLD": 2.0,
            "BUILD2": 6.0,
            "HORIZONTAL": 2.0,
        },
    )
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(750.0, 0.0, 3000.0),
        t3=Point3D(2300.0, 0.0, 3070.0),
        config=config,
    )

    assert str(result.summary["same_direction_profile_mode_requested"]) == "auto"
    assert str(result.summary["same_direction_profile_mode_applied"]) == "j_curve"


def test_same_direction_forced_j_reports_infeasible_limits() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        same_direction_profile_mode=SAME_DIRECTION_PROFILE_J_CURVE,
        dls_build_min_deg_per_30m=0.1,
        dls_build_max_deg_per_30m=2.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD1": 2.0,
            "HOLD": 2.0,
            "BUILD2": 2.0,
            "HORIZONTAL": 2.0,
        },
    )
    with pytest.raises(PlanningError, match="J-profile is not feasible"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(750.0, 0.0, 3000.0),
            t3=Point3D(2300.0, 0.0, 3070.0),
            config=config,
        )


def test_same_direction_forced_j_rejects_non_coplanar_case() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(same_direction_profile_mode=SAME_DIRECTION_PROFILE_J_CURVE)
    with pytest.raises(PlanningError, match="only supported for coplanar"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(1400.0, 500.0, 2500.0),
            t3=Point3D(2200.0, 1700.0, 2580.0),
            config=config,
        )


def test_planner_supports_turn_in_build_for_non_planar_same_direction_geometry() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        md_step_m=5.0,
        md_step_control_m=1.0,
        pos_tolerance_m=2.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=6.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD1": 6.0,
            "HOLD": 2.0,
            "BUILD2": 6.0,
            "HORIZONTAL": 2.0,
        },
    )

    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1400.0, 500.0, 2500.0),
        t3=Point3D(2200.0, 1700.0, 2580.0),
        config=config,
    )

    assert result.summary["distance_t1_m"] <= config.pos_tolerance_m
    assert result.summary["distance_t3_m"] <= config.pos_tolerance_m
    assert float(result.summary["azimuth_turn_deg"]) > 1.0

    build2 = result.stations[result.stations["segment"] == "BUILD2"]
    assert float(build2["AZI_deg"].max() - build2["AZI_deg"].min()) > 1.0


def test_reverse_non_planar_turn_respects_objective_mode_for_build_dls() -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        md_step_m=5.0,
        md_step_control_m=1.0,
        pos_tolerance_m=1.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=6.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD_REV": 6.0,
            "HOLD_REV": 2.0,
            "DROP_REV": 6.0,
            "BUILD1": 6.0,
            "HOLD": 2.0,
            "BUILD2": 6.0,
            "HORIZONTAL": 2.0,
        },
    )
    config_hold = _fast_config(**base_kwargs, objective_mode="maximize_hold")
    config_min_dls = _fast_config(**base_kwargs, objective_mode="minimize_build_dls")

    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(300.0, 300.0, 2000.0)
    t3 = Point3D(900.0, 1200.0, 2075.0)
    result_hold = planner.plan(surface=surface, t1=t1, t3=t3, config=config_hold)
    result_min_dls = planner.plan(surface=surface, t1=t1, t3=t3, config=config_min_dls)

    assert result_min_dls.summary["max_dls_build1_deg_per_30m"] <= result_hold.summary["max_dls_build1_deg_per_30m"] + 1e-6
    assert result_min_dls.summary["max_dls_build2_deg_per_30m"] <= result_hold.summary["max_dls_build2_deg_per_30m"] + 1e-6


def test_non_planar_turn_solver_honors_max_total_md_limit() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        md_step_m=5.0,
        md_step_control_m=1.0,
        pos_tolerance_m=1.0,
        max_total_md_m=800.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=6.0,
    )

    with pytest.raises(PlanningError, match="No valid TURN solution found"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(1400.0, 500.0, 2500.0),
            t3=Point3D(2200.0, 1700.0, 2580.0),
            config=config,
        )


def test_planner_keeps_solution_when_total_md_exceeds_soft_limit() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        md_step_m=10.0,
        md_step_control_m=2.0,
        pos_tolerance_m=2.0,
        max_total_md_postcheck_m=100.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=10.0,
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
    summary = result.summary
    assert float(summary["md_total_m"]) > float(config.max_total_md_postcheck_m)
    assert str(summary["md_postcheck_exceeded"]) == "yes"
    assert float(summary["md_postcheck_excess_m"]) > 0.0


def test_planner_validates_invalid_dls_bounds() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
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
            config=_fast_config(kop_min_vertical_m=-1.0),
        )

    with pytest.raises(PlanningError, match="kop_search_grid_size must be >= 2"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=_fast_config(kop_search_grid_size=1),
        )

    with pytest.raises(PlanningError, match="reverse_inc_min_deg must be < reverse_inc_max_deg"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=_fast_config(reverse_inc_min_deg=45.0, reverse_inc_max_deg=45.0),
        )


def test_planner_allows_post_entry_build_to_match_t3_with_entry_inc_target() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(entry_inc_target_deg=86.0, entry_inc_tolerance_deg=2.0)
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert abs(float(result.summary["entry_inc_deg"]) - 86.0) <= 2.0
    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["horizontal_adjust_length_m"]) > 0.0


def test_planner_reports_overbend_requirement_when_max_inc_is_too_low() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        entry_inc_target_deg=86.0,
        entry_inc_tolerance_deg=2.0,
        max_inc_deg=88.0,
    )

    with pytest.raises(PlanningError, match="without overbend"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2200.0),
            config=config,
        )


def test_planner_rejects_unknown_objective_mode() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(objective_mode="unsupported_mode")  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="objective_mode must be one of"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_planner_rejects_unknown_turn_solver_mode() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(turn_solver_mode="unsupported_turn_solver")  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="turn_solver_mode must be one of"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_planner_rejects_unknown_same_direction_profile_mode() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(same_direction_profile_mode="unsupported_profile_mode")  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="same_direction_profile_mode must be one of"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_planner_validates_turn_solver_numeric_controls() -> None:
    planner = TrajectoryPlanner()
    with pytest.raises(PlanningError, match="turn_solver_qmc_samples must be non-negative"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=_fast_config(turn_solver_qmc_samples=-1),
        )

    with pytest.raises(PlanningError, match="turn_solver_local_starts must be >= 1"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=_fast_config(turn_solver_local_starts=0),
        )
    with pytest.raises(PlanningError, match="objective_auto_turn_threshold_deg must be in \\[0, 180\\]"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=_fast_config(objective_auto_turn_threshold_deg=-1.0),
        )


def test_objective_auto_switch_selects_min_turn_candidate_when_threshold_exceeded() -> None:
    cfg = _fast_config(
        objective_mode="maximize_hold",
        objective_auto_switch_to_turn=True,
        objective_auto_turn_threshold_deg=20.0,
    )
    hold_favored = _profile_stub(hold_length_m=500.0, turn_deg=34.0, dls_build=1.2)
    turn_favored = _profile_stub(hold_length_m=420.0, turn_deg=7.0, dls_build=1.4)
    runtime = planner_module._SolverRuntimeContext(
        parallel_requested_jobs=1,
        objective_mode_requested=str(cfg.objective_mode),
        objective_mode_applied=str(cfg.objective_mode),
    )

    selected = planner_module._select_best_candidate_by_config(
        candidates=[hold_favored, turn_favored],
        config=cfg,
        runtime_context=runtime,
    )

    assert selected is turn_favored
    assert runtime.objective_mode_requested == "maximize_hold"
    assert runtime.objective_mode_applied == "minimize_azimuth_turn"
    assert runtime.objective_auto_switched is True
    assert runtime.objective_auto_switch_trigger_turn_deg == pytest.approx(34.0)


def test_objective_auto_switch_keeps_maximize_hold_below_threshold() -> None:
    cfg = _fast_config(
        objective_mode="maximize_hold",
        objective_auto_switch_to_turn=True,
        objective_auto_turn_threshold_deg=40.0,
    )
    hold_favored = _profile_stub(hold_length_m=500.0, turn_deg=34.0, dls_build=1.2)
    turn_favored = _profile_stub(hold_length_m=420.0, turn_deg=7.0, dls_build=1.4)
    runtime = planner_module._SolverRuntimeContext(
        parallel_requested_jobs=1,
        objective_mode_requested=str(cfg.objective_mode),
        objective_mode_applied=str(cfg.objective_mode),
    )

    selected = planner_module._select_best_candidate_by_config(
        candidates=[hold_favored, turn_favored],
        config=cfg,
        runtime_context=runtime,
    )

    assert selected is hold_favored
    assert runtime.objective_mode_requested == "maximize_hold"
    assert runtime.objective_mode_applied == "maximize_hold"
    assert runtime.objective_auto_switched is False


def test_planner_validates_adaptive_and_parallel_controls() -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
    )

    with pytest.raises(PlanningError, match="adaptive_grid_initial_size must be >= 2"):
        planner.plan(**base_kwargs, config=_fast_config(adaptive_grid_initial_size=1))

    with pytest.raises(PlanningError, match="adaptive_grid_refine_levels must be >= 0"):
        planner.plan(**base_kwargs, config=_fast_config(adaptive_grid_refine_levels=-1))

    with pytest.raises(PlanningError, match="adaptive_grid_top_k must be >= 1"):
        planner.plan(**base_kwargs, config=_fast_config(adaptive_grid_top_k=0))

    with pytest.raises(PlanningError, match="parallel_jobs must be >= 1"):
        planner.plan(**base_kwargs, config=_fast_config(parallel_jobs=0))


@pytest.mark.parametrize("objective_mode", ["maximize_hold", "minimize_build_dls"])
def test_adaptive_search_objective_is_not_worse_than_dense_baseline(
    objective_mode: str,
) -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        md_step_m=10.0,
        md_step_control_m=2.0,
        pos_tolerance_m=2.0,
        entry_inc_target_deg=86.0,
        entry_inc_tolerance_deg=2.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=10.0,
        objective_mode=objective_mode,
        kop_search_grid_size=31,
        adaptive_grid_initial_size=3,
        adaptive_grid_refine_levels=0,
        adaptive_grid_top_k=1,
        parallel_jobs=1,
    )
    dense_config = _fast_config(**base_kwargs, adaptive_grid_enabled=False)
    adaptive_config = _fast_config(
        **base_kwargs,
        adaptive_grid_enabled=True,
        adaptive_dense_check_enabled=True,
    )

    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(600.0, 800.0, 2400.0)
    t3 = Point3D(1500.0, 2000.0, 2500.0)
    result_dense = planner.plan(surface=surface, t1=t1, t3=t3, config=dense_config)
    result_adaptive = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=adaptive_config,
    )

    hold_dense = float(result_dense.summary["hold_length_m"])
    hold_adaptive = float(result_adaptive.summary["hold_length_m"])
    dls_dense = float(result_dense.summary["max_dls_build1_deg_per_30m"])
    dls_adaptive = float(result_adaptive.summary["max_dls_build1_deg_per_30m"])

    if objective_mode == "maximize_hold":
        assert hold_adaptive >= hold_dense - 1e-6
        if abs(hold_adaptive - hold_dense) <= 1e-4:
            assert dls_adaptive <= dls_dense + 1e-6
    else:
        assert dls_adaptive <= dls_dense + 1e-6
        if abs(dls_adaptive - dls_dense) <= 1e-4:
            assert hold_adaptive >= hold_dense - 1e-6

    assert str(result_dense.summary["solver_adaptive_dense_check"]) == "no"
    assert str(result_adaptive.summary["solver_adaptive_dense_check"]) == "yes"


def test_adaptive_dense_check_can_be_disabled() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        md_step_m=10.0,
        md_step_control_m=2.0,
        pos_tolerance_m=2.0,
        entry_inc_target_deg=86.0,
        entry_inc_tolerance_deg=2.0,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=10.0,
        kop_search_grid_size=31,
        adaptive_grid_enabled=True,
        adaptive_dense_check_enabled=False,
        adaptive_grid_initial_size=5,
        adaptive_grid_refine_levels=1,
        adaptive_grid_top_k=2,
        parallel_jobs=1,
    )
    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )
    assert str(result.summary["solver_adaptive_grid_enabled"]) == "yes"
    assert str(result.summary["solver_adaptive_dense_check"]) == "no"


def test_parallel_fallback_is_reported_in_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_BrokenExecutor":
            raise RuntimeError("forced parallel failure for test")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    monkeypatch.setattr("pywp.planner.ProcessPoolExecutor", _BrokenExecutor)
    planner = TrajectoryPlanner()
    config = _fast_config(
        parallel_jobs=2,
        adaptive_grid_enabled=False,
        kop_search_grid_size=21,
    )

    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert str(result.summary["solver_parallel_status"]) == "fallback_to_sequential"
    assert str(result.summary["solver_parallel_fallback"]) == "yes"
    assert float(result.summary["solver_parallel_fallback_batches"]) >= 1.0
    assert "forced parallel failure for test" in str(
        result.summary["solver_parallel_fallback_reason"]
    )


def test_reverse_turn_summary_uses_configured_turn_solver_depth_without_hidden_minima() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        pos_tolerance_m=1.0,
        turn_solver_qmc_samples=0,
        turn_solver_local_starts=1,
    )

    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["solver_turn_qmc_samples"]) == pytest.approx(0.0)
    assert float(result.summary["solver_turn_local_starts"]) == pytest.approx(1.0)


def test_reverse_turn_single_start_uses_azimuth_jitter_for_stability() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        pos_tolerance_m=2.0,
        turn_solver_qmc_samples=0,
        turn_solver_local_starts=1,
        adaptive_dense_check_enabled=False,
    )

    result = planner.plan(
        surface=Point3D(598863.0, 7411139.0, 0.0),
        t1=Point3D(598533.0, 7410844.0, 3769.876),
        t3=Point3D(596660.0, 7409688.0, 3771.395),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    # Regression guard: a single-start TURN solve should avoid the previous
    # high-turn local minimum (~25.9 deg for this geometry).
    assert float(result.summary["azimuth_turn_deg"]) < 24.0


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
    config_hold = _fast_config(**base_kwargs, objective_mode="maximize_hold")
    config_min_dls = _fast_config(**base_kwargs, objective_mode="minimize_build_dls")

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


def test_objective_mode_minimize_azimuth_turn_prefers_smaller_turn_for_reverse_turn_case() -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        md_step_m=10.0,
        md_step_control_m=2.0,
        pos_tolerance_m=2.0,
        turn_solver_qmc_samples=0,
        turn_solver_local_starts=1,
        adaptive_dense_check_enabled=False,
        dls_build_min_deg_per_30m=0.1,
        dls_build_max_deg_per_30m=3.0,
    )
    config_hold = _fast_config(**base_kwargs, objective_mode="maximize_hold")
    config_turn = _fast_config(
        **base_kwargs,
        objective_mode="minimize_azimuth_turn",
    )

    surface = Point3D(598863.0, 7411139.0, 0.0)
    t1 = Point3D(599168.0, 7411032.0, 3799.701)
    t3 = Point3D(601041.0, 7412188.0, 3798.500)
    result_hold = planner.plan(surface=surface, t1=t1, t3=t3, config=config_hold)
    result_turn = planner.plan(surface=surface, t1=t1, t3=t3, config=config_turn)

    assert float(result_hold.summary["distance_t1_m"]) <= config_hold.pos_tolerance_m
    assert float(result_hold.summary["distance_t3_m"]) <= config_hold.pos_tolerance_m
    assert float(result_turn.summary["distance_t1_m"]) <= config_turn.pos_tolerance_m
    assert float(result_turn.summary["distance_t3_m"]) <= config_turn.pos_tolerance_m
    assert float(result_turn.summary["azimuth_turn_deg"]) <= float(
        result_hold.summary["azimuth_turn_deg"]
    ) + 1e-6


def test_objective_mode_minimize_total_md_prefers_shorter_well_path() -> None:
    planner = TrajectoryPlanner()
    base_kwargs = dict(
        md_step_m=5.0,
        md_step_control_m=1.0,
        pos_tolerance_m=2.0,
        turn_solver_qmc_samples=8,
        turn_solver_local_starts=4,
        adaptive_dense_check_enabled=False,
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=6.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD1": 6.0,
            "HOLD": 2.0,
            "BUILD2": 6.0,
            "HORIZONTAL": 2.0,
        },
    )
    config_hold = _fast_config(**base_kwargs, objective_mode="maximize_hold")
    config_md = _fast_config(**base_kwargs, objective_mode="minimize_total_md")

    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(1400.0, 500.0, 2500.0)
    t3 = Point3D(2200.0, 1700.0, 2580.0)
    result_hold = planner.plan(surface=surface, t1=t1, t3=t3, config=config_hold)
    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=config_md)

    assert float(result_hold.summary["distance_t1_m"]) <= config_hold.pos_tolerance_m
    assert float(result_hold.summary["distance_t3_m"]) <= config_hold.pos_tolerance_m
    assert float(result_md.summary["distance_t1_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["distance_t3_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["md_total_m"]) <= float(
        result_hold.summary["md_total_m"]
    ) + 1e-6


def test_horizontal_segment_respects_horizontal_dls_limit_with_post_entry_smoothing() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
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
    assert float(horizontal["DLS_deg_per_30m"].max(skipna=True)) <= config.dls_limits_deg_per_30m["HORIZONTAL"] + 1e-6
    assert float(result.summary["horizontal_adjust_length_m"]) >= 0.0


def test_profile_has_all_segments_in_expected_order() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
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
    config = _fast_config(
        dls_build_min_deg_per_30m=0.1,
        dls_build_max_deg_per_30m=0.2,
        dls_limits_deg_per_30m={
            "BUILD1": 0.2,
            "HOLD": 2.0,
            "BUILD2": 0.2,
            "HORIZONTAL": 2.0,
        },
    )

    with pytest.raises(PlanningError, match="BUILD max DLS") as exc_info:
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )
    message = str(exc_info.value)
    assert "Reasons and actions" in message
    assert "BUILD DLS upper bound is insufficient" in message


def test_planner_respects_max_total_md_limit() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
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
    config = _fast_config(
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
        config=_fast_config(),
    )

    vertical = result.stations[result.stations["segment"] == "VERTICAL"]
    assert not vertical.empty
    assert float(vertical["MD_m"].max()) >= 300.0 - 1e-6
    assert result.summary["kop_md_m"] >= 300.0 - 1e-6


def test_planner_raises_when_kop_min_vertical_exceeds_t1_depth() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(kop_min_vertical_m=2500.0)
    with pytest.raises(PlanningError, match="minimum vertical before KOP is too deep"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_planner_builds_reverse_profile_when_classification_requires_it() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
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
    config = _fast_config(
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
    config = _fast_config(
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


def test_reverse_profile_is_feasible_around_dls_3_4_to_3_5_for_typical_case() -> None:
    planner = TrajectoryPlanner()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(210.0, 280.0, 2000.0)
    t3 = Point3D(810.0, 1080.0, 2069.9268)

    hold_values: list[float] = []
    distance_values: list[float] = []
    for dmax in (3.4, 3.5):
        config = _fast_config(
            dls_build_min_deg_per_30m=0.5,
            dls_build_max_deg_per_30m=dmax,
            dls_limits_deg_per_30m={
                "VERTICAL": 1.0,
                "BUILD_REV": dmax,
                "HOLD_REV": 2.0,
                "DROP_REV": dmax,
                "BUILD1": dmax,
                "HOLD": 2.0,
                "BUILD2": dmax,
                "HORIZONTAL": 2.0,
            },
        )
        result = planner.plan(surface=surface, t1=t1, t3=t3, config=config)
        hold_values.append(float(result.summary["hold_inc_deg"]))
        distance_values.extend(
            [
                float(result.summary["distance_t1_m"]),
                float(result.summary["distance_t3_m"]),
            ]
        )

    assert len(hold_values) == 2
    assert all(np.isfinite(value) for value in hold_values)
    assert all(0.0 <= value <= 95.0 for value in hold_values)
    assert all(value <= 2.0 + 1e-6 for value in distance_values)


def test_reverse_profile_keeps_non_degenerate_forward_structure() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(
        dls_build_min_deg_per_30m=0.5,
        dls_build_max_deg_per_30m=3.5,
        min_structural_segment_m=30.0,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD_REV": 3.5,
            "HOLD_REV": 2.0,
            "DROP_REV": 3.5,
            "BUILD1": 3.5,
            "HOLD": 2.0,
            "BUILD2": 3.5,
            "HORIZONTAL": 2.0,
        },
    )

    result = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(210.0, 280.0, 2000.0),
        t3=Point3D(810.0, 1080.0, 2069.9268),
        config=config,
    )
    stations = result.stations

    for segment in ("BUILD1", "HOLD", "BUILD2"):
        block = stations.loc[stations["segment"] == segment, "MD_m"]
        assert not block.empty
        assert float(block.max() - block.min()) >= config.min_structural_segment_m - 1e-6


def test_planner_validates_structural_segment_resolution_config() -> None:
    planner = TrajectoryPlanner()

    with pytest.raises(PlanningError, match="min_structural_segment_m must be positive"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=_fast_config(min_structural_segment_m=0.0),
        )

    with pytest.raises(PlanningError, match="min_structural_segment_m must be >= md_step_control_m"):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 0.0, 2500.0),
            t3=Point3D(1500.0, 0.0, 2600.0),
            config=_fast_config(min_structural_segment_m=1.0, md_step_control_m=2.0),
        )
