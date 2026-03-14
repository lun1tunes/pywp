from __future__ import annotations

from pywp.models import (
    DEFAULT_BUILD_DLS_MAX_DEG_PER_30M,
    OBJECTIVE_MINIMIZE_TOTAL_MD,
    TURN_SOLVER_LEAST_SQUARES,
    TrajectoryConfig,
    build_segment_dls_limits_deg_per_30m,
)
from pywp.planner_config import (
    CFG_DEFAULTS,
    OBJECTIVE_OPTIONS,
    TURN_SOLVER_OPTIONS,
    build_segment_dls_limits,
    build_trajectory_config,
    normalize_build_dls_bounds,
)


def test_option_dictionaries_cover_supported_modes() -> None:
    assert OBJECTIVE_MINIMIZE_TOTAL_MD in OBJECTIVE_OPTIONS
    assert TURN_SOLVER_LEAST_SQUARES in TURN_SOLVER_OPTIONS


def test_normalize_build_dls_bounds_orders_values() -> None:
    low, high = normalize_build_dls_bounds(3.0, 0.5)
    assert low == 0.5
    assert high == 3.0


def test_build_segment_dls_limits_applies_shared_build_limit() -> None:
    limits = build_segment_dls_limits(2.75)
    assert limits["BUILD1"] == 2.75
    assert limits["BUILD2"] == 2.75
    assert limits["VERTICAL"] == 1.0
    assert limits["HOLD"] == 2.0
    assert limits["HORIZONTAL"] == 2.0


def test_trajectory_config_defaults_use_shared_segment_limit_builder() -> None:
    cfg = TrajectoryConfig()
    expected = build_segment_dls_limits_deg_per_30m(
        DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
    )
    assert cfg.dls_build_max_deg_per_30m == DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
    assert cfg.dls_limits_deg_per_30m == expected


def test_trajectory_config_auto_syncs_segment_limits_when_build_max_overridden() -> None:
    cfg = TrajectoryConfig(dls_build_max_deg_per_30m=5.5)
    assert cfg.dls_limits_deg_per_30m["BUILD1"] == 5.5
    assert cfg.dls_limits_deg_per_30m["BUILD2"] == 5.5


def test_build_trajectory_config_pins_min_build_dls_to_zero_and_applies_limits() -> None:
    config = build_trajectory_config(
        md_step_m=CFG_DEFAULTS.md_step_m,
        md_step_control_m=CFG_DEFAULTS.md_step_control_m,
        pos_tolerance_m=CFG_DEFAULTS.pos_tolerance_m,
        entry_inc_target_deg=CFG_DEFAULTS.entry_inc_target_deg,
        entry_inc_tolerance_deg=CFG_DEFAULTS.entry_inc_tolerance_deg,
        max_inc_deg=CFG_DEFAULTS.max_inc_deg,
        dls_build_max_deg_per_30m=0.8,
        kop_min_vertical_m=CFG_DEFAULTS.kop_min_vertical_m,
        turn_solver_mode=CFG_DEFAULTS.turn_solver_mode,
        turn_solver_max_restarts=CFG_DEFAULTS.turn_solver_max_restarts,
    )

    assert config.dls_build_min_deg_per_30m == 0.0
    assert config.dls_build_max_deg_per_30m == 0.8
    assert config.dls_limits_deg_per_30m["BUILD1"] == 0.8
    assert config.dls_limits_deg_per_30m["BUILD2"] == 0.8
    assert config.objective_mode == OBJECTIVE_MINIMIZE_TOTAL_MD
    assert config.turn_solver_max_restarts == CFG_DEFAULTS.turn_solver_max_restarts
