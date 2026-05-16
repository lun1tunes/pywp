from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from pywp.models import (
    DEFAULT_BUILD_DLS_MAX_DEG_PER_30M,
    OPTIMIZATION_MINIMIZE_MD,
    OPTIMIZATION_NONE,
    Point3D,
    TURN_SOLVER_LEAST_SQUARES,
    TrajectoryConfig,
    build_segment_dls_limits_deg_per_30m,
)
from pywp.planner_config import (
    CFG_DEFAULTS,
    OPTIMIZATION_OPTIONS,
    TURN_SOLVER_OPTIONS,
    build_segment_dls_limits,
    build_trajectory_config,
    normalize_build_dls_bounds,
)


def test_option_dictionaries_cover_supported_modes() -> None:
    assert OPTIMIZATION_NONE in OPTIMIZATION_OPTIONS
    assert OPTIMIZATION_MINIMIZE_MD in OPTIMIZATION_OPTIONS
    assert TURN_SOLVER_LEAST_SQUARES in TURN_SOLVER_OPTIONS


def test_normalize_build_dls_bounds_orders_values() -> None:
    low, high = normalize_build_dls_bounds(3.0, 0.5)
    assert low == 0.5
    assert high == 3.0


def test_build_segment_dls_limits_applies_shared_build_limit() -> None:
    limits = build_segment_dls_limits(2.75)
    assert limits["BUILD1"] == 2.75
    assert limits["BUILD2"] == 2.75
    assert limits["HORIZONTAL"] == 2.75
    assert limits["VERTICAL"] == 1.0
    assert limits["HOLD"] == 2.0


def test_trajectory_config_defaults_use_shared_segment_limit_builder() -> None:
    cfg = TrajectoryConfig()
    expected = build_segment_dls_limits_deg_per_30m(DEFAULT_BUILD_DLS_MAX_DEG_PER_30M)
    assert cfg.dls_build_max_deg_per_30m == DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
    assert cfg.dls_limits_deg_per_30m == expected


def test_trajectory_config_auto_syncs_segment_limits_when_build_max_overridden() -> (
    None
):
    cfg = TrajectoryConfig(dls_build_max_deg_per_30m=5.5)
    assert cfg.dls_limits_deg_per_30m["BUILD1"] == 5.5
    assert cfg.dls_limits_deg_per_30m["BUILD2"] == 5.5
    assert cfg.dls_limits_deg_per_30m["HORIZONTAL"] == 5.5


def test_trajectory_config_validated_copy_revalidates_and_syncs_limits() -> None:
    cfg = TrajectoryConfig()
    updated = cfg.validated_copy(dls_build_max_deg_per_30m=4.5)

    assert updated.dls_build_max_deg_per_30m == 4.5
    assert updated.dls_limits_deg_per_30m["BUILD1"] == 4.5
    assert updated.dls_limits_deg_per_30m["BUILD2"] == 4.5
    assert updated.dls_limits_deg_per_30m["HORIZONTAL"] == 4.5

    with pytest.raises(ValidationError, match="least_squares|de_hybrid"):
        cfg.validated_copy(turn_solver_mode="unsupported_turn_solver")


def test_trajectory_config_rejects_cross_field_invalid_values_at_model_boundary() -> (
    None
):
    with pytest.raises(
        ValidationError, match="entry_inc_target_deg cannot exceed max_inc_deg"
    ):
        TrajectoryConfig(entry_inc_target_deg=85.0, max_inc_deg=80.0)

    with pytest.raises(
        ValidationError,
        match="dls_build_min_deg_per_30m cannot exceed dls_build_max_deg_per_30m",
    ):
        TrajectoryConfig(dls_build_min_deg_per_30m=4.0, dls_build_max_deg_per_30m=3.0)

    with pytest.raises(
        ValidationError, match="min_structural_segment_m must be >= md_step_control_m"
    ):
        TrajectoryConfig(min_structural_segment_m=1.0, md_step_control_m=2.0)


def test_trajectory_config_strips_legacy_dls_map_and_derives_segment_limits() -> None:
    cfg = TrajectoryConfig(
        dls_build_max_deg_per_30m=5.5,
        dls_limits_deg_per_30m={"HORIZONTAL": 3.5},
    )

    assert cfg.dls_limits_deg_per_30m == {
        "VERTICAL": 1.0,
        "BUILD1": 5.5,
        "HOLD": 2.0,
        "BUILD2": 5.5,
        "HORIZONTAL": 5.5,
    }
    assert "dls_limits_deg_per_30m" not in cfg.model_dump()


def test_trajectory_config_rejects_unknown_dls_segment_names() -> None:
    with pytest.raises(ValidationError, match="Unsupported DLS segment names"):
        TrajectoryConfig(dls_limits_deg_per_30m={"BUILD3": 4.0})


def test_trajectory_config_strips_removed_legacy_fields_for_backward_compatibility() -> (
    None
):
    cfg = TrajectoryConfig.model_validate(
        {
            "objective_mode": "minimize_total_md",
            "max_total_md_m": 100.0,
        }
    )
    assert cfg.optimization_mode == OPTIMIZATION_MINIMIZE_MD
    assert cfg.max_total_md_postcheck_m == CFG_DEFAULTS.max_total_md_postcheck_m


def test_trajectory_config_rejects_unknown_optimization_mode() -> None:
    with pytest.raises(ValidationError, match="none|minimize_md|minimize_kop"):
        TrajectoryConfig(optimization_mode="unsupported_optimization")  # type: ignore[arg-type]


def test_point3d_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValidationError):
        Point3D(x=math.nan, y=0.0, z=0.0)

    with pytest.raises(ValidationError):
        Point3D(x=0.0, y=math.inf, z=0.0)


def test_build_trajectory_config_pins_min_build_dls_to_zero_and_applies_limits() -> (
    None
):
    config = build_trajectory_config(
        md_step_m=CFG_DEFAULTS.md_step_m,
        md_step_control_m=CFG_DEFAULTS.md_step_control_m,
        lateral_tolerance_m=CFG_DEFAULTS.lateral_tolerance_m,
        vertical_tolerance_m=CFG_DEFAULTS.vertical_tolerance_m,
        entry_inc_target_deg=CFG_DEFAULTS.entry_inc_target_deg,
        entry_inc_tolerance_deg=CFG_DEFAULTS.entry_inc_tolerance_deg,
        max_inc_deg=CFG_DEFAULTS.max_inc_deg,
        dls_build_max_deg_per_30m=0.8,
        kop_min_vertical_m=CFG_DEFAULTS.kop_min_vertical_m,
        optimization_mode=OPTIMIZATION_NONE,
        turn_solver_mode=CFG_DEFAULTS.turn_solver_mode,
        turn_solver_max_restarts=CFG_DEFAULTS.turn_solver_max_restarts,
    )

    assert config.dls_build_min_deg_per_30m == 0.0
    assert config.dls_build_max_deg_per_30m == 0.8
    assert config.dls_limits_deg_per_30m["BUILD1"] == 0.8
    assert config.dls_limits_deg_per_30m["BUILD2"] == 0.8
    assert config.dls_limits_deg_per_30m["HORIZONTAL"] == 0.8
    assert config.optimization_mode == OPTIMIZATION_NONE
    assert config.turn_solver_max_restarts == CFG_DEFAULTS.turn_solver_max_restarts
    assert config.offer_j_profile is True
