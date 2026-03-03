from __future__ import annotations

from pywp.models import (
    OBJECTIVE_MAXIMIZE_HOLD,
    OBJECTIVE_MINIMIZE_AZIMUTH_TURN,
    OBJECTIVE_MINIMIZE_BUILD_DLS,
    OBJECTIVE_MINIMIZE_TOTAL_MD,
    SAME_DIRECTION_PROFILE_AUTO,
    SAME_DIRECTION_PROFILE_CLASSIC,
    SAME_DIRECTION_PROFILE_J_CURVE,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    TrajectoryConfig,
)

CFG_DEFAULTS = TrajectoryConfig()

OBJECTIVE_OPTIONS = {
    OBJECTIVE_MAXIMIZE_HOLD: "Максимизировать длину HOLD",
    OBJECTIVE_MINIMIZE_BUILD_DLS: "Минимизировать ПИ на BUILD",
    OBJECTIVE_MINIMIZE_AZIMUTH_TURN: "Минимизировать азимутальный доворот",
    OBJECTIVE_MINIMIZE_TOTAL_MD: "Минимизировать итоговую MD",
}

TURN_SOLVER_OPTIONS = {
    TURN_SOLVER_LEAST_SQUARES: "Least Squares (TRF, рекомендуется)",
    TURN_SOLVER_DE_HYBRID: "DE Hybrid (глобальный + локальный)",
}
SAME_DIRECTION_PROFILE_OPTIONS = {
    SAME_DIRECTION_PROFILE_AUTO: "Авто (рекомендованный)",
    SAME_DIRECTION_PROFILE_CLASSIC: "Классический (2 BUILD + HOLD)",
    SAME_DIRECTION_PROFILE_J_CURVE: "J-профиль (1 BUILD до t1)",
}


def normalize_build_dls_bounds(
    dls_build_min_deg_per_30m: float,
    dls_build_max_deg_per_30m: float,
) -> tuple[float, float]:
    min_value = float(min(dls_build_min_deg_per_30m, dls_build_max_deg_per_30m))
    max_value = float(max(dls_build_min_deg_per_30m, dls_build_max_deg_per_30m))
    return min_value, max_value


def build_segment_dls_limits(build_dls_max_deg_per_30m: float) -> dict[str, float]:
    build_limit = float(max(build_dls_max_deg_per_30m, 0.0))
    return {
        "VERTICAL": 1.0,
        "BUILD_REV": build_limit,
        "HOLD_REV": 2.0,
        "DROP_REV": build_limit,
        "BUILD1": build_limit,
        "HOLD": 2.0,
        "BUILD2": build_limit,
        "HORIZONTAL": 2.0,
    }


def build_trajectory_config(
    *,
    md_step_m: float,
    md_step_control_m: float,
    pos_tolerance_m: float,
    entry_inc_target_deg: float,
    entry_inc_tolerance_deg: float,
    max_inc_deg: float,
    dls_build_min_deg_per_30m: float,
    dls_build_max_deg_per_30m: float,
    kop_min_vertical_m: float,
    objective_mode: str,
    objective_auto_switch_to_turn: bool = CFG_DEFAULTS.objective_auto_switch_to_turn,
    objective_auto_turn_threshold_deg: float = CFG_DEFAULTS.objective_auto_turn_threshold_deg,
    turn_solver_mode: str,
    turn_solver_qmc_samples: int,
    turn_solver_local_starts: int,
    same_direction_profile_mode: str = SAME_DIRECTION_PROFILE_AUTO,
    adaptive_grid_enabled: bool = True,
    adaptive_dense_check_enabled: bool = True,
    adaptive_grid_initial_size: int = 11,
    adaptive_grid_refine_levels: int = 2,
    adaptive_grid_top_k: int = 6,
    parallel_jobs: int = 1,
    profile_cache_enabled: bool = True,
    max_total_md_postcheck_m: float = 6500.0,
) -> TrajectoryConfig:
    min_build, max_build = normalize_build_dls_bounds(
        dls_build_min_deg_per_30m=dls_build_min_deg_per_30m,
        dls_build_max_deg_per_30m=dls_build_max_deg_per_30m,
    )
    return TrajectoryConfig(
        md_step_m=float(md_step_m),
        md_step_control_m=float(md_step_control_m),
        pos_tolerance_m=float(pos_tolerance_m),
        entry_inc_target_deg=float(entry_inc_target_deg),
        entry_inc_tolerance_deg=float(entry_inc_tolerance_deg),
        max_inc_deg=float(max_inc_deg),
        dls_build_min_deg_per_30m=min_build,
        dls_build_max_deg_per_30m=max_build,
        kop_min_vertical_m=float(kop_min_vertical_m),
        objective_mode=str(objective_mode),
        objective_auto_switch_to_turn=bool(objective_auto_switch_to_turn),
        objective_auto_turn_threshold_deg=float(objective_auto_turn_threshold_deg),
        turn_solver_mode=str(turn_solver_mode),
        same_direction_profile_mode=str(same_direction_profile_mode),
        turn_solver_qmc_samples=int(turn_solver_qmc_samples),
        turn_solver_local_starts=int(turn_solver_local_starts),
        adaptive_grid_enabled=bool(adaptive_grid_enabled),
        adaptive_dense_check_enabled=bool(adaptive_dense_check_enabled),
        adaptive_grid_initial_size=int(adaptive_grid_initial_size),
        adaptive_grid_refine_levels=int(adaptive_grid_refine_levels),
        adaptive_grid_top_k=int(adaptive_grid_top_k),
        parallel_jobs=int(parallel_jobs),
        profile_cache_enabled=bool(profile_cache_enabled),
        max_total_md_postcheck_m=float(max_total_md_postcheck_m),
        dls_limits_deg_per_30m=build_segment_dls_limits(
            build_dls_max_deg_per_30m=max_build
        ),
    )
