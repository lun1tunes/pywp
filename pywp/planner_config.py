from __future__ import annotations

from pywp.models import (
    OBJECTIVE_MAXIMIZE_HOLD,
    OBJECTIVE_MINIMIZE_BUILD_DLS,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    TrajectoryConfig,
)

CFG_DEFAULTS = TrajectoryConfig()

OBJECTIVE_OPTIONS = {
    OBJECTIVE_MAXIMIZE_HOLD: "Максимизировать длину HOLD",
    OBJECTIVE_MINIMIZE_BUILD_DLS: "Минимизировать DLS на BUILD",
}

TURN_SOLVER_OPTIONS = {
    TURN_SOLVER_LEAST_SQUARES: "Least Squares (TRF, рекомендуется)",
    TURN_SOLVER_DE_HYBRID: "DE Hybrid (глобальный + локальный)",
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
    turn_solver_mode: str,
    turn_solver_qmc_samples: int,
    turn_solver_local_starts: int,
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
        turn_solver_mode=str(turn_solver_mode),
        turn_solver_qmc_samples=int(turn_solver_qmc_samples),
        turn_solver_local_starts=int(turn_solver_local_starts),
        dls_limits_deg_per_30m=build_segment_dls_limits(
            build_dls_max_deg_per_30m=max_build
        ),
    )
