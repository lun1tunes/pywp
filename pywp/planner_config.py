from __future__ import annotations

from pywp.models import (
    INTERPOLATION_RODRIGUES,
    INTERPOLATION_SLERP,
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    OPTIMIZATION_MINIMIZE_KOP,
    OPTIMIZATION_MINIMIZE_MD,
    OPTIMIZATION_NONE,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    TrajectoryConfig,
    build_segment_dls_limits_deg_per_30m,
)

CFG_DEFAULTS = TrajectoryConfig()

OPTIMIZATION_OPTIONS = {
    OPTIMIZATION_NONE: "Без оптимизации",
    OPTIMIZATION_MINIMIZE_MD: "Минимизация MD",
    OPTIMIZATION_MINIMIZE_KOP: "Минимизация KOP",
}

_INTERNAL_OPTIMIZATION_DISPLAY_OPTIONS = {
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE: "Anti-collision avoidance",
}

TURN_SOLVER_OPTIONS = {
    TURN_SOLVER_LEAST_SQUARES: "Least Squares (TRF, рекомендуется)",
    TURN_SOLVER_DE_HYBRID: "DE Hybrid (глобальный + локальный)",
}

INTERPOLATION_METHOD_OPTIONS = {
    INTERPOLATION_RODRIGUES: "Rodrigues (рекомендуется)",
    INTERPOLATION_SLERP: "SLERP (классический)",
}


def optimization_display_label(mode: str) -> str:
    text = str(mode).strip()
    if not text:
        return "—"
    return OPTIMIZATION_OPTIONS.get(
        text,
        _INTERNAL_OPTIMIZATION_DISPLAY_OPTIONS.get(text, text),
    )


def normalize_build_dls_bounds(
    dls_build_min_deg_per_30m: float,
    dls_build_max_deg_per_30m: float,
) -> tuple[float, float]:
    min_value = float(min(dls_build_min_deg_per_30m, dls_build_max_deg_per_30m))
    max_value = float(max(dls_build_min_deg_per_30m, dls_build_max_deg_per_30m))
    return min_value, max_value


def build_segment_dls_limits(build_dls_max_deg_per_30m: float) -> dict[str, float]:
    return build_segment_dls_limits_deg_per_30m(
        build_dls_max_deg_per_30m=build_dls_max_deg_per_30m
    )


def build_trajectory_config(
    *,
    md_step_m: float,
    md_step_control_m: float,
    lateral_tolerance_m: float,
    vertical_tolerance_m: float,
    entry_inc_target_deg: float,
    entry_inc_tolerance_deg: float,
    max_inc_deg: float,
    dls_build_max_deg_per_30m: float,
    kop_min_vertical_m: float,
    optimization_mode: str,
    turn_solver_mode: str,
    turn_solver_max_restarts: int,
    max_total_md_postcheck_m: float = 6500.0,
    interpolation_method: str = INTERPOLATION_RODRIGUES,
    offer_j_profile: bool = True,
) -> TrajectoryConfig:
    max_build = float(max(dls_build_max_deg_per_30m, 0.0))
    return TrajectoryConfig(
        md_step_m=float(md_step_m),
        md_step_control_m=float(md_step_control_m),
        lateral_tolerance_m=float(lateral_tolerance_m),
        vertical_tolerance_m=float(vertical_tolerance_m),
        entry_inc_target_deg=float(entry_inc_target_deg),
        entry_inc_tolerance_deg=float(entry_inc_tolerance_deg),
        max_inc_deg=float(max_inc_deg),
        dls_build_min_deg_per_30m=0.0,
        dls_build_max_deg_per_30m=max_build,
        kop_min_vertical_m=float(kop_min_vertical_m),
        optimization_mode=str(optimization_mode),
        turn_solver_mode=str(turn_solver_mode),
        turn_solver_max_restarts=int(turn_solver_max_restarts),
        max_total_md_postcheck_m=float(max_total_md_postcheck_m),
        interpolation_method=str(interpolation_method),
        offer_j_profile=bool(offer_j_profile),
    )
