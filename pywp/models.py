from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

OBJECTIVE_MAXIMIZE_HOLD = "maximize_hold"
OBJECTIVE_MINIMIZE_BUILD_DLS = "minimize_build_dls"
OBJECTIVE_MINIMIZE_AZIMUTH_TURN = "minimize_azimuth_turn"
OBJECTIVE_MINIMIZE_TOTAL_MD = "minimize_total_md"
ALLOWED_OBJECTIVE_MODES = (
    OBJECTIVE_MAXIMIZE_HOLD,
    OBJECTIVE_MINIMIZE_BUILD_DLS,
    OBJECTIVE_MINIMIZE_AZIMUTH_TURN,
    OBJECTIVE_MINIMIZE_TOTAL_MD,
)
ObjectiveMode = Literal[
    "maximize_hold",
    "minimize_build_dls",
    "minimize_azimuth_turn",
    "minimize_total_md",
]
TURN_SOLVER_LEAST_SQUARES = "least_squares"
TURN_SOLVER_DE_HYBRID = "de_hybrid"
ALLOWED_TURN_SOLVER_MODES = (TURN_SOLVER_LEAST_SQUARES, TURN_SOLVER_DE_HYBRID)
TurnSolverMode = Literal["least_squares", "de_hybrid"]
SAME_DIRECTION_PROFILE_AUTO = "auto"
SAME_DIRECTION_PROFILE_CLASSIC = "classic"
SAME_DIRECTION_PROFILE_J_CURVE = "j_curve"
ALLOWED_SAME_DIRECTION_PROFILE_MODES = (
    SAME_DIRECTION_PROFILE_AUTO,
    SAME_DIRECTION_PROFILE_CLASSIC,
    SAME_DIRECTION_PROFILE_J_CURVE,
)
SameDirectionProfileMode = Literal["auto", "classic", "j_curve"]

DEFAULT_BUILD_DLS_MAX_DEG_PER_30M = 3.0

_SEGMENT_DLS_ORDER: tuple[str, ...] = (
    "VERTICAL",
    "BUILD1",
    "HOLD",
    "BUILD2",
    "HORIZONTAL",
)
_SEGMENT_DLS_FIXED_LIMITS: dict[str, float] = {
    "VERTICAL": 1.0,
    "HOLD": 2.0,
    "HORIZONTAL": 2.0,
}
_SEGMENT_DLS_BUILD_CONTROLLED: set[str] = {
    "BUILD1",
    "BUILD2",
}


def build_segment_dls_limits_deg_per_30m(
    build_dls_max_deg_per_30m: float,
) -> Dict[str, float]:
    """Build default segment DLS limits from a single BUILD max value."""

    build_limit = float(max(build_dls_max_deg_per_30m, 0.0))
    limits: Dict[str, float] = {}
    for segment_name in _SEGMENT_DLS_ORDER:
        if segment_name in _SEGMENT_DLS_BUILD_CONTROLLED:
            limits[segment_name] = build_limit
        else:
            limits[segment_name] = float(_SEGMENT_DLS_FIXED_LIMITS[segment_name])
    return limits


@dataclass(frozen=True)
class Point3D:
    """Cartesian point in meters: X=East, Y=North, Z=TVD (positive down)."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TrajectoryConfig:
    md_step_m: float = 10.0
    md_step_control_m: float = 2.0
    pos_tolerance_m: float = 2.0
    entry_inc_target_deg: float = 86.0
    entry_inc_tolerance_deg: float = 2.0
    max_inc_deg: float = 95.0

    dls_build_min_deg_per_30m: float = 0.0
    dls_build_max_deg_per_30m: float = DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
    kop_min_vertical_m: float = 550.0
    kop_search_grid_size: int = 81
    adaptive_grid_initial_size: int = 11
    adaptive_grid_refine_levels: int = 2
    adaptive_grid_top_k: int = 6
    adaptive_grid_enabled: bool = True
    adaptive_dense_check_enabled: bool = False
    parallel_jobs: int = 1
    profile_cache_enabled: bool = True
    same_direction_profile_mode: SameDirectionProfileMode = (
        SAME_DIRECTION_PROFILE_AUTO
    )

    max_total_md_m: float = 12000.0
    # Post-processing MD threshold for user-facing validation only.
    # Does not participate in solver search/optimization constraints.
    max_total_md_postcheck_m: float = 6500.0
    objective_mode: ObjectiveMode = OBJECTIVE_MAXIMIZE_HOLD
    objective_auto_switch_to_turn: bool = True
    objective_auto_turn_threshold_deg: float = 24.0
    turn_solver_mode: TurnSolverMode = TURN_SOLVER_LEAST_SQUARES
    turn_solver_qmc_samples: int = 32
    turn_solver_local_starts: int = 1
    # Minimum MD span for BUILD/HOLD/BUILD sections. 30 m aligns with the common DLS reference interval (deg/30m).
    min_structural_segment_m: float = 30.0

    dls_limits_deg_per_30m: Dict[str, float] = field(
        default_factory=lambda: build_segment_dls_limits_deg_per_30m(
            DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
        )
    )


@dataclass(frozen=True)
class PlannerResult:
    stations: "pandas.DataFrame"
    summary: Dict[str, float | str]
    azimuth_deg: float
    md_t1_m: float
