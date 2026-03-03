from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

OBJECTIVE_MAXIMIZE_HOLD = "maximize_hold"
OBJECTIVE_MINIMIZE_BUILD_DLS = "minimize_build_dls"
ALLOWED_OBJECTIVE_MODES = (OBJECTIVE_MAXIMIZE_HOLD, OBJECTIVE_MINIMIZE_BUILD_DLS)
ObjectiveMode = Literal["maximize_hold", "minimize_build_dls"]
TURN_SOLVER_LEAST_SQUARES = "least_squares"
TURN_SOLVER_DE_HYBRID = "de_hybrid"
ALLOWED_TURN_SOLVER_MODES = (TURN_SOLVER_LEAST_SQUARES, TURN_SOLVER_DE_HYBRID)
TurnSolverMode = Literal["least_squares", "de_hybrid"]


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

    dls_build_min_deg_per_30m: float = 0.1
    dls_build_max_deg_per_30m: float = 3.0
    kop_min_vertical_m: float = 550.0
    kop_search_grid_size: int = 81
    reverse_inc_min_deg: float = 8.0
    reverse_inc_max_deg: float = 80.0
    reverse_inc_grid_size: int = 49
    adaptive_grid_initial_size: int = 11
    adaptive_grid_refine_levels: int = 2
    adaptive_grid_top_k: int = 6
    adaptive_grid_enabled: bool = True
    adaptive_dense_check_enabled: bool = False
    parallel_jobs: int = 1
    profile_cache_enabled: bool = True

    max_total_md_m: float = 12000.0
    # Post-processing MD threshold for user-facing validation only.
    # Does not participate in solver search/optimization constraints.
    max_total_md_postcheck_m: float = 6500.0
    objective_mode: ObjectiveMode = OBJECTIVE_MAXIMIZE_HOLD
    turn_solver_mode: TurnSolverMode = TURN_SOLVER_LEAST_SQUARES
    turn_solver_qmc_samples: int = 32
    turn_solver_local_starts: int = 1
    # Minimum MD span for BUILD/HOLD/BUILD sections. 30 m aligns with the common DLS reference interval (deg/30m).
    min_structural_segment_m: float = 30.0

    dls_limits_deg_per_30m: Dict[str, float] = field(
        default_factory=lambda: {
            "VERTICAL": 1.0,
            "BUILD_REV": 3.0,
            "HOLD_REV": 2.0,
            "DROP_REV": 3.0,
            "BUILD1": 3.0,
            "HOLD": 2.0,
            "BUILD2": 3.0,
            "HORIZONTAL": 2.0,
        }
    )


@dataclass(frozen=True)
class PlannerResult:
    stations: "pandas.DataFrame"
    summary: Dict[str, float | str]
    azimuth_deg: float
    md_t1_m: float
