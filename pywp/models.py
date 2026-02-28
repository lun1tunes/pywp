from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

OBJECTIVE_MAXIMIZE_HOLD = "maximize_hold"
OBJECTIVE_MINIMIZE_BUILD_DLS = "minimize_build_dls"
ALLOWED_OBJECTIVE_MODES = (OBJECTIVE_MAXIMIZE_HOLD, OBJECTIVE_MINIMIZE_BUILD_DLS)
ObjectiveMode = Literal["maximize_hold", "minimize_build_dls"]


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

    dls_build_min_deg_per_30m: float = 0.5
    dls_build_max_deg_per_30m: float = 10.0
    kop_min_vertical_m: float = 300.0
    kop_search_grid_size: int = 81

    max_total_md_m: float = 12000.0
    objective_mode: ObjectiveMode = OBJECTIVE_MAXIMIZE_HOLD

    dls_limits_deg_per_30m: Dict[str, float] = field(
        default_factory=lambda: {
            "VERTICAL": 1.0,
            "BUILD1": 10.0,
            "HOLD": 2.0,
            "BUILD2": 10.0,
            "HORIZONTAL": 2.0,
        }
    )


@dataclass(frozen=True)
class PlannerResult:
    stations: "pandas.DataFrame"
    summary: Dict[str, float]
    azimuth_deg: float
    md_t1_m: float
