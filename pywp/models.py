from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


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

    max_total_md_m: float = 12000.0

    dls_limits_deg_per_30m: Dict[str, float] = field(
        default_factory=lambda: {
            "VERTICAL": 1.0,
            "BUILD": 10.0,
            "HOLD": 2.0,
        }
    )


@dataclass(frozen=True)
class PlannerResult:
    stations: "pandas.DataFrame"
    summary: Dict[str, float]
    azimuth_deg: float
    md_t1_m: float
