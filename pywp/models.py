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
    angle_tolerance_deg: float = 1.0

    inc_entry_min_deg: float = 60.0
    inc_entry_max_deg: float = 86.0

    dls_build1_min_deg_per_30m: float = 2.0
    dls_build1_max_deg_per_30m: float = 8.0
    dls_build2_min_deg_per_30m: float = 2.0
    dls_build2_max_deg_per_30m: float = 10.0

    max_total_md_m: float = 12000.0

    dls_limits_deg_per_30m: Dict[str, float] = field(
        default_factory=lambda: {
            "VERTICAL": 1.0,
            "BUILD1": 8.0,
            "HOLD1": 2.0,
            "HOLD2": 2.0,
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
