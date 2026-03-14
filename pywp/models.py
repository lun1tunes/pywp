from __future__ import annotations

from typing import Dict, Literal

import pandas as pd
from pydantic import Field, field_validator, model_validator

from pywp.pydantic_base import FrozenArbitraryModel, FrozenModel

OBJECTIVE_MINIMIZE_TOTAL_MD = "minimize_total_md"
ALLOWED_OBJECTIVE_MODES = (OBJECTIVE_MINIMIZE_TOTAL_MD,)
ObjectiveMode = Literal["minimize_total_md"]
TURN_SOLVER_LEAST_SQUARES = "least_squares"
TURN_SOLVER_DE_HYBRID = "de_hybrid"
ALLOWED_TURN_SOLVER_MODES = (TURN_SOLVER_LEAST_SQUARES, TURN_SOLVER_DE_HYBRID)
TurnSolverMode = Literal["least_squares", "de_hybrid"]

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
SummaryValue = float | int | str | bool
SummaryDict = dict[str, SummaryValue]


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


class Point3D(FrozenModel):
    """Cartesian point in meters: X=East, Y=North, Z=TVD (positive down)."""

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        super().__init__(x=x, y=y, z=z)


class TrajectoryConfig(FrozenModel):
    md_step_m: float = 10.0
    md_step_control_m: float = 2.0
    pos_tolerance_m: float = 2.0
    entry_inc_target_deg: float = 86.0
    entry_inc_tolerance_deg: float = 2.0
    max_inc_deg: float = 95.0

    dls_build_min_deg_per_30m: float = 0.0
    dls_build_max_deg_per_30m: float = DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
    kop_min_vertical_m: float = 550.0

    max_total_md_m: float = 12000.0
    # Post-processing MD threshold for user-facing validation only.
    # Does not participate in solver search/optimization constraints.
    max_total_md_postcheck_m: float = 6500.0
    objective_mode: ObjectiveMode = OBJECTIVE_MINIMIZE_TOTAL_MD
    turn_solver_mode: TurnSolverMode = TURN_SOLVER_LEAST_SQUARES
    turn_solver_max_restarts: int = 2
    # Minimum MD span for BUILD/HOLD/BUILD sections. 30 m aligns with the common DLS reference interval (deg/30m).
    min_structural_segment_m: float = 30.0

    dls_limits_deg_per_30m: Dict[str, float] = Field(
        default_factory=lambda: build_segment_dls_limits_deg_per_30m(
            DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
        )
    )

    @field_validator("dls_limits_deg_per_30m", mode="before")
    @classmethod
    def _normalize_dls_limits(cls, value: object) -> Dict[str, float]:
        if value is None:
            return build_segment_dls_limits_deg_per_30m(
                DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
            )
        limits = dict(value)  # type: ignore[arg-type]
        return {str(key): float(raw_value) for key, raw_value in limits.items()}

    @model_validator(mode="after")
    def _sync_build_limits(self) -> "TrajectoryConfig":
        default_limits = build_segment_dls_limits_deg_per_30m(
            DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
        )
        current_limits = dict(self.dls_limits_deg_per_30m)
        if (
            current_limits == default_limits
            and abs(
                float(self.dls_build_max_deg_per_30m)
                - float(DEFAULT_BUILD_DLS_MAX_DEG_PER_30M)
            )
            > 1e-9
        ):
            object.__setattr__(
                self,
                "dls_limits_deg_per_30m",
                build_segment_dls_limits_deg_per_30m(
                    float(self.dls_build_max_deg_per_30m)
                ),
            )
        return self


class PlannerResult(FrozenArbitraryModel):
    stations: pd.DataFrame
    summary: SummaryDict
    azimuth_deg: float
    md_t1_m: float
