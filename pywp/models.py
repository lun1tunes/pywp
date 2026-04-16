from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

import pandas as pd
from pydantic import Field, field_validator, model_validator

from pywp.constants import SMALL
from pywp.planner_types import PlanningError
from pywp.pydantic_base import FrozenArbitraryModel, FrozenModel

OPTIMIZATION_NONE = "none"
OPTIMIZATION_MINIMIZE_MD = "minimize_md"
OPTIMIZATION_MINIMIZE_KOP = "minimize_kop"
OPTIMIZATION_ANTI_COLLISION_AVOIDANCE = "anti_collision_avoidance"
ALLOWED_OPTIMIZATION_MODES = (
    OPTIMIZATION_NONE,
    OPTIMIZATION_MINIMIZE_MD,
    OPTIMIZATION_MINIMIZE_KOP,
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
)
OptimizationMode = Literal[
    "none",
    "minimize_md",
    "minimize_kop",
    "anti_collision_avoidance",
]
TURN_SOLVER_LEAST_SQUARES = "least_squares"
TURN_SOLVER_DE_HYBRID = "de_hybrid"
ALLOWED_TURN_SOLVER_MODES = (TURN_SOLVER_LEAST_SQUARES, TURN_SOLVER_DE_HYBRID)
TurnSolverMode = Literal["least_squares", "de_hybrid"]

INTERPOLATION_SLERP = "slerp"
INTERPOLATION_RODRIGUES = "rodrigues"
ALLOWED_INTERPOLATION_METHODS = (INTERPOLATION_RODRIGUES, INTERPOLATION_SLERP)
InterpolationMethod = Literal["rodrigues", "slerp"]

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
FiniteScalar = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteScalar = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFiniteScalar = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
EntryIncTargetScalar = Annotated[float, Field(gt=0.0, lt=90.0, allow_inf_nan=False)]
MaxIncScalar = Annotated[float, Field(gt=0.0, le=120.0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
SegmentDlsLimitScalar = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


def build_segment_dls_limits_deg_per_30m(
    build_dls_max_deg_per_30m: float,
) -> dict[str, float]:
    """Build default segment DLS limits from a single BUILD max value."""

    build_limit = float(max(build_dls_max_deg_per_30m, 0.0))
    limits: dict[str, float] = {}
    for segment_name in _SEGMENT_DLS_ORDER:
        if segment_name in _SEGMENT_DLS_BUILD_CONTROLLED:
            limits[segment_name] = build_limit
        else:
            limits[segment_name] = float(_SEGMENT_DLS_FIXED_LIMITS[segment_name])
    return limits


class Point3D(FrozenModel):
    """Cartesian point in meters: X=East, Y=North, Z=TVD (positive down)."""

    x: FiniteScalar
    y: FiniteScalar
    z: FiniteScalar

    def __init__(self, *args: float, **data: float):
        if args:
            if data:
                raise TypeError("Point3D accepts either positional x/y/z or keyword fields, not both.")
            if len(args) != 3:
                raise TypeError(f"Point3D expected 3 positional arguments, got {len(args)}.")
            data = {"x": float(args[0]), "y": float(args[1]), "z": float(args[2])}
        super().__init__(**data)


class TrajectoryConfig(FrozenModel):
    md_step_m: PositiveFiniteScalar = 10.0
    md_step_control_m: PositiveFiniteScalar = 2.0
    lateral_tolerance_m: PositiveFiniteScalar = 30.0
    vertical_tolerance_m: PositiveFiniteScalar = 2.0
    entry_inc_target_deg: EntryIncTargetScalar = 86.0
    entry_inc_tolerance_deg: NonNegativeFiniteScalar = 2.0
    max_inc_deg: MaxIncScalar = 95.0

    dls_build_min_deg_per_30m: NonNegativeFiniteScalar = 0.0
    dls_build_max_deg_per_30m: NonNegativeFiniteScalar = (
        DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
    )
    kop_min_vertical_m: NonNegativeFiniteScalar = 550.0

    # Post-processing MD threshold for user-facing validation only.
    # Does not participate in solver search/optimization constraints.
    max_total_md_postcheck_m: PositiveFiniteScalar = 6500.0
    optimization_mode: OptimizationMode = OPTIMIZATION_MINIMIZE_MD
    turn_solver_mode: TurnSolverMode = TURN_SOLVER_LEAST_SQUARES
    turn_solver_max_restarts: NonNegativeInt = 2
    interpolation_method: InterpolationMethod = INTERPOLATION_RODRIGUES
    # Minimum MD span for BUILD/HOLD/BUILD sections. 30 m aligns with the common DLS reference interval (deg/30m).
    min_structural_segment_m: PositiveFiniteScalar = 30.0

    dls_limits_deg_per_30m: dict[str, SegmentDlsLimitScalar] = Field(
        default_factory=lambda: build_segment_dls_limits_deg_per_30m(
            DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_removed_legacy_fields(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        legacy_pos_tolerance = payload.pop("pos_tolerance_m", None)
        if legacy_pos_tolerance is not None:
            if "lateral_tolerance_m" not in payload:
                payload["lateral_tolerance_m"] = legacy_pos_tolerance
            if "vertical_tolerance_m" not in payload:
                payload["vertical_tolerance_m"] = legacy_pos_tolerance
        payload.pop("objective_mode", None)
        payload.pop("max_total_md_m", None)
        return payload

    @field_validator("dls_limits_deg_per_30m", mode="before")
    @classmethod
    def _normalize_dls_limits(
        cls, value: object
    ) -> dict[str, SegmentDlsLimitScalar]:
        defaults = build_segment_dls_limits_deg_per_30m(
            DEFAULT_BUILD_DLS_MAX_DEG_PER_30M
        )
        if value is None:
            return defaults
        try:
            raw_limits = dict(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "dls_limits_deg_per_30m must be a mapping of segment names to limits."
            ) from exc

        unknown_segments = sorted(
            {str(key) for key in raw_limits.keys()}.difference(_SEGMENT_DLS_ORDER)
        )
        if unknown_segments:
            raise ValueError(
                "Unsupported DLS segment names: "
                + ", ".join(unknown_segments)
            )

        limits = dict(defaults)
        for key, raw_value in raw_limits.items():
            limits[str(key)] = float(raw_value)
        return {
            segment_name: float(limits[segment_name]) for segment_name in _SEGMENT_DLS_ORDER
        }

    def validate_for_planning(self) -> None:
        """Validate configuration limits that cannot be checked via isolated Pydantic fields."""
        if self.md_step_m <= 0.0 or self.md_step_control_m <= 0.0:
            raise PlanningError("MD steps must be positive.")
        if self.lateral_tolerance_m <= 0.0:
            raise PlanningError("lateral_tolerance_m must be positive.")
        if self.vertical_tolerance_m <= 0.0:
            raise PlanningError("vertical_tolerance_m must be positive.")
        if self.entry_inc_target_deg <= 0.0 or self.entry_inc_target_deg >= 90.0:
            raise PlanningError("entry_inc_target_deg must be in (0, 90).")
        if self.kop_min_vertical_m < 0.0:
            raise PlanningError("kop_min_vertical_m must be non-negative.")
        if self.entry_inc_tolerance_deg < 0.0:
            raise PlanningError("entry_inc_tolerance_deg must be non-negative.")
        if self.max_inc_deg <= 0.0 or self.max_inc_deg > 120.0:
            raise PlanningError("max_inc_deg must be in (0, 120].")
        if self.entry_inc_target_deg > self.max_inc_deg + SMALL:
            raise PlanningError("entry_inc_target_deg cannot exceed max_inc_deg.")
        if self.dls_build_min_deg_per_30m < 0.0:
            raise PlanningError("dls_build_min_deg_per_30m cannot be negative.")
        if self.dls_build_max_deg_per_30m < 0.0:
            raise PlanningError("dls_build_max_deg_per_30m cannot be negative.")
        if self.dls_build_min_deg_per_30m > self.dls_build_max_deg_per_30m:
            raise PlanningError(
                "dls_build_min_deg_per_30m cannot exceed dls_build_max_deg_per_30m."
            )
        if self.max_total_md_postcheck_m <= 0.0:
            raise PlanningError("max_total_md_postcheck_m must be positive.")
        if self.min_structural_segment_m <= 0.0:
            raise PlanningError("min_structural_segment_m must be positive.")
        if self.min_structural_segment_m < self.md_step_control_m:
            raise PlanningError("min_structural_segment_m must be >= md_step_control_m.")
        try:
            turn_solver_max_restarts = float(self.turn_solver_max_restarts)
        except (TypeError, ValueError) as exc:
            raise PlanningError(
                "turn_solver_max_restarts must be a non-negative integer."
            ) from exc
        if int(turn_solver_max_restarts) != turn_solver_max_restarts:
            raise PlanningError("turn_solver_max_restarts must be an integer.")
        if int(turn_solver_max_restarts) < 0:
            raise PlanningError("turn_solver_max_restarts must be non-negative.")
        for segment, limit in self.dls_limits_deg_per_30m.items():
            if limit < 0.0:
                raise PlanningError(f"DLS limit for segment {segment} cannot be negative.")

    @model_validator(mode="after")
    def _validate_and_sync_limits(self) -> "TrajectoryConfig":
        if float(self.entry_inc_target_deg) > float(self.max_inc_deg):
            raise ValueError("entry_inc_target_deg cannot exceed max_inc_deg.")
        if float(self.dls_build_min_deg_per_30m) > float(self.dls_build_max_deg_per_30m):
            raise ValueError(
                "dls_build_min_deg_per_30m cannot exceed dls_build_max_deg_per_30m."
            )
        if float(self.min_structural_segment_m) < float(self.md_step_control_m):
            raise ValueError("min_structural_segment_m must be >= md_step_control_m.")

        current_limits = dict(self.dls_limits_deg_per_30m)
        build_limit = float(self.dls_build_max_deg_per_30m)
        for segment_name in _SEGMENT_DLS_BUILD_CONTROLLED:
            current_limits[segment_name] = build_limit
        object.__setattr__(
            self,
            "dls_limits_deg_per_30m",
            {
                segment_name: float(current_limits[segment_name])
                for segment_name in _SEGMENT_DLS_ORDER
            },
        )
        return self

    @property
    def pos_tolerance_m(self) -> float:
        """Legacy compatibility alias.

        Prefer the explicit `lateral_tolerance_m` and `vertical_tolerance_m`.
        """

        return float(max(self.lateral_tolerance_m, self.vertical_tolerance_m))


class PlannerResult(FrozenArbitraryModel):
    stations: pd.DataFrame
    summary: SummaryDict
    azimuth_deg: float
    md_t1_m: float
