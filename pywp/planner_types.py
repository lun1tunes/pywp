from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pywp.classification import TRAJECTORY_REVERSE_DIRECTION


class PlanningError(RuntimeError):
    """Raised when trajectory planning fails due to infeasible constraints or geometry."""

    pass


__all__ = [
    "PlanningError",
    "SectionGeometry",
    "ProfileParameters",
    "PostEntrySection",
    "TurnSearchSettings",
    "OptimizationOutcome",
    "TurnSolveResult",
    "CandidateOptimizationEvaluation",
    "EndpointState",
    "ProfileEndpointEvaluation",
    "ProgressCallback",
    "_emit_progress",
    "_scaled_progress_callback",
]


@dataclass(frozen=True)
class SectionGeometry:
    s1_m: float
    z1_m: float
    ds_13_m: float
    dz_13_m: float
    azimuth_entry_deg: float
    azimuth_surface_t1_deg: float
    inc_entry_deg: float
    inc_required_t1_t3_deg: float
    t1_cross_m: float
    t3_cross_m: float
    t1_east_m: float
    t1_north_m: float
    t1_tvd_m: float

    def is_coplanar(self, tolerance_m: float) -> bool:
        return bool(
            abs(self.t1_cross_m) <= tolerance_m and abs(self.t3_cross_m) <= tolerance_m
        )

    def is_zero_azimuth_turn(self, target_direction: str, tolerance_m: float) -> bool:
        if str(target_direction) == TRAJECTORY_REVERSE_DIRECTION:
            return False
        return self.is_coplanar(tolerance_m=tolerance_m)


@dataclass(frozen=True)
class ProfileParameters:
    kop_vertical_m: float
    inc_entry_deg: float
    inc_required_t1_t3_deg: float
    inc_hold_deg: float
    dls_build1_deg_per_30m: float
    dls_build2_deg_per_30m: float
    build1_length_m: float
    hold_length_m: float
    build2_length_m: float
    horizontal_length_m: float
    horizontal_adjust_length_m: float
    horizontal_hold_length_m: float
    horizontal_inc_deg: float
    horizontal_dls_deg_per_30m: float
    azimuth_hold_deg: float
    azimuth_entry_deg: float
    profile_family: str = "unified"
    build1_controls: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def md_t1_m(self) -> float:
        return float(
            self.kop_vertical_m
            + self.build1_length_m
            + self.hold_length_m
            + self.build2_length_m
        )

    @property
    def md_total_m(self) -> float:
        return float(self.md_t1_m + self.horizontal_length_m)


@dataclass(frozen=True)
class PostEntrySection:
    total_length_m: float
    transition_length_m: float
    hold_length_m: float
    hold_inc_deg: float
    transition_dls_deg_per_30m: float


@dataclass(frozen=True)
class TurnSearchSettings:
    restart_index: int
    search_depth_scale: float
    seed_lattice_points: int
    local_max_nfev: int
    de_maxiter: int
    de_popsize: int


@dataclass(frozen=True)
class OptimizationOutcome:
    mode: str
    status: str
    objective_value: float
    theoretical_lower_bound: float
    absolute_gap_value: float
    relative_gap_pct: float
    seeds_used: int
    runs_used: int


@dataclass(frozen=True)
class TurnSolveResult:
    params: ProfileParameters
    optimization: OptimizationOutcome


@dataclass(frozen=True)
class CandidateOptimizationEvaluation:
    candidate: ProfileParameters | None
    t1_miss_m: float
    md_total_m: float
    kop_vertical_m: float
    t1_margin_m: float
    build1_margin_m: float
    build2_margin_m: float
    max_inc_margin_deg: float
    horizontal_dls_margin_deg_per_30m: float

    @property
    def feasible(self) -> bool:
        tolerance = 1e-9
        return bool(
            self.candidate is not None
            and self.t1_margin_m >= -tolerance
            and self.build1_margin_m >= -tolerance
            and self.build2_margin_m >= -tolerance
            and self.max_inc_margin_deg >= -tolerance
            and self.horizontal_dls_margin_deg_per_30m >= -tolerance
        )


@dataclass(frozen=True)
class EndpointState:
    md_m: float
    east_m: float
    north_m: float
    tvd_m: float
    inc_deg: float
    azi_deg: float


@dataclass(frozen=True)
class ProfileEndpointEvaluation:
    t1: EndpointState
    t3: EndpointState


ProgressCallback = Callable[[str, float], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    message: str,
    fraction: float,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, float(max(0.0, min(1.0, fraction))))


def _scaled_progress_callback(
    progress_callback: ProgressCallback | None,
    start_fraction: float,
    end_fraction: float,
) -> ProgressCallback | None:
    if progress_callback is None:
        return None
    start = float(max(0.0, min(1.0, start_fraction)))
    end = float(max(start, min(1.0, end_fraction)))
    span = end - start

    def wrapped(message: str, local_fraction: float) -> None:
        local = float(max(0.0, min(1.0, local_fraction)))
        progress_callback(message, start + span * local)

    return wrapped
