from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, minimize_scalar
from scipy.stats import qmc

from pywp.classification import (
    TRAJECTORY_REVERSE_DIRECTION,
    TRAJECTORY_SAME_DIRECTION,
    WellClassification,
    classify_trajectory_type,
    classify_well,
    complexity_label,
    trajectory_type_label,
)
from pywp.mcm import add_dls, compute_positions_min_curv, dogleg_angle_rad, minimum_curvature_increment
from pywp.models import (
    ALLOWED_TURN_SOLVER_MODES,
    ALLOWED_OBJECTIVE_MODES,
    OBJECTIVE_MAXIMIZE_HOLD,
    OBJECTIVE_MINIMIZE_BUILD_DLS,
    PlannerResult,
    Point3D,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    TrajectoryConfig,
)
from pywp.segments import BuildSegment, HoldSegment, HorizontalSegment, VerticalSegment
from pywp.trajectory import WellTrajectory

# Directional-planning formulas use standard circular-arc relations:
# arc length L = R * dI(rad), ds = R * (cos(I0)-cos(I1)), dTVD = R * (sin(I1)-sin(I0)).
# DLS/R conversion: DLS(deg/30m) = 30 * (180/pi) / R.
# References used in this task:
# - IADC drilling lexicon (minimum curvature): https://iadclexicon.org/minimum-curvature-method/
# - IADC drilling lexicon (dogleg severity): https://iadclexicon.org/dogleg-severity/
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
SMALL = 1e-9


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class SectionGeometry:
    s1_m: float
    z1_m: float
    ds_13_m: float
    dz_13_m: float
    azimuth_entry_deg: float
    azimuth_surface_t1_deg: float
    inc_entry_deg: float
    t1_cross_m: float
    t3_cross_m: float
    t1_east_m: float
    t1_north_m: float
    t1_tvd_m: float


@dataclass(frozen=True)
class ProfileParameters:
    trajectory_type: str
    kop_vertical_m: float
    reverse_inc_deg: float
    reverse_hold_length_m: float
    reverse_dls_deg_per_30m: float
    inc_entry_deg: float
    inc_hold_deg: float
    dls_build1_deg_per_30m: float
    dls_build2_deg_per_30m: float
    build1_length_m: float
    hold_length_m: float
    build2_length_m: float
    horizontal_length_m: float
    azimuth_hold_deg: float
    azimuth_entry_deg: float

    @property
    def reverse_build_length_m(self) -> float:
        if self.reverse_inc_deg <= SMALL or self.reverse_dls_deg_per_30m <= SMALL:
            return 0.0
        return float(_radius_from_dls(self.reverse_dls_deg_per_30m) * self.reverse_inc_deg * DEG2RAD)

    @property
    def reverse_total_length_m(self) -> float:
        return float(2.0 * self.reverse_build_length_m + self.reverse_hold_length_m)

    @property
    def md_t1_m(self) -> float:
        return float(
            self.kop_vertical_m
            + self.reverse_total_length_m
            + self.build1_length_m
            + self.hold_length_m
            + self.build2_length_m
        )

    @property
    def md_total_m(self) -> float:
        return float(self.md_t1_m + self.horizontal_length_m)


class TrajectoryPlanner:
    def plan(
        self,
        surface: Point3D,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
    ) -> PlannerResult:
        _validate_config(config)

        geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
        horizontal_offset_t1_m = _horizontal_offset(surface=surface, point=t1)
        trajectory_type = classify_trajectory_type(gv_m=geometry.z1_m, horizontal_offset_t1_m=horizontal_offset_t1_m)

        if trajectory_type == TRAJECTORY_REVERSE_DIRECTION:
            if _is_geometry_coplanar(geometry=geometry, tolerance_m=config.pos_tolerance_m):
                params = _solve_reverse_direction_profile(geometry=geometry, config=config)
            else:
                params = _solve_reverse_direction_profile_with_turn(geometry=geometry, config=config)
        else:
            if _is_geometry_coplanar(geometry=geometry, tolerance_m=config.pos_tolerance_m):
                params = _solve_same_direction_profile(geometry=geometry, config=config)
            else:
                params = _solve_same_direction_profile_with_turn(geometry=geometry, config=config)

        trajectory = _build_trajectory(params=params)

        control = compute_positions_min_curv(
            trajectory.stations(md_step_m=config.md_step_control_m),
            start=surface,
        )
        control = add_dls(control)

        classification = classify_well(
            gv_m=geometry.z1_m,
            horizontal_offset_t1_m=horizontal_offset_t1_m,
            hold_inc_deg=params.inc_hold_deg,
        )

        summary = _build_summary(
            df=control,
            t1=t1,
            t3=t3,
            md_t1_m=params.md_t1_m,
            params=params,
            horizontal_offset_t1_m=horizontal_offset_t1_m,
            classification=classification,
            config=config,
        )
        _assert_solution_is_valid(summary=summary, config=config)

        output = compute_positions_min_curv(trajectory.stations(md_step_m=config.md_step_m), start=surface)
        output = add_dls(output)

        return PlannerResult(
            stations=output,
            summary=summary,
            azimuth_deg=geometry.azimuth_entry_deg,
            md_t1_m=params.md_t1_m,
        )


def _build_section_geometry(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    config: TrajectoryConfig,
) -> SectionGeometry:
    azimuth_entry_deg = _azimuth_deg_from_points(t1=t1, t3=t3)
    azimuth_surface_t1_deg = _azimuth_deg_from_pair(surface=surface, target=t1)
    s1_m, c1_m, z1_m = _project_to_section_axis(surface=surface, point=t1, azimuth_deg=azimuth_entry_deg)
    s3_m, c3_m, z3_m = _project_to_section_axis(surface=surface, point=t3, azimuth_deg=azimuth_entry_deg)

    if z1_m <= 0.0:
        raise PlanningError("t1 must be below surface in TVD.")

    ds_13_m = s3_m - s1_m
    dz_13_m = z3_m - z1_m
    if ds_13_m <= 0.0:
        raise PlanningError("Invalid t1->t3 geometry: along-section offset must be positive.")

    inc_entry_deg = _inclination_from_displacement(ds_13=ds_13_m, dz_13=dz_13_m)
    if abs(inc_entry_deg - config.entry_inc_target_deg) > config.entry_inc_tolerance_deg:
        raise PlanningError(
            f"INC at t1 from target geometry is {inc_entry_deg:.2f} deg, "
            f"outside required {config.entry_inc_target_deg:.1f}±{config.entry_inc_tolerance_deg:.1f} deg."
        )

    return SectionGeometry(
        s1_m=float(s1_m),
        z1_m=float(z1_m),
        ds_13_m=float(ds_13_m),
        dz_13_m=float(dz_13_m),
        azimuth_entry_deg=float(azimuth_entry_deg),
        azimuth_surface_t1_deg=float(azimuth_surface_t1_deg),
        inc_entry_deg=float(inc_entry_deg),
        t1_cross_m=float(c1_m),
        t3_cross_m=float(c3_m),
        t1_east_m=float(t1.x - surface.x),
        t1_north_m=float(t1.y - surface.y),
        t1_tvd_m=float(t1.z - surface.z),
    )


def _is_geometry_coplanar(geometry: SectionGeometry, tolerance_m: float) -> bool:
    return bool(abs(geometry.t1_cross_m) <= tolerance_m and abs(geometry.t3_cross_m) <= tolerance_m)


def _solve_same_direction_profile(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
) -> ProfileParameters:
    lower, upper = _effective_dls_search_bounds(
        config=config,
        constrained_segments=("BUILD1", "BUILD2"),
    )

    kop_candidates = _iter_kop_candidates(geometry=geometry, config=config)
    best: ProfileParameters | None = None
    for kop_vertical_m in kop_candidates:
        candidate = _optimize_profile_for_objective(
            lower_dls=lower,
            upper_dls=upper,
            profile_builder=lambda dls: _profile_same_direction(
                geometry=geometry,
                dls_build_deg_per_30m=dls,
                kop_vertical_m=kop_vertical_m,
                min_structural_segment_m=config.min_structural_segment_m,
            ),
            objective_mode=config.objective_mode,
            config=config,
        )
        if candidate is None:
            continue
        best = _select_better_candidate(current=best, candidate=candidate, objective_mode=config.objective_mode)

    if best is not None:
        return best

    available_z_for_builds = geometry.z1_m - float(config.kop_min_vertical_m)
    if available_z_for_builds <= SMALL:
        raise PlanningError(
            "No valid profile: minimum vertical before KOP is too deep for t1 TVD. "
            f"t1 TVD is {geometry.z1_m:.2f} m, requested minimum vertical is {config.kop_min_vertical_m:.2f} m."
        )

    required_dls = _required_dls_for_t1_reach(
        s1_m=geometry.s1_m,
        z1_m=available_z_for_builds,
        inc_entry_deg=geometry.inc_entry_deg,
    )
    raise PlanningError(
        "No valid VERTICAL->BUILD1->HOLD->BUILD2->HORIZONTAL solution found within configured limits. "
        f"Try increasing BUILD max DLS above {required_dls:.2f} deg/30m, reducing minimum vertical before KOP, "
        "or relaxing constraints."
    )


def _solve_same_direction_profile_with_turn(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
) -> ProfileParameters:
    lower, upper = _effective_dls_search_bounds(
        config=config,
        constrained_segments=("BUILD1", "BUILD2"),
    )
    min_segment_m = max(float(config.min_structural_segment_m), SMALL)
    horizontal_length_m = float(np.hypot(geometry.ds_13_m, geometry.dz_13_m))
    if horizontal_length_m <= SMALL:
        raise PlanningError("Invalid t1->t3 geometry: horizontal section length must be positive.")

    kop_min = float(max(config.kop_min_vertical_m, 0.0))
    kop_max = float(geometry.z1_m - min_segment_m)
    if kop_min >= kop_max:
        raise PlanningError(
            "No valid profile: minimum vertical before KOP is too deep for t1 TVD. "
            f"t1 TVD is {geometry.z1_m:.2f} m, requested minimum vertical is {config.kop_min_vertical_m:.2f} m."
        )

    inc_hold_min = 0.5
    inc_hold_max = float(geometry.inc_entry_deg - 0.1)
    if inc_hold_max <= inc_hold_min:
        raise PlanningError("No valid TURN solution: INC at t1 is too small for BUILD1->BUILD2 structure.")

    mandatory_md_without_hold = kop_min + horizontal_length_m + 2.0 * min_segment_m
    hold_max = float(max(min_segment_m, config.max_total_md_m - mandatory_md_without_hold))
    if hold_max <= min_segment_m + SMALL:
        raise PlanningError(
            "No valid TURN solution found for non-coplanar targets within configured limits. "
            "Maximum total MD is too restrictive for mandatory VERTICAL/BUILD/HORIZONTAL sections."
        )

    bounds = (
        (kop_min, kop_max),
        (lower, upper),
        (inc_hold_min, inc_hold_max),
        (min_segment_m, hold_max),
        (0.0, 360.0),
    )

    base_kop = float(np.clip(geometry.z1_m * 0.25, kop_min, kop_max))
    shallow_kop = float(np.clip(geometry.z1_m * 0.12, kop_min, kop_max))
    deep_kop = float(np.clip(geometry.z1_m * 0.4, kop_min, kop_max))
    mid_dls = 0.5 * (lower + upper)

    inc_low = float(np.clip(geometry.inc_entry_deg * 0.35, inc_hold_min, inc_hold_max))
    inc_mid = float(np.clip(geometry.inc_entry_deg * 0.55, inc_hold_min, inc_hold_max))
    inc_high = float(np.clip(geometry.inc_entry_deg * 0.75, inc_hold_min, inc_hold_max))

    hold_guess = float(np.clip(max(min_segment_m, geometry.s1_m * 0.35), min_segment_m, hold_max))
    hold_low = float(np.clip(hold_guess * 0.6, min_segment_m, hold_max))
    hold_high = float(np.clip(hold_guess * 1.5, min_segment_m, hold_max))

    azimuth_mid = _mid_azimuth_deg(geometry.azimuth_surface_t1_deg, geometry.azimuth_entry_deg)
    seeds = [
        np.array([base_kop, mid_dls, inc_mid, hold_guess, geometry.azimuth_surface_t1_deg], dtype=float),
        np.array([base_kop, mid_dls, inc_mid, hold_guess, geometry.azimuth_entry_deg], dtype=float),
        np.array([base_kop, mid_dls, inc_mid, hold_guess, azimuth_mid], dtype=float),
        np.array([shallow_kop, lower, inc_low, hold_low, geometry.azimuth_surface_t1_deg], dtype=float),
        np.array([shallow_kop, upper, inc_high, hold_high, geometry.azimuth_surface_t1_deg], dtype=float),
        np.array([deep_kop, lower, inc_low, hold_low, geometry.azimuth_entry_deg], dtype=float),
        np.array([deep_kop, upper, inc_high, hold_high, geometry.azimuth_entry_deg], dtype=float),
        np.array([base_kop, lower, inc_high, hold_guess, azimuth_mid], dtype=float),
        np.array([base_kop, upper, inc_low, hold_guess, azimuth_mid], dtype=float),
    ]
    def profile_builder(values: np.ndarray) -> ProfileParameters | None:
        kop_vertical_m, dls_build, inc_hold_deg, hold_length_m, azimuth_hold_deg = values.tolist()
        return _profile_same_direction_with_turn(
            geometry=geometry,
            dls_build_deg_per_30m=float(dls_build),
            kop_vertical_m=float(kop_vertical_m),
            inc_hold_deg=float(inc_hold_deg),
            hold_length_m=float(hold_length_m),
            azimuth_hold_deg=float(azimuth_hold_deg),
            min_structural_segment_m=min_segment_m,
        )

    best, best_miss = _solve_turn_profile_multistart(
        bounds=bounds,
        seed_vectors=seeds,
        qmc_samples=int(max(config.turn_solver_qmc_samples, 0)),
        qmc_seed=42,
        max_local_starts=int(max(config.turn_solver_local_starts, 1)),
        profile_builder=profile_builder,
        objective_mode=config.objective_mode,
        turn_solver_mode=config.turn_solver_mode,
        config=config,
        target_point_east_north_tvd=(geometry.t1_east_m, geometry.t1_north_m, geometry.t1_tvd_m),
    )
    if best is not None:
        return best

    raise PlanningError(
        "No valid TURN solution found for non-coplanar targets within configured limits. "
        f"Closest miss to t1 is {best_miss:.2f} m. "
        "Try increasing BUILD max DLS, reducing minimum vertical before KOP, or relaxing tolerance."
    )


def _solve_reverse_direction_profile(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
) -> ProfileParameters:
    lower, upper = _effective_dls_search_bounds(
        config=config,
        constrained_segments=("BUILD_REV", "DROP_REV", "BUILD1", "BUILD2"),
    )

    kop_candidates = _iter_kop_candidates(geometry=geometry, config=config)
    reverse_inc_candidates = _iter_reverse_inc_candidates(config=config)

    best: ProfileParameters | None = None
    for kop_vertical_m in kop_candidates:
        for reverse_inc_deg in reverse_inc_candidates:
            candidate = _optimize_profile_for_objective(
                lower_dls=lower,
                upper_dls=upper,
                profile_builder=lambda dls: _profile_reverse_direction(
                    geometry=geometry,
                    dls_build_deg_per_30m=dls,
                    kop_vertical_m=kop_vertical_m,
                    reverse_inc_deg=reverse_inc_deg,
                    min_structural_segment_m=config.min_structural_segment_m,
                ),
                objective_mode=config.objective_mode,
                config=config,
            )
            if candidate is None:
                continue
            best = _select_better_candidate(current=best, candidate=candidate, objective_mode=config.objective_mode)

    if best is not None:
        return best

    raise PlanningError(
        "No valid reverse-direction profile found within configured limits. "
        "Try increasing BUILD max DLS, reducing minimum vertical before KOP, "
        "or widening reverse INC search limits."
    )


def _solve_reverse_direction_profile_with_turn(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
) -> ProfileParameters:
    lower, upper = _effective_dls_search_bounds(
        config=config,
        constrained_segments=("BUILD_REV", "DROP_REV", "BUILD1", "BUILD2"),
    )
    min_segment_m = max(float(config.min_structural_segment_m), SMALL)
    horizontal_length_m = float(np.hypot(geometry.ds_13_m, geometry.dz_13_m))
    if horizontal_length_m <= SMALL:
        raise PlanningError("Invalid t1->t3 geometry: horizontal section length must be positive.")

    kop_min = float(max(config.kop_min_vertical_m, 0.0))
    kop_max = float(geometry.z1_m - min_segment_m)
    if kop_min >= kop_max:
        raise PlanningError(
            "No valid reverse profile: minimum vertical before KOP is too deep for t1 TVD. "
            f"t1 TVD is {geometry.z1_m:.2f} m, requested minimum vertical is {config.kop_min_vertical_m:.2f} m."
        )

    inc_hold_min = 0.5
    inc_hold_max = float(geometry.inc_entry_deg - 0.1)
    if inc_hold_max <= inc_hold_min:
        raise PlanningError("No valid reverse TURN solution: INC at t1 is too small for BUILD1->BUILD2 structure.")

    reverse_inc_min = float(max(config.reverse_inc_min_deg, 0.1))
    reverse_inc_max = float(min(config.reverse_inc_max_deg, 89.0))
    if reverse_inc_max <= reverse_inc_min:
        raise PlanningError("No valid reverse TURN solution: reverse INC bounds are invalid.")

    mandatory_md_without_holds = kop_min + horizontal_length_m + 4.0 * min_segment_m
    hold_upper = float(max(min_segment_m, config.max_total_md_m - mandatory_md_without_holds))
    reverse_hold_upper = hold_upper
    if hold_upper <= min_segment_m + SMALL:
        raise PlanningError(
            "No valid reverse TURN solution found for non-coplanar targets within configured limits. "
            "Maximum total MD is too restrictive for mandatory reverse and forward sections."
        )

    bounds = (
        (kop_min, kop_max),
        (lower, upper),
        (reverse_inc_min, reverse_inc_max),
        (min_segment_m, reverse_hold_upper),
        (inc_hold_min, inc_hold_max),
        (min_segment_m, hold_upper),
        (0.0, 360.0),
    )

    base_kop = float(np.clip(geometry.z1_m * 0.25, kop_min, kop_max))
    shallow_kop = float(np.clip(geometry.z1_m * 0.1, kop_min, kop_max))
    deep_kop = float(np.clip(geometry.z1_m * 0.4, kop_min, kop_max))
    mid_dls = 0.5 * (lower + upper)

    reverse_mid = float(np.clip(0.5 * (reverse_inc_min + reverse_inc_max), reverse_inc_min, reverse_inc_max))
    reverse_low = float(np.clip(reverse_inc_min * 1.2, reverse_inc_min, reverse_inc_max))
    reverse_high = float(np.clip(reverse_inc_max * 0.8, reverse_inc_min, reverse_inc_max))
    inc_low = float(np.clip(geometry.inc_entry_deg * 0.35, inc_hold_min, inc_hold_max))
    inc_mid = float(np.clip(geometry.inc_entry_deg * 0.55, inc_hold_min, inc_hold_max))
    inc_high = float(np.clip(geometry.inc_entry_deg * 0.75, inc_hold_min, inc_hold_max))

    hold_guess = float(np.clip(max(min_segment_m, geometry.s1_m * 0.3), min_segment_m, hold_upper))
    reverse_hold_guess = float(np.clip(max(min_segment_m, geometry.s1_m * 0.2), min_segment_m, reverse_hold_upper))
    hold_low = float(np.clip(hold_guess * 0.6, min_segment_m, hold_upper))
    hold_high = float(np.clip(hold_guess * 1.6, min_segment_m, hold_upper))
    reverse_hold_low = float(np.clip(reverse_hold_guess * 0.5, min_segment_m, reverse_hold_upper))
    reverse_hold_high = float(np.clip(reverse_hold_guess * 1.8, min_segment_m, reverse_hold_upper))

    azimuth_mid = _mid_azimuth_deg(geometry.azimuth_surface_t1_deg, geometry.azimuth_entry_deg)
    seeds = [
        np.array(
            [base_kop, mid_dls, reverse_mid, reverse_hold_guess, inc_mid, hold_guess, geometry.azimuth_surface_t1_deg],
            dtype=float,
        ),
        np.array(
            [base_kop, mid_dls, reverse_mid, reverse_hold_guess, inc_mid, hold_guess, geometry.azimuth_entry_deg],
            dtype=float,
        ),
        np.array([base_kop, mid_dls, reverse_mid, reverse_hold_guess, inc_mid, hold_guess, azimuth_mid], dtype=float),
        np.array(
            [shallow_kop, lower, reverse_low, reverse_hold_low, inc_low, hold_low, geometry.azimuth_surface_t1_deg],
            dtype=float,
        ),
        np.array(
            [deep_kop, upper, reverse_high, reverse_hold_high, inc_high, hold_high, geometry.azimuth_entry_deg],
            dtype=float,
        ),
    ]
    def profile_builder(values: np.ndarray) -> ProfileParameters | None:
        (
            kop_vertical_m,
            dls_build,
            reverse_inc_deg,
            reverse_hold_length_m,
            inc_hold_deg,
            hold_length_m,
            azimuth_hold_deg,
        ) = values.tolist()
        return _profile_reverse_direction_with_turn(
            geometry=geometry,
            dls_build_deg_per_30m=float(dls_build),
            kop_vertical_m=float(kop_vertical_m),
            reverse_inc_deg=float(reverse_inc_deg),
            reverse_hold_length_m=float(reverse_hold_length_m),
            inc_hold_deg=float(inc_hold_deg),
            hold_length_m=float(hold_length_m),
            azimuth_hold_deg=float(azimuth_hold_deg),
            min_structural_segment_m=min_segment_m,
        )

    best, best_miss = _solve_turn_profile_multistart(
        bounds=bounds,
        seed_vectors=seeds,
        qmc_samples=int(max(config.turn_solver_qmc_samples, 36)),
        qmc_seed=42,
        max_local_starts=int(max(config.turn_solver_local_starts, 12)),
        profile_builder=profile_builder,
        objective_mode=config.objective_mode,
        turn_solver_mode=config.turn_solver_mode,
        config=config,
        target_point_east_north_tvd=(geometry.t1_east_m, geometry.t1_north_m, geometry.t1_tvd_m),
    )
    if best is not None:
        return best

    raise PlanningError(
        "No valid reverse TURN solution found for non-coplanar targets within configured limits. "
        f"Closest miss to t1 is {best_miss:.2f} m. "
        "Try increasing BUILD max DLS, reducing minimum vertical before KOP, or relaxing tolerance."
    )


def _solve_turn_profile_multistart(
    bounds: tuple[tuple[float, float], ...],
    seed_vectors: list[np.ndarray],
    qmc_samples: int,
    qmc_seed: int,
    max_local_starts: int,
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    objective_mode: str,
    turn_solver_mode: str,
    config: TrajectoryConfig,
    target_point_east_north_tvd: tuple[float, float, float],
) -> tuple[ProfileParameters | None, float]:
    # For TURN cases we solve endpoint matching as bounded nonlinear least-squares in 3D:
    # residuals = [dE, dN, dTVD, max(0, MD_excess)] with multiple starts.
    # A feasible candidate is then selected by the configured engineering objective.
    lower = np.array([pair[0] for pair in bounds], dtype=float)
    upper = np.array([pair[1] for pair in bounds], dtype=float)
    pos_tol = max(float(config.pos_tolerance_m), 1e-9)
    md_scale = max(float(config.max_total_md_m), 1.0)
    target_e, target_n, target_z = target_point_east_north_tvd

    all_seed_vectors = _seed_vectors_with_qmc(
        deterministic_seeds=seed_vectors,
        lower_bounds=lower,
        upper_bounds=upper,
        qmc_samples=qmc_samples,
        qmc_seed=qmc_seed,
    )

    def residuals(values: np.ndarray) -> np.ndarray:
        candidate = profile_builder(values)
        if candidate is None:
            return np.full(4, 1e6, dtype=float)
        end_e, end_n, end_z = _estimate_t1_endpoint_for_profile(candidate)
        md_excess = max(0.0, float(candidate.md_total_m - config.max_total_md_m))
        return np.array(
            [
                (end_e - target_e) / pos_tol,
                (end_n - target_n) / pos_tol,
                (end_z - target_z) / pos_tol,
                md_excess / md_scale,
            ],
            dtype=float,
        )

    best_candidate: ProfileParameters | None = None
    best_miss = np.inf
    scored_starts: list[tuple[float, np.ndarray]] = []

    for seed in all_seed_vectors:
        clipped_seed = _clip_to_bounds(seed, bounds=bounds)
        start_residuals = residuals(clipped_seed)
        start_score = float(np.linalg.norm(start_residuals))
        scored_starts.append((start_score, clipped_seed))

        candidate = profile_builder(clipped_seed)
        if candidate is None:
            continue
        endpoint = _estimate_t1_endpoint_for_profile(candidate)
        miss = _distance_3d(
            endpoint[0],
            endpoint[1],
            endpoint[2],
            target_e,
            target_n,
            target_z,
        )
        best_miss = min(best_miss, miss)
        if miss <= config.pos_tolerance_m + SMALL and _is_candidate_feasible(candidate=candidate, config=config):
            best_candidate = _select_better_candidate(
                current=best_candidate,
                candidate=candidate,
                objective_mode=objective_mode,
            )

    scored_starts.sort(key=lambda item: item[0])
    local_starts = [seed for _, seed in scored_starts[: max(1, int(max_local_starts))]]

    if turn_solver_mode == TURN_SOLVER_DE_HYBRID:
        de_result = differential_evolution(
            func=lambda vector: float(np.linalg.norm(residuals(np.asarray(vector, dtype=float)))),
            bounds=list(bounds),
            strategy="best1bin",
            maxiter=35,
            popsize=8,
            tol=1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=qmc_seed,
            init="latinhypercube",
            polish=False,
            updating="deferred",
            workers=1,
        )
        if de_result.success and np.all(np.isfinite(de_result.x)):
            local_starts = [_clip_to_bounds(np.asarray(de_result.x, dtype=float), bounds=bounds), *local_starts]
    elif turn_solver_mode != TURN_SOLVER_LEAST_SQUARES:
        raise PlanningError(f"Unsupported TURN solver mode: {turn_solver_mode}")

    # Deduplicate starts while preserving order.
    deduped: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for seed in local_starts:
        key = tuple(np.round(seed, decimals=8).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seed)
    local_starts = deduped[: max(1, int(max_local_starts) + 1)]

    for start in local_starts:
        solution = least_squares(
            residuals,
            x0=start,
            bounds=(lower, upper),
            method="trf",
            jac="2-point",
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=300,
        )
        probes = [start]
        if solution.success and np.all(np.isfinite(solution.x)):
            probes.append(_clip_to_bounds(np.asarray(solution.x, dtype=float), bounds=bounds))

        for probe in probes:
            candidate = profile_builder(probe)
            if candidate is None:
                continue
            endpoint = _estimate_t1_endpoint_for_profile(candidate)
            miss = _distance_3d(
                endpoint[0],
                endpoint[1],
                endpoint[2],
                target_e,
                target_n,
                target_z,
            )
            best_miss = min(best_miss, miss)
            if miss > config.pos_tolerance_m + SMALL:
                continue
            if not _is_candidate_feasible(candidate=candidate, config=config):
                continue
            best_candidate = _select_better_candidate(
                current=best_candidate,
                candidate=candidate,
                objective_mode=objective_mode,
            )

    return best_candidate, float(best_miss)


def _seed_vectors_with_qmc(
    deterministic_seeds: list[np.ndarray],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    qmc_samples: int,
    qmc_seed: int,
) -> list[np.ndarray]:
    vectors = [np.asarray(seed, dtype=float) for seed in deterministic_seeds]
    sample_count = int(max(qmc_samples, 0))
    if sample_count <= 0:
        return vectors
    if np.any(upper_bounds <= lower_bounds + SMALL):
        return vectors

    sampler = qmc.LatinHypercube(d=len(lower_bounds), seed=qmc_seed)
    unit = sampler.random(n=sample_count)
    scaled = qmc.scale(unit, l_bounds=lower_bounds, u_bounds=upper_bounds)
    vectors.extend(np.asarray(row, dtype=float) for row in scaled)
    return vectors


def _profile_same_direction(
    geometry: SectionGeometry,
    dls_build_deg_per_30m: float,
    kop_vertical_m: float,
    min_structural_segment_m: float,
) -> ProfileParameters | None:
    if dls_build_deg_per_30m <= 0.0 or kop_vertical_m < 0.0:
        return None

    z_after_preamble_m = geometry.z1_m - kop_vertical_m
    if z_after_preamble_m <= SMALL:
        return None

    return _build_profile_from_effective_targets(
        geometry=geometry,
        dls_build_deg_per_30m=dls_build_deg_per_30m,
        s_to_t1_m=geometry.s1_m,
        z_to_t1_m=z_after_preamble_m,
        kop_vertical_m=kop_vertical_m,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        reverse_inc_deg=0.0,
        reverse_hold_length_m=0.0,
        reverse_dls_deg_per_30m=0.0,
        min_structural_segment_m=min_structural_segment_m,
    )


def _profile_reverse_direction(
    geometry: SectionGeometry,
    dls_build_deg_per_30m: float,
    kop_vertical_m: float,
    reverse_inc_deg: float,
    min_structural_segment_m: float,
) -> ProfileParameters | None:
    if dls_build_deg_per_30m <= 0.0 or kop_vertical_m < 0.0:
        return None
    if reverse_inc_deg <= SMALL or reverse_inc_deg >= 89.9:
        return None

    radius_m = _radius_from_dls(dls_build_deg_per_30m)
    reverse_inc_rad = reverse_inc_deg * DEG2RAD
    sin_reverse = np.sin(reverse_inc_rad)
    cos_reverse = np.cos(reverse_inc_rad)

    reverse_base_offset_m = 2.0 * radius_m * (1.0 - cos_reverse)
    reverse_base_depth_m = 2.0 * radius_m * sin_reverse

    z_after_base_reverse_m = geometry.z1_m - kop_vertical_m - reverse_base_depth_m
    if z_after_base_reverse_m <= SMALL:
        return None

    inc_entry_rad = geometry.inc_entry_deg * DEG2RAD
    forward_min_offset_m = radius_m * (1.0 - np.cos(inc_entry_rad))
    required_extra_offset_m = max(forward_min_offset_m - (geometry.s1_m + reverse_base_offset_m) + SMALL, 0.0)

    if sin_reverse <= SMALL:
        return None

    reverse_hold_length_m = required_extra_offset_m / sin_reverse
    reverse_total_depth_m = reverse_base_depth_m + reverse_hold_length_m * cos_reverse
    z_after_reverse_m = geometry.z1_m - kop_vertical_m - reverse_total_depth_m
    if z_after_reverse_m <= SMALL:
        return None

    s_after_reverse_m = geometry.s1_m + reverse_base_offset_m + reverse_hold_length_m * sin_reverse

    candidate = _build_profile_from_effective_targets(
        geometry=geometry,
        dls_build_deg_per_30m=dls_build_deg_per_30m,
        s_to_t1_m=s_after_reverse_m,
        z_to_t1_m=z_after_reverse_m,
        kop_vertical_m=kop_vertical_m,
        trajectory_type=TRAJECTORY_REVERSE_DIRECTION,
        reverse_inc_deg=reverse_inc_deg,
        reverse_hold_length_m=reverse_hold_length_m,
        reverse_dls_deg_per_30m=dls_build_deg_per_30m,
        min_structural_segment_m=min_structural_segment_m,
    )
    if candidate is None:
        return None

    if candidate.reverse_total_length_m <= SMALL:
        return None

    return candidate


def _profile_reverse_direction_with_turn(
    geometry: SectionGeometry,
    dls_build_deg_per_30m: float,
    kop_vertical_m: float,
    reverse_inc_deg: float,
    reverse_hold_length_m: float,
    inc_hold_deg: float,
    hold_length_m: float,
    azimuth_hold_deg: float,
    min_structural_segment_m: float,
) -> ProfileParameters | None:
    if dls_build_deg_per_30m <= SMALL or kop_vertical_m < 0.0:
        return None
    if reverse_inc_deg <= SMALL or reverse_inc_deg >= 89.5:
        return None
    if hold_length_m <= SMALL or reverse_hold_length_m <= SMALL:
        return None
    if inc_hold_deg <= SMALL or inc_hold_deg >= geometry.inc_entry_deg - SMALL:
        return None

    horizontal_length_m = float(np.hypot(geometry.ds_13_m, geometry.dz_13_m))
    if horizontal_length_m <= SMALL:
        return None

    radius_m = _radius_from_dls(dls_build_deg_per_30m)
    min_segment = max(float(min_structural_segment_m), SMALL)

    reverse_build_length_m = float(radius_m * np.radians(reverse_inc_deg))
    build1_length_m = float(radius_m * np.radians(inc_hold_deg))
    build2_dogleg_rad = float(
        dogleg_angle_rad(
            inc_hold_deg,
            azimuth_hold_deg,
            geometry.inc_entry_deg,
            geometry.azimuth_entry_deg,
        )
    )
    build2_length_m = float(radius_m * build2_dogleg_rad)

    if (
        reverse_build_length_m < min_segment - SMALL
        or reverse_hold_length_m < min_segment - SMALL
        or build1_length_m < min_segment - SMALL
        or hold_length_m < min_segment - SMALL
        or build2_length_m < min_segment - SMALL
    ):
        return None

    return ProfileParameters(
        trajectory_type=TRAJECTORY_REVERSE_DIRECTION,
        kop_vertical_m=float(kop_vertical_m),
        reverse_inc_deg=float(reverse_inc_deg),
        reverse_hold_length_m=float(reverse_hold_length_m),
        reverse_dls_deg_per_30m=float(dls_build_deg_per_30m),
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_hold_deg=float(inc_hold_deg),
        dls_build1_deg_per_30m=float(dls_build_deg_per_30m),
        dls_build2_deg_per_30m=float(dls_build_deg_per_30m),
        build1_length_m=build1_length_m,
        hold_length_m=float(hold_length_m),
        build2_length_m=build2_length_m,
        horizontal_length_m=horizontal_length_m,
        azimuth_hold_deg=_normalize_azimuth_deg(azimuth_hold_deg),
        azimuth_entry_deg=_normalize_azimuth_deg(geometry.azimuth_entry_deg),
    )


def _build_profile_from_effective_targets(
    geometry: SectionGeometry,
    dls_build_deg_per_30m: float,
    s_to_t1_m: float,
    z_to_t1_m: float,
    kop_vertical_m: float,
    trajectory_type: str,
    reverse_inc_deg: float,
    reverse_hold_length_m: float,
    reverse_dls_deg_per_30m: float,
    min_structural_segment_m: float,
) -> ProfileParameters | None:
    horizontal_length_m = float(np.hypot(geometry.ds_13_m, geometry.dz_13_m))
    if horizontal_length_m <= SMALL:
        return None

    inc_entry_rad = geometry.inc_entry_deg * DEG2RAD
    if inc_entry_rad <= SMALL or inc_entry_rad >= (np.pi / 2.0) - SMALL:
        return None

    radius_m = _radius_from_dls(dls_build_deg_per_30m)

    # Aggregate forward build displacement to t1 for BUILD1 (0->Ih) + BUILD2 (Ih->Ientry):
    # ds_build_total = R*(1-cos(Ientry)), dz_build_total = R*sin(Ientry)
    a_m = s_to_t1_m - radius_m * (1.0 - np.cos(inc_entry_rad))
    b_m = z_to_t1_m - radius_m * np.sin(inc_entry_rad)
    if a_m <= SMALL or b_m <= SMALL:
        return None

    inc_hold_rad = float(np.arctan2(a_m, b_m))
    if inc_hold_rad <= SMALL or inc_hold_rad >= inc_entry_rad - SMALL:
        return None
    inc_hold_deg = float(inc_hold_rad * RAD2DEG)

    hold_length_m = float(np.hypot(a_m, b_m))
    build1_length_m = float(radius_m * inc_hold_rad)
    build2_length_m = float(radius_m * (inc_entry_rad - inc_hold_rad))
    if hold_length_m <= SMALL or build1_length_m <= SMALL or build2_length_m <= SMALL:
        return None
    min_segment = max(float(min_structural_segment_m), SMALL)
    if (
        hold_length_m < min_segment - SMALL
        or build1_length_m < min_segment - SMALL
        or build2_length_m < min_segment - SMALL
    ):
        return None

    return ProfileParameters(
        trajectory_type=trajectory_type,
        kop_vertical_m=float(kop_vertical_m),
        reverse_inc_deg=float(max(reverse_inc_deg, 0.0)),
        reverse_hold_length_m=float(max(reverse_hold_length_m, 0.0)),
        reverse_dls_deg_per_30m=float(max(reverse_dls_deg_per_30m, 0.0)),
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_hold_deg=inc_hold_deg,
        dls_build1_deg_per_30m=float(dls_build_deg_per_30m),
        dls_build2_deg_per_30m=float(dls_build_deg_per_30m),
        build1_length_m=build1_length_m,
        hold_length_m=hold_length_m,
        build2_length_m=build2_length_m,
        horizontal_length_m=horizontal_length_m,
        azimuth_hold_deg=float(geometry.azimuth_entry_deg),
        azimuth_entry_deg=float(geometry.azimuth_entry_deg),
    )


def _profile_same_direction_with_turn(
    geometry: SectionGeometry,
    dls_build_deg_per_30m: float,
    kop_vertical_m: float,
    inc_hold_deg: float,
    hold_length_m: float,
    azimuth_hold_deg: float,
    min_structural_segment_m: float,
) -> ProfileParameters | None:
    if dls_build_deg_per_30m <= SMALL or kop_vertical_m < 0.0:
        return None
    if hold_length_m <= SMALL:
        return None
    if inc_hold_deg <= SMALL or inc_hold_deg >= geometry.inc_entry_deg - SMALL:
        return None

    horizontal_length_m = float(np.hypot(geometry.ds_13_m, geometry.dz_13_m))
    if horizontal_length_m <= SMALL:
        return None

    min_segment = max(float(min_structural_segment_m), SMALL)
    radius_m = _radius_from_dls(dls_build_deg_per_30m)
    build1_dogleg_rad = float(dogleg_angle_rad(0.0, azimuth_hold_deg, inc_hold_deg, azimuth_hold_deg))
    build2_dogleg_rad = float(
        dogleg_angle_rad(
            inc_hold_deg,
            azimuth_hold_deg,
            geometry.inc_entry_deg,
            geometry.azimuth_entry_deg,
        )
    )

    build1_length_m = float(radius_m * build1_dogleg_rad)
    build2_length_m = float(radius_m * build2_dogleg_rad)
    if (
        build1_length_m < min_segment - SMALL
        or hold_length_m < min_segment - SMALL
        or build2_length_m < min_segment - SMALL
    ):
        return None

    return ProfileParameters(
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        kop_vertical_m=float(kop_vertical_m),
        reverse_inc_deg=0.0,
        reverse_hold_length_m=0.0,
        reverse_dls_deg_per_30m=0.0,
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_hold_deg=float(inc_hold_deg),
        dls_build1_deg_per_30m=float(dls_build_deg_per_30m),
        dls_build2_deg_per_30m=float(dls_build_deg_per_30m),
        build1_length_m=build1_length_m,
        hold_length_m=float(hold_length_m),
        build2_length_m=build2_length_m,
        horizontal_length_m=horizontal_length_m,
        azimuth_hold_deg=_normalize_azimuth_deg(azimuth_hold_deg),
        azimuth_entry_deg=_normalize_azimuth_deg(geometry.azimuth_entry_deg),
    )


def _estimate_t1_endpoint_for_profile(profile: ProfileParameters) -> tuple[float, float, float]:
    north_m = 0.0
    east_m = 0.0
    tvd_m = float(profile.kop_vertical_m)

    if profile.trajectory_type == TRAJECTORY_REVERSE_DIRECTION and profile.reverse_total_length_m > SMALL:
        opposite_azimuth_deg = _opposite_azimuth(profile.azimuth_hold_deg)
        reverse_build_length_m = float(profile.reverse_build_length_m)

        dn, de, dz = minimum_curvature_increment(
            md1_m=0.0,
            inc1_deg=0.0,
            azi1_deg=opposite_azimuth_deg,
            md2_m=reverse_build_length_m,
            inc2_deg=profile.reverse_inc_deg,
            azi2_deg=opposite_azimuth_deg,
        )
        north_m += dn
        east_m += de
        tvd_m += dz

        dn, de, dz = minimum_curvature_increment(
            md1_m=0.0,
            inc1_deg=profile.reverse_inc_deg,
            azi1_deg=opposite_azimuth_deg,
            md2_m=float(profile.reverse_hold_length_m),
            inc2_deg=profile.reverse_inc_deg,
            azi2_deg=opposite_azimuth_deg,
        )
        north_m += dn
        east_m += de
        tvd_m += dz

        dn, de, dz = minimum_curvature_increment(
            md1_m=0.0,
            inc1_deg=profile.reverse_inc_deg,
            azi1_deg=opposite_azimuth_deg,
            md2_m=reverse_build_length_m,
            inc2_deg=0.0,
            azi2_deg=opposite_azimuth_deg,
        )
        north_m += dn
        east_m += de
        tvd_m += dz

    dn, de, dz = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=0.0,
        azi1_deg=profile.azimuth_hold_deg,
        md2_m=float(profile.build1_length_m),
        inc2_deg=profile.inc_hold_deg,
        azi2_deg=profile.azimuth_hold_deg,
    )
    north_m += dn
    east_m += de
    tvd_m += dz

    dn, de, dz = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=profile.inc_hold_deg,
        azi1_deg=profile.azimuth_hold_deg,
        md2_m=float(profile.hold_length_m),
        inc2_deg=profile.inc_hold_deg,
        azi2_deg=profile.azimuth_hold_deg,
    )
    north_m += dn
    east_m += de
    tvd_m += dz

    dn, de, dz = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=profile.inc_hold_deg,
        azi1_deg=profile.azimuth_hold_deg,
        md2_m=float(profile.build2_length_m),
        inc2_deg=profile.inc_entry_deg,
        azi2_deg=profile.azimuth_entry_deg,
    )
    north_m += dn
    east_m += de
    tvd_m += dz

    return float(east_m), float(north_m), float(tvd_m)


def _clip_to_bounds(values: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> np.ndarray:
    clipped = np.asarray(values, dtype=float).copy()
    for idx, (low, high) in enumerate(bounds):
        clipped[idx] = float(np.clip(clipped[idx], low, high))
    return clipped


def _normalize_azimuth_deg(azimuth_deg: float) -> float:
    return float(np.mod(float(azimuth_deg), 360.0))


def _shortest_azimuth_delta_deg(azimuth_from_deg: float, azimuth_to_deg: float) -> float:
    delta = _normalize_azimuth_deg(azimuth_to_deg - azimuth_from_deg)
    if delta > 180.0:
        delta -= 360.0
    return float(delta)


def _mid_azimuth_deg(azimuth_from_deg: float, azimuth_to_deg: float) -> float:
    return _normalize_azimuth_deg(azimuth_from_deg + 0.5 * _shortest_azimuth_delta_deg(azimuth_from_deg, azimuth_to_deg))


def _iter_kop_candidates(geometry: SectionGeometry, config: TrajectoryConfig) -> np.ndarray:
    kop_min = float(max(config.kop_min_vertical_m, 0.0))
    kop_max = float(geometry.z1_m - SMALL)
    if kop_min >= kop_max:
        return np.array([], dtype=float)

    grid_size = int(max(config.kop_search_grid_size, 2))
    if grid_size == 2:
        return np.array([kop_min, kop_max], dtype=float)
    return np.linspace(kop_min, kop_max, grid_size, dtype=float)


def _iter_reverse_inc_candidates(config: TrajectoryConfig) -> np.ndarray:
    inc_min = float(max(config.reverse_inc_min_deg, 0.0))
    inc_max = float(min(config.reverse_inc_max_deg, 89.5))
    if inc_min >= inc_max:
        return np.array([], dtype=float)

    grid_size = int(max(config.reverse_inc_grid_size, 2))
    return np.linspace(inc_min, inc_max, grid_size, dtype=float)


def _effective_dls_search_bounds(config: TrajectoryConfig, constrained_segments: tuple[str, ...]) -> tuple[float, float]:
    lower = float(config.dls_build_min_deg_per_30m)
    upper = float(config.dls_build_max_deg_per_30m)

    for segment in constrained_segments:
        segment_limit = config.dls_limits_deg_per_30m.get(segment)
        if segment_limit is None:
            continue
        upper = min(upper, float(segment_limit))

    if upper <= SMALL:
        segments = ", ".join(constrained_segments)
        raise PlanningError(
            f"No feasible BUILD DLS: constrained upper bound is {upper:.2f} deg/30m for segments [{segments}]."
        )
    if lower > upper + SMALL:
        segments = ", ".join(constrained_segments)
        raise PlanningError(
            "No feasible BUILD DLS interval after applying segment limits: "
            f"min={lower:.2f} deg/30m, max={upper:.2f} deg/30m for [{segments}]."
        )
    return lower, upper


def _candidate_rank(candidate: ProfileParameters, objective_mode: str) -> tuple[float, float, float, float]:
    if objective_mode == OBJECTIVE_MINIMIZE_BUILD_DLS:
        return (
            candidate.dls_build1_deg_per_30m,
            -candidate.hold_length_m,
            -candidate.kop_vertical_m,
            candidate.reverse_total_length_m,
        )

    return (
        -candidate.hold_length_m,
        candidate.dls_build1_deg_per_30m,
        -candidate.kop_vertical_m,
        candidate.reverse_total_length_m,
    )


def _select_better_candidate(
    current: ProfileParameters | None,
    candidate: ProfileParameters,
    objective_mode: str,
) -> ProfileParameters:
    if current is None:
        return candidate
    if _candidate_rank(candidate, objective_mode) < _candidate_rank(current, objective_mode):
        return candidate
    return current


def _optimize_profile_for_objective(
    lower_dls: float,
    upper_dls: float,
    profile_builder: Callable[[float], ProfileParameters | None],
    objective_mode: str,
    config: TrajectoryConfig,
) -> ProfileParameters | None:
    if objective_mode == OBJECTIVE_MINIMIZE_BUILD_DLS:
        return _optimize_minimize_build_dls(
            lower_dls=lower_dls,
            upper_dls=upper_dls,
            profile_builder=profile_builder,
            config=config,
        )

    return _optimize_maximize_hold(
        lower_dls=lower_dls,
        upper_dls=upper_dls,
        profile_builder=profile_builder,
        config=config,
    )


def _optimize_maximize_hold(
    lower_dls: float,
    upper_dls: float,
    profile_builder: Callable[[float], ProfileParameters | None],
    config: TrajectoryConfig,
) -> ProfileParameters | None:
    def objective(dls_build: float) -> float:
        candidate = profile_builder(float(dls_build))
        if not _is_candidate_feasible(candidate=candidate, config=config):
            return 1e12 + abs(float(dls_build) - lower_dls)
        return -candidate.hold_length_m

    optimum = minimize_scalar(objective, bounds=(lower_dls, upper_dls), method="bounded", options={"xatol": 1e-5})
    probes = [lower_dls, upper_dls, float(optimum.x), float(optimum.x) - 1e-3, float(optimum.x) + 1e-3]

    best: ProfileParameters | None = None
    best_hold = -1.0
    for dls in probes:
        dls_clamped = float(np.clip(dls, lower_dls, upper_dls))
        candidate = profile_builder(dls_clamped)
        if not _is_candidate_feasible(candidate=candidate, config=config):
            continue
        if candidate.hold_length_m > best_hold:
            best = candidate
            best_hold = candidate.hold_length_m
    return best


def _optimize_minimize_build_dls(
    lower_dls: float,
    upper_dls: float,
    profile_builder: Callable[[float], ProfileParameters | None],
    config: TrajectoryConfig,
) -> ProfileParameters | None:
    lower_candidate = profile_builder(lower_dls)
    if _is_candidate_feasible(candidate=lower_candidate, config=config):
        return lower_candidate

    grid = np.linspace(lower_dls, upper_dls, 400, dtype=float)
    candidates: list[ProfileParameters | None] = [profile_builder(float(dls)) for dls in grid]
    feasible_mask = [bool(_is_candidate_feasible(candidate=cand, config=config)) for cand in candidates]
    try:
        first_feasible_idx = feasible_mask.index(True)
    except ValueError:
        return None

    if first_feasible_idx == 0:
        refined_candidate = candidates[0]
        return refined_candidate if isinstance(refined_candidate, ProfileParameters) else None

    lo = float(grid[first_feasible_idx - 1])
    hi = float(grid[first_feasible_idx])
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        mid_candidate = profile_builder(mid)
        if _is_candidate_feasible(candidate=mid_candidate, config=config):
            hi = mid
        else:
            lo = mid

    candidate = profile_builder(hi)
    if _is_candidate_feasible(candidate=candidate, config=config):
        return candidate
    return None


def _is_candidate_feasible(candidate: ProfileParameters | None, config: TrajectoryConfig) -> bool:
    if candidate is None:
        return False
    return bool(candidate.md_total_m <= config.max_total_md_m + SMALL)


def _required_dls_for_t1_reach(s1_m: float, z1_m: float, inc_entry_deg: float) -> float:
    inc_entry_rad = inc_entry_deg * DEG2RAD
    one_minus_cos = max(1.0 - np.cos(inc_entry_rad), SMALL)
    sin_entry = max(np.sin(inc_entry_rad), SMALL)
    radius_limit_m = min(s1_m / one_minus_cos, z1_m / sin_entry)
    return _dls_from_radius(radius_limit_m)


def _build_trajectory(params: ProfileParameters) -> WellTrajectory:
    segments = [
        VerticalSegment(
            length_m=params.kop_vertical_m,
            azi_deg=params.azimuth_hold_deg,
            name="VERTICAL",
        )
    ]

    if params.trajectory_type == TRAJECTORY_REVERSE_DIRECTION and params.reverse_total_length_m > SMALL:
        opposite_azimuth_deg = _opposite_azimuth(params.azimuth_hold_deg)
        segments.append(
            BuildSegment(
                inc_from_deg=0.0,
                inc_to_deg=params.reverse_inc_deg,
                dls_deg_per_30m=params.reverse_dls_deg_per_30m,
                azi_deg=opposite_azimuth_deg,
                name="BUILD_REV",
            )
        )
        if params.reverse_hold_length_m > SMALL:
            segments.append(
                HoldSegment(
                    length_m=params.reverse_hold_length_m,
                    inc_deg=params.reverse_inc_deg,
                    azi_deg=opposite_azimuth_deg,
                    name="HOLD_REV",
                )
            )
        segments.append(
            BuildSegment(
                inc_from_deg=params.reverse_inc_deg,
                inc_to_deg=0.0,
                dls_deg_per_30m=params.reverse_dls_deg_per_30m,
                azi_deg=opposite_azimuth_deg,
                name="DROP_REV",
            )
        )

    segments.extend(
        [
            BuildSegment(
                inc_from_deg=0.0,
                inc_to_deg=params.inc_hold_deg,
                dls_deg_per_30m=params.dls_build1_deg_per_30m,
                azi_deg=params.azimuth_hold_deg,
                name="BUILD1",
            ),
            HoldSegment(
                length_m=params.hold_length_m,
                inc_deg=params.inc_hold_deg,
                azi_deg=params.azimuth_hold_deg,
                name="HOLD",
            ),
            BuildSegment(
                inc_from_deg=params.inc_hold_deg,
                inc_to_deg=params.inc_entry_deg,
                dls_deg_per_30m=params.dls_build2_deg_per_30m,
                azi_deg=params.azimuth_hold_deg,
                azi_to_deg=params.azimuth_entry_deg,
                name="BUILD2",
            ),
            HorizontalSegment(
                length_m=params.horizontal_length_m,
                inc_deg=params.inc_entry_deg,
                azi_deg=params.azimuth_entry_deg,
                name="HORIZONTAL",
            ),
        ]
    )
    return WellTrajectory(segments=segments)


def _build_summary(
    df: pd.DataFrame,
    t1: Point3D,
    t3: Point3D,
    md_t1_m: float,
    params: ProfileParameters,
    horizontal_offset_t1_m: float,
    classification: WellClassification,
    config: TrajectoryConfig,
) -> dict[str, float | str]:
    t1_idx = int((df["MD_m"] - md_t1_m).abs().idxmin())
    t1_row = df.loc[t1_idx]
    t3_row = df.iloc[-1]

    distance_t1 = _distance_3d(t1_row["X_m"], t1_row["Y_m"], t1_row["Z_m"], t1.x, t1.y, t1.z)
    distance_t3 = _distance_3d(t3_row["X_m"], t3_row["Y_m"], t3_row["Z_m"], t3.x, t3.y, t3.z)

    max_dls = float(np.nanmax(df["DLS_deg_per_30m"].to_numpy()))
    summary: dict[str, float | str] = {
        "distance_t1_m": float(distance_t1),
        "distance_t3_m": float(distance_t3),
        "kop_vertical_m": float(params.kop_vertical_m),
        "kop_md_m": float(params.kop_vertical_m),
        "entry_inc_deg": float(t1_row["INC_deg"]),
        "entry_inc_target_deg": float(config.entry_inc_target_deg),
        "entry_inc_tolerance_deg": float(config.entry_inc_tolerance_deg),
        "hold_inc_deg": float(params.inc_hold_deg),
        "max_dls_total_deg_per_30m": max_dls,
        "md_total_m": float(df["MD_m"].iloc[-1]),
        "t1_horizontal_offset_m": float(horizontal_offset_t1_m),
        "horizontal_length_m": float(params.horizontal_length_m),
        "trajectory_type": trajectory_type_label(classification.trajectory_type),
        "well_complexity": complexity_label(classification.complexity),
        "well_complexity_by_offset": complexity_label(classification.complexity_by_offset),
        "well_complexity_by_hold": complexity_label(classification.complexity_by_hold),
        "reverse_inc_deg": float(params.reverse_inc_deg),
        "reverse_hold_length_m": float(params.reverse_hold_length_m),
        "hold_azimuth_deg": float(params.azimuth_hold_deg),
        "entry_azimuth_deg": float(params.azimuth_entry_deg),
        "azimuth_turn_deg": float(abs(_shortest_azimuth_delta_deg(params.azimuth_hold_deg, params.azimuth_entry_deg))),
    }

    summary["class_reverse_offset_min_m"] = float(classification.limits.reverse_min_m)
    summary["class_reverse_offset_max_m"] = float(classification.limits.reverse_max_m)
    summary["class_offset_ordinary_max_m"] = float(classification.limits.ordinary_offset_max_m)
    summary["class_offset_complex_max_m"] = float(classification.limits.complex_offset_max_m)
    summary["class_hold_ordinary_max_deg"] = float(classification.limits.hold_ordinary_max_deg)
    summary["class_hold_complex_max_deg"] = float(classification.limits.hold_complex_max_deg)

    for segment, limit in config.dls_limits_deg_per_30m.items():
        seg_max = float(df.loc[df["segment"] == segment, "DLS_deg_per_30m"].max(skipna=True))
        if np.isnan(seg_max):
            seg_max = 0.0
        summary[f"max_dls_{segment.lower()}_deg_per_30m"] = seg_max
        summary[f"dls_limit_{segment.lower()}_deg_per_30m"] = float(limit)
    return summary


def _assert_solution_is_valid(summary: dict[str, float | str], config: TrajectoryConfig) -> None:
    if float(summary["distance_t1_m"]) > config.pos_tolerance_m:
        raise PlanningError("Failed to hit t1 within tolerance.")
    if float(summary["distance_t3_m"]) > config.pos_tolerance_m:
        raise PlanningError("Failed to hit t3 within tolerance.")
    if abs(float(summary["entry_inc_deg"]) - config.entry_inc_target_deg) > config.entry_inc_tolerance_deg + 1e-6:
        raise PlanningError("Entry inclination at t1 is outside required target range.")
    for segment, limit in config.dls_limits_deg_per_30m.items():
        actual = float(summary.get(f"max_dls_{segment.lower()}_deg_per_30m", 0.0))
        if actual > limit + 1e-6:
            raise PlanningError(f"DLS limit exceeded on segment {segment}: {actual:.2f} > {limit:.2f}")


def _validate_config(config: TrajectoryConfig) -> None:
    if config.md_step_m <= 0.0 or config.md_step_control_m <= 0.0:
        raise PlanningError("MD steps must be positive.")
    if config.pos_tolerance_m <= 0.0:
        raise PlanningError("Position tolerance must be positive.")
    if config.kop_min_vertical_m < 0.0:
        raise PlanningError("kop_min_vertical_m must be non-negative.")
    if config.kop_search_grid_size < 2:
        raise PlanningError("kop_search_grid_size must be >= 2.")
    if config.reverse_inc_min_deg < 0.0:
        raise PlanningError("reverse_inc_min_deg must be non-negative.")
    if config.reverse_inc_max_deg > 89.5:
        raise PlanningError("reverse_inc_max_deg must be <= 89.5.")
    if config.reverse_inc_min_deg >= config.reverse_inc_max_deg:
        raise PlanningError("reverse_inc_min_deg must be < reverse_inc_max_deg.")
    if config.reverse_inc_grid_size < 2:
        raise PlanningError("reverse_inc_grid_size must be >= 2.")
    if config.entry_inc_tolerance_deg < 0.0:
        raise PlanningError("entry_inc_tolerance_deg must be non-negative.")
    if config.dls_build_min_deg_per_30m > config.dls_build_max_deg_per_30m:
        raise PlanningError("dls_build_min_deg_per_30m cannot exceed dls_build_max_deg_per_30m.")
    if config.max_total_md_m <= 0.0:
        raise PlanningError("max_total_md_m must be positive.")
    if config.min_structural_segment_m <= 0.0:
        raise PlanningError("min_structural_segment_m must be positive.")
    if config.min_structural_segment_m < config.md_step_control_m:
        raise PlanningError("min_structural_segment_m must be >= md_step_control_m.")
    if config.objective_mode not in ALLOWED_OBJECTIVE_MODES:
        allowed = ", ".join(ALLOWED_OBJECTIVE_MODES)
        raise PlanningError(f"objective_mode must be one of: {allowed}.")
    if config.turn_solver_mode not in ALLOWED_TURN_SOLVER_MODES:
        allowed_turn = ", ".join(ALLOWED_TURN_SOLVER_MODES)
        raise PlanningError(f"turn_solver_mode must be one of: {allowed_turn}.")
    if config.turn_solver_qmc_samples < 0:
        raise PlanningError("turn_solver_qmc_samples must be non-negative.")
    if config.turn_solver_local_starts < 1:
        raise PlanningError("turn_solver_local_starts must be >= 1.")
    for segment, limit in config.dls_limits_deg_per_30m.items():
        if limit < 0.0:
            raise PlanningError(f"DLS limit for segment {segment} cannot be negative.")


def _azimuth_deg_from_points(t1: Point3D, t3: Point3D) -> float:
    return _azimuth_deg_from_pair(surface=t1, target=t3)


def _azimuth_deg_from_pair(surface: Point3D, target: Point3D) -> float:
    dn = target.y - surface.y
    de = target.x - surface.x
    if np.isclose(dn, 0.0) and np.isclose(de, 0.0):
        raise PlanningError("Azimuth is undefined for overlapping plan coordinates.")
    azimuth_rad = np.arctan2(de, dn)
    return float(np.mod(azimuth_rad * RAD2DEG, 360.0))


def _project_to_section_axis(surface: Point3D, point: Point3D, azimuth_deg: float) -> tuple[float, float, float]:
    dn = point.y - surface.y
    de = point.x - surface.x
    az = azimuth_deg * DEG2RAD
    along = dn * np.cos(az) + de * np.sin(az)
    cross = -dn * np.sin(az) + de * np.cos(az)
    z = point.z - surface.z
    return float(along), float(cross), float(z)


def _inclination_from_displacement(ds_13: float, dz_13: float) -> float:
    # Inclination is measured from vertical: tan(INC) = horizontal / vertical.
    return float(np.degrees(np.arctan2(ds_13, dz_13)))


def _dls_from_radius(radius_m: float) -> float:
    return float(30.0 * RAD2DEG / radius_m)


def _radius_from_dls(dls_deg_per_30m: float) -> float:
    return float(30.0 * RAD2DEG / dls_deg_per_30m)


def _distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))


def _horizontal_offset(surface: Point3D, point: Point3D) -> float:
    return float(np.hypot(point.x - surface.x, point.y - surface.y))


def _opposite_azimuth(azimuth_deg: float) -> float:
    return float(np.mod(azimuth_deg + 180.0, 360.0))
