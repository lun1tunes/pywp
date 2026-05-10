from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, minimize

from pywp.anticollision_optimization import (
    AntiCollisionClearanceEvaluation,
    AntiCollisionOptimizationContext,
    evaluate_candidate_anti_collision_clearance,
)
from pywp.classification import classify_trajectory_type
from pywp.mcm import (
    add_dls,
    compute_positions_min_curv,
    dogleg_angle_rad,
    minimum_curvature_increment,
)
from pywp.models import (
    ALLOWED_TURN_SOLVER_MODES,
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    OPTIMIZATION_MINIMIZE_KOP,
    OPTIMIZATION_MINIMIZE_MD,
    OPTIMIZATION_NONE,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    PlannerResult,
    Point3D,
    TrajectoryConfig,
)
from pywp.planner_geometry import (
    _build_section_geometry,
    _dls_from_radius,
    _horizontal_offset,
    _mid_azimuth_deg,
    _normalize_azimuth_deg,
    _radius_from_dls,
    _required_dls_for_t1_reach,
    _shortest_azimuth_delta_deg,
)
from pywp.planner_types import (
    CandidateOptimizationEvaluation,
    OptimizationOutcome,
    PlanningError,
    PostEntrySection,
    ProfileParameters,
    ProgressCallback,
    SectionGeometry,
    TurnSearchSettings,
    TurnSolveResult,
    _emit_progress,
    _scaled_progress_callback,
)
from pywp.planner_validation import (
    _build_validated_control_and_summary,
    _candidate_turn_deg,
    _estimate_t1_endpoint_for_profile,
    _is_candidate_feasible,
)
from pywp.trajectory import WellTrajectory

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
SMALL = 1e-9

TRAJECTORY_MODEL_LABEL = "Unified J Profile + Build + Azimuth Turn"
TURN_RESTART_GROWTH_FACTOR = 1.6
NUMERICAL_BUILD_DLS_FLOOR_DEG_PER_30M = 0.1
MD_OPTIMIZATION_THEORETICAL_GAP_FRACTION = 0.05
KOP_OPTIMIZATION_TOLERANCE_M = 1e-3
OPTIMIZATION_MAX_SEEDS = 4
OPTIMIZATION_IMPROVEMENT_TOLERANCE = 1e-6
MD_BOUNDARY_REFINEMENT_MAX_BASE_CANDIDATES = 2
MD_BOUNDARY_REFINEMENT_MAX_NFEV_FACTOR = 0.18
KOP_BOUNDARY_REFINEMENT_MAX_BASE_CANDIDATES = 2
KOP_BOUNDARY_REFINEMENT_MAX_NFEV_FACTOR = 0.18
MD_2D_REFINEMENT_MAX_BASE_CANDIDATES = 2
MD_2D_REFINEMENT_MAX_SEEDS = 8
MD_2D_REFINEMENT_MAXITER = 36
MD_BOUNDARY_EXTREMUM_BUILD_TOLERANCE = 1e-6
MD_BOUNDARY_EXTREMUM_KOP_TOLERANCE_M = 1e-3
ANTI_COLLISION_SF_TOLERANCE = 1e-3
ANTI_COLLISION_OBJECTIVE_PENALTY = 1_000.0
ANTI_COLLISION_KOP_PREFERENCE_WEIGHT = 0.25
ANTI_COLLISION_BUILD1_PREFERENCE_WEIGHT = 0.20
ANTI_COLLISION_KEEP_KOP_WEIGHT = 0.60
ANTI_COLLISION_KEEP_BUILD1_WEIGHT = 0.50
ANTI_COLLISION_BUILD2_PREFERENCE_WEIGHT = 0.18
ANTI_COLLISION_SAMPLE_STEP_M = 50.0
ANTI_COLLISION_MAX_STARTS = 12
ANTI_COLLISION_MAXITER = 72
EARLY_ANTI_COLLISION_MAX_STARTS = 8
EARLY_ANTI_COLLISION_MAXITER = 48
LATE_ANTI_COLLISION_BUILD2_MAX_STARTS = 6
LATE_ANTI_COLLISION_BUILD2_MAXITER = 48
LATE_ANTI_COLLISION_KEEP_KOP_TOLERANCE_M = 5.0
LATE_ANTI_COLLISION_KEEP_BUILD1_TOLERANCE_DEG_PER_30M = 0.05


def _target_miss_components(
    endpoint: np.ndarray,
    target_point: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    dx_m = float(endpoint[0] - target_point[0])
    dy_m = float(endpoint[1] - target_point[1])
    dz_m = float(endpoint[2] - target_point[2])
    lateral_m = float(np.hypot(dx_m, dy_m))
    vertical_m = float(abs(dz_m))
    distance_m = float(np.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m))
    return dx_m, dy_m, dz_m, lateral_m, vertical_m, distance_m


def _target_miss_within_tolerance(
    *,
    lateral_m: float,
    vertical_m: float,
    config: TrajectoryConfig,
    slack_m: float = 0.0,
) -> bool:
    slack = max(float(slack_m), 0.0)
    return bool(
        float(lateral_m) <= float(config.lateral_tolerance_m) + slack + SMALL
        and float(vertical_m) <= float(config.vertical_tolerance_m) + slack + SMALL
    )


def _target_miss_margin_m(
    *,
    lateral_m: float,
    vertical_m: float,
    config: TrajectoryConfig,
) -> float:
    return float(
        min(
            float(config.lateral_tolerance_m) - float(lateral_m),
            float(config.vertical_tolerance_m) - float(vertical_m),
        )
    )


def _normalized_target_miss(
    *,
    endpoint: np.ndarray,
    target_point: np.ndarray,
    config: TrajectoryConfig,
) -> float:
    _, _, _, lateral_m, vertical_m, _ = _target_miss_components(endpoint, target_point)
    lateral_scale = max(float(config.lateral_tolerance_m), 1e-9)
    vertical_scale = max(float(config.vertical_tolerance_m), 1e-9)
    return float(
        np.hypot(
            float(lateral_m) / lateral_scale,
            float(vertical_m) / vertical_scale,
        )
    )


class TrajectoryPlanner:
    def plan(
        self,
        surface: Point3D,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
        progress_callback: ProgressCallback | None = None,
        optimization_context: AntiCollisionOptimizationContext | None = None,
    ) -> PlannerResult:
        _emit_progress(progress_callback, "Планировщик: проверка конфигурации.", 0.03)
        config.validate_for_planning()

        _emit_progress(
            progress_callback, "Планировщик: подготовка геометрии цели.", 0.10
        )
        geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
        horizontal_offset_t1_m = _horizontal_offset(surface=surface, point=t1)
        target_direction = classify_trajectory_type(
            gv_m=float(geometry.z1_m),
            horizontal_offset_t1_m=float(horizontal_offset_t1_m),
        )

        zero_azimuth_turn = geometry.is_zero_azimuth_turn(
            target_direction=target_direction,
            tolerance_m=float(config.lateral_tolerance_m),
        )
        solver_mode_text = (
            "Солвер: единая оптимизация траектории (поворот по азимуту = 0)."
            if zero_azimuth_turn
            else "Солвер: единая оптимизация траектории с азимутальным поворотом."
        )
        _emit_progress(progress_callback, solver_mode_text, 0.18)
        (
            params,
            optimization_outcome,
            turn_search_settings,
            turn_restarts_used,
            trajectory,
            control,
            summary,
        ) = _solve_turn_with_restarts(
            surface=surface,
            t1=t1,
            t3=t3,
            geometry=geometry,
            horizontal_offset_t1_m=horizontal_offset_t1_m,
            config=config,
            optimization_context=optimization_context,
            zero_azimuth_turn=zero_azimuth_turn,
            progress_callback=_scaled_progress_callback(
                progress_callback=progress_callback,
                start_fraction=0.16,
                end_fraction=0.90,
            ),
        )

        _emit_progress(
            progress_callback, "Планировщик: формирование выходной инклинометрии.", 0.96
        )
        try:
            output = compute_positions_min_curv(
                trajectory.stations(md_step_m=config.md_step_m),
                start=surface,
            )
            output = add_dls(output)
        except ValueError as exc:
            raise PlanningError(
                "Не удалось построить выходную инклинометрию методом минимальной кривизны. "
                f"Причина: {exc}"
            ) from exc
        _emit_progress(progress_callback, "Планировщик: результат готов.", 1.00)

        return PlannerResult(
            stations=output,
            summary=summary,
            azimuth_deg=geometry.azimuth_entry_deg,
            md_t1_m=params.md_t1_m,
        )


def _profile_zero_azimuth_turn_continuous(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
    post_entry: PostEntrySection,
    build_dls_deg_per_30m: float,
) -> ProfileParameters | None:
    kop_vertical_m = _minimal_feasible_zero_azimuth_turn_kop(
        geometry=geometry,
        config=config,
        build_dls_deg_per_30m=build_dls_deg_per_30m,
    )
    if kop_vertical_m is None:
        return None
    return _build_profile_from_effective_targets(
        geometry=geometry,
        dls_build_deg_per_30m=build_dls_deg_per_30m,
        s_to_t1_m=geometry.s1_m,
        z_to_t1_m=geometry.z1_m - kop_vertical_m,
        kop_vertical_m=kop_vertical_m,
        min_build_segment_m=float(config.min_structural_segment_m),
        post_entry=post_entry,
    )


def _profile_zero_azimuth_turn_j(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
    post_entry: PostEntrySection,
    build_dls_max_deg_per_30m: float,
) -> ProfileParameters | None:
    inc_entry_rad = float(geometry.inc_entry_deg * DEG2RAD)
    one_minus_cos = float(1.0 - np.cos(inc_entry_rad))
    sin_inc = float(np.sin(inc_entry_rad))
    if one_minus_cos <= SMALL or sin_inc <= SMALL:
        return None

    radius_m = float(geometry.s1_m / one_minus_cos)
    if radius_m <= SMALL:
        return None
    dls_build = float(_dls_from_radius(radius_m))
    if dls_build > float(build_dls_max_deg_per_30m) + SMALL:
        return None

    kop_vertical_m = float(geometry.z1_m - radius_m * sin_inc)
    if kop_vertical_m < float(max(config.kop_min_vertical_m, 0.0)) - SMALL:
        return None

    build1_length_m = float(radius_m * inc_entry_rad)
    if build1_length_m < float(config.min_structural_segment_m) - SMALL:
        return None

    return ProfileParameters(
        kop_vertical_m=float(max(kop_vertical_m, 0.0)),
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_required_t1_t3_deg=float(geometry.inc_required_t1_t3_deg),
        inc_hold_deg=float(geometry.inc_entry_deg),
        dls_build1_deg_per_30m=dls_build,
        dls_build2_deg_per_30m=0.0,
        build1_length_m=build1_length_m,
        hold_length_m=0.0,
        build2_length_m=0.0,
        horizontal_length_m=float(post_entry.total_length_m),
        horizontal_adjust_length_m=float(post_entry.transition_length_m),
        horizontal_hold_length_m=float(post_entry.hold_length_m),
        horizontal_inc_deg=float(post_entry.hold_inc_deg),
        horizontal_dls_deg_per_30m=float(post_entry.transition_dls_deg_per_30m),
        azimuth_hold_deg=float(geometry.azimuth_entry_deg),
        azimuth_entry_deg=float(geometry.azimuth_entry_deg),
    )


def _minimal_feasible_zero_azimuth_turn_kop(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
    build_dls_deg_per_30m: float,
) -> float | None:
    radius_m = _radius_from_dls(build_dls_deg_per_30m)
    inc_entry_rad = float(geometry.inc_entry_deg * DEG2RAD)
    if radius_m <= SMALL or inc_entry_rad <= SMALL:
        return None

    a_m = float(geometry.s1_m - radius_m * (1.0 - np.cos(inc_entry_rad)))
    b0_m = float(geometry.z1_m - radius_m * np.sin(inc_entry_rad))
    if a_m <= SMALL or b0_m <= SMALL:
        return None

    min_build = float(max(config.min_structural_segment_m, SMALL))
    build_angle_min_rad = float(min_build / radius_m)
    if build_angle_min_rad >= inc_entry_rad - SMALL:
        return None

    kop_lower = float(max(config.kop_min_vertical_m, 0.0))
    if build_angle_min_rad > SMALL:
        kop_lower = max(kop_lower, float(b0_m - a_m / np.tan(build_angle_min_rad)))

    build2_angle_max_rad = float(inc_entry_rad - build_angle_min_rad)
    kop_upper = float(b0_m - SMALL)
    if build2_angle_max_rad <= SMALL:
        return None
    kop_upper = min(kop_upper, float(b0_m - a_m / np.tan(build2_angle_max_rad)))

    if kop_lower > kop_upper + SMALL:
        return None
    return float(np.clip(kop_lower, 0.0, kop_upper))


def _solve_turn_with_restarts(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    geometry: SectionGeometry,
    horizontal_offset_t1_m: float,
    config: TrajectoryConfig,
    optimization_context: AntiCollisionOptimizationContext | None,
    zero_azimuth_turn: bool,
    progress_callback: ProgressCallback | None = None,
) -> tuple[
    ProfileParameters,
    OptimizationOutcome,
    TurnSearchSettings,
    int,
    WellTrajectory,
    pd.DataFrame,
    dict[str, float | str],
]:
    max_attempts = int(max(config.turn_solver_max_restarts, 0)) + 1
    last_error: PlanningError | None = None

    for restart_index in range(max_attempts):
        search_settings = _turn_search_settings(restart_index=restart_index)
        attempt_start = 0.02 + 0.88 * float(restart_index / max(max_attempts, 1))
        attempt_end = 0.02 + 0.88 * float((restart_index + 1) / max(max_attempts, 1))
        _emit_progress(
            progress_callback,
            (
                f"Солвер: попытка {restart_index + 1}/{max_attempts}, "
                f"глубина поиска x{search_settings.search_depth_scale:.2f}."
            ),
            attempt_start,
        )
        attempt_progress = _scaled_progress_callback(
            progress_callback=progress_callback,
            start_fraction=attempt_start + 0.03 * (attempt_end - attempt_start),
            end_fraction=attempt_start + 0.78 * (attempt_end - attempt_start),
        )
        try:
            solve_result = _solve_turn_profile(
                geometry=geometry,
                config=config,
                surface=surface,
                optimization_context=optimization_context,
                zero_azimuth_turn=zero_azimuth_turn,
                search_settings=search_settings,
                progress_callback=attempt_progress,
            )
            params = solve_result.params
            _emit_progress(
                progress_callback,
                "Солвер: контрольный расчет и валидация.",
                attempt_start + 0.86 * (attempt_end - attempt_start),
            )
            trajectory, control, summary = _build_validated_control_and_summary(
                surface=surface,
                t1=t1,
                t3=t3,
                geometry=geometry,
                horizontal_offset_t1_m=horizontal_offset_t1_m,
                params=params,
                optimization_outcome=solve_result.optimization,
                config=config,
                turn_search_settings=search_settings,
                turn_restarts_used=restart_index,
            )
            return (
                params,
                solve_result.optimization,
                search_settings,
                restart_index,
                trajectory,
                control,
                summary,
            )
        except PlanningError as exc:
            last_error = exc
            if restart_index >= max_attempts - 1 or not _is_retryable_solver_error(
                str(exc)
            ):
                raise
            next_settings = _turn_search_settings(restart_index=restart_index + 1)
            _emit_progress(
                progress_callback,
                (
                    f"Солвер: рестарт {restart_index + 1}/{max_attempts - 1}, "
                    f"увеличиваем дискретность поиска до x{next_settings.search_depth_scale:.2f}."
                ),
                attempt_end,
            )

    if last_error is None:
        raise PlanningError("Trajectory solver failed without explicit error.")
    raise last_error


def _turn_search_settings(restart_index: int) -> TurnSearchSettings:
    level = int(max(restart_index, 0))
    depth_scale = float(TURN_RESTART_GROWTH_FACTOR**level)
    lattice_points = 0 if level == 0 else 1 + 2 * level
    return TurnSearchSettings(
        restart_index=level,
        search_depth_scale=depth_scale,
        seed_lattice_points=lattice_points,
        local_max_nfev=max(240, int(round(240.0 * depth_scale))),
        de_maxiter=max(24, int(round(24.0 * depth_scale))),
        de_popsize=max(7, int(round(7.0 * (1.45**level)))),
    )


def _is_retryable_solver_error(message: str) -> bool:
    source = str(message or "")
    if "Closest miss to t1" in source:
        return True
    if source.startswith("Failed to hit t1 within tolerance."):
        return True
    if source.startswith("Failed to hit t3 within tolerance."):
        return True
    return False


def _solve_turn_profile(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
    surface: Point3D,
    optimization_context: AntiCollisionOptimizationContext | None,
    zero_azimuth_turn: bool,
    search_settings: TurnSearchSettings,
    progress_callback: ProgressCallback | None = None,
) -> TurnSolveResult:
    build_dls_upper = _resolve_build_dls_max(
        config=config,
        constrained_segments=("BUILD1", "BUILD2"),
    )
    build_dls_lower = _effective_build_dls_lower_bound(
        config=config,
        upper_dls_deg_per_30m=build_dls_upper,
    )
    horizontal_dls = _resolve_horizontal_dls(config=config)

    post_entry = _solve_post_entry_section(
        ds_m=geometry.ds_13_m,
        dz_m=geometry.dz_13_m,
        inc_entry_deg=geometry.inc_entry_deg,
        dls_deg_per_30m=horizontal_dls,
        max_inc_deg=float(config.max_inc_deg),
    )
    if post_entry is None:
        diagnostics = _diagnose_post_entry_constraints(
            geometry=geometry,
            config=config,
            horizontal_dls_deg_per_30m=horizontal_dls,
        )
        raise PlanningError(
            "No valid trajectory solution found within configured limits. "
            "Post-entry section to t3 is not feasible."
            + _format_failure_diagnostics(diagnostics)
        )

    if build_dls_upper > build_dls_lower + SMALL:
        _emit_progress(
            progress_callback,
            (
                "Солвер: совместный поиск по геометрии и BUILD ПИ "
                f"в диапазоне [{build_dls_lower:.2f}, {build_dls_upper:.2f}] deg/30m."
            ),
            0.06,
        )

    inc_hold_min = 0.5
    inc_hold_max = float(geometry.inc_entry_deg - 0.5)
    kop_min = float(max(config.kop_min_vertical_m, 0.0))
    kop_max = float(max(geometry.z1_m - SMALL, kop_min + SMALL))
    hold_max = float(max(geometry.s1_m + geometry.z1_m, 1000.0))

    if inc_hold_max <= inc_hold_min + SMALL:
        raise PlanningError(
            "No valid trajectory solution found within configured limits. "
            "INC at t1 is too small for BUILD1->BUILD2 structure."
        )
    if kop_min >= kop_max:
        raise PlanningError(
            "No valid trajectory solution found within configured limits. "
            f"t1 TVD is {geometry.z1_m:.2f} m, requested minimum vertical is {config.kop_min_vertical_m:.2f} m."
        )

    if zero_azimuth_turn:
        bounds = (
            (0.0, 1.0),
            (kop_min, kop_max),
            (inc_hold_min, inc_hold_max),
            (0.0, hold_max),
        )
    else:
        bounds = (
            (0.0, 1.0),
            (kop_min, kop_max),
            (inc_hold_min, inc_hold_max),
            (0.0, hold_max),
            (0.0, 360.0),
        )
    target_point = np.array(
        [geometry.t1_east_m, geometry.t1_north_m, geometry.t1_tvd_m],
        dtype=float,
    )

    preferred_profiles: list[ProfileParameters] = []
    if zero_azimuth_turn:
        for build_dls in _preferred_build_dls_values(
            lower_dls_deg_per_30m=build_dls_lower,
            upper_dls_deg_per_30m=build_dls_upper,
        ):
            continuous_candidate = _profile_zero_azimuth_turn_continuous(
                geometry=geometry,
                config=config,
                post_entry=post_entry,
                build_dls_deg_per_30m=build_dls,
            )
            if continuous_candidate is not None:
                preferred_profiles.append(continuous_candidate)
            j_candidate = _profile_zero_azimuth_turn_j(
                geometry=geometry,
                config=config,
                post_entry=post_entry,
                build_dls_max_deg_per_30m=build_dls,
            )
            if j_candidate is not None:
                preferred_profiles.append(j_candidate)

    profile_builder = _make_turn_profile_builder(
        geometry=geometry,
        zero_azimuth_turn=zero_azimuth_turn,
        lower_dls_deg_per_30m=build_dls_lower,
        upper_dls_deg_per_30m=build_dls_upper,
        min_build_segment_m=float(config.min_structural_segment_m),
        post_entry=post_entry,
        split_build=False,
    )

    seed_vectors = _turn_seed_vectors(
        geometry=geometry,
        build_dls_lower_deg_per_30m=build_dls_lower,
        build_dls_upper_deg_per_30m=build_dls_upper,
        bounds=bounds,
        search_settings=search_settings,
        zero_azimuth_turn=zero_azimuth_turn,
        preferred_profiles=tuple(preferred_profiles),
    )
    _emit_progress(
        progress_callback,
        (
            f"Солвер: стартовые приближения "
            f"({len(seed_vectors)} шт., depth x{search_settings.search_depth_scale:.2f})."
        ),
        0.12,
    )

    candidates: list[ProfileParameters] = []
    best_miss = np.inf
    best_lateral_m = np.inf
    best_vertical_m = np.inf
    best_miss_build_dls = float(build_dls_upper)
    for preferred_profile in preferred_profiles:
        endpoint = np.array(
            _estimate_t1_endpoint_for_profile(preferred_profile), dtype=float
        )
        _, _, _, lateral_m, vertical_m, _ = _target_miss_components(
            endpoint, target_point
        )
        miss = _normalized_target_miss(
            endpoint=endpoint,
            target_point=target_point,
            config=config,
        )
        best_miss = min(best_miss, miss)
        if miss <= best_miss + SMALL:
            best_miss_build_dls = float(preferred_profile.dls_build1_deg_per_30m)
            best_lateral_m = float(lateral_m)
            best_vertical_m = float(vertical_m)
        if not _target_miss_within_tolerance(
            lateral_m=lateral_m,
            vertical_m=vertical_m,
            config=config,
        ):
            continue
        if not _is_candidate_feasible(candidate=preferred_profile, config=config):
            continue
        candidates.append(preferred_profile)
    if zero_azimuth_turn and candidates:
        if str(config.optimization_mode) == OPTIMIZATION_NONE:
            selected = min(
                candidates,
                key=lambda candidate: (
                    candidate.md_total_m,
                    candidate.build2_length_m,
                ),
            )
            _emit_progress(
                progress_callback,
                "Солвер: аналитический zero-turn кандидат принят без дополнительной оптимизации.",
                1.00,
            )
            return TurnSolveResult(
                params=selected,
                optimization=OptimizationOutcome(
                    mode=str(config.optimization_mode),
                    status="off",
                    objective_value=float(selected.md_total_m),
                    theoretical_lower_bound=0.0,
                    absolute_gap_value=0.0,
                    relative_gap_pct=0.0,
                    seeds_used=len(candidates),
                    runs_used=0,
                ),
            )

    if str(config.turn_solver_mode) == TURN_SOLVER_DE_HYBRID:
        _emit_progress(progress_callback, "Солвер: глобальный DE-поиск.", 0.32)
        de_result = differential_evolution(
            func=lambda vector: _turn_scalar_cost(
                values=np.asarray(vector, dtype=float),
                profile_builder=profile_builder,
                target_point=target_point,
                config=config,
            ),
            bounds=list(bounds),
            strategy="best1bin",
            maxiter=search_settings.de_maxiter,
            popsize=search_settings.de_popsize,
            tol=1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            updating="deferred",
            workers=1,
        )
        if de_result.success and np.all(np.isfinite(de_result.x)):
            seed_vectors = [
                _clip_to_bounds(np.asarray(de_result.x, dtype=float), bounds=bounds),
                *seed_vectors,
            ]
    elif str(config.turn_solver_mode) != TURN_SOLVER_LEAST_SQUARES:
        allowed = ", ".join(ALLOWED_TURN_SOLVER_MODES)
        raise PlanningError(f"turn_solver_mode must be one of: {allowed}.")

    seed_vectors = _dedupe_seed_vectors(seed_vectors)
    _emit_progress(
        progress_callback,
        (
            f"Солвер: локальная оптимизация ({len(seed_vectors)} стартов, "
            f"max_nfev={search_settings.local_max_nfev})."
        ),
        0.55,
    )

    lower = np.array([item[0] for item in bounds], dtype=float)
    upper = np.array([item[1] for item in bounds], dtype=float)
    total_starts = len(seed_vectors)
    for index, seed in enumerate(seed_vectors, start=1):
        clipped_seed = _clip_to_bounds(seed, bounds=bounds)
        solution = least_squares(
            fun=lambda values: _turn_residuals(
                values=np.asarray(values, dtype=float),
                profile_builder=profile_builder,
                target_point=target_point,
                config=config,
            ),
            x0=clipped_seed,
            bounds=(lower, upper),
            method="trf",
            jac="2-point",
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=search_settings.local_max_nfev,
        )
        probes = [clipped_seed]
        if solution.success and np.all(np.isfinite(solution.x)):
            probes.append(
                _clip_to_bounds(np.asarray(solution.x, dtype=float), bounds=bounds)
            )

        for probe in probes:
            candidate = profile_builder(probe)
            if candidate is None:
                continue
            endpoint = np.array(
                _estimate_t1_endpoint_for_profile(candidate), dtype=float
            )
            _, _, _, lateral_m, vertical_m, _ = _target_miss_components(
                endpoint, target_point
            )
            miss = _normalized_target_miss(
                endpoint=endpoint,
                target_point=target_point,
                config=config,
            )
            if miss < best_miss - SMALL or abs(miss - best_miss) <= SMALL:
                best_miss = miss
                best_miss_build_dls = float(candidate.dls_build1_deg_per_30m)
                best_lateral_m = float(lateral_m)
                best_vertical_m = float(vertical_m)
            if not _target_miss_within_tolerance(
                lateral_m=lateral_m,
                vertical_m=vertical_m,
                config=config,
            ):
                continue
            if not _is_candidate_feasible(candidate=candidate, config=config):
                continue
            candidates.append(candidate)

        _emit_progress(
            progress_callback,
            f"Солвер: локальные решатели {index}/{total_starts}.",
            0.60 + 0.36 * float(index / max(total_starts, 1)),
        )

    if candidates:
        _emit_progress(progress_callback, "Солвер: найден допустимый профиль.", 1.00)
        return _select_feasible_candidate(
            candidates=candidates,
            surface=surface,
            geometry=geometry,
            post_entry=post_entry,
            config=config,
            optimization_context=optimization_context,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=build_dls_lower,
            upper_dls_deg_per_30m=build_dls_upper,
            bounds=bounds,
            profile_builder=profile_builder,
            target_point=target_point,
            search_settings=search_settings,
            progress_callback=progress_callback,
        )

    diagnostics = _diagnose_same_direction_no_solution(
        geometry=geometry,
        config=config,
        lower_dls=build_dls_lower,
        upper_dls=build_dls_upper,
        required_dls=_required_dls_for_t1_reach(
            s1_m=geometry.s1_m,
            z1_m=max(geometry.z1_m - float(config.kop_min_vertical_m), SMALL),
            inc_entry_deg=geometry.inc_entry_deg,
        ),
    )
    if build_dls_upper > build_dls_lower + SMALL:
        diagnostics.append(
            "Solver searched BUILD DLS interval "
            f"[{build_dls_lower:.2f}, {build_dls_upper:.2f}] deg/30m; "
            f"closest candidate was at {best_miss_build_dls:.2f} deg/30m."
        )
    diagnostics.append(
        "Solver endpoint miss to t1 after optimization is "
        f"lateral {best_lateral_m:.2f} m / vertical {best_vertical_m:.2f} m "
        f"(tolerances {config.lateral_tolerance_m:.2f} / {config.vertical_tolerance_m:.2f} m)."
    )
    raise PlanningError(
        "No valid trajectory solution found within configured limits. "
        "Closest miss to t1 is "
        f"lateral {best_lateral_m:.2f} m / vertical {best_vertical_m:.2f} m."
        + _format_failure_diagnostics(diagnostics)
    )


def _effective_build_dls_lower_bound(
    config: TrajectoryConfig,
    upper_dls_deg_per_30m: float,
) -> float:
    lower = float(max(config.dls_build_min_deg_per_30m, 0.0))
    if lower <= SMALL:
        lower = min(float(upper_dls_deg_per_30m), NUMERICAL_BUILD_DLS_FLOOR_DEG_PER_30M)
    return float(min(max(lower, SMALL), upper_dls_deg_per_30m))


def _decode_build_dls_from_unit(
    unit_value: float,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
) -> float:
    lower = float(max(lower_dls_deg_per_30m, SMALL))
    upper = float(max(upper_dls_deg_per_30m, lower))
    if upper - lower <= SMALL:
        return upper
    unit = float(np.clip(unit_value, 0.0, 1.0))
    return float(lower + unit * (upper - lower))


def _encode_build_dls_to_unit(
    build_dls_deg_per_30m: float,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
) -> float:
    lower = float(max(lower_dls_deg_per_30m, SMALL))
    upper = float(max(upper_dls_deg_per_30m, lower))
    if upper - lower <= SMALL:
        return 1.0
    return float(
        np.clip((float(build_dls_deg_per_30m) - lower) / (upper - lower), 0.0, 1.0)
    )


def _preferred_build_dls_values(
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
) -> tuple[float, ...]:
    lower = float(max(lower_dls_deg_per_30m, SMALL))
    upper = float(max(upper_dls_deg_per_30m, lower))
    if upper - lower <= SMALL:
        return (upper,)
    mid = 0.5 * (lower + upper)
    return (upper, mid, lower)


def _split_build_optimization_bounds(
    *,
    bounds: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    build_bounds = (float(bounds[0][0]), float(bounds[0][1]))
    return (build_bounds, build_bounds, *bounds[1:])


def _make_turn_profile_builder(
    *,
    geometry: SectionGeometry,
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    min_build_segment_m: float,
    post_entry: PostEntrySection,
    split_build: bool,
) -> Callable[[np.ndarray], ProfileParameters | None]:
    def builder(values: np.ndarray) -> ProfileParameters | None:
        values_list = values.tolist()
        build1_dls = _decode_build_dls_from_unit(
            unit_value=float(values_list[0]),
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        )
        if split_build:
            build2_dls = _decode_build_dls_from_unit(
                unit_value=float(values_list[1]),
                lower_dls_deg_per_30m=lower_dls_deg_per_30m,
                upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            )
            kop_index = 2
        else:
            build2_dls = build1_dls
            kop_index = 1
        kop_vertical_m = float(values_list[kop_index])
        inc_hold_deg = float(values_list[kop_index + 1])
        hold_length_m = float(values_list[kop_index + 2])
        azimuth_hold_deg = (
            float(geometry.azimuth_entry_deg)
            if zero_azimuth_turn
            else float(values_list[kop_index + 3])
        )
        return _profile_same_direction_with_turn(
            geometry=geometry,
            dls_build1_deg_per_30m=build1_dls,
            dls_build2_deg_per_30m=build2_dls,
            kop_vertical_m=kop_vertical_m,
            inc_hold_deg=inc_hold_deg,
            hold_length_m=hold_length_m,
            azimuth_hold_deg=azimuth_hold_deg,
            min_build_segment_m=min_build_segment_m,
            post_entry=post_entry,
        )

    return builder


def _default_candidate_sort_key(
    candidate: ProfileParameters,
) -> tuple[float, float, float]:
    return (
        candidate.md_total_m,
        _candidate_turn_deg(candidate),
        -max(candidate.dls_build1_deg_per_30m, candidate.dls_build2_deg_per_30m),
    )


def _collect_early_anti_collision_stage_candidates(
    *,
    baseline_vector: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    config: TrajectoryConfig,
    split_build: bool,
) -> list[ProfileParameters]:
    kop_index = 2 if split_build else 1
    build1_units = (1.0, 0.85, 0.7)
    kop_fractions = (0.0, 0.15, 0.30, 0.45)
    kop_lower = float(bounds[kop_index][0])
    kop_upper = float(bounds[kop_index][1])
    kop_span = max(kop_upper - kop_lower, 1.0)
    candidates: list[ProfileParameters] = []
    seen: set[tuple[float, float, float]] = set()
    for build1_unit in build1_units:
        for kop_fraction in kop_fractions:
            values = np.asarray(baseline_vector, dtype=float).copy()
            values[0] = float(np.clip(build1_unit, bounds[0][0], bounds[0][1]))
            values[kop_index] = float(
                np.clip(
                    kop_lower + kop_fraction * kop_span,
                    bounds[kop_index][0],
                    bounds[kop_index][1],
                )
            )
            refined_candidates = _refine_profile_with_fixed_components(
                seed_vector=values,
                fixed_components={
                    0: float(values[0]),
                    kop_index: float(values[kop_index]),
                },
                bounds=bounds,
                profile_builder=profile_builder,
                target_point=target_point,
                config=config,
                max_nfev=36,
            )
            if not refined_candidates:
                evaluation = _evaluate_candidate_for_optimization(
                    values=values,
                    profile_builder=profile_builder,
                    target_point=target_point,
                    config=config,
                )
                if evaluation.feasible and evaluation.candidate is not None:
                    refined_candidates = [evaluation.candidate]
            for candidate in refined_candidates:
                key = (
                    round(float(candidate.dls_build1_deg_per_30m), 6),
                    round(float(candidate.kop_vertical_m), 6),
                    round(float(candidate.md_total_m), 6),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _collect_late_build2_seed_vectors(
    *,
    seed_vectors: list[np.ndarray],
    baseline_vector: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    kop_bound_index: int,
) -> list[np.ndarray]:
    build2_index = 1
    build2_upper = float(bounds[build2_index][1])
    build2_lower = float(bounds[build2_index][0])
    hold_index = kop_bound_index + 2
    hold_upper = float(bounds[hold_index][1]) if hold_index < len(bounds) else 0.0
    focused: list[np.ndarray] = []
    baseline = _clip_to_bounds(np.asarray(baseline_vector, dtype=float), bounds=bounds)
    focused.append(baseline)
    build2_upper_seed = baseline.copy()
    build2_upper_seed[build2_index] = build2_upper
    focused.append(build2_upper_seed)
    for seed in seed_vectors[: min(3, len(seed_vectors))]:
        adjusted = _clip_to_bounds(np.asarray(seed, dtype=float), bounds=bounds)
        adjusted[0] = float(baseline[0])
        adjusted[kop_bound_index] = float(baseline[kop_bound_index])
        focused.append(adjusted.copy())
        build2_mid_seed = adjusted.copy()
        build2_mid_seed[build2_index] = float(
            0.5 * (float(build2_mid_seed[build2_index]) + build2_upper)
        )
        focused.append(build2_mid_seed)
        build2_upper_variant = adjusted.copy()
        build2_upper_variant[build2_index] = build2_upper
        focused.append(build2_upper_variant)
        if hold_index < len(bounds):
            shorter_hold_variant = build2_upper_variant.copy()
            shorter_hold_variant[hold_index] = float(
                max(
                    hold_upper * 0.25,
                    float(bounds[hold_index][0]),
                )
            )
            focused.append(shorter_hold_variant)
            longer_hold_variant = build2_upper_variant.copy()
            longer_hold_variant[hold_index] = float(
                max(
                    float(longer_hold_variant[hold_index]),
                    hold_upper * 0.75,
                )
            )
            focused.append(longer_hold_variant)
        build2_lower_variant = adjusted.copy()
        build2_lower_variant[build2_index] = build2_lower
        focused.append(build2_lower_variant)
    return _dedupe_seed_vectors(focused)[:LATE_ANTI_COLLISION_BUILD2_MAX_STARTS]


def _optimization_objective_value(candidate: ProfileParameters, mode: str) -> float:
    if mode == OPTIMIZATION_MINIMIZE_MD:
        return float(candidate.md_total_m)
    if mode == OPTIMIZATION_MINIMIZE_KOP:
        return float(candidate.kop_vertical_m)
    return float(candidate.md_total_m)


def _optimization_candidate_sort_key(
    candidate: ProfileParameters,
    mode: str,
) -> tuple[float, float, float, float]:
    if mode == OPTIMIZATION_MINIMIZE_KOP:
        return (
            float(candidate.kop_vertical_m),
            float(candidate.md_total_m),
            _candidate_turn_deg(candidate),
            -max(
                float(candidate.dls_build1_deg_per_30m),
                float(candidate.dls_build2_deg_per_30m),
            ),
        )
    return (
        float(candidate.md_total_m),
        float(candidate.kop_vertical_m),
        _candidate_turn_deg(candidate),
        -max(
            float(candidate.dls_build1_deg_per_30m),
            float(candidate.dls_build2_deg_per_30m),
        ),
    )


def _theoretical_objective_lower_bound(
    *,
    geometry: SectionGeometry,
    config: TrajectoryConfig,
    mode: str,
) -> float:
    if mode == OPTIMIZATION_MINIMIZE_KOP:
        return float(max(config.kop_min_vertical_m, 0.0))
    t1_straight_m = float(
        np.sqrt(
            geometry.t1_east_m * geometry.t1_east_m
            + geometry.t1_north_m * geometry.t1_north_m
            + geometry.t1_tvd_m * geometry.t1_tvd_m
        )
    )
    cross_delta_m = float(geometry.t3_cross_m - geometry.t1_cross_m)
    t1_t3_straight_m = float(
        np.sqrt(
            geometry.ds_13_m * geometry.ds_13_m
            + cross_delta_m * cross_delta_m
            + geometry.dz_13_m * geometry.dz_13_m
        )
    )
    return float(max(t1_straight_m + t1_t3_straight_m, SMALL))


def _optimization_absolute_gap(
    *,
    objective_value: float,
    theoretical_lower_bound: float,
) -> float:
    return float(max(float(objective_value) - float(theoretical_lower_bound), 0.0))


def _optimization_relative_gap_pct(
    *,
    objective_value: float,
    theoretical_lower_bound: float,
) -> float:
    absolute_gap = _optimization_absolute_gap(
        objective_value=objective_value,
        theoretical_lower_bound=theoretical_lower_bound,
    )
    denominator = float(max(abs(theoretical_lower_bound), 1.0))
    return float(100.0 * absolute_gap / denominator)


def _optimization_target_reached(
    *,
    objective_value: float,
    theoretical_lower_bound: float,
    mode: str,
) -> bool:
    if mode == OPTIMIZATION_MINIMIZE_KOP:
        return bool(
            objective_value <= theoretical_lower_bound + KOP_OPTIMIZATION_TOLERANCE_M
        )
    return bool(
        objective_value
        <= theoretical_lower_bound * (1.0 + MD_OPTIMIZATION_THEORETICAL_GAP_FRACTION)
        + SMALL
    )


def _is_md_boundary_extremum_candidate(
    *,
    candidate: ProfileParameters,
    build_dls_upper_deg_per_30m: float,
    kop_lower_m: float,
) -> bool:
    return bool(
        abs(
            float(candidate.dls_build1_deg_per_30m) - float(build_dls_upper_deg_per_30m)
        )
        <= MD_BOUNDARY_EXTREMUM_BUILD_TOLERANCE
        and abs(
            float(candidate.dls_build2_deg_per_30m) - float(build_dls_upper_deg_per_30m)
        )
        <= MD_BOUNDARY_EXTREMUM_BUILD_TOLERANCE
        and abs(float(candidate.kop_vertical_m) - float(kop_lower_m))
        <= MD_BOUNDARY_EXTREMUM_KOP_TOLERANCE_M
    )


def _select_anti_collision_candidate(
    *,
    candidates: list[ProfileParameters],
    surface: Point3D,
    config: TrajectoryConfig,
    optimization_context: AntiCollisionOptimizationContext,
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    search_settings: TurnSearchSettings,
    optimization_bounds: tuple[tuple[float, float], ...] | None = None,
    optimization_profile_builder: (
        Callable[[np.ndarray], ProfileParameters | None] | None
    ) = None,
    split_build: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TurnSolveResult:
    active_bounds = optimization_bounds or bounds
    active_profile_builder = optimization_profile_builder or profile_builder
    kop_bound_index = 2 if split_build else 1
    baseline = min(candidates, key=_default_candidate_sort_key)
    baseline_vector = _candidate_to_search_vector(
        candidate=baseline,
        zero_azimuth_turn=zero_azimuth_turn,
        lower_dls_deg_per_30m=lower_dls_deg_per_30m,
        upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        bounds=active_bounds,
        split_build=split_build,
    )
    keep_kop_reference = float(baseline_vector[kop_bound_index])
    if optimization_context.baseline_kop_vertical_m is not None:
        keep_kop_reference = float(
            np.clip(
                float(optimization_context.baseline_kop_vertical_m),
                float(active_bounds[kop_bound_index][0]),
                float(active_bounds[kop_bound_index][1]),
            )
        )
    keep_build1_reference = float(baseline_vector[0])
    if optimization_context.baseline_build1_dls_deg_per_30m is not None:
        keep_build1_reference = float(
            np.clip(
                _encode_build_dls_to_unit(
                    build_dls_deg_per_30m=float(
                        optimization_context.baseline_build1_dls_deg_per_30m
                    ),
                    lower_dls_deg_per_30m=lower_dls_deg_per_30m,
                    upper_dls_deg_per_30m=upper_dls_deg_per_30m,
                ),
                float(active_bounds[0][0]),
                float(active_bounds[0][1]),
            )
        )
    constrained_baseline_vector = np.asarray(baseline_vector, dtype=float).copy()
    constrained_baseline_vector[0] = float(keep_build1_reference)
    constrained_baseline_vector[kop_bound_index] = float(keep_kop_reference)
    distance_baseline_vector = (
        constrained_baseline_vector
        if bool(
            optimization_context.prefer_keep_kop
            or optimization_context.prefer_keep_build1
        )
        else baseline_vector
    )
    clearance_cache: dict[tuple[float, ...], AntiCollisionClearanceEvaluation] = {}

    def candidate_key(candidate: ProfileParameters) -> tuple[float, ...]:
        vector = _candidate_to_search_vector(
            candidate=candidate,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=active_bounds,
            split_build=split_build,
        )
        return tuple(np.round(vector, decimals=8).tolist())

    def clearance_for_candidate(
        candidate: ProfileParameters,
    ) -> AntiCollisionClearanceEvaluation:
        key = candidate_key(candidate)
        if key not in clearance_cache:
            clearance_cache[key] = evaluate_candidate_anti_collision_clearance(
                candidate=candidate,
                surface=surface,
                context=optimization_context,
            )
        return clearance_cache[key]

    def anti_collision_sort_key(
        candidate: ProfileParameters,
    ) -> tuple[float, float, float, float, float, float, float]:
        clearance = clearance_for_candidate(candidate)
        kop_rank = (
            float(candidate.kop_vertical_m)
            if bool(optimization_context.prefer_lower_kop)
            else 0.0
        )
        build1_rank = (
            -float(candidate.dls_build1_deg_per_30m)
            if bool(optimization_context.prefer_higher_build1)
            else 0.0
        )
        deviation_rank = _candidate_vector_distance(
            candidate=candidate,
            baseline_vector=distance_baseline_vector,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=active_bounds,
            split_build=split_build,
        )
        return (
            max(
                float(optimization_context.sf_target)
                - float(clearance.min_separation_factor),
                0.0,
            ),
            kop_rank,
            build1_rank,
            deviation_rank,
            float(candidate.md_total_m),
            float(candidate.kop_vertical_m),
            _candidate_turn_deg(candidate),
        )

    early_anti_collision_stage = bool(
        optimization_context.prefer_lower_kop
        and optimization_context.prefer_higher_build1
    )
    late_build2_stage = bool(
        split_build
        and (
            optimization_context.prefer_adjust_build2
            or (
                optimization_context.prefer_keep_kop
                and optimization_context.prefer_keep_build1
            )
        )
        and not early_anti_collision_stage
    )
    ranked_all = sorted(candidates, key=anti_collision_sort_key)
    if late_build2_stage:
        baseline_build1_dls = optimization_context.baseline_build1_dls_deg_per_30m
        ranked = [
            candidate
            for candidate in ranked_all
            if abs(float(candidate.kop_vertical_m) - float(keep_kop_reference))
            <= LATE_ANTI_COLLISION_KEEP_KOP_TOLERANCE_M
            and (
                baseline_build1_dls is None
                or abs(
                    float(candidate.dls_build1_deg_per_30m) - float(baseline_build1_dls)
                )
                <= LATE_ANTI_COLLISION_KEEP_BUILD1_TOLERANCE_DEG_PER_30M
            )
        ]
    else:
        ranked = list(ranked_all)
    best_seed = ranked[0] if ranked else ranked_all[0]
    best_seed_clearance = clearance_for_candidate(best_seed)
    if (
        ranked
        and float(best_seed_clearance.min_separation_factor)
        >= float(optimization_context.sf_target) - ANTI_COLLISION_SF_TOLERANCE
    ):
        return TurnSolveResult(
            params=best_seed,
            optimization=OptimizationOutcome(
                mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
                status="sf_target_reached",
                objective_value=max(
                    float(optimization_context.sf_target)
                    - float(best_seed_clearance.min_separation_factor),
                    0.0,
                ),
                theoretical_lower_bound=0.0,
                absolute_gap_value=max(
                    float(optimization_context.sf_target)
                    - float(best_seed_clearance.min_separation_factor),
                    0.0,
                ),
                relative_gap_pct=0.0,
                seeds_used=len(ranked),
                runs_used=0,
            ),
        )

    seed_vectors = [
        _candidate_to_search_vector(
            candidate=item,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=active_bounds,
            split_build=split_build,
        )
        for item in (ranked if ranked else ranked_all)[:OPTIMIZATION_MAX_SEEDS]
    ]
    seed_vectors.extend(
        _collect_optimization_seed_vectors(
            candidates=ranked if ranked else ranked_all,
            mode=OPTIMIZATION_MINIMIZE_MD,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=active_bounds,
            split_build=split_build,
        )
    )
    optimization_seed_vectors = _dedupe_seed_vectors(seed_vectors)
    optimized_candidates: list[ProfileParameters] = list(ranked)
    if early_anti_collision_stage:
        optimized_candidates.extend(
            _collect_early_anti_collision_stage_candidates(
                baseline_vector=baseline_vector,
                bounds=active_bounds,
                profile_builder=active_profile_builder,
                target_point=target_point,
                config=config,
                split_build=split_build,
            )
        )
        optimization_seed_vectors = _dedupe_seed_vectors(
            [
                _candidate_to_search_vector(
                    candidate=item,
                    zero_azimuth_turn=zero_azimuth_turn,
                    lower_dls_deg_per_30m=lower_dls_deg_per_30m,
                    upper_dls_deg_per_30m=upper_dls_deg_per_30m,
                    bounds=active_bounds,
                    split_build=split_build,
                )
                for item in optimized_candidates[: max(OPTIMIZATION_MAX_SEEDS, 1)]
            ]
        )[:EARLY_ANTI_COLLISION_MAX_STARTS]
    elif late_build2_stage:
        optimization_seed_vectors = _collect_late_build2_seed_vectors(
            seed_vectors=optimization_seed_vectors,
            baseline_vector=constrained_baseline_vector,
            bounds=active_bounds,
            kop_bound_index=kop_bound_index,
        )
    else:
        optimization_seed_vectors = optimization_seed_vectors[
            :ANTI_COLLISION_MAX_STARTS
        ]
    runs_used = 0

    eval_cache: dict[
        tuple[float, ...],
        tuple[CandidateOptimizationEvaluation, AntiCollisionClearanceEvaluation | None],
    ] = {}

    def evaluate(
        values: np.ndarray,
    ) -> tuple[
        CandidateOptimizationEvaluation, AntiCollisionClearanceEvaluation | None
    ]:
        vector = tuple(np.asarray(values, dtype=float).tolist())
        cached = eval_cache.get(vector)
        if cached is not None:
            return cached
        base_eval = _evaluate_candidate_for_optimization(
            values=np.asarray(values, dtype=float),
            profile_builder=active_profile_builder,
            target_point=target_point,
            config=config,
        )
        clearance: AntiCollisionClearanceEvaluation | None = None
        if base_eval.feasible and base_eval.candidate is not None:
            clearance = clearance_cache.get(candidate_key(base_eval.candidate))
            if clearance is None:
                clearance = evaluate_candidate_anti_collision_clearance(
                    candidate=base_eval.candidate,
                    surface=surface,
                    context=optimization_context,
                )
                clearance_cache[candidate_key(base_eval.candidate)] = clearance
        else:
            clearance = None
        eval_cache[vector] = (base_eval, clearance)
        return base_eval, clearance

    def objective(values: np.ndarray) -> float:
        base_eval, clearance = evaluate(values)
        if not base_eval.feasible or base_eval.candidate is None or clearance is None:
            return 1e12
        deficit = max(
            float(optimization_context.sf_target)
            - float(clearance.min_separation_factor),
            0.0,
        )
        deviation = _search_vector_distance(
            values=np.asarray(values, dtype=float),
            baseline_vector=distance_baseline_vector,
            bounds=active_bounds,
        )
        kop_preference = 0.0
        if bool(optimization_context.prefer_lower_kop):
            kop_span = max(
                float(active_bounds[kop_bound_index][1])
                - float(active_bounds[kop_bound_index][0]),
                1.0,
            )
            kop_preference = (
                ANTI_COLLISION_KOP_PREFERENCE_WEIGHT
                * max(
                    float(base_eval.kop_vertical_m)
                    - float(active_bounds[kop_bound_index][0]),
                    0.0,
                )
                / kop_span
            )
        build1_preference = 0.0
        if bool(optimization_context.prefer_higher_build1):
            build1_span = max(
                float(active_bounds[0][1]) - float(active_bounds[0][0]),
                1e-6,
            )
            build1_preference = (
                ANTI_COLLISION_BUILD1_PREFERENCE_WEIGHT
                * max(float(active_bounds[0][1]) - float(values[0]), 0.0)
                / build1_span
            )
        keep_kop_penalty = 0.0
        if bool(optimization_context.prefer_keep_kop):
            kop_span = max(
                float(active_bounds[kop_bound_index][1])
                - float(active_bounds[kop_bound_index][0]),
                1.0,
            )
            keep_kop_penalty = (
                ANTI_COLLISION_KEEP_KOP_WEIGHT
                * (
                    max(
                        abs(float(values[kop_bound_index]) - float(keep_kop_reference)),
                        0.0,
                    )
                    / kop_span
                )
                ** 2
            )
        keep_build1_penalty = 0.0
        if bool(optimization_context.prefer_keep_build1):
            build1_span = max(
                float(active_bounds[0][1]) - float(active_bounds[0][0]),
                1e-6,
            )
            keep_build1_penalty = (
                ANTI_COLLISION_KEEP_BUILD1_WEIGHT
                * (
                    max(abs(float(values[0]) - float(keep_build1_reference)), 0.0)
                    / build1_span
                )
                ** 2
            )
        build2_preference = 0.0
        if late_build2_stage:
            build2_span = max(
                float(active_bounds[1][1]) - float(active_bounds[1][0]),
                1e-6,
            )
            build2_preference = (
                ANTI_COLLISION_BUILD2_PREFERENCE_WEIGHT
                * max(float(active_bounds[1][1]) - float(values[1]), 0.0)
                / build2_span
            )
        return (
            ANTI_COLLISION_OBJECTIVE_PENALTY * deficit * deficit
            + kop_preference
            + build1_preference
            + keep_kop_penalty
            + keep_build1_penalty
            + build2_preference
            + deviation
            + 1e-4 * float(base_eval.md_total_m)
        )

    constraints = [
        {"type": "ineq", "fun": lambda values: evaluate(values)[0].t1_margin_m},
        {"type": "ineq", "fun": lambda values: evaluate(values)[0].build1_margin_m},
        {"type": "ineq", "fun": lambda values: evaluate(values)[0].build2_margin_m},
        {"type": "ineq", "fun": lambda values: evaluate(values)[0].max_inc_margin_deg},
        {
            "type": "ineq",
            "fun": lambda values: evaluate(values)[0].horizontal_dls_margin_deg_per_30m,
        },
    ]
    maxiter = max(
        36, min(120, int(round(0.24 * float(search_settings.local_max_nfev))))
    )
    if early_anti_collision_stage:
        maxiter = min(maxiter, EARLY_ANTI_COLLISION_MAXITER)
    elif late_build2_stage:
        maxiter = min(maxiter, LATE_ANTI_COLLISION_BUILD2_MAXITER)
    else:
        maxiter = min(maxiter, ANTI_COLLISION_MAXITER)
    if progress_callback is not None:
        _emit_progress(
            progress_callback,
            (
                "Оптимизация: anti-collision avoidance, "
                f"локальные старты {len(optimization_seed_vectors)}."
            ),
            0.97,
        )
    if early_anti_collision_stage:
        selected = min(optimized_candidates, key=anti_collision_sort_key)
        selected_clearance = clearance_for_candidate(selected)
        seed_key = anti_collision_sort_key(best_seed)
        selected_key = anti_collision_sort_key(selected)
        improved = bool(selected_key < seed_key)
        objective_gap = max(
            float(optimization_context.sf_target)
            - float(selected_clearance.min_separation_factor),
            0.0,
        )
        return TurnSolveResult(
            params=selected,
            optimization=OptimizationOutcome(
                mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
                status=(
                    "sf_target_reached"
                    if float(selected_clearance.min_separation_factor)
                    >= float(optimization_context.sf_target)
                    - ANTI_COLLISION_SF_TOLERANCE
                    else ("clearance_improved" if improved else "seed_selected")
                ),
                objective_value=float(objective_gap),
                theoretical_lower_bound=0.0,
                absolute_gap_value=float(objective_gap),
                relative_gap_pct=(
                    float(
                        100.0
                        * objective_gap
                        / max(float(optimization_context.sf_target), 1.0)
                    )
                    if objective_gap > 0.0
                    else 0.0
                ),
                seeds_used=len(optimization_seed_vectors),
                runs_used=0,
            ),
        )
    if late_build2_stage:
        fixed_indices = {0, kop_bound_index}
        reduced_indices = [
            index for index in range(len(active_bounds)) if index not in fixed_indices
        ]
        reduced_bounds = [active_bounds[index] for index in reduced_indices]
        for seed_vector in optimization_seed_vectors:
            base_seed = _clip_to_bounds(seed_vector, bounds=active_bounds)
            base_seed[0] = float(keep_build1_reference)
            base_seed[kop_bound_index] = float(keep_kop_reference)
            reduced_seed = np.asarray(
                [float(base_seed[index]) for index in reduced_indices],
                dtype=float,
            )

            def expand_reduced(values_reduced: np.ndarray) -> np.ndarray:
                expanded = np.asarray(base_seed, dtype=float).copy()
                for reduced_index, free_index in enumerate(reduced_indices):
                    expanded[free_index] = float(values_reduced[reduced_index])
                expanded[0] = float(keep_build1_reference)
                expanded[kop_bound_index] = float(keep_kop_reference)
                return _clip_to_bounds(expanded, bounds=active_bounds)

            runs_used += 1
            result = minimize(
                fun=lambda values_reduced: objective(
                    expand_reduced(np.asarray(values_reduced, dtype=float))
                ),
                x0=reduced_seed,
                method="SLSQP",
                bounds=list(reduced_bounds),
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda values_reduced: evaluate(
                            expand_reduced(np.asarray(values_reduced, dtype=float))
                        )[0].t1_margin_m,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda values_reduced: evaluate(
                            expand_reduced(np.asarray(values_reduced, dtype=float))
                        )[0].build1_margin_m,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda values_reduced: evaluate(
                            expand_reduced(np.asarray(values_reduced, dtype=float))
                        )[0].build2_margin_m,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda values_reduced: evaluate(
                            expand_reduced(np.asarray(values_reduced, dtype=float))
                        )[0].max_inc_margin_deg,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda values_reduced: evaluate(
                            expand_reduced(np.asarray(values_reduced, dtype=float))
                        )[0].horizontal_dls_margin_deg_per_30m,
                    },
                ],
                options={
                    "maxiter": int(maxiter),
                    "ftol": 1e-8,
                    "disp": False,
                },
            )
            probes = [expand_reduced(reduced_seed)]
            if np.all(np.isfinite(result.x)):
                probes.append(expand_reduced(np.asarray(result.x, dtype=float)))
            for probe in probes:
                base_eval, clearance = evaluate(probe)
                if (
                    not base_eval.feasible
                    or base_eval.candidate is None
                    or clearance is None
                ):
                    continue
                optimized_candidates.append(base_eval.candidate)
                clearance_cache[candidate_key(base_eval.candidate)] = clearance

            if optimized_candidates:
                current_best = min(optimized_candidates, key=anti_collision_sort_key)
                current_clearance = clearance_for_candidate(current_best)
                if (
                    float(current_clearance.min_separation_factor)
                    >= float(optimization_context.sf_target)
                    - ANTI_COLLISION_SF_TOLERANCE
                ):
                    break
    else:
        for seed_vector in optimization_seed_vectors:
            seed = _clip_to_bounds(seed_vector, bounds=active_bounds)
            runs_used += 1
            result = minimize(
                fun=objective,
                x0=seed,
                method="SLSQP",
                bounds=list(active_bounds),
                constraints=constraints,
                options={
                    "maxiter": int(maxiter),
                    "ftol": 1e-8,
                    "disp": False,
                },
            )
            probes = [seed]
            if np.all(np.isfinite(result.x)):
                probes.append(
                    _clip_to_bounds(
                        np.asarray(result.x, dtype=float), bounds=active_bounds
                    )
                )
            for probe in probes:
                base_eval, clearance = evaluate(probe)
                if (
                    not base_eval.feasible
                    or base_eval.candidate is None
                    or clearance is None
                ):
                    continue
                optimized_candidates.append(base_eval.candidate)
                clearance_cache[candidate_key(base_eval.candidate)] = clearance

            current_best = min(optimized_candidates, key=anti_collision_sort_key)
            current_clearance = clearance_for_candidate(current_best)
            if (
                float(current_clearance.min_separation_factor)
                >= float(optimization_context.sf_target) - ANTI_COLLISION_SF_TOLERANCE
            ):
                break

    if late_build2_stage and not optimized_candidates:
        raise PlanningError(
            "Late anti-collision BUILD2/HOLD adjustment could not find a valid "
            "trajectory within the preserved KOP/BUILD1 profile and current limits. "
            "Relax t1 tolerance / DLS limits or revise target spacing."
        )

    selected = min(optimized_candidates, key=anti_collision_sort_key)
    selected_clearance = clearance_for_candidate(selected)
    seed_key = anti_collision_sort_key(best_seed)
    selected_key = anti_collision_sort_key(selected)
    improved = bool(selected_key < seed_key)
    objective_gap = max(
        float(optimization_context.sf_target)
        - float(selected_clearance.min_separation_factor),
        0.0,
    )
    return TurnSolveResult(
        params=selected,
        optimization=OptimizationOutcome(
            mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
            status=(
                "sf_target_reached"
                if float(selected_clearance.min_separation_factor)
                >= float(optimization_context.sf_target) - ANTI_COLLISION_SF_TOLERANCE
                else ("clearance_improved" if improved else "seed_selected")
            ),
            objective_value=float(objective_gap),
            theoretical_lower_bound=0.0,
            absolute_gap_value=float(objective_gap),
            relative_gap_pct=(
                float(
                    100.0
                    * objective_gap
                    / max(float(optimization_context.sf_target), 1.0)
                )
                if objective_gap > 0.0
                else 0.0
            ),
            seeds_used=len(optimization_seed_vectors),
            runs_used=int(runs_used),
        ),
    )


def _candidate_vector_distance(
    *,
    candidate: ProfileParameters,
    baseline_vector: np.ndarray,
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    split_build: bool = False,
) -> float:
    values = _candidate_to_search_vector(
        candidate=candidate,
        zero_azimuth_turn=zero_azimuth_turn,
        lower_dls_deg_per_30m=lower_dls_deg_per_30m,
        upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        bounds=bounds,
        split_build=split_build,
    )
    return _search_vector_distance(
        values=values,
        baseline_vector=baseline_vector,
        bounds=bounds,
    )


def _search_vector_distance(
    *,
    values: np.ndarray,
    baseline_vector: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
) -> float:
    vector = np.asarray(values, dtype=float)
    baseline = np.asarray(baseline_vector, dtype=float)
    spans = np.array(
        [max(float(upper - lower), 1.0) for lower, upper in bounds],
        dtype=float,
    )
    normalized = (vector - baseline) / spans
    return float(np.linalg.norm(normalized))


def _candidate_to_search_vector(
    *,
    candidate: ProfileParameters,
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    split_build: bool = False,
) -> np.ndarray:
    vector = [
        _encode_build_dls_to_unit(
            build_dls_deg_per_30m=float(candidate.dls_build1_deg_per_30m),
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        ),
    ]
    if split_build:
        vector.append(
            _encode_build_dls_to_unit(
                build_dls_deg_per_30m=float(candidate.dls_build2_deg_per_30m),
                lower_dls_deg_per_30m=lower_dls_deg_per_30m,
                upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            )
        )
    vector.extend(
        [
            float(candidate.kop_vertical_m),
            float(candidate.inc_hold_deg),
            float(candidate.hold_length_m),
        ]
    )
    if not zero_azimuth_turn:
        vector.append(float(_normalize_azimuth_deg(candidate.azimuth_hold_deg)))
    return _clip_to_bounds(np.asarray(vector, dtype=float), bounds=bounds)


def _dedupe_profile_candidates(
    *,
    candidates: list[ProfileParameters],
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
) -> list[ProfileParameters]:
    deduped: list[ProfileParameters] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        vector = _candidate_to_search_vector(
            candidate=candidate,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
        )
        key = tuple(np.round(vector, decimals=8).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _evaluate_candidate_for_optimization(
    *,
    values: np.ndarray,
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    config: TrajectoryConfig,
) -> CandidateOptimizationEvaluation:
    candidate = profile_builder(np.asarray(values, dtype=float))
    return _evaluate_profile_candidate(
        candidate=candidate, target_point=target_point, config=config
    )


def _evaluate_profile_candidate(
    *,
    candidate: ProfileParameters | None,
    target_point: np.ndarray,
    config: TrajectoryConfig,
) -> CandidateOptimizationEvaluation:
    if candidate is None:
        return CandidateOptimizationEvaluation(
            candidate=None,
            t1_miss_m=1e9,
            md_total_m=1e12,
            kop_vertical_m=1e12,
            t1_margin_m=-1e9,
            build1_margin_m=-1e9,
            build2_margin_m=-1e9,
            max_inc_margin_deg=-1e9,
            horizontal_dls_margin_deg_per_30m=-1e9,
        )

    endpoint = np.array(_estimate_t1_endpoint_for_profile(candidate), dtype=float)
    _, _, _, miss_t1_lateral_m, miss_t1_vertical_m, miss_t1_m = _target_miss_components(
        endpoint,
        target_point,
    )
    max_inc_actual_deg = float(
        max(
            candidate.inc_hold_deg,
            candidate.inc_entry_deg,
            candidate.horizontal_inc_deg,
        )
    )
    horizontal_limit = float(config.dls_limits_deg_per_30m.get("HORIZONTAL", np.inf))
    return CandidateOptimizationEvaluation(
        candidate=candidate,
        t1_miss_m=miss_t1_m,
        md_total_m=float(candidate.md_total_m),
        kop_vertical_m=float(candidate.kop_vertical_m),
        t1_margin_m=_target_miss_margin_m(
            lateral_m=miss_t1_lateral_m,
            vertical_m=miss_t1_vertical_m,
            config=config,
        ),
        build1_margin_m=float(
            candidate.build1_length_m - config.min_structural_segment_m
        ),
        build2_margin_m=float(
            candidate.build2_length_m - config.min_structural_segment_m
        ),
        max_inc_margin_deg=float(config.max_inc_deg - max_inc_actual_deg),
        horizontal_dls_margin_deg_per_30m=float(
            horizontal_limit - candidate.horizontal_dls_deg_per_30m
        ),
    )


def _collect_optimization_seed_vectors(
    *,
    candidates: list[ProfileParameters],
    mode: str,
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    split_build: bool = False,
) -> list[np.ndarray]:
    if not candidates:
        return []
    ranked = sorted(
        candidates,
        key=lambda candidate: _optimization_candidate_sort_key(candidate, mode),
    )
    seed_candidates = list(ranked[:OPTIMIZATION_MAX_SEEDS])
    if mode == OPTIMIZATION_MINIMIZE_KOP:
        seed_candidates.append(min(ranked, key=_default_candidate_sort_key))
    else:
        seed_candidates.append(
            min(
                ranked,
                key=lambda candidate: (candidate.kop_vertical_m, candidate.md_total_m),
            )
        )
    seed_vectors = [
        _candidate_to_search_vector(
            candidate=candidate,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
            split_build=split_build,
        )
        for candidate in seed_candidates
    ]
    if mode == OPTIMIZATION_MINIMIZE_MD and seed_vectors:
        build1_upper = float(bounds[0][1])
        build2_upper = float(bounds[1][1]) if split_build else build1_upper
        kop_index = 2 if split_build else 1
        kop_lower = float(bounds[kop_index][0])
        md_boundary_variants: list[np.ndarray] = []
        for seed in seed_vectors[: min(3, len(seed_vectors))]:
            kop_lower_seed = np.asarray(seed, dtype=float).copy()
            kop_lower_seed[kop_index] = kop_lower
            if split_build:
                build1_upper_seed = np.asarray(seed, dtype=float).copy()
                build1_upper_seed[0] = build1_upper
                build2_upper_seed = np.asarray(seed, dtype=float).copy()
                build2_upper_seed[1] = build2_upper
                combined_seed = np.asarray(seed, dtype=float).copy()
                combined_seed[0] = build1_upper
                combined_seed[1] = build2_upper
                combined_kop_seed = np.asarray(combined_seed, dtype=float).copy()
                combined_kop_seed[kop_index] = kop_lower
                midpoint_seed = np.asarray(seed, dtype=float).copy()
                midpoint_seed[0] = float(0.5 * (midpoint_seed[0] + build1_upper))
                midpoint_seed[1] = float(0.5 * (midpoint_seed[1] + build2_upper))
                midpoint_seed[kop_index] = float(
                    0.5 * (midpoint_seed[kop_index] + kop_lower)
                )
                md_boundary_variants.extend(
                    [
                        build1_upper_seed,
                        build2_upper_seed,
                        combined_seed,
                        kop_lower_seed,
                        combined_kop_seed,
                        midpoint_seed,
                    ]
                )
            else:
                build_upper_seed = np.asarray(seed, dtype=float).copy()
                build_upper_seed[0] = build1_upper
                combined_seed = np.asarray(seed, dtype=float).copy()
                combined_seed[0] = build1_upper
                combined_seed[kop_index] = kop_lower
                midpoint_seed = np.asarray(seed, dtype=float).copy()
                midpoint_seed[0] = float(0.5 * (midpoint_seed[0] + build1_upper))
                midpoint_seed[kop_index] = float(
                    0.5 * (midpoint_seed[kop_index] + kop_lower)
                )
                md_boundary_variants.extend(
                    [
                        build_upper_seed,
                        kop_lower_seed,
                        combined_seed,
                        midpoint_seed,
                    ]
                )
        seed_vectors.extend(md_boundary_variants)
    if mode == OPTIMIZATION_MINIMIZE_KOP and seed_vectors:
        kop_index = 2 if split_build else 1
        kop_lower = float(bounds[kop_index][0])
        build1_upper = float(bounds[0][1])
        build2_upper = float(bounds[1][1]) if split_build else build1_upper
        lowered_variants: list[np.ndarray] = []
        for seed in seed_vectors[: min(3, len(seed_vectors))]:
            midpoint_seed = np.asarray(seed, dtype=float).copy()
            midpoint_seed[kop_index] = float(
                0.5 * (midpoint_seed[kop_index] + kop_lower)
            )
            lower_bound_seed = np.asarray(seed, dtype=float).copy()
            lower_bound_seed[kop_index] = kop_lower
            if split_build:
                lower_bound_build_upper_seed = np.asarray(seed, dtype=float).copy()
                lower_bound_build_upper_seed[0] = build1_upper
                lower_bound_build_upper_seed[1] = build2_upper
                lower_bound_build_upper_seed[kop_index] = kop_lower
                build1_upper_midpoint_seed = np.asarray(seed, dtype=float).copy()
                build1_upper_midpoint_seed[0] = float(
                    0.5 * (build1_upper_midpoint_seed[0] + build1_upper)
                )
                build1_upper_midpoint_seed[kop_index] = kop_lower
                build2_upper_midpoint_seed = np.asarray(seed, dtype=float).copy()
                build2_upper_midpoint_seed[1] = float(
                    0.5 * (build2_upper_midpoint_seed[1] + build2_upper)
                )
                build2_upper_midpoint_seed[kop_index] = kop_lower
                lowered_variants.extend(
                    [
                        midpoint_seed,
                        lower_bound_seed,
                        lower_bound_build_upper_seed,
                        build1_upper_midpoint_seed,
                        build2_upper_midpoint_seed,
                    ]
                )
            else:
                lower_bound_build_upper_seed = np.asarray(seed, dtype=float).copy()
                lower_bound_build_upper_seed[0] = build1_upper
                lower_bound_build_upper_seed[kop_index] = kop_lower
                build_upper_midpoint_seed = np.asarray(seed, dtype=float).copy()
                build_upper_midpoint_seed[0] = float(
                    0.5 * (build_upper_midpoint_seed[0] + build1_upper)
                )
                build_upper_midpoint_seed[kop_index] = kop_lower
                lowered_variants.extend(
                    [
                        midpoint_seed,
                        lower_bound_seed,
                        lower_bound_build_upper_seed,
                        build_upper_midpoint_seed,
                    ]
                )
        seed_vectors.extend(lowered_variants)
    return _dedupe_seed_vectors(seed_vectors)


def _refine_profile_with_fixed_components(
    *,
    seed_vector: np.ndarray,
    fixed_components: dict[int, float],
    bounds: tuple[tuple[float, float], ...],
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    config: TrajectoryConfig,
    max_nfev: int,
) -> list[ProfileParameters]:
    seed = _clip_to_bounds(np.asarray(seed_vector, dtype=float), bounds=bounds)
    fixed = {int(index): float(value) for index, value in fixed_components.items()}
    free_indices = [index for index in range(len(bounds)) if index not in fixed]
    if not free_indices:
        candidate = profile_builder(seed)
        if candidate is not None:
            endpoint = np.array(
                _estimate_t1_endpoint_for_profile(candidate), dtype=float
            )
            _, _, _, lateral_m, vertical_m, _ = _target_miss_components(
                endpoint, target_point
            )
            if _target_miss_within_tolerance(
                lateral_m=lateral_m,
                vertical_m=vertical_m,
                config=config,
            ) and _is_candidate_feasible(candidate, config):
                return [candidate]
        return []

    lower = np.array([bounds[index][0] for index in free_indices], dtype=float)
    upper = np.array([bounds[index][1] for index in free_indices], dtype=float)
    x0 = np.array([seed[index] for index in free_indices], dtype=float)

    def _compose(free_values: np.ndarray) -> np.ndarray:
        full = seed.copy()
        for index, value in fixed.items():
            full[index] = float(value)
        for free_index, value in zip(
            free_indices, np.asarray(free_values, dtype=float)
        ):
            full[free_index] = float(value)
        return _clip_to_bounds(full, bounds=bounds)

    solution = least_squares(
        fun=lambda free_values: _turn_residuals(
            values=_compose(np.asarray(free_values, dtype=float)),
            profile_builder=profile_builder,
            target_point=target_point,
            config=config,
        ),
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=int(max(max_nfev, 20)),
    )

    probes = [_compose(x0)]
    if bool(solution.success) and np.all(np.isfinite(solution.x)):
        probes.append(_compose(np.asarray(solution.x, dtype=float)))

    refined: list[ProfileParameters] = []
    for probe in probes:
        candidate = profile_builder(probe)
        if candidate is None:
            continue
        endpoint = np.array(_estimate_t1_endpoint_for_profile(candidate), dtype=float)
        _, _, _, lateral_m, vertical_m, _ = _target_miss_components(
            endpoint, target_point
        )
        if not _target_miss_within_tolerance(
            lateral_m=lateral_m,
            vertical_m=vertical_m,
            config=config,
        ):
            continue
        if not _is_candidate_feasible(candidate, config):
            continue
        refined.append(candidate)
    return refined


def _boundary_refine_md_candidates(
    *,
    candidates: list[ProfileParameters],
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    config: TrajectoryConfig,
    search_settings: TurnSearchSettings,
) -> tuple[list[ProfileParameters], int]:
    if not candidates:
        return [], 0

    ranked = sorted(candidates, key=_default_candidate_sort_key)
    base_candidates = ranked[:MD_BOUNDARY_REFINEMENT_MAX_BASE_CANDIDATES]
    max_nfev = max(
        40,
        int(
            round(
                MD_BOUNDARY_REFINEMENT_MAX_NFEV_FACTOR
                * float(search_settings.local_max_nfev)
            )
        ),
    )
    build_upper = float(bounds[0][1])
    kop_lower = float(bounds[1][0])
    boundary_specs = (
        {0: build_upper},
        {1: kop_lower},
        {0: build_upper, 1: kop_lower},
    )

    refined: list[ProfileParameters] = []
    runs_used = 0
    for candidate in base_candidates:
        seed_vector = _candidate_to_search_vector(
            candidate=candidate,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
        )
        for fixed_components in boundary_specs:
            runs_used += 1
            refined.extend(
                _refine_profile_with_fixed_components(
                    seed_vector=seed_vector,
                    fixed_components=fixed_components,
                    bounds=bounds,
                    profile_builder=profile_builder,
                    target_point=target_point,
                    config=config,
                    max_nfev=max_nfev,
                )
            )
    return refined, int(runs_used)


def _boundary_refine_kop_candidates(
    *,
    candidates: list[ProfileParameters],
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    config: TrajectoryConfig,
    search_settings: TurnSearchSettings,
) -> tuple[list[ProfileParameters], int]:
    if not candidates:
        return [], 0

    ranked = sorted(
        candidates,
        key=lambda candidate: _optimization_candidate_sort_key(
            candidate, OPTIMIZATION_MINIMIZE_KOP
        ),
    )
    base_candidates = ranked[:KOP_BOUNDARY_REFINEMENT_MAX_BASE_CANDIDATES]
    max_nfev = max(
        40,
        int(
            round(
                KOP_BOUNDARY_REFINEMENT_MAX_NFEV_FACTOR
                * float(search_settings.local_max_nfev)
            )
        ),
    )
    build_upper = float(bounds[0][1])
    kop_lower = float(bounds[1][0])
    boundary_specs = (
        {1: kop_lower},
        {0: build_upper, 1: kop_lower},
    )

    refined: list[ProfileParameters] = []
    runs_used = 0
    for candidate in base_candidates:
        seed_vector = _candidate_to_search_vector(
            candidate=candidate,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
        )
        for fixed_components in boundary_specs:
            runs_used += 1
            refined.extend(
                _refine_profile_with_fixed_components(
                    seed_vector=seed_vector,
                    fixed_components=fixed_components,
                    bounds=bounds,
                    profile_builder=profile_builder,
                    target_point=target_point,
                    config=config,
                    max_nfev=max_nfev,
                )
            )
    return refined, int(runs_used)


def _direction_vector_ned(
    *,
    inc_deg: float,
    azimuth_deg: float,
) -> np.ndarray:
    inc_rad = float(np.radians(inc_deg))
    azimuth_rad = float(np.radians(azimuth_deg))
    return np.array(
        [
            np.sin(inc_rad) * np.cos(azimuth_rad),
            np.sin(inc_rad) * np.sin(azimuth_rad),
            np.cos(inc_rad),
        ],
        dtype=float,
    )


def _arc_increment_ned(
    *,
    length_m: float,
    inc_from_deg: float,
    azimuth_from_deg: float,
    inc_to_deg: float,
    azimuth_to_deg: float,
) -> np.ndarray:
    dn, de, dz = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=float(inc_from_deg),
        azi1_deg=float(azimuth_from_deg),
        md2_m=float(length_m),
        inc2_deg=float(inc_to_deg),
        azi2_deg=float(azimuth_to_deg),
    )
    return np.array([dn, de, dz], dtype=float)


def _recover_profile_from_build_and_kop(
    *,
    geometry: SectionGeometry,
    build_dls_deg_per_30m: float,
    kop_vertical_m: float,
    min_build_segment_m: float,
    post_entry: PostEntrySection,
    zero_azimuth_turn: bool,
    reference_candidates: tuple[ProfileParameters, ...] = (),
) -> ProfileParameters | None:
    if zero_azimuth_turn:
        return _build_profile_from_effective_targets(
            geometry=geometry,
            dls_build_deg_per_30m=build_dls_deg_per_30m,
            s_to_t1_m=geometry.s1_m,
            z_to_t1_m=geometry.z1_m - kop_vertical_m,
            kop_vertical_m=kop_vertical_m,
            min_build_segment_m=min_build_segment_m,
            post_entry=post_entry,
        )
    return _recover_turn_profile_from_build_and_kop(
        geometry=geometry,
        build_dls_deg_per_30m=build_dls_deg_per_30m,
        kop_vertical_m=kop_vertical_m,
        min_build_segment_m=min_build_segment_m,
        post_entry=post_entry,
        reference_candidates=reference_candidates,
    )


def _recover_turn_profile_from_build_and_kop(
    *,
    geometry: SectionGeometry,
    build_dls_deg_per_30m: float,
    kop_vertical_m: float,
    min_build_segment_m: float,
    post_entry: PostEntrySection,
    reference_candidates: tuple[ProfileParameters, ...] = (),
) -> ProfileParameters | None:
    if build_dls_deg_per_30m <= SMALL or kop_vertical_m < 0.0:
        return None

    target_from_kop = np.array(
        [
            float(geometry.t1_north_m),
            float(geometry.t1_east_m),
            float(geometry.t1_tvd_m - kop_vertical_m),
        ],
        dtype=float,
    )
    if target_from_kop[2] <= SMALL:
        return None

    radius_m = _radius_from_dls(build_dls_deg_per_30m)
    min_build = max(float(min_build_segment_m), SMALL)
    pos_scale = 1.0

    def build_candidate(
        inc_hold_deg: float, azimuth_hold_deg: float
    ) -> tuple[ProfileParameters | None, np.ndarray, float]:
        inc_hold = float(inc_hold_deg)
        azimuth_hold = float(_normalize_azimuth_deg(azimuth_hold_deg))
        if inc_hold <= SMALL or inc_hold >= float(geometry.inc_entry_deg) - SMALL:
            return None, np.full(3, 1e9, dtype=float), -1e9

        build1_dogleg_rad = float(
            dogleg_angle_rad(0.0, azimuth_hold, inc_hold, azimuth_hold)
        )
        build2_dogleg_rad = float(
            dogleg_angle_rad(
                inc_hold,
                azimuth_hold,
                geometry.inc_entry_deg,
                geometry.azimuth_entry_deg,
            )
        )
        build1_length_m = float(radius_m * build1_dogleg_rad)
        build2_length_m = float(radius_m * build2_dogleg_rad)
        if build1_length_m < min_build - SMALL or build2_length_m < min_build - SMALL:
            return None, np.full(3, 1e9, dtype=float), -1e9

        arc1 = _arc_increment_ned(
            length_m=build1_length_m,
            inc_from_deg=0.0,
            azimuth_from_deg=azimuth_hold,
            inc_to_deg=inc_hold,
            azimuth_to_deg=azimuth_hold,
        )
        arc2 = _arc_increment_ned(
            length_m=build2_length_m,
            inc_from_deg=inc_hold,
            azimuth_from_deg=azimuth_hold,
            inc_to_deg=geometry.inc_entry_deg,
            azimuth_to_deg=geometry.azimuth_entry_deg,
        )
        hold_direction = _direction_vector_ned(
            inc_deg=inc_hold, azimuth_deg=azimuth_hold
        )
        remainder = target_from_kop - arc1 - arc2
        hold_length_m = float(np.dot(remainder, hold_direction))
        perpendicular = remainder - hold_length_m * hold_direction
        if hold_length_m < -SMALL:
            return None, perpendicular, hold_length_m

        candidate = _profile_same_direction_with_turn(
            geometry=geometry,
            dls_build1_deg_per_30m=build_dls_deg_per_30m,
            dls_build2_deg_per_30m=build_dls_deg_per_30m,
            kop_vertical_m=kop_vertical_m,
            inc_hold_deg=inc_hold,
            hold_length_m=float(max(hold_length_m, 0.0)),
            azimuth_hold_deg=azimuth_hold,
            min_build_segment_m=min_build_segment_m,
            post_entry=post_entry,
        )
        return candidate, perpendicular, hold_length_m

    def residual(values: np.ndarray) -> np.ndarray:
        candidate, perpendicular, hold_length_m = build_candidate(
            inc_hold_deg=float(values[0]),
            azimuth_hold_deg=float(values[1]),
        )
        if candidate is None:
            return np.array([1e6, 1e6, 1e6, 1e6], dtype=float)
        negative_hold_penalty = float(max(-hold_length_m, 0.0))
        return np.array(
            [
                float(perpendicular[0] / pos_scale),
                float(perpendicular[1] / pos_scale),
                float(perpendicular[2] / pos_scale),
                negative_hold_penalty / pos_scale,
            ],
            dtype=float,
        )

    inc_seed_default = float(
        np.clip(geometry.inc_entry_deg * 0.6, 0.5, geometry.inc_entry_deg - 0.5)
    )
    az_seed_default = float(
        _mid_azimuth_deg(geometry.azimuth_surface_t1_deg, geometry.azimuth_entry_deg)
    )
    seeds: list[np.ndarray] = []
    for reference_candidate in reference_candidates:
        seeds.append(
            np.array(
                [
                    float(
                        np.clip(
                            reference_candidate.inc_hold_deg,
                            0.5,
                            geometry.inc_entry_deg - 0.5,
                        )
                    ),
                    float(_normalize_azimuth_deg(reference_candidate.azimuth_hold_deg)),
                ],
                dtype=float,
            )
        )
    seeds.extend(
        [
            np.array([inc_seed_default, az_seed_default], dtype=float),
            np.array(
                [inc_seed_default, float(geometry.azimuth_surface_t1_deg)], dtype=float
            ),
            np.array(
                [inc_seed_default, float(geometry.azimuth_entry_deg)], dtype=float
            ),
        ]
    )

    bounds_lower = np.array([0.5, 0.0], dtype=float)
    bounds_upper = np.array([float(geometry.inc_entry_deg - 0.5), 360.0], dtype=float)
    best_candidate: ProfileParameters | None = None
    best_score = np.inf

    for seed in _dedupe_seed_vectors(seeds):
        clipped_seed = np.clip(seed, bounds_lower, bounds_upper)
        solution = least_squares(
            fun=lambda values: residual(np.asarray(values, dtype=float)),
            x0=clipped_seed,
            bounds=(bounds_lower, bounds_upper),
            method="trf",
            jac="2-point",
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=80,
        )
        probes = [clipped_seed]
        if bool(solution.success) and np.all(np.isfinite(solution.x)):
            probes.append(
                np.clip(np.asarray(solution.x, dtype=float), bounds_lower, bounds_upper)
            )

        for probe in probes:
            candidate, perpendicular, hold_length_m = build_candidate(
                inc_hold_deg=float(probe[0]),
                azimuth_hold_deg=float(probe[1]),
            )
            if candidate is None or hold_length_m < -SMALL:
                continue
            score = float(np.linalg.norm(perpendicular))
            if score < best_score:
                best_score = score
                best_candidate = candidate

    return best_candidate


def _two_dimensional_md_refine_candidates(
    *,
    candidates: list[ProfileParameters],
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    geometry: SectionGeometry,
    post_entry: PostEntrySection,
    target_point: np.ndarray,
    config: TrajectoryConfig,
) -> tuple[list[ProfileParameters], int]:
    if not candidates:
        return [], 0

    ranked = sorted(candidates, key=_default_candidate_sort_key)
    base_candidates = ranked[:MD_2D_REFINEMENT_MAX_BASE_CANDIDATES]
    build_upper = float(bounds[0][1])
    kop_lower = float(bounds[1][0])
    seeds: list[np.ndarray] = []
    for candidate in base_candidates:
        build_unit = _encode_build_dls_to_unit(
            build_dls_deg_per_30m=float(candidate.dls_build1_deg_per_30m),
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        )
        current = np.array([build_unit, float(candidate.kop_vertical_m)], dtype=float)
        seeds.extend(
            [
                current,
                np.array([build_upper, current[1]], dtype=float),
                np.array([current[0], kop_lower], dtype=float),
                np.array([build_upper, kop_lower], dtype=float),
            ]
        )
    seed_vectors = _dedupe_seed_vectors(
        [
            _clip_to_bounds(
                np.asarray(seed, dtype=float),
                bounds=(bounds[0], bounds[1]),
            )
            for seed in seeds
        ]
    )[:MD_2D_REFINEMENT_MAX_SEEDS]

    if not seed_vectors:
        return [], 0

    base_pairs = [
        (
            _encode_build_dls_to_unit(
                build_dls_deg_per_30m=float(candidate.dls_build1_deg_per_30m),
                lower_dls_deg_per_30m=lower_dls_deg_per_30m,
                upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            ),
            float(candidate.kop_vertical_m),
            candidate,
        )
        for candidate in base_candidates
    ]
    optimized: list[ProfileParameters] = []
    runs_used = 0
    eval_cache: dict[tuple[float, float], CandidateOptimizationEvaluation] = {}

    def evaluate(values: np.ndarray) -> CandidateOptimizationEvaluation:
        pair = tuple(np.round(np.asarray(values, dtype=float), decimals=10).tolist())
        cached = eval_cache.get(pair)
        if cached is not None:
            return cached

        build_dls = _decode_build_dls_from_unit(
            unit_value=float(values[0]),
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        )
        kop_vertical_m = float(values[1])
        ordered_reference_candidates = tuple(
            item[2]
            for item in sorted(
                base_pairs,
                key=lambda item: (
                    abs(float(values[0]) - float(item[0]))
                    + abs(kop_vertical_m - float(item[1])),
                    _default_candidate_sort_key(item[2]),
                ),
            )
        )
        candidate = _recover_profile_from_build_and_kop(
            geometry=geometry,
            build_dls_deg_per_30m=build_dls,
            kop_vertical_m=kop_vertical_m,
            min_build_segment_m=float(config.min_structural_segment_m),
            post_entry=post_entry,
            zero_azimuth_turn=zero_azimuth_turn,
            reference_candidates=ordered_reference_candidates,
        )
        evaluation = _evaluate_profile_candidate(
            candidate=candidate,
            target_point=target_point,
            config=config,
        )
        eval_cache[pair] = evaluation
        return evaluation

    objective_bounds = [bounds[0], bounds[1]]
    for seed in seed_vectors:
        seed_pair = _clip_to_bounds(
            np.asarray(seed, dtype=float), bounds=tuple(objective_bounds)
        )
        runs_used += 1
        result = minimize(
            fun=lambda values: float(
                evaluate(np.asarray(values, dtype=float)).md_total_m
            ),
            x0=seed_pair,
            method="SLSQP",
            bounds=list(objective_bounds),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda values: evaluate(
                        np.asarray(values, dtype=float)
                    ).t1_margin_m,
                },
                {
                    "type": "ineq",
                    "fun": lambda values: evaluate(
                        np.asarray(values, dtype=float)
                    ).build1_margin_m,
                },
                {
                    "type": "ineq",
                    "fun": lambda values: evaluate(
                        np.asarray(values, dtype=float)
                    ).build2_margin_m,
                },
                {
                    "type": "ineq",
                    "fun": lambda values: evaluate(
                        np.asarray(values, dtype=float)
                    ).max_inc_margin_deg,
                },
                {
                    "type": "ineq",
                    "fun": lambda values: evaluate(
                        np.asarray(values, dtype=float)
                    ).horizontal_dls_margin_deg_per_30m,
                },
            ],
            options={
                "maxiter": MD_2D_REFINEMENT_MAXITER,
                "ftol": 1e-8,
                "disp": False,
            },
        )
        probes = [seed_pair]
        if bool(result.success) and np.all(np.isfinite(result.x)):
            probes.append(
                _clip_to_bounds(
                    np.asarray(result.x, dtype=float), bounds=tuple(objective_bounds)
                )
            )

        for probe in probes:
            evaluation = evaluate(probe)
            if evaluation.feasible and evaluation.candidate is not None:
                optimized.append(evaluation.candidate)
    return optimized, int(runs_used)


def _select_feasible_candidate(
    *,
    candidates: list[ProfileParameters],
    surface: Point3D,
    geometry: SectionGeometry,
    post_entry: PostEntrySection,
    config: TrajectoryConfig,
    optimization_context: AntiCollisionOptimizationContext | None,
    zero_azimuth_turn: bool,
    lower_dls_deg_per_30m: float,
    upper_dls_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    search_settings: TurnSearchSettings,
    progress_callback: ProgressCallback | None = None,
) -> TurnSolveResult:
    deduped_candidates = _dedupe_profile_candidates(
        candidates=candidates,
        zero_azimuth_turn=zero_azimuth_turn,
        lower_dls_deg_per_30m=lower_dls_deg_per_30m,
        upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        bounds=bounds,
    )
    if not deduped_candidates:
        raise PlanningError("No feasible candidate available for final selection.")

    optimization_mode = str(config.optimization_mode)
    if optimization_mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE:
        if optimization_context is None:
            raise PlanningError(
                "Anti-collision optimization mode requires an optimization context."
            )
        optimization_bounds = _split_build_optimization_bounds(bounds=bounds)
        optimization_profile_builder = _make_turn_profile_builder(
            geometry=geometry,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            min_build_segment_m=float(config.min_structural_segment_m),
            post_entry=post_entry,
            split_build=True,
        )
        return _select_anti_collision_candidate(
            candidates=deduped_candidates,
            surface=surface,
            config=config,
            optimization_context=optimization_context,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
            profile_builder=profile_builder,
            target_point=target_point,
            search_settings=search_settings,
            optimization_bounds=optimization_bounds,
            optimization_profile_builder=optimization_profile_builder,
            split_build=True,
            progress_callback=progress_callback,
        )
    if optimization_mode == OPTIMIZATION_NONE:
        selected = min(deduped_candidates, key=_default_candidate_sort_key)
        return TurnSolveResult(
            params=selected,
            optimization=OptimizationOutcome(
                mode=optimization_mode,
                status="off",
                objective_value=float(selected.md_total_m),
                theoretical_lower_bound=0.0,
                absolute_gap_value=0.0,
                relative_gap_pct=0.0,
                seeds_used=len(deduped_candidates),
                runs_used=0,
            ),
        )

    lower_bound = _theoretical_objective_lower_bound(
        geometry=geometry,
        config=config,
        mode=optimization_mode,
    )
    ranked = sorted(
        deduped_candidates,
        key=lambda candidate: _optimization_candidate_sort_key(
            candidate, optimization_mode
        ),
    )
    best_seed = ranked[0]
    seed_sort_key = _optimization_candidate_sort_key(best_seed, optimization_mode)
    best_seed_objective = _optimization_objective_value(best_seed, optimization_mode)
    if _optimization_target_reached(
        objective_value=best_seed_objective,
        theoretical_lower_bound=lower_bound,
        mode=optimization_mode,
    ):
        status = (
            "within_md_theoretical_gap"
            if optimization_mode == OPTIMIZATION_MINIMIZE_MD
            else "at_min_kop_limit"
        )
        return TurnSolveResult(
            params=best_seed,
            optimization=OptimizationOutcome(
                mode=optimization_mode,
                status=status,
                objective_value=best_seed_objective,
                theoretical_lower_bound=lower_bound,
                absolute_gap_value=_optimization_absolute_gap(
                    objective_value=best_seed_objective,
                    theoretical_lower_bound=lower_bound,
                ),
                relative_gap_pct=_optimization_relative_gap_pct(
                    objective_value=best_seed_objective,
                    theoretical_lower_bound=lower_bound,
                ),
                seeds_used=len(ranked),
                runs_used=0,
            ),
        )

    optimization_bounds = _split_build_optimization_bounds(bounds=bounds)
    optimization_profile_builder = _make_turn_profile_builder(
        geometry=geometry,
        zero_azimuth_turn=zero_azimuth_turn,
        lower_dls_deg_per_30m=lower_dls_deg_per_30m,
        upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        min_build_segment_m=float(config.min_structural_segment_m),
        post_entry=post_entry,
        split_build=True,
    )
    optimization_seed_vectors = _collect_optimization_seed_vectors(
        candidates=ranked,
        mode=optimization_mode,
        zero_azimuth_turn=zero_azimuth_turn,
        lower_dls_deg_per_30m=lower_dls_deg_per_30m,
        upper_dls_deg_per_30m=upper_dls_deg_per_30m,
        bounds=optimization_bounds,
        split_build=True,
    )
    optimized_candidates: list[ProfileParameters] = list(ranked)
    runs_used = 0
    kop_lower_bound = float(bounds[1][0])

    if optimization_mode == OPTIMIZATION_MINIMIZE_MD:
        boundary_candidates, boundary_runs_used = _boundary_refine_md_candidates(
            candidates=ranked,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
            profile_builder=profile_builder,
            target_point=target_point,
            config=config,
            search_settings=search_settings,
        )
        optimized_candidates.extend(boundary_candidates)
        runs_used += int(boundary_runs_used)
        current_best_after_boundary = min(
            optimized_candidates,
            key=lambda candidate: _optimization_candidate_sort_key(
                candidate, optimization_mode
            ),
        )
        current_best_objective = _optimization_objective_value(
            current_best_after_boundary,
            optimization_mode,
        )
        if _optimization_target_reached(
            objective_value=current_best_objective,
            theoretical_lower_bound=lower_bound,
            mode=optimization_mode,
        ):
            improved = bool(
                _optimization_candidate_sort_key(
                    current_best_after_boundary, optimization_mode
                )
                < seed_sort_key
            )
            return TurnSolveResult(
                params=current_best_after_boundary,
                optimization=OptimizationOutcome(
                    mode=optimization_mode,
                    status=(
                        "within_md_theoretical_gap_after_boundary"
                        if improved
                        else "within_md_theoretical_gap"
                    ),
                    objective_value=current_best_objective,
                    theoretical_lower_bound=lower_bound,
                    absolute_gap_value=_optimization_absolute_gap(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    relative_gap_pct=_optimization_relative_gap_pct(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    seeds_used=len(optimization_seed_vectors),
                    runs_used=int(runs_used),
                ),
            )

        two_d_candidates, two_d_runs_used = _two_dimensional_md_refine_candidates(
            candidates=optimized_candidates,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
            geometry=geometry,
            post_entry=post_entry,
            target_point=target_point,
            config=config,
        )
        optimized_candidates.extend(two_d_candidates)
        runs_used += int(two_d_runs_used)
        current_best_after_2d = min(
            optimized_candidates,
            key=lambda candidate: _optimization_candidate_sort_key(
                candidate, optimization_mode
            ),
        )
        current_best_objective = _optimization_objective_value(
            current_best_after_2d,
            optimization_mode,
        )
        if _is_md_boundary_extremum_candidate(
            candidate=current_best_after_2d,
            build_dls_upper_deg_per_30m=upper_dls_deg_per_30m,
            kop_lower_m=kop_lower_bound,
        ):
            improved = bool(
                _optimization_candidate_sort_key(
                    current_best_after_2d, optimization_mode
                )
                < seed_sort_key
            )
            return TurnSolveResult(
                params=current_best_after_2d,
                optimization=OptimizationOutcome(
                    mode=optimization_mode,
                    status=("at_md_boundary_extremum" if improved else "seed_selected"),
                    objective_value=current_best_objective,
                    theoretical_lower_bound=lower_bound,
                    absolute_gap_value=_optimization_absolute_gap(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    relative_gap_pct=_optimization_relative_gap_pct(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    seeds_used=len(optimization_seed_vectors),
                    runs_used=int(runs_used),
                ),
            )
        if _optimization_target_reached(
            objective_value=current_best_objective,
            theoretical_lower_bound=lower_bound,
            mode=optimization_mode,
        ):
            improved = bool(
                _optimization_candidate_sort_key(
                    current_best_after_2d, optimization_mode
                )
                < seed_sort_key
            )
            return TurnSolveResult(
                params=current_best_after_2d,
                optimization=OptimizationOutcome(
                    mode=optimization_mode,
                    status=(
                        "within_md_theoretical_gap_after_2d"
                        if improved
                        else "within_md_theoretical_gap"
                    ),
                    objective_value=current_best_objective,
                    theoretical_lower_bound=lower_bound,
                    absolute_gap_value=_optimization_absolute_gap(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    relative_gap_pct=_optimization_relative_gap_pct(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    seeds_used=len(optimization_seed_vectors),
                    runs_used=int(runs_used),
                ),
            )

    if optimization_mode == OPTIMIZATION_MINIMIZE_KOP:
        boundary_candidates, boundary_runs_used = _boundary_refine_kop_candidates(
            candidates=ranked,
            zero_azimuth_turn=zero_azimuth_turn,
            lower_dls_deg_per_30m=lower_dls_deg_per_30m,
            upper_dls_deg_per_30m=upper_dls_deg_per_30m,
            bounds=bounds,
            profile_builder=profile_builder,
            target_point=target_point,
            config=config,
            search_settings=search_settings,
        )
        optimized_candidates.extend(boundary_candidates)
        runs_used += int(boundary_runs_used)
        current_best_after_boundary = min(
            optimized_candidates,
            key=lambda candidate: _optimization_candidate_sort_key(
                candidate, optimization_mode
            ),
        )
        current_best_objective = _optimization_objective_value(
            current_best_after_boundary,
            optimization_mode,
        )
        if _optimization_target_reached(
            objective_value=current_best_objective,
            theoretical_lower_bound=lower_bound,
            mode=optimization_mode,
        ):
            return TurnSolveResult(
                params=current_best_after_boundary,
                optimization=OptimizationOutcome(
                    mode=optimization_mode,
                    status="at_min_kop_limit",
                    objective_value=current_best_objective,
                    theoretical_lower_bound=lower_bound,
                    absolute_gap_value=_optimization_absolute_gap(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    relative_gap_pct=_optimization_relative_gap_pct(
                        objective_value=current_best_objective,
                        theoretical_lower_bound=lower_bound,
                    ),
                    seeds_used=len(optimization_seed_vectors),
                    runs_used=int(runs_used),
                ),
            )

    if progress_callback is not None:
        _emit_progress(
            progress_callback,
            (
                f"Оптимизация: режим `{optimization_mode}`, "
                f"локальные старты {len(optimization_seed_vectors)} с независимыми BUILD1/BUILD2."
            ),
            0.97,
        )

    last_values: tuple[float, ...] | None = None
    last_eval: CandidateOptimizationEvaluation | None = None

    def evaluate(values: np.ndarray) -> CandidateOptimizationEvaluation:
        nonlocal last_values, last_eval
        vector = tuple(np.asarray(values, dtype=float).tolist())
        if last_values == vector and last_eval is not None:
            return last_eval
        last_values = vector
        last_eval = _evaluate_candidate_for_optimization(
            values=np.asarray(values, dtype=float),
            profile_builder=optimization_profile_builder,
            target_point=target_point,
            config=config,
        )
        return last_eval

    def objective(values: np.ndarray) -> float:
        evaluation = evaluate(values)
        if not evaluation.feasible or evaluation.candidate is None:
            return 1e12
        if optimization_mode == OPTIMIZATION_MINIMIZE_KOP:
            return float(evaluation.kop_vertical_m)
        return float(evaluation.md_total_m)

    constraints = [
        {"type": "ineq", "fun": lambda values: evaluate(values).t1_margin_m},
        {"type": "ineq", "fun": lambda values: evaluate(values).build1_margin_m},
        {"type": "ineq", "fun": lambda values: evaluate(values).build2_margin_m},
        {"type": "ineq", "fun": lambda values: evaluate(values).max_inc_margin_deg},
        {
            "type": "ineq",
            "fun": lambda values: evaluate(values).horizontal_dls_margin_deg_per_30m,
        },
    ]

    maxiter = max(
        40, min(140, int(round(0.30 * float(search_settings.local_max_nfev))))
    )
    for seed_vector in optimization_seed_vectors:
        seed = _clip_to_bounds(seed_vector, bounds=optimization_bounds)
        runs_used += 1
        result = minimize(
            fun=objective,
            x0=seed,
            method="SLSQP",
            bounds=list(optimization_bounds),
            constraints=constraints,
            options={
                "maxiter": int(maxiter),
                "ftol": 1e-8,
                "disp": False,
            },
        )
        probes = [seed]
        if bool(result.success) and np.all(np.isfinite(result.x)):
            probes.append(
                _clip_to_bounds(
                    np.asarray(result.x, dtype=float),
                    bounds=optimization_bounds,
                )
            )
        for probe in probes:
            evaluation = evaluate(probe)
            if not evaluation.feasible or evaluation.candidate is None:
                continue
            optimized_candidates.append(evaluation.candidate)

        current_best = min(
            optimized_candidates,
            key=lambda candidate: _optimization_candidate_sort_key(
                candidate, optimization_mode
            ),
        )
        current_best_objective = _optimization_objective_value(
            current_best, optimization_mode
        )
        if _optimization_target_reached(
            objective_value=current_best_objective,
            theoretical_lower_bound=lower_bound,
            mode=optimization_mode,
        ):
            break

    selected = min(
        optimized_candidates,
        key=lambda candidate: _optimization_candidate_sort_key(
            candidate, optimization_mode
        ),
    )
    selected_objective = _optimization_objective_value(selected, optimization_mode)
    selected_sort_key = _optimization_candidate_sort_key(selected, optimization_mode)
    improved = bool(
        selected_sort_key < seed_sort_key
        and (
            selected_objective
            < best_seed_objective - OPTIMIZATION_IMPROVEMENT_TOLERANCE
            or selected_sort_key != seed_sort_key
        )
    )
    return TurnSolveResult(
        params=selected,
        optimization=OptimizationOutcome(
            mode=optimization_mode,
            status="refined" if improved else "seed_selected",
            objective_value=selected_objective,
            theoretical_lower_bound=lower_bound,
            absolute_gap_value=_optimization_absolute_gap(
                objective_value=selected_objective,
                theoretical_lower_bound=lower_bound,
            ),
            relative_gap_pct=_optimization_relative_gap_pct(
                objective_value=selected_objective,
                theoretical_lower_bound=lower_bound,
            ),
            seeds_used=len(optimization_seed_vectors),
            runs_used=int(runs_used),
        ),
    )


def _turn_seed_vectors(
    geometry: SectionGeometry,
    build_dls_lower_deg_per_30m: float,
    build_dls_upper_deg_per_30m: float,
    bounds: tuple[tuple[float, float], ...],
    search_settings: TurnSearchSettings,
    zero_azimuth_turn: bool,
    preferred_profiles: tuple[ProfileParameters, ...] = (),
) -> list[np.ndarray]:
    build_unit_min, build_unit_max = bounds[0]
    kop_min, kop_max = bounds[1]
    inc_min, inc_max = bounds[2]
    hold_min, hold_max = bounds[3]
    build_units = _build_unit_seed_values(search_settings=search_settings)
    build_units = np.array(
        [
            float(np.clip(build_unit, build_unit_min, build_unit_max))
            for build_unit in build_units
        ],
        dtype=float,
    )
    build_mid = float(np.clip(0.5, build_unit_min, build_unit_max))

    hold_guess = float(np.clip(max(geometry.s1_m * 0.25, 50.0), hold_min, hold_max))
    hold_high = float(np.clip(hold_guess * 1.8, hold_min, hold_max))
    kop_mid = float(np.clip(geometry.z1_m * 0.45, kop_min, kop_max))
    kop_high = float(np.clip(geometry.z1_m * 0.70, kop_min, kop_max))
    inc_low = float(np.clip(geometry.inc_entry_deg * 0.35, inc_min, inc_max))
    inc_mid = float(np.clip(geometry.inc_entry_deg * 0.55, inc_min, inc_max))
    inc_high = float(np.clip(geometry.inc_entry_deg * 0.75, inc_min, inc_max))
    if zero_azimuth_turn:
        seed_vectors = [
            np.array([build_unit, kop_min, inc_mid, hold_min], dtype=float)
            for build_unit in build_units
        ]
        seed_vectors.extend(
            np.array([build_unit, kop_min, inc_mid, hold_guess], dtype=float)
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array([build_unit, kop_mid, inc_mid, hold_guess], dtype=float)
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array([build_unit, kop_mid, inc_low, hold_high], dtype=float)
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array([build_unit, kop_high, inc_high, hold_min], dtype=float)
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array(
                [
                    _encode_build_dls_to_unit(
                        build_dls_deg_per_30m=float(candidate.dls_build1_deg_per_30m),
                        lower_dls_deg_per_30m=build_dls_lower_deg_per_30m,
                        upper_dls_deg_per_30m=build_dls_upper_deg_per_30m,
                    ),
                    float(candidate.kop_vertical_m),
                    float(candidate.inc_hold_deg),
                    float(candidate.hold_length_m),
                ],
                dtype=float,
            )
            for candidate in preferred_profiles
        )
    else:
        az_surface = float(geometry.azimuth_surface_t1_deg)
        az_entry = float(geometry.azimuth_entry_deg)
        az_mid = float(_mid_azimuth_deg(az_surface, az_entry))
        seed_vectors = [
            np.array([build_unit, kop_min, inc_mid, hold_min, az_surface], dtype=float)
            for build_unit in build_units
        ]
        seed_vectors.extend(
            np.array(
                [build_unit, kop_min, inc_mid, hold_guess, az_surface], dtype=float
            )
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array([build_unit, kop_min, inc_high, hold_guess, az_entry], dtype=float)
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array([build_unit, kop_mid, inc_mid, hold_guess, az_mid], dtype=float)
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array([build_unit, kop_mid, inc_low, hold_high, az_surface], dtype=float)
            for build_unit in build_units
        )
        seed_vectors.extend(
            np.array([build_unit, kop_high, inc_high, hold_min, az_entry], dtype=float)
            for build_unit in build_units
        )
    lattice_points = int(max(search_settings.seed_lattice_points, 0))
    if lattice_points <= 0:
        return _dedupe_seed_vectors(seed_vectors)

    hold_grid_max = float(
        np.clip(
            max(
                hold_guess * (1.8 + 0.4 * float(search_settings.restart_index)),
                120.0,
            ),
            hold_min,
            hold_max,
        )
    )
    kop_values = _axis_samples(kop_min, kop_max, lattice_points)
    inc_values = _axis_samples(inc_min, inc_max, lattice_points)
    hold_values = _axis_samples(hold_min, hold_grid_max, lattice_points)
    if zero_azimuth_turn:
        seed_vectors.extend(
            np.array([build_mid, kop, inc_mid, hold_guess], dtype=float)
            for kop in kop_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop_mid, inc, hold_guess], dtype=float)
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop_mid, inc_mid, hold], dtype=float)
            for hold in hold_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop, inc, hold_guess], dtype=float)
            for kop in kop_values
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop_mid, inc, hold], dtype=float)
            for inc in inc_values
            for hold in hold_values
        )
        seed_vectors.extend(
            np.array([build_unit, kop_mid, inc_mid, hold_guess], dtype=float)
            for build_unit in build_units
        )
    else:
        azimuth_values = _turn_azimuth_samples(az_surface, az_entry, lattice_points)
        seed_vectors.extend(
            np.array([build_mid, kop, inc_mid, hold_guess, az_mid], dtype=float)
            for kop in kop_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop_mid, inc, hold_guess, az_mid], dtype=float)
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop_mid, inc_mid, hold, az_mid], dtype=float)
            for hold in hold_values
        )
        seed_vectors.extend(
            np.array(
                [build_mid, kop_mid, inc_mid, hold_guess, azimuth_deg], dtype=float
            )
            for azimuth_deg in azimuth_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop, inc, hold_guess, az_mid], dtype=float)
            for kop in kop_values
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([build_mid, kop_mid, inc_mid, hold, azimuth_deg], dtype=float)
            for hold in hold_values
            for azimuth_deg in azimuth_values
        )
        seed_vectors.extend(
            np.array([build_unit, kop_mid, inc_mid, hold_guess, az_mid], dtype=float)
            for build_unit in build_units
        )
    return _dedupe_seed_vectors(seed_vectors)


def _build_unit_seed_values(search_settings: TurnSearchSettings) -> np.ndarray:
    level = max(int(search_settings.restart_index), 0)
    if level <= 0:
        return np.array([1.0], dtype=float)
    if level == 1:
        return np.array([1.0, 0.2], dtype=float)
    return np.array([1.0, 0.6, 0.2], dtype=float)


def _axis_samples(lower: float, upper: float, count: int) -> np.ndarray:
    if count <= 1 or abs(float(upper) - float(lower)) <= SMALL:
        return np.array([0.5 * (float(lower) + float(upper))], dtype=float)
    return np.linspace(float(lower), float(upper), int(count), dtype=float)


def _turn_azimuth_samples(
    azimuth_from_deg: float,
    azimuth_to_deg: float,
    count: int,
) -> np.ndarray:
    if count <= 1:
        return np.array(
            [_mid_azimuth_deg(azimuth_from_deg, azimuth_to_deg)], dtype=float
        )
    delta_deg = _shortest_azimuth_delta_deg(azimuth_from_deg, azimuth_to_deg)
    fractions = np.linspace(0.0, 1.0, int(count), dtype=float)
    return np.array(
        [
            _normalize_azimuth_deg(float(azimuth_from_deg + fraction * delta_deg))
            for fraction in fractions
        ],
        dtype=float,
    )


def _turn_residuals(
    values: np.ndarray,
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    config: TrajectoryConfig,
) -> np.ndarray:
    candidate = profile_builder(values)
    if candidate is None:
        return np.full(3, 1e6, dtype=float)

    endpoint = np.array(_estimate_t1_endpoint_for_profile(candidate), dtype=float)
    lateral_scale = max(float(config.lateral_tolerance_m), 1e-9)
    vertical_scale = max(float(config.vertical_tolerance_m), 1e-9)
    residuals = np.array(
        [
            (endpoint[0] - target_point[0]) / lateral_scale,
            (endpoint[1] - target_point[1]) / lateral_scale,
            (endpoint[2] - target_point[2]) / vertical_scale,
        ],
        dtype=float,
    )
    return residuals


def _turn_scalar_cost(
    values: np.ndarray,
    profile_builder: Callable[[np.ndarray], ProfileParameters | None],
    target_point: np.ndarray,
    config: TrajectoryConfig,
) -> float:
    candidate = profile_builder(values)
    if candidate is None:
        return 1e12
    endpoint = np.array(_estimate_t1_endpoint_for_profile(candidate), dtype=float)
    miss = _normalized_target_miss(
        endpoint=endpoint,
        target_point=target_point,
        config=config,
    )
    return float(miss * 1e6 + candidate.md_total_m)


def _build_profile_from_effective_targets(
    geometry: SectionGeometry,
    dls_build_deg_per_30m: float,
    s_to_t1_m: float,
    z_to_t1_m: float,
    kop_vertical_m: float,
    min_build_segment_m: float,
    post_entry: PostEntrySection,
) -> ProfileParameters | None:
    inc_entry_rad = geometry.inc_entry_deg * DEG2RAD
    if inc_entry_rad <= SMALL or inc_entry_rad >= (np.pi / 2.0) - SMALL:
        return None

    radius_m = _radius_from_dls(dls_build_deg_per_30m)
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
    min_build = max(float(min_build_segment_m), SMALL)
    if build1_length_m < min_build - SMALL or build2_length_m < min_build - SMALL:
        return None

    return ProfileParameters(
        kop_vertical_m=float(kop_vertical_m),
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_required_t1_t3_deg=float(geometry.inc_required_t1_t3_deg),
        inc_hold_deg=inc_hold_deg,
        dls_build1_deg_per_30m=float(dls_build_deg_per_30m),
        dls_build2_deg_per_30m=float(dls_build_deg_per_30m),
        build1_length_m=build1_length_m,
        hold_length_m=float(max(hold_length_m, 0.0)),
        build2_length_m=build2_length_m,
        horizontal_length_m=float(post_entry.total_length_m),
        horizontal_adjust_length_m=float(post_entry.transition_length_m),
        horizontal_hold_length_m=float(post_entry.hold_length_m),
        horizontal_inc_deg=float(post_entry.hold_inc_deg),
        horizontal_dls_deg_per_30m=float(post_entry.transition_dls_deg_per_30m),
        azimuth_hold_deg=float(geometry.azimuth_entry_deg),
        azimuth_entry_deg=float(geometry.azimuth_entry_deg),
    )


def _profile_same_direction_with_turn(
    geometry: SectionGeometry,
    dls_build1_deg_per_30m: float,
    dls_build2_deg_per_30m: float,
    kop_vertical_m: float,
    inc_hold_deg: float,
    hold_length_m: float,
    azimuth_hold_deg: float,
    min_build_segment_m: float,
    post_entry: PostEntrySection,
) -> ProfileParameters | None:
    if (
        dls_build1_deg_per_30m <= SMALL
        or dls_build2_deg_per_30m <= SMALL
        or kop_vertical_m < 0.0
    ):
        return None
    if hold_length_m < -SMALL:
        return None
    if inc_hold_deg <= SMALL or inc_hold_deg >= geometry.inc_entry_deg - SMALL:
        return None

    min_build = max(float(min_build_segment_m), SMALL)
    radius_build1_m = _radius_from_dls(dls_build1_deg_per_30m)
    radius_build2_m = _radius_from_dls(dls_build2_deg_per_30m)
    build1_dogleg_rad = float(
        dogleg_angle_rad(0.0, azimuth_hold_deg, inc_hold_deg, azimuth_hold_deg)
    )
    build2_dogleg_rad = float(
        dogleg_angle_rad(
            inc_hold_deg,
            azimuth_hold_deg,
            geometry.inc_entry_deg,
            geometry.azimuth_entry_deg,
        )
    )
    build1_length_m = float(radius_build1_m * build1_dogleg_rad)
    build2_length_m = float(radius_build2_m * build2_dogleg_rad)
    if build1_length_m < min_build - SMALL or build2_length_m < min_build - SMALL:
        return None

    return ProfileParameters(
        kop_vertical_m=float(kop_vertical_m),
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_required_t1_t3_deg=float(geometry.inc_required_t1_t3_deg),
        inc_hold_deg=float(inc_hold_deg),
        dls_build1_deg_per_30m=float(dls_build1_deg_per_30m),
        dls_build2_deg_per_30m=float(dls_build2_deg_per_30m),
        build1_length_m=build1_length_m,
        hold_length_m=float(max(hold_length_m, 0.0)),
        build2_length_m=build2_length_m,
        horizontal_length_m=float(post_entry.total_length_m),
        horizontal_adjust_length_m=float(post_entry.transition_length_m),
        horizontal_hold_length_m=float(post_entry.hold_length_m),
        horizontal_inc_deg=float(post_entry.hold_inc_deg),
        horizontal_dls_deg_per_30m=float(post_entry.transition_dls_deg_per_30m),
        azimuth_hold_deg=_normalize_azimuth_deg(azimuth_hold_deg),
        azimuth_entry_deg=_normalize_azimuth_deg(geometry.azimuth_entry_deg),
    )


def _solve_post_entry_section(
    ds_m: float,
    dz_m: float,
    inc_entry_deg: float,
    dls_deg_per_30m: float,
    max_inc_deg: float,
) -> PostEntrySection | None:
    if ds_m <= SMALL:
        return None

    inc_entry_rad = float(np.radians(inc_entry_deg))
    max_inc_rad = float(np.radians(max_inc_deg))
    if (
        inc_entry_rad < -SMALL
        or max_inc_rad <= SMALL
        or inc_entry_rad > max_inc_rad + SMALL
    ):
        return None

    if dls_deg_per_30m <= SMALL:
        sin_inc = float(np.sin(inc_entry_rad))
        cos_inc = float(np.cos(inc_entry_rad))
        mismatch = ds_m * cos_inc - dz_m * sin_inc
        if abs(mismatch) > 1e-3:
            return None
        hold_length_m = ds_m * sin_inc + dz_m * cos_inc
        if hold_length_m < -1e-3:
            return None
        hold_length_m = float(max(hold_length_m, 0.0))
        return PostEntrySection(
            total_length_m=hold_length_m,
            transition_length_m=0.0,
            hold_length_m=hold_length_m,
            hold_inc_deg=float(inc_entry_deg),
            transition_dls_deg_per_30m=0.0,
        )

    radius_m = _radius_from_dls(dls_deg_per_30m)
    tolerance = 1e-3

    def arc_displacement(inc_from_rad: float, inc_to_rad: float) -> tuple[float, float]:
        delta_rad = float(inc_to_rad - inc_from_rad)
        if abs(delta_rad) <= SMALL:
            return 0.0, 0.0
        direction = 1.0 if delta_rad > 0.0 else -1.0
        ds_arc = float(
            radius_m * (np.cos(inc_from_rad) - np.cos(inc_to_rad)) / direction
        )
        dz_arc = float(
            radius_m * (np.sin(inc_to_rad) - np.sin(inc_from_rad)) / direction
        )
        return ds_arc, dz_arc

    def residual(inc_hold_rad: float) -> float:
        ds_arc, dz_arc = arc_displacement(
            inc_from_rad=inc_entry_rad,
            inc_to_rad=inc_hold_rad,
        )
        ds_rem = ds_m - ds_arc
        dz_rem = dz_m - dz_arc
        return float(ds_rem * np.cos(inc_hold_rad) - dz_rem * np.sin(inc_hold_rad))

    def build_candidate(inc_hold_rad: float) -> PostEntrySection | None:
        ds_arc, dz_arc = arc_displacement(
            inc_from_rad=inc_entry_rad,
            inc_to_rad=inc_hold_rad,
        )
        ds_rem = ds_m - ds_arc
        dz_rem = dz_m - dz_arc
        hold_length_m = float(
            ds_rem * np.sin(inc_hold_rad) + dz_rem * np.cos(inc_hold_rad)
        )
        if hold_length_m < -tolerance:
            return None
        hold_length_m = float(max(hold_length_m, 0.0))

        ds_pred = float(ds_arc + hold_length_m * np.sin(inc_hold_rad))
        dz_pred = float(dz_arc + hold_length_m * np.cos(inc_hold_rad))
        miss = float(np.hypot(ds_pred - ds_m, dz_pred - dz_m))
        if miss > 5e-3:
            return None

        transition_length_m = float(radius_m * abs(inc_hold_rad - inc_entry_rad))
        return PostEntrySection(
            total_length_m=float(transition_length_m + hold_length_m),
            transition_length_m=transition_length_m,
            hold_length_m=hold_length_m,
            hold_inc_deg=float(np.degrees(inc_hold_rad)),
            transition_dls_deg_per_30m=float(dls_deg_per_30m),
        )

    samples = np.linspace(0.0, max_inc_rad, 721, dtype=float)
    values = np.array([residual(float(angle)) for angle in samples], dtype=float)
    candidate_angles: list[float] = []
    for idx in range(len(samples) - 1):
        f0 = values[idx]
        f1 = values[idx + 1]
        a0 = float(samples[idx])
        a1 = float(samples[idx + 1])
        if abs(f0) <= tolerance:
            candidate_angles.append(a0)
            continue
        if abs(f1) <= tolerance:
            candidate_angles.append(a1)
            continue
        if np.signbit(f0) == np.signbit(f1):
            continue
        lo = a0
        hi = a1
        flo = f0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fmid = residual(mid)
            if abs(fmid) <= tolerance:
                lo = mid
                hi = mid
                break
            if np.signbit(fmid) == np.signbit(flo):
                lo = mid
                flo = fmid
            else:
                hi = mid
        candidate_angles.append(0.5 * (lo + hi))

    if not candidate_angles:
        min_idx = int(np.argmin(np.abs(values)))
        candidate_angles.append(float(samples[min_idx]))

    candidates: list[PostEntrySection] = []
    for angle in candidate_angles:
        clamped = float(np.clip(angle, 0.0, max_inc_rad))
        candidate = build_candidate(clamped)
        if candidate is None:
            continue
        if candidate.hold_inc_deg > max_inc_deg + 1e-6:
            continue
        candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: (
            abs(candidate.hold_inc_deg - inc_entry_deg),
            candidate.total_length_m,
        )
    )
    return candidates[0]


def _required_post_entry_dls(
    geometry: SectionGeometry,
    inc_entry_deg: float,
    max_inc_deg: float,
    dls_min_deg_per_30m: float,
    dls_max_deg_per_30m: float,
) -> float | None:
    low = max(float(dls_min_deg_per_30m), SMALL)
    high = max(float(dls_max_deg_per_30m), low)
    if (
        _solve_post_entry_section(
            ds_m=geometry.ds_13_m,
            dz_m=geometry.dz_13_m,
            inc_entry_deg=inc_entry_deg,
            dls_deg_per_30m=low,
            max_inc_deg=max_inc_deg,
        )
        is not None
    ):
        return low
    if (
        _solve_post_entry_section(
            ds_m=geometry.ds_13_m,
            dz_m=geometry.dz_13_m,
            inc_entry_deg=inc_entry_deg,
            dls_deg_per_30m=high,
            max_inc_deg=max_inc_deg,
        )
        is None
    ):
        return None

    lo = low
    hi = high
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        candidate = _solve_post_entry_section(
            ds_m=geometry.ds_13_m,
            dz_m=geometry.dz_13_m,
            inc_entry_deg=inc_entry_deg,
            dls_deg_per_30m=mid,
            max_inc_deg=max_inc_deg,
        )
        if candidate is None:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _diagnose_post_entry_constraints(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
    horizontal_dls_deg_per_30m: float,
) -> list[str]:
    messages: list[str] = []
    if geometry.inc_required_t1_t3_deg > config.max_inc_deg + SMALL:
        messages.append(
            "t1->t3 geometry requires INC "
            f"{geometry.inc_required_t1_t3_deg:.2f} deg, above max INC {config.max_inc_deg:.2f} deg "
            "(overbend would be required). Make t3 deeper and/or reduce t1->t3 horizontal offset."
        )
        return messages

    current = _solve_post_entry_section(
        ds_m=geometry.ds_13_m,
        dz_m=geometry.dz_13_m,
        inc_entry_deg=geometry.inc_entry_deg,
        dls_deg_per_30m=horizontal_dls_deg_per_30m,
        max_inc_deg=config.max_inc_deg,
    )
    if current is not None:
        return messages

    required_dls = _required_post_entry_dls(
        geometry=geometry,
        inc_entry_deg=geometry.inc_entry_deg,
        max_inc_deg=config.max_inc_deg,
        dls_min_deg_per_30m=horizontal_dls_deg_per_30m,
        dls_max_deg_per_30m=max(horizontal_dls_deg_per_30m, 30.0),
    )
    if required_dls is not None:
        messages.append(
            "Post-entry t1->t3 connection is not feasible with BUILD/HORIZONTAL DLS limit "
            f"{horizontal_dls_deg_per_30m:.2f} deg/30m; requires about {required_dls:.2f} deg/30m. "
            "Increase BUILD/HORIZONTAL DLS limit or move t3 closer to t1 in section."
        )
    else:
        messages.append(
            "Post-entry t1->t3 connection is infeasible even with high DLS scan up to 30 deg/30m. "
            "Adjust geometry (increase t3 TVD and/or reduce t1->t3 horizontal offset) or relax entry INC target/max INC."
        )
    return messages


def _diagnose_same_direction_no_solution(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
    lower_dls: float,
    upper_dls: float,
    required_dls: float,
) -> list[str]:
    messages: list[str] = []
    horizontal_dls = _resolve_horizontal_dls(config=config)
    if geometry.s1_m <= SMALL:
        messages.append(
            "t1 lies behind the t1->t3 entry axis for the current entry azimuth. "
            f"Along-section offset to t1 is {geometry.s1_m:.2f} m. "
            "This is a reverse-entry geometry: tighter BUILD can shorten translation before t1 and make the target harder to reach."
        )
    build_vertical_available_m = max(
        geometry.z1_m - float(config.kop_min_vertical_m), 0.0
    )
    if (
        np.isfinite(required_dls)
        and required_dls > 0.0
        and upper_dls + SMALL < required_dls
    ):
        messages.append(
            "BUILD DLS upper bound is insufficient for t1 reach: "
            f"available max {upper_dls:.2f} deg/30m, required about {required_dls:.2f} deg/30m, "
            f"build_vertical_available={build_vertical_available_m:.1f} m, "
            f"kop_min_vertical={float(config.kop_min_vertical_m):.1f} m, "
            f"t1_tvd={geometry.z1_m:.1f} m."
        )
    if config.kop_min_vertical_m >= geometry.z1_m - SMALL:
        messages.append(
            "Minimum VERTICAL before KOP is too deep for current t1 TVD. "
            f"kop_min_vertical={config.kop_min_vertical_m:.1f} m, t1 TVD={geometry.z1_m:.1f} m."
        )
    elif build_vertical_available_m < 0.25 * geometry.z1_m:
        messages.append(
            "Minimum VERTICAL before KOP leaves very little room for BUILD section. "
            f"kop_min_vertical={config.kop_min_vertical_m:.1f} m, t1 TVD={geometry.z1_m:.1f} m, "
            f"available BUILD vertical={build_vertical_available_m:.1f} m."
        )
    messages.extend(
        _diagnose_post_entry_constraints(
            geometry=geometry,
            config=config,
            horizontal_dls_deg_per_30m=horizontal_dls,
        )
    )
    if lower_dls > upper_dls + SMALL:
        messages.append(
            "BUILD DLS interval is empty after constraints. "
            f"Configured range [{lower_dls:.2f}, {upper_dls:.2f}] deg/30m."
        )
    return messages


def _format_failure_diagnostics(lines: list[str]) -> str:
    compact: list[str] = []
    for line in lines:
        text = str(line).strip()
        if not text or text in compact:
            continue
        compact.append(text)
    if not compact:
        return ""
    return "\nReasons and actions:\n- " + "\n- ".join(compact)


def _resolve_build_dls_max(
    config: TrajectoryConfig,
    constrained_segments: tuple[str, ...],
) -> float:
    build_dls = float(config.dls_build_max_deg_per_30m)
    for segment in constrained_segments:
        segment_limit = config.dls_limits_deg_per_30m.get(segment)
        if segment_limit is None:
            continue
        build_dls = min(build_dls, float(segment_limit))
    if build_dls <= SMALL:
        segments = ", ".join(constrained_segments)
        raise PlanningError(
            f"No feasible BUILD DLS: constrained upper bound is {build_dls:.2f} deg/30m for segments [{segments}]."
        )
    return build_dls


def _resolve_horizontal_dls(config: TrajectoryConfig) -> float:
    horizontal_limit = float(config.dls_limits_deg_per_30m.get("HORIZONTAL", 0.0))
    return max(horizontal_limit, 0.0)


def _dedupe_seed_vectors(seed_vectors: list[np.ndarray]) -> list[np.ndarray]:
    deduped: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seed_vectors:
        key = tuple(np.round(np.asarray(seed, dtype=float), decimals=8).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(np.asarray(seed, dtype=float))
    return deduped


def _clip_to_bounds(
    values: np.ndarray, bounds: tuple[tuple[float, float], ...]
) -> np.ndarray:
    clipped = np.asarray(values, dtype=float).copy()
    for idx, (low, high) in enumerate(bounds):
        clipped[idx] = float(np.clip(clipped[idx], low, high))
    return clipped
