from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from pywp.classification import (
    TRAJECTORY_REVERSE_DIRECTION,
    WellClassification,
    classify_trajectory_type,
    classify_well,
    complexity_label,
    trajectory_type_label,
)
from pywp.mcm import (
    add_dls,
    compute_positions_min_curv,
    dogleg_angle_rad,
    minimum_curvature_increment,
)
from pywp.models import (
    ALLOWED_TURN_SOLVER_MODES,
    PlannerResult,
    Point3D,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    TrajectoryConfig,
)
from pywp.segments import BuildSegment, HoldSegment, HorizontalSegment, VerticalSegment
from pywp.trajectory import WellTrajectory

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
SMALL = 1e-9

TRAJECTORY_MODEL_LABEL = "Unified J Profile + Build + Azimuth Turn"
TURN_RESTART_GROWTH_FACTOR = 1.6


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
    inc_required_t1_t3_deg: float
    t1_cross_m: float
    t3_cross_m: float
    t1_east_m: float
    t1_north_m: float
    t1_tvd_m: float


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


class TrajectoryPlanner:
    def plan(
        self,
        surface: Point3D,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> PlannerResult:
        _emit_progress(progress_callback, "Планировщик: проверка конфигурации.", 0.03)
        _validate_config(config)

        _emit_progress(progress_callback, "Планировщик: подготовка геометрии цели.", 0.10)
        geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
        horizontal_offset_t1_m = _horizontal_offset(surface=surface, point=t1)
        target_direction = classify_trajectory_type(
            gv_m=float(geometry.z1_m),
            horizontal_offset_t1_m=float(horizontal_offset_t1_m),
        )

        zero_azimuth_turn = _is_zero_azimuth_turn_geometry(
            geometry=geometry,
            target_direction=target_direction,
            tolerance_m=config.pos_tolerance_m,
        )
        solver_mode_text = (
            "Солвер: единая оптимизация траектории (поворот по азимуту = 0)."
            if zero_azimuth_turn
            else "Солвер: единая оптимизация траектории с азимутальным поворотом."
        )
        _emit_progress(progress_callback, solver_mode_text, 0.18)
        (
            params,
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
            zero_azimuth_turn=zero_azimuth_turn,
            progress_callback=_scaled_progress_callback(
                progress_callback=progress_callback,
                start_fraction=0.16,
                end_fraction=0.90,
            ),
        )

        _emit_progress(progress_callback, "Планировщик: формирование выходной инклинометрии.", 0.96)
        output = compute_positions_min_curv(
            trajectory.stations(md_step_m=config.md_step_m),
            start=surface,
        )
        output = add_dls(output)
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
    zero_azimuth_turn: bool,
    progress_callback: ProgressCallback | None = None,
) -> tuple[
    ProfileParameters,
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
            params = _solve_turn_profile(
                geometry=geometry,
                config=config,
                zero_azimuth_turn=zero_azimuth_turn,
                search_settings=search_settings,
                progress_callback=attempt_progress,
            )
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
                config=config,
                turn_search_settings=search_settings,
                turn_restarts_used=restart_index,
            )
            return (
                params,
                search_settings,
                restart_index,
                trajectory,
                control,
                summary,
            )
        except PlanningError as exc:
            last_error = exc
            if restart_index >= max_attempts - 1 or not _is_retryable_solver_error(str(exc)):
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


def _build_validated_control_and_summary(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    geometry: SectionGeometry,
    horizontal_offset_t1_m: float,
    params: ProfileParameters,
    config: TrajectoryConfig,
    turn_search_settings: TurnSearchSettings | None,
    turn_restarts_used: int,
) -> tuple[WellTrajectory, pd.DataFrame, dict[str, float | str]]:
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
        turn_search_settings=turn_search_settings,
        turn_restarts_used=turn_restarts_used,
    )
    _assert_solution_is_valid(summary=summary, config=config)
    return trajectory, control, summary


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
    zero_azimuth_turn: bool,
    search_settings: TurnSearchSettings,
    progress_callback: ProgressCallback | None = None,
) -> ProfileParameters:
    build_dls = _resolve_build_dls_max(
        config=config,
        constrained_segments=("BUILD1", "BUILD2"),
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

    radius_m = _radius_from_dls(build_dls)
    min_build_deg = float((max(config.min_structural_segment_m, SMALL) / radius_m) * RAD2DEG)
    inc_hold_min = float(max(min_build_deg, 0.5))
    inc_hold_max = float(geometry.inc_entry_deg - min_build_deg)
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
            (kop_min, kop_max),
            (inc_hold_min, inc_hold_max),
            (0.0, hold_max),
        )
    else:
        bounds = (
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

    def profile_builder(values: np.ndarray) -> ProfileParameters | None:
        values_list = values.tolist()
        kop_vertical_m = float(values_list[0])
        inc_hold_deg = float(values_list[1])
        hold_length_m = float(values_list[2])
        azimuth_hold_deg = (
            float(geometry.azimuth_entry_deg)
            if zero_azimuth_turn
            else float(values_list[3])
        )
        return _profile_same_direction_with_turn(
            geometry=geometry,
            dls_build_deg_per_30m=build_dls,
            kop_vertical_m=kop_vertical_m,
            inc_hold_deg=inc_hold_deg,
            hold_length_m=hold_length_m,
            azimuth_hold_deg=azimuth_hold_deg,
            min_build_segment_m=float(config.min_structural_segment_m),
            post_entry=post_entry,
        )

    seed_vectors = _turn_seed_vectors(
        geometry=geometry,
        bounds=bounds,
        search_settings=search_settings,
        zero_azimuth_turn=zero_azimuth_turn,
        preferred_profiles=tuple(preferred_profiles),
    )
    _emit_progress(
        progress_callback,
        (
            f"Солвер: подготовка стартовых приближений "
            f"({len(seed_vectors)} шт., depth x{search_settings.search_depth_scale:.2f})."
        ),
        0.12,
    )

    candidates: list[ProfileParameters] = []
    best_miss = np.inf
    for preferred_profile in preferred_profiles:
        endpoint = np.array(
            _estimate_t1_endpoint_for_profile(preferred_profile), dtype=float
        )
        miss = float(_distance_3d(*endpoint, *target_point))
        best_miss = min(best_miss, miss)
        if miss > config.pos_tolerance_m + SMALL:
            continue
        if not _is_candidate_feasible(candidate=preferred_profile, config=config):
            continue
        candidates.append(preferred_profile)
    if zero_azimuth_turn and candidates:
        _emit_progress(
            progress_callback,
            "Солвер: аналитический zero-turn кандидат принят без дополнительной оптимизации.",
            1.00,
        )
        return min(
            candidates,
            key=lambda candidate: (
                candidate.md_total_m,
                candidate.build2_length_m,
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
            probes.append(_clip_to_bounds(np.asarray(solution.x, dtype=float), bounds=bounds))

        for probe in probes:
            candidate = profile_builder(probe)
            if candidate is None:
                continue
            endpoint = np.array(_estimate_t1_endpoint_for_profile(candidate), dtype=float)
            miss = float(_distance_3d(*endpoint, *target_point))
            best_miss = min(best_miss, miss)
            if miss > config.pos_tolerance_m + SMALL:
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
        return min(
            candidates,
            key=lambda candidate: (
                candidate.md_total_m,
                _candidate_turn_deg(candidate),
            ),
        )

    diagnostics = _diagnose_same_direction_no_solution(
        geometry=geometry,
        config=config,
        lower_dls=build_dls,
        upper_dls=build_dls,
        required_dls=_required_dls_for_t1_reach(
            s1_m=geometry.s1_m,
            z1_m=max(geometry.z1_m - float(config.kop_min_vertical_m), SMALL),
            inc_entry_deg=geometry.inc_entry_deg,
        ),
    )
    diagnostics.append(
        f"Solver endpoint miss to t1 after optimization is {best_miss:.2f} m (tolerance {config.pos_tolerance_m:.2f} m)."
    )
    raise PlanningError(
        "No valid trajectory solution found within configured limits. "
        f"Closest miss to t1 is {best_miss:.2f} m."
        + _format_failure_diagnostics(diagnostics)
    )


def _turn_seed_vectors(
    geometry: SectionGeometry,
    bounds: tuple[tuple[float, float], ...],
    search_settings: TurnSearchSettings,
    zero_azimuth_turn: bool,
    preferred_profiles: tuple[ProfileParameters, ...] = (),
) -> list[np.ndarray]:
    kop_min, kop_max = bounds[0]
    inc_min, inc_max = bounds[1]
    hold_min, hold_max = bounds[2]

    hold_guess = float(np.clip(max(geometry.s1_m * 0.25, 50.0), hold_min, hold_max))
    hold_high = float(np.clip(hold_guess * 1.8, hold_min, hold_max))
    kop_mid = float(np.clip(geometry.z1_m * 0.45, kop_min, kop_max))
    kop_high = float(np.clip(geometry.z1_m * 0.70, kop_min, kop_max))
    inc_low = float(np.clip(geometry.inc_entry_deg * 0.35, inc_min, inc_max))
    inc_mid = float(np.clip(geometry.inc_entry_deg * 0.55, inc_min, inc_max))
    inc_high = float(np.clip(geometry.inc_entry_deg * 0.75, inc_min, inc_max))
    if zero_azimuth_turn:
        seed_vectors = [
            np.array([kop_min, inc_mid, hold_min], dtype=float),
            np.array([kop_min, inc_mid, hold_guess], dtype=float),
            np.array([kop_mid, inc_mid, hold_guess], dtype=float),
            np.array([kop_mid, inc_low, hold_high], dtype=float),
            np.array([kop_high, inc_high, hold_min], dtype=float),
        ]
        seed_vectors.extend(
            np.array(
                [
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
            np.array([kop_min, inc_mid, hold_min, az_surface], dtype=float),
            np.array([kop_min, inc_mid, hold_guess, az_surface], dtype=float),
            np.array([kop_min, inc_high, hold_guess, az_entry], dtype=float),
            np.array([kop_mid, inc_mid, hold_guess, az_mid], dtype=float),
            np.array([kop_mid, inc_low, hold_high, az_surface], dtype=float),
            np.array([kop_high, inc_high, hold_min, az_entry], dtype=float),
        ]
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
            np.array([kop, inc_mid, hold_guess], dtype=float)
            for kop in kop_values
        )
        seed_vectors.extend(
            np.array([kop_mid, inc, hold_guess], dtype=float)
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([kop_mid, inc_mid, hold], dtype=float)
            for hold in hold_values
        )
        seed_vectors.extend(
            np.array([kop, inc, hold_guess], dtype=float)
            for kop in kop_values
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([kop_mid, inc, hold], dtype=float)
            for inc in inc_values
            for hold in hold_values
        )
    else:
        azimuth_values = _turn_azimuth_samples(az_surface, az_entry, lattice_points)
        seed_vectors.extend(
            np.array([kop, inc_mid, hold_guess, az_mid], dtype=float)
            for kop in kop_values
        )
        seed_vectors.extend(
            np.array([kop_mid, inc, hold_guess, az_mid], dtype=float)
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([kop_mid, inc_mid, hold, az_mid], dtype=float)
            for hold in hold_values
        )
        seed_vectors.extend(
            np.array([kop_mid, inc_mid, hold_guess, azimuth_deg], dtype=float)
            for azimuth_deg in azimuth_values
        )
        seed_vectors.extend(
            np.array([kop, inc, hold_guess, az_mid], dtype=float)
            for kop in kop_values
            for inc in inc_values
        )
        seed_vectors.extend(
            np.array([kop_mid, inc_mid, hold, azimuth_deg], dtype=float)
            for hold in hold_values
            for azimuth_deg in azimuth_values
        )
    return _dedupe_seed_vectors(seed_vectors)


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
        return np.array([_mid_azimuth_deg(azimuth_from_deg, azimuth_to_deg)], dtype=float)
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
        return np.full(4, 1e6, dtype=float)

    endpoint = np.array(_estimate_t1_endpoint_for_profile(candidate), dtype=float)
    pos_scale = max(float(config.pos_tolerance_m), 1e-9)
    md_scale = max(float(config.max_total_md_m), 1.0)
    md_excess = max(0.0, float(candidate.md_total_m - config.max_total_md_m))
    residuals = np.array(
        [
            (endpoint[0] - target_point[0]) / pos_scale,
            (endpoint[1] - target_point[1]) / pos_scale,
            (endpoint[2] - target_point[2]) / pos_scale,
            md_excess / md_scale,
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
    miss = float(_distance_3d(*endpoint, *target_point))
    md_excess = max(0.0, float(candidate.md_total_m - config.max_total_md_m))
    return float(miss * 1e6 + md_excess * 1e4 + candidate.md_total_m)


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
    dls_build_deg_per_30m: float,
    kop_vertical_m: float,
    inc_hold_deg: float,
    hold_length_m: float,
    azimuth_hold_deg: float,
    min_build_segment_m: float,
    post_entry: PostEntrySection,
) -> ProfileParameters | None:
    if dls_build_deg_per_30m <= SMALL or kop_vertical_m < 0.0:
        return None
    if hold_length_m < -SMALL:
        return None
    if inc_hold_deg <= SMALL or inc_hold_deg >= geometry.inc_entry_deg - SMALL:
        return None

    min_build = max(float(min_build_segment_m), SMALL)
    radius_m = _radius_from_dls(dls_build_deg_per_30m)
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
    build1_length_m = float(radius_m * build1_dogleg_rad)
    build2_length_m = float(radius_m * build2_dogleg_rad)
    if build1_length_m < min_build - SMALL or build2_length_m < min_build - SMALL:
        return None

    return ProfileParameters(
        kop_vertical_m=float(kop_vertical_m),
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_required_t1_t3_deg=float(geometry.inc_required_t1_t3_deg),
        inc_hold_deg=float(inc_hold_deg),
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
    if inc_entry_rad < -SMALL or max_inc_rad <= SMALL or inc_entry_rad > max_inc_rad + SMALL:
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

    def residual(inc_hold_rad: float) -> float:
        ds_arc = radius_m * (np.cos(inc_entry_rad) - np.cos(inc_hold_rad))
        dz_arc = radius_m * (np.sin(inc_hold_rad) - np.sin(inc_entry_rad))
        ds_rem = ds_m - ds_arc
        dz_rem = dz_m - dz_arc
        return float(ds_rem * np.cos(inc_hold_rad) - dz_rem * np.sin(inc_hold_rad))

    def build_candidate(inc_hold_rad: float) -> PostEntrySection | None:
        ds_arc = radius_m * (np.cos(inc_entry_rad) - np.cos(inc_hold_rad))
        dz_arc = radius_m * (np.sin(inc_hold_rad) - np.sin(inc_entry_rad))
        ds_rem = ds_m - ds_arc
        dz_rem = dz_m - dz_arc
        hold_length_m = float(ds_rem * np.sin(inc_hold_rad) + dz_rem * np.cos(inc_hold_rad))
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
    if _solve_post_entry_section(
        ds_m=geometry.ds_13_m,
        dz_m=geometry.dz_13_m,
        inc_entry_deg=inc_entry_deg,
        dls_deg_per_30m=low,
        max_inc_deg=max_inc_deg,
    ) is not None:
        return low
    if _solve_post_entry_section(
        ds_m=geometry.ds_13_m,
        dz_m=geometry.dz_13_m,
        inc_entry_deg=inc_entry_deg,
        dls_deg_per_30m=high,
        max_inc_deg=max_inc_deg,
    ) is None:
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
            "Post-entry t1->t3 connection is not feasible with HORIZONTAL DLS limit "
            f"{horizontal_dls_deg_per_30m:.2f} deg/30m; requires about {required_dls:.2f} deg/30m. "
            "Increase HORIZONTAL DLS limit or move t3 closer to t1 in section."
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
    if upper_dls + SMALL < required_dls:
        messages.append(
            "BUILD DLS upper bound is insufficient for t1 reach: "
            f"available max {upper_dls:.2f} deg/30m, required about {required_dls:.2f} deg/30m."
        )
    if config.kop_min_vertical_m >= geometry.z1_m - SMALL:
        messages.append(
            "Minimum VERTICAL before KOP is too deep for current t1 TVD. "
            f"kop_min_vertical={config.kop_min_vertical_m:.1f} m, t1 TVD={geometry.z1_m:.1f} m."
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


def _estimate_t1_endpoint_for_profile(profile: ProfileParameters) -> tuple[float, float, float]:
    north_m = 0.0
    east_m = 0.0
    tvd_m = float(profile.kop_vertical_m)

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


def _build_trajectory(params: ProfileParameters) -> WellTrajectory:
    segments = [
        VerticalSegment(
            length_m=params.kop_vertical_m,
            azi_deg=params.azimuth_hold_deg,
            name="VERTICAL",
        )
    ]
    if params.build1_length_m > SMALL:
        segments.append(
            BuildSegment(
                inc_from_deg=0.0,
                inc_to_deg=params.inc_hold_deg,
                dls_deg_per_30m=params.dls_build1_deg_per_30m,
                azi_deg=params.azimuth_hold_deg,
                name="BUILD1",
            )
        )
    if params.hold_length_m > SMALL:
        segments.append(
            HoldSegment(
                length_m=params.hold_length_m,
                inc_deg=params.inc_hold_deg,
                azi_deg=params.azimuth_hold_deg,
                name="HOLD",
            )
        )
    if params.build2_length_m > SMALL:
        segments.append(
            BuildSegment(
                inc_from_deg=params.inc_hold_deg,
                inc_to_deg=params.inc_entry_deg,
                dls_deg_per_30m=params.dls_build2_deg_per_30m,
                azi_deg=params.azimuth_hold_deg,
                azi_to_deg=params.azimuth_entry_deg,
                name="BUILD2",
            )
        )
    if (
        params.horizontal_adjust_length_m > SMALL
        and abs(params.horizontal_inc_deg - params.inc_entry_deg) > 1e-6
    ):
        segments.append(
            BuildSegment(
                inc_from_deg=params.inc_entry_deg,
                inc_to_deg=params.horizontal_inc_deg,
                dls_deg_per_30m=params.horizontal_dls_deg_per_30m,
                azi_deg=params.azimuth_entry_deg,
                name="HORIZONTAL",
            )
        )
    if params.horizontal_hold_length_m > SMALL:
        segments.append(
            HorizontalSegment(
                length_m=params.horizontal_hold_length_m,
                inc_deg=params.horizontal_inc_deg,
                azi_deg=params.azimuth_entry_deg,
                name="HORIZONTAL",
            )
        )
    return WellTrajectory(segments)

def _candidate_turn_deg(candidate: ProfileParameters) -> float:
    return float(
        abs(
            _shortest_azimuth_delta_deg(
                candidate.azimuth_hold_deg,
                candidate.azimuth_entry_deg,
            )
        )
    )


def _build_summary(
    df: pd.DataFrame,
    t1: Point3D,
    t3: Point3D,
    md_t1_m: float,
    params: ProfileParameters,
    horizontal_offset_t1_m: float,
    classification: WellClassification,
    config: TrajectoryConfig,
    turn_search_settings: TurnSearchSettings | None = None,
    turn_restarts_used: int = 0,
) -> dict[str, float | str]:
    t1_idx = int((df["MD_m"] - md_t1_m).abs().idxmin())
    t1_row = df.loc[t1_idx]
    t3_row = df.iloc[-1]

    distance_t1 = _distance_3d(t1_row["X_m"], t1_row["Y_m"], t1_row["Z_m"], t1.x, t1.y, t1.z)
    distance_t3 = _distance_3d(t3_row["X_m"], t3_row["Y_m"], t3_row["Z_m"], t3.x, t3.y, t3.z)

    max_dls = float(np.nanmax(df["DLS_deg_per_30m"].to_numpy()))
    max_inc_actual = float(np.nanmax(df["INC_deg"].to_numpy()))
    md_total_m = float(df["MD_m"].iloc[-1])
    md_postcheck_limit_m = float(config.max_total_md_postcheck_m)
    md_postcheck_excess_m = float(max(0.0, md_total_m - md_postcheck_limit_m))
    md_postcheck_exceeded = bool(md_postcheck_excess_m > 1e-6)

    summary: dict[str, float | str] = {
        "distance_t1_m": float(distance_t1),
        "distance_t3_m": float(distance_t3),
        "kop_vertical_m": float(params.kop_vertical_m),
        "kop_md_m": float(params.kop_vertical_m),
        "entry_inc_deg": float(t1_row["INC_deg"]),
        "entry_inc_target_deg": float(config.entry_inc_target_deg),
        "entry_inc_tolerance_deg": float(config.entry_inc_tolerance_deg),
        "max_inc_deg": float(config.max_inc_deg),
        "max_inc_actual_deg": max_inc_actual,
        "inc_required_t1_t3_deg": float(params.inc_required_t1_t3_deg),
        "horizontal_adjust_length_m": float(params.horizontal_adjust_length_m),
        "horizontal_hold_length_m": float(params.horizontal_hold_length_m),
        "horizontal_inc_deg": float(params.horizontal_inc_deg),
        "hold_inc_deg": float(params.inc_hold_deg),
        "hold_length_m": float(params.hold_length_m),
        "max_dls_total_deg_per_30m": max_dls,
        "md_total_m": md_total_m,
        "max_total_md_postcheck_m": md_postcheck_limit_m,
        "md_postcheck_excess_m": md_postcheck_excess_m,
        "md_postcheck_exceeded": "yes" if md_postcheck_exceeded else "no",
        "t1_horizontal_offset_m": float(horizontal_offset_t1_m),
        "horizontal_length_m": float(params.horizontal_length_m),
        "trajectory_type": TRAJECTORY_MODEL_LABEL,
        "trajectory_target_direction": trajectory_type_label(
            classification.trajectory_type
        ),
        "well_complexity": complexity_label(classification.complexity),
        "well_complexity_by_offset": complexity_label(classification.complexity_by_offset),
        "well_complexity_by_hold": complexity_label(classification.complexity_by_hold),
        "hold_azimuth_deg": float(params.azimuth_hold_deg),
        "entry_azimuth_deg": float(params.azimuth_entry_deg),
        "azimuth_turn_deg": float(
            abs(
                _shortest_azimuth_delta_deg(
                    params.azimuth_hold_deg,
                    params.azimuth_entry_deg,
                )
            )
        ),
        "solver_turn_mode": str(config.turn_solver_mode),
        "solver_turn_max_restarts": int(config.turn_solver_max_restarts),
        "solver_turn_restarts_used": int(turn_restarts_used),
        "solver_turn_attempts_used": (
            int(turn_restarts_used + 1) if turn_search_settings is not None else 0
        ),
        "solver_turn_search_depth_scale": (
            float(turn_search_settings.search_depth_scale)
            if turn_search_settings is not None
            else 1.0
        ),
        "solver_turn_seed_lattice_points": (
            int(turn_search_settings.seed_lattice_points)
            if turn_search_settings is not None
            else 0
        ),
        "solver_turn_local_max_nfev": (
            int(turn_search_settings.local_max_nfev)
            if turn_search_settings is not None
            else 0
        ),
        "solver_strategy": "unified_azimuth_turn_minimize_total_md",
        "class_reverse_offset_min_m": float(classification.limits.reverse_min_m),
        "class_reverse_offset_max_m": float(classification.limits.reverse_max_m),
        "class_offset_ordinary_max_m": float(classification.limits.ordinary_offset_max_m),
        "class_offset_complex_max_m": float(classification.limits.complex_offset_max_m),
        "class_hold_ordinary_max_deg": float(classification.limits.hold_ordinary_max_deg),
        "class_hold_complex_max_deg": float(classification.limits.hold_complex_max_deg),
    }

    for segment, limit in config.dls_limits_deg_per_30m.items():
        seg_max = float(df.loc[df["segment"] == segment, "DLS_deg_per_30m"].max(skipna=True))
        if np.isnan(seg_max):
            seg_max = 0.0
        summary[f"max_dls_{segment.lower()}_deg_per_30m"] = seg_max
        summary[f"dls_limit_{segment.lower()}_deg_per_30m"] = float(limit)
    return summary


def _assert_solution_is_valid(summary: dict[str, float | str], config: TrajectoryConfig) -> None:
    if float(summary["distance_t1_m"]) > config.pos_tolerance_m:
        raise PlanningError(
            "Failed to hit t1 within tolerance. "
            f"Miss={float(summary['distance_t1_m']):.2f} m, tolerance={config.pos_tolerance_m:.2f} m. "
            "Increase BUILD DLS limit, relax tolerance, or adjust target geometry."
        )
    if float(summary["distance_t3_m"]) > config.pos_tolerance_m:
        raise PlanningError(
            "Failed to hit t3 within tolerance. "
            f"Miss={float(summary['distance_t3_m']):.2f} m, tolerance={config.pos_tolerance_m:.2f} m. "
            "Increase HORIZONTAL DLS limit and/or max INC, or move t3 closer/deeper relative to t1."
        )
    if abs(float(summary["entry_inc_deg"]) - config.entry_inc_target_deg) > config.entry_inc_tolerance_deg + 1e-6:
        raise PlanningError("Entry inclination at t1 is outside required target range.")
    if float(summary["max_inc_actual_deg"]) > config.max_inc_deg + 1e-6:
        raise PlanningError(
            "Trajectory exceeds configured max INC. "
            f"max actual INC={float(summary['max_inc_actual_deg']):.2f} deg, "
            f"limit={config.max_inc_deg:.2f} deg."
        )
    for segment, limit in config.dls_limits_deg_per_30m.items():
        actual = float(summary.get(f"max_dls_{segment.lower()}_deg_per_30m", 0.0))
        if actual > limit + 1e-6:
            raise PlanningError(f"DLS limit exceeded on segment {segment}: {actual:.2f} > {limit:.2f}")


def _is_candidate_feasible(candidate: ProfileParameters | None, config: TrajectoryConfig) -> bool:
    if candidate is None:
        return False
    if candidate.md_total_m > config.max_total_md_m + SMALL:
        return False
    horizontal_limit = config.dls_limits_deg_per_30m.get("HORIZONTAL")
    if horizontal_limit is not None and candidate.horizontal_dls_deg_per_30m > float(horizontal_limit) + SMALL:
        return False
    if candidate.horizontal_inc_deg > config.max_inc_deg + SMALL:
        return False
    return True


def _validate_config(config: TrajectoryConfig) -> None:
    if config.md_step_m <= 0.0 or config.md_step_control_m <= 0.0:
        raise PlanningError("MD steps must be positive.")
    if config.pos_tolerance_m <= 0.0:
        raise PlanningError("Position tolerance must be positive.")
    if config.entry_inc_target_deg <= 0.0 or config.entry_inc_target_deg >= 90.0:
        raise PlanningError("entry_inc_target_deg must be in (0, 90).")
    if config.kop_min_vertical_m < 0.0:
        raise PlanningError("kop_min_vertical_m must be non-negative.")
    if config.entry_inc_tolerance_deg < 0.0:
        raise PlanningError("entry_inc_tolerance_deg must be non-negative.")
    if config.max_inc_deg <= 0.0 or config.max_inc_deg > 120.0:
        raise PlanningError("max_inc_deg must be in (0, 120].")
    if config.entry_inc_target_deg > config.max_inc_deg + SMALL:
        raise PlanningError("entry_inc_target_deg cannot exceed max_inc_deg.")
    if config.dls_build_min_deg_per_30m < 0.0:
        raise PlanningError("dls_build_min_deg_per_30m cannot be negative.")
    if config.dls_build_max_deg_per_30m < 0.0:
        raise PlanningError("dls_build_max_deg_per_30m cannot be negative.")
    if config.dls_build_min_deg_per_30m > config.dls_build_max_deg_per_30m:
        raise PlanningError("dls_build_min_deg_per_30m cannot exceed dls_build_max_deg_per_30m.")
    if config.max_total_md_m <= 0.0:
        raise PlanningError("max_total_md_m must be positive.")
    if config.max_total_md_postcheck_m <= 0.0:
        raise PlanningError("max_total_md_postcheck_m must be positive.")
    if config.min_structural_segment_m <= 0.0:
        raise PlanningError("min_structural_segment_m must be positive.")
    if config.min_structural_segment_m < config.md_step_control_m:
        raise PlanningError("min_structural_segment_m must be >= md_step_control_m.")
    if config.turn_solver_mode not in ALLOWED_TURN_SOLVER_MODES:
        allowed_turn = ", ".join(ALLOWED_TURN_SOLVER_MODES)
        raise PlanningError(f"turn_solver_mode must be one of: {allowed_turn}.")
    try:
        turn_solver_max_restarts = float(config.turn_solver_max_restarts)
    except (TypeError, ValueError) as exc:
        raise PlanningError("turn_solver_max_restarts must be a non-negative integer.") from exc
    if int(turn_solver_max_restarts) != turn_solver_max_restarts:
        raise PlanningError("turn_solver_max_restarts must be an integer.")
    if int(turn_solver_max_restarts) < 0:
        raise PlanningError("turn_solver_max_restarts must be non-negative.")
    for segment, limit in config.dls_limits_deg_per_30m.items():
        if limit < 0.0:
            raise PlanningError(f"DLS limit for segment {segment} cannot be negative.")


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

    inc_required_t1_t3_deg = _inclination_from_displacement(ds_13=ds_13_m, dz_13=dz_13_m)
    inc_entry_deg = float(config.entry_inc_target_deg)
    if inc_entry_deg > config.max_inc_deg + SMALL:
        raise PlanningError(
            "Entry INC target exceeds configured max INC. "
            f"entry_inc_target={inc_entry_deg:.2f} deg, max_inc={config.max_inc_deg:.2f} deg. "
            "Reduce entry_inc_target_deg or increase max_inc_deg."
        )
    if inc_required_t1_t3_deg > config.max_inc_deg + SMALL:
        raise PlanningError(
            "With current global max INC the t1->t3 geometry is infeasible without overbend. "
            f"Required straight INC is {inc_required_t1_t3_deg:.2f} deg, "
            f"max INC is {config.max_inc_deg:.2f} deg. "
            "To make target drillable, move t3 deeper and/or closer to t1 in horizontal projection, "
            "or increase max_inc_deg."
        )

    return SectionGeometry(
        s1_m=float(s1_m),
        z1_m=float(z1_m),
        ds_13_m=float(ds_13_m),
        dz_13_m=float(dz_13_m),
        azimuth_entry_deg=float(azimuth_entry_deg),
        azimuth_surface_t1_deg=float(azimuth_surface_t1_deg),
        inc_entry_deg=float(inc_entry_deg),
        inc_required_t1_t3_deg=float(inc_required_t1_t3_deg),
        t1_cross_m=float(c1_m),
        t3_cross_m=float(c3_m),
        t1_east_m=float(t1.x - surface.x),
        t1_north_m=float(t1.y - surface.y),
        t1_tvd_m=float(t1.z - surface.z),
    )


def _is_geometry_coplanar(geometry: SectionGeometry, tolerance_m: float) -> bool:
    return bool(abs(geometry.t1_cross_m) <= tolerance_m and abs(geometry.t3_cross_m) <= tolerance_m)


def _is_zero_azimuth_turn_geometry(
    geometry: SectionGeometry,
    target_direction: str,
    tolerance_m: float,
) -> bool:
    if str(target_direction) == TRAJECTORY_REVERSE_DIRECTION:
        return False
    return _is_geometry_coplanar(geometry=geometry, tolerance_m=tolerance_m)


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
    return _normalize_azimuth_deg(
        azimuth_from_deg
        + 0.5 * _shortest_azimuth_delta_deg(azimuth_from_deg, azimuth_to_deg)
    )


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
    return float(np.degrees(np.arctan2(ds_13, dz_13)))


def _dls_from_radius(radius_m: float) -> float:
    return float(30.0 * RAD2DEG / radius_m)


def _radius_from_dls(dls_deg_per_30m: float) -> float:
    return float(30.0 * RAD2DEG / dls_deg_per_30m)


def _required_dls_for_t1_reach(s1_m: float, z1_m: float, inc_entry_deg: float) -> float:
    inc_entry_rad = inc_entry_deg * DEG2RAD
    one_minus_cos = max(1.0 - np.cos(inc_entry_rad), SMALL)
    sin_entry = max(np.sin(inc_entry_rad), SMALL)
    radius_limit_m = min(s1_m / one_minus_cos, z1_m / sin_entry)
    return _dls_from_radius(radius_limit_m)


def _distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))


def _horizontal_offset(surface: Point3D, point: Point3D) -> float:
    return float(np.hypot(point.x - surface.x, point.y - surface.y))
