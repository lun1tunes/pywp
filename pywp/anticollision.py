from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from pickle import PicklingError
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from pywp.constants import SMALL
from pywp.models import Point3D
from pywp.parallel import process_pool_context
from pywp.uncertainty import (
    DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    PlanningUncertaintyModel,
    UncertaintyEllipseSample,
    UncertaintyStationSample,
    UncertaintyTubeMesh,
    WellUncertaintyOverlay,
    build_uncertainty_overlay,
    build_uncertainty_station_samples,
    local_uncertainty_axes_xyz,
)

TARGET_NONE = ""
TARGET_T1 = "t1"
TARGET_T3 = "t3"
_PAIR_XY_PREFILTER_DIAMETER_FACTOR = 1.5
_PAIR_TERMINAL_PREFILTER_DIAMETER_FACTOR = 1.35
DEFINITIVE_SCAN_STEP_M = 10.0
DEFINITIVE_LOCAL_REFINE_STEP_M = 5.0
DEFINITIVE_LOCAL_REFINE_TRIGGER_SF = 4.0
_MAX_OVERLAP_GEOMETRY_RINGS_PER_CORRIDOR = 8
_MAX_LOCAL_REFINE_SEED_PAIRS_PER_WELL_PAIR = 4
_SCAN_MAX_SAMPLES = 1_000_000
_ISCWSA_DISPLAY_MAX_ELLIPSES_FOR_ANTI_COLLISION = 240


@dataclass(frozen=True)
class AntiCollisionSample:
    md_m: float
    center_xyz: tuple[float, float, float]
    covariance_xyz: np.ndarray
    covariance_xyz_random: np.ndarray
    covariance_xyz_systematic: np.ndarray
    covariance_xyz_global: np.ndarray
    global_source_vectors_xyz: tuple[tuple[str, np.ndarray], ...]
    inc_deg: float = 0.0
    azi_deg: float = 0.0
    target_label: str = TARGET_NONE


@dataclass(frozen=True)
class AntiCollisionWell:
    name: str
    color: str
    overlay: WellUncertaintyOverlay
    samples: tuple[AntiCollisionSample, ...]
    stations: pd.DataFrame
    surface: Point3D
    t1: Point3D | None
    t3: Point3D | None
    md_t1_m: float | None
    md_t3_m: float | None
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = ()
    well_kind: str = "project"
    is_reference_only: bool = False
    normal_support_radii_1sigma_m: np.ndarray | None = None


@dataclass(frozen=True)
class AntiCollisionZone:
    well_a: str
    well_b: str
    classification: str
    priority_rank: int
    label_a: str
    label_b: str
    md_a_m: float
    md_b_m: float
    center_distance_m: float
    combined_radius_m: float
    overlap_depth_m: float
    separation_factor: float
    hotspot_xyz: tuple[float, float, float]
    display_radius_m: float


@dataclass(frozen=True)
class AntiCollisionCorridor:
    well_a: str
    well_b: str
    classification: str
    priority_rank: int
    label_a: str
    label_b: str
    md_a_start_m: float
    md_a_end_m: float
    md_b_start_m: float
    md_b_end_m: float
    md_a_values_m: np.ndarray
    md_b_values_m: np.ndarray
    label_a_values: tuple[str, ...]
    label_b_values: tuple[str, ...]
    midpoint_xyz: np.ndarray
    overlap_rings_xyz: tuple[np.ndarray, ...]
    overlap_core_radius_m: np.ndarray
    separation_factor_values: np.ndarray
    overlap_depth_values_m: np.ndarray


@dataclass(frozen=True)
class AntiCollisionWellSegment:
    well_name: str
    md_start_m: float
    md_end_m: float
    classification: str
    priority_rank: int


@dataclass(frozen=True)
class AntiCollisionReportEvent:
    well_a: str
    well_b: str
    classification: str
    priority_rank: int
    label_a: str
    label_b: str
    md_a_start_m: float
    md_a_end_m: float
    md_b_start_m: float
    md_b_end_m: float
    min_separation_factor: float
    max_overlap_depth_m: float
    min_center_distance_m: float
    merged_corridor_count: int


@dataclass(frozen=True)
class AntiCollisionReportEventGroup:
    event: AntiCollisionReportEvent
    corridors: tuple[AntiCollisionCorridor, ...]


@dataclass(frozen=True)
class AntiCollisionAnalysis:
    wells: tuple[AntiCollisionWell, ...]
    corridors: tuple[AntiCollisionCorridor, ...]
    well_segments: tuple[AntiCollisionWellSegment, ...]
    zones: tuple[AntiCollisionZone, ...]
    pair_count: int
    overlapping_pair_count: int
    target_overlap_pair_count: int
    worst_separation_factor: float | None


@dataclass(frozen=True)
class AntiCollisionPairCacheEntry:
    well_a: str
    well_b: str
    signature_a: str
    signature_b: str
    corridors: tuple[AntiCollisionCorridor, ...]
    zones: tuple[AntiCollisionZone, ...]
    build_overlap_geometry: bool = True


@dataclass(frozen=True)
class AntiCollisionIncrementalStats:
    reused_well_count: int = 0
    rebuilt_well_count: int = 0
    reused_pair_count: int = 0
    recalculated_pair_count: int = 0


@dataclass(frozen=True)
class AntiCollisionProgress:
    pair_count: int
    completed_pair_count: int
    reused_pair_count: int = 0
    recalculated_pair_count: int = 0
    prefiltered_pair_count: int = 0
    elapsed_s: float = 0.0
    parallel_workers: int = 0


@dataclass(frozen=True)
class _AntiCollisionLateralEnvelope:
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    max_lateral_radius_m: float
    surface_x_m: float
    surface_y_m: float
    terminal_x_m: float
    terminal_y_m: float
    terminal_z_m: float
    terminal_lateral_radius_m: float
    terminal_spatial_radius_m: float


@dataclass(frozen=True)
class _AntiCollisionPairJob:
    job_index: int
    left_index: int
    right_index: int
    build_overlap_geometry: bool


@dataclass(frozen=True)
class _AntiCollisionPairCalculation:
    job_index: int
    left_index: int
    right_index: int
    corridors: tuple[AntiCollisionCorridor, ...]
    zones: tuple[AntiCollisionZone, ...]


@dataclass(frozen=True)
class _AntiCollisionPairOutcome:
    job_index: int
    pair_key: tuple[str, str]
    cache_entry: AntiCollisionPairCacheEntry
    reused: bool = False
    recalculated: bool = False
    prefiltered: bool = False


_PARALLEL_ANTI_COLLISION_WELLS: tuple[AntiCollisionWell, ...] = ()


def anti_collision_method_caption(
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> str:
    if model.iscwsa_tool_code:
        model_name = (
            str(model.iscwsa_tool_code).replace("iscwsa_", "").replace("_", " ").upper()
        )
        return (
            f"Anti-collision scan выполнен по 2σ {model_name} "
            "tool-error covariance. Для каждой пары скважин считается "
            f"отчетная сетка с шагом до {DEFINITIVE_SCAN_STEP_M:.0f} м, "
            f"а зоны с SF≤{DEFINITIVE_LOCAL_REFINE_TRIGGER_SF:.0f} локально уточняются до "
            f"{DEFINITIVE_LOCAL_REFINE_STEP_M:.0f} м. "
            "расстояние между центрами сечений и geometric directional 2σ "
            "overlap radius: random/systematic terms суммируются по двум "
            "стволам, а global terms считаются через относительный вклад "
            "общих error sources между скважинами. "
            "Красный volume/polygon показывает приближенное общее пересечение "
            "двух конусов: в каждом конфликтном сечении строится "
            "polygon-intersection uncertainty-контуров в общей локальной "
            "плоскости. Красные участки стволов показывают MD-интервалы "
            "с overlap. Приоритет в отчете отдается пересечениям в t1/t3."
        )
    return (
        "Anti-collision scan выполнен по planning-level 2σ конусам неопределенности. "
        f"Для каждой пары скважин считается отчетная сетка с шагом до {DEFINITIVE_SCAN_STEP_M:.0f} м, "
        f"а зоны с SF≤{DEFINITIVE_LOCAL_REFINE_TRIGGER_SF:.0f} локально уточняются до "
        f"{DEFINITIVE_LOCAL_REFINE_STEP_M:.0f} м. "
        "расстояние между центрами сечений и "
        "geometric directional 2σ overlap radius как сумма опорных радиусов "
        "двух uncertainty-конусов. Красный volume/polygon показывает "
        "приближенное общее пересечение двух конусов: в каждом конфликтном "
        "сечении строится polygon-intersection uncertainty-контуров в общей "
        "локальной плоскости. Красные участки стволов показывают MD-интервалы "
        "с overlap. Приоритет в отчете отдается пересечениям в t1/t3."
        f" Базовая модель uncertainty: INC/AZI = {float(model.sigma_inc_deg):.2f}°/"
        f"{float(model.sigma_azi_deg):.2f}° (1σ), lateral drift "
        f"{float(model.sigma_lateral_drift_m_per_1000m):.1f} м/1000м (1σ), "
        "взвешенный по латеральной экспозиции sin(INC)."
    )


def build_anti_collision_well(
    *,
    name: str,
    color: str,
    stations: pd.DataFrame,
    surface: Point3D,
    t1: Point3D | None,
    t3: Point3D | None,
    azimuth_deg: float,
    md_t1_m: float | None,
    md_t3_m: float | None = None,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    include_display_geometry: bool = True,
    well_kind: str = "project",
    is_reference_only: bool = False,
    analysis_sample_step_m: float | None = None,
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = (),
) -> AntiCollisionWell:
    md_t3_m = (
        float(md_t3_m)
        if md_t3_m is not None
        else (float(stations["MD_m"].iloc[-1]) if len(stations) else None)
    )
    required_md_values = tuple(
        float(value) for value in (md_t1_m, md_t3_m) if value is not None
    )
    if include_display_geometry:
        overlay_model = _display_geometry_sampling_model(
            model=model,
            sample_step_m=analysis_sample_step_m,
        )
        overlay = build_uncertainty_overlay(
            stations=stations,
            surface=surface,
            azimuth_deg=azimuth_deg,
            model=overlay_model,
            required_md_m=required_md_values,
        )
    else:
        overlay = WellUncertaintyOverlay(samples=(), model=model)
    analysis_model = _analysis_sampling_model(
        model=model,
        sample_step_m=analysis_sample_step_m,
    )
    samples = tuple(
        _build_collision_sample_from_station_sample(
            sample=sample,
            md_t1_m=_optional_float(md_t1_m),
            md_t3_m=_optional_float(md_t3_m),
        )
        for sample in build_uncertainty_station_samples(
            stations=stations,
            model=analysis_model,
            required_md_m=required_md_values,
        )
    )
    return AntiCollisionWell(
        name=str(name),
        color=str(color),
        overlay=overlay,
        samples=samples,
        stations=stations.copy(),
        surface=surface,
        t1=t1,
        t3=t3,
        target_pairs=tuple(target_pairs),
        md_t1_m=None if md_t1_m is None else float(md_t1_m),
        md_t3_m=None if md_t3_m is None else float(md_t3_m),
        well_kind=str(well_kind),
        is_reference_only=bool(is_reference_only),
        normal_support_radii_1sigma_m=_max_normal_support_radii_for_samples(
            samples=samples,
            confidence_scale=1.0,
        ),
    )


def _analysis_sampling_model(
    *,
    model: PlanningUncertaintyModel,
    sample_step_m: float | None,
) -> PlanningUncertaintyModel:
    if sample_step_m is None:
        return model
    step_m = float(sample_step_m)
    if step_m <= 0.0:
        raise ValueError("analysis_sample_step_m must be positive.")
    return replace(
        model,
        sample_step_m=step_m,
        max_display_ellipses=max(int(model.max_display_ellipses), _SCAN_MAX_SAMPLES),
        min_refined_step_m=min(float(model.min_refined_step_m), step_m),
    )


def _display_geometry_sampling_model(
    *,
    model: PlanningUncertaintyModel,
    sample_step_m: float | None,
) -> PlanningUncertaintyModel:
    if sample_step_m is None or not bool(model.iscwsa_tool_code):
        return model
    step_m = float(sample_step_m)
    if step_m <= 0.0:
        raise ValueError("analysis_sample_step_m must be positive.")
    return replace(
        model,
        sample_step_m=step_m,
        max_display_ellipses=max(
            int(model.max_display_ellipses),
            _ISCWSA_DISPLAY_MAX_ELLIPSES_FOR_ANTI_COLLISION,
        ),
        min_refined_step_m=min(float(model.min_refined_step_m), step_m),
    )


def analyze_anti_collision(
    wells: list[AntiCollisionWell] | tuple[AntiCollisionWell, ...],
    *,
    build_overlap_geometry: bool = True,
    pair_filter: Callable[[AntiCollisionWell, AntiCollisionWell], bool] | None = None,
    progress_callback: Callable[[AntiCollisionProgress], None] | None = None,
    parallel_workers: int = 0,
) -> AntiCollisionAnalysis:
    analysis, _, _ = analyze_anti_collision_incremental(
        wells,
        build_overlap_geometry=build_overlap_geometry,
        pair_filter=pair_filter,
        progress_callback=progress_callback,
        parallel_workers=parallel_workers,
    )
    return analysis


def _pair_cache_key(
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
) -> tuple[str, str]:
    return tuple(sorted((str(well_a.name), str(well_b.name))))


def analyze_anti_collision_incremental(
    wells: list[AntiCollisionWell] | tuple[AntiCollisionWell, ...],
    *,
    build_overlap_geometry: bool = True,
    pair_filter: Callable[[AntiCollisionWell, AntiCollisionWell], bool] | None = None,
    well_signature_by_name: Mapping[str, str] | None = None,
    previous_pair_cache: (
        Mapping[tuple[str, str], AntiCollisionPairCacheEntry] | None
    ) = None,
    reused_well_count: int = 0,
    rebuilt_well_count: int = 0,
    progress_callback: Callable[[AntiCollisionProgress], None] | None = None,
    parallel_workers: int = 0,
) -> tuple[
    AntiCollisionAnalysis,
    dict[tuple[str, str], AntiCollisionPairCacheEntry],
    AntiCollisionIncrementalStats,
]:
    ordered_wells = tuple(wells)
    lateral_envelopes = tuple(
        _lateral_envelope_for_prefilter(well) for well in ordered_wells
    )
    previous_pairs = dict(previous_pair_cache or {})
    signatures = {
        str(name): str(value) for name, value in (well_signature_by_name or {}).items()
    }
    pair_jobs = _anti_collision_pair_jobs(
        ordered_wells=ordered_wells,
        build_overlap_geometry=build_overlap_geometry,
        pair_filter=pair_filter,
    )
    pair_count = len(pair_jobs)
    reused_pair_count = 0
    recalculated_pair_count = 0
    prefiltered_pair_count = 0
    completed_pair_count = 0
    started_at = perf_counter()

    def notify_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            AntiCollisionProgress(
                pair_count=int(pair_count),
                completed_pair_count=int(completed_pair_count),
                reused_pair_count=int(reused_pair_count),
                recalculated_pair_count=int(recalculated_pair_count),
                prefiltered_pair_count=int(prefiltered_pair_count),
                elapsed_s=float(perf_counter() - started_at),
                parallel_workers=int(max(parallel_workers, 0)),
            )
        )

    notify_progress()
    outcomes: list[_AntiCollisionPairOutcome] = []
    recalculation_jobs: list[_AntiCollisionPairJob] = []
    for job in pair_jobs:
        well_a = ordered_wells[int(job.left_index)]
        well_b = ordered_wells[int(job.right_index)]
        pair_key = _pair_cache_key(well_a, well_b)
        signature_a = signatures.get(str(well_a.name), "")
        signature_b = signatures.get(str(well_b.name), "")
        previous = previous_pairs.get(pair_key)
        previous_signatures = (
            {
                str(previous.well_a): str(previous.signature_a),
                str(previous.well_b): str(previous.signature_b),
            }
            if previous is not None
            else {}
        )
        current_signatures = {
            str(well_a.name): signature_a,
            str(well_b.name): signature_b,
        }
        previous_has_overlap_geometry = bool(
            getattr(previous, "build_overlap_geometry", True)
        )
        geometry_cache_compatible = previous_has_overlap_geometry or not bool(
            build_overlap_geometry
        )
        if (
            previous is not None
            and previous_signatures == current_signatures
            and geometry_cache_compatible
        ):
            outcomes.append(
                _AntiCollisionPairOutcome(
                    job_index=int(job.job_index),
                    pair_key=pair_key,
                    cache_entry=previous,
                    reused=True,
                )
            )
            reused_pair_count += 1
            completed_pair_count += 1
            notify_progress()
            continue

        recalculated_pair_count += 1
        if _pair_prefilter_xy_far_apart(
            lateral_envelope_a=lateral_envelopes[int(job.left_index)],
            lateral_envelope_b=lateral_envelopes[int(job.right_index)],
        ):
            outcomes.append(
                _AntiCollisionPairOutcome(
                    job_index=int(job.job_index),
                    pair_key=pair_key,
                    cache_entry=AntiCollisionPairCacheEntry(
                        well_a=str(well_a.name),
                        well_b=str(well_b.name),
                        signature_a=signature_a,
                        signature_b=signature_b,
                        corridors=(),
                        zones=(),
                        build_overlap_geometry=bool(build_overlap_geometry),
                    ),
                    recalculated=True,
                    prefiltered=True,
                )
            )
            prefiltered_pair_count += 1
            completed_pair_count += 1
            notify_progress()
            continue
        recalculation_jobs.append(job)

    def notify_recalculated_pair_done() -> None:
        nonlocal completed_pair_count
        completed_pair_count += 1
        notify_progress()

    calculation_results = _calculate_pair_overlap_jobs(
        ordered_wells=ordered_wells,
        jobs=recalculation_jobs,
        parallel_workers=int(parallel_workers),
        progress_callback=notify_recalculated_pair_done,
    )
    for result in calculation_results:
        well_a = ordered_wells[int(result.left_index)]
        well_b = ordered_wells[int(result.right_index)]
        pair_key = _pair_cache_key(well_a, well_b)
        signature_a = signatures.get(str(well_a.name), "")
        signature_b = signatures.get(str(well_b.name), "")
        outcomes.append(
            _AntiCollisionPairOutcome(
                job_index=int(result.job_index),
                pair_key=pair_key,
                cache_entry=AntiCollisionPairCacheEntry(
                    well_a=str(well_a.name),
                    well_b=str(well_b.name),
                    signature_a=signature_a,
                    signature_b=signature_b,
                    corridors=tuple(result.corridors),
                    zones=tuple(result.zones),
                    build_overlap_geometry=bool(build_overlap_geometry),
                ),
                recalculated=True,
            )
        )

    corridors: list[AntiCollisionCorridor] = []
    zones: list[AntiCollisionZone] = []
    pair_cache: dict[tuple[str, str], AntiCollisionPairCacheEntry] = {}
    for outcome in sorted(outcomes, key=lambda item: int(item.job_index)):
        corridors.extend(outcome.cache_entry.corridors)
        zones.extend(outcome.cache_entry.zones)
        pair_cache[outcome.pair_key] = outcome.cache_entry

    zones = sorted(
        zones,
        key=lambda zone: (
            int(zone.priority_rank),
            float(zone.separation_factor),
            -float(zone.overlap_depth_m),
            str(zone.well_a),
            str(zone.well_b),
        ),
    )
    pair_keys = {
        tuple(sorted((corridor.well_a, corridor.well_b))) for corridor in corridors
    }
    target_pair_keys = {
        tuple(sorted((corridor.well_a, corridor.well_b)))
        for corridor in corridors
        if int(corridor.priority_rank) < 2
    }
    worst_sf = (
        None
        if not corridors
        else float(
            min(
                float(np.min(corridor.separation_factor_values))
                for corridor in corridors
            )
        )
    )
    analysis = AntiCollisionAnalysis(
        wells=ordered_wells,
        corridors=tuple(corridors),
        well_segments=tuple(_collect_well_overlap_segments(corridors, ordered_wells)),
        zones=tuple(zones),
        pair_count=int(pair_count),
        overlapping_pair_count=int(len(pair_keys)),
        target_overlap_pair_count=int(len(target_pair_keys)),
        worst_separation_factor=worst_sf,
    )
    return (
        analysis,
        pair_cache,
        AntiCollisionIncrementalStats(
            reused_well_count=int(reused_well_count),
            rebuilt_well_count=int(rebuilt_well_count),
            reused_pair_count=int(reused_pair_count),
            recalculated_pair_count=int(recalculated_pair_count),
        ),
    )


def _anti_collision_pair_jobs(
    *,
    ordered_wells: tuple[AntiCollisionWell, ...],
    build_overlap_geometry: bool,
    pair_filter: Callable[[AntiCollisionWell, AntiCollisionWell], bool] | None,
) -> list[_AntiCollisionPairJob]:
    jobs: list[_AntiCollisionPairJob] = []
    for left_index in range(len(ordered_wells)):
        for right_index in range(left_index + 1, len(ordered_wells)):
            well_a = ordered_wells[left_index]
            well_b = ordered_wells[right_index]
            if not _should_analyze_pair(
                well_a=well_a,
                well_b=well_b,
                pair_filter=pair_filter,
            ):
                continue
            jobs.append(
                _AntiCollisionPairJob(
                    job_index=len(jobs),
                    left_index=int(left_index),
                    right_index=int(right_index),
                    build_overlap_geometry=bool(build_overlap_geometry),
                )
            )
    return jobs


def _calculate_pair_overlap_jobs(
    *,
    ordered_wells: tuple[AntiCollisionWell, ...],
    jobs: list[_AntiCollisionPairJob],
    parallel_workers: int,
    progress_callback: Callable[[], None] | None,
) -> list[_AntiCollisionPairCalculation]:
    if not jobs:
        return []
    def calculate_serial(
        *,
        progress_limit: int | None = None,
    ) -> list[_AntiCollisionPairCalculation]:
        results: list[_AntiCollisionPairCalculation] = []
        progress_count = 0
        for job in jobs:
            results.append(
                _calculate_pair_overlap_job_serial(
                    ordered_wells=ordered_wells,
                    job=job,
                )
            )
            if (
                progress_callback is not None
                and (progress_limit is None or progress_count < progress_limit)
            ):
                progress_callback()
                progress_count += 1
        return results

    workers = int(max(parallel_workers, 0))
    if workers <= 1 or len(jobs) <= 1:
        return calculate_serial()

    workers = min(workers, len(jobs))
    results_by_index: dict[int, _AntiCollisionPairCalculation] = {}
    completed_parallel_count = 0
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=process_pool_context(allow_stdin_fork=True),
            initializer=_initialize_parallel_pair_worker,
            initargs=(ordered_wells,),
        ) as executor:
            futures = {
                executor.submit(_calculate_pair_overlap_job_parallel, job): job
                for job in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                results_by_index[int(result.job_index)] = result
                completed_parallel_count += 1
                if progress_callback is not None:
                    progress_callback()
    except (BrokenProcessPool, PicklingError, OSError, RuntimeError, ValueError):
        return calculate_serial(
            progress_limit=max(len(jobs) - completed_parallel_count, 0)
        )
    return [
        results_by_index[int(job.job_index)]
        for job in sorted(jobs, key=lambda item: int(item.job_index))
    ]

def _initialize_parallel_pair_worker(
    ordered_wells: tuple[AntiCollisionWell, ...],
) -> None:
    global _PARALLEL_ANTI_COLLISION_WELLS
    _PARALLEL_ANTI_COLLISION_WELLS = tuple(ordered_wells)


def _calculate_pair_overlap_job_parallel(
    job: _AntiCollisionPairJob,
) -> _AntiCollisionPairCalculation:
    if not _PARALLEL_ANTI_COLLISION_WELLS:
        raise RuntimeError("Anti-collision worker was not initialized.")
    return _calculate_pair_overlap_job_serial(
        ordered_wells=_PARALLEL_ANTI_COLLISION_WELLS,
        job=job,
    )


def _calculate_pair_overlap_job_serial(
    *,
    ordered_wells: tuple[AntiCollisionWell, ...],
    job: _AntiCollisionPairJob,
) -> _AntiCollisionPairCalculation:
    well_a = ordered_wells[int(job.left_index)]
    well_b = ordered_wells[int(job.right_index)]
    pair_corridors = tuple(
        _pair_overlap_corridors(
            well_a=well_a,
            well_b=well_b,
            build_overlap_geometry=bool(job.build_overlap_geometry),
        )
    )
    pair_zones = tuple(_corridor_summary_zone(corridor) for corridor in pair_corridors)
    return _AntiCollisionPairCalculation(
        job_index=int(job.job_index),
        left_index=int(job.left_index),
        right_index=int(job.right_index),
        corridors=pair_corridors,
        zones=pair_zones,
    )


def _should_analyze_pair(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    pair_filter: Callable[[AntiCollisionWell, AntiCollisionWell], bool] | None = None,
) -> bool:
    if bool(well_a.is_reference_only) and bool(well_b.is_reference_only):
        return False
    if pair_filter is not None and not bool(pair_filter(well_a, well_b)):
        return False
    return True


def _lateral_envelope_for_prefilter(
    well: AntiCollisionWell,
) -> _AntiCollisionLateralEnvelope:
    if {"X_m", "Y_m"}.issubset(well.stations.columns):
        x_values = well.stations["X_m"].to_numpy(dtype=float)
        y_values = well.stations["Y_m"].to_numpy(dtype=float)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]
    else:
        x_values = np.asarray(
            [sample.center_xyz[0] for sample in well.samples], dtype=float
        )
        y_values = np.asarray(
            [sample.center_xyz[1] for sample in well.samples], dtype=float
        )
    if x_values.size == 0 or y_values.size == 0:
        surface_x = float(well.surface.x)
        surface_y = float(well.surface.y)
        x_values = np.asarray([surface_x], dtype=float)
        y_values = np.asarray([surface_y], dtype=float)
    terminal_center_xyz = (
        tuple(float(value) for value in well.samples[-1].center_xyz)
        if well.samples
        else (
            float(x_values[-1]),
            float(y_values[-1]),
            float(
                well.stations["Z_m"].to_numpy(dtype=float)[-1]
                if {"Z_m"}.issubset(well.stations.columns) and len(well.stations)
                else float(well.surface.z)
            ),
        )
    )
    max_lateral_radius_m = max(
        (
            _sample_xy_confidence_radius_m(
                covariance_xyz=sample.covariance_xyz,
                confidence_scale=float(well.overlay.model.confidence_scale),
            )
            for sample in well.samples
        ),
        default=0.0,
    )
    terminal_lateral_radius_m = (
        _sample_xy_confidence_radius_m(
            covariance_xyz=well.samples[-1].covariance_xyz,
            confidence_scale=float(well.overlay.model.confidence_scale),
        )
        if well.samples
        else 0.0
    )
    terminal_spatial_radius_m = (
        _sample_3d_confidence_radius_m(
            covariance_xyz=well.samples[-1].covariance_xyz,
            confidence_scale=float(well.overlay.model.confidence_scale),
        )
        if well.samples
        else 0.0
    )
    return _AntiCollisionLateralEnvelope(
        min_x_m=float(np.min(x_values)),
        max_x_m=float(np.max(x_values)),
        min_y_m=float(np.min(y_values)),
        max_y_m=float(np.max(y_values)),
        max_lateral_radius_m=float(max_lateral_radius_m),
        surface_x_m=float(well.surface.x),
        surface_y_m=float(well.surface.y),
        terminal_x_m=float(terminal_center_xyz[0]),
        terminal_y_m=float(terminal_center_xyz[1]),
        terminal_z_m=float(terminal_center_xyz[2]),
        terminal_lateral_radius_m=float(terminal_lateral_radius_m),
        terminal_spatial_radius_m=float(terminal_spatial_radius_m),
    )


def _sample_xy_confidence_radius_m(
    *,
    covariance_xyz: np.ndarray,
    confidence_scale: float,
) -> float:
    covariance_xy = np.asarray(covariance_xyz, dtype=float)[:2, :2]
    if covariance_xy.shape != (2, 2) or not np.all(np.isfinite(covariance_xy)):
        return 0.0
    eigenvalues = np.linalg.eigvalsh(covariance_xy)
    principal_variance = float(max(np.max(eigenvalues), 0.0))
    return float(max(confidence_scale, 0.0)) * float(np.sqrt(principal_variance))


def _sample_3d_confidence_radius_m(
    *,
    covariance_xyz: np.ndarray,
    confidence_scale: float,
) -> float:
    covariance = np.asarray(covariance_xyz, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return 0.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    principal_variance = float(max(np.max(eigenvalues), 0.0))
    return float(max(confidence_scale, 0.0)) * float(np.sqrt(principal_variance))


def _pair_prefilter_xy_far_apart(
    *,
    lateral_envelope_a: _AntiCollisionLateralEnvelope,
    lateral_envelope_b: _AntiCollisionLateralEnvelope,
) -> bool:
    gap_x_m = max(
        0.0,
        float(lateral_envelope_a.min_x_m) - float(lateral_envelope_b.max_x_m),
        float(lateral_envelope_b.min_x_m) - float(lateral_envelope_a.max_x_m),
    )
    gap_y_m = max(
        0.0,
        float(lateral_envelope_a.min_y_m) - float(lateral_envelope_b.max_y_m),
        float(lateral_envelope_b.min_y_m) - float(lateral_envelope_a.max_y_m),
    )
    xy_gap_m = float(np.hypot(gap_x_m, gap_y_m))
    max_lateral_diameter_m = 2.0 * max(
        float(lateral_envelope_a.max_lateral_radius_m),
        float(lateral_envelope_b.max_lateral_radius_m),
    )
    threshold_m = float(_PAIR_XY_PREFILTER_DIAMETER_FACTOR) * max_lateral_diameter_m
    return bool(xy_gap_m > threshold_m)


def _pair_prefilter_terminal_far_apart(
    *,
    lateral_envelope_a: _AntiCollisionLateralEnvelope,
    lateral_envelope_b: _AntiCollisionLateralEnvelope,
) -> bool:
    surface_xy_distance_m = float(
        np.hypot(
            float(lateral_envelope_a.surface_x_m)
            - float(lateral_envelope_b.surface_x_m),
            float(lateral_envelope_a.surface_y_m)
            - float(lateral_envelope_b.surface_y_m),
        )
    )
    terminal_xy_distance_m = float(
        np.hypot(
            float(lateral_envelope_a.terminal_x_m)
            - float(lateral_envelope_b.terminal_x_m),
            float(lateral_envelope_a.terminal_y_m)
            - float(lateral_envelope_b.terminal_y_m),
        )
    )
    terminal_3d_distance_m = float(
        np.linalg.norm(
            np.asarray(
                [
                    float(lateral_envelope_a.terminal_x_m)
                    - float(lateral_envelope_b.terminal_x_m),
                    float(lateral_envelope_a.terminal_y_m)
                    - float(lateral_envelope_b.terminal_y_m),
                    float(lateral_envelope_a.terminal_z_m)
                    - float(lateral_envelope_b.terminal_z_m),
                ],
                dtype=float,
            )
        )
    )
    terminal_xy_diameter_m = 2.0 * max(
        float(lateral_envelope_a.terminal_lateral_radius_m),
        float(lateral_envelope_b.terminal_lateral_radius_m),
    )
    terminal_3d_diameter_m = 2.0 * max(
        float(lateral_envelope_a.terminal_spatial_radius_m),
        float(lateral_envelope_b.terminal_spatial_radius_m),
    )
    xy_cutoff_m = (
        float(_PAIR_TERMINAL_PREFILTER_DIAMETER_FACTOR) * terminal_xy_diameter_m
    )
    spatial_cutoff_m = (
        float(_PAIR_TERMINAL_PREFILTER_DIAMETER_FACTOR) * terminal_3d_diameter_m
    )
    if surface_xy_distance_m <= xy_cutoff_m:
        return False
    return bool(
        terminal_xy_distance_m > xy_cutoff_m
        and terminal_3d_distance_m > spatial_cutoff_m
    )


def _segment_types_for_interval(
    analysis: AntiCollisionAnalysis,
    well_name: str,
    md_start_m: float,
    md_end_m: float,
) -> str:
    """Extract segment type names (VERTICAL, HOLD, BUILD1, etc.) for given MD interval.

    Returns comma-separated unique segment types that overlap with the interval.
    Uses the stations dataframe from AntiCollisionWell to get actual segment types.
    """
    # Find the well in analysis
    well = None
    for w in analysis.wells:
        if w.name == well_name:
            well = w
            break
    if well is None or well.stations is None or well.stations.empty:
        # Fallback to well_segments if stations not available
        segments: list[str] = []
        for segment in analysis.well_segments:
            if segment.well_name != well_name:
                continue
            if segment.md_end_m < md_start_m or segment.md_start_m > md_end_m:
                continue
            segments.append(segment.classification)
        if not segments:
            return "—"
        seen: set[str] = set()
        unique_segments: list[str] = []
        for seg in segments:
            if seg not in seen:
                seen.add(seg)
                unique_segments.append(seg)
        return ", ".join(unique_segments)

    # Use stations dataframe to get segment types
    df = well.stations
    if "segment" not in df.columns:
        return "—"

    # Filter stations within the MD interval
    mask = (df["MD_m"] >= md_start_m) & (df["MD_m"] <= md_end_m)
    segment_names = df.loc[mask, "segment"].dropna().astype(str).tolist()

    if not segment_names:
        return "—"

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_segments: list[str] = []
    for seg in segment_names:
        seg_upper = seg.upper()
        if seg_upper not in seen:
            seen.add(seg_upper)
            unique_segments.append(seg_upper)
    return ", ".join(unique_segments)


def anti_collision_report_rows(
    analysis: AntiCollisionAnalysis,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in anti_collision_report_events(analysis):
        segment_a = _segment_types_for_interval(
            analysis,
            event.well_a,
            event.md_a_start_m,
            event.md_a_end_m,
        )
        segment_b = _segment_types_for_interval(
            analysis,
            event.well_b,
            event.md_b_start_m,
            event.md_b_end_m,
        )
        rows.append(
            {
                "Приоритет": _priority_label_from_parts(
                    classification=event.classification
                ),
                "Скважина A": event.well_a,
                "Скважина B": event.well_b,
                "Участок A": segment_a,
                "Участок B": segment_b,
                "Интервал A, м": _md_interval_label(
                    float(event.md_a_start_m),
                    float(event.md_a_end_m),
                ),
                "Интервал B, м": _md_interval_label(
                    float(event.md_b_start_m),
                    float(event.md_b_end_m),
                ),
                "SF min": float(event.min_separation_factor),
                "Overlap max, м": float(event.max_overlap_depth_m),
                "Мин. расстояние, м": float(event.min_center_distance_m),
            }
        )
    return rows


def anti_collision_report_events(
    analysis: AntiCollisionAnalysis,
) -> tuple[AntiCollisionReportEvent, ...]:
    return tuple(group.event for group in anti_collision_report_event_groups(analysis))


def anti_collision_report_event_groups(
    analysis: AntiCollisionAnalysis,
) -> tuple[AntiCollisionReportEventGroup, ...]:
    if not analysis.corridors:
        return ()
    merge_tolerance_m = _corridor_merge_tolerance_m(analysis.wells)
    sorted_corridors = sorted(
        analysis.corridors,
        key=lambda corridor: (
            str(corridor.well_a),
            str(corridor.well_b),
            float(corridor.md_a_start_m),
            float(corridor.md_b_start_m),
            int(corridor.priority_rank),
        ),
    )
    groups: list[AntiCollisionReportEventGroup] = []
    current = _corridor_to_report_event(sorted_corridors[0])
    current_corridors = [sorted_corridors[0]]
    for corridor in sorted_corridors[1:]:
        if _report_event_group_can_merge(
            current,
            corridor,
            tolerance_m=merge_tolerance_m,
        ):
            current = _merge_report_event_with_corridor(current, corridor)
            current_corridors.append(corridor)
            continue
        groups.append(
            AntiCollisionReportEventGroup(
                event=current,
                corridors=tuple(current_corridors),
            )
        )
        current = _corridor_to_report_event(corridor)
        current_corridors = [corridor]
    groups.append(
        AntiCollisionReportEventGroup(
            event=current,
            corridors=tuple(current_corridors),
        )
    )
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                int(group.event.priority_rank),
                str(group.event.well_a),
                str(group.event.well_b),
                str(group.event.label_a),
                str(group.event.label_b),
                float(group.event.md_a_start_m),
                float(group.event.md_b_start_m),
                float(group.event.min_separation_factor),
                -float(group.event.max_overlap_depth_m),
            ),
        )
    )


def collision_zone_plan_polygon(
    zone: AntiCollisionZone,
    *,
    point_count: int = 40,
) -> np.ndarray:
    return _circle_polygon_xy(
        center_x=float(zone.hotspot_xyz[0]),
        center_y=float(zone.hotspot_xyz[1]),
        radius_m=float(zone.display_radius_m),
        point_count=int(point_count),
    )


def collision_corridor_plan_polygon(
    corridor: AntiCollisionCorridor,
) -> np.ndarray:
    rings_xyz = [np.asarray(ring, dtype=float) for ring in corridor.overlap_rings_xyz]
    if not rings_xyz:
        return np.empty((0, 2), dtype=float)
    rings_xy = [ring[:, :2] for ring in rings_xyz]
    if len(rings_xy) == 1:
        ring_xy = np.asarray(rings_xy[0], dtype=float)
        if len(ring_xy) == 0:
            return np.empty((0, 2), dtype=float)
        return np.vstack([ring_xy, ring_xy[0]])

    positive_side: list[np.ndarray] = []
    negative_side: list[np.ndarray] = []
    centers_xy = np.asarray(
        [np.mean(np.asarray(ring, dtype=float), axis=0)[:2] for ring in rings_xyz],
        dtype=float,
    )
    for index, ring_xy in enumerate(rings_xy):
        center_xy = np.asarray(centers_xy[index], dtype=float)
        tangent_xy = _centerline_tangent_xy(centers_xy, index)
        tangent_norm = float(np.linalg.norm(tangent_xy))
        if tangent_norm <= SMALL:
            tangent_xy = np.array([1.0, 0.0], dtype=float)
            tangent_norm = 1.0
        tangent_xy = tangent_xy / tangent_norm
        normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
        offsets = (np.asarray(ring_xy, dtype=float) - center_xy[None, :]) @ normal_xy
        positive_side.append(np.asarray(ring_xy[int(np.argmax(offsets))], dtype=float))
        negative_side.append(np.asarray(ring_xy[int(np.argmin(offsets))], dtype=float))
    polygon = np.vstack(
        [
            np.asarray(positive_side, dtype=float),
            np.asarray(negative_side[::-1], dtype=float),
        ]
    )
    return np.vstack([polygon, polygon[0]])


def collision_corridor_tube_mesh(
    corridor: AntiCollisionCorridor,
) -> UncertaintyTubeMesh | None:
    rings = [
        _open_ring(np.asarray(ring, dtype=float)) for ring in corridor.overlap_rings_xyz
    ]
    rings = [
        ring
        for ring in rings
        if ring.ndim == 2
        and ring.shape[0] >= 3
        and ring.shape[1] == 3
        and np.all(np.isfinite(ring))
    ]
    if not rings:
        return None
    if len(rings) == 1:
        ring = np.asarray(rings[0], dtype=float)
        center = np.mean(ring, axis=0)
        radius = float(np.nanmax(np.linalg.norm(ring - center[None, :], axis=1)))
        normal = _ring_plane_normal(ring, center)
        thickness = float(np.clip(max(radius, 1.0) * 0.25, 2.0, 30.0))
        rings = [
            np.asarray(ring - normal[None, :] * thickness * 0.5, dtype=float),
            np.asarray(ring + normal[None, :] * thickness * 0.5, dtype=float),
        ]

    vertices_list: list[np.ndarray] = []
    triangles_i: list[int] = []
    triangles_j: list[int] = []
    triangles_k: list[int] = []
    vertex_count = 0
    ring_starts: list[int] = []
    ring_sizes: list[int] = []

    for ring in rings:
        ring_start_index = int(vertex_count)
        ring_starts.append(ring_start_index)
        ring_sizes.append(int(len(ring)))
        vertices_list.append(ring)
        vertex_count += int(len(ring))

    for ring_index in range(len(rings) - 1):
        current_start = ring_starts[ring_index]
        next_start = ring_starts[ring_index + 1]
        points_per_ring = min(ring_sizes[ring_index], ring_sizes[ring_index + 1])
        for point_index in range(points_per_ring):
            next_point_index = (point_index + 1) % points_per_ring
            a = current_start + point_index
            b = current_start + next_point_index
            c = next_start + point_index
            d = next_start + next_point_index
            triangles_i.extend([a, b])
            triangles_j.extend([c, c])
            triangles_k.extend([b, d])

    for ring_start_index, ring in (
        (ring_starts[0], rings[0]),
        (ring_starts[-1], rings[-1]),
    ):
        center_index = int(vertex_count)
        vertices_list.append(np.mean(np.asarray(ring, dtype=float), axis=0)[None, :])
        vertex_count += 1
        points_per_ring = int(len(ring))
        for point_index in range(points_per_ring):
            next_index = (point_index + 1) % points_per_ring
            triangles_i.append(center_index)
            triangles_j.append(ring_start_index + point_index)
            triangles_k.append(ring_start_index + next_index)

    vertices = np.vstack(vertices_list)
    return UncertaintyTubeMesh(
        vertices_xyz=np.asarray(vertices, dtype=float),
        i=np.asarray(triangles_i, dtype=int),
        j=np.asarray(triangles_j, dtype=int),
        k=np.asarray(triangles_k, dtype=int),
    )


def collision_corridor_point_sphere_mesh(
    corridor: AntiCollisionCorridor,
    *,
    lat_steps: int = 10,
    lon_steps: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rings = [np.asarray(ring, dtype=float) for ring in corridor.overlap_rings_xyz]
    if not rings:
        return (
            np.empty((0, 0), dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty((0, 0), dtype=float),
        )
    ring = np.asarray(rings[0], dtype=float)
    center = np.mean(ring, axis=0)
    radius = float(max(np.max(np.linalg.norm(ring - center[None, :], axis=1)), 1.0))
    lat = np.linspace(0.0, np.pi, int(max(lat_steps, 4)))
    lon = np.linspace(0.0, 2.0 * np.pi, int(max(lon_steps, 8)))
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    x = center[0] + radius * np.sin(lat_grid) * np.cos(lon_grid)
    y = center[1] + radius * np.sin(lat_grid) * np.sin(lon_grid)
    z = center[2] + radius * np.cos(lat_grid)
    return x, y, z


def collision_zone_sphere_mesh(
    zone: AntiCollisionZone,
    *,
    lat_steps: int = 10,
    lon_steps: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = float(max(zone.display_radius_m, 1.0))
    center = np.asarray(zone.hotspot_xyz, dtype=float)
    lat = np.linspace(0.0, np.pi, int(max(lat_steps, 4)))
    lon = np.linspace(0.0, 2.0 * np.pi, int(max(lon_steps, 8)))
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    x = center[0] + radius * np.sin(lat_grid) * np.cos(lon_grid)
    y = center[1] + radius * np.sin(lat_grid) * np.sin(lon_grid)
    z = center[2] + radius * np.cos(lat_grid)
    return x, y, z


def _build_collision_sample(
    *,
    sample: UncertaintyEllipseSample,
    md_t1_m: float | None,
    md_t3_m: float | None,
) -> AntiCollisionSample:
    return AntiCollisionSample(
        md_m=float(sample.md_m),
        center_xyz=tuple(float(value) for value in sample.center_xyz),
        covariance_xyz=np.asarray(sample.covariance_xyz, dtype=float),
        covariance_xyz_random=np.asarray(sample.covariance_xyz_random, dtype=float),
        covariance_xyz_systematic=np.asarray(
            sample.covariance_xyz_systematic, dtype=float
        ),
        covariance_xyz_global=np.asarray(sample.covariance_xyz_global, dtype=float),
        global_source_vectors_xyz=tuple(
            (source_name, np.asarray(source_vector, dtype=float))
            for source_name, source_vector in sample.global_source_vectors_xyz
        ),
        inc_deg=float(getattr(sample, "inc_deg", 0.0)),
        azi_deg=float(getattr(sample, "azi_deg", 0.0)),
        target_label=_target_label(
            md_m=float(sample.md_m),
            md_t1_m=_optional_float(md_t1_m),
            md_t3_m=_optional_float(md_t3_m),
        ),
    )


def _build_collision_sample_from_station_sample(
    *,
    sample: UncertaintyStationSample,
    md_t1_m: float | None,
    md_t3_m: float | None,
) -> AntiCollisionSample:
    return AntiCollisionSample(
        md_m=float(sample.md_m),
        center_xyz=tuple(float(value) for value in sample.center_xyz),
        covariance_xyz=np.asarray(sample.covariance_xyz, dtype=float),
        covariance_xyz_random=np.asarray(sample.covariance_xyz_random, dtype=float),
        covariance_xyz_systematic=np.asarray(
            sample.covariance_xyz_systematic, dtype=float
        ),
        covariance_xyz_global=np.asarray(sample.covariance_xyz_global, dtype=float),
        global_source_vectors_xyz=tuple(
            (source_name, np.asarray(source_vector, dtype=float))
            for source_name, source_vector in sample.global_source_vectors_xyz
        ),
        inc_deg=float(sample.inc_deg),
        azi_deg=float(sample.azi_deg),
        target_label=_target_label(
            md_m=float(sample.md_m),
            md_t1_m=_optional_float(md_t1_m),
            md_t3_m=_optional_float(md_t3_m),
        ),
    )


def _target_label(*, md_m: float, md_t1_m: float | None, md_t3_m: float | None) -> str:
    if md_t1_m is not None and abs(float(md_m) - float(md_t1_m)) <= 1e-6:
        return TARGET_T1
    if md_t3_m is not None and abs(float(md_m) - float(md_t3_m)) <= 1e-6:
        return TARGET_T3
    return TARGET_NONE


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _independent_sample_covariance(sample: AntiCollisionSample) -> np.ndarray:
    return np.asarray(sample.covariance_xyz_random, dtype=float) + np.asarray(
        sample.covariance_xyz_systematic, dtype=float
    )


def _pair_relative_covariance_matrix(
    sample_a: AntiCollisionSample,
    sample_b: AntiCollisionSample,
) -> np.ndarray:
    covariance = _independent_sample_covariance(
        sample_a
    ) + _independent_sample_covariance(sample_b)
    vectors_a = {
        str(source_name): np.asarray(source_vector, dtype=float)
        for source_name, source_vector in sample_a.global_source_vectors_xyz
    }
    vectors_b = {
        str(source_name): np.asarray(source_vector, dtype=float)
        for source_name, source_vector in sample_b.global_source_vectors_xyz
    }
    for source_name in sorted(set(vectors_a).union(vectors_b)):
        vector_a = vectors_a.get(source_name, np.zeros(3, dtype=float))
        vector_b = vectors_b.get(source_name, np.zeros(3, dtype=float))
        relative = vector_a - vector_b
        covariance = covariance + np.outer(relative, relative)
    return 0.5 * (covariance + covariance.T)


def _normal_projection_matrix(sample: AntiCollisionSample) -> np.ndarray:
    tangent = local_uncertainty_axes_xyz(
        inc_deg=float(sample.inc_deg),
        azi_deg=float(sample.azi_deg),
    )[0]
    tangent = np.asarray(tangent, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= SMALL:
        return np.eye(3, dtype=float)
    tangent = tangent / tangent_norm
    return np.eye(3, dtype=float) - np.outer(tangent, tangent)


def _pair_normal_projected_relative_covariance_matrix(
    sample_a: AntiCollisionSample,
    sample_b: AntiCollisionSample,
) -> np.ndarray:
    projection_a = _normal_projection_matrix(sample_a)
    projection_b = _normal_projection_matrix(sample_b)
    covariance_a = _independent_sample_covariance(sample_a)
    covariance_b = _independent_sample_covariance(sample_b)
    covariance = (
        projection_a @ covariance_a @ projection_a.T
        + projection_b @ covariance_b @ projection_b.T
    )
    vectors_a = {
        str(source_name): np.asarray(source_vector, dtype=float)
        for source_name, source_vector in sample_a.global_source_vectors_xyz
    }
    vectors_b = {
        str(source_name): np.asarray(source_vector, dtype=float)
        for source_name, source_vector in sample_b.global_source_vectors_xyz
    }
    for source_name in sorted(set(vectors_a).union(vectors_b)):
        relative = projection_a @ vectors_a.get(source_name, np.zeros(3, dtype=float))
        relative -= projection_b @ vectors_b.get(source_name, np.zeros(3, dtype=float))
        covariance = covariance + np.outer(relative, relative)
    return 0.5 * (covariance + covariance.T)


def _pair_distance_combined_radius_matrices(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    confidence_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    samples_a = well_a.samples
    samples_b = well_b.samples
    centers_a = np.asarray([sample.center_xyz for sample in samples_a], dtype=float)
    centers_b = np.asarray([sample.center_xyz for sample in samples_b], dtype=float)
    delta = centers_a[:, None, :] - centers_b[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    max_radius_a = _normal_support_radii_for_well(
        well=well_a,
        confidence_scale=confidence_scale,
    )
    max_radius_b = _normal_support_radii_for_well(
        well=well_b,
        confidence_scale=confidence_scale,
    )
    conservative_radius = max_radius_a[:, None] + max_radius_b[None, :]
    candidate_mask = distance <= (
        float(DEFINITIVE_LOCAL_REFINE_TRIGGER_SF)
        * np.maximum(conservative_radius, SMALL)
    )
    combined_radius = np.zeros_like(distance, dtype=float)
    candidate_indices = np.argwhere(candidate_mask)
    if candidate_indices.size == 0:
        return distance, combined_radius

    index_a_values = candidate_indices[:, 0].astype(int, copy=False)
    index_b_values = candidate_indices[:, 1].astype(int, copy=False)
    _, candidate_radii = _evaluated_pair_distances_and_radii_by_indices(
        samples_a=samples_a,
        samples_b=samples_b,
        index_a_values=index_a_values,
        index_b_values=index_b_values,
        centers_a=centers_a,
        centers_b=centers_b,
        confidence_scale=confidence_scale,
    )
    combined_radius[index_a_values, index_b_values] = candidate_radii
    return distance, combined_radius


def _normal_support_radii_for_well(
    *,
    well: AntiCollisionWell,
    confidence_scale: float,
) -> np.ndarray:
    cached = well.normal_support_radii_1sigma_m
    if cached is not None:
        cached_array = np.asarray(cached, dtype=float)
        if cached_array.shape == (len(well.samples),) and np.all(
            np.isfinite(cached_array)
        ):
            return float(max(confidence_scale, SMALL)) * cached_array
    return _max_normal_support_radii_for_samples(
        samples=well.samples,
        confidence_scale=confidence_scale,
    )


def _max_normal_support_radii_for_samples(
    *,
    samples: tuple[AntiCollisionSample, ...],
    confidence_scale: float,
) -> np.ndarray:
    radii: list[float] = []
    scale = float(max(confidence_scale, SMALL))
    for sample in samples:
        projection = _normal_projection_matrix(sample)
        covariance = (
            projection @ np.asarray(sample.covariance_xyz, dtype=float) @ projection.T
        )
        eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        radii.append(scale * float(np.sqrt(max(float(np.max(eigenvalues)), 0.0))))
    return np.asarray(radii, dtype=float)


def _evaluated_pair_distances_and_radii_by_indices(
    *,
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    index_a_values: np.ndarray,
    index_b_values: np.ndarray,
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    confidence_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    index_a = np.asarray(index_a_values, dtype=int)
    index_b = np.asarray(index_b_values, dtype=int)
    pair_centers_a = np.asarray(centers_a, dtype=float)[index_a]
    pair_centers_b = np.asarray(centers_b, dtype=float)[index_b]
    delta = pair_centers_a - pair_centers_b
    distance = np.linalg.norm(delta, axis=1)
    direction = np.zeros_like(delta, dtype=float)
    np.divide(delta, distance[:, None], out=direction, where=distance[:, None] > SMALL)
    zero_mask = distance <= SMALL
    if np.any(zero_mask):
        zero_offsets = np.where(zero_mask)[0]
        for offset in zero_offsets.tolist():
            covariance = _pair_normal_projected_relative_covariance_matrix(
                samples_a[int(index_a[int(offset)])],
                samples_b[int(index_b[int(offset)])],
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            direction[int(offset)] = eigenvectors[:, int(np.argmax(eigenvalues))]

    covariances_a = np.asarray(
        [_independent_sample_covariance(sample) for sample in samples_a],
        dtype=float,
    )[index_a]
    covariances_b = np.asarray(
        [_independent_sample_covariance(sample) for sample in samples_b],
        dtype=float,
    )[index_b]
    tangents_a = _sample_tangent_array(samples_a)[index_a]
    tangents_b = _sample_tangent_array(samples_b)[index_b]
    direction_a = _normal_plane_projected_direction_from_tangents(
        direction=direction,
        tangents=tangents_a,
    )
    direction_b = _normal_plane_projected_direction_from_tangents(
        direction=direction,
        tangents=tangents_b,
    )
    directional_sigma2_a = np.einsum(
        "pi,pij,pj->p", direction_a, covariances_a, direction_a
    )
    directional_sigma2_b = np.einsum(
        "pi,pij,pj->p", direction_b, covariances_b, direction_b
    )
    scale = float(max(confidence_scale, SMALL))
    radius_global = scale * np.sqrt(
        np.clip(
            _directional_global_relative_sigma2_for_index_arrays(
                samples_a=samples_a,
                samples_b=samples_b,
                index_a=index_a,
                index_b=index_b,
                direction_a=direction_a,
                direction_b=direction_b,
            ),
            0.0,
            None,
        )
    )
    combined_radius = (
        scale * np.sqrt(np.clip(directional_sigma2_a, 0.0, None))
        + scale * np.sqrt(np.clip(directional_sigma2_b, 0.0, None))
        + radius_global
    )
    return distance, combined_radius


def _sample_tangent_array(samples: tuple[AntiCollisionSample, ...]) -> np.ndarray:
    return np.asarray(
        [
            local_uncertainty_axes_xyz(
                inc_deg=float(sample.inc_deg),
                azi_deg=float(sample.azi_deg),
            )[0]
            for sample in samples
        ],
        dtype=float,
    )


def _normal_plane_projected_direction_from_tangents(
    *,
    direction: np.ndarray,
    tangents: np.ndarray,
) -> np.ndarray:
    tangent_component = np.sum(
        np.asarray(direction, dtype=float) * np.asarray(tangents, dtype=float),
        axis=1,
        keepdims=True,
    )
    return np.asarray(direction, dtype=float) - tangent_component * np.asarray(
        tangents, dtype=float
    )


def _directional_global_relative_sigma2_for_index_arrays(
    *,
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    index_a: np.ndarray,
    index_b: np.ndarray,
    direction_a: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray:
    source_names = sorted(
        {
            str(source_name)
            for sample in (*samples_a, *samples_b)
            for source_name, _ in sample.global_source_vectors_xyz
        }
    )
    sigma2 = np.zeros(len(index_a), dtype=float)
    for source_name in source_names:
        vectors_a = _source_vector_array(samples_a, source_name)[index_a]
        vectors_b = _source_vector_array(samples_b, source_name)[index_b]
        projected = np.einsum("pi,pi->p", direction_a, vectors_a) - np.einsum(
            "pi,pi->p", direction_b, vectors_b
        )
        sigma2 += projected * projected
    return sigma2


def _source_vector_array(
    samples: tuple[AntiCollisionSample, ...],
    source_name: str,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for sample in samples:
        sample_vectors = {
            str(name): np.asarray(vector, dtype=float)
            for name, vector in sample.global_source_vectors_xyz
        }
        vectors.append(sample_vectors.get(source_name, np.zeros(3, dtype=float)))
    return np.asarray(vectors, dtype=float)


def _score_matrix(distance: np.ndarray, combined_radius: np.ndarray) -> np.ndarray:
    return np.divide(
        distance,
        np.maximum(combined_radius, SMALL),
        out=np.full_like(distance, np.inf, dtype=float),
        where=combined_radius > SMALL,
    )


def _local_refine_candidate_pairs(score: np.ndarray) -> set[tuple[int, int]]:
    if score.size == 0 or not np.any(np.isfinite(score)):
        return set()
    trigger = float(DEFINITIVE_LOCAL_REFINE_TRIGGER_SF)
    row_best_j = np.argmin(score, axis=1)
    col_best_i = np.argmin(score, axis=0)
    pairs: set[tuple[int, int]] = set()
    for index_a, index_b in enumerate(row_best_j.tolist()):
        if float(score[int(index_a), int(index_b)]) <= trigger:
            pairs.add((int(index_a), int(index_b)))
    for index_b, index_a in enumerate(col_best_i.tolist()):
        if float(score[int(index_a), int(index_b)]) <= trigger:
            pairs.add((int(index_a), int(index_b)))
    return _representative_refine_pairs_by_cluster(
        pairs=pairs,
        score=score,
        max_pairs=int(_MAX_LOCAL_REFINE_SEED_PAIRS_PER_WELL_PAIR),
    )


def _representative_refine_pairs_by_cluster(
    *,
    pairs: set[tuple[int, int]],
    score: np.ndarray,
    max_pairs: int,
) -> set[tuple[int, int]]:
    if not pairs:
        return set()
    sorted_pairs = sorted(pairs)
    clusters: list[list[tuple[int, int]]] = []
    current_cluster: list[tuple[int, int]] = []
    previous_pair: tuple[int, int] | None = None
    for pair in sorted_pairs:
        if previous_pair is None or (
            abs(int(pair[0]) - int(previous_pair[0])) <= 2
            and abs(int(pair[1]) - int(previous_pair[1])) <= 2
        ):
            current_cluster.append(pair)
        else:
            clusters.append(current_cluster)
            current_cluster = [pair]
        previous_pair = pair
    if current_cluster:
        clusters.append(current_cluster)

    representatives = [
        min(cluster, key=lambda pair: _refine_pair_sort_key(pair, score=score))
        for cluster in clusters
    ]
    if len(representatives) <= max_pairs:
        return set(representatives)
    ranked_pairs = sorted(
        representatives,
        key=lambda pair: _refine_pair_sort_key(pair, score=score),
    )
    return set(ranked_pairs[:max_pairs])


def _refine_pair_sort_key(
    pair: tuple[int, int],
    *,
    score: np.ndarray,
) -> tuple[float, int, int]:
    return (
        float(score[int(pair[0]), int(pair[1])]),
        int(pair[0]),
        int(pair[1]),
    )


def _pair_overlap_corridors(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    build_overlap_geometry: bool,
) -> list[AntiCollisionCorridor]:
    if not well_a.samples or not well_b.samples:
        return []

    confidence_scale = float(max(well_a.overlay.model.confidence_scale, SMALL))
    distance, combined_radius = _pair_distance_combined_radius_matrices(
        well_a=well_a,
        well_b=well_b,
        confidence_scale=confidence_scale,
    )
    score = _score_matrix(distance, combined_radius)
    refine_pairs = _local_refine_candidate_pairs(score)
    overlap_mask = (combined_radius > SMALL) & (distance <= combined_radius)
    if not refine_pairs and not np.any(overlap_mask):
        return []

    row_best_j = np.argmin(score, axis=1)
    col_best_i = np.argmin(score, axis=0)

    matched_pairs: set[tuple[int, int]] = set()
    for index_a, index_b in enumerate(row_best_j.tolist()):
        if overlap_mask[int(index_a), int(index_b)]:
            matched_pairs.add((int(index_a), int(index_b)))
    for index_b, index_a in enumerate(col_best_i.tolist()):
        if overlap_mask[int(index_a), int(index_b)]:
            matched_pairs.add((int(index_a), int(index_b)))

    overlap_indices = np.argwhere(overlap_mask)
    for index_a, index_b in overlap_indices.tolist():
        sample_a = well_a.samples[int(index_a)]
        sample_b = well_b.samples[int(index_b)]
        if sample_a.target_label or sample_b.target_label:
            matched_pairs.add((int(index_a), int(index_b)))

    matched_pairs = _geometry_confirmed_pairs(
        pairs=matched_pairs,
        samples_a=well_a.samples,
        samples_b=well_b.samples,
        model_a=well_a.overlay.model,
        model_b=well_b.overlay.model,
    )

    refined_corridors: list[AntiCollisionCorridor] = []
    if refine_pairs:
        refined_corridors = _build_local_refined_pair_corridors(
            well_a=well_a,
            well_b=well_b,
            coarse_pairs=sorted(refine_pairs),
            confidence_scale=confidence_scale,
            build_overlap_geometry=build_overlap_geometry,
        )
        if refined_corridors and not matched_pairs:
            return refined_corridors
    if not matched_pairs:
        return []
    if refined_corridors:
        matched_pairs = _pairs_outside_refined_corridors(
            pairs=matched_pairs,
            samples_a=well_a.samples,
            samples_b=well_b.samples,
            refined_corridors=refined_corridors,
        )
        if not matched_pairs:
            return refined_corridors
    return (
        _build_pair_corridors(
            well_a=well_a,
            well_b=well_b,
            samples_a=well_a.samples,
            samples_b=well_b.samples,
            pairs=sorted(matched_pairs),
            distance=distance,
            combined_radius=combined_radius,
            build_overlap_geometry=build_overlap_geometry,
        )
        + refined_corridors
    )


def _pairs_outside_refined_corridors(
    *,
    pairs: set[tuple[int, int]],
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    refined_corridors: list[AntiCollisionCorridor],
) -> set[tuple[int, int]]:
    if not pairs or not refined_corridors:
        return set(pairs)
    remaining: set[tuple[int, int]] = set()
    tolerance_m = float(DEFINITIVE_LOCAL_REFINE_STEP_M) * 0.5 + 1e-6
    for index_a, index_b in pairs:
        md_a = float(samples_a[int(index_a)].md_m)
        md_b = float(samples_b[int(index_b)].md_m)
        if any(
            float(corridor.md_a_start_m) - tolerance_m
            <= md_a
            <= float(corridor.md_a_end_m) + tolerance_m
            and float(corridor.md_b_start_m) - tolerance_m
            <= md_b
            <= float(corridor.md_b_end_m) + tolerance_m
            for corridor in refined_corridors
        ):
            continue
        remaining.add((int(index_a), int(index_b)))
    return remaining


def _build_local_refined_pair_corridors(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    coarse_pairs: list[tuple[int, int]],
    confidence_scale: float,
    build_overlap_geometry: bool,
) -> list[AntiCollisionCorridor]:
    if not coarse_pairs:
        return []
    samples_by_md_a: dict[float, AntiCollisionSample] = {}
    samples_by_md_b: dict[float, AntiCollisionSample] = {}
    evaluated_md_pairs: set[tuple[float, float]] = set()

    for index_a, index_b in coarse_pairs:
        local_a = _local_refined_samples_for_index(well_a, int(index_a))
        local_b = _local_refined_samples_for_index(well_b, int(index_b))
        for sample in local_a:
            samples_by_md_a[round(float(sample.md_m), 6)] = sample
        for sample in local_b:
            samples_by_md_b[round(float(sample.md_m), 6)] = sample
        for sample_a in local_a:
            for sample_b in local_b:
                evaluated_md_pairs.add(
                    (round(float(sample_a.md_m), 6), round(float(sample_b.md_m), 6))
                )

    samples_a = tuple(
        sample
        for _, sample in sorted(samples_by_md_a.items(), key=lambda item: item[0])
    )
    samples_b = tuple(
        sample
        for _, sample in sorted(samples_by_md_b.items(), key=lambda item: item[0])
    )
    if not samples_a or not samples_b or not evaluated_md_pairs:
        return []

    index_by_md_a = {
        round(float(sample.md_m), 6): index for index, sample in enumerate(samples_a)
    }
    index_by_md_b = {
        round(float(sample.md_m), 6): index for index, sample in enumerate(samples_b)
    }
    distance = np.full((len(samples_a), len(samples_b)), np.inf, dtype=float)
    combined_radius = np.zeros((len(samples_a), len(samples_b)), dtype=float)
    evaluated_pairs: list[tuple[int, int]] = []
    for md_a, md_b in sorted(evaluated_md_pairs):
        index_a = index_by_md_a.get(md_a)
        index_b = index_by_md_b.get(md_b)
        if index_a is None or index_b is None:
            continue
        evaluated_pairs.append((int(index_a), int(index_b)))
    if not evaluated_pairs:
        return []

    pair_distances, pair_radii = _evaluated_pair_distances_and_radii(
        samples_a=samples_a,
        samples_b=samples_b,
        pairs=evaluated_pairs,
        confidence_scale=confidence_scale,
    )
    for pair_index, (index_a, index_b) in enumerate(evaluated_pairs):
        distance[int(index_a), int(index_b)] = float(pair_distances[pair_index])
        combined_radius[int(index_a), int(index_b)] = float(pair_radii[pair_index])

    overlap_mask = (combined_radius > SMALL) & (distance <= combined_radius)
    if not np.any(overlap_mask):
        return []
    score = _score_matrix(distance, combined_radius)
    row_best_j = np.argmin(score, axis=1)
    col_best_i = np.argmin(score, axis=0)
    matched_pairs: set[tuple[int, int]] = set()
    for index_a, index_b in enumerate(row_best_j.tolist()):
        if overlap_mask[int(index_a), int(index_b)]:
            matched_pairs.add((int(index_a), int(index_b)))
    for index_b, index_a in enumerate(col_best_i.tolist()):
        if overlap_mask[int(index_a), int(index_b)]:
            matched_pairs.add((int(index_a), int(index_b)))
    matched_pairs = _geometry_confirmed_pairs(
        pairs=matched_pairs,
        samples_a=samples_a,
        samples_b=samples_b,
        model_a=well_a.overlay.model,
        model_b=well_b.overlay.model,
    )
    if not matched_pairs:
        return []
    return _build_pair_corridors(
        well_a=well_a,
        well_b=well_b,
        samples_a=samples_a,
        samples_b=samples_b,
        pairs=sorted(matched_pairs),
        distance=distance,
        combined_radius=combined_radius,
        build_overlap_geometry=build_overlap_geometry,
        interval_half_width_cap_m=0.5 * float(DEFINITIVE_LOCAL_REFINE_STEP_M),
    )


def _geometry_confirmed_pairs(
    *,
    pairs: set[tuple[int, int]],
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    model_a: PlanningUncertaintyModel,
    model_b: PlanningUncertaintyModel,
) -> set[tuple[int, int]]:
    if not pairs:
        return set()

    ring_cache_a: dict[int, np.ndarray] = {}
    ring_cache_b: dict[int, np.ndarray] = {}
    confirmed: set[tuple[int, int]] = set()
    for index_a, index_b in sorted(pairs):
        index_a = int(index_a)
        index_b = int(index_b)
        ring_a = ring_cache_a.get(index_a)
        if ring_a is None:
            ring_a = _sample_uncertainty_ring_xyz(samples_a[index_a], model=model_a)
            ring_cache_a[index_a] = ring_a
        ring_b = ring_cache_b.get(index_b)
        if ring_b is None:
            ring_b = _sample_uncertainty_ring_xyz(samples_b[index_b], model=model_b)
            ring_cache_b[index_b] = ring_b
        overlap_ring = _overlap_ring_between_samples(
            ring_a_xyz=ring_a,
            ring_b_xyz=ring_b,
            center_xyz=0.5
            * (
                np.asarray(samples_a[index_a].center_xyz, dtype=float)
                + np.asarray(samples_b[index_b].center_xyz, dtype=float)
            ),
        )
        if len(overlap_ring) >= 3:
            confirmed.add((index_a, index_b))
    return confirmed


def _local_refined_samples_for_index(
    well: AntiCollisionWell,
    index: int,
) -> tuple[AntiCollisionSample, ...]:
    samples = well.samples
    if not samples:
        return ()
    safe_index = int(np.clip(index, 0, len(samples) - 1))
    md_start = _sample_interval_start(samples, safe_index)
    md_end = _sample_interval_end(samples, safe_index)
    step_m = float(DEFINITIVE_LOCAL_REFINE_STEP_M)
    md_values = [float(md_start)]
    next_md = float(md_start + step_m)
    while next_md < float(md_end) - 1e-6:
        md_values.append(float(next_md))
        next_md += step_m
    if abs(float(md_values[-1]) - float(md_end)) > 1e-6:
        md_values.append(float(md_end))
    center_md = float(samples[safe_index].md_m)
    if all(abs(center_md - value) > 1e-6 for value in md_values):
        md_values.append(center_md)
    return tuple(
        _collision_sample_at_md(well, md_m=float(md_m))
        for md_m in sorted(set(round(float(value), 6) for value in md_values))
    )


def _collision_sample_at_md(
    well: AntiCollisionWell,
    *,
    md_m: float,
) -> AntiCollisionSample:
    interpolated = _interpolate_collision_sample(well.samples, md_m=float(md_m))
    state = _interpolated_station_state_for_collision(well.stations, md_m=float(md_m))
    if state is None:
        return interpolated
    return AntiCollisionSample(
        md_m=float(interpolated.md_m),
        center_xyz=(
            float(state["x_m"]),
            float(state["y_m"]),
            float(state["z_m"]),
        ),
        covariance_xyz=interpolated.covariance_xyz,
        covariance_xyz_random=interpolated.covariance_xyz_random,
        covariance_xyz_systematic=interpolated.covariance_xyz_systematic,
        covariance_xyz_global=interpolated.covariance_xyz_global,
        global_source_vectors_xyz=interpolated.global_source_vectors_xyz,
        inc_deg=float(state["inc_deg"]),
        azi_deg=float(state["azi_deg"]),
        target_label=_target_label(
            md_m=float(interpolated.md_m),
            md_t1_m=_optional_float(well.md_t1_m),
            md_t3_m=_optional_float(well.md_t3_m),
        ),
    )


def _interpolated_station_state_for_collision(
    stations: pd.DataFrame,
    *,
    md_m: float,
) -> dict[str, float] | None:
    required = {"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"}
    if not required.issubset(stations.columns) or stations.empty:
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    finite = np.isfinite(md_values)
    if not np.any(finite):
        return None
    md_values = md_values[finite]
    order = np.argsort(md_values)
    md_values = md_values[order]
    md = float(np.clip(float(md_m), float(md_values[0]), float(md_values[-1])))

    def column_values(name: str) -> np.ndarray:
        return stations[name].to_numpy(dtype=float)[finite][order]

    azi_rad = np.unwrap(np.deg2rad(column_values("AZI_deg")))
    return {
        "md_m": md,
        "inc_deg": float(np.interp(md, md_values, column_values("INC_deg"))),
        "azi_deg": float(np.rad2deg(np.interp(md, md_values, azi_rad)) % 360.0),
        "x_m": float(np.interp(md, md_values, column_values("X_m"))),
        "y_m": float(np.interp(md, md_values, column_values("Y_m"))),
        "z_m": float(np.interp(md, md_values, column_values("Z_m"))),
    }


def _interpolate_collision_sample(
    samples: tuple[AntiCollisionSample, ...],
    *,
    md_m: float,
) -> AntiCollisionSample:
    md = round(float(md_m), 6)
    md_values = np.asarray([sample.md_m for sample in samples], dtype=float)
    exact_matches = np.where(np.isclose(md_values, md, atol=1e-6))[0]
    if exact_matches.size:
        return samples[int(exact_matches[0])]
    right = int(np.searchsorted(md_values, md, side="right"))
    if right <= 0:
        return samples[0]
    if right >= len(samples):
        return samples[-1]
    left_sample = samples[right - 1]
    right_sample = samples[right]
    left_md = float(left_sample.md_m)
    right_md = float(right_sample.md_m)
    fraction = float((md - left_md) / max(right_md - left_md, SMALL))

    def lerp_array(left: np.ndarray, right_value: np.ndarray) -> np.ndarray:
        return np.asarray(left, dtype=float) + fraction * (
            np.asarray(right_value, dtype=float) - np.asarray(left, dtype=float)
        )

    target_label = TARGET_NONE
    for sample in (left_sample, right_sample):
        if sample.target_label and abs(float(sample.md_m) - md) <= 1e-6:
            target_label = str(sample.target_label)
            break

    source_names = sorted(
        {
            str(source_name)
            for sample in (left_sample, right_sample)
            for source_name, _ in sample.global_source_vectors_xyz
        }
    )
    left_vectors = {
        str(source_name): np.asarray(vector, dtype=float)
        for source_name, vector in left_sample.global_source_vectors_xyz
    }
    right_vectors = {
        str(source_name): np.asarray(vector, dtype=float)
        for source_name, vector in right_sample.global_source_vectors_xyz
    }
    return AntiCollisionSample(
        md_m=float(md),
        center_xyz=tuple(
            float(value)
            for value in lerp_array(
                np.asarray(left_sample.center_xyz, dtype=float),
                np.asarray(right_sample.center_xyz, dtype=float),
            )
        ),
        covariance_xyz=lerp_array(
            left_sample.covariance_xyz, right_sample.covariance_xyz
        ),
        covariance_xyz_random=lerp_array(
            left_sample.covariance_xyz_random,
            right_sample.covariance_xyz_random,
        ),
        covariance_xyz_systematic=lerp_array(
            left_sample.covariance_xyz_systematic,
            right_sample.covariance_xyz_systematic,
        ),
        covariance_xyz_global=lerp_array(
            left_sample.covariance_xyz_global,
            right_sample.covariance_xyz_global,
        ),
        global_source_vectors_xyz=tuple(
            (
                source_name,
                lerp_array(
                    left_vectors.get(source_name, np.zeros(3, dtype=float)),
                    right_vectors.get(source_name, np.zeros(3, dtype=float)),
                ),
            )
            for source_name in source_names
        ),
        inc_deg=float(
            left_sample.inc_deg
            + fraction * (right_sample.inc_deg - left_sample.inc_deg)
        ),
        azi_deg=_interpolated_azimuth_deg(
            left_sample.azi_deg,
            right_sample.azi_deg,
            fraction,
        ),
        target_label=target_label,
    )


def _interpolated_azimuth_deg(
    left_deg: float, right_deg: float, fraction: float
) -> float:
    left = float(left_deg)
    delta = ((float(right_deg) - left + 180.0) % 360.0) - 180.0
    return float((left + float(fraction) * delta) % 360.0)


def _evaluated_pair_distances_and_radii(
    *,
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    pairs: list[tuple[int, int]],
    confidence_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    pair_samples_a = tuple(samples_a[int(index_a)] for index_a, _ in pairs)
    pair_samples_b = tuple(samples_b[int(index_b)] for _, index_b in pairs)
    centers_a = np.asarray(
        [sample.center_xyz for sample in pair_samples_a], dtype=float
    )
    centers_b = np.asarray(
        [sample.center_xyz for sample in pair_samples_b], dtype=float
    )
    delta = centers_a - centers_b
    distance = np.linalg.norm(delta, axis=1)
    direction = np.zeros_like(delta, dtype=float)
    np.divide(delta, distance[:, None], out=direction, where=distance[:, None] > SMALL)
    zero_mask = distance <= SMALL
    if np.any(zero_mask):
        for index in np.where(zero_mask)[0].tolist():
            covariance = _pair_normal_projected_relative_covariance_matrix(
                pair_samples_a[int(index)],
                pair_samples_b[int(index)],
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            direction[int(index)] = eigenvectors[:, int(np.argmax(eigenvalues))]
    covariances_a = np.asarray(
        [_independent_sample_covariance(sample) for sample in pair_samples_a],
        dtype=float,
    )
    covariances_b = np.asarray(
        [_independent_sample_covariance(sample) for sample in pair_samples_b],
        dtype=float,
    )
    direction_a = _normal_plane_projected_direction_for_pairs(
        samples=pair_samples_a,
        direction=direction,
    )
    direction_b = _normal_plane_projected_direction_for_pairs(
        samples=pair_samples_b,
        direction=direction,
    )
    directional_sigma2_a = np.einsum(
        "pi,pij,pj->p", direction_a, covariances_a, direction_a
    )
    directional_sigma2_b = np.einsum(
        "pi,pij,pj->p", direction_b, covariances_b, direction_b
    )
    scale = float(max(confidence_scale, SMALL))
    radius_global = scale * np.sqrt(
        np.clip(
            _directional_global_relative_sigma2_for_pairs(
                samples_a=pair_samples_a,
                samples_b=pair_samples_b,
                direction=direction,
            ),
            0.0,
            None,
        )
    )
    combined_radius = (
        scale * np.sqrt(np.clip(directional_sigma2_a, 0.0, None))
        + scale * np.sqrt(np.clip(directional_sigma2_b, 0.0, None))
        + radius_global
    )
    return distance, combined_radius


def _directional_global_relative_sigma2_for_pairs(
    *,
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    direction: np.ndarray,
) -> np.ndarray:
    direction_a = _normal_plane_projected_direction_for_pairs(
        samples=samples_a,
        direction=direction,
    )
    direction_b = _normal_plane_projected_direction_for_pairs(
        samples=samples_b,
        direction=direction,
    )
    source_names = sorted(
        {
            str(source_name)
            for sample in (*samples_a, *samples_b)
            for source_name, _ in sample.global_source_vectors_xyz
        }
    )
    sigma2 = np.zeros(len(samples_a), dtype=float)
    for source_name in source_names:
        vectors_a = np.asarray(
            [
                dict(sample.global_source_vectors_xyz).get(
                    source_name,
                    np.zeros(3, dtype=float),
                )
                for sample in samples_a
            ],
            dtype=float,
        )
        vectors_b = np.asarray(
            [
                dict(sample.global_source_vectors_xyz).get(
                    source_name,
                    np.zeros(3, dtype=float),
                )
                for sample in samples_b
            ],
            dtype=float,
        )
        projected = np.einsum("pi,pi->p", direction_a, vectors_a) - np.einsum(
            "pi,pi->p", direction_b, vectors_b
        )
        sigma2 += projected * projected
    return sigma2


def _normal_plane_projected_direction_for_pairs(
    *,
    samples: tuple[AntiCollisionSample, ...],
    direction: np.ndarray,
) -> np.ndarray:
    direction_array = np.asarray(direction, dtype=float)
    tangents = np.asarray(
        [
            local_uncertainty_axes_xyz(
                inc_deg=float(sample.inc_deg),
                azi_deg=float(sample.azi_deg),
            )[0]
            for sample in samples
        ],
        dtype=float,
    )
    tangent_component = np.sum(direction_array * tangents, axis=1, keepdims=True)
    return direction_array - tangent_component * tangents


def _build_pair_corridors(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    pairs: list[tuple[int, int]],
    distance: np.ndarray,
    combined_radius: np.ndarray,
    build_overlap_geometry: bool,
    interval_half_width_cap_m: float | None = None,
) -> list[AntiCollisionCorridor]:
    corridors: list[AntiCollisionCorridor] = []
    current_pairs: list[tuple[int, int]] = []

    for index_a, index_b in sorted(
        pairs,
        key=lambda pair: (
            0.5 * (samples_a[pair[0]].md_m + samples_b[pair[1]].md_m),
            pair[0],
            pair[1],
        ),
    ):
        next_key = _pair_class_key(
            label_a=samples_a[int(index_a)].target_label,
            label_b=samples_b[int(index_b)].target_label,
        )
        if not current_pairs:
            current_pairs.append((int(index_a), int(index_b)))
            continue
        prev_a, prev_b = current_pairs[-1]
        previous_key = _pair_class_key(
            label_a=samples_a[int(prev_a)].target_label,
            label_b=samples_b[int(prev_b)].target_label,
        )
        if (
            _pairs_are_contiguous(
                previous_pair=(prev_a, prev_b),
                next_pair=(int(index_a), int(index_b)),
                samples_a=samples_a,
                samples_b=samples_b,
            )
            and previous_key == next_key
        ):
            current_pairs.append((int(index_a), int(index_b)))
            continue
        corridors.append(
            _build_single_corridor(
                well_a=well_a,
                well_b=well_b,
                samples_a=samples_a,
                samples_b=samples_b,
                pairs=current_pairs,
                distance=distance,
                combined_radius=combined_radius,
                build_overlap_geometry=build_overlap_geometry,
                interval_half_width_cap_m=interval_half_width_cap_m,
            )
        )
        current_pairs = [(int(index_a), int(index_b))]

    if current_pairs:
        corridors.append(
            _build_single_corridor(
                well_a=well_a,
                well_b=well_b,
                samples_a=samples_a,
                samples_b=samples_b,
                pairs=current_pairs,
                distance=distance,
                combined_radius=combined_radius,
                build_overlap_geometry=build_overlap_geometry,
                interval_half_width_cap_m=interval_half_width_cap_m,
            )
        )
    return corridors


def _pairs_are_contiguous(
    *,
    previous_pair: tuple[int, int],
    next_pair: tuple[int, int],
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
) -> bool:
    previous_md_a = float(samples_a[int(previous_pair[0])].md_m)
    next_md_a = float(samples_a[int(next_pair[0])].md_m)
    previous_md_b = float(samples_b[int(previous_pair[1])].md_m)
    next_md_b = float(samples_b[int(next_pair[1])].md_m)
    step_a = _local_sample_step_near_pair(samples_a, previous_pair[0], next_pair[0])
    step_b = _local_sample_step_near_pair(samples_b, previous_pair[1], next_pair[1])
    if abs(next_md_a - previous_md_a) > 2.1 * max(step_a, SMALL):
        return False
    if abs(next_md_b - previous_md_b) > 2.1 * max(step_b, SMALL):
        return False
    return (
        abs(next_md_a - previous_md_a) > SMALL or abs(next_md_b - previous_md_b) > SMALL
    )


def _local_sample_step_near_pair(
    samples: tuple[AntiCollisionSample, ...],
    index_a: int,
    index_b: int,
) -> float:
    if len(samples) < 2:
        return 1.0
    left = max(min(int(index_a), int(index_b)) - 1, 0)
    right = min(max(int(index_a), int(index_b)) + 2, len(samples) - 1)
    md_values = np.asarray(
        [sample.md_m for sample in samples[left : right + 1]], dtype=float
    )
    diffs = np.diff(np.sort(md_values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if diffs.size == 0:
        return float(DEFINITIVE_LOCAL_REFINE_STEP_M)
    return float(np.median(diffs))


def _pair_class_key(*, label_a: str, label_b: str) -> tuple[int, str, str]:
    classification, priority_rank = _classify_pair_labels(
        label_a=str(label_a),
        label_b=str(label_b),
    )
    return int(priority_rank), str(label_a), str(label_b)


def _build_single_corridor(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    samples_a: tuple[AntiCollisionSample, ...],
    samples_b: tuple[AntiCollisionSample, ...],
    pairs: list[tuple[int, int]],
    distance: np.ndarray,
    combined_radius: np.ndarray,
    build_overlap_geometry: bool,
    interval_half_width_cap_m: float | None = None,
) -> AntiCollisionCorridor:
    ordered_pairs = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    midpoint_points: list[np.ndarray] = []
    core_radii: list[float] = []
    sf_values: list[float] = []
    overlap_depth_values: list[float] = []
    point_meta: list[tuple[int, str, str]] = []
    md_a_values: list[float] = []
    md_b_values: list[float] = []
    label_a_values: list[str] = []
    label_b_values: list[str] = []
    overlap_rings_xyz: list[np.ndarray] = []
    overlap_ring_inputs: list[
        tuple[AntiCollisionSample, AntiCollisionSample, np.ndarray]
    ] = []

    for index_a, index_b in ordered_pairs:
        sample_a = samples_a[int(index_a)]
        sample_b = samples_b[int(index_b)]
        if bool(well_a.is_reference_only) or bool(well_b.is_reference_only):
            label_a = TARGET_NONE
            label_b = TARGET_NONE
        else:
            label_a = str(sample_a.target_label)
            label_b = str(sample_b.target_label)
        combined_radius_m = float(combined_radius[int(index_a), int(index_b)])
        center_distance_m = float(distance[int(index_a), int(index_b)])
        overlap_depth_m = float(max(combined_radius_m - center_distance_m, 0.0))
        sf = (
            float(center_distance_m / combined_radius_m)
            if combined_radius_m > SMALL
            else float("inf")
        )
        midpoint_points.append(
            0.5
            * (
                np.asarray(sample_a.center_xyz, dtype=float)
                + np.asarray(sample_b.center_xyz, dtype=float)
            )
        )
        core_radii.append(_overlap_core_radius_m(overlap_depth_m=overlap_depth_m))
        if build_overlap_geometry:
            overlap_ring_inputs.append(
                (
                    sample_a,
                    sample_b,
                    np.asarray(midpoint_points[-1], dtype=float),
                )
            )
        sf_values.append(sf)
        overlap_depth_values.append(overlap_depth_m)
        priority_rank = _classify_pair_labels(
            label_a=label_a,
            label_b=label_b,
        )[1]
        point_meta.append((int(priority_rank), label_a, label_b))
        md_a_values.append(float(sample_a.md_m))
        md_b_values.append(float(sample_b.md_m))
        label_a_values.append(label_a)
        label_b_values.append(label_b)

    best_meta = min(point_meta, key=lambda item: item[0])
    best_priority = int(best_meta[0])
    classification = {
        0: "target-target",
        1: "target-trajectory",
    }.get(best_priority, "trajectory")
    if build_overlap_geometry:
        for offset in _selected_overlap_geometry_offsets(
            point_count=len(overlap_ring_inputs),
            separation_factor_values=sf_values,
        ):
            sample_a, sample_b, midpoint_xyz = overlap_ring_inputs[int(offset)]
            overlap_ring = _overlap_ring_between_samples(
                ring_a_xyz=_sample_uncertainty_ring_xyz(
                    sample_a,
                    model=well_a.overlay.model,
                ),
                ring_b_xyz=_sample_uncertainty_ring_xyz(
                    sample_b,
                    model=well_b.overlay.model,
                ),
                center_xyz=midpoint_xyz,
            )
            if len(overlap_ring) >= 3:
                overlap_rings_xyz.append(overlap_ring)
    first_index_a = min(index_a for index_a, _ in ordered_pairs)
    last_index_a = max(index_a for index_a, _ in ordered_pairs)
    first_index_b = min(index_b for _, index_b in ordered_pairs)
    last_index_b = max(index_b for _, index_b in ordered_pairs)
    return AntiCollisionCorridor(
        well_a=well_a.name,
        well_b=well_b.name,
        classification=classification,
        priority_rank=best_priority,
        label_a=str(best_meta[1]),
        label_b=str(best_meta[2]),
        md_a_start_m=_sample_interval_start(
            samples_a,
            first_index_a,
            half_width_cap_m=interval_half_width_cap_m,
        ),
        md_a_end_m=_sample_interval_end(
            samples_a,
            last_index_a,
            half_width_cap_m=interval_half_width_cap_m,
        ),
        md_b_start_m=_sample_interval_start(
            samples_b,
            first_index_b,
            half_width_cap_m=interval_half_width_cap_m,
        ),
        md_b_end_m=_sample_interval_end(
            samples_b,
            last_index_b,
            half_width_cap_m=interval_half_width_cap_m,
        ),
        md_a_values_m=np.asarray(md_a_values, dtype=float),
        md_b_values_m=np.asarray(md_b_values, dtype=float),
        label_a_values=tuple(label_a_values),
        label_b_values=tuple(label_b_values),
        midpoint_xyz=np.asarray(midpoint_points, dtype=float),
        overlap_rings_xyz=tuple(
            np.asarray(ring, dtype=float) for ring in overlap_rings_xyz
        ),
        overlap_core_radius_m=np.asarray(core_radii, dtype=float),
        separation_factor_values=np.asarray(sf_values, dtype=float),
        overlap_depth_values_m=np.asarray(overlap_depth_values, dtype=float),
    )


def _selected_overlap_geometry_offsets(
    *,
    point_count: int,
    separation_factor_values: list[float],
) -> tuple[int, ...]:
    count = int(point_count)
    if count <= 0:
        return ()
    max_count = int(min(count, _MAX_OVERLAP_GEOMETRY_RINGS_PER_CORRIDOR))
    if count <= max_count:
        return tuple(range(count))

    priority_offsets = [0, count - 1]
    if separation_factor_values:
        priority_offsets.append(
            int(np.argmin(np.asarray(separation_factor_values, dtype=float)))
        )
    priority_offsets.extend(
        int(round(value)) for value in np.linspace(0, count - 1, max_count)
    )

    selected: list[int] = []
    for offset in priority_offsets:
        safe_offset = int(np.clip(int(offset), 0, count - 1))
        if safe_offset not in selected:
            selected.append(safe_offset)
        if len(selected) >= max_count:
            break
    return tuple(sorted(selected))


def _sample_interval_start(
    samples: tuple[AntiCollisionSample, ...],
    index: int,
    *,
    half_width_cap_m: float | None = None,
) -> float:
    current_md = float(samples[int(index)].md_m)
    if int(index) == 0:
        return current_md
    previous_md = float(samples[int(index) - 1].md_m)
    start_m = 0.5 * (previous_md + current_md)
    if half_width_cap_m is not None:
        start_m = max(start_m, current_md - float(half_width_cap_m))
    return float(start_m)


def _sample_interval_end(
    samples: tuple[AntiCollisionSample, ...],
    index: int,
    *,
    half_width_cap_m: float | None = None,
) -> float:
    current_md = float(samples[int(index)].md_m)
    if int(index) >= len(samples) - 1:
        return current_md
    next_md = float(samples[int(index) + 1].md_m)
    end_m = 0.5 * (current_md + next_md)
    if half_width_cap_m is not None:
        end_m = min(end_m, current_md + float(half_width_cap_m))
    return float(end_m)


def _overlap_core_radius_m(*, overlap_depth_m: float) -> float:
    return float(max(3.0, 0.5 * float(overlap_depth_m)))


def _sample_uncertainty_ring_xyz(
    sample: AntiCollisionSample,
    *,
    model: PlanningUncertaintyModel,
) -> np.ndarray:
    center = np.asarray(sample.center_xyz, dtype=float)
    tangent, primary_axis, secondary_axis = local_uncertainty_axes_xyz(
        inc_deg=float(sample.inc_deg),
        azi_deg=float(sample.azi_deg),
    )
    if abs(float(sample.inc_deg)) < float(model.near_vertical_isotropic_threshold_deg):
        primary_axis, secondary_axis = _stable_normal_plane_basis(tangent)
    semi_primary_m, semi_secondary_m, primary_axis, secondary_axis = (
        _normal_plane_ellipse_axes_from_covariance(
            covariance_xyz=np.asarray(sample.covariance_xyz, dtype=float),
            tangent=tangent,
            primary_axis=primary_axis,
            secondary_axis=secondary_axis,
            confidence_scale=float(model.confidence_scale),
        )
    )
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        int(max(model.ellipse_points, 12)),
        endpoint=False,
    )
    return (
        center[None, :]
        + np.cos(angles)[:, None] * (semi_primary_m * primary_axis[None, :])
        + np.sin(angles)[:, None] * (semi_secondary_m * secondary_axis[None, :])
    )


def _normal_plane_ellipse_axes_from_covariance(
    *,
    covariance_xyz: np.ndarray,
    tangent: np.ndarray,
    primary_axis: np.ndarray,
    secondary_axis: np.ndarray,
    confidence_scale: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    tangent = np.asarray(tangent, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= SMALL:
        tangent = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        tangent = tangent / tangent_norm
    u = np.asarray(primary_axis, dtype=float)
    u = u - tangent * float(np.dot(u, tangent))
    u_norm = float(np.linalg.norm(u))
    if u_norm <= SMALL:
        u, v = _stable_normal_plane_basis(tangent)
    else:
        u = u / u_norm
        v = np.cross(tangent, u)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= SMALL:
            u, v = _stable_normal_plane_basis(tangent)
        else:
            v = v / v_norm
    covariance = np.asarray(covariance_xyz, dtype=float)
    projected = np.array(
        [
            [float(u @ covariance @ u), float(u @ covariance @ v)],
            [float(v @ covariance @ u), float(v @ covariance @ v)],
        ],
        dtype=float,
    )
    projected = 0.5 * (projected + projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    axis_major = eigenvectors[0, 0] * u + eigenvectors[1, 0] * v
    axis_minor = eigenvectors[0, 1] * u + eigenvectors[1, 1] * v
    axis_major_norm = float(np.linalg.norm(axis_major))
    axis_minor_norm = float(np.linalg.norm(axis_minor))
    if axis_major_norm <= SMALL or axis_minor_norm <= SMALL:
        axis_major, axis_minor = u, v
    else:
        axis_major = axis_major / axis_major_norm
        axis_minor = axis_minor / axis_minor_norm
    scale = float(max(confidence_scale, 0.0))
    return (
        scale * float(np.sqrt(eigenvalues[0])),
        scale * float(np.sqrt(eigenvalues[1])),
        np.asarray(axis_major, dtype=float),
        np.asarray(axis_minor, dtype=float),
    )


def _overlap_ring_between_samples(
    *,
    ring_a_xyz: np.ndarray,
    ring_b_xyz: np.ndarray,
    center_xyz: np.ndarray,
    resample_points: int = 32,
) -> np.ndarray:
    ring_a = _open_ring(np.asarray(ring_a_xyz, dtype=float))
    ring_b = _open_ring(np.asarray(ring_b_xyz, dtype=float))
    if len(ring_a) < 3 or len(ring_b) < 3:
        return np.empty((0, 3), dtype=float)

    basis_u, basis_v = _shared_overlap_plane_basis(ring_a=ring_a, ring_b=ring_b)
    center = np.asarray(center_xyz, dtype=float)
    polygon_a_2d = _project_ring_to_plane(ring_a, center, basis_u, basis_v)
    polygon_b_2d = _project_ring_to_plane(ring_b, center, basis_u, basis_v)
    polygon_a_2d = _ensure_ccw_convex_polygon(polygon_a_2d)
    polygon_b_2d = _ensure_ccw_convex_polygon(polygon_b_2d)
    overlap_polygon_2d = _convex_polygon_intersection(polygon_a_2d, polygon_b_2d)
    if len(overlap_polygon_2d) < 3:
        return np.empty((0, 3), dtype=float)
    resampled_polygon_2d = _resample_closed_polygon_2d(
        overlap_polygon_2d,
        point_count=int(resample_points),
    )
    return (
        center[None, :]
        + resampled_polygon_2d[:, 0:1] * basis_u[None, :]
        + resampled_polygon_2d[:, 1:2] * basis_v[None, :]
    )


def _open_ring(ring_xyz: np.ndarray) -> np.ndarray:
    ring = np.asarray(ring_xyz, dtype=float)
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1], atol=SMALL):
        return ring[:-1]
    return ring


def _shared_overlap_plane_basis(
    *,
    ring_a: np.ndarray,
    ring_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center_a = np.mean(np.asarray(ring_a, dtype=float), axis=0)
    center_b = np.mean(np.asarray(ring_b, dtype=float), axis=0)
    normal_a = _ring_plane_normal(ring_a, center_a)
    normal_b = _ring_plane_normal(ring_b, center_b)
    if float(np.dot(normal_a, normal_b)) < 0.0:
        normal_b = -normal_b
    normal = normal_a + normal_b
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        normal = normal_a if float(np.linalg.norm(normal_a)) > SMALL else normal_b
        normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        normal = center_b - center_a
        normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        normal_norm = 1.0
    normal = normal / normal_norm
    return _stable_normal_plane_basis(normal)


def _ring_plane_normal(ring_xyz: np.ndarray, center_xyz: np.ndarray) -> np.ndarray:
    offsets = (
        np.asarray(ring_xyz, dtype=float) - np.asarray(center_xyz, dtype=float)[None, :]
    )
    if len(offsets) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    _, _, vh = np.linalg.svd(offsets, full_matrices=False)
    if vh.ndim != 2 or vh.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    normal = np.asarray(vh[-1], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / normal_norm


def _project_ring_to_plane(
    ring_xyz: np.ndarray,
    center_xyz: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> np.ndarray:
    offsets = (
        np.asarray(ring_xyz, dtype=float) - np.asarray(center_xyz, dtype=float)[None, :]
    )
    return np.column_stack(
        [
            offsets @ np.asarray(basis_u, dtype=float),
            offsets @ np.asarray(basis_v, dtype=float),
        ]
    )


def _ensure_ccw_convex_polygon(polygon_2d: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_2d, dtype=float)
    if len(polygon) >= 2 and np.allclose(polygon[0], polygon[-1], atol=SMALL):
        polygon = polygon[:-1]
    if len(polygon) < 3:
        return polygon
    centroid = np.mean(polygon, axis=0)
    angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
    ordered = polygon[np.argsort(angles)]
    area2 = np.sum(
        ordered[:, 0] * np.roll(ordered[:, 1], -1)
        - ordered[:, 1] * np.roll(ordered[:, 0], -1)
    )
    if area2 < 0.0:
        ordered = ordered[::-1]
    return ordered


def _convex_polygon_intersection(
    subject_polygon: np.ndarray,
    clip_polygon: np.ndarray,
) -> np.ndarray:
    output = np.asarray(subject_polygon, dtype=float)
    clip = np.asarray(clip_polygon, dtype=float)
    if len(output) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=float)

    def inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
        edge = edge_end - edge_start
        rel = point - edge_start
        return float(edge[0] * rel[1] - edge[1] * rel[0]) >= -SMALL

    def intersection(
        start: np.ndarray,
        end: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
    ) -> np.ndarray:
        line = end - start
        edge = edge_end - edge_start
        denom = float(line[0] * edge[1] - line[1] * edge[0])
        if abs(denom) <= 1e-12:
            return np.asarray(end, dtype=float)
        diff = edge_start - start
        t = float((diff[0] * edge[1] - diff[1] * edge[0]) / denom)
        return start + t * line

    for edge_start, edge_end in zip(clip, np.roll(clip, -1, axis=0)):
        input_list = np.asarray(output, dtype=float)
        if len(input_list) == 0:
            break
        output_points: list[np.ndarray] = []
        prev_point = np.asarray(input_list[-1], dtype=float)
        prev_inside = inside(prev_point, edge_start, edge_end)
        for current_point in input_list:
            current = np.asarray(current_point, dtype=float)
            current_inside = inside(current, edge_start, edge_end)
            if current_inside:
                if not prev_inside:
                    output_points.append(
                        intersection(prev_point, current, edge_start, edge_end)
                    )
                output_points.append(current)
            elif prev_inside:
                output_points.append(
                    intersection(prev_point, current, edge_start, edge_end)
                )
            prev_point = current
            prev_inside = current_inside
        output = np.asarray(output_points, dtype=float)
    if len(output) < 3:
        return np.empty((0, 2), dtype=float)
    return _ensure_ccw_convex_polygon(output)


def _resample_closed_polygon_2d(
    polygon_2d: np.ndarray, *, point_count: int
) -> np.ndarray:
    polygon = _ensure_ccw_convex_polygon(np.asarray(polygon_2d, dtype=float))
    if len(polygon) < 3:
        return polygon
    closed = np.vstack([polygon, polygon[0]])
    segment_vectors = np.diff(closed, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    perimeter = float(np.sum(segment_lengths))
    if perimeter <= SMALL:
        return polygon
    target_distances = np.linspace(
        0.0, perimeter, int(max(point_count, 12)), endpoint=False
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    resampled: list[np.ndarray] = []
    segment_index = 0
    for distance in target_distances:
        while (
            segment_index < len(segment_lengths) - 1
            and cumulative[segment_index + 1] < distance - SMALL
        ):
            segment_index += 1
        segment_start = closed[segment_index]
        segment_end = closed[segment_index + 1]
        segment_length = float(segment_lengths[segment_index])
        if segment_length <= 1e-12:
            resampled.append(np.asarray(segment_start, dtype=float))
            continue
        local_t = float((distance - cumulative[segment_index]) / segment_length)
        resampled.append(segment_start + local_t * (segment_end - segment_start))
    return np.asarray(resampled, dtype=float)


def _corridor_summary_zone(corridor: AntiCollisionCorridor) -> AntiCollisionZone:
    worst_index = _corridor_representative_index(corridor)
    midpoint_xyz = np.asarray(corridor.midpoint_xyz, dtype=float)
    overlap_depth_values = np.asarray(corridor.overlap_depth_values_m, dtype=float)
    radii = np.asarray(corridor.overlap_core_radius_m, dtype=float)
    overlap_depth_m = float(overlap_depth_values[worst_index])
    separation_factor = float(corridor.separation_factor_values[worst_index])
    combined_radius_m = (
        overlap_depth_m / max(1.0 - separation_factor, SMALL)
        if separation_factor < 1.0
        else overlap_depth_m
    )
    center_distance_m = max(combined_radius_m - overlap_depth_m, 0.0)
    return AntiCollisionZone(
        well_a=corridor.well_a,
        well_b=corridor.well_b,
        classification=corridor.classification,
        priority_rank=int(corridor.priority_rank),
        label_a=corridor.label_a,
        label_b=corridor.label_b,
        md_a_m=float(np.asarray(corridor.md_a_values_m, dtype=float)[worst_index]),
        md_b_m=float(np.asarray(corridor.md_b_values_m, dtype=float)[worst_index]),
        center_distance_m=float(center_distance_m),
        combined_radius_m=float(combined_radius_m),
        overlap_depth_m=float(overlap_depth_m),
        separation_factor=float(separation_factor),
        hotspot_xyz=(
            float(midpoint_xyz[worst_index, 0]),
            float(midpoint_xyz[worst_index, 1]),
            float(midpoint_xyz[worst_index, 2]),
        ),
        display_radius_m=float(radii[worst_index]),
    )


def _corridor_to_report_event(
    corridor: AntiCollisionCorridor,
) -> AntiCollisionReportEvent:
    return AntiCollisionReportEvent(
        well_a=str(corridor.well_a),
        well_b=str(corridor.well_b),
        classification=str(corridor.classification),
        priority_rank=int(corridor.priority_rank),
        label_a=str(corridor.label_a),
        label_b=str(corridor.label_b),
        md_a_start_m=float(corridor.md_a_start_m),
        md_a_end_m=float(corridor.md_a_end_m),
        md_b_start_m=float(corridor.md_b_start_m),
        md_b_end_m=float(corridor.md_b_end_m),
        min_separation_factor=float(np.min(corridor.separation_factor_values)),
        max_overlap_depth_m=float(np.max(corridor.overlap_depth_values_m)),
        min_center_distance_m=_corridor_min_center_distance_m(corridor),
        merged_corridor_count=1,
    )


def _report_event_group_can_merge(
    event: AntiCollisionReportEvent,
    corridor: AntiCollisionCorridor,
    *,
    tolerance_m: float,
) -> bool:
    if str(event.well_a) != str(corridor.well_a) or str(event.well_b) != str(
        corridor.well_b
    ):
        return False
    if (
        str(event.classification) == "target-target"
        or str(corridor.classification) == "target-target"
    ):
        if (
            str(event.classification) != str(corridor.classification)
            or str(event.label_a) != str(corridor.label_a)
            or str(event.label_b) != str(corridor.label_b)
        ):
            return False
    if not _target_labels_are_merge_compatible(event, corridor):
        return False
    overlap_or_touch_a = (
        float(corridor.md_a_start_m) <= float(event.md_a_end_m) + tolerance_m
    )
    overlap_or_touch_b = (
        float(corridor.md_b_start_m) <= float(event.md_b_end_m) + tolerance_m
    )
    return bool(overlap_or_touch_a and overlap_or_touch_b)


def _merge_report_event_with_corridor(
    event: AntiCollisionReportEvent,
    corridor: AntiCollisionCorridor,
) -> AntiCollisionReportEvent:
    classification, priority_rank, label_a, label_b = _merged_report_event_meta(
        event,
        corridor,
    )
    return AntiCollisionReportEvent(
        well_a=event.well_a,
        well_b=event.well_b,
        classification=classification,
        priority_rank=int(priority_rank),
        label_a=label_a,
        label_b=label_b,
        md_a_start_m=min(float(event.md_a_start_m), float(corridor.md_a_start_m)),
        md_a_end_m=max(float(event.md_a_end_m), float(corridor.md_a_end_m)),
        md_b_start_m=min(float(event.md_b_start_m), float(corridor.md_b_start_m)),
        md_b_end_m=max(float(event.md_b_end_m), float(corridor.md_b_end_m)),
        min_separation_factor=min(
            float(event.min_separation_factor),
            float(np.min(corridor.separation_factor_values)),
        ),
        max_overlap_depth_m=max(
            float(event.max_overlap_depth_m),
            float(np.max(corridor.overlap_depth_values_m)),
        ),
        min_center_distance_m=min(
            float(event.min_center_distance_m),
            _corridor_min_center_distance_m(corridor),
        ),
        merged_corridor_count=int(event.merged_corridor_count) + 1,
    )


def _corridor_min_center_distance_m(corridor: AntiCollisionCorridor) -> float:
    sf_values = np.asarray(corridor.separation_factor_values, dtype=float)
    overlap_depth_values = np.asarray(corridor.overlap_depth_values_m, dtype=float)
    safe_mask = sf_values < 0.999999
    center_distances = np.full_like(sf_values, np.inf)
    center_distances[safe_mask] = (
        sf_values[safe_mask]
        * overlap_depth_values[safe_mask]
        / (1.0 - sf_values[safe_mask])
    )
    return float(np.min(center_distances))


def _merged_report_event_meta(
    event: AntiCollisionReportEvent,
    corridor: AntiCollisionCorridor,
) -> tuple[str, int, str, str]:
    event_key = (
        int(event.priority_rank),
        float(event.min_separation_factor),
        -float(event.max_overlap_depth_m),
    )
    corridor_key = (
        int(corridor.priority_rank),
        float(np.min(corridor.separation_factor_values)),
        -float(np.max(corridor.overlap_depth_values_m)),
    )
    if corridor_key < event_key:
        return (
            str(corridor.classification),
            int(corridor.priority_rank),
            str(corridor.label_a),
            str(corridor.label_b),
        )
    return (
        str(event.classification),
        int(event.priority_rank),
        str(event.label_a),
        str(event.label_b),
    )


def _target_labels_are_merge_compatible(
    event: AntiCollisionReportEvent,
    corridor: AntiCollisionCorridor,
) -> bool:
    event_labels = _target_label_signature(event.label_a, event.label_b)
    corridor_labels = _target_label_signature(corridor.label_a, corridor.label_b)
    return bool(
        not event_labels or not corridor_labels or event_labels == corridor_labels
    )


def _target_label_signature(label_a: str, label_b: str) -> tuple[str, ...]:
    left = str(label_a)
    right = str(label_b)
    return (left, right) if left or right else ()


def _corridor_merge_tolerance_m(
    wells: tuple[AntiCollisionWell, ...],
) -> float:
    return float(
        max((_well_sample_step_m(well) for well in wells), default=100.0) * 1.05
    )


def _well_sample_step_m(well: AntiCollisionWell) -> float:
    md_values = np.asarray([sample.md_m for sample in well.samples], dtype=float)
    if md_values.size < 2:
        return float(well.overlay.model.sample_step_m)
    diffs = np.diff(np.sort(md_values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if diffs.size == 0:
        return float(well.overlay.model.sample_step_m)
    return float(np.median(diffs))


def _md_interval_label(md_start_m: float, md_end_m: float) -> str:
    start = float(md_start_m)
    end = float(md_end_m)
    if abs(end - start) <= 0.5:
        return f"{start:.0f}"
    return f"{start:.0f} - {end:.0f}"


def _corridor_representative_index(corridor: AntiCollisionCorridor) -> int:
    label_a_values = tuple(str(value) for value in corridor.label_a_values)
    label_b_values = tuple(str(value) for value in corridor.label_b_values)
    separation_values = np.asarray(corridor.separation_factor_values, dtype=float)
    ranked_indices = sorted(
        range(len(separation_values)),
        key=lambda index: (
            _classify_pair_labels(
                label_a=label_a_values[index],
                label_b=label_b_values[index],
            )[1],
            float(separation_values[index]),
            -float(np.asarray(corridor.overlap_depth_values_m, dtype=float)[index]),
        ),
    )
    return int(ranked_indices[0])


def _collect_well_overlap_segments(
    corridors: list[AntiCollisionCorridor],
    wells: tuple[AntiCollisionWell, ...],
) -> list[AntiCollisionWellSegment]:
    if not corridors:
        return []
    step_tolerance_m = float(
        max((_well_sample_step_m(well) for well in wells), default=100.0) * 1.05
    )
    raw_segments: list[AntiCollisionWellSegment] = []
    for corridor in corridors:
        raw_segments.append(
            AntiCollisionWellSegment(
                well_name=corridor.well_a,
                md_start_m=float(corridor.md_a_start_m),
                md_end_m=float(corridor.md_a_end_m),
                classification=corridor.classification,
                priority_rank=int(corridor.priority_rank),
            )
        )
        raw_segments.append(
            AntiCollisionWellSegment(
                well_name=corridor.well_b,
                md_start_m=float(corridor.md_b_start_m),
                md_end_m=float(corridor.md_b_end_m),
                classification=corridor.classification,
                priority_rank=int(corridor.priority_rank),
            )
        )
    merged: list[AntiCollisionWellSegment] = []
    for well_name in sorted({segment.well_name for segment in raw_segments}):
        well_segments = sorted(
            [segment for segment in raw_segments if segment.well_name == well_name],
            key=lambda segment: (float(segment.md_start_m), float(segment.md_end_m)),
        )
        current = well_segments[0]
        for segment in well_segments[1:]:
            if float(segment.md_start_m) <= float(current.md_end_m) + step_tolerance_m:
                current = AntiCollisionWellSegment(
                    well_name=current.well_name,
                    md_start_m=float(current.md_start_m),
                    md_end_m=max(float(current.md_end_m), float(segment.md_end_m)),
                    classification=(
                        current.classification
                        if int(current.priority_rank) <= int(segment.priority_rank)
                        else segment.classification
                    ),
                    priority_rank=min(
                        int(current.priority_rank), int(segment.priority_rank)
                    ),
                )
                continue
            merged.append(current)
            current = segment
        merged.append(current)
    return merged


def _classify_pair_labels(*, label_a: str, label_b: str) -> tuple[str, int]:
    if label_a and label_b:
        return "target-target", 0
    if label_a or label_b:
        return "target-trajectory", 1
    return "trajectory", 2


def _priority_label_from_parts(*, classification: str) -> str:
    if classification == "target-target":
        return "Цели ↔ цели"
    if classification == "target-trajectory":
        return "Цель ↔ траектория"
    return "Траектория ↔ траектория"


def _zone_location_label_from_parts(*, label_a: str, label_b: str) -> str:
    left = str(label_a) or "траектория"
    right = str(label_b) or "траектория"
    return f"{left} ↔ {right}"


def _circle_polygon_xy(
    *,
    center_x: float,
    center_y: float,
    radius_m: float,
    point_count: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(max(point_count, 12)), endpoint=False)
    polygon = np.column_stack(
        [
            float(center_x) + float(radius_m) * np.cos(angles),
            float(center_y) + float(radius_m) * np.sin(angles),
        ]
    )
    return np.vstack([polygon, polygon[0]])


def _centerline_tangent_xy(centers_xy: np.ndarray, index: int) -> np.ndarray:
    current = np.asarray(centers_xy[int(index)], dtype=float)
    if int(index) == 0:
        return np.asarray(centers_xy[1], dtype=float) - current
    if int(index) == len(centers_xy) - 1:
        return current - np.asarray(centers_xy[-2], dtype=float)
    return np.asarray(centers_xy[int(index) + 1], dtype=float) - np.asarray(
        centers_xy[int(index) - 1], dtype=float
    )


def _centerline_tangent_xyz(centers_xyz: np.ndarray, index: int) -> np.ndarray:
    current = np.asarray(centers_xyz[int(index)], dtype=float)
    if int(index) == 0:
        return np.asarray(centers_xyz[1], dtype=float) - current
    if int(index) == len(centers_xyz) - 1:
        return current - np.asarray(centers_xyz[-2], dtype=float)
    return np.asarray(centers_xyz[int(index) + 1], dtype=float) - np.asarray(
        centers_xyz[int(index) - 1], dtype=float
    )


def _stable_normal_plane_basis(
    tangent_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(tangent_xyz, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        tangent = np.array([0.0, 0.0, 1.0], dtype=float)
        tangent_norm = 1.0
    tangent = tangent / tangent_norm
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(tangent, reference))) > 0.98:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    basis_1 = np.cross(reference, tangent)
    basis_1_norm = float(np.linalg.norm(basis_1))
    if basis_1_norm <= 1e-12:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_1 = np.cross(reference, tangent)
        basis_1_norm = float(np.linalg.norm(basis_1))
    basis_1 = basis_1 / basis_1_norm
    basis_2 = np.cross(tangent, basis_1)
    basis_2 = basis_2 / float(np.linalg.norm(basis_2))
    return basis_1, basis_2


def _align_ring_for_continuity(
    *,
    ring_open_xyz: np.ndarray,
    previous_ring_open_xyz: np.ndarray | None,
) -> np.ndarray:
    current = np.asarray(ring_open_xyz, dtype=float)
    if previous_ring_open_xyz is None:
        return current
    previous = np.asarray(previous_ring_open_xyz, dtype=float)
    if current.shape != previous.shape or current.ndim != 2 or current.shape[0] < 3:
        return current

    best_ring = current
    best_cost = float(np.mean(np.linalg.norm(current - previous, axis=1)))
    for candidate_base in (current, current[::-1]):
        for shift in range(current.shape[0]):
            candidate = np.roll(candidate_base, shift=shift, axis=0)
            candidate_cost = float(
                np.mean(np.linalg.norm(candidate - previous, axis=1))
            )
            if candidate_cost + SMALL < best_cost:
                best_ring = candidate
                best_cost = candidate_cost
    return best_ring
