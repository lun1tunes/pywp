from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import hashlib
from pickle import PicklingError
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionIncrementalStats,
    AntiCollisionPairCacheEntry,
    AntiCollisionWell,
    analyze_anti_collision_incremental,
    build_anti_collision_well,
)
from pywp.eclipse_welltrack import (
    WelltrackRecord,
    welltrack_points_to_target_pairs,
)
from pywp.multi_horizontal import extend_plan_with_multi_horizontal_targets
from pywp.models import TrajectoryConfig
from pywp.parallel import process_pool_context
from pywp.planner import TrajectoryPlanner
from pywp.reference_trajectories import (
    ImportedTrajectoryWell,
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_KIND_COLORS,
    reference_well_collision_name,
    reference_well_duplicate_name_keys,
)
from pywp.uncertainty import PlanningUncertaintyModel
from pywp.welltrack_batch import SuccessfulWellPlan

# Zones where both wells are at MD below this threshold are treated as
# "near-surface" and excluded from the scoring metric.  On shared-surface
# pads every well pair has SF≈0 at the surface, blinding the optimizer.
_SURFACE_MD_THRESHOLD_M = 600.0

# Minimum improvement in score to accept a swap (avoids noise).
_IMPROVEMENT_EPS = 0.005

# Maximum number of greedy iterations (each tests all viable swaps for the
# worst actionable collision pair).
_MAX_ITERATIONS = 8

# Maximum number of distinct worst-zone pairs to try per iteration when the
# top-1 worst zone cannot be improved.
_MAX_WORST_ZONES_PER_ITERATION = 3

# Keep pad-order optimization interactive.  A candidate is expensive: it
# recalculates two trajectories and rescoring anti-collision pairs can still
# take several seconds even with cache.  The bounded search below aims for a
# useful local improvement in about two minutes for one pad.
_MAX_CANDIDATE_EVALUATIONS = 24
_MAX_CANDIDATES_PER_ITERATION = 12
_MAX_CANDIDATES_PER_ZONE = 8
_MAX_RUNTIME_S = 120.0


@dataclass(frozen=True)
class _PadOptimizationScore:
    target_zone_count: int
    overlap_pair_count: int
    zone_count: int
    severe_zone_count: int
    worst_sf: float
    mean_sf: float


def _recalculate_well_from_dicts(
    record_dict: dict, config_dict: dict,
) -> dict | None:
    """Worker-safe version that accepts/returns plain dicts.

    Streamlit's script rerunner can reload modules, making the Pydantic
    class objects in the main process differ from those in ``sys.modules``.
    Pickle checks identity and raises ``PicklingError``.  By serialising
    to dicts before crossing the process boundary we avoid this entirely.
    """
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)

    record = WelltrackRecord.model_validate(record_dict)
    config = TrajectoryConfig.model_validate(config_dict)
    result = recalculate_well(record, config)
    if result is None:
        return None
    return result.model_dump()


def recalculate_well(
    record: WelltrackRecord, config: TrajectoryConfig,
) -> SuccessfulWellPlan | None:
    start_t = perf_counter()
    try:
        surface, target_pairs = welltrack_points_to_target_pairs(record.points)
        t1, t3 = target_pairs[0]
    except (ValueError, IndexError):
        return None
    planner = TrajectoryPlanner()
    try:
        result = planner.plan(surface=surface, t1=t1, t3=t3, config=config)
        if len(target_pairs) > 1:
            result = extend_plan_with_multi_horizontal_targets(
                base_result=result,
                target_pairs=target_pairs,
                config=config,
            )
            t3 = target_pairs[-1][1]
        return SuccessfulWellPlan(
            name=record.name,
            surface=surface,
            t1=t1,
            t3=t3,
            target_pairs=target_pairs,
            stations=result.stations,
            summary=result.summary,
            azimuth_deg=result.azimuth_deg,
            md_t1_m=result.md_t1_m,
            config=config,
            runtime_s=perf_counter() - start_t,
        )
    except Exception:
        return None


def _actionable_pad_zones(
    analysis: AntiCollisionAnalysis,
    pad_well_names: set[str],
    fixed_well_names: set[str] | None = None,
) -> list:
    """Return zones affected by movable wells on this pad."""
    fixed_names = {str(name) for name in (fixed_well_names or set())}
    pad_names = {str(name) for name in pad_well_names}
    movable_names = pad_names - fixed_names
    zones: list = []
    for zone in analysis.zones:
        zone_names = {str(zone.well_a), str(zone.well_b)}
        if not zone_names.intersection(pad_names):
            continue
        if not zone_names.intersection(movable_names):
            continue
        if zone_names.issubset(fixed_names):
            continue
        if (
            float(zone.md_a_m) <= _SURFACE_MD_THRESHOLD_M
            and float(zone.md_b_m) <= _SURFACE_MD_THRESHOLD_M
        ):
            continue
        zones.append(zone)
    return zones


def score_analysis(
    analysis: AntiCollisionAnalysis | None,
    pad_well_names: set[str],
    fixed_well_names: set[str] | None = None,
) -> float:
    """Evaluate collision severity affected by movable pad wells. Higher is better.

    Near-surface zones (both wells MD ≤ threshold) are excluded so that
    shared-surface pads don't produce a permanent SF=0 floor.
    """
    if analysis is None:
        return float("-inf")

    pad_sfs = [
        float(z.separation_factor)
        for z in _actionable_pad_zones(
            analysis,
            pad_well_names,
            fixed_well_names=fixed_well_names,
        )
    ]
    if not pad_sfs:
        return float("inf")

    # Primary: worst (min) SF.  Secondary: mean SF (small weight to
    # disambiguate ties and reward globally better arrangements).
    return float(min(pad_sfs)) + 0.001 * float(np.mean(pad_sfs))


def _optimization_score(
    analysis: AntiCollisionAnalysis | None,
    pad_well_names: set[str],
    fixed_well_names: set[str] | None = None,
) -> _PadOptimizationScore:
    if analysis is None:
        return _PadOptimizationScore(
            target_zone_count=1_000_000,
            overlap_pair_count=1_000_000,
            zone_count=1_000_000,
            severe_zone_count=1_000_000,
            worst_sf=float("-inf"),
            mean_sf=float("-inf"),
        )

    zones = _actionable_pad_zones(
        analysis,
        pad_well_names,
        fixed_well_names=fixed_well_names,
    )
    if not zones:
        return _PadOptimizationScore(
            target_zone_count=0,
            overlap_pair_count=0,
            zone_count=0,
            severe_zone_count=0,
            worst_sf=float("inf"),
            mean_sf=float("inf"),
        )

    sfs = np.asarray([float(zone.separation_factor) for zone in zones], dtype=float)
    pair_keys = {
        tuple(sorted((str(zone.well_a), str(zone.well_b)))) for zone in zones
    }
    target_pair_keys = {
        tuple(sorted((str(zone.well_a), str(zone.well_b))))
        for zone in zones
        if int(getattr(zone, "priority_rank", 2)) < 2
    }
    return _PadOptimizationScore(
        target_zone_count=len(target_pair_keys),
        overlap_pair_count=len(pair_keys),
        zone_count=len(zones),
        severe_zone_count=int(np.count_nonzero(sfs < 1.0)),
        worst_sf=float(np.min(sfs)),
        mean_sf=float(np.mean(sfs)),
    )


def _score_is_improvement(
    candidate: _PadOptimizationScore,
    current: _PadOptimizationScore,
) -> bool:
    for field_name in (
        "target_zone_count",
        "overlap_pair_count",
        "zone_count",
        "severe_zone_count",
    ):
        candidate_value = int(getattr(candidate, field_name))
        current_value = int(getattr(current, field_name))
        if candidate_value > current_value:
            return False

    for field_name in (
        "target_zone_count",
        "overlap_pair_count",
        "zone_count",
        "severe_zone_count",
    ):
        candidate_value = int(getattr(candidate, field_name))
        current_value = int(getattr(current, field_name))
        if candidate_value < current_value:
            return True

    if float(candidate.worst_sf) > float(current.worst_sf) + _IMPROVEMENT_EPS:
        return True
    if (
        abs(float(candidate.worst_sf) - float(current.worst_sf)) <= _IMPROVEMENT_EPS
        and float(candidate.mean_sf) > float(current.mean_sf) + _IMPROVEMENT_EPS
    ):
        return True
    return False


def _score_text(score: _PadOptimizationScore) -> str:
    if int(score.zone_count) <= 0:
        return "конфликтов 0"
    return (
        f"target-пар {int(score.target_zone_count)}, "
        f"пар {int(score.overlap_pair_count)}, "
        f"зон {int(score.zone_count)}, "
        f"SF min {float(score.worst_sf):.3f}, "
        f"SF mean {float(score.mean_sf):.3f}"
    )


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)) or float(seconds) < 1.0:
        return "оценка уточняется"
    total_seconds = int(round(float(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    if minutes <= 0:
        return f"~{sec} с"
    return f"~{minutes} мин {sec:02d} с"


def _surface_xyz_by_name(
    records: list[WelltrackRecord],
) -> dict[str, tuple[float, float, float]]:
    surfaces: dict[str, tuple[float, float, float]] = {}
    for record in records:
        if not record.points:
            continue
        point = record.points[0]
        surfaces[str(record.name)] = (
            float(point.x),
            float(point.y),
            float(point.z),
        )
    return surfaces


def _surface_distance(
    surfaces: Mapping[str, tuple[float, float, float]],
    left: str,
    right: str,
) -> float:
    left_point = surfaces.get(str(left))
    right_point = surfaces.get(str(right))
    if left_point is None or right_point is None:
        return float("inf")
    return float(
        np.linalg.norm(
            np.asarray(left_point, dtype=float) - np.asarray(right_point, dtype=float)
        )
    )


def _hot_well_rank_by_name(actionable_zones: list) -> dict[str, tuple[int, float, int]]:
    ranks: dict[str, list[float]] = {}
    target_counts: dict[str, int] = {}
    for zone in actionable_zones:
        sf = float(getattr(zone, "separation_factor", float("inf")))
        is_target = int(getattr(zone, "priority_rank", 2)) < 2
        for name in (str(zone.well_a), str(zone.well_b)):
            ranks.setdefault(name, []).append(sf)
            if is_target:
                target_counts[name] = target_counts.get(name, 0) + 1
    return {
        name: (
            -int(target_counts.get(name, 0)),
            float(min(values)) if values else float("inf"),
            -int(len(values)),
        )
        for name, values in ranks.items()
    }


def _candidate_swaps_for_zone(
    *,
    zone: object,
    actionable_zones: list,
    movable_names: set[str],
    surfaces: Mapping[str, tuple[float, float, float]],
    last_accepted_pair: tuple[str, str] | None,
    remaining_budget: int,
) -> list[tuple[str, str]]:
    """Build a small, high-value swap list for the current worst zone."""
    if remaining_budget <= 0:
        return []
    wa, wb = str(zone.well_a), str(zone.well_b)
    active_names = [name for name in (wa, wb) if name in movable_names]
    if not active_names:
        return []

    hot_rank = _hot_well_rank_by_name(actionable_zones)
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(left: str, right: str) -> None:
        if left == right:
            return
        if left not in movable_names or right not in movable_names:
            return
        key = tuple(sorted((str(left), str(right))))
        if key == last_accepted_pair or key in seen:
            return
        seen.add(key)
        candidates.append((str(left), str(right)))

    if wa in movable_names and wb in movable_names:
        add_candidate(wa, wb)

    for active_name in active_names:
        ranked_others = sorted(
            (name for name in movable_names if name != active_name),
            key=lambda name: (
                hot_rank.get(str(name), (0, float("inf"), 0)),
                _surface_distance(surfaces, active_name, str(name)),
                str(name),
            ),
        )
        for other in ranked_others:
            add_candidate(active_name, other)
            if len(candidates) >= min(_MAX_CANDIDATES_PER_ZONE, remaining_budget):
                return candidates

    return candidates[: min(_MAX_CANDIDATES_PER_ZONE, remaining_budget)]


# ---------------------------------------------------------------------------
#  Lightweight AC-well builder (no display geometry — fast).
# ---------------------------------------------------------------------------
def _build_ac_well_light(
    success: SuccessfulWellPlan,
    model: PlanningUncertaintyModel,
) -> AntiCollisionWell:
    return build_anti_collision_well(
        name=success.name,
        color="#A0A0A0",
        stations=success.stations,
        surface=success.surface,
        t1=success.t1,
        t3=success.t3,
        azimuth_deg=float(success.azimuth_deg),
        md_t1_m=float(success.md_t1_m),
        model=model,
        include_display_geometry=False,
        well_kind="project",
        is_reference_only=False,
    )


def _build_ref_ac_well_light(
    ref: ImportedTrajectoryWell,
    model: PlanningUncertaintyModel,
    *,
    collision_name: str | None = None,
) -> AntiCollisionWell:
    return build_anti_collision_well(
        name=str(collision_name) if collision_name is not None else str(ref.name),
        color=REFERENCE_WELL_KIND_COLORS.get(str(ref.kind), "#A0A0A0"),
        stations=ref.stations,
        surface=ref.surface,
        t1=None,
        t3=None,
        azimuth_deg=float(ref.azimuth_deg),
        md_t1_m=None,
        md_t3_m=None,
        model=model,
        include_display_geometry=False,
        well_kind=str(ref.kind),
        is_reference_only=True,
    )


def _reference_uncertainty_model(
    *,
    reference_well: ImportedTrajectoryWell,
    collision_name: str,
    default_model: PlanningUncertaintyModel,
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ),
) -> PlanningUncertaintyModel:
    if not reference_uncertainty_models_by_name:
        return default_model
    if str(getattr(reference_well, "kind", "")) != REFERENCE_WELL_ACTUAL:
        return default_model
    return (
        reference_uncertainty_models_by_name.get(str(collision_name))
        or reference_uncertainty_models_by_name.get(str(reference_well.name))
        or default_model
    )


def _analysis_well_signature(
    name: str,
    success: object,
    model: PlanningUncertaintyModel,
) -> str:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(name).encode("utf-8"))
    _update_model_signature(digest, model)
    for attr_name in ("surface", "t1", "t3"):
        point = getattr(success, attr_name, None)
        if point is None:
            digest.update(b"none")
            continue
        digest.update(
            np.asarray(
                [
                    float(getattr(point, "x", 0.0)),
                    float(getattr(point, "y", 0.0)),
                    float(getattr(point, "z", 0.0)),
                ],
                dtype=np.float64,
            ).tobytes()
        )
    digest.update(
        np.asarray(
            [
                float(getattr(success, "azimuth_deg", 0.0) or 0.0),
                float(getattr(success, "md_t1_m", 0.0) or 0.0),
            ],
            dtype=np.float64,
        ).tobytes()
    )
    stations = getattr(success, "stations", None)
    if isinstance(stations, pd.DataFrame):
        columns = [
            column
            for column in ("MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m")
            if column in stations.columns
        ]
        digest.update(str(tuple(columns)).encode("utf-8"))
        digest.update(str(len(stations)).encode("utf-8"))
        if columns:
            digest.update(stations.loc[:, columns].to_numpy(dtype=np.float64).tobytes())
    else:
        digest.update(repr(id(success)).encode("utf-8"))
    return digest.hexdigest()


def _reference_well_signature(
    collision_name: str,
    ref: ImportedTrajectoryWell,
    model: PlanningUncertaintyModel,
) -> str:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(collision_name).encode("utf-8"))
    digest.update(str(ref.name).encode("utf-8"))
    digest.update(str(ref.kind).encode("utf-8"))
    _update_model_signature(digest, model)
    digest.update(
        np.asarray(
            [float(ref.surface.x), float(ref.surface.y), float(ref.surface.z)],
            dtype=np.float64,
        ).tobytes()
    )
    digest.update(np.asarray([float(ref.azimuth_deg)], dtype=np.float64).tobytes())
    columns = [
        column
        for column in ("MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m")
        if column in ref.stations.columns
    ]
    digest.update(str(tuple(columns)).encode("utf-8"))
    digest.update(str(len(ref.stations)).encode("utf-8"))
    if columns:
        digest.update(ref.stations.loc[:, columns].to_numpy(dtype=np.float64).tobytes())
    return digest.hexdigest()


def _update_model_signature(digest: object, model: PlanningUncertaintyModel) -> None:
    digest.update(str(getattr(model, "iscwsa_tool_code", "") or "").encode("utf-8"))
    env = getattr(model, "iscwsa_environment", None)
    values = [
        float(getattr(model, "sigma_inc_deg", 0.0)),
        float(getattr(model, "sigma_azi_deg", 0.0)),
        float(getattr(model, "sigma_lateral_drift_m_per_1000m", 0.0)),
        float(getattr(model, "confidence_scale", 0.0)),
        float(getattr(model, "sample_step_m", 0.0)),
        float(getattr(model, "min_refined_step_m", 0.0)),
        float(getattr(model, "directional_refine_threshold_deg", 0.0)),
        float(getattr(env, "gtot_mps2", 0.0)),
        float(getattr(env, "mtot_nt", 0.0)),
        float(getattr(env, "dip_deg", 0.0)),
        float(getattr(env, "declination_deg", 0.0)),
        float(getattr(env, "lateral_singularity_inc_deg", 0.0)),
    ]
    digest.update(np.asarray(values, dtype=np.float64).tobytes())


def _cached_or_built_ac_well(
    *,
    name: str,
    signature_by_name: dict[str, str],
    previous_well_cache: Mapping[str, tuple[str, AntiCollisionWell]] | None,
    builder: Callable[[], AntiCollisionWell],
) -> AntiCollisionWell:
    previous = (previous_well_cache or {}).get(str(name))
    if previous is not None and len(previous) == 2:
        previous_signature, previous_well = previous
        if (
            str(signature_by_name.get(str(name), "")) == str(previous_signature)
            and isinstance(previous_well, AntiCollisionWell)
        ):
            return previous_well
    return builder()


def _analyze_incremental_from_ac_wells(
    ac_wells: dict[str, AntiCollisionWell],
    ref_ac_wells: tuple[AntiCollisionWell, ...],
    *,
    well_signature_by_name: Mapping[str, str] | None = None,
    previous_pair_cache: (
        Mapping[tuple[str, str], AntiCollisionPairCacheEntry] | None
    ) = None,
    parallel_workers: int = 0,
) -> tuple[
    AntiCollisionAnalysis,
    dict[tuple[str, str], AntiCollisionPairCacheEntry],
    AntiCollisionIncrementalStats,
]:
    """Run pairwise analysis from pre-built AC well objects (no geometry)."""
    all_wells = list(ac_wells.values()) + list(ref_ac_wells)
    return analyze_anti_collision_incremental(
        all_wells,
        build_overlap_geometry=False,
        well_signature_by_name=well_signature_by_name,
        previous_pair_cache=previous_pair_cache,
        parallel_workers=int(parallel_workers),
    )


def _swap_surfaces_and_recalculate(
    records: list[WelltrackRecord],
    successes: dict[str, SuccessfulWellPlan],
    name_a: str,
    name_b: str,
    config_by_name: dict[str, TrajectoryConfig],
    pool: ProcessPoolExecutor | None = None,
) -> tuple[list[WelltrackRecord], dict[str, SuccessfulWellPlan]] | None:
    """Swap surface points of two wells, recalculate both, return new state or None."""
    g_a = next((i for i, r in enumerate(records) if r.name == name_a), None)
    g_b = next((i for i, r in enumerate(records) if r.name == name_b), None)
    if g_a is None or g_b is None:
        return None

    pts_a = records[g_a].points
    pts_b = records[g_b].points
    if not pts_a or not pts_b:
        return None

    # Skip if surfaces are identical (swap would be a no-op).
    sa, sb = pts_a[0], pts_b[0]
    if (
        abs(float(sa.x) - float(sb.x)) < 0.01
        and abs(float(sa.y) - float(sb.y)) < 0.01
        and abs(float(sa.z) - float(sb.z)) < 0.01
    ):
        return None

    new_records = list(records)  # shallow copy — only replace swapped entries
    new_records[g_a] = WelltrackRecord(
        name=records[g_a].name, points=(pts_b[0], *pts_a[1:]),
    )
    new_records[g_b] = WelltrackRecord(
        name=records[g_b].name, points=(pts_a[0], *pts_b[1:]),
    )

    cfg_a = config_by_name.get(name_a)
    cfg_b = config_by_name.get(name_b)
    if cfg_a is None or cfg_b is None:
        return None

    # Parallel trajectory recalculation — the two wells are independent.
    # We submit dicts to avoid PicklingError when Streamlit reloads modules
    # (class identity changes between reruns).
    if pool is not None:
        try:
            future_a = pool.submit(
                _recalculate_well_from_dicts,
                new_records[g_a].model_dump(),
                cfg_a.model_dump(),
            )
            future_b = pool.submit(
                _recalculate_well_from_dicts,
                new_records[g_b].model_dump(),
                cfg_b.model_dump(),
            )
            raw_a = future_a.result()
            raw_b = future_b.result()
            r_a = (
                SuccessfulWellPlan.model_validate(raw_a)
                if raw_a is not None
                else None
            )
            r_b = (
                SuccessfulWellPlan.model_validate(raw_b)
                if raw_b is not None
                else None
            )
        except (BrokenProcessPool, PicklingError, OSError, RuntimeError, ValueError):
            r_a = recalculate_well(new_records[g_a], cfg_a)
            r_b = recalculate_well(new_records[g_b], cfg_b)
    else:
        r_a = recalculate_well(new_records[g_a], cfg_a)
        r_b = recalculate_well(new_records[g_b], cfg_b)

    if r_a is None or r_b is None:
        return None

    new_successes = dict(successes)
    new_successes[r_a.name] = r_a
    new_successes[r_b.name] = r_b
    return new_records, new_successes


def optimize_pad_order(
    records: list[WelltrackRecord],
    success_dict: dict[str, SuccessfulWellPlan],
    pad_well_names: set[str],
    uncertainty_model: PlanningUncertaintyModel,
    reference_wells: list[ImportedTrajectoryWell],
    config_by_name: dict[str, TrajectoryConfig],
    progress_callback: Callable[[int, str], None],
    fixed_well_names: set[str] | None = None,
    initial_analysis: AntiCollisionAnalysis | None = None,
    previous_well_cache: Mapping[str, tuple[str, AntiCollisionWell]] | None = None,
    previous_pair_cache: (
        Mapping[tuple[str, str], AntiCollisionPairCacheEntry] | None
    ) = None,
    well_signature_by_name: Mapping[str, str] | None = None,
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    parallel_workers: int = 0,
) -> tuple[list[WelltrackRecord], dict[str, SuccessfulWellPlan], bool]:

    pad_names = {
        str(name)
        for name in pad_well_names
        if str(name) in success_dict
    }
    if len(pad_names) < 2:
        return records, success_dict, False
    fixed_names = {
        str(name)
        for name in (fixed_well_names or set())
        if str(name) in pad_names
    }
    movable_names = set(pad_names) - fixed_names
    if len(movable_names) < 2:
        progress_callback(
            100,
            "Оптимизация не запущена: для перестановки нужно минимум две "
            "незафиксированные скважины.",
        )
        return records, success_dict, False

    # Check that at least 2 distinct surface locations exist; otherwise
    # swapping surfaces is a no-op and we can exit early.
    surface_set: set[tuple[float, float, float]] = set()
    for r in records:
        if r.name in movable_names and r.points:
            s = r.points[0]
            surface_set.add((round(float(s.x), 2), round(float(s.y), 2), round(float(s.z), 2)))
    if len(surface_set) < 2:
        progress_callback(100, "Все скважины на одной точке — перестановка не даст эффекта.")
        return records, success_dict, False

    ref_wells = tuple(reference_wells)
    # Use a Streamlit-safe context: spawn on Windows/macOS, forkserver on Linux.
    _mp_ctx = process_pool_context()
    try:
        pool: ProcessPoolExecutor | None = ProcessPoolExecutor(
            max_workers=2,
            mp_context=_mp_ctx,
        )
    except (BrokenProcessPool, PicklingError, OSError, RuntimeError, ValueError):
        pool = None

    # --- Pre-build lightweight AC wells (no display geometry). ---
    # Reused across candidates; only swapped wells are rebuilt.
    current_signatures = {
        str(name): str(signature)
        for name, signature in (well_signature_by_name or {}).items()
    }
    duplicate_reference_name_keys = reference_well_duplicate_name_keys(ref_wells)
    planned_names = tuple(success_dict.keys())
    reference_collision_items = tuple(
        (
            reference_well_collision_name(
                rw,
                planned_names=planned_names,
                duplicate_name_keys=duplicate_reference_name_keys,
            ),
            rw,
        )
        for rw in ref_wells
    )
    reference_model_by_collision_name = {
        str(collision_name): _reference_uncertainty_model(
            reference_well=rw,
            collision_name=str(collision_name),
            default_model=uncertainty_model,
            reference_uncertainty_models_by_name=(
                reference_uncertainty_models_by_name
            ),
        )
        for collision_name, rw in reference_collision_items
    }
    for collision_name, rw in reference_collision_items:
        reference_model = reference_model_by_collision_name[str(collision_name)]
        current_signatures.setdefault(
            str(collision_name),
            _reference_well_signature(str(collision_name), rw, reference_model),
        )
    ref_ac_wells = tuple(
        _cached_or_built_ac_well(
            name=str(collision_name),
            signature_by_name=current_signatures,
            previous_well_cache=previous_well_cache,
            builder=lambda rw=rw, collision_name=str(
                collision_name
            ), reference_model=reference_model_by_collision_name[
                str(collision_name)
            ]: _build_ref_ac_well_light(
                rw,
                reference_model,
                collision_name=collision_name,
            ),
        )
        for collision_name, rw in reference_collision_items
    )
    ac_well_cache: dict[str, AntiCollisionWell] = {}
    for name, success in success_dict.items():
        current_signatures.setdefault(
            str(name),
            _analysis_well_signature(str(name), success, uncertainty_model),
        )
        ac_well_cache[str(name)] = _cached_or_built_ac_well(
            name=str(name),
            signature_by_name=current_signatures,
            previous_well_cache=previous_well_cache,
            builder=lambda success=success: _build_ac_well_light(
                success,
                uncertainty_model,
            ),
        )

    if initial_analysis is not None and previous_pair_cache is not None:
        analysis = initial_analysis
        current_pair_cache = dict(previous_pair_cache)
    else:
        analysis, current_pair_cache, initial_stats = _analyze_incremental_from_ac_wells(
            ac_well_cache,
            ref_ac_wells,
            well_signature_by_name=current_signatures,
            previous_pair_cache=previous_pair_cache,
            parallel_workers=int(parallel_workers),
        )
        if initial_stats.reused_pair_count:
            progress_callback(
                0,
                (
                    "Стартовая оценка anti-collision использовала кэш: "
                    f"{int(initial_stats.reused_pair_count)} пар."
                ),
            )
    best_score = _optimization_score(
        analysis,
        pad_names,
        fixed_well_names=fixed_names,
    )
    best_score_value = score_analysis(
        analysis,
        pad_names,
        fixed_well_names=fixed_names,
    )

    best_records = list(records)
    best_successes = dict(success_dict)
    improved = False
    last_accepted_pair: tuple[str, str] | None = None
    optimization_started = perf_counter()
    evaluated_candidate_count = 0
    accepted_swap_count = 0
    stopped_by_budget = False
    surfaces_by_name = _surface_xyz_by_name(best_records)

    progress_callback(0, f"Начальный score: {_score_text(best_score)}.")

    try:
        for iteration in range(1, _MAX_ITERATIONS + 1):
            if evaluated_candidate_count >= _MAX_CANDIDATE_EVALUATIONS:
                stopped_by_budget = True
                break
            if perf_counter() - optimization_started >= _MAX_RUNTIME_S:
                stopped_by_budget = True
                break
            actionable = _actionable_pad_zones(
                analysis,
                pad_names,
                fixed_well_names=fixed_names,
            )
            if not actionable:
                break

            actionable.sort(key=lambda z: float(z.separation_factor))

            # Try up to _MAX_WORST_ZONES_PER_ITERATION distinct worst-zone pairs.
            swap_accepted = False
            seen_pairs_this_iter: set[tuple[str, str]] = set()
            tested_candidates_this_iter = 0

            for zone in actionable[:_MAX_WORST_ZONES_PER_ITERATION]:
                if evaluated_candidate_count >= _MAX_CANDIDATE_EVALUATIONS:
                    stopped_by_budget = True
                    break
                if perf_counter() - optimization_started >= _MAX_RUNTIME_S:
                    stopped_by_budget = True
                    break
                if tested_candidates_this_iter >= _MAX_CANDIDATES_PER_ITERATION:
                    break
                wa, wb = str(zone.well_a), str(zone.well_b)
                pair_key = tuple(sorted((wa, wb)))
                if pair_key in seen_pairs_this_iter:
                    continue
                seen_pairs_this_iter.add(pair_key)

                zone_progress_percent = min(
                    94,
                    4
                    + int(
                        90
                        * evaluated_candidate_count
                        / max(_MAX_CANDIDATE_EVALUATIONS, 1)
                    ),
                )
                progress_callback(
                    zone_progress_percent,
                    f"Итерация {iteration}: анализ {wa} ↔ {wb} (SF={float(zone.separation_factor):.3f})...",
                )

                remaining_global_budget = (
                    _MAX_CANDIDATE_EVALUATIONS - evaluated_candidate_count
                )
                remaining_iter_budget = (
                    _MAX_CANDIDATES_PER_ITERATION - tested_candidates_this_iter
                )
                swap_candidates = _candidate_swaps_for_zone(
                    zone=zone,
                    actionable_zones=actionable,
                    movable_names=movable_names,
                    surfaces=surfaces_by_name,
                    last_accepted_pair=last_accepted_pair,
                    remaining_budget=min(remaining_global_budget, remaining_iter_budget),
                )

                best_candidate = None
                best_candidate_score = best_score
                best_candidate_score_value = best_score_value
                best_candidate_ac_updates: dict[str, AntiCollisionWell] | None = None
                best_candidate_signatures: dict[str, str] | None = None
                best_candidate_pair_cache: (
                    dict[tuple[str, str], AntiCollisionPairCacheEntry] | None
                ) = None
                best_candidate_analysis: AntiCollisionAnalysis | None = None
                best_candidate_pair_key: tuple[str, str] | None = None

                for name_1, name_2 in swap_candidates:
                    if evaluated_candidate_count >= _MAX_CANDIDATE_EVALUATIONS:
                        stopped_by_budget = True
                        break
                    elapsed_before = perf_counter() - optimization_started
                    if elapsed_before >= _MAX_RUNTIME_S:
                        stopped_by_budget = True
                        break
                    candidate_pair_key = tuple(sorted((str(name_1), str(name_2))))
                    if candidate_pair_key == last_accepted_pair:
                        continue
                    evaluated_candidate_count += 1
                    tested_candidates_this_iter += 1
                    avg_candidate_s = elapsed_before / max(evaluated_candidate_count - 1, 1)
                    remaining_candidates = (
                        _MAX_CANDIDATE_EVALUATIONS - evaluated_candidate_count + 1
                    )
                    eta_text = _format_eta(avg_candidate_s * remaining_candidates)
                    progress_percent = min(
                        94,
                        4
                        + int(
                            90
                            * evaluated_candidate_count
                            / max(_MAX_CANDIDATE_EVALUATIONS, 1)
                        ),
                    )
                    progress_callback(
                        progress_percent,
                        (
                            f"Кандидат {evaluated_candidate_count}/"
                            f"{_MAX_CANDIDATE_EVALUATIONS}: перестановка "
                            f"{name_1} ↔ {name_2}. ETA {eta_text}. "
                            f"Лучший score: {_score_text(best_score)}."
                        ),
                    )
                    result = _swap_surfaces_and_recalculate(
                        best_records, best_successes, name_1, name_2, config_by_name,
                        pool=pool,
                    )
                    if result is None:
                        progress_callback(
                            progress_percent,
                            (
                                f"Кандидат {evaluated_candidate_count}: "
                                f"{name_1} ↔ {name_2} пропущен — траектория "
                                "после перестановки не построилась."
                            ),
                        )
                        continue

                    cand_records, cand_successes = result

                    # Incremental AC well update: only rebuild the 2 changed wells.
                    ac_well_1 = _build_ac_well_light(
                        cand_successes[name_1],
                        uncertainty_model,
                    )
                    ac_well_2 = _build_ac_well_light(
                        cand_successes[name_2],
                        uncertainty_model,
                    )
                    cand_ac_wells = dict(ac_well_cache)
                    cand_ac_wells[name_1] = ac_well_1
                    cand_ac_wells[name_2] = ac_well_2
                    cand_signatures = dict(current_signatures)
                    cand_signatures[name_1] = _analysis_well_signature(
                        name_1,
                        cand_successes[name_1],
                        uncertainty_model,
                    )
                    cand_signatures[name_2] = _analysis_well_signature(
                        name_2,
                        cand_successes[name_2],
                        uncertainty_model,
                    )

                    cand_analysis, cand_pair_cache, cand_stats = (
                        _analyze_incremental_from_ac_wells(
                            cand_ac_wells,
                            ref_ac_wells,
                            well_signature_by_name=cand_signatures,
                            previous_pair_cache=current_pair_cache,
                            parallel_workers=int(parallel_workers),
                        )
                    )
                    cand_score = _optimization_score(
                        cand_analysis,
                        pad_names,
                        fixed_well_names=fixed_names,
                    )
                    cand_score_value = score_analysis(
                        cand_analysis,
                        pad_names,
                        fixed_well_names=fixed_names,
                    )
                    progress_callback(
                        progress_percent,
                        (
                            f"Кандидат {evaluated_candidate_count}: "
                            f"{name_1} ↔ {name_2} оценен. "
                            f"{_score_text(cand_score)}; кэш пар "
                            f"{int(getattr(cand_stats, 'reused_pair_count', 0))} reused / "
                            f"{int(getattr(cand_stats, 'recalculated_pair_count', 0))} recalculated."
                        ),
                    )

                    if _score_is_improvement(cand_score, best_candidate_score):
                        best_candidate_score = cand_score
                        best_candidate_score_value = cand_score_value
                        best_candidate = (cand_records, cand_successes)
                        best_candidate_ac_updates = {
                            name_1: ac_well_1,
                            name_2: ac_well_2,
                        }
                        best_candidate_signatures = cand_signatures
                        best_candidate_pair_cache = cand_pair_cache
                        best_candidate_analysis = cand_analysis
                        best_candidate_pair_key = candidate_pair_key

                if best_candidate is not None:
                    best_records, best_successes = best_candidate
                    best_score = best_candidate_score
                    best_score_value = best_candidate_score_value
                    current_signatures = dict(
                        best_candidate_signatures or current_signatures
                    )
                    current_pair_cache = dict(
                        best_candidate_pair_cache or current_pair_cache
                    )
                    analysis = best_candidate_analysis or analysis
                    # Update the AC well cache with the accepted swap.
                    if best_candidate_ac_updates is not None:
                        ac_well_cache.update(best_candidate_ac_updates)
                    last_accepted_pair = best_candidate_pair_key
                    surfaces_by_name = _surface_xyz_by_name(best_records)
                    swap_accepted = True
                    improved = True
                    accepted_swap_count += 1
                    progress_callback(
                        min(
                            96,
                            4
                            + int(
                                90
                                * evaluated_candidate_count
                                / max(_MAX_CANDIDATE_EVALUATIONS, 1)
                            ),
                        ),
                        (
                            f"Принята перестановка {last_accepted_pair[0]} ↔ "
                            f"{last_accepted_pair[1]}. Новый score: "
                            f"{_score_text(best_score)}."
                        ),
                    )
                    break  # restart from updated state

            if evaluated_candidate_count >= _MAX_CANDIDATE_EVALUATIONS:
                stopped_by_budget = True
                break
            if perf_counter() - optimization_started >= _MAX_RUNTIME_S:
                stopped_by_budget = True
                break
            if not swap_accepted:
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    elapsed_total = perf_counter() - optimization_started
    budget_text = (
        " Остановлено по лимиту времени/кандидатов."
        if stopped_by_budget
        else ""
    )
    progress_callback(
        100,
        (
            f"Готово за {elapsed_total:.1f} с: принято {accepted_swap_count}, "
            f"проверено {evaluated_candidate_count} кандидатов. Лучший score: "
            f"{_score_text(best_score)} ({best_score_value:.3f}).{budget_text}"
        ),
    )
    return best_records, best_successes, improved
