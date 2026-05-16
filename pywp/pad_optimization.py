from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
from typing import Callable

import numpy as np

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionWell,
    analyze_anti_collision,
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


def _analyze_from_ac_wells(
    ac_wells: dict[str, AntiCollisionWell],
    ref_ac_wells: tuple[AntiCollisionWell, ...],
) -> AntiCollisionAnalysis:
    """Run pairwise analysis from pre-built AC well objects (no geometry)."""
    all_wells = list(ac_wells.values()) + list(ref_ac_wells)
    return analyze_anti_collision(all_wells, build_overlap_geometry=False)


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
        except (BrokenProcessPool, OSError, RuntimeError, ValueError):
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
    except (BrokenProcessPool, OSError, RuntimeError, ValueError):
        pool = None

    # --- Pre-build lightweight AC wells (no display geometry). ---
    # Reused across candidates; only swapped wells are rebuilt.
    duplicate_reference_name_keys = reference_well_duplicate_name_keys(ref_wells)
    ref_ac_wells = tuple(
        _build_ref_ac_well_light(
            rw,
            uncertainty_model,
            collision_name=reference_well_collision_name(
                rw,
                planned_names=tuple(success_dict.keys()),
                duplicate_name_keys=duplicate_reference_name_keys,
            ),
        )
        for rw in ref_wells
    )
    ac_well_cache: dict[str, AntiCollisionWell] = {
        name: _build_ac_well_light(success_dict[name], uncertainty_model)
        for name in success_dict
    }

    analysis = _analyze_from_ac_wells(ac_well_cache, ref_ac_wells)
    best_score = score_analysis(
        analysis,
        pad_names,
        fixed_well_names=fixed_names,
    )

    best_records = list(records)
    best_successes = dict(success_dict)
    improved = False
    tried_pairs: set[tuple[str, str]] = set()

    progress_callback(0, f"Начальный score: {best_score:.3f}...")

    try:
        for iteration in range(1, _MAX_ITERATIONS + 1):
            analysis = _analyze_from_ac_wells(ac_well_cache, ref_ac_wells)
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

            for zone in actionable[:_MAX_WORST_ZONES_PER_ITERATION]:
                wa, wb = str(zone.well_a), str(zone.well_b)
                pair_key = tuple(sorted((wa, wb)))
                if pair_key in seen_pairs_this_iter:
                    continue
                seen_pairs_this_iter.add(pair_key)

                progress_callback(
                    int(iteration * 100 / _MAX_ITERATIONS),
                    f"Итерация {iteration}: анализ {wa} ↔ {wb} (SF={float(zone.separation_factor):.3f})...",
                )

                # Generate all distinct swaps involving non-fixed wells from the
                # worst pair. Fixed wells keep their surface slot unchanged.
                active_names = [
                    name for name in (wa, wb) if name in movable_names
                ]
                swap_candidates: list[tuple[str, str]] = []
                seen_candidate_keys: set[tuple[str, str]] = set()
                for active_name in active_names:
                    for other in sorted(movable_names - {active_name}):
                        candidate_key = tuple(sorted((active_name, other)))
                        if candidate_key in seen_candidate_keys:
                            continue
                        seen_candidate_keys.add(candidate_key)
                        swap_candidates.append((active_name, other))

                best_candidate = None
                best_candidate_score = best_score
                best_candidate_ac_updates: dict[str, AntiCollisionWell] | None = None

                for name_1, name_2 in swap_candidates:
                    ck = tuple(sorted((name_1, name_2)))
                    if ck in tried_pairs:
                        continue
                    tried_pairs.add(ck)

                    result = _swap_surfaces_and_recalculate(
                        best_records, best_successes, name_1, name_2, config_by_name,
                        pool=pool,
                    )
                    if result is None:
                        continue

                    cand_records, cand_successes = result

                    # Incremental AC well update: only rebuild the 2 changed wells.
                    ac_well_1 = _build_ac_well_light(cand_successes[name_1], uncertainty_model)
                    ac_well_2 = _build_ac_well_light(cand_successes[name_2], uncertainty_model)
                    cand_ac_wells = dict(ac_well_cache)
                    cand_ac_wells[name_1] = ac_well_1
                    cand_ac_wells[name_2] = ac_well_2

                    cand_analysis = _analyze_from_ac_wells(cand_ac_wells, ref_ac_wells)
                    cand_score = score_analysis(
                        cand_analysis,
                        pad_names,
                        fixed_well_names=fixed_names,
                    )

                    if cand_score > best_candidate_score + _IMPROVEMENT_EPS:
                        best_candidate_score = cand_score
                        best_candidate = (cand_records, cand_successes)
                        best_candidate_ac_updates = {name_1: ac_well_1, name_2: ac_well_2}

                if best_candidate is not None:
                    best_records, best_successes = best_candidate
                    best_score = best_candidate_score
                    # Update the AC well cache with the accepted swap.
                    ac_well_cache.update(best_candidate_ac_updates)
                    swap_accepted = True
                    improved = True
                    break  # restart from updated state

            if not swap_accepted:
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    progress_callback(100, f"Готово! Лучший score: {best_score:.3f}")
    return best_records, best_successes, improved
