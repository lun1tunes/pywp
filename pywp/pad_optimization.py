from __future__ import annotations

import copy

import numpy as np

from pywp.eclipse_welltrack import WelltrackRecord, welltrack_points_to_targets
from pywp.ptc_core import _source_surface_xyz
from pywp.welltrack_batch import SuccessfulWellPlan
from pywp.anticollision_rerun import (
    build_anti_collision_analysis_for_successes,
)
from pywp.uncertainty import PlanningUncertaintyModel
from pywp.reference_trajectories import ImportedTrajectoryWell
from pywp.planner import TrajectoryPlanner
from pywp.anticollision import AntiCollisionAnalysis


def recalculate_well(
    record: WelltrackRecord, config: TrajectoryConfig
) -> SuccessfulWellPlan | None:
    from time import perf_counter

    start_t = perf_counter()
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    planner = TrajectoryPlanner()
    try:
        result = planner.plan(surface=surface, t1=t1, t3=t3, config=config)
        return SuccessfulWellPlan(
            name=record.name,
            surface=surface,
            t1=t1,
            t3=t3,
            stations=result.stations,
            summary=result.summary,
            azimuth_deg=result.azimuth_deg,
            md_t1_m=result.md_t1_m,
            config=config,
            runtime_s=perf_counter() - start_t,
        )
    except Exception:
        return None


def score_analysis(
    analysis: AntiCollisionAnalysis | None, pad_well_names: set[str]
) -> float:
    """Evaluate severity of collisions specifically within the pad. Higher is better."""
    if analysis is None:
        return float("-inf")

    # We want to minimize the severity. We can use the sum of 1/SF for bad collisions, or minimum SF.
    pad_sfs = [
        float(z.separation_factor)
        for z in analysis.zones
        if z.well_a in pad_well_names and z.well_b in pad_well_names
    ]
    if not pad_sfs:
        return float("inf")

    # Metric: primary min SF, secondary average SF
    return float(min(pad_sfs)) + 0.001 * float(np.mean(pad_sfs))


def optimize_pad_order(
    records: list[WelltrackRecord],
    success_dict: dict[str, SuccessfulWellPlan],
    pad_well_names: set[str],
    uncertainty_model: PlanningUncertaintyModel,
    reference_wells: list[ImportedTrajectoryWell],
    config_by_name: dict[str, TrajectoryConfig],
    progress_callback: callable,
) -> tuple[list[WelltrackRecord], dict[str, SuccessfulWellPlan], bool]:

    pad_records = [r for r in records if r.name in pad_well_names and r.name in success_dict]
    if len(pad_records) < 2:
        return records, success_dict, False

    slots = [_source_surface_xyz(r) for r in pad_records]
    if any(s is None for s in slots):
        return records, success_dict, False

    current_analysis = build_anti_collision_analysis_for_successes(
        tuple(success_dict.values()),
        model=uncertainty_model,
        reference_wells=tuple(reference_wells),
    )
    current_score = score_analysis(current_analysis, pad_well_names)

    best_score = current_score
    best_records = copy.deepcopy(records)
    best_successes = copy.deepcopy(success_dict)

    # Simple heuristic: sort slots by principal component to find "extremes"
    coords = np.array(slots)
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    pc1 = vh[0]
    projections = centered @ pc1

    # Sort slots by projection
    slot_indices = np.argsort(projections)
    end_slots = [slot_indices[0], slot_indices[-1]]

    iteration = 0
    max_iterations = 5
    improved = False

    progress_callback(0, f"Начальный SF: {best_score:.2f}...")

    while iteration < max_iterations:
        iteration += 1

        # Find worst collision pair on the pad
        current_analysis = build_anti_collision_analysis_for_successes(
            tuple(best_successes.values()),
            model=uncertainty_model,
            reference_wells=tuple(reference_wells),
        )
        pad_zones = [
            z
            for z in current_analysis.zones
            if z.well_a in pad_well_names and z.well_b in pad_well_names
        ]
        if not pad_zones:
            break

        pad_zones.sort(key=lambda z: float(z.separation_factor))
        worst_zone = pad_zones[0]
        wa, wb = worst_zone.well_a, worst_zone.well_b

        progress_callback(
            int(iteration * 100 / max_iterations), f"Анализ {wa} и {wb}..."
        )

        # Try swapping them to the furthest slots if they are not already there
        # We need the indices of wa and wb in the current pad_records
        wa_idx = next(i for i, r in enumerate(pad_records) if r.name == wa)
        wb_idx = next(i for i, r in enumerate(pad_records) if r.name == wb)

        # Possible candidates to swap with: ends of the pad
        swap_candidates = []
        for end_slot_idx in end_slots:
            if end_slot_idx != wa_idx and end_slot_idx != wb_idx:
                swap_candidates.append((wa_idx, end_slot_idx))
                swap_candidates.append((wb_idx, end_slot_idx))

        # Also try just swapping wa and wb directly
        swap_candidates.append((wa_idx, wb_idx))

        swap_found = False
        for idx1, idx2 in swap_candidates:
            # Try swap
            test_records = copy.deepcopy(best_records)
            test_successes = copy.deepcopy(best_successes)

            # Find the actual global indices
            g_idx1 = next(
                i
                for i, r in enumerate(test_records)
                if r.name == pad_records[idx1].name
            )
            g_idx2 = next(
                i
                for i, r in enumerate(test_records)
                if r.name == pad_records[idx2].name
            )

            # Swap surface points
            pts1 = test_records[g_idx1].points
            pts2 = test_records[g_idx2].points

            new_pts1 = (pts2[0], *pts1[1:])
            new_pts2 = (pts1[0], *pts2[1:])

            test_records[g_idx1] = WelltrackRecord(
                name=test_records[g_idx1].name, points=new_pts1
            )
            test_records[g_idx2] = WelltrackRecord(
                name=test_records[g_idx2].name, points=new_pts2
            )

            r1 = recalculate_well(
                test_records[g_idx1], config_by_name[test_records[g_idx1].name]
            )
            r2 = recalculate_well(
                test_records[g_idx2], config_by_name[test_records[g_idx2].name]
            )

            if r1 is None or r2 is None:
                continue

            test_successes[r1.name] = r1
            test_successes[r2.name] = r2

            test_analysis = build_anti_collision_analysis_for_successes(
                tuple(test_successes.values()),
                model=uncertainty_model,
                reference_wells=tuple(reference_wells),
            )
            test_score = score_analysis(test_analysis, pad_well_names)

            if test_score > best_score + 0.01:
                best_score = test_score
                best_records = test_records
                best_successes = test_successes

                # Update pad_records reference for next iteration
                pad_records = [
                    r for r in best_records if r.name in pad_well_names
                ]
                swap_found = True
                improved = True
                break

        if not swap_found:
            # If no swap for the worst pair improved things, we are likely stuck in a local minimum for this heuristic.
            break

    progress_callback(100, f"Готово! Лучший SF: {best_score:.2f}")
    return best_records, best_successes, improved
