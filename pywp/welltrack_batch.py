from __future__ import annotations

import logging
import multiprocessing
import sys
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Mapping
from time import perf_counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from pydantic import field_validator

from pywp.eclipse_welltrack import WelltrackRecord, welltrack_points_to_targets
from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
    evaluate_stations_anti_collision_clearance,
)
from pywp.anticollision_rerun import (
    DYNAMIC_CLUSTER_PLAN_ACTIVE,
    DYNAMIC_CLUSTER_PLAN_BLOCKED,
    DYNAMIC_CLUSTER_PLAN_RESOLVED,
    DynamicClusterExecutionPlan,
    build_anti_collision_analysis_for_successes,
    build_anticollision_well_contexts,
    build_dynamic_cluster_execution_plan,
)
from pywp.anticollision_recommendations import (
    build_anti_collision_recommendation_clusters,
    build_anti_collision_recommendations,
)
from pywp.anticollision_stage import (
    ANTI_COLLISION_STAGE_EARLY_KOP_BUILD1,
    ANTI_COLLISION_STAGE_LATE_TRAJECTORY,
    anti_collision_stage_from_context,
)
from pywp.models import Point3D, SummaryDict, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner
from pywp.pydantic_base import FrozenArbitraryModel, coerce_model_like
from pywp.reference_trajectories import ImportedTrajectoryWell
from pywp.solver_diagnostics import summarize_problem_ru
from pywp.uncertainty import PlanningUncertaintyModel
from pywp.ui_utils import dls_to_pi

ProgressCallback = Callable[[int, int, str], None]
SolverProgressCallback = Callable[[int, int, str, str, float], None]
RecordDoneCallback = Callable[[int, int, str, dict[str, Any]], None]


@dataclass(frozen=True)
class DynamicClusterExecutionContext:
    target_well_names: tuple[str, ...]
    uncertainty_model: PlanningUncertaintyModel
    initial_successes: tuple["SuccessfulWellPlan", ...]
    reference_wells: tuple[ImportedTrajectoryWell, ...] = ()


_MAX_DYNAMIC_CLUSTER_PASSES = 3


@dataclass(frozen=True)
class BatchEvaluationMetadata:
    executed_well_names: tuple[str, ...] = ()
    skipped_selected_names: tuple[str, ...] = ()
    cluster_resolved_early: bool = False
    cluster_blocked: bool = False
    cluster_blocking_reason: str | None = None


class SuccessfulWellPlan(FrozenArbitraryModel):
    name: str
    surface: Point3D
    t1: Point3D
    t3: Point3D
    stations: pd.DataFrame
    summary: SummaryDict
    azimuth_deg: float
    md_t1_m: float
    config: TrajectoryConfig
    runtime_s: float | None = None
    md_postcheck_exceeded: bool = False
    md_postcheck_message: str = ""

    @field_validator("surface", "t1", "t3", mode="before")
    @classmethod
    def _coerce_point3d(cls, value: object) -> Point3D:
        return coerce_model_like(value, Point3D)

    @field_validator("config", mode="before")
    @classmethod
    def _coerce_config(cls, value: object) -> TrajectoryConfig:
        return coerce_model_like(value, TrajectoryConfig)


def rebuild_optimization_context(
    *,
    context: AntiCollisionOptimizationContext | None,
    reference_success_by_name: Mapping[str, SuccessfulWellPlan],
    strict_missing_references: bool = False,
) -> AntiCollisionOptimizationContext | None:
    if context is None:
        return None
    references = []
    changed = False
    for reference in context.references:
        success = reference_success_by_name.get(str(reference.well_name))
        if success is None:
            if strict_missing_references:
                return None
            references.append(reference)
            continue
        try:
            updated_reference = build_anti_collision_reference_path(
                well_name=str(success.name),
                stations=success.stations,
                md_start_m=float(reference.md_start_m),
                md_end_m=float(reference.md_end_m),
                sample_step_m=float(context.sample_step_m),
                model=context.uncertainty_model,
            )
        except ValueError:
            if strict_missing_references:
                return None
            references.append(reference)
            continue
        references.append(updated_reference)
        changed = True
    if not changed:
        return context
    return AntiCollisionOptimizationContext(
        candidate_md_start_m=float(context.candidate_md_start_m),
        candidate_md_end_m=float(context.candidate_md_end_m),
        sf_target=float(context.sf_target),
        sample_step_m=float(context.sample_step_m),
        uncertainty_model=context.uncertainty_model,
        references=tuple(references),
        prefer_lower_kop=bool(context.prefer_lower_kop),
        prefer_higher_build1=bool(context.prefer_higher_build1),
        prefer_keep_kop=bool(context.prefer_keep_kop),
        prefer_keep_build1=bool(context.prefer_keep_build1),
        prefer_adjust_build2=bool(context.prefer_adjust_build2),
        baseline_kop_vertical_m=(
            float(context.baseline_kop_vertical_m)
            if context.baseline_kop_vertical_m is not None
            else None
        ),
        baseline_build1_dls_deg_per_30m=(
            float(context.baseline_build1_dls_deg_per_30m)
            if context.baseline_build1_dls_deg_per_30m is not None
            else None
        ),
        interpolation_method=str(context.interpolation_method),
    )



def _normalized_attempted_anticollision_stages(
    summary: Mapping[str, object],
) -> list[str]:
    raw_value = summary.get("anti_collision_attempted_stages")
    if isinstance(raw_value, str):
        items = raw_value.split("|")
    else:
        items = []
    stage_value = summary.get("anti_collision_stage")
    if isinstance(stage_value, str):
        items.append(stage_value)
    known_stages = {
        ANTI_COLLISION_STAGE_EARLY_KOP_BUILD1,
        ANTI_COLLISION_STAGE_LATE_TRAJECTORY,
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        stage = str(item).strip()
        if stage not in known_stages or stage in seen:
            continue
        normalized.append(stage)
        seen.add(stage)
    return normalized


def _success_with_attempted_anticollision_stage(
    *,
    success: SuccessfulWellPlan,
    stage: str | None,
) -> SuccessfulWellPlan:
    stage_name = str(stage or "").strip()
    if not stage_name:
        return success
    summary = dict(success.summary)
    attempted_stages = _normalized_attempted_anticollision_stages(summary)
    if stage_name not in attempted_stages:
        attempted_stages.append(stage_name)
    summary["anti_collision_attempted_stages"] = "|".join(attempted_stages)
    return success.validated_copy(summary=summary)


def _evaluate_record_from_dicts(
    record_dict: dict,
    config_dict: dict,
    optimization_context_dict: dict | None = None,
) -> tuple[dict[str, Any], dict | None]:
    """Worker entry-point that accepts/returns plain dicts.

    Streamlit's script rerunner can reload modules, making Pydantic class
    objects in the main process differ from those in ``sys.modules``.
    Pickle checks identity and raises ``PicklingError``.  By serialising
    to dicts before crossing the process boundary we avoid this entirely.
    """
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    record = WelltrackRecord.model_validate(record_dict)
    config = TrajectoryConfig.model_validate(config_dict)
    opt_ctx: AntiCollisionOptimizationContext | None = None
    if optimization_context_dict is not None:
        # AntiCollisionOptimizationContext is a @dataclass, not Pydantic
        opt_ctx = AntiCollisionOptimizationContext(**optimization_context_dict)
    row, success = _evaluate_record_standalone(record, config, opt_ctx)
    return row, success.model_dump() if success is not None else None


def _evaluate_record_standalone(
    record: WelltrackRecord,
    config: TrajectoryConfig,
    optimization_context: AntiCollisionOptimizationContext | None = None,
) -> tuple[dict[str, Any], SuccessfulWellPlan | None]:
    """Top-level picklable version of ``_evaluate_record`` for multiprocessing.

    Creates its own ``TrajectoryPlanner`` inside the worker process so that
    nothing unpicklable needs to cross the process boundary.
    """
    base_row = WelltrackBatchPlanner._base_row(record=record)
    if len(record.points) != 3:
        base_row["Статус"] = "Ошибка формата"
        base_row["Проблема"] = (
            f"Ожидалось 3 точки (S, t1, t3), получено {len(record.points)}."
        )
        return base_row, None

    try:
        surface, t1, t3 = welltrack_points_to_targets(record.points)
        started = perf_counter()
        planner = TrajectoryPlanner()
        plan_kwargs: dict[str, Any] = {
            "surface": surface,
            "t1": t1,
            "t3": t3,
            "config": config,
        }
        if optimization_context is not None:
            plan_kwargs["optimization_context"] = optimization_context
        result = planner.plan(**plan_kwargs)
        runtime_s = float(perf_counter() - started)
    except (ValueError, PlanningError) as exc:
        base_row["Статус"] = "Ошибка расчета"
        base_row["Проблема"] = summarize_problem_ru(str(exc))
        return base_row, None

    summary = dict(result.summary)
    ac_stage = anti_collision_stage_from_context(optimization_context)
    if ac_stage is not None:
        summary["anti_collision_stage"] = ac_stage
        summary["anti_collision_attempted_stages"] = ac_stage
    md_total_m = float(summary.get("md_total_m", 0.0))
    md_limit_m = float(summary.get("max_total_md_postcheck_m", 0.0))
    md_postcheck_excess_m = float(summary.get("md_postcheck_excess_m", 0.0))
    md_postcheck_exceeded = bool(md_postcheck_excess_m > 1e-6)
    md_postcheck_message = ""
    if md_postcheck_exceeded:
        md_postcheck_message = (
            "Превышен лимит итоговой MD (постпроверка): "
            f"{md_total_m:.2f} м > {md_limit_m:.2f} м (+{md_postcheck_excess_m:.2f} м)."
        )

    success = SuccessfulWellPlan(
        name=record.name,
        surface=surface,
        t1=t1,
        t3=t3,
        stations=result.stations,
        summary=summary,
        azimuth_deg=result.azimuth_deg,
        md_t1_m=result.md_t1_m,
        config=config,
        runtime_s=runtime_s,
        md_postcheck_exceeded=md_postcheck_exceeded,
        md_postcheck_message=md_postcheck_message,
    )
    row = WelltrackBatchPlanner._row_from_success(record=record, success=success)
    return row, success


class WelltrackBatchPlanner:
    def __init__(self, planner: TrajectoryPlanner | None = None):
        self._planner = planner or TrajectoryPlanner()
        self._last_evaluation_metadata = BatchEvaluationMetadata()

    @property
    def last_evaluation_metadata(self) -> BatchEvaluationMetadata:
        return self._last_evaluation_metadata

    def evaluate(
        self,
        records: Iterable[WelltrackRecord],
        selected_names: set[str],
        config: TrajectoryConfig,
        selected_order: list[str] | None = None,
        config_by_name: dict[str, TrajectoryConfig] | None = None,
        optimization_context_by_name: dict[str, AntiCollisionOptimizationContext] | None = None,
        dynamic_cluster_context: DynamicClusterExecutionContext | None = None,
        progress_callback: ProgressCallback | None = None,
        solver_progress_callback: SolverProgressCallback | None = None,
        record_done_callback: RecordDoneCallback | None = None,
        parallel_workers: int = 0,
    ) -> tuple[list[dict[str, Any]], list[SuccessfulWellPlan]]:
        selected_records = self._selected_records_in_order(
            records=records,
            selected_names=selected_names,
            selected_order=selected_order,
        )

        # ------------------------------------------------------------------
        # Parallel fast-path: when workers > 1 and no dynamic cluster
        # context (which requires iterative sequential execution), submit
        # all selected wells to a process pool.
        # ------------------------------------------------------------------
        if (
            int(parallel_workers) > 1
            and dynamic_cluster_context is None
            and len(selected_records) > 1
        ):
            return self._evaluate_parallel(
                selected_records=selected_records,
                config=config,
                config_by_name=config_by_name,
                optimization_context_by_name=optimization_context_by_name,
                progress_callback=progress_callback,
                record_done_callback=record_done_callback,
                parallel_workers=int(parallel_workers),
            )

        summary_rows: list[dict[str, Any]] = []
        successes: list[SuccessfulWellPlan] = []
        total = len(selected_records)
        total_planned_steps = int(total)
        selected_records_by_name = {
            str(record.name): record for record in selected_records
        }
        initial_success_by_name = {
            str(item.name): item
            for item in (
                dynamic_cluster_context.initial_successes
                if dynamic_cluster_context is not None
                else ()
            )
        }
        remaining_selected_names = [
            str(record.name) for record in selected_records
        ]
        recalculated_success_by_name: dict[str, SuccessfulWellPlan] = {}
        executed_well_names: list[str] = []
        skipped_selected_names: list[str] = []
        cluster_resolved_early = False
        cluster_blocked = False
        cluster_blocking_reason: str | None = None
        dynamic_cluster_pass_count = 1 if dynamic_cluster_context is not None else 0
        previous_cluster_score = self._initial_cluster_score(dynamic_cluster_context)
        dynamic_cluster_prefer_trajectory_stage = False

        while remaining_selected_names or dynamic_cluster_context is not None:
            if not remaining_selected_names:
                (
                    reseed_names,
                    reseed_reason,
                    reseed_resolved,
                    current_cluster_score,
                    next_prefer_trajectory_stage,
                ) = (
                    self._extend_dynamic_cluster_queue(
                        selected_records_by_name=selected_records_by_name,
                        dynamic_cluster_context=dynamic_cluster_context,
                        recalculated_success_by_name=recalculated_success_by_name,
                        previous_cluster_score=previous_cluster_score,
                        dynamic_cluster_pass_count=dynamic_cluster_pass_count,
                    )
                )
                dynamic_cluster_prefer_trajectory_stage = bool(
                    next_prefer_trajectory_stage
                )
                if current_cluster_score is not None:
                    previous_cluster_score = current_cluster_score
                if reseed_names:
                    remaining_selected_names.extend(reseed_names)
                    dynamic_cluster_pass_count += 1
                    total_planned_steps = max(
                        int(total_planned_steps),
                        int(len(executed_well_names) + len(remaining_selected_names)),
                    )
                else:
                    if reseed_resolved:
                        cluster_resolved_early = True
                    elif reseed_reason:
                        cluster_blocked = True
                        cluster_blocking_reason = str(reseed_reason)
                    break
            dynamic_cluster_plan, pruned_names = self._refresh_dynamic_cluster_plan(
                remaining_selected_names=remaining_selected_names,
                dynamic_cluster_context=dynamic_cluster_context,
                recalculated_success_by_name=recalculated_success_by_name,
                prefer_trajectory_stage=dynamic_cluster_prefer_trajectory_stage,
            )
            if pruned_names:
                skipped_selected_names.extend(
                    name for name in pruned_names if name not in skipped_selected_names
                )
                if dynamic_cluster_plan is not None:
                    resolution_state = str(dynamic_cluster_plan.resolution_state).strip()
                    if resolution_state == DYNAMIC_CLUSTER_PLAN_RESOLVED:
                        cluster_resolved_early = True
                    elif resolution_state == DYNAMIC_CLUSTER_PLAN_BLOCKED:
                        cluster_blocked = True
                        cluster_blocking_reason = (
                            str(dynamic_cluster_plan.blocking_reason).strip()
                            if dynamic_cluster_plan.blocking_reason is not None
                            else None
                        )
            if not remaining_selected_names:
                break
            index = len(executed_well_names) + 1
            total = max(int(total_planned_steps), int(index))
            record, runtime_override = self._next_record_for_evaluation(
                selected_records_by_name=selected_records_by_name,
                remaining_selected_names=remaining_selected_names,
                selected_order=selected_order,
                base_config=config,
                config_by_name=config_by_name,
                optimization_context_by_name=optimization_context_by_name,
                dynamic_cluster_plan=dynamic_cluster_plan,
                recalculated_success_by_name=recalculated_success_by_name,
            )
            remaining_selected_names.remove(str(record.name))
            executed_well_names.append(str(record.name))
            if progress_callback is not None:
                progress_callback(index, total, record.name)

            planner_progress_callback = None
            if solver_progress_callback is not None:
                index_i = int(index)
                total_i = int(total)
                name_i = str(record.name)

                def _planner_progress(stage_text: str, stage_fraction: float) -> None:
                    solver_progress_callback(
                        index_i,
                        total_i,
                        name_i,
                        stage_text,
                        stage_fraction,
                    )

                planner_progress_callback = _planner_progress

            row, success = self._evaluate_record(
                record=record,
                config=runtime_override["config"],
                optimization_context=runtime_override["optimization_context"],
                planner_progress_callback=planner_progress_callback,
            )
            if success is not None:
                optimization_context = runtime_override["optimization_context"]
                previous_success = recalculated_success_by_name.get(str(record.name))
                if previous_success is None:
                    previous_success = initial_success_by_name.get(str(record.name))
                retained_success = None
                if dynamic_cluster_context is not None:
                    current_success_by_name = dict(initial_success_by_name)
                    current_success_by_name.update(recalculated_success_by_name)
                    retained_success = self._select_cluster_monotonic_anticollision_success(
                        candidate_success=success,
                        existing_success=previous_success,
                        current_success_by_name=current_success_by_name,
                        dynamic_cluster_context=dynamic_cluster_context,
                    )
                    if (
                        retained_success is None
                        and not isinstance(
                            optimization_context,
                            AntiCollisionOptimizationContext,
                        )
                    ):
                        retained_success = self._select_monotonic_anticollision_success(
                            candidate_success=success,
                            existing_success=previous_success,
                            optimization_context=optimization_context,
                        )
                elif isinstance(optimization_context, AntiCollisionOptimizationContext):
                    retained_success = self._select_monotonic_anticollision_success(
                        candidate_success=success,
                        existing_success=previous_success,
                        optimization_context=optimization_context,
                    )
                if retained_success is not None:
                    retained_success = _success_with_attempted_anticollision_stage(
                        success=retained_success,
                        stage=(
                            str(success.summary.get("anti_collision_stage")).strip()
                            if success.summary.get("anti_collision_stage") is not None
                            else anti_collision_stage_from_context(optimization_context)
                        ),
                    )
                    success = retained_success
                    row = self._row_from_success(record=record, success=retained_success)
                if success is not None and previous_success is not None:
                    for attempted_stage in _normalized_attempted_anticollision_stages(
                        previous_success.summary
                    ):
                        success = _success_with_attempted_anticollision_stage(
                            success=success,
                            stage=attempted_stage,
                        )
            summary_rows.append(row)
            if success is not None:
                successes.append(success)
                recalculated_success_by_name[str(success.name)] = success
            if record_done_callback is not None:
                record_done_callback(index, total, record.name, row)

        self._last_evaluation_metadata = BatchEvaluationMetadata(
            executed_well_names=tuple(executed_well_names),
            skipped_selected_names=tuple(skipped_selected_names),
            cluster_resolved_early=bool(cluster_resolved_early),
            cluster_blocked=bool(cluster_blocked),
            cluster_blocking_reason=cluster_blocking_reason,
        )
        return summary_rows, successes

    def _evaluate_parallel(
        self,
        selected_records: list[WelltrackRecord],
        config: TrajectoryConfig,
        config_by_name: dict[str, TrajectoryConfig] | None,
        optimization_context_by_name: dict[str, AntiCollisionOptimizationContext] | None,
        progress_callback: ProgressCallback | None,
        record_done_callback: RecordDoneCallback | None,
        parallel_workers: int,
    ) -> tuple[list[dict[str, Any]], list[SuccessfulWellPlan]]:
        """Execute selected wells in parallel using a process pool."""
        _mp_method = "spawn" if sys.platform == "win32" else "forkserver"
        _mp_ctx = multiprocessing.get_context(_mp_method)
        total = len(selected_records)
        workers = min(int(parallel_workers), total)

        # Preserve submission order so results come back in the same order.
        ordered_names: list[str] = [str(r.name) for r in selected_records]
        future_to_name: dict[Future, str] = {}
        results_by_name: dict[str, tuple[dict[str, Any], SuccessfulWellPlan | None]] = {}

        pool = ProcessPoolExecutor(max_workers=workers, mp_context=_mp_ctx)
        try:
            for record in selected_records:
                well_config = (
                    (config_by_name or {}).get(str(record.name)) or config
                )
                opt_ctx = (
                    (optimization_context_by_name or {}).get(str(record.name))
                )
                fut = pool.submit(
                    _evaluate_record_from_dicts,
                    record.model_dump(),
                    well_config.model_dump(),
                    asdict(opt_ctx) if opt_ctx is not None else None,
                )
                future_to_name[fut] = str(record.name)

            completed_count = 0
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                completed_count += 1
                try:
                    row, success_dict = fut.result()
                    success = (
                        SuccessfulWellPlan.model_validate(success_dict)
                        if success_dict is not None
                        else None
                    )
                except Exception as exc:  # noqa: BLE001
                    row = self._base_row(
                        record=next(
                            r for r in selected_records if str(r.name) == name
                        )
                    )
                    row["Статус"] = "Ошибка расчета"
                    row["Проблема"] = summarize_problem_ru(str(exc))
                    success = None
                results_by_name[name] = (row, success)
                if progress_callback is not None:
                    progress_callback(completed_count, total, name)
                if record_done_callback is not None:
                    record_done_callback(completed_count, total, name, row)
        finally:
            pool.shutdown(wait=True)

        # Assemble results in original submission order.
        summary_rows: list[dict[str, Any]] = []
        successes: list[SuccessfulWellPlan] = []
        executed_well_names: list[str] = []
        for name in ordered_names:
            row, success = results_by_name.get(name, (
                self._base_row(
                    record=next(
                        r for r in selected_records if str(r.name) == name
                    )
                ),
                None,
            ))
            summary_rows.append(row)
            if success is not None:
                successes.append(success)
            executed_well_names.append(name)

        self._last_evaluation_metadata = BatchEvaluationMetadata(
            executed_well_names=tuple(executed_well_names),
            skipped_selected_names=(),
            cluster_resolved_early=False,
            cluster_blocked=False,
            cluster_blocking_reason=None,
        )
        return summary_rows, successes

    @staticmethod
    def _cluster_anticollision_score(
        *,
        success_by_name: Mapping[str, SuccessfulWellPlan],
        dynamic_cluster_context: DynamicClusterExecutionContext,
    ) -> tuple[float, float, int]:
        successes = list(success_by_name.values())
        if len(successes) < 2:
            return 1e6, 0.0, 0
        analysis = build_anti_collision_analysis_for_successes(
            successes,
            model=dynamic_cluster_context.uncertainty_model,
            reference_wells=dynamic_cluster_context.reference_wells,
            include_display_geometry=False,
            build_overlap_geometry=False,
        )
        recommendations = build_anti_collision_recommendations(
            analysis,
            well_context_by_name=build_anticollision_well_contexts(successes),
        )
        clusters = build_anti_collision_recommendation_clusters(recommendations)
        target_set = {
            str(name)
            for name in dynamic_cluster_context.target_well_names
            if str(name).strip()
        }
        relevant_clusters = [
            cluster
            for cluster in clusters
            if target_set.intersection(str(name) for name in cluster.well_names)
        ]
        if not relevant_clusters:
            return 1e6, 0.0, 0
        worst_sf = min(float(cluster.worst_separation_factor) for cluster in relevant_clusters)
        max_overlap = max(
            float(recommendation.max_overlap_depth_m)
            for cluster in relevant_clusters
            for recommendation in cluster.recommendations
        )
        recommendation_count = sum(
            int(cluster.recommendation_count) for cluster in relevant_clusters
        )
        return worst_sf, max_overlap, recommendation_count

    @staticmethod
    def _initial_cluster_score(
        dynamic_cluster_context: DynamicClusterExecutionContext | None,
    ) -> tuple[float, float, int] | None:
        if dynamic_cluster_context is None:
            return None
        success_by_name = {
            str(item.name): item for item in dynamic_cluster_context.initial_successes
        }
        if not success_by_name:
            return None
        return WelltrackBatchPlanner._cluster_anticollision_score(
            success_by_name=success_by_name,
            dynamic_cluster_context=dynamic_cluster_context,
        )

    @staticmethod
    def _cluster_score_improved(
        *,
        current_score: tuple[float, float, int] | None,
        previous_score: tuple[float, float, int] | None,
    ) -> bool:
        if current_score is None:
            return False
        if previous_score is None:
            return True
        current_sf, current_overlap, current_count = current_score
        previous_sf, previous_overlap, previous_count = previous_score
        sf_tolerance = 1e-3
        overlap_tolerance = 1e-3
        if current_sf > previous_sf + sf_tolerance:
            return True
        if (
            abs(current_sf - previous_sf) <= sf_tolerance
            and current_overlap < previous_overlap - overlap_tolerance
        ):
            return True
        if (
            abs(current_sf - previous_sf) <= sf_tolerance
            and abs(current_overlap - previous_overlap) <= overlap_tolerance
            and current_count < previous_count
        ):
            return True
        return False

    @staticmethod
    def _well_local_anticollision_score(
        *,
        success_by_name: Mapping[str, SuccessfulWellPlan],
        dynamic_cluster_context: DynamicClusterExecutionContext,
        target_well_name: str,
    ) -> tuple[float, float, int]:
        successes = list(success_by_name.values())
        if len(successes) < 2 or str(target_well_name) not in success_by_name:
            return 1e6, 0.0, 0
        analysis = build_anti_collision_analysis_for_successes(
            successes,
            model=dynamic_cluster_context.uncertainty_model,
            reference_wells=dynamic_cluster_context.reference_wells,
            include_display_geometry=False,
            build_overlap_geometry=False,
        )
        recommendations = build_anti_collision_recommendations(
            analysis,
            well_context_by_name=build_anticollision_well_contexts(successes),
        )
        target_set = {
            str(name)
            for name in dynamic_cluster_context.target_well_names
            if str(name).strip()
        }
        well_name = str(target_well_name)
        relevant_recommendations = [
            recommendation
            for recommendation in recommendations
            if well_name in {str(recommendation.well_a), str(recommendation.well_b)}
            and target_set.intersection(
                {str(recommendation.well_a), str(recommendation.well_b)}
            )
        ]
        if not relevant_recommendations:
            return 1e6, 0.0, 0
        worst_sf = min(
            float(recommendation.min_separation_factor)
            for recommendation in relevant_recommendations
        )
        max_overlap = max(
            float(recommendation.max_overlap_depth_m)
            for recommendation in relevant_recommendations
        )
        return worst_sf, max_overlap, len(relevant_recommendations)

    @staticmethod
    def _select_cluster_monotonic_anticollision_success(
        *,
        candidate_success: SuccessfulWellPlan,
        existing_success: SuccessfulWellPlan | None,
        current_success_by_name: Mapping[str, SuccessfulWellPlan],
        dynamic_cluster_context: DynamicClusterExecutionContext | None,
    ) -> SuccessfulWellPlan | None:
        if dynamic_cluster_context is None or existing_success is None:
            return None
        current_existing = dict(current_success_by_name)
        current_existing[str(existing_success.name)] = existing_success
        current_candidate = dict(current_success_by_name)
        current_candidate[str(candidate_success.name)] = candidate_success
        target_well_name = str(candidate_success.name)
        (
            existing_local_sf,
            existing_local_overlap,
            existing_local_count,
        ) = WelltrackBatchPlanner._well_local_anticollision_score(
            success_by_name=current_existing,
            dynamic_cluster_context=dynamic_cluster_context,
            target_well_name=target_well_name,
        )
        (
            candidate_local_sf,
            candidate_local_overlap,
            candidate_local_count,
        ) = WelltrackBatchPlanner._well_local_anticollision_score(
            success_by_name=current_candidate,
            dynamic_cluster_context=dynamic_cluster_context,
            target_well_name=target_well_name,
        )
        existing_sf, existing_overlap, existing_count = (
            WelltrackBatchPlanner._cluster_anticollision_score(
                success_by_name=current_existing,
                dynamic_cluster_context=dynamic_cluster_context,
            )
        )
        candidate_sf, candidate_overlap, candidate_count = (
            WelltrackBatchPlanner._cluster_anticollision_score(
                success_by_name=current_candidate,
                dynamic_cluster_context=dynamic_cluster_context,
            )
        )
        sf_tolerance = 1e-3
        overlap_tolerance = 1e-3
        if candidate_local_sf < existing_local_sf - sf_tolerance:
            return existing_success
        if (
            candidate_local_sf <= existing_local_sf + sf_tolerance
            and candidate_local_overlap > existing_local_overlap + overlap_tolerance
        ):
            return existing_success
        if (
            abs(candidate_local_sf - existing_local_sf) <= sf_tolerance
            and abs(candidate_local_overlap - existing_local_overlap)
            <= overlap_tolerance
            and candidate_local_count > existing_local_count
        ):
            return existing_success
        if candidate_sf < existing_sf - sf_tolerance:
            return existing_success
        if (
            candidate_sf <= existing_sf + sf_tolerance
            and candidate_overlap > existing_overlap + overlap_tolerance
        ):
            return existing_success
        if (
            abs(candidate_sf - existing_sf) <= sf_tolerance
            and abs(candidate_overlap - existing_overlap) <= overlap_tolerance
            and candidate_count > existing_count
        ):
            return existing_success
        return None

    @staticmethod
    def summary_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    @staticmethod
    def _summary_target_direction_label(value: object) -> str:
        text = str(value).strip()
        if not text or text == "—":
            return "—"
        text_lower = text.lower()
        if "обрат" in text_lower:
            return "В обратном направлении"
        return "В прямом направлении"

    @staticmethod
    def _base_row(record: WelltrackRecord) -> dict[str, Any]:
        return {
            "Скважина": record.name,
            "Точек": len(record.points),
            "Статус": "Не рассчитана",
            "Рестарты решателя": "—",
            "Модель траектории": "—",
            "Классификация целей": "—",
            "Сложность": "—",
            "Отход t1, м": "—",
            "KOP MD, м": "—",
            "Длина ГС, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс ПИ, deg/10m": "—",
            "Макс MD, м": "—",
            "Проблема": "",
        }

    @staticmethod
    def _selected_records_in_order(
        *,
        records: Iterable[WelltrackRecord],
        selected_names: set[str],
        selected_order: list[str] | None,
    ) -> list[WelltrackRecord]:
        ordered_records = list(records)
        selected_name_set = {str(name) for name in selected_names}
        by_name = {str(record.name): record for record in ordered_records}
        resolved: list[WelltrackRecord] = []
        seen: set[str] = set()
        for name in selected_order or ():
            well_name = str(name)
            if well_name not in selected_name_set or well_name in seen:
                continue
            record = by_name.get(well_name)
            if record is None:
                continue
            resolved.append(record)
            seen.add(well_name)
        for record in ordered_records:
            well_name = str(record.name)
            if well_name not in selected_name_set or well_name in seen:
                continue
            resolved.append(record)
            seen.add(well_name)
        return resolved

    def _next_record_for_evaluation(
        self,
        *,
        selected_records_by_name: dict[str, WelltrackRecord],
        remaining_selected_names: list[str],
        selected_order: list[str] | None,
        base_config: TrajectoryConfig,
        config_by_name: dict[str, TrajectoryConfig] | None,
        optimization_context_by_name: dict[str, AntiCollisionOptimizationContext] | None,
        dynamic_cluster_plan: DynamicClusterExecutionPlan | None,
        recalculated_success_by_name: dict[str, SuccessfulWellPlan],
    ) -> tuple[WelltrackRecord, dict[str, object]]:
        runtime_override = self._runtime_override_for_next_record(
            base_config=base_config,
            config_by_name=config_by_name,
            optimization_context_by_name=optimization_context_by_name,
            dynamic_cluster_plan=dynamic_cluster_plan,
            recalculated_success_by_name=recalculated_success_by_name,
        )
        if runtime_override is not None:
            record_name = str(runtime_override["well_name"])
            record = selected_records_by_name[record_name]
            return record, runtime_override
        ordered_names = [
            str(name)
            for name in (selected_order or remaining_selected_names)
            if str(name) in set(remaining_selected_names)
        ]
        next_name = ordered_names[0] if ordered_names else str(remaining_selected_names[0])
        record = selected_records_by_name[next_name]
        context = self._resolve_optimization_context(
            context=(optimization_context_by_name or {}).get(str(record.name)),
            recalculated_success_by_name=recalculated_success_by_name,
        )
        config = (config_by_name or {}).get(str(record.name), base_config)
        return record, {
            "well_name": str(record.name),
            "config": config,
            "optimization_context": context,
        }

    def _runtime_override_for_next_record(
        self,
        *,
        base_config: TrajectoryConfig,
        config_by_name: dict[str, TrajectoryConfig] | None,
        optimization_context_by_name: dict[str, AntiCollisionOptimizationContext] | None,
        dynamic_cluster_plan: DynamicClusterExecutionPlan | None,
        recalculated_success_by_name: dict[str, SuccessfulWellPlan],
    ) -> dict[str, object] | None:
        if (
            dynamic_cluster_plan is None
            or str(dynamic_cluster_plan.resolution_state).strip()
            != DYNAMIC_CLUSTER_PLAN_ACTIVE
            or not dynamic_cluster_plan.ordered_well_names
        ):
            return None
        well_name = str(dynamic_cluster_plan.ordered_well_names[0])
        payload = dict(dynamic_cluster_plan.prepared_by_well.get(well_name, {}))
        update_fields = dict(payload.get("update_fields", {}))
        config = (
            base_config.validated_copy(**update_fields)
            if update_fields
            else (config_by_name or {}).get(well_name, base_config)
        )
        context = payload.get("optimization_context")
        if isinstance(context, AntiCollisionOptimizationContext):
            context = self._resolve_optimization_context(
                context=context,
                recalculated_success_by_name=recalculated_success_by_name,
            )
        elif context is None:
            context = self._resolve_optimization_context(
                context=(optimization_context_by_name or {}).get(well_name),
                recalculated_success_by_name=recalculated_success_by_name,
            )
        return {
            "well_name": well_name,
            "config": config,
            "optimization_context": context,
            "cluster_plan": dynamic_cluster_plan,
        }

    @staticmethod
    def _refresh_dynamic_cluster_plan(
        *,
        remaining_selected_names: list[str],
        dynamic_cluster_context: DynamicClusterExecutionContext | None,
        recalculated_success_by_name: dict[str, SuccessfulWellPlan],
        prefer_trajectory_stage: bool = False,
    ) -> tuple[DynamicClusterExecutionPlan | None, tuple[str, ...]]:
        if dynamic_cluster_context is None or not remaining_selected_names:
            return None, ()
        cluster_scoped_remaining = tuple(
            str(name)
            for name in remaining_selected_names
            if str(name) in set(dynamic_cluster_context.target_well_names)
        )
        if not cluster_scoped_remaining:
            return None, ()
        current_success_by_name = {
            str(item.name): item for item in dynamic_cluster_context.initial_successes
        }
        current_success_by_name.update(recalculated_success_by_name)
        plan = build_dynamic_cluster_execution_plan(
            successes=list(current_success_by_name.values()),
            selected_names=set(str(name) for name in remaining_selected_names),
            target_well_names=dynamic_cluster_context.target_well_names,
            uncertainty_model=dynamic_cluster_context.uncertainty_model,
            reference_wells=dynamic_cluster_context.reference_wells,
            prefer_trajectory_stage=bool(prefer_trajectory_stage),
        )
        if plan is None:
            blocking_reason = (
                "Iterative cluster-aware execution не нашел актуальных шагов "
                "для оставшихся скважин. Текущий прогон остановлен, чтобы не "
                "пересчитывать их базовыми настройками."
            )
            fallback_plan = DynamicClusterExecutionPlan(
                cluster=None,
                ordered_well_names=(),
                prepared_by_well={},
                skipped_wells=cluster_scoped_remaining,
                resolution_state=DYNAMIC_CLUSTER_PLAN_BLOCKED,
                blocking_reason=blocking_reason,
            )
            for name in cluster_scoped_remaining:
                remaining_selected_names.remove(str(name))
            return fallback_plan, cluster_scoped_remaining
        resolution_state = str(plan.resolution_state).strip()
        if resolution_state in {
            DYNAMIC_CLUSTER_PLAN_RESOLVED,
            DYNAMIC_CLUSTER_PLAN_BLOCKED,
        }:
            for name in cluster_scoped_remaining:
                remaining_selected_names.remove(str(name))
            return plan, cluster_scoped_remaining
        return plan, ()

    @staticmethod
    def _extend_dynamic_cluster_queue(
        *,
        selected_records_by_name: Mapping[str, WelltrackRecord],
        dynamic_cluster_context: DynamicClusterExecutionContext | None,
        recalculated_success_by_name: Mapping[str, SuccessfulWellPlan],
        previous_cluster_score: tuple[float, float, int] | None,
        dynamic_cluster_pass_count: int,
    ) -> tuple[
        tuple[str, ...],
        str | None,
        bool,
        tuple[float, float, int] | None,
        bool,
    ]:
        if dynamic_cluster_context is None:
            return (), None, False, previous_cluster_score, False
        if dynamic_cluster_pass_count >= _MAX_DYNAMIC_CLUSTER_PASSES:
            return (
                (),
                (
                    "Cluster-level anti-collision пересчет достиг лимита проходов "
                    f"({_MAX_DYNAMIC_CLUSTER_PASSES}). Нужен новый запуск или ручная корректировка."
                ),
                False,
                previous_cluster_score,
                False,
            )
        current_success_by_name = {
            str(item.name): item for item in dynamic_cluster_context.initial_successes
        }
        current_success_by_name.update(recalculated_success_by_name)
        current_cluster_score = WelltrackBatchPlanner._cluster_anticollision_score(
            success_by_name=current_success_by_name,
            dynamic_cluster_context=dynamic_cluster_context,
        )
        improved_since_previous = WelltrackBatchPlanner._cluster_score_improved(
            current_score=current_cluster_score,
            previous_score=previous_cluster_score,
        )
        prefer_trajectory_stage = bool(dynamic_cluster_pass_count >= 1)
        plan = build_dynamic_cluster_execution_plan(
            successes=list(current_success_by_name.values()),
            selected_names=set(selected_records_by_name),
            target_well_names=dynamic_cluster_context.target_well_names,
            uncertainty_model=dynamic_cluster_context.uncertainty_model,
            reference_wells=dynamic_cluster_context.reference_wells,
            prefer_trajectory_stage=prefer_trajectory_stage,
        )
        if plan is None:
            return (), None, False, current_cluster_score, prefer_trajectory_stage
        if (
            dynamic_cluster_pass_count > 0
            and not improved_since_previous
            and not (
                prefer_trajectory_stage
                and str(plan.resolution_state).strip() == DYNAMIC_CLUSTER_PLAN_ACTIVE
                and bool(plan.ordered_well_names)
            )
        ):
            return (
                (),
                (
                    "Следующий cluster-level проход не дал заметного улучшения SF/overlap. "
                    "Автоматический пересчет остановлен."
                ),
                False,
                current_cluster_score,
                prefer_trajectory_stage,
            )
        resolution_state = str(plan.resolution_state).strip()
        if resolution_state == DYNAMIC_CLUSTER_PLAN_RESOLVED:
            return (), None, True, current_cluster_score, prefer_trajectory_stage
        if resolution_state == DYNAMIC_CLUSTER_PLAN_BLOCKED:
            return (
                (),
                str(plan.blocking_reason).strip() if plan.blocking_reason is not None else None,
                False,
                current_cluster_score,
                prefer_trajectory_stage,
            )
        pending_names = tuple(
            str(name)
            for name in plan.ordered_well_names
            if str(name) in selected_records_by_name
        )
        if not pending_names:
            return (
                (),
                "Для следующего cluster-level прохода не осталось скважин в текущем наборе выбора.",
                False,
                current_cluster_score,
                prefer_trajectory_stage,
            )
        return pending_names, None, False, current_cluster_score, prefer_trajectory_stage

    @staticmethod
    def _resolve_optimization_context(
        *,
        context: AntiCollisionOptimizationContext | None,
        recalculated_success_by_name: dict[str, SuccessfulWellPlan],
    ) -> AntiCollisionOptimizationContext | None:
        return rebuild_optimization_context(
            context=context,
            reference_success_by_name=recalculated_success_by_name,
            strict_missing_references=False,
        )

    @staticmethod
    def _row_from_success(
        *,
        record: WelltrackRecord,
        success: SuccessfulWellPlan,
    ) -> dict[str, Any]:
        summary = dict(success.summary)
        md_total_m = float(summary.get("md_total_m", 0.0))
        t1_offset = float(
            np.hypot(success.t1.x - success.surface.x, success.t1.y - success.surface.y)
        )
        row = WelltrackBatchPlanner._base_row(record=record)
        row.update(
            {
                "Статус": "OK",
                "Рестарты решателя": str(
                    int(float(summary.get("solver_turn_restarts_used", 0.0)))
                ),
                "Модель траектории": str(summary.get("trajectory_type", "—")),
                "Классификация целей": WelltrackBatchPlanner._summary_target_direction_label(
                    summary.get("trajectory_target_direction", "—")
                ),
                "Сложность": str(summary.get("well_complexity", "—")),
                "Отход t1, м": f"{t1_offset:.2f}",
                "Мин VERTICAL до KOP, м": f"{float(success.config.kop_min_vertical_m):.2f}",
                "KOP MD, м": f"{float(summary.get('kop_md_m', 0.0)):.2f}",
                "Длина ГС, м": (
                    f"{float(summary.get('horizontal_length_m', 0.0)):.2f}"
                ),
                "INC в t1, deg": f"{float(summary.get('entry_inc_deg', 0.0)):.2f}",
                "ЗУ HOLD, deg": f"{float(summary.get('hold_inc_deg', 0.0)):.2f}",
                "Макс ПИ, deg/10m": (
                    f"{dls_to_pi(float(summary.get('max_dls_total_deg_per_30m', 0.0))):.2f}"
                ),
                "Макс MD, м": f"{md_total_m:.2f}",
                "Проблема": str(success.md_postcheck_message or ""),
            }
        )
        return row

    @staticmethod
    def _select_monotonic_anticollision_success(
        *,
        candidate_success: SuccessfulWellPlan,
        existing_success: SuccessfulWellPlan | None,
        optimization_context: object,
    ) -> SuccessfulWellPlan | None:
        if existing_success is None:
            return None
        if not isinstance(optimization_context, AntiCollisionOptimizationContext):
            return None
        if str(candidate_success.config.optimization_mode).strip() != "anti_collision_avoidance":
            return None
        try:
            existing_clearance = evaluate_stations_anti_collision_clearance(
                stations=existing_success.stations,
                context=optimization_context,
            )
            candidate_clearance = evaluate_stations_anti_collision_clearance(
                stations=candidate_success.stations,
                context=optimization_context,
            )
        except ValueError:
            return None

        sf_tolerance = 1e-3
        overlap_tolerance = 1e-3
        existing_sf = float(existing_clearance.min_separation_factor)
        candidate_sf = float(candidate_clearance.min_separation_factor)
        existing_overlap = float(existing_clearance.max_overlap_depth_m)
        candidate_overlap = float(candidate_clearance.max_overlap_depth_m)

        if candidate_sf < existing_sf - sf_tolerance:
            return existing_success
        if candidate_sf <= existing_sf + sf_tolerance and (
            candidate_overlap > existing_overlap + overlap_tolerance
        ):
            return existing_success
        if (
            abs(candidate_sf - existing_sf) <= sf_tolerance
            and abs(candidate_overlap - existing_overlap) <= overlap_tolerance
        ):
            candidate_md = float(candidate_success.summary.get("md_total_m", 0.0))
            existing_md = float(existing_success.summary.get("md_total_m", 0.0))
            if candidate_md > existing_md + 1e-6:
                return existing_success
        return None

    def _evaluate_record(
        self,
        record: WelltrackRecord,
        config: TrajectoryConfig,
        optimization_context: AntiCollisionOptimizationContext | None = None,
        planner_progress_callback: Callable[[str, float], None] | None = None,
    ) -> tuple[dict[str, Any], SuccessfulWellPlan | None]:
        row = self._base_row(record=record)
        if len(record.points) != 3:
            row["Статус"] = "Ошибка формата"
            row["Проблема"] = f"Ожидалось 3 точки (S, t1, t3), получено {len(record.points)}."
            return row, None

        try:
            surface, t1, t3 = welltrack_points_to_targets(record.points)
            started = perf_counter()
            plan_kwargs: dict[str, Any] = {
                "surface": surface,
                "t1": t1,
                "t3": t3,
                "config": config,
                "progress_callback": planner_progress_callback,
            }
            if optimization_context is not None:
                plan_kwargs["optimization_context"] = optimization_context
            result = self._planner.plan(**plan_kwargs)
            runtime_s = float(perf_counter() - started)
        except (ValueError, PlanningError) as exc:
            row["Статус"] = "Ошибка расчета"
            row["Проблема"] = summarize_problem_ru(str(exc))
            return row, None
        summary = dict(result.summary)
        anti_collision_stage = anti_collision_stage_from_context(optimization_context)
        if anti_collision_stage is not None:
            summary["anti_collision_stage"] = anti_collision_stage
            summary["anti_collision_attempted_stages"] = anti_collision_stage
        md_total_m = float(summary.get("md_total_m", 0.0))
        md_limit_m = float(summary.get("max_total_md_postcheck_m", 0.0))
        md_postcheck_excess_m = float(summary.get("md_postcheck_excess_m", 0.0))
        md_postcheck_exceeded = bool(md_postcheck_excess_m > 1e-6)
        md_postcheck_message = ""
        if md_postcheck_exceeded:
            md_postcheck_message = (
                "Превышен лимит итоговой MD (постпроверка): "
                f"{md_total_m:.2f} м > {md_limit_m:.2f} м (+{md_postcheck_excess_m:.2f} м)."
            )

        success = SuccessfulWellPlan(
            name=record.name,
            surface=surface,
            t1=t1,
            t3=t3,
            stations=result.stations,
            summary=summary,
            azimuth_deg=result.azimuth_deg,
            md_t1_m=result.md_t1_m,
            config=config,
            runtime_s=runtime_s,
            md_postcheck_exceeded=md_postcheck_exceeded,
            md_postcheck_message=md_postcheck_message,
        )
        row = self._row_from_success(record=record, success=success)
        return row, success


def merge_batch_results(
    *,
    records: Iterable[WelltrackRecord],
    existing_rows: Iterable[dict[str, Any]] | None,
    existing_successes: Iterable[SuccessfulWellPlan] | None,
    new_rows: Iterable[dict[str, Any]],
    new_successes: Iterable[SuccessfulWellPlan],
) -> tuple[list[dict[str, Any]], list[SuccessfulWellPlan]]:
    """Merge a partial batch run with previously stored results.

    Rows are always returned in WELLTRACK order. Missing wells are represented by
    a "Не рассчитана" base row so the page can show a stable full-table view even
    after running only a subset of wells.
    """

    ordered_records = list(records)
    ordered_names = [str(record.name) for record in ordered_records]
    ordered_name_set = set(ordered_names)

    rows_by_name: dict[str, dict[str, Any]] = {
        str(record.name): WelltrackBatchPlanner._base_row(record)
        for record in ordered_records
    }
    for row in existing_rows or ():
        name = str(row.get("Скважина", "")).strip()
        if name in ordered_name_set:
            rows_by_name[name] = dict(row)
    rerun_names: set[str] = set()
    for row in new_rows:
        name = str(row.get("Скважина", "")).strip()
        if name in ordered_name_set:
            rows_by_name[name] = dict(row)
            rerun_names.add(name)

    successes_by_name: dict[str, SuccessfulWellPlan] = {}
    for success in existing_successes or ():
        name = str(success.name)
        if name in ordered_name_set:
            successes_by_name[name] = success
    for name in rerun_names:
        successes_by_name.pop(name, None)
    for success in new_successes:
        name = str(success.name)
        if name in ordered_name_set:
            successes_by_name[name] = success

    merged_rows = [rows_by_name[name] for name in ordered_names]
    merged_successes = [
        successes_by_name[name]
        for name in ordered_names
        if name in successes_by_name
    ]
    return merged_rows, merged_successes


def recommended_batch_selection(
    *,
    records: Iterable[WelltrackRecord],
    summary_rows: Iterable[dict[str, Any]] | None,
) -> list[str]:
    """Return wells that still need user attention.

    Before the first run every well is recommended. After that only not-yet-run,
    failed or warning-bearing wells stay preselected.
    """

    ordered_names = [str(record.name) for record in records]
    if summary_rows is None:
        return ordered_names

    rows_by_name = {
        str(row.get("Скважина", "")).strip(): row for row in summary_rows
    }
    recommended: list[str] = []
    for name in ordered_names:
        row = rows_by_name.get(name)
        if row is None:
            recommended.append(name)
            continue
        status = str(row.get("Статус", "")).strip()
        problem_text = str(row.get("Проблема", "")).strip()
        if status != "OK" or problem_text:
            recommended.append(name)
    return recommended
