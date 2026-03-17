from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping
from time import perf_counter

import numpy as np
import pandas as pd
from pydantic import field_validator

from pywp.eclipse_welltrack import WelltrackRecord, welltrack_points_to_targets
from pywp.models import Point3D, SummaryDict, TrajectoryConfig
from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
)
from pywp.optimization_reference import compute_unoptimized_reference
from pywp.planner import PlanningError, TrajectoryPlanner
from pywp.pydantic_base import FrozenArbitraryModel, coerce_model_like
from pywp.solver_diagnostics import summarize_problem_ru
from pywp.ui_utils import dls_to_pi

ProgressCallback = Callable[[int, int, str], None]
SolverProgressCallback = Callable[[int, int, str, str, float], None]
RecordDoneCallback = Callable[[int, int, str, dict[str, Any]], None]


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
    baseline_summary: SummaryDict | None = None
    baseline_runtime_s: float | None = None
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
    )


def ensure_successful_plan_baseline(
    *,
    success: SuccessfulWellPlan,
    planner: TrajectoryPlanner | None = None,
) -> SuccessfulWellPlan:
    if success.baseline_summary is not None:
        return success
    if str(success.config.optimization_mode) == "none":
        return success
    reference = compute_unoptimized_reference(
        planner=planner or TrajectoryPlanner(),
        surface=success.surface,
        t1=success.t1,
        t3=success.t3,
        config=success.config,
    )
    if reference is None:
        return success
    return success.validated_copy(
        baseline_summary=reference.summary,
        baseline_runtime_s=reference.runtime_s,
    )


class WelltrackBatchPlanner:
    def __init__(self, planner: TrajectoryPlanner | None = None):
        self._planner = planner or TrajectoryPlanner()

    def evaluate(
        self,
        records: Iterable[WelltrackRecord],
        selected_names: set[str],
        config: TrajectoryConfig,
        selected_order: list[str] | None = None,
        config_by_name: dict[str, TrajectoryConfig] | None = None,
        optimization_context_by_name: dict[str, AntiCollisionOptimizationContext] | None = None,
        progress_callback: ProgressCallback | None = None,
        solver_progress_callback: SolverProgressCallback | None = None,
        record_done_callback: RecordDoneCallback | None = None,
    ) -> tuple[list[dict[str, Any]], list[SuccessfulWellPlan]]:
        summary_rows: list[dict[str, Any]] = []
        successes: list[SuccessfulWellPlan] = []
        selected_records = self._selected_records_in_order(
            records=records,
            selected_names=selected_names,
            selected_order=selected_order,
        )
        total = len(selected_records)
        recalculated_success_by_name: dict[str, SuccessfulWellPlan] = {}

        for index, record in enumerate(selected_records, start=1):
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
                config=(config_by_name or {}).get(str(record.name), config),
                optimization_context=self._resolve_optimization_context(
                    context=(optimization_context_by_name or {}).get(str(record.name)),
                    recalculated_success_by_name=recalculated_success_by_name,
                ),
                planner_progress_callback=planner_progress_callback,
            )
            summary_rows.append(row)
            if success is not None:
                successes.append(success)
                recalculated_success_by_name[str(success.name)] = success
            if record_done_callback is not None:
                record_done_callback(index, total, record.name, row)

        return summary_rows, successes

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
            "Горизонтальный отход t1, м": "—",
            "KOP MD, м": "—",
            "Длина HORIZONTAL, м": "—",
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
        t1_offset = float(np.hypot(t1.x - surface.x, t1.y - surface.y))
        summary = result.summary
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

        row.update(
            {
                "Статус": "OK",
                "Рестарты решателя": str(
                    int(float(summary.get("solver_turn_restarts_used", 0.0)))
                ),
                "Модель траектории": str(summary.get("trajectory_type", "—")),
                "Классификация целей": self._summary_target_direction_label(
                    summary.get("trajectory_target_direction", "—")
                ),
                "Сложность": str(summary.get("well_complexity", "—")),
                "Горизонтальный отход t1, м": f"{t1_offset:.2f}",
                "KOP MD, м": f"{float(summary.get('kop_md_m', 0.0)):.2f}",
                "Длина HORIZONTAL, м": f"{float(summary.get('horizontal_length_m', 0.0)):.2f}",
                "INC в t1, deg": f"{float(summary.get('entry_inc_deg', 0.0)):.2f}",
                "ЗУ HOLD, deg": f"{float(summary.get('hold_inc_deg', 0.0)):.2f}",
                "Макс ПИ, deg/10m": f"{dls_to_pi(float(summary.get('max_dls_total_deg_per_30m', 0.0))):.2f}",
                "Макс MD, м": f"{md_total_m:.2f}",
                "Проблема": md_postcheck_message,
            }
        )
        success = SuccessfulWellPlan(
            name=record.name,
            surface=surface,
            t1=t1,
            t3=t3,
            stations=result.stations,
            summary=result.summary,
            azimuth_deg=result.azimuth_deg,
            md_t1_m=result.md_t1_m,
            config=config,
            runtime_s=runtime_s,
            baseline_summary=None,
            baseline_runtime_s=None,
            md_postcheck_exceeded=md_postcheck_exceeded,
            md_postcheck_message=md_postcheck_message,
        )
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
