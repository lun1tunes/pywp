from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from pywp.eclipse_welltrack import WelltrackRecord, welltrack_points_to_targets
from pywp.models import Point3D, SummaryDict, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner
from pywp.pydantic_base import FrozenArbitraryModel
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
    md_postcheck_exceeded: bool = False
    md_postcheck_message: str = ""


class WelltrackBatchPlanner:
    def __init__(self, planner: TrajectoryPlanner | None = None):
        self._planner = planner or TrajectoryPlanner()

    def evaluate(
        self,
        records: Iterable[WelltrackRecord],
        selected_names: set[str],
        config: TrajectoryConfig,
        progress_callback: ProgressCallback | None = None,
        solver_progress_callback: SolverProgressCallback | None = None,
        record_done_callback: RecordDoneCallback | None = None,
    ) -> tuple[list[dict[str, Any]], list[SuccessfulWellPlan]]:
        summary_rows: list[dict[str, Any]] = []
        successes: list[SuccessfulWellPlan] = []
        selected_records = [record for record in records if record.name in selected_names]
        total = len(selected_records)

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
                config=config,
                planner_progress_callback=planner_progress_callback,
            )
            summary_rows.append(row)
            if success is not None:
                successes.append(success)
            if record_done_callback is not None:
                record_done_callback(index, total, record.name, row)

        return summary_rows, successes

    @staticmethod
    def summary_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    @staticmethod
    def _base_row(record: WelltrackRecord) -> dict[str, Any]:
        return {
            "Скважина": record.name,
            "Точек": len(record.points),
            "Статус": "Не рассчитана",
            "Модель траектории": "—",
            "Классификация целей": "—",
            "Сложность": "—",
            "Горизонтальный отход t1, м": "—",
            "Длина HORIZONTAL, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс ПИ, deg/10m": "—",
            "Макс MD, м": "—",
            "Проблема": "",
        }

    def _evaluate_record(
        self,
        record: WelltrackRecord,
        config: TrajectoryConfig,
        planner_progress_callback: Callable[[str, float], None] | None = None,
    ) -> tuple[dict[str, Any], SuccessfulWellPlan | None]:
        row = self._base_row(record=record)
        if len(record.points) != 3:
            row["Статус"] = "Ошибка формата"
            row["Проблема"] = f"Ожидалось 3 точки (S, t1, t3), получено {len(record.points)}."
            return row, None

        try:
            surface, t1, t3 = welltrack_points_to_targets(record.points)
            result = self._planner.plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                progress_callback=planner_progress_callback,
            )
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
                "Модель траектории": str(summary.get("trajectory_type", "—")),
                "Классификация целей": str(summary.get("trajectory_target_direction", "—")),
                "Сложность": str(summary.get("well_complexity", "—")),
                "Горизонтальный отход t1, м": f"{t1_offset:.2f}",
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
            md_postcheck_exceeded=md_postcheck_exceeded,
            md_postcheck_message=md_postcheck_message,
        )
        return row, success
