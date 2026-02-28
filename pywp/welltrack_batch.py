from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pywp.eclipse_welltrack import WelltrackRecord, welltrack_points_to_targets
from pywp.models import Point3D, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner


@dataclass(frozen=True)
class SuccessfulWellPlan:
    name: str
    surface: Point3D
    t1: Point3D
    t3: Point3D
    stations: pd.DataFrame
    summary: dict[str, float | str]
    azimuth_deg: float
    md_t1_m: float
    config: TrajectoryConfig


class WelltrackBatchPlanner:
    def __init__(self, planner: TrajectoryPlanner | None = None):
        self._planner = planner or TrajectoryPlanner()

    def evaluate(
        self,
        records: Iterable[WelltrackRecord],
        selected_names: set[str],
        config: TrajectoryConfig,
    ) -> tuple[list[dict[str, Any]], list[SuccessfulWellPlan]]:
        summary_rows: list[dict[str, Any]] = []
        successes: list[SuccessfulWellPlan] = []

        for record in records:
            if record.name not in selected_names:
                continue

            row = self._base_row(record=record)
            if len(record.points) != 3:
                row["Статус"] = "Ошибка формата"
                row["Проблема"] = f"Ожидалось 3 точки (S, t1, t3), получено {len(record.points)}."
                summary_rows.append(row)
                continue

            try:
                surface, t1, t3 = welltrack_points_to_targets(record.points)
                result = self._planner.plan(surface=surface, t1=t1, t3=t3, config=config)
            except (ValueError, PlanningError) as exc:
                row["Статус"] = "Ошибка расчета"
                row["Проблема"] = str(exc)
                summary_rows.append(row)
                continue

            t1_offset = float(np.hypot(t1.x - surface.x, t1.y - surface.y))
            summary = result.summary
            row.update(
                {
                    "Статус": "OK",
                    "Тип траектории": str(summary.get("trajectory_type", "—")),
                    "Сложность": str(summary.get("well_complexity", "—")),
                    "Горизонтальный отход t1, м": f"{t1_offset:.2f}",
                    "INC в t1, deg": f"{float(summary.get('entry_inc_deg', 0.0)):.2f}",
                    "ЗУ HOLD, deg": f"{float(summary.get('hold_inc_deg', 0.0)):.2f}",
                    "Макс DLS, deg/30m": f"{float(summary.get('max_dls_total_deg_per_30m', 0.0)):.2f}",
                    "Проблема": "",
                }
            )
            summary_rows.append(row)
            successes.append(
                SuccessfulWellPlan(
                    name=record.name,
                    surface=surface,
                    t1=t1,
                    t3=t3,
                    stations=result.stations,
                    summary=result.summary,
                    azimuth_deg=result.azimuth_deg,
                    md_t1_m=result.md_t1_m,
                    config=config,
                )
            )

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
            "Тип траектории": "—",
            "Сложность": "—",
            "Горизонтальный отход t1, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс DLS, deg/30m": "—",
            "Проблема": "",
        }
