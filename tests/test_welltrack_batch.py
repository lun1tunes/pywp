from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import PlannerResult, TrajectoryConfig
from pywp.welltrack_batch import (
    SuccessfulWellPlan,
    WelltrackBatchPlanner,
    merge_batch_results,
    recommended_batch_selection,
)


def _fast_batch_config(**overrides: Any) -> TrajectoryConfig:
    base: dict[str, Any] = {
        "md_step_m": 10.0,
        "md_step_control_m": 2.0,
        "pos_tolerance_m": 2.0,
        "turn_solver_max_restarts": 0,
    }
    base.update(overrides)
    return TrajectoryConfig(**base)


class _StubPlanner:
    def plan(
        self,
        *,
        surface: Any,
        t1: Any,
        t3: Any,
        config: TrajectoryConfig,
        progress_callback: Any = None,
    ) -> PlannerResult:
        if progress_callback is not None:
            progress_callback("Планировщик: подготовка", 0.0)
            progress_callback("Планировщик: завершение", 1.0)

        stations = pd.DataFrame(
            {
                "MD_m": [0.0, 1000.0, 2000.0],
                "X_m": [surface.x, t1.x, t3.x],
                "Y_m": [surface.y, t1.y, t3.y],
                "Z_m": [surface.z, t1.z, t3.z],
                "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
            }
        )
        summary: dict[str, float | str] = {
            "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
            "trajectory_target_direction": "Цели в одном направлении",
            "well_complexity": "Обычная",
            "horizontal_length_m": 1000.0,
            "entry_inc_deg": 86.0,
            "hold_inc_deg": 20.0,
            "max_dls_total_deg_per_30m": 3.0,
            "md_total_m": 2200.0,
            "max_total_md_postcheck_m": float(config.max_total_md_postcheck_m),
            "md_postcheck_excess_m": 0.0,
        }
        return PlannerResult(stations=stations, summary=summary, azimuth_deg=0.0, md_t1_m=1000.0)


@pytest.mark.integration
def test_batch_planner_evaluates_success_and_format_error() -> None:
    records = [
        WelltrackRecord(
            name="OK-1",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="BAD-2",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=100.0, z=1000.0, md=1000.0),
            ),
        ),
    ]

    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={"OK-1", "BAD-2"},
        config=_fast_batch_config(),
    )

    assert len(rows) == 2
    by_name = {row["Скважина"]: row for row in rows}
    assert by_name["OK-1"]["Статус"] == "OK"
    assert by_name["BAD-2"]["Статус"] == "Ошибка формата"
    assert len(successes) == 1
    assert successes[0].name == "OK-1"


def test_batch_planner_reports_progress_callback_for_selected_wells() -> None:
    records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="SKIP-ME",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=100.0, z=1000.0, md=1000.0),
                WelltrackPoint(x=110.0, y=110.0, z=1010.0, md=1015.0),
            ),
        ),
    ]
    seen: list[tuple[int, int, str]] = []

    def on_progress(index: int, total: int, name: str) -> None:
        seen.append((index, total, name))

    rows, successes = WelltrackBatchPlanner(planner=_StubPlanner()).evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        config=_fast_batch_config(),
        progress_callback=on_progress,
    )

    assert len(rows) == 2
    assert len(successes) == 2
    assert seen == [(1, 2, "WELL-A"), (2, 2, "WELL-B")]


def test_batch_planner_reports_solver_stages_and_record_done_callbacks() -> None:
    records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
    ]
    solver_seen: list[tuple[int, int, str, str, float]] = []
    done_seen: list[tuple[int, int, str, str]] = []

    def on_solver(
        index: int, total: int, name: str, stage_text: str, stage_fraction: float
    ) -> None:
        solver_seen.append((index, total, name, stage_text, stage_fraction))

    def on_done(index: int, total: int, name: str, row: dict[str, object]) -> None:
        done_seen.append((index, total, name, str(row.get("Статус", "—"))))

    rows, successes = WelltrackBatchPlanner(planner=_StubPlanner()).evaluate(
        records=records,
        selected_names={"WELL-A"},
        config=_fast_batch_config(),
        solver_progress_callback=on_solver,
        record_done_callback=on_done,
    )

    assert len(rows) == 1
    assert len(successes) == 1
    assert done_seen == [(1, 1, "WELL-A", "OK")]
    assert solver_seen
    fractions = [item[4] for item in solver_seen]
    assert min(fractions) >= 0.0
    assert max(fractions) <= 1.0
    stage_names = [item[3] for item in solver_seen]
    assert any("Планировщик" in stage for stage in stage_names)


@pytest.mark.integration
def test_batch_planner_real_solver_emits_solver_progress_and_done_callbacks() -> None:
    records = [
        WelltrackRecord(
            name="REAL-CB-1",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
    ]
    solver_seen: list[tuple[int, int, str, str, float]] = []
    done_seen: list[tuple[int, int, str, str]] = []

    def on_solver(
        index: int, total: int, name: str, stage_text: str, stage_fraction: float
    ) -> None:
        solver_seen.append((index, total, name, stage_text, stage_fraction))

    def on_done(index: int, total: int, name: str, row: dict[str, object]) -> None:
        done_seen.append((index, total, name, str(row.get("Статус", "—"))))

    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={"REAL-CB-1"},
        config=_fast_batch_config(),
        solver_progress_callback=on_solver,
        record_done_callback=on_done,
    )

    assert len(rows) == 1
    assert len(successes) == 1
    assert rows[0]["Статус"] == "OK"
    assert solver_seen
    assert done_seen == [(1, 1, "REAL-CB-1", "OK")]
    fractions = [item[4] for item in solver_seen]
    assert min(fractions) >= 0.0
    assert max(fractions) <= 1.0
    assert any(float(value) > 0.0 for value in fractions)


@pytest.mark.integration
def test_batch_planner_keeps_success_when_md_postcheck_exceeded() -> None:
    records = [
        WelltrackRecord(
            name="WELL-MD",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
    ]
    cfg = _fast_batch_config().validated_copy(max_total_md_postcheck_m=100.0)
    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={"WELL-MD"},
        config=cfg,
    )

    assert len(rows) == 1
    assert len(successes) == 1
    row = rows[0]
    assert row["Статус"] == "OK"
    assert float(row["Макс MD, м"]) > float(cfg.max_total_md_postcheck_m)
    assert "Превышен лимит итоговой MD" in str(row["Проблема"])
    assert successes[0].md_postcheck_exceeded is True
    assert "Превышен лимит итоговой MD" in successes[0].md_postcheck_message


def test_successful_well_plan_accepts_model_like_inputs() -> None:
    class LegacyPoint(BaseModel):
        x: float
        y: float
        z: float

    class LegacyConfig(BaseModel):
        md_step_m: float = 10.0
        md_step_control_m: float = 2.0
        pos_tolerance_m: float = 2.0
        entry_inc_target_deg: float = 86.0
        entry_inc_tolerance_deg: float = 2.0
        max_inc_deg: float = 95.0
        dls_build_min_deg_per_30m: float = 0.0
        dls_build_max_deg_per_30m: float = 3.0
        kop_min_vertical_m: float = 550.0
        max_total_md_m: float = 12000.0
        max_total_md_postcheck_m: float = 6500.0
        objective_mode: str = "minimize_total_md"
        turn_solver_mode: str = "least_squares"
        turn_solver_max_restarts: int = 2
        min_structural_segment_m: float = 30.0
        dls_limits_deg_per_30m: dict[str, float] = {
            "VERTICAL": 1.0,
            "BUILD1": 3.0,
            "HOLD": 2.0,
            "BUILD2": 3.0,
            "HORIZONTAL": 2.0,
        }

    plan = SuccessfulWellPlan(
        name="WELL-1",
        surface=LegacyPoint(x=0.0, y=0.0, z=0.0),
        t1=LegacyPoint(x=600.0, y=800.0, z=2400.0),
        t3=LegacyPoint(x=1500.0, y=2000.0, z=2500.0),
        stations=pd.DataFrame({"MD_m": [0.0], "X_m": [0.0], "Y_m": [0.0], "Z_m": [0.0]}),
        summary={"trajectory_type": "Unified J Profile + Build + Azimuth Turn"},
        azimuth_deg=0.0,
        md_t1_m=1000.0,
        config=LegacyConfig(),
    )

    assert plan.surface.x == 0.0
    assert plan.config.turn_solver_mode == "least_squares"


def test_merge_batch_results_preserves_other_wells_and_adds_not_calculated_rows() -> None:
    records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-C",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=700.0, y=760.0, z=2200.0, md=2300.0),
                WelltrackPoint(x=1600.0, y=1960.0, z=2350.0, md=3350.0),
            ),
        ),
    ]
    planner = WelltrackBatchPlanner(planner=_StubPlanner())
    cfg = _fast_batch_config()
    first_rows, first_successes = planner.evaluate(
        records=records,
        selected_names={"WELL-A"},
        config=cfg,
    )
    second_rows = [
        {
            "Скважина": "WELL-B",
            "Точек": 3,
            "Статус": "Ошибка расчета",
            "Модель траектории": "—",
            "Классификация целей": "—",
            "Сложность": "—",
            "Горизонтальный отход t1, м": "—",
            "Длина HORIZONTAL, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс ПИ, deg/10m": "—",
            "Макс MD, м": "—",
            "Проблема": "Solver endpoint miss to t1.",
        }
    ]

    merged_rows, merged_successes = merge_batch_results(
        records=records,
        existing_rows=first_rows,
        existing_successes=first_successes,
        new_rows=second_rows,
        new_successes=[],
    )

    assert [row["Скважина"] for row in merged_rows] == ["WELL-A", "WELL-B", "WELL-C"]
    by_name = {str(row["Скважина"]): row for row in merged_rows}
    assert by_name["WELL-A"]["Статус"] == "OK"
    assert by_name["WELL-B"]["Статус"] == "Ошибка расчета"
    assert by_name["WELL-C"]["Статус"] == "Не рассчитана"
    assert [item.name for item in merged_successes] == ["WELL-A"]


def test_merge_batch_results_replaces_old_success_for_rerun_well() -> None:
    records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
    ]
    planner = WelltrackBatchPlanner(planner=_StubPlanner())
    cfg = _fast_batch_config()
    first_rows, first_successes = planner.evaluate(
        records=records,
        selected_names={"WELL-A"},
        config=cfg,
    )
    rerun_rows = [
        {
            "Скважина": "WELL-A",
            "Точек": 3,
            "Статус": "Ошибка расчета",
            "Модель траектории": "—",
            "Классификация целей": "—",
            "Сложность": "—",
            "Горизонтальный отход t1, м": "—",
            "Длина HORIZONTAL, м": "—",
            "INC в t1, deg": "—",
            "ЗУ HOLD, deg": "—",
            "Макс ПИ, deg/10m": "—",
            "Макс MD, м": "—",
            "Проблема": "Failure on rerun.",
        }
    ]

    merged_rows, merged_successes = merge_batch_results(
        records=records,
        existing_rows=first_rows,
        existing_successes=first_successes,
        new_rows=rerun_rows,
        new_successes=[],
    )

    assert merged_rows[0]["Статус"] == "Ошибка расчета"
    assert merged_successes == []


def test_recommended_batch_selection_prefers_unresolved_wells_only() -> None:
    records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-C",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=700.0, y=760.0, z=2200.0, md=2300.0),
                WelltrackPoint(x=1600.0, y=1960.0, z=2350.0, md=3350.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-D",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=720.0, y=740.0, z=2180.0, md=2280.0),
                WelltrackPoint(x=1620.0, y=1940.0, z=2330.0, md=3330.0),
            ),
        ),
    ]
    summary_rows = [
        {
            "Скважина": "WELL-A",
            "Статус": "OK",
            "Проблема": "",
        },
        {
            "Скважина": "WELL-B",
            "Статус": "Ошибка расчета",
            "Проблема": "Endpoint miss.",
        },
        {
            "Скважина": "WELL-C",
            "Статус": "OK",
            "Проблема": "Превышен лимит итоговой MD.",
        },
    ]

    initial = recommended_batch_selection(records=records, summary_rows=None)
    follow_up = recommended_batch_selection(records=records, summary_rows=summary_rows)

    assert initial == ["WELL-A", "WELL-B", "WELL-C", "WELL-D"]
    assert follow_up == ["WELL-B", "WELL-C", "WELL-D"]
