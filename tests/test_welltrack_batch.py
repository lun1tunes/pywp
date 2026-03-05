from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import PlannerResult, TrajectoryConfig
from pywp.welltrack_batch import WelltrackBatchPlanner


def _fast_batch_config(**overrides: Any) -> TrajectoryConfig:
    base: dict[str, Any] = {
        "md_step_m": 10.0,
        "md_step_control_m": 2.0,
        "pos_tolerance_m": 2.0,
        "kop_search_grid_size": 21,
        "adaptive_grid_enabled": True,
        "adaptive_dense_check_enabled": False,
        "adaptive_grid_initial_size": 5,
        "adaptive_grid_refine_levels": 1,
        "adaptive_grid_top_k": 2,
        "parallel_jobs": 1,
        "turn_solver_qmc_samples": 0,
        "turn_solver_local_starts": 1,
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
            "trajectory_type": "J Profile + Continious Build",
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
    cfg = replace(_fast_batch_config(), max_total_md_postcheck_m=100.0)
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
