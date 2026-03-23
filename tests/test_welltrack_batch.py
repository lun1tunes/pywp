from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from pywp.anticollision_optimization import (
    AntiCollisionClearanceEvaluation,
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
)
from pywp.anticollision_rerun import (
    DYNAMIC_CLUSTER_PLAN_ACTIVE,
    DynamicClusterExecutionPlan,
    build_anti_collision_analysis_for_successes,
    build_anticollision_well_contexts,
    build_dynamic_cluster_execution_plan,
)
from pywp.anticollision_recommendations import build_anti_collision_recommendations
from pywp.anticollision_recommendations import (
    AntiCollisionClusterActionStep,
    AntiCollisionRecommendationCluster,
)
from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackRecord,
    parse_welltrack_text,
)
from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.uncertainty import (
    DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    DEFAULT_UNCERTAINTY_PRESET,
    planning_uncertainty_model_for_preset,
)
from pywp.welltrack_batch import (
    DynamicClusterExecutionContext,
    ensure_successful_plan_baseline,
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
        optimization_context: Any = None,
        progress_callback: Any = None,
    ) -> PlannerResult:
        if progress_callback is not None:
            progress_callback("Планировщик: подготовка", 0.0)
            progress_callback("Планировщик: завершение", 1.0)

        stations = pd.DataFrame(
            {
                "MD_m": [0.0, 1000.0, 2000.0],
                "INC_deg": [0.0, 45.0, 90.0],
                "AZI_deg": [0.0, 90.0, 90.0],
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
            "solver_turn_restarts_used": 0.0,
            "solver_turn_max_restarts": float(config.turn_solver_max_restarts),
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
    assert by_name["OK-1"]["Рестарты решателя"] == "0"
    assert by_name["OK-1"]["Классификация целей"] == "В прямом направлении"
    assert by_name["OK-1"]["KOP MD, м"] == "550.00"
    assert by_name["BAD-2"]["Статус"] == "Ошибка формата"
    assert by_name["BAD-2"]["Рестарты решателя"] == "—"
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


def test_batch_planner_writes_restart_count_from_solver_summary() -> None:
    class _RestartStubPlanner(_StubPlanner):
        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            progress_callback: Any = None,
        ) -> PlannerResult:
            result = super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                progress_callback=progress_callback,
            )
            summary = dict(result.summary)
            summary["solver_turn_restarts_used"] = 1.0
            summary["solver_turn_max_restarts"] = float(config.turn_solver_max_restarts)
            return PlannerResult(
                stations=result.stations,
                summary=summary,
                azimuth_deg=result.azimuth_deg,
                md_t1_m=result.md_t1_m,
            )

    records = [
        WelltrackRecord(
            name="WELL-R",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
    ]

    rows, _ = WelltrackBatchPlanner(planner=_RestartStubPlanner()).evaluate(
        records=records,
        selected_names={"WELL-R"},
        config=_fast_batch_config(turn_solver_max_restarts=2),
    )

    assert rows[0]["Статус"] == "OK"
    assert rows[0]["Рестарты решателя"] == "1"


def test_batch_planner_maps_reverse_target_direction_to_short_report_label() -> None:
    class _ReverseDirectionStubPlanner(_StubPlanner):
        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            progress_callback: Any = None,
        ) -> PlannerResult:
            result = super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                progress_callback=progress_callback,
            )
            summary = dict(result.summary)
            summary["trajectory_target_direction"] = "Цели в обратном направлении"
            summary["kop_md_m"] = 725.0
            return PlannerResult(
                stations=result.stations,
                summary=summary,
                azimuth_deg=result.azimuth_deg,
                md_t1_m=result.md_t1_m,
            )

    records = [
        WelltrackRecord(
            name="WELL-REV",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
    ]

    rows, _ = WelltrackBatchPlanner(planner=_ReverseDirectionStubPlanner()).evaluate(
        records=records,
        selected_names={"WELL-REV"},
        config=_fast_batch_config(),
    )

    assert rows[0]["Классификация целей"] == "В обратном направлении"
    assert rows[0]["KOP MD, м"] == "725.00"


def test_batch_planner_applies_per_well_config_overrides() -> None:
    class _ConfigCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.optimization_by_target_x: dict[float, list[str]] = {}

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            progress_callback: Any = None,
        ) -> PlannerResult:
            self.optimization_by_target_x.setdefault(float(t1.x), []).append(
                str(config.optimization_mode)
            )
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                progress_callback=progress_callback,
            )

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
    ]
    planner = _ConfigCapturePlanner()
    base_config = _fast_batch_config()
    override_config = base_config.validated_copy(optimization_mode="minimize_kop")

    rows, successes = WelltrackBatchPlanner(planner=planner).evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        config=base_config,
        config_by_name={"WELL-B": override_config},
    )

    assert len(rows) == 2
    assert len(successes) == 2
    assert planner.optimization_by_target_x[600.0] == ["none"]
    assert "minimize_kop" in planner.optimization_by_target_x[650.0]


def test_batch_planner_respects_selected_execution_order() -> None:
    class _OrderCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.seen_names: list[float] = []

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            optimization_context: Any = None,
            progress_callback: Any = None,
        ) -> PlannerResult:
            self.seen_names.append(float(t1.x))
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                optimization_context=optimization_context,
                progress_callback=progress_callback,
            )

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
    ]
    planner = _OrderCapturePlanner()

    rows, successes = WelltrackBatchPlanner(planner=planner).evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        selected_order=["WELL-B", "WELL-A"],
        config=_fast_batch_config(),
    )

    assert len(rows) == 2
    assert len(successes) == 2
    assert planner.seen_names == [650.0, 600.0]


def test_batch_planner_rebuilds_reference_paths_from_recalculated_earlier_steps() -> None:
    class _ReferenceCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.reference_terminal_x_by_target_x: dict[float, list[float]] = {}

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            optimization_context: Any = None,
            progress_callback: Any = None,
        ) -> PlannerResult:
            if optimization_context is not None:
                terminal_x = float(optimization_context.references[0].xyz_m[-1, 0])
                self.reference_terminal_x_by_target_x.setdefault(float(t1.x), []).append(
                    terminal_x
                )
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                optimization_context=optimization_context,
                progress_callback=progress_callback,
            )

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
    ]
    legacy_reference_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 45.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [999.0, 999.0, 999.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 1000.0, 2000.0],
        }
    )
    base_context = AntiCollisionOptimizationContext(
        candidate_md_start_m=1000.0,
        candidate_md_end_m=2000.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(
            build_anti_collision_reference_path(
                well_name="WELL-A",
                stations=legacy_reference_stations,
                md_start_m=1000.0,
                md_end_m=2000.0,
                sample_step_m=50.0,
                model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
            ),
        ),
    )
    planner = _ReferenceCapturePlanner()

    rows, successes = WelltrackBatchPlanner(planner=planner).evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        selected_order=["WELL-A", "WELL-B"],
        config=_fast_batch_config(),
        optimization_context_by_name={"WELL-B": base_context},
    )

    assert len(rows) == 2
    assert len(successes) == 2
    assert planner.reference_terminal_x_by_target_x[650.0] == [1500.0]


def test_batch_planner_dynamic_cluster_context_recomputes_execution_order_per_step(monkeypatch) -> None:
    import pywp.welltrack_batch as batch_module

    class _OrderCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.seen_names: list[str] = []

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            optimization_context: Any = None,
            progress_callback: Any = None,
        ) -> PlannerResult:
            self.seen_names.append(str(t1.x))
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                optimization_context=optimization_context,
                progress_callback=progress_callback,
            )

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
    ]

    calls = {"count": 0}

    def fake_dynamic_plan(
        *,
        successes,
        selected_names,
        target_well_names,
        uncertainty_model,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-A", "WELL-B"),
                prepared_by_well={
                    "WELL-A": {
                        "update_fields": {"optimization_mode": "anti_collision_avoidance"},
                        "optimization_context": None,
                    },
                    "WELL-B": {
                        "update_fields": {"optimization_mode": "minimize_kop"},
                        "optimization_context": None,
                    },
                },
                skipped_wells=(),
            )
        if calls["count"] == 2:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-B",),
                prepared_by_well={
                    "WELL-B": {
                        "update_fields": {"optimization_mode": "minimize_kop"},
                        "optimization_context": None,
                    }
                },
                skipped_wells=(),
            )
        return DynamicClusterExecutionPlan(
            cluster=None,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state="resolved",
        )

    monkeypatch.setattr(
        batch_module,
        "build_dynamic_cluster_execution_plan",
        fake_dynamic_plan,
    )

    planner = _OrderCapturePlanner()
    rows, successes = WelltrackBatchPlanner(planner=planner).evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        selected_order=["WELL-B", "WELL-A"],
        config=_fast_batch_config(),
        dynamic_cluster_context=DynamicClusterExecutionContext(
            target_well_names=("WELL-A", "WELL-B"),
            uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
            initial_successes=(),
        ),
    )

    assert len(rows) == 2
    assert len(successes) == 2
    assert planner.seen_names == ["600.0", "650.0"]


def test_build_dynamic_cluster_execution_plan_uses_real_cluster_filtering() -> None:
    successes = [
        SuccessfulWellPlan(
            name="WELL-A",
            surface={"x": 0.0, "y": 0.0, "z": 0.0},
            t1={"x": 1000.0, "y": 0.0, "z": 0.0},
            t3={"x": 2000.0, "y": 0.0, "z": 0.0},
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 1000.0, 2000.0],
                    "INC_deg": [0.0, 90.0, 90.0],
                    "AZI_deg": [90.0, 90.0, 90.0],
                    "X_m": [0.0, 1000.0, 2000.0],
                    "Y_m": [0.0, 0.0, 0.0],
                    "Z_m": [0.0, 0.0, 0.0],
                    "DLS_deg_per_30m": [0.0, 0.0, 0.0],
                    "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                }
            ),
            summary={
                "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
                "trajectory_target_direction": "Цели в одном направлении",
                "well_complexity": "Обычная",
                "optimization_mode": "none",
                "azimuth_turn_deg": 0.0,
                "horizontal_length_m": 1000.0,
                "entry_inc_deg": 90.0,
                "hold_inc_deg": 90.0,
                "build_dls_selected_deg_per_30m": 0.0,
                "max_dls_total_deg_per_30m": 0.0,
                "kop_md_m": 700.0,
                "max_inc_actual_deg": 90.0,
                "max_inc_deg": 95.0,
                "md_total_m": 2000.0,
                "max_total_md_postcheck_m": 6500.0,
                "md_postcheck_excess_m": 0.0,
            },
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            config=TrajectoryConfig(),
        ),
        SuccessfulWellPlan(
            name="WELL-B",
            surface={"x": 0.0, "y": 40.0, "z": 0.0},
            t1={"x": 1000.0, "y": 40.0, "z": 0.0},
            t3={"x": 2000.0, "y": 40.0, "z": 0.0},
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 1000.0, 2000.0],
                    "INC_deg": [0.0, 90.0, 90.0],
                    "AZI_deg": [90.0, 90.0, 90.0],
                    "X_m": [0.0, 1000.0, 2000.0],
                    "Y_m": [40.0, 40.0, 40.0],
                    "Z_m": [0.0, 0.0, 0.0],
                    "DLS_deg_per_30m": [0.0, 0.0, 0.0],
                    "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                }
            ),
            summary={
                "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
                "trajectory_target_direction": "Цели в одном направлении",
                "well_complexity": "Обычная",
                "optimization_mode": "none",
                "azimuth_turn_deg": 0.0,
                "horizontal_length_m": 1000.0,
                "entry_inc_deg": 90.0,
                "hold_inc_deg": 90.0,
                "build_dls_selected_deg_per_30m": 0.0,
                "max_dls_total_deg_per_30m": 0.0,
                "kop_md_m": 650.0,
                "max_inc_actual_deg": 90.0,
                "max_inc_deg": 95.0,
                "md_total_m": 2000.0,
                "max_total_md_postcheck_m": 6500.0,
                "md_postcheck_excess_m": 0.0,
            },
            azimuth_deg=90.0,
            md_t1_m=1000.0,
            config=TrajectoryConfig(),
        ),
    ]

    plan = build_dynamic_cluster_execution_plan(
        successes=successes,
        selected_names={"WELL-A", "WELL-B"},
        target_well_names=("WELL-A", "WELL-B"),
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    )

    assert plan is not None
    assert plan.resolution_state in {DYNAMIC_CLUSTER_PLAN_ACTIVE, "blocked"}
    assert plan.cluster is not None
    assert set(plan.cluster.well_names) == {"WELL-A", "WELL-B"}
    assert set(plan.ordered_well_names).issubset({"WELL-A", "WELL-B"})


def test_build_dynamic_cluster_execution_plan_stages_mixed_and_trajectory_frontier(
    monkeypatch,
) -> None:
    import pywp.anticollision_rerun as rerun_module

    successes = [
        SuccessfulWellPlan(
            name="well_02",
            surface={"x": 0.0, "y": 0.0, "z": 0.0},
            t1={"x": 1000.0, "y": 0.0, "z": 2000.0},
            t3={"x": 2000.0, "y": 0.0, "z": 2200.0},
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 1000.0, 2000.0],
                    "INC_deg": [0.0, 60.0, 85.0],
                    "AZI_deg": [0.0, 45.0, 45.0],
                    "X_m": [0.0, 700.0, 2000.0],
                    "Y_m": [0.0, 300.0, 0.0],
                    "Z_m": [0.0, 1800.0, 2200.0],
                    "segment": ["VERTICAL", "BUILD1", "BUILD2"],
                }
            ),
            summary={"md_total_m": 2200.0, "trajectory_target_direction": "Цели в одном направлении"},
            azimuth_deg=45.0,
            md_t1_m=1800.0,
            config=TrajectoryConfig(),
        ),
        SuccessfulWellPlan(
            name="well_05",
            surface={"x": 0.0, "y": 0.0, "z": 0.0},
            t1={"x": 900.0, "y": 100.0, "z": 2000.0},
            t3={"x": 2100.0, "y": 200.0, "z": 2200.0},
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 1000.0, 2000.0],
                    "INC_deg": [0.0, 55.0, 84.0],
                    "AZI_deg": [0.0, 42.0, 42.0],
                    "X_m": [0.0, 650.0, 2100.0],
                    "Y_m": [0.0, 280.0, 200.0],
                    "Z_m": [0.0, 1750.0, 2200.0],
                    "segment": ["VERTICAL", "BUILD1", "BUILD2"],
                }
            ),
            summary={"md_total_m": 2200.0, "trajectory_target_direction": "Цели в одном направлении"},
            azimuth_deg=42.0,
            md_t1_m=1750.0,
            config=TrajectoryConfig(),
        ),
    ]

    cluster = AntiCollisionRecommendationCluster(
        cluster_id="ac-cluster-001",
        well_names=("well_01", "well_02", "well_03", "well_04", "well_05"),
        recommendations=(),
        recommendation_count=4,
        target_conflict_count=0,
        vertical_conflict_count=3,
        trajectory_conflict_count=1,
        worst_separation_factor=0.29,
        summary="cluster",
        detail="detail",
        expected_maneuver="mixed",
        blocking_advisory=None,
        rerun_order_label="well_02 → well_05 → well_04",
        first_rerun_well="well_02",
        first_rerun_maneuver="mixed",
        action_steps=(
            AntiCollisionClusterActionStep(
                order_rank=1,
                well_name="well_02",
                category="mixed",
                optimization_mode="anti_collision_avoidance",
                expected_maneuver="mixed",
                reason="r1",
                related_recommendation_count=2,
                worst_separation_factor=0.29,
            ),
            AntiCollisionClusterActionStep(
                order_rank=2,
                well_name="well_05",
                category="trajectory_review",
                optimization_mode="anti_collision_avoidance",
                expected_maneuver="build2",
                reason="r2",
                related_recommendation_count=1,
                worst_separation_factor=0.29,
            ),
            AntiCollisionClusterActionStep(
                order_rank=3,
                well_name="well_04",
                category="reduce_kop",
                optimization_mode="minimize_kop",
                expected_maneuver="kop",
                reason="r3",
                related_recommendation_count=1,
                worst_separation_factor=0.0,
            ),
        ),
        can_prepare_rerun=True,
        affected_wells=("well_02", "well_05", "well_04"),
        action_label="prepare",
    )
    monkeypatch.setattr(
        rerun_module,
        "build_anti_collision_analysis_for_successes",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        rerun_module,
        "build_anti_collision_recommendations",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        rerun_module,
        "build_anti_collision_recommendation_clusters",
        lambda recommendations: (cluster,),
    )
    monkeypatch.setattr(
        rerun_module,
        "build_cluster_prepared_overrides",
        lambda cluster, **kwargs: (
            {
                "well_02": {"update_fields": {"optimization_mode": "anti_collision_avoidance"}},
                "well_05": {"update_fields": {"optimization_mode": "anti_collision_avoidance"}},
                "well_04": {"update_fields": {"optimization_mode": "minimize_kop"}},
            },
            [],
        ),
    )

    plan = build_dynamic_cluster_execution_plan(
        successes=successes,
        selected_names={"well_02", "well_05", "well_04"},
        target_well_names=("well_02", "well_05", "well_04"),
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    )

    assert plan is not None
    assert plan.resolution_state == DYNAMIC_CLUSTER_PLAN_ACTIVE
    assert plan.ordered_well_names == ("well_02", "well_05")
    assert set(plan.prepared_by_well) == {"well_02", "well_05"}


def test_batch_planner_dynamic_cluster_context_stops_when_cluster_resolves_early(
    monkeypatch,
) -> None:
    import pywp.welltrack_batch as batch_module

    class _OrderCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.seen_names: list[str] = []

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            optimization_context: Any = None,
            progress_callback: Any = None,
        ) -> PlannerResult:
            self.seen_names.append(str(t1.x))
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                optimization_context=optimization_context,
                progress_callback=progress_callback,
            )

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
    ]
    calls = {"count": 0}

    def fake_dynamic_plan(
        *,
        successes,
        selected_names,
        target_well_names,
        uncertainty_model,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-A", "WELL-B"),
                prepared_by_well={
                    "WELL-A": {
                        "update_fields": {
                            "optimization_mode": "anti_collision_avoidance"
                        },
                        "optimization_context": None,
                    },
                    "WELL-B": {
                        "update_fields": {"optimization_mode": "minimize_kop"},
                        "optimization_context": None,
                    },
                },
                skipped_wells=(),
            )
        return DynamicClusterExecutionPlan(
            cluster=None,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state="resolved",
        )

    monkeypatch.setattr(
        batch_module,
        "build_dynamic_cluster_execution_plan",
        fake_dynamic_plan,
    )

    planner = WelltrackBatchPlanner(planner=_OrderCapturePlanner())
    rows, successes = planner.evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        selected_order=["WELL-A", "WELL-B"],
        config=_fast_batch_config(),
        dynamic_cluster_context=DynamicClusterExecutionContext(
            target_well_names=("WELL-A", "WELL-B"),
            uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
            initial_successes=(),
        ),
    )

    assert len(rows) == 1
    assert len(successes) == 1
    assert planner._planner.seen_names == ["600.0"]
    assert planner.last_evaluation_metadata.cluster_resolved_early is True
    assert planner.last_evaluation_metadata.cluster_blocked is False
    assert planner.last_evaluation_metadata.skipped_selected_names == ("WELL-B",)


def test_batch_planner_dynamic_cluster_context_stops_when_cluster_becomes_blocked(
    monkeypatch,
) -> None:
    import pywp.welltrack_batch as batch_module

    class _OrderCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.seen_names: list[str] = []

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            optimization_context: Any = None,
            progress_callback: Any = None,
        ) -> PlannerResult:
            self.seen_names.append(str(t1.x))
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                optimization_context=optimization_context,
                progress_callback=progress_callback,
            )

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
    ]
    calls = {"count": 0}

    def fake_dynamic_plan(
        *,
        successes,
        selected_names,
        target_well_names,
        uncertainty_model,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-A", "WELL-B"),
                prepared_by_well={
                    "WELL-A": {
                        "update_fields": {"optimization_mode": "minimize_kop"},
                        "optimization_context": None,
                    },
                    "WELL-B": {
                        "update_fields": {"optimization_mode": "anti_collision_avoidance"},
                        "optimization_context": None,
                    },
                },
                skipped_wells=(),
            )
        return DynamicClusterExecutionPlan(
            cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state="blocked",
            blocking_reason="Сначала решить spacing целей.",
        )

    monkeypatch.setattr(
        batch_module,
        "build_dynamic_cluster_execution_plan",
        fake_dynamic_plan,
    )

    planner = WelltrackBatchPlanner(planner=_OrderCapturePlanner())
    rows, successes = planner.evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        selected_order=["WELL-A", "WELL-B"],
        config=_fast_batch_config(),
        dynamic_cluster_context=DynamicClusterExecutionContext(
            target_well_names=("WELL-A", "WELL-B"),
            uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
            initial_successes=(),
        ),
    )

    assert len(rows) == 1
    assert len(successes) == 1
    assert planner._planner.seen_names == ["600.0"]
    assert planner.last_evaluation_metadata.cluster_resolved_early is False
    assert planner.last_evaluation_metadata.cluster_blocked is True
    assert planner.last_evaluation_metadata.cluster_blocking_reason == (
        "Сначала решить spacing целей."
    )
    assert planner.last_evaluation_metadata.skipped_selected_names == ("WELL-B",)


def test_batch_planner_dynamic_cluster_context_stops_when_no_remaining_actionable_wells(
    monkeypatch,
) -> None:
    import pywp.welltrack_batch as batch_module

    class _OrderCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.seen_names: list[str] = []

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            optimization_context: Any = None,
            progress_callback: Any = None,
        ) -> PlannerResult:
            self.seen_names.append(str(t1.x))
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                optimization_context=optimization_context,
                progress_callback=progress_callback,
            )

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
    ]
    calls = {"count": 0}

    def fake_dynamic_plan(
        *,
        successes,
        selected_names,
        target_well_names,
        uncertainty_model,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-A",),
                prepared_by_well={
                    "WELL-A": {
                        "update_fields": {
                            "optimization_mode": "anti_collision_avoidance"
                        },
                        "optimization_context": None,
                    }
                },
                skipped_wells=(),
            )
        return None

    monkeypatch.setattr(
        batch_module,
        "build_dynamic_cluster_execution_plan",
        fake_dynamic_plan,
    )

    planner = WelltrackBatchPlanner(planner=_OrderCapturePlanner())
    rows, successes = planner.evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        selected_order=["WELL-A", "WELL-B"],
        config=_fast_batch_config(),
        dynamic_cluster_context=DynamicClusterExecutionContext(
            target_well_names=("WELL-A", "WELL-B"),
            uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
            initial_successes=(),
        ),
    )

    assert len(rows) == 1
    assert len(successes) == 1
    assert planner._planner.seen_names == ["600.0"]
    assert planner.last_evaluation_metadata.cluster_resolved_early is False
    assert planner.last_evaluation_metadata.cluster_blocked is True
    assert planner.last_evaluation_metadata.skipped_selected_names == ("WELL-B",)
    assert "не нашел актуальных шагов" in str(
        planner.last_evaluation_metadata.cluster_blocking_reason
    ).lower()


def test_batch_planner_dynamic_cluster_context_reseeds_second_cluster_pass(
    monkeypatch,
) -> None:
    import pywp.welltrack_batch as batch_module

    class _OrderCapturePlanner(_StubPlanner):
        def __init__(self) -> None:
            self.seen_names: list[str] = []

        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            optimization_context: Any = None,
            progress_callback: Any = None,
        ) -> PlannerResult:
            self.seen_names.append(str(t1.x))
            return super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                optimization_context=optimization_context,
                progress_callback=progress_callback,
            )

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
    ]
    calls = {"count": 0}

    def fake_dynamic_plan(
        *,
        successes,
        selected_names,
        target_well_names,
        uncertainty_model,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-A",),
                prepared_by_well={
                    "WELL-A": {
                        "update_fields": {
                            "optimization_mode": "anti_collision_avoidance"
                        },
                        "optimization_context": None,
                    }
                },
                skipped_wells=(),
            )
        if calls["count"] == 2:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-B",),
                prepared_by_well={
                    "WELL-B": {
                        "update_fields": {
                            "optimization_mode": "anti_collision_avoidance"
                        },
                        "optimization_context": None,
                    }
                },
                skipped_wells=(),
            )
        if calls["count"] == 3:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-A",),
                prepared_by_well={
                    "WELL-A": {
                        "update_fields": {
                            "optimization_mode": "anti_collision_avoidance"
                        },
                        "optimization_context": None,
                    }
                },
                skipped_wells=(),
            )
        if calls["count"] == 4:
            return DynamicClusterExecutionPlan(
                cluster=type("Cluster", (), {"cluster_id": "ac-cluster-001"})(),
                ordered_well_names=("WELL-A",),
                prepared_by_well={
                    "WELL-A": {
                        "update_fields": {
                            "optimization_mode": "anti_collision_avoidance"
                        },
                        "optimization_context": None,
                    }
                },
                skipped_wells=(),
            )
        return DynamicClusterExecutionPlan(
            cluster=None,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state="resolved",
        )

    monkeypatch.setattr(
        batch_module,
        "build_dynamic_cluster_execution_plan",
        fake_dynamic_plan,
    )

    planner = WelltrackBatchPlanner(planner=_OrderCapturePlanner())
    rows, successes = planner.evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        selected_order=["WELL-A", "WELL-B"],
        config=_fast_batch_config(),
        dynamic_cluster_context=DynamicClusterExecutionContext(
            target_well_names=("WELL-A", "WELL-B"),
            uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
            initial_successes=(),
        ),
    )

    assert len(rows) == 3
    assert len(successes) == 3
    assert planner._planner.seen_names == ["600.0", "650.0", "600.0"]
    assert planner.last_evaluation_metadata.cluster_blocked is True
    assert "не дал заметного улучшения" in str(
        planner.last_evaluation_metadata.cluster_blocking_reason
    ).lower()


def test_monotonic_anticollision_success_keeps_existing_when_new_sf_is_worse(
    monkeypatch,
) -> None:
    import pywp.welltrack_batch as batch_module

    def make_success(
        *,
        name: str,
        optimization_mode: str,
        x_last_m: float,
        md_total_m: float,
    ) -> SuccessfulWellPlan:
        return SuccessfulWellPlan(
            name=name,
            surface={"x": 0.0, "y": 0.0, "z": 0.0},
            t1={"x": 600.0, "y": 800.0, "z": 2400.0},
            t3={"x": 1500.0, "y": 2000.0, "z": 2500.0},
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 1000.0, 2000.0],
                    "INC_deg": [0.0, 45.0, 90.0],
                    "AZI_deg": [0.0, 90.0, 90.0],
                    "X_m": [0.0, 800.0, x_last_m],
                    "Y_m": [0.0, 900.0, 2000.0],
                    "Z_m": [0.0, 2200.0, 2500.0],
                    "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                }
            ),
            summary={
                "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
                "trajectory_target_direction": "Цели в одном направлении",
                "well_complexity": "Обычная",
                "horizontal_length_m": 1000.0,
                "entry_inc_deg": 86.0,
                "hold_inc_deg": 20.0,
                "max_dls_total_deg_per_30m": 3.0,
                "md_total_m": float(md_total_m),
                "max_total_md_postcheck_m": 6500.0,
                "md_postcheck_excess_m": 0.0,
                "solver_turn_restarts_used": 0.0,
            },
            azimuth_deg=0.0,
            md_t1_m=1000.0,
            config=TrajectoryConfig(optimization_mode=optimization_mode),
        )

    existing_success = make_success(
        name="WELL-A",
        optimization_mode="anti_collision_avoidance",
        x_last_m=1400.0,
        md_total_m=2100.0,
    )
    candidate_success = make_success(
        name="WELL-A",
        optimization_mode="anti_collision_avoidance",
        x_last_m=1200.0,
        md_total_m=2050.0,
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=1500.0,
        candidate_md_end_m=2500.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
    )

    def fake_clearance(*, stations, context):
        x_last = float(stations["X_m"].iloc[-1])
        if x_last > 1300.0:
            return AntiCollisionClearanceEvaluation(
                min_separation_factor=0.65,
                max_overlap_depth_m=3.0,
            )
        return AntiCollisionClearanceEvaluation(
            min_separation_factor=0.39,
            max_overlap_depth_m=8.0,
        )

    monkeypatch.setattr(
        batch_module,
        "evaluate_stations_anti_collision_clearance",
        fake_clearance,
    )

    retained = WelltrackBatchPlanner._select_monotonic_anticollision_success(
        candidate_success=candidate_success,
        existing_success=existing_success,
        optimization_context=context,
    )

    assert retained is existing_success


def test_cluster_monotonic_anticollision_success_keeps_existing_when_cluster_score_worsens(
    monkeypatch,
) -> None:
    def make_success(*, name: str, md_total_m: float) -> SuccessfulWellPlan:
        return SuccessfulWellPlan(
            name=name,
            surface={"x": 0.0, "y": 0.0, "z": 0.0},
            t1={"x": 600.0, "y": 800.0, "z": 2400.0},
            t3={"x": 1500.0, "y": 2000.0, "z": 2500.0},
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 1000.0, 2000.0],
                    "INC_deg": [0.0, 45.0, 90.0],
                    "AZI_deg": [0.0, 90.0, 90.0],
                    "X_m": [0.0, 800.0, 1500.0],
                    "Y_m": [0.0, 900.0, 2000.0],
                    "Z_m": [0.0, 2200.0, 2500.0],
                    "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                }
            ),
            summary={
                "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
                "trajectory_target_direction": "Цели в одном направлении",
                "well_complexity": "Обычная",
                "horizontal_length_m": 1000.0,
                "entry_inc_deg": 86.0,
                "hold_inc_deg": 20.0,
                "max_dls_total_deg_per_30m": 3.0,
                "md_total_m": float(md_total_m),
                "max_total_md_postcheck_m": 6500.0,
                "md_postcheck_excess_m": 0.0,
                "solver_turn_restarts_used": 0.0,
            },
            azimuth_deg=0.0,
            md_t1_m=1000.0,
            config=TrajectoryConfig(optimization_mode="anti_collision_avoidance"),
        )

    existing_success = make_success(name="WELL-A", md_total_m=2100.0)
    candidate_success = make_success(name="WELL-A", md_total_m=2050.0)
    other_success = make_success(name="WELL-B", md_total_m=2200.0)
    context = DynamicClusterExecutionContext(
        target_well_names=("WELL-A", "WELL-B"),
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        initial_successes=(existing_success, other_success),
    )

    calls = {"count": 0}

    def fake_cluster_score(*, success_by_name, dynamic_cluster_context):
        calls["count"] += 1
        if float(success_by_name["WELL-A"].summary["md_total_m"]) > 2090.0:
            return 0.65, 3.0, 4
        return 0.39, 8.0, 5

    monkeypatch.setattr(
        WelltrackBatchPlanner,
        "_cluster_anticollision_score",
        staticmethod(fake_cluster_score),
    )

    retained = WelltrackBatchPlanner._select_cluster_monotonic_anticollision_success(
        candidate_success=candidate_success,
        existing_success=existing_success,
        current_success_by_name={"WELL-A": existing_success, "WELL-B": other_success},
        dynamic_cluster_context=context,
    )

    assert calls["count"] == 2
    assert retained is existing_success


def test_cluster_monotonic_anticollision_success_keeps_existing_when_well_local_score_worsens(
    monkeypatch,
) -> None:
    def make_success(*, name: str, optimization_mode: str, md_total_m: float) -> SuccessfulWellPlan:
        return SuccessfulWellPlan(
            name=name,
            surface={"x": 0.0, "y": 0.0, "z": 0.0},
            t1={"x": 600.0, "y": 800.0, "z": 2400.0},
            t3={"x": 1500.0, "y": 2000.0, "z": 2500.0},
            stations=pd.DataFrame(
                {
                    "MD_m": [0.0, 1000.0, 2000.0],
                    "INC_deg": [0.0, 45.0, 90.0],
                    "AZI_deg": [0.0, 90.0, 90.0],
                    "X_m": [0.0, 800.0, 1500.0],
                    "Y_m": [0.0, 900.0, 2000.0],
                    "Z_m": [0.0, 2200.0, 2500.0],
                    "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                }
            ),
            summary={
                "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
                "trajectory_target_direction": "Цели в одном направлении",
                "well_complexity": "Обычная",
                "horizontal_length_m": 1000.0,
                "entry_inc_deg": 86.0,
                "hold_inc_deg": 20.0,
                "max_dls_total_deg_per_30m": 3.0,
                "md_total_m": float(md_total_m),
                "max_total_md_postcheck_m": 6500.0,
                "md_postcheck_excess_m": 0.0,
                "solver_turn_restarts_used": 0.0,
            },
            azimuth_deg=0.0,
            md_t1_m=1000.0,
            config=TrajectoryConfig(optimization_mode=optimization_mode),
        )

    existing_success = make_success(
        name="WELL-A",
        optimization_mode="anti_collision_avoidance",
        md_total_m=2100.0,
    )
    candidate_success = make_success(
        name="WELL-A",
        optimization_mode="minimize_kop",
        md_total_m=2050.0,
    )
    other_success = make_success(
        name="WELL-B",
        optimization_mode="none",
        md_total_m=2200.0,
    )
    context = DynamicClusterExecutionContext(
        target_well_names=("WELL-A", "WELL-B"),
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        initial_successes=(existing_success, other_success),
    )

    def fake_cluster_score(*, success_by_name, dynamic_cluster_context):
        return 0.55, 3.0, 4

    def fake_well_local_score(*, success_by_name, dynamic_cluster_context, target_well_name):
        if str(success_by_name["WELL-A"].config.optimization_mode) == "anti_collision_avoidance":
            return 0.72, 2.0, 1
        return 0.31, 7.0, 2

    monkeypatch.setattr(
        WelltrackBatchPlanner,
        "_cluster_anticollision_score",
        staticmethod(fake_cluster_score),
    )
    monkeypatch.setattr(
        WelltrackBatchPlanner,
        "_well_local_anticollision_score",
        staticmethod(fake_well_local_score),
    )

    retained = WelltrackBatchPlanner._select_cluster_monotonic_anticollision_success(
        candidate_success=candidate_success,
        existing_success=existing_success,
        current_success_by_name={"WELL-A": existing_success, "WELL-B": other_success},
        dynamic_cluster_context=context,
    )

    assert retained is existing_success


def test_batch_planner_defers_unoptimized_reference_for_optimized_mode() -> None:
    class _OptimizationStubPlanner(_StubPlanner):
        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            progress_callback: Any = None,
        ) -> PlannerResult:
            result = super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                progress_callback=progress_callback,
            )
            summary = dict(result.summary)
            summary["optimization_mode"] = str(config.optimization_mode)
            summary["md_total_m"] = 2100.0 if str(config.optimization_mode) != "none" else 2200.0
            return PlannerResult(
                stations=result.stations,
                summary=summary,
                azimuth_deg=result.azimuth_deg,
                md_t1_m=result.md_t1_m,
            )

    records = [
        WelltrackRecord(
            name="WELL-OPT",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
    ]

    rows, successes = WelltrackBatchPlanner(planner=_OptimizationStubPlanner()).evaluate(
        records=records,
        selected_names={"WELL-OPT"},
        config=_fast_batch_config(optimization_mode="minimize_md"),
    )

    assert len(rows) == 1
    assert len(successes) == 1
    success = successes[0]
    assert success.runtime_s is not None
    assert success.baseline_runtime_s is None
    assert success.baseline_summary is None
    assert float(success.summary["md_total_m"]) == 2100.0


def test_ensure_successful_plan_baseline_computes_reference_lazily() -> None:
    class _OptimizationStubPlanner(_StubPlanner):
        def plan(
            self,
            *,
            surface: Any,
            t1: Any,
            t3: Any,
            config: TrajectoryConfig,
            progress_callback: Any = None,
        ) -> PlannerResult:
            result = super().plan(
                surface=surface,
                t1=t1,
                t3=t3,
                config=config,
                progress_callback=progress_callback,
            )
            summary = dict(result.summary)
            summary["optimization_mode"] = str(config.optimization_mode)
            summary["md_total_m"] = 2100.0 if str(config.optimization_mode) != "none" else 2200.0
            return PlannerResult(
                stations=result.stations,
                summary=summary,
                azimuth_deg=result.azimuth_deg,
                md_t1_m=result.md_t1_m,
            )

    success = SuccessfulWellPlan(
        name="WELL-OPT",
        surface={"x": 0.0, "y": 0.0, "z": 0.0},
        t1={"x": 600.0, "y": 800.0, "z": 2400.0},
        t3={"x": 1500.0, "y": 2000.0, "z": 2500.0},
        stations=_StubPlanner().plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=_fast_batch_config(optimization_mode="minimize_md"),
        ).stations,
        summary={"md_total_m": 2100.0, "optimization_mode": "minimize_md"},
        azimuth_deg=0.0,
        md_t1_m=1000.0,
        config=_fast_batch_config(optimization_mode="minimize_md"),
    )

    updated = ensure_successful_plan_baseline(
        success=success,
        planner=_OptimizationStubPlanner(),
    )

    assert updated.baseline_runtime_s is not None
    assert updated.baseline_summary is not None
    assert float(updated.baseline_summary["md_total_m"]) == 2200.0


@pytest.mark.integration
def test_cluster_rerun_on_welltracks3_keeps_well02_anticollision_solution() -> None:
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS3.INC").read_text(encoding="utf-8")
    )
    base_config = _fast_batch_config(kop_min_vertical_m=550.0, optimization_mode="none")
    planner = WelltrackBatchPlanner()
    rows, successes = planner.evaluate(
        records=records,
        selected_names={str(record.name) for record in records},
        config=base_config,
    )
    model = planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET)
    dynamic_context = DynamicClusterExecutionContext(
        target_well_names=tuple(str(success.name) for success in successes),
        uncertainty_model=model,
        initial_successes=tuple(successes),
    )
    merged_rows = list(rows)
    merged_successes = list(successes)

    for _ in range(3):
        plan = build_dynamic_cluster_execution_plan(
            successes=list(merged_successes),
            selected_names={str(record.name) for record in records},
            target_well_names=dynamic_context.target_well_names,
            uncertainty_model=model,
        )
        assert plan is not None
        if _ == 0:
            assert plan.ordered_well_names[:2] == ("well_02", "well_05")
        config_by_name = {
            str(well_name): base_config.validated_copy(
                **dict(payload.get("update_fields", {}))
            )
            for well_name, payload in plan.prepared_by_well.items()
        }
        optimization_context_by_name = {
            str(well_name): payload["optimization_context"]
            for well_name, payload in plan.prepared_by_well.items()
            if payload.get("optimization_context") is not None
        }
        new_rows, new_successes = planner.evaluate(
            records=records,
            selected_names={str(record.name) for record in records},
            selected_order=list(plan.ordered_well_names),
            config=base_config,
            config_by_name=config_by_name,
            optimization_context_by_name=optimization_context_by_name,
            dynamic_cluster_context=dynamic_context,
        )
        merged_rows, merged_successes = merge_batch_results(
            records=records,
            existing_rows=merged_rows,
            existing_successes=merged_successes,
            new_rows=new_rows,
            new_successes=new_successes,
        )

    success_by_name = {str(item.name): item for item in merged_successes}
    well_02 = success_by_name["well_02"]
    assert str(well_02.config.optimization_mode) == "anti_collision_avoidance"

    analysis = build_anti_collision_analysis_for_successes(
        list(merged_successes),
        model=model,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=build_anticollision_well_contexts(list(merged_successes)),
    )
    trajectory_conflicts_02_05 = [
        recommendation
        for recommendation in recommendations
        if {str(recommendation.well_a), str(recommendation.well_b)}
        == {"well_02", "well_05"}
        and str(recommendation.category) == "trajectory_review"
    ]
    assert not trajectory_conflicts_02_05


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


@pytest.mark.integration
def test_cluster_rerun_on_welltracks3_keeps_well02_anticollision_solution() -> None:
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS3.INC").read_text(encoding="utf-8")
    )
    base_config = _fast_batch_config(
        kop_min_vertical_m=550.0,
        optimization_mode="none",
    )
    planner = WelltrackBatchPlanner()
    rows, successes = planner.evaluate(
        records=records,
        selected_names={str(record.name) for record in records},
        config=base_config,
    )
    model = planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET)
    dynamic_context = DynamicClusterExecutionContext(
        target_well_names=tuple(str(success.name) for success in successes),
        uncertainty_model=model,
        initial_successes=tuple(successes),
    )
    merged_rows = list(rows)
    merged_successes = list(successes)

    for _ in range(3):
        plan = build_dynamic_cluster_execution_plan(
            successes=list(merged_successes),
            selected_names={str(record.name) for record in records},
            target_well_names=dynamic_context.target_well_names,
            uncertainty_model=model,
        )
        assert plan is not None
        config_by_name = {
            str(well_name): base_config.validated_copy(
                **dict(payload.get("update_fields", {}))
            )
            for well_name, payload in plan.prepared_by_well.items()
        }
        optimization_context_by_name = {
            str(well_name): payload["optimization_context"]
            for well_name, payload in plan.prepared_by_well.items()
            if payload.get("optimization_context") is not None
        }
        new_rows, new_successes = planner.evaluate(
            records=records,
            selected_names={str(record.name) for record in records},
            selected_order=list(plan.ordered_well_names),
            config=base_config,
            config_by_name=config_by_name,
            optimization_context_by_name=optimization_context_by_name,
            dynamic_cluster_context=dynamic_context,
        )
        merged_rows, merged_successes = merge_batch_results(
            records=records,
            existing_rows=merged_rows,
            existing_successes=merged_successes,
            new_rows=new_rows,
            new_successes=new_successes,
        )

    success_by_name = {str(item.name): item for item in merged_successes}
    well_02 = success_by_name["well_02"]
    assert str(well_02.config.optimization_mode) == "anti_collision_avoidance"

    analysis = build_anti_collision_analysis_for_successes(
        list(merged_successes),
        model=model,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=build_anticollision_well_contexts(list(merged_successes)),
    )
    trajectory_conflicts_02_05 = [
        recommendation
        for recommendation in recommendations
        if {str(recommendation.well_a), str(recommendation.well_b)}
        == {"well_02", "well_05"}
        and str(recommendation.category) == "trajectory_review"
    ]
    assert not trajectory_conflicts_02_05
