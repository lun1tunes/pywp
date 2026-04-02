from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from pywp.models import OPTIMIZATION_NONE, Point3D, SummaryDict, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner


@dataclass(frozen=True)
class OptimizationReference:
    summary: SummaryDict
    runtime_s: float


def compute_unoptimized_reference(
    *,
    planner: TrajectoryPlanner,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    config: TrajectoryConfig,
    progress_callback: Callable[[str, float], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> OptimizationReference | None:
    if str(config.optimization_mode) == OPTIMIZATION_NONE:
        return None

    reference_config = config.validated_copy(optimization_mode=OPTIMIZATION_NONE)
    started = perf_counter()
    try:
        if progress_callback is not None:
            progress_callback("Расчёт baseline без оптимизации...", 0.0)
        result = planner.plan(
            surface=surface,
            t1=t1,
            t3=t3,
            config=reference_config,
            progress_callback=progress_callback,
        )
    except (PlanningError, ValueError):
        if on_error is not None:
            on_error("Baseline без оптимизации не построен.")
        return None
    elapsed_s = perf_counter() - started
    return OptimizationReference(summary=result.summary, runtime_s=float(elapsed_s))
