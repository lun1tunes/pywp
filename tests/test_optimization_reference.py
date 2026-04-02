from __future__ import annotations

from pywp.models import Point3D, TrajectoryConfig
from pywp.optimization_reference import compute_unoptimized_reference
from pywp.planner import PlanningError


class _FailingPlanner:
    def plan(self, **_: object):
        raise PlanningError("baseline failed")


def test_compute_unoptimized_reference_reports_error_callback_on_failure() -> None:
    errors: list[str] = []

    result = compute_unoptimized_reference(
        planner=_FailingPlanner(),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=TrajectoryConfig(optimization_mode="minimize_md"),
        on_error=errors.append,
    )

    assert result is None
    assert errors == ["Baseline без оптимизации не построен."]
