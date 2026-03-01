from __future__ import annotations

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import TrajectoryConfig
from pywp.welltrack_batch import WelltrackBatchPlanner


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
        config=TrajectoryConfig(),
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

    rows, successes = WelltrackBatchPlanner().evaluate(
        records=records,
        selected_names={"WELL-A", "WELL-B"},
        config=TrajectoryConfig(),
        progress_callback=on_progress,
    )

    assert len(rows) == 2
    assert len(successes) == 2
    assert seen == [(1, 2, "WELL-A"), (2, 2, "WELL-B")]
