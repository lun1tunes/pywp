import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest
from pywp.eclipse_welltrack import WelltrackRecord, WelltrackPoint
from pywp.models import TrajectoryConfig, Point3D
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL
from pywp.welltrack_batch import SuccessfulWellPlan
from pywp import pad_optimization
from pywp.pad_optimization import optimize_pad_order
from pywp.anticollision import AntiCollisionAnalysis, AntiCollisionZone

def test_optimize_pad_order_no_records():
    records = []
    successes = {}
    model = DEFAULT_PLANNING_UNCERTAINTY_MODEL
    def cb(pct, msg): pass
    res_rec, res_suc, improved = optimize_pad_order(records, successes, set(), model, [], {}, cb)
    assert not improved


@pytest.mark.integration
def test_pad_swap_recalculation_runs_under_spawn_context() -> None:
    records = [
        WelltrackRecord(
            name="PAD-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="PAD-B",
            points=(
                WelltrackPoint(x=50.0, y=50.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=850.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1550.0, y=2050.0, z=2500.0, md=3500.0),
            ),
        ),
    ]
    config = TrajectoryConfig(
        md_step_m=10.0,
        md_step_control_m=2.0,
        pos_tolerance_m=2.0,
        turn_solver_max_restarts=0,
    )

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        result = pad_optimization._swap_surfaces_and_recalculate(
            records=records,
            successes={},
            name_a="PAD-A",
            name_b="PAD-B",
            config_by_name={"PAD-A": config, "PAD-B": config},
            pool=pool,
        )

    assert result is not None
    swapped_records, swapped_successes = result
    assert swapped_records[0].points[0].x == pytest.approx(50.0)
    assert swapped_records[1].points[0].x == pytest.approx(0.0)
    assert set(swapped_successes) == {"PAD-A", "PAD-B"}
