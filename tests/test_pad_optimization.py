import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

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


def test_optimize_pad_order_does_not_swap_fixed_wells(monkeypatch) -> None:
    records = [
        WelltrackRecord(
            name="FIXED",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1000.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-A",
            points=(
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=520.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1020.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-B",
            points=(
                WelltrackPoint(x=40.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=540.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1040.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
    ]
    success_dict = {str(record.name): object() for record in records}
    analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="FIXED",
                well_b="MOVE-A",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.25,
            ),
        )
    )
    calls: list[tuple[str, str]] = []

    class DummyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self, *, wait: bool = True) -> None:
            pass

    def fake_swap(
        _records,
        _successes,
        name_a,
        name_b,
        _config_by_name,
        pool=None,
    ):
        calls.append((str(name_a), str(name_b)))
        return None

    monkeypatch.setattr(pad_optimization, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(
        pad_optimization,
        "_build_ac_well_light",
        lambda _success, _model: object(),
    )
    monkeypatch.setattr(
        pad_optimization,
        "_analyze_from_ac_wells",
        lambda _ac_wells, _ref_ac_wells: analysis,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_swap_surfaces_and_recalculate",
        fake_swap,
    )

    _, _, improved = optimize_pad_order(
        records=records,
        success_dict=success_dict,
        pad_well_names={"FIXED", "MOVE-A", "MOVE-B"},
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        reference_wells=[],
        config_by_name={},
        progress_callback=lambda _pct, _msg: None,
        fixed_well_names={"FIXED"},
    )

    assert improved is False
    assert calls == [("MOVE-A", "MOVE-B")]


def test_optimize_pad_order_ignores_fixed_fixed_score_floor(monkeypatch) -> None:
    records = [
        WelltrackRecord(
            name="FIXED-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1000.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="FIXED-B",
            points=(
                WelltrackPoint(x=10.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=510.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1010.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-A",
            points=(
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=520.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1020.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-B",
            points=(
                WelltrackPoint(x=40.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=540.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1040.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
    ]
    success_dict = {str(record.name): object() for record in records}
    initial_analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="FIXED-A",
                well_b="FIXED-B",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.10,
            ),
            SimpleNamespace(
                well_a="MOVE-A",
                well_b="MOVE-B",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.50,
            ),
        )
    )
    candidate_analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="FIXED-A",
                well_b="FIXED-B",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.10,
            ),
            SimpleNamespace(
                well_a="MOVE-A",
                well_b="MOVE-B",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.80,
            ),
        )
    )
    analyze_calls = 0
    swap_calls: list[tuple[str, str]] = []

    class DummyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self, *, wait: bool = True) -> None:
            pass

    def fake_analyze(_ac_wells, _ref_ac_wells):
        nonlocal analyze_calls
        analyze_calls += 1
        if analyze_calls <= 2:
            return initial_analysis
        return candidate_analysis

    def fake_swap(
        _records,
        successes,
        name_a,
        name_b,
        _config_by_name,
        pool=None,
    ):
        swap_calls.append((str(name_a), str(name_b)))
        return list(_records), dict(successes)

    monkeypatch.setattr(pad_optimization, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(
        pad_optimization,
        "_build_ac_well_light",
        lambda success, _model: success,
    )
    monkeypatch.setattr(pad_optimization, "_analyze_from_ac_wells", fake_analyze)
    monkeypatch.setattr(
        pad_optimization,
        "_swap_surfaces_and_recalculate",
        fake_swap,
    )

    _, _, improved = optimize_pad_order(
        records=records,
        success_dict=success_dict,
        pad_well_names={"FIXED-A", "FIXED-B", "MOVE-A", "MOVE-B"},
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        reference_wells=[],
        config_by_name={},
        progress_callback=lambda _pct, _msg: None,
        fixed_well_names={"FIXED-A", "FIXED-B"},
    )

    assert improved is True
    assert swap_calls == [("MOVE-A", "MOVE-B")]


def test_score_analysis_includes_reference_conflicts_touching_movable_pad_wells() -> None:
    analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="MOVE-A",
                well_b="FACT-1",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.40,
            ),
            SimpleNamespace(
                well_a="OTHER-PAD",
                well_b="FACT-2",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.10,
            ),
        )
    )

    score = pad_optimization.score_analysis(
        analysis,
        pad_well_names={"MOVE-A", "MOVE-B"},
        fixed_well_names=set(),
    )

    assert score == pytest.approx(0.4004)


def test_score_analysis_ignores_reference_conflicts_touching_only_fixed_pad_wells() -> None:
    analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="FIXED-A",
                well_b="FACT-1",
                md_a_m=1000.0,
                md_b_m=1000.0,
                separation_factor=0.10,
            ),
        )
    )

    score = pad_optimization.score_analysis(
        analysis,
        pad_well_names={"FIXED-A", "MOVE-A"},
        fixed_well_names={"FIXED-A"},
    )

    assert score == float("inf")


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


def test_pad_swap_recalculation_falls_back_when_pool_breaks() -> None:
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

    class BrokenPool:
        def submit(self, *args, **kwargs):
            raise pad_optimization.BrokenProcessPool("spawn pool unavailable")

    result = pad_optimization._swap_surfaces_and_recalculate(
        records=records,
        successes={},
        name_a="PAD-A",
        name_b="PAD-B",
        config_by_name={"PAD-A": config, "PAD-B": config},
        pool=BrokenPool(),
    )

    assert result is not None
    swapped_records, swapped_successes = result
    assert swapped_records[0].points[0].x == pytest.approx(50.0)
    assert swapped_records[1].points[0].x == pytest.approx(0.0)
    assert set(swapped_successes) == {"PAD-A", "PAD-B"}
