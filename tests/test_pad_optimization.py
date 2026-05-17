import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from pywp.eclipse_welltrack import WelltrackRecord, WelltrackPoint
from pywp.models import TrajectoryConfig, Point3D
from pywp.reference_trajectories import ImportedTrajectoryWell
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL
from pywp.welltrack_batch import SuccessfulWellPlan
from pywp import pad_optimization
from pywp.pad_optimization import optimize_pad_order
from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionPairCacheEntry,
    AntiCollisionZone,
)


def _fake_incremental_result(analysis, pair_cache=None, reused_pair_count=0):
    return (
        analysis,
        dict(pair_cache or {}),
        SimpleNamespace(
            reused_pair_count=int(reused_pair_count),
            recalculated_pair_count=0,
            prefiltered_pair_count=0,
            reused_well_count=0,
            rebuilt_well_count=0,
        ),
    )


def _straight_reference_well(
    name: str = "REF",
    *,
    kind: str = "actual",
) -> ImportedTrajectoryWell:
    return ImportedTrajectoryWell(
        name=name,
        kind=kind,
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 1000.0],
                "INC_deg": [0.0, 90.0],
                "AZI_deg": [90.0, 90.0],
                "X_m": [0.0, 1000.0],
                "Y_m": [0.0, 0.0],
                "Z_m": [0.0, 1000.0],
            }
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        azimuth_deg=90.0,
    )


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
        "_analyze_incremental_from_ac_wells",
        lambda _ac_wells, _ref_ac_wells, **_kwargs: _fake_incremental_result(
            analysis
        ),
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


def test_optimize_pad_order_keeps_fixed_surface_when_swap_is_accepted(
    monkeypatch,
) -> None:
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
    initial_analysis = SimpleNamespace(
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
    improved_analysis = SimpleNamespace(zones=())
    analyze_calls = 0
    swap_calls: list[tuple[str, str]] = []

    class DummyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self, *, wait: bool = True) -> None:
            pass

    def fake_analyze(_ac_wells, _ref_ac_wells, **_kwargs):
        nonlocal analyze_calls
        analyze_calls += 1
        if analyze_calls == 1:
            return _fake_incremental_result(initial_analysis)
        return _fake_incremental_result(improved_analysis)

    def fake_swap(source_records, successes, name_a, name_b, _config_by_name, pool=None):
        swap_calls.append((str(name_a), str(name_b)))
        assert "FIXED" not in {str(name_a), str(name_b)}
        indices = {str(record.name): idx for idx, record in enumerate(source_records)}
        idx_a = indices[str(name_a)]
        idx_b = indices[str(name_b)]
        pts_a = source_records[idx_a].points
        pts_b = source_records[idx_b].points
        new_records = list(source_records)
        new_records[idx_a] = WelltrackRecord(
            name=source_records[idx_a].name,
            points=(pts_b[0], *pts_a[1:]),
        )
        new_records[idx_b] = WelltrackRecord(
            name=source_records[idx_b].name,
            points=(pts_a[0], *pts_b[1:]),
        )
        return new_records, dict(successes)

    monkeypatch.setattr(pad_optimization, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(
        pad_optimization,
        "_build_ac_well_light",
        lambda success, _model: success,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_analyze_incremental_from_ac_wells",
        fake_analyze,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_swap_surfaces_and_recalculate",
        fake_swap,
    )

    new_records, _, improved = optimize_pad_order(
        records=records,
        success_dict={str(record.name): object() for record in records},
        pad_well_names={"FIXED", "MOVE-A", "MOVE-B"},
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        reference_wells=[],
        config_by_name={},
        progress_callback=lambda _pct, _msg: None,
        fixed_well_names={"FIXED"},
    )

    by_name = {str(record.name): record for record in new_records}
    assert improved is True
    assert swap_calls == [("MOVE-A", "MOVE-B")]
    assert by_name["FIXED"].points[0] == records[0].points[0]
    assert by_name["MOVE-A"].points[0] == records[2].points[0]
    assert by_name["MOVE-B"].points[0] == records[1].points[0]


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
        if analyze_calls == 1:
            return _fake_incremental_result(initial_analysis)
        return _fake_incremental_result(candidate_analysis)

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
    monkeypatch.setattr(
        pad_optimization,
        "_analyze_incremental_from_ac_wells",
        lambda ac_wells, ref_ac_wells, **_kwargs: fake_analyze(
            ac_wells,
            ref_ac_wells,
        ),
    )
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


def test_optimize_pad_order_rejects_swap_that_increases_collision_zones(
    monkeypatch,
) -> None:
    records = [
        WelltrackRecord(
            name="MOVE-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1000.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-B",
            points=(
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=520.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1020.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
    ]
    success_dict = {str(record.name): object() for record in records}
    initial_analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="MOVE-A",
                well_b="MOVE-B",
                md_a_m=1000.0,
                md_b_m=1000.0,
                priority_rank=2,
                separation_factor=0.25,
            ),
        )
    )
    candidate_analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="MOVE-A",
                well_b="MOVE-B",
                md_a_m=1000.0,
                md_b_m=1000.0,
                priority_rank=2,
                separation_factor=0.85,
            ),
            SimpleNamespace(
                well_a="MOVE-A",
                well_b="MOVE-B",
                md_a_m=1500.0,
                md_b_m=1500.0,
                priority_rank=2,
                separation_factor=0.90,
            ),
        )
    )
    analyze_calls = 0

    class DummyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self, *, wait: bool = True) -> None:
            pass

    def fake_analyze(_ac_wells, _ref_ac_wells, **_kwargs):
        nonlocal analyze_calls
        analyze_calls += 1
        if analyze_calls == 1:
            return _fake_incremental_result(initial_analysis)
        return _fake_incremental_result(candidate_analysis)

    monkeypatch.setattr(pad_optimization, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(
        pad_optimization,
        "_build_ac_well_light",
        lambda success, _model: success,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_analyze_incremental_from_ac_wells",
        fake_analyze,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_swap_surfaces_and_recalculate",
        lambda records, successes, *_args, **_kwargs: (list(records), dict(successes)),
    )

    _, _, improved = optimize_pad_order(
        records=records,
        success_dict=success_dict,
        pad_well_names={"MOVE-A", "MOVE-B"},
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        reference_wells=[],
        config_by_name={},
        progress_callback=lambda _pct, _msg: None,
    )

    assert improved is False


def test_score_improvement_does_not_trade_target_pair_for_more_total_pairs() -> None:
    current = pad_optimization._PadOptimizationScore(
        target_zone_count=1,
        overlap_pair_count=1,
        zone_count=1,
        severe_zone_count=1,
        worst_sf=0.25,
        mean_sf=0.25,
    )
    candidate = pad_optimization._PadOptimizationScore(
        target_zone_count=0,
        overlap_pair_count=2,
        zone_count=2,
        severe_zone_count=1,
        worst_sf=0.80,
        mean_sf=0.80,
    )

    assert pad_optimization._score_is_improvement(candidate, current) is False


def test_score_improvement_does_not_trade_fewer_pairs_for_more_zones() -> None:
    current = pad_optimization._PadOptimizationScore(
        target_zone_count=0,
        overlap_pair_count=2,
        zone_count=2,
        severe_zone_count=0,
        worst_sf=0.50,
        mean_sf=0.60,
    )
    candidate = pad_optimization._PadOptimizationScore(
        target_zone_count=0,
        overlap_pair_count=1,
        zone_count=3,
        severe_zone_count=0,
        worst_sf=0.90,
        mean_sf=0.90,
    )

    assert pad_optimization._score_is_improvement(candidate, current) is False


def test_score_improvement_accepts_clean_target_pair_reduction() -> None:
    current = pad_optimization._PadOptimizationScore(
        target_zone_count=1,
        overlap_pair_count=2,
        zone_count=2,
        severe_zone_count=1,
        worst_sf=0.25,
        mean_sf=0.35,
    )
    candidate = pad_optimization._PadOptimizationScore(
        target_zone_count=0,
        overlap_pair_count=2,
        zone_count=2,
        severe_zone_count=1,
        worst_sf=0.25,
        mean_sf=0.35,
    )

    assert pad_optimization._score_is_improvement(candidate, current) is True


def test_optimize_pad_order_reuses_existing_pair_cache_for_candidate_scoring(
    monkeypatch,
) -> None:
    records = [
        WelltrackRecord(
            name="MOVE-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1000.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-B",
            points=(
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=520.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1020.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
    ]
    success_dict = {str(record.name): object() for record in records}
    initial_analysis = SimpleNamespace(
        zones=(
            SimpleNamespace(
                well_a="MOVE-A",
                well_b="MOVE-B",
                md_a_m=1000.0,
                md_b_m=1000.0,
                priority_rank=2,
                separation_factor=0.25,
            ),
        )
    )
    candidate_analysis = SimpleNamespace(zones=())
    previous_pair_cache = {
        ("MOVE-A", "MOVE-B"): AntiCollisionPairCacheEntry(
            well_a="MOVE-A",
            well_b="MOVE-B",
            signature_a="sig-a",
            signature_b="sig-b",
            corridors=(),
            zones=(),
            build_overlap_geometry=True,
        )
    }
    captured_previous_caches: list[dict] = []

    class DummyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self, *, wait: bool = True) -> None:
            pass

    def fake_analyze(_ac_wells, _ref_ac_wells, **kwargs):
        captured_previous_caches.append(dict(kwargs.get("previous_pair_cache") or {}))
        return _fake_incremental_result(candidate_analysis, pair_cache={})

    monkeypatch.setattr(pad_optimization, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(
        pad_optimization,
        "_build_ac_well_light",
        lambda success, _model: success,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_analyze_incremental_from_ac_wells",
        fake_analyze,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_swap_surfaces_and_recalculate",
        lambda records, successes, *_args, **_kwargs: (list(records), dict(successes)),
    )

    _, _, improved = optimize_pad_order(
        records=records,
        success_dict=success_dict,
        pad_well_names={"MOVE-A", "MOVE-B"},
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        reference_wells=[],
        config_by_name={},
        progress_callback=lambda _pct, _msg: None,
        initial_analysis=initial_analysis,
        previous_pair_cache=previous_pair_cache,
        well_signature_by_name={"MOVE-A": "sig-a", "MOVE-B": "sig-b"},
    )

    assert improved is True
    assert captured_previous_caches == [previous_pair_cache]


def test_cached_or_built_ac_well_does_not_reuse_without_current_signature() -> None:
    previous_well = pad_optimization._build_ref_ac_well_light(
        _straight_reference_well(),
        DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        collision_name="REF",
    )
    built = []

    result = pad_optimization._cached_or_built_ac_well(
        name="REF",
        signature_by_name={},
        previous_well_cache={"REF": ("stale-signature", previous_well)},
        builder=lambda: built.append("built") or previous_well,
    )

    assert result is previous_well
    assert built == ["built"]


def test_optimize_pad_order_uses_reference_specific_uncertainty_model(
    monkeypatch,
) -> None:
    records = [
        WelltrackRecord(
            name="MOVE-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1000.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-B",
            points=(
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=520.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1020.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
    ]
    reference = _straight_reference_well("FACT-UNKNOWN")
    default_model = DEFAULT_PLANNING_UNCERTAINTY_MODEL
    unknown_model = replace(
        default_model,
        sigma_azi_deg=default_model.sigma_azi_deg * 2.0,
    )
    built_reference_models = []

    class DummyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self, *, wait: bool = True) -> None:
            pass

    monkeypatch.setattr(pad_optimization, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(
        pad_optimization,
        "_build_ac_well_light",
        lambda success, _model: success,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_build_ref_ac_well_light",
        lambda ref, model, **_kwargs: built_reference_models.append(model) or ref,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_analyze_incremental_from_ac_wells",
        lambda _ac_wells, _ref_ac_wells, **_kwargs: _fake_incremental_result(
            SimpleNamespace(zones=())
        ),
    )

    optimize_pad_order(
        records=records,
        success_dict={record.name: object() for record in records},
        pad_well_names={"MOVE-A", "MOVE-B"},
        uncertainty_model=default_model,
        reference_wells=[reference],
        config_by_name={},
        progress_callback=lambda _pct, _msg: None,
        reference_uncertainty_models_by_name={"FACT-UNKNOWN": unknown_model},
    )

    assert built_reference_models == [unknown_model]


def test_optimize_pad_order_keeps_approved_reference_on_default_uncertainty_model(
    monkeypatch,
) -> None:
    records = [
        WelltrackRecord(
            name="MOVE-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=500.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1000.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
        WelltrackRecord(
            name="MOVE-B",
            points=(
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=520.0, y=0.0, z=2000.0, md=2000.0),
                WelltrackPoint(x=1020.0, y=0.0, z=2100.0, md=3000.0),
            ),
        ),
    ]
    reference = _straight_reference_well("PROJECT-APPROVED", kind="approved")
    default_model = DEFAULT_PLANNING_UNCERTAINTY_MODEL
    unknown_model = replace(
        default_model,
        sigma_azi_deg=default_model.sigma_azi_deg * 2.0,
    )
    built_reference_models = []

    class DummyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self, *, wait: bool = True) -> None:
            pass

    monkeypatch.setattr(pad_optimization, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(
        pad_optimization,
        "_build_ac_well_light",
        lambda success, _model: success,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_build_ref_ac_well_light",
        lambda ref, model, **_kwargs: built_reference_models.append(model) or ref,
    )
    monkeypatch.setattr(
        pad_optimization,
        "_analyze_incremental_from_ac_wells",
        lambda _ac_wells, _ref_ac_wells, **_kwargs: _fake_incremental_result(
            SimpleNamespace(zones=())
        ),
    )

    optimize_pad_order(
        records=records,
        success_dict={record.name: object() for record in records},
        pad_well_names={"MOVE-A", "MOVE-B"},
        uncertainty_model=default_model,
        reference_wells=[reference],
        config_by_name={},
        progress_callback=lambda _pct, _msg: None,
        reference_uncertainty_models_by_name={"PROJECT-APPROVED": unknown_model},
    )

    assert built_reference_models == [default_model]


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
