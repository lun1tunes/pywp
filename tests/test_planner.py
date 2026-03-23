from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from pywp.anticollision_optimization import (
    AntiCollisionClearanceEvaluation,
    AntiCollisionOptimizationContext,
)
from pywp.eclipse_welltrack import parse_welltrack_text, welltrack_points_to_targets
from pywp.models import (
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    OPTIMIZATION_MINIMIZE_KOP,
    OPTIMIZATION_MINIMIZE_MD,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    Point3D,
    TrajectoryConfig,
)
from pywp.planner import PlanningError, TrajectoryPlanner
from pywp.planner_types import CandidateOptimizationEvaluation, ProfileParameters
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL

pytestmark = pytest.mark.integration


def _fast_config(**overrides: object) -> TrajectoryConfig:
    base = {
        "md_step_m": 10.0,
        "md_step_control_m": 2.0,
        "pos_tolerance_m": 2.0,
        "entry_inc_target_deg": 86.0,
        "entry_inc_tolerance_deg": 2.0,
        "dls_build_max_deg_per_30m": 6.0,
        "max_total_md_postcheck_m": 20000.0,
        "turn_solver_mode": TURN_SOLVER_LEAST_SQUARES,
    }
    base.update(overrides)
    return TrajectoryConfig(**base)


def test_same_direction_reference_case_solves_with_minimum_kop() -> None:
    config = _fast_config(kop_min_vertical_m=550.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert abs(float(result.summary["entry_inc_deg"]) - config.entry_inc_target_deg) <= config.entry_inc_tolerance_deg
    assert float(result.summary["kop_md_m"]) == pytest.approx(550.0, abs=1e-6)
    assert float(result.summary["azimuth_turn_deg"]) == pytest.approx(0.0, abs=1e-6)
    assert str(result.summary["trajectory_type"]) == "Unified J Profile + Build + Azimuth Turn"
    assert float(result.summary["max_dls_build1_deg_per_30m"]) <= 6.0 + 1e-6
    assert float(result.summary["max_dls_build2_deg_per_30m"]) <= 6.0 + 1e-6


def test_translated_surface_coordinates_preserve_exact_endpoint_validation() -> None:
    config = _fast_config(turn_solver_max_restarts=0)
    surface = Point3D(598863.0, 7411139.0, 0.0)
    result = TrajectoryPlanner().plan(
        surface=surface,
        t1=Point3D(599463.0, 7411939.0, 2400.0),
        t3=Point3D(600363.0, 7413139.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["t1_exact_x_m"]) == pytest.approx(599463.0, abs=1e-4)
    assert float(result.summary["t1_exact_y_m"]) == pytest.approx(7411939.0, abs=1e-4)
    assert float(result.summary["t3_exact_x_m"]) == pytest.approx(600363.0, abs=1e-4)
    assert float(result.summary["t3_exact_y_m"]) == pytest.approx(7413139.0, abs=1e-4)


def test_higher_min_vertical_pushes_kop_up_deterministically() -> None:
    config_low = _fast_config(kop_min_vertical_m=550.0)
    config_high = _fast_config(kop_min_vertical_m=900.0)
    planner = TrajectoryPlanner()

    result_low = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_low,
    )
    result_high = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_high,
    )

    assert float(result_low.summary["kop_md_m"]) == pytest.approx(550.0, abs=1e-6)
    assert float(result_high.summary["kop_md_m"]) >= 900.0 - 1e-6
    assert float(result_high.summary["md_total_m"]) >= float(result_low.summary["md_total_m"])


def test_higher_build_dls_reduces_total_md() -> None:
    planner = TrajectoryPlanner()
    config_soft = _fast_config(dls_build_max_deg_per_30m=3.0)
    config_hard = _fast_config(dls_build_max_deg_per_30m=6.0)

    result_soft = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_soft,
    )
    result_hard = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_hard,
    )

    assert float(result_hard.summary["md_total_m"]) <= float(result_soft.summary["md_total_m"])


def test_md_optimization_stops_within_theoretical_gap_when_solution_is_already_short() -> None:
    planner = TrajectoryPlanner()
    config_none = _fast_config(kop_min_vertical_m=200.0)
    config_md = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
    )

    result_none = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config_none,
    )
    result_md = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config_md,
    )

    assert float(result_md.summary["distance_t1_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["distance_t3_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["md_total_m"]) <= float(result_none.summary["md_total_m"]) + 1e-6
    assert str(result_md.summary["optimization_mode"]) == OPTIMIZATION_MINIMIZE_MD
    assert str(result_md.summary["optimization_status"]) == "within_md_theoretical_gap"
    assert float(result_md.summary["optimization_relative_gap_pct"]) <= 5.0 + 1e-6
    assert int(result_md.summary["optimization_runs_used"]) == 0


def test_md_optimization_seed_policy_adds_build_and_kop_boundary_variants() -> None:
    import pywp.planner as planner_module

    candidate = planner_module.ProfileParameters(
        kop_vertical_m=950.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=320.0,
        hold_length_m=480.0,
        build2_length_m=210.0,
        horizontal_length_m=1200.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1090.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )

    seeds = planner_module._collect_optimization_seed_vectors(
        candidates=[candidate],
        mode=OPTIMIZATION_MINIMIZE_MD,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
    )

    rounded = [tuple(np.round(seed, decimals=6).tolist()) for seed in seeds]
    assert len(seeds) >= 5
    assert any(seed[0] == pytest.approx(1.0) for seed in seeds)
    assert any(seed[1] == pytest.approx(200.0) for seed in seeds)
    assert any(
        seed[0] == pytest.approx(1.0) and seed[1] == pytest.approx(200.0)
        for seed in seeds
    )
    assert len(rounded) == len(set(rounded))


def test_split_build_optimization_seed_policy_adds_independent_build_variants() -> None:
    import pywp.planner as planner_module

    candidate = planner_module.ProfileParameters(
        kop_vertical_m=950.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=320.0,
        hold_length_m=480.0,
        build2_length_m=210.0,
        horizontal_length_m=1200.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1090.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    bounds = (
        (0.0, 1.0),
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )

    seeds = planner_module._collect_optimization_seed_vectors(
        candidates=[candidate],
        mode=OPTIMIZATION_MINIMIZE_MD,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        split_build=True,
    )

    rounded = [tuple(np.round(seed, decimals=6).tolist()) for seed in seeds]
    assert len(seeds) >= 6
    assert any(seed[0] == pytest.approx(1.0) for seed in seeds)
    assert any(seed[1] == pytest.approx(1.0) for seed in seeds)
    assert any(
        seed[0] == pytest.approx(1.0)
        and seed[1] == pytest.approx(1.0)
        and seed[2] == pytest.approx(200.0)
        for seed in seeds
    )
    assert len(rounded) == len(set(rounded))


def test_anti_collision_selection_prefers_clearance_improving_candidate(monkeypatch) -> None:
    import pywp.planner as planner_module

    seed_candidate = ProfileParameters(
        kop_vertical_m=760.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=320.0,
        hold_length_m=480.0,
        build2_length_m=210.0,
        horizontal_length_m=1200.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1090.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    improved_candidate = ProfileParameters(
        kop_vertical_m=690.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=4.4,
        dls_build2_deg_per_30m=4.4,
        build1_length_m=300.0,
        hold_length_m=420.0,
        build2_length_m=240.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    config = _fast_config(optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE)
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=1200.0,
        candidate_md_end_m=2200.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
    )
    improved_vector = planner_module._candidate_to_search_vector(
        candidate=improved_candidate,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
    )

    def fake_clearance(
        *,
        candidate: ProfileParameters,
        surface: Point3D,
        context: AntiCollisionOptimizationContext,
    ) -> AntiCollisionClearanceEvaluation:
        if float(candidate.kop_vertical_m) == pytest.approx(improved_candidate.kop_vertical_m):
            return AntiCollisionClearanceEvaluation(
                min_separation_factor=1.08,
                max_overlap_depth_m=0.0,
            )
        return AntiCollisionClearanceEvaluation(
            min_separation_factor=0.62,
            max_overlap_depth_m=14.0,
        )

    def fake_evaluate_candidate(
        *,
        values: np.ndarray,
        profile_builder,
        target_point: np.ndarray,
        config: TrajectoryConfig,
    ) -> CandidateOptimizationEvaluation:
        candidate = (
            improved_candidate
            if float(values[1]) <= improved_candidate.kop_vertical_m + 1e-6
            else seed_candidate
        )
        return CandidateOptimizationEvaluation(
            candidate=candidate,
            t1_miss_m=0.25,
            md_total_m=float(candidate.md_total_m),
            kop_vertical_m=float(candidate.kop_vertical_m),
            t1_margin_m=1.75,
            build1_margin_m=50.0,
            build2_margin_m=50.0,
            max_inc_margin_deg=4.0,
            horizontal_dls_margin_deg_per_30m=0.25,
        )

    def fake_minimize(*, fun, x0, method, bounds, constraints, options):
        assert method == "SLSQP"
        return SimpleNamespace(success=True, x=np.asarray(improved_vector, dtype=float))

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        fake_clearance,
    )
    monkeypatch.setattr(
        planner_module,
        "_evaluate_candidate_for_optimization",
        fake_evaluate_candidate,
    )
    monkeypatch.setattr(planner_module, "minimize", fake_minimize)

    result = planner_module._select_anti_collision_candidate(
        candidates=[seed_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: None,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
    )

    assert result.params == improved_candidate
    assert result.optimization.mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    assert result.optimization.status == "sf_target_reached"
    assert result.optimization.runs_used >= 1


def test_anti_collision_selection_can_choose_split_build_candidate(monkeypatch) -> None:
    import pywp.planner as planner_module

    seed_candidate = ProfileParameters(
        kop_vertical_m=760.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=320.0,
        hold_length_m=480.0,
        build2_length_m=210.0,
        horizontal_length_m=1200.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1090.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    improved_candidate = ProfileParameters(
        kop_vertical_m=760.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=5.4,
        dls_build2_deg_per_30m=2.8,
        build1_length_m=244.0,
        hold_length_m=452.0,
        build2_length_m=307.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    config = _fast_config(optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE)
    base_bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    optimization_bounds = (
        (0.0, 1.0),
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=1200.0,
        candidate_md_end_m=2200.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
    )
    improved_vector = planner_module._candidate_to_search_vector(
        candidate=improved_candidate,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=optimization_bounds,
        split_build=True,
    )

    def fake_clearance(
        *,
        candidate: ProfileParameters,
        surface: Point3D,
        context: AntiCollisionOptimizationContext,
    ) -> AntiCollisionClearanceEvaluation:
        if (
            float(candidate.dls_build1_deg_per_30m)
            == pytest.approx(improved_candidate.dls_build1_deg_per_30m)
            and float(candidate.dls_build2_deg_per_30m)
            == pytest.approx(improved_candidate.dls_build2_deg_per_30m)
        ):
            return AntiCollisionClearanceEvaluation(
                min_separation_factor=1.08,
                max_overlap_depth_m=0.0,
            )
        return AntiCollisionClearanceEvaluation(
            min_separation_factor=0.62,
            max_overlap_depth_m=14.0,
        )

    def fake_evaluate_candidate(
        *,
        values: np.ndarray,
        profile_builder,
        target_point: np.ndarray,
        config: TrajectoryConfig,
    ) -> CandidateOptimizationEvaluation:
        candidate = (
            improved_candidate
            if abs(float(values[0]) - float(values[1])) > 1e-6
            else seed_candidate
        )
        return CandidateOptimizationEvaluation(
            candidate=candidate,
            t1_miss_m=0.25,
            md_total_m=float(candidate.md_total_m),
            kop_vertical_m=float(candidate.kop_vertical_m),
            t1_margin_m=1.75,
            build1_margin_m=50.0,
            build2_margin_m=50.0,
            max_inc_margin_deg=4.0,
            horizontal_dls_margin_deg_per_30m=0.25,
        )

    def fake_minimize(*, fun, x0, method, bounds, constraints, options):
        assert method == "SLSQP"
        return SimpleNamespace(success=True, x=np.asarray(improved_vector, dtype=float))

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        fake_clearance,
    )
    monkeypatch.setattr(
        planner_module,
        "_evaluate_candidate_for_optimization",
        fake_evaluate_candidate,
    )
    monkeypatch.setattr(planner_module, "minimize", fake_minimize)

    result = planner_module._select_anti_collision_candidate(
        candidates=[seed_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=base_bounds,
        profile_builder=lambda values: None,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
        optimization_bounds=optimization_bounds,
        optimization_profile_builder=lambda values: None,
        split_build=True,
    )

    assert result.params == improved_candidate
    assert result.params.dls_build1_deg_per_30m != pytest.approx(
        result.params.dls_build2_deg_per_30m
    )


def test_anti_collision_selection_caches_repeated_vector_evaluations(monkeypatch) -> None:
    import pywp.planner as planner_module

    candidate = ProfileParameters(
        kop_vertical_m=760.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=320.0,
        hold_length_m=480.0,
        build2_length_m=210.0,
        horizontal_length_m=1200.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1090.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    config = _fast_config(optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE)
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=1200.0,
        candidate_md_end_m=2200.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
    )
    call_counts = {"evaluate": 0, "clearance": 0}

    def fake_clearance(
        *,
        candidate: ProfileParameters,
        surface: Point3D,
        context: AntiCollisionOptimizationContext,
    ) -> AntiCollisionClearanceEvaluation:
        call_counts["clearance"] += 1
        return AntiCollisionClearanceEvaluation(
            min_separation_factor=0.62,
            max_overlap_depth_m=14.0,
        )

    def fake_evaluate_candidate(
        *,
        values: np.ndarray,
        profile_builder,
        target_point: np.ndarray,
        config: TrajectoryConfig,
    ) -> CandidateOptimizationEvaluation:
        call_counts["evaluate"] += 1
        return CandidateOptimizationEvaluation(
            candidate=candidate,
            t1_miss_m=0.25,
            md_total_m=float(candidate.md_total_m),
            kop_vertical_m=float(candidate.kop_vertical_m),
            t1_margin_m=1.75,
            build1_margin_m=50.0,
            build2_margin_m=50.0,
            max_inc_margin_deg=4.0,
            horizontal_dls_margin_deg_per_30m=0.25,
        )

    def fake_minimize(*, fun, x0, method, bounds, constraints, options):
        assert method == "SLSQP"
        fun(x0)
        fun(np.asarray(x0, dtype=float))
        for constraint in constraints:
            constraint["fun"](x0)
            constraint["fun"](np.asarray(x0, dtype=float))
        return SimpleNamespace(success=True, x=np.asarray(x0, dtype=float))

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        fake_clearance,
    )
    monkeypatch.setattr(
        planner_module,
        "_evaluate_candidate_for_optimization",
        fake_evaluate_candidate,
    )
    monkeypatch.setattr(planner_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        planner_module,
        "_collect_optimization_seed_vectors",
        lambda **kwargs: [],
    )

    result = planner_module._select_anti_collision_candidate(
        candidates=[candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: None,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
    )

    assert result.params == candidate
    assert call_counts["evaluate"] == 1
    assert call_counts["clearance"] == 1


def test_md_optimization_boundary_refinement_can_improve_total_md() -> None:
    planner = TrajectoryPlanner()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(650.8546533242657, 919.0538224668888, 2681.8358083035228)
    t3 = Point3D(1981.7245876002041, 1405.5328339992557, 2774.652111333409)
    config_none = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode="none",
        max_total_md_postcheck_m=25000.0,
    )
    config_md = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        max_total_md_postcheck_m=25000.0,
    )

    result_none = planner.plan(surface=surface, t1=t1, t3=t3, config=config_none)
    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=config_md)

    assert float(result_md.summary["distance_t1_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["distance_t3_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["md_total_m"]) + 100.0 < float(result_none.summary["md_total_m"])
    assert str(result_md.summary["optimization_status"]) == "within_md_theoretical_gap_after_boundary"
    assert int(result_md.summary["optimization_runs_used"]) >= 6


def test_anti_collision_candidate_can_prefer_lower_kop_for_mixed_maneuver_context(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module

    lower_kop_candidate = ProfileParameters(
        kop_vertical_m=600.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=4.4,
        dls_build2_deg_per_30m=4.4,
        build1_length_m=300.0,
        hold_length_m=420.0,
        build2_length_m=240.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    baseline_candidate = ProfileParameters(
        kop_vertical_m=820.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=4.4,
        dls_build2_deg_per_30m=4.4,
        build1_length_m=300.0,
        hold_length_m=420.0,
        build2_length_m=240.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    config = _fast_config(optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE)
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=1200.0,
        candidate_md_end_m=2200.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
        prefer_lower_kop=True,
    )

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        lambda **kwargs: AntiCollisionClearanceEvaluation(
            min_separation_factor=1.05,
            max_overlap_depth_m=0.0,
        ),
    )

    result = planner_module._select_anti_collision_candidate(
        candidates=[baseline_candidate, lower_kop_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: None,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
    )

    assert result.params == lower_kop_candidate
    assert result.optimization.status == "sf_target_reached"


def test_anti_collision_candidate_can_prefer_higher_build1_for_early_maneuver_context(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module

    stronger_build1_candidate = ProfileParameters(
        kop_vertical_m=700.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=5.4,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=280.0,
        hold_length_m=420.0,
        build2_length_m=210.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    softer_build1_candidate = ProfileParameters(
        kop_vertical_m=700.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=3.8,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=280.0,
        hold_length_m=420.0,
        build2_length_m=210.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    config = _fast_config(optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE)
    bounds = (
        (0.0, 1.0),
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=500.0,
        candidate_md_end_m=1800.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
        prefer_higher_build1=True,
    )

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        lambda **kwargs: AntiCollisionClearanceEvaluation(
            min_separation_factor=1.05,
            max_overlap_depth_m=0.0,
        ),
    )

    result = planner_module._select_anti_collision_candidate(
        candidates=[softer_build1_candidate, stronger_build1_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: None,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
        split_build=True,
    )

    assert result.params == stronger_build1_candidate


def test_anti_collision_candidate_can_keep_early_profile_for_late_maneuver_context(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module

    stable_candidate = ProfileParameters(
        kop_vertical_m=700.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=4.2,
        dls_build2_deg_per_30m=2.8,
        build1_length_m=280.0,
        hold_length_m=420.0,
        build2_length_m=210.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    drifted_candidate = ProfileParameters(
        kop_vertical_m=980.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=2.7,
        dls_build2_deg_per_30m=2.8,
        build1_length_m=280.0,
        hold_length_m=420.0,
        build2_length_m=210.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    config = _fast_config(optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE)
    bounds = (
        (0.0, 1.0),
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=3900.0,
        candidate_md_end_m=4300.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
        prefer_keep_kop=True,
        prefer_keep_build1=True,
    )

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        lambda **kwargs: AntiCollisionClearanceEvaluation(
            min_separation_factor=1.05,
            max_overlap_depth_m=0.0,
        ),
    )

    result = planner_module._select_anti_collision_candidate(
        candidates=[stable_candidate, drifted_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: None,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
        split_build=True,
    )

    assert result.params == stable_candidate


def test_anti_collision_early_stage_uses_reduced_kop_build1_search(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module

    candidate = ProfileParameters(
        kop_vertical_m=700.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=5.0,
        dls_build2_deg_per_30m=3.0,
        build1_length_m=280.0,
        hold_length_m=420.0,
        build2_length_m=210.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=38.0,
        azimuth_entry_deg=67.0,
    )
    config = _fast_config(optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE)
    bounds = (
        (0.0, 1.0),
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=500.0,
        candidate_md_end_m=1800.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
        prefer_lower_kop=True,
        prefer_higher_build1=True,
    )

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        lambda **kwargs: AntiCollisionClearanceEvaluation(
            min_separation_factor=0.5,
            max_overlap_depth_m=5.0,
        ),
    )
    minimize_calls: list[np.ndarray] = []

    def _fake_minimize(*, fun, x0, method, bounds, constraints, options):
        minimize_calls.append(np.asarray(x0, dtype=float))
        class _Result:
            x = np.asarray(x0, dtype=float)
        return _Result()

    monkeypatch.setattr(planner_module, "minimize", _fake_minimize)

    result = planner_module._select_anti_collision_candidate(
        candidates=[candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        optimization_bounds=bounds,
        profile_builder=lambda values: candidate,
        optimization_profile_builder=lambda values: candidate,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
        split_build=True,
    )

    assert result.optimization.runs_used == len(minimize_calls)
    assert 1 <= len(minimize_calls) <= planner_module.EARLY_ANTI_COLLISION_MAX_STARTS
    assert all(call.shape == (2,) for call in minimize_calls)
    assert result.optimization.status in {"seed_selected", "clearance_improved"}


def test_md_optimization_2d_refinement_can_improve_total_md_when_boundary_stage_is_unavailable(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module

    monkeypatch.setattr(
        planner_module,
        "_boundary_refine_md_candidates",
        lambda **kwargs: ([], 0),
    )

    planner = TrajectoryPlanner()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(650.8546533242657, 919.0538224668888, 2681.8358083035228)
    t3 = Point3D(1981.7245876002041, 1405.5328339992557, 2774.652111333409)
    config_none = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode="none",
        max_total_md_postcheck_m=25000.0,
    )
    config_md = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        max_total_md_postcheck_m=25000.0,
    )

    result_none = planner.plan(surface=surface, t1=t1, t3=t3, config=config_none)
    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=config_md)

    assert float(result_md.summary["distance_t1_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["distance_t3_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["md_total_m"]) + 100.0 < float(result_none.summary["md_total_m"])
    assert str(result_md.summary["optimization_status"]) in {
        "within_md_theoretical_gap_after_2d",
        "at_md_boundary_extremum",
    }
    assert int(result_md.summary["optimization_runs_used"]) >= 4


def test_removed_legacy_max_total_md_does_not_create_hidden_solver_limit() -> None:
    config = TrajectoryConfig.model_validate(
        {
            "turn_solver_max_restarts": 0,
            "max_total_md_m": 100.0,
            "max_total_md_postcheck_m": 20000.0,
        }
    )
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["md_total_m"]) > 100.0


def test_recover_profile_from_build_and_kop_is_deterministic_for_reference_order() -> None:
    import pywp.planner as planner_module

    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(650.8546533242657, 919.0538224668888, 2681.8358083035228)
    t3 = Point3D(1981.7245876002041, 1405.5328339992557, 2774.652111333409)
    config = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        max_total_md_postcheck_m=25000.0,
    )
    geometry = planner_module._build_section_geometry(surface, t1, t3, config)
    post_entry = planner_module._solve_post_entry_section(
        ds_m=geometry.ds_13_m,
        dz_m=geometry.dz_13_m,
        inc_entry_deg=geometry.inc_entry_deg,
        dls_deg_per_30m=planner_module._resolve_horizontal_dls(config),
        max_inc_deg=float(config.max_inc_deg),
    )
    assert post_entry is not None
    zero_azimuth_turn = planner_module._is_zero_azimuth_turn_geometry(
        geometry=geometry,
        target_direction=planner_module.classify_trajectory_type(
            gv_m=float(geometry.z1_m),
            horizontal_offset_t1_m=planner_module._horizontal_offset(surface=surface, point=t1),
        ),
        tolerance_m=config.pos_tolerance_m,
    )
    assert zero_azimuth_turn is False

    md_params = planner_module._solve_turn_profile(
        surface=surface,
        geometry=geometry,
        config=config,
        optimization_context=None,
        zero_azimuth_turn=zero_azimuth_turn,
        search_settings=planner_module._turn_search_settings(0),
    ).params
    kop_params = planner_module._solve_turn_profile(
        surface=surface,
        geometry=geometry,
        config=_fast_config(
            kop_min_vertical_m=200.0,
            optimization_mode=OPTIMIZATION_MINIMIZE_KOP,
            max_total_md_postcheck_m=25000.0,
        ),
        optimization_context=None,
        zero_azimuth_turn=zero_azimuth_turn,
        search_settings=planner_module._turn_search_settings(0),
    ).params

    build_dls = 0.5 * (
        float(md_params.dls_build1_deg_per_30m)
        + float(kop_params.dls_build1_deg_per_30m)
    )
    kop_vertical_m = 0.5 * (
        float(md_params.kop_vertical_m)
        + float(kop_params.kop_vertical_m)
    )

    profile_a = planner_module._recover_profile_from_build_and_kop(
        geometry=geometry,
        build_dls_deg_per_30m=build_dls,
        kop_vertical_m=kop_vertical_m,
        min_build_segment_m=float(config.min_structural_segment_m),
        post_entry=post_entry,
        zero_azimuth_turn=zero_azimuth_turn,
        reference_candidates=(md_params, kop_params),
    )
    profile_b = planner_module._recover_profile_from_build_and_kop(
        geometry=geometry,
        build_dls_deg_per_30m=build_dls,
        kop_vertical_m=kop_vertical_m,
        min_build_segment_m=float(config.min_structural_segment_m),
        post_entry=post_entry,
        zero_azimuth_turn=zero_azimuth_turn,
        reference_candidates=(kop_params, md_params),
    )

    assert profile_a is not None
    assert profile_b is not None
    assert profile_a.md_total_m == pytest.approx(profile_b.md_total_m, abs=1e-6)
    assert profile_a.inc_hold_deg == pytest.approx(profile_b.inc_hold_deg, abs=1e-6)
    assert profile_a.hold_length_m == pytest.approx(profile_b.hold_length_m, abs=1e-6)
    assert profile_a.azimuth_hold_deg == pytest.approx(profile_b.azimuth_hold_deg, abs=1e-6)


def test_solver_relaxes_build_dls_below_max_for_reverse_entry_geometry() -> None:
    config = _fast_config(
        dls_build_min_deg_per_30m=0.0,
        dls_build_max_deg_per_30m=3.0,
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
    )
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(2500.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["build_dls_selected_deg_per_30m"]) < float(
        config.dls_build_max_deg_per_30m
    )
    assert 1.5 <= float(result.summary["build_dls_selected_deg_per_30m"]) <= 2.6
    assert str(result.summary["build_dls_relaxed_from_max"]) == "yes"
    assert int(result.summary["solver_turn_restarts_used"]) == 0


def test_kop_optimization_reduces_kop_for_reverse_entry_geometry() -> None:
    planner = TrajectoryPlanner()
    config_none = _fast_config(
        dls_build_min_deg_per_30m=0.0,
        dls_build_max_deg_per_30m=6.0,
        kop_min_vertical_m=200.0,
        optimization_mode="none",
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
    )
    config_kop = _fast_config(
        dls_build_min_deg_per_30m=0.0,
        dls_build_max_deg_per_30m=6.0,
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_KOP,
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
    )

    result_none = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(2500.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_none,
    )
    result_kop = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(2500.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config_kop,
    )

    assert float(result_kop.summary["distance_t1_m"]) <= config_kop.pos_tolerance_m
    assert float(result_kop.summary["distance_t3_m"]) <= config_kop.pos_tolerance_m
    assert float(result_kop.summary["kop_md_m"]) + 1.0 < float(result_none.summary["kop_md_m"])
    assert str(result_kop.summary["optimization_mode"]) == OPTIMIZATION_MINIMIZE_KOP
    assert str(result_kop.summary["optimization_status"]) in {"refined", "at_min_kop_limit"}
    assert int(result_kop.summary["optimization_runs_used"]) > 0


def test_kop_optimization_hits_minimum_kop_limit_for_shallow_turn_case() -> None:
    planner = TrajectoryPlanner()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(334.0, 46.0, 3769.0)
    t3 = Point3D(1464.0, 649.0, 3806.0)
    config_md = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
    )
    config_kop = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_KOP,
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
    )

    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=config_md)
    result_kop = planner.plan(surface=surface, t1=t1, t3=t3, config=config_kop)

    assert float(result_kop.summary["distance_t1_m"]) <= config_kop.pos_tolerance_m
    assert float(result_kop.summary["distance_t3_m"]) <= config_kop.pos_tolerance_m
    assert float(result_kop.summary["kop_md_m"]) == pytest.approx(550.0, abs=1e-6)
    assert float(result_kop.summary["kop_md_m"]) == pytest.approx(
        float(result_md.summary["kop_md_m"]),
        abs=1e-6,
    )
    assert float(result_kop.summary["md_total_m"]) <= float(result_md.summary["md_total_m"]) + 1e-6
    assert str(result_kop.summary["optimization_status"]) == "at_min_kop_limit"
    assert int(result_kop.summary["optimization_runs_used"]) <= 4


def test_md_optimization_skips_full_slsqp_when_2d_stage_confirms_boundary_extremum() -> None:
    planner = TrajectoryPlanner()
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS3.INC").read_text(encoding="utf-8")
    )
    target_record = next(record for record in records if str(record.name) == "well_02")
    surface, t1, t3 = welltrack_points_to_targets(target_record.points)
    config_md = TrajectoryConfig(
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        turn_solver_max_restarts=2,
    )
    config_kop = TrajectoryConfig(
        optimization_mode=OPTIMIZATION_MINIMIZE_KOP,
        turn_solver_max_restarts=2,
    )

    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=config_md)
    result_kop = planner.plan(surface=surface, t1=t1, t3=t3, config=config_kop)

    assert str(result_md.summary["optimization_status"]) == "at_md_boundary_extremum"
    assert int(result_md.summary["optimization_runs_used"]) <= 14
    assert float(result_md.summary["kop_md_m"]) == pytest.approx(
        float(result_kop.summary["kop_md_m"]),
        abs=1e-6,
    )
    assert float(result_md.summary["md_total_m"]) == pytest.approx(
        float(result_kop.summary["md_total_m"]),
        abs=1e-6,
    )
    assert float(result_md.summary["build_dls_selected_deg_per_30m"]) == pytest.approx(
        float(result_kop.summary["build_dls_selected_deg_per_30m"]),
        abs=1e-6,
    )


def test_fixed_high_build_dls_reports_reverse_entry_geometry() -> None:
    config = _fast_config(
        dls_build_min_deg_per_30m=3.0,
        dls_build_max_deg_per_30m=3.0,
        turn_solver_max_restarts=1,
        max_total_md_postcheck_m=20000.0,
    )

    with pytest.raises(PlanningError, match="reverse-entry geometry"):
        TrajectoryPlanner().plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(2500.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=config,
        )


def test_non_planar_turn_solver_least_squares_hits_targets() -> None:
    config = _fast_config(turn_solver_mode=TURN_SOLVER_LEAST_SQUARES, pos_tolerance_m=2.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["azimuth_turn_deg"]) > 1.0
    assert str(result.summary["solver_turn_mode"]) == TURN_SOLVER_LEAST_SQUARES
    assert int(result.summary["solver_turn_max_restarts"]) == int(config.turn_solver_max_restarts)
    assert int(result.summary["solver_turn_restarts_used"]) >= 0


def test_non_planar_solver_handles_post_entry_inc_drop_with_default_limits() -> None:
    config = TrajectoryConfig(turn_solver_max_restarts=0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(900.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["horizontal_inc_deg"]) < float(result.summary["entry_inc_deg"])
    assert int(result.summary["solver_turn_restarts_used"]) == 0
    assert float(result.summary["azimuth_turn_deg"]) > 1.0


@pytest.mark.slow
def test_non_planar_turn_solver_de_hybrid_hits_targets() -> None:
    config = _fast_config(turn_solver_mode=TURN_SOLVER_DE_HYBRID, pos_tolerance_m=2.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert float(result.summary["azimuth_turn_deg"]) > 1.0
    assert str(result.summary["solver_turn_mode"]) == TURN_SOLVER_DE_HYBRID


def test_reverse_direction_geometry_uses_turn_branch_and_solves() -> None:
    config = _fast_config(pos_tolerance_m=2.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 400.0, 3000.0),
        t3=Point3D(1020.0, 1360.0, 3083.9122),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert str(result.summary["trajectory_target_direction"]) == "Цели в обратном направлении"
    assert np.isfinite(float(result.summary["azimuth_turn_deg"]))


def test_turn_solver_retries_with_deeper_search_when_first_attempt_fails(monkeypatch) -> None:
    import pywp.planner as planner_module

    original = planner_module._solve_turn_profile
    seen_restarts: list[int] = []

    def flaky_turn_solver(*args, **kwargs):
        search_settings = kwargs["search_settings"]
        seen_restarts.append(int(search_settings.restart_index))
        if len(seen_restarts) == 1:
            raise PlanningError(
                "No valid trajectory solution found within configured limits. "
                "Closest miss to t1 is 7.87 m.\n"
                "Reasons and actions:\n"
                "- Solver endpoint miss to t1 after optimization is 7.87 m (tolerance 2.00 m). "
                "Best analytical delta: dX=1.00 m, dY=7.50 m, dZ=1.80 m."
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(planner_module, "_solve_turn_profile", flaky_turn_solver)
    config = _fast_config(turn_solver_max_restarts=1, pos_tolerance_m=2.0)

    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=config,
    )

    assert seen_restarts == [0, 1]
    assert int(result.summary["solver_turn_restarts_used"]) == 1
    assert float(result.summary["solver_turn_search_depth_scale"]) > 1.0


def test_planner_rejects_negative_turn_restart_budget() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _fast_config(turn_solver_max_restarts=-1)


def test_planner_keeps_solution_when_total_md_exceeds_soft_limit() -> None:
    config = _fast_config(max_total_md_postcheck_m=100.0)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["md_total_m"]) > float(config.max_total_md_postcheck_m)
    assert str(result.summary["md_postcheck_exceeded"]) == "yes"
    assert float(result.summary["md_postcheck_excess_m"]) > 0.0


def test_exact_endpoint_metrics_are_stable_across_control_grid_density() -> None:
    planner = TrajectoryPlanner()
    coarse = _fast_config(md_step_control_m=30.0)
    fine = _fast_config(md_step_control_m=0.5)

    result_coarse = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=coarse,
    )
    result_fine = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 300.0, 2000.0),
        t3=Point3D(900.0, 1200.0, 2075.0),
        config=fine,
    )

    for key in (
        "distance_t1_m",
        "distance_t3_m",
        "entry_inc_deg",
        "t1_exact_x_m",
        "t1_exact_y_m",
        "t1_exact_z_m",
        "t3_exact_x_m",
        "t3_exact_y_m",
        "t3_exact_z_m",
    ):
        assert float(result_coarse.summary[key]) == pytest.approx(
            float(result_fine.summary[key]),
            abs=1e-9,
        )


def test_planner_reports_exact_miss_components_in_failure_message() -> None:
    planner = TrajectoryPlanner()
    config = _fast_config(pos_tolerance_m=0.0001)

    with pytest.raises(PlanningError, match="Analytical delta: dX=.*dY=.*dZ="):
        planner.plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(300.0, 300.0, 2000.0),
            t3=Point3D(900.0, 1200.0, 2075.0),
            config=config,
        )


def test_planner_validates_negative_build_dls_bounds() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _fast_config(dls_build_min_deg_per_30m=-0.1)

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _fast_config(dls_build_max_deg_per_30m=-0.1)


def test_planner_validates_supported_entry_inc_target_range() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        _fast_config(entry_inc_target_deg=-5.0)

    with pytest.raises(ValidationError, match="less than 90"):
        _fast_config(entry_inc_target_deg=95.0, max_inc_deg=110.0)


def test_planner_rejects_unknown_turn_solver_mode() -> None:
    with pytest.raises(ValidationError, match="least_squares|de_hybrid"):
        _fast_config(turn_solver_mode="unsupported_turn_solver")  # type: ignore[arg-type]
