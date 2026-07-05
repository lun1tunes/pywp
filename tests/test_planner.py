from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pywp.anticollision_optimization import (
    AntiCollisionClearanceEvaluation,
    AntiCollisionOptimizationContext,
)
from pywp.eclipse_welltrack import parse_welltrack_text, welltrack_points_to_targets
from pywp.models import (
    J_PROFILE_POLICY_PREFER,
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    OPTIMIZATION_MINIMIZE_KOP,
    OPTIMIZATION_MINIMIZE_MD,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
    PlannerResult,
    Point3D,
    TrajectoryConfig,
)
from pywp.planner import PlanningError, TrajectoryPlanner
from pywp.planner_types import CandidateOptimizationEvaluation, ProfileParameters
from pywp.ptc_target_import_dev import parse_dev_target_file
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL

pytestmark = pytest.mark.integration


def _fast_config(**overrides: object) -> TrajectoryConfig:
    base = {
        "md_step_m": 10.0,
        "md_step_control_m": 2.0,
        "lateral_tolerance_m": 30.0,
        "vertical_tolerance_m": 2.0,
        "entry_inc_target_deg": 86.0,
        "entry_inc_tolerance_deg": 2.0,
        "dls_build_max_deg_per_30m": 6.0,
        "max_total_md_postcheck_m": 20000.0,
        "turn_solver_mode": TURN_SOLVER_LEAST_SQUARES,
    }
    base.update(overrides)
    return TrajectoryConfig(**base)


def _classic_j_reference_targets() -> tuple[Point3D, Point3D, Point3D]:
    entry_inc_deg = 60.0
    build_dls_deg_per_30m = 3.0
    radius_m = 30.0 * 180.0 / np.pi / build_dls_deg_per_30m
    build_lateral_m = radius_m * (1.0 - np.cos(np.radians(entry_inc_deg)))
    build_vertical_m = radius_m * np.sin(np.radians(entry_inc_deg))
    kop_vertical_m = 550.0
    horizontal_step_m = 1000.0
    t1_tvd_m = kop_vertical_m + build_vertical_m
    t3_tvd_m = t1_tvd_m + horizontal_step_m / np.tan(np.radians(entry_inc_deg))

    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(0.0, build_lateral_m, t1_tvd_m)
    t3 = Point3D(0.0, build_lateral_m + horizontal_step_m, t3_tvd_m)
    return surface, t1, t3


def test_plan_multi_target_delegates_to_plan_for_two_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = TrajectoryPlanner()
    surface = Point3D(0.0, 0.0, 0.0)
    targets = (
        Point3D(600.0, 800.0, 2400.0),
        Point3D(1500.0, 2000.0, 2500.0),
    )
    config = _fast_config()
    expected = PlannerResult(
        stations=pd.DataFrame({"MD_m": [0.0]}),
        summary={"trajectory_type": "delegated"},
        azimuth_deg=12.0,
        md_t1_m=345.0,
    )
    captured: dict[str, object] = {}

    def _fake_plan(
        self: TrajectoryPlanner,
        *,
        surface: Point3D,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
        progress_callback: object = None,
        optimization_context: object = None,
    ) -> PlannerResult:
        captured["surface"] = surface
        captured["t1"] = t1
        captured["t3"] = t3
        captured["config"] = config
        captured["optimization_context"] = optimization_context
        return expected

    monkeypatch.setattr(TrajectoryPlanner, "plan", _fake_plan)

    result = planner.plan_multi_target(
        surface=surface,
        targets=targets,
        config=config,
    )

    assert result is expected
    assert captured["surface"] == surface
    assert captured["t1"] == targets[0]
    assert captured["t3"] == targets[1]
    assert captured["config"] == config


def test_plan_multi_target_extends_base_plan_to_final_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywp.planner as planner_module

    planner = TrajectoryPlanner()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(100.0, 0.0, 200.0)
    t2 = Point3D(200.0, 0.0, 300.0)
    t3 = Point3D(260.0, 50.0, 330.0)
    config = _fast_config(max_total_md_postcheck_m=5000.0)
    base_result = PlannerResult(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 100.0, 250.0],
                "INC_deg": [0.0, 70.0, 86.0],
                "AZI_deg": [0.0, 0.0, 0.0],
                "X_m": [surface.x, t1.x, t2.x],
                "Y_m": [surface.y, t1.y, t2.y],
                "Z_m": [surface.z, t1.z, t2.z],
                "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
                "DLS_deg_per_30m": [0.0, 1.0, 1.5],
            }
        ),
        summary={
            "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
            "horizontal_length_m": 150.0,
            "max_total_md_postcheck_m": 5000.0,
            "md_total_m": 250.0,
            "md_postcheck_excess_m": 0.0,
            "md_postcheck_exceeded": "no",
        },
        azimuth_deg=0.0,
        md_t1_m=100.0,
    )
    validation_call: dict[str, object] = {}

    def _fake_plan(
        self: TrajectoryPlanner,
        *,
        surface: Point3D,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
        progress_callback: object = None,
        optimization_context: object = None,
    ) -> PlannerResult:
        return base_result

    def _fake_smooth_transition_rows(**kwargs: object) -> list[dict[str, object]]:
        assert kwargs["segment_name"] == "HORIZONTAL_BUILD1"
        return [
            {
                "MD_m": 340.0,
                "INC_deg": 87.0,
                "AZI_deg": 10.0,
                "X_m": 230.0,
                "Y_m": 20.0,
                "Z_m": 315.0,
                "segment": "HORIZONTAL_BUILD1",
                "DLS_deg_per_30m": 1.8,
            },
            {
                "MD_m": 430.0,
                "INC_deg": 88.0,
                "AZI_deg": 15.0,
                "X_m": t3.x,
                "Y_m": t3.y,
                "Z_m": t3.z,
                "segment": "HORIZONTAL_BUILD1",
                "DLS_deg_per_30m": 2.1,
            },
        ]

    def _fake_add_dls(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "DLS_deg_per_30m" not in out.columns:
            out["DLS_deg_per_30m"] = 0.0
        out["DLS_deg_per_30m"] = out["DLS_deg_per_30m"].fillna(0.0)
        return out

    def _fake_validate_extended_stations(
        *,
        stations: pd.DataFrame,
        final_target: Point3D,
        config: TrajectoryConfig,
        post_t1_start_md_m: float,
    ) -> None:
        validation_call["stations"] = stations.copy()
        validation_call["final_target"] = final_target
        validation_call["post_t1_start_md_m"] = post_t1_start_md_m

    monkeypatch.setattr(TrajectoryPlanner, "plan", _fake_plan)
    monkeypatch.setattr(
        planner_module,
        "_smooth_transition_rows",
        _fake_smooth_transition_rows,
    )
    monkeypatch.setattr(planner_module, "add_dls", _fake_add_dls)
    monkeypatch.setattr(
        planner_module,
        "_validate_extended_stations",
        _fake_validate_extended_stations,
    )

    result = planner.plan_multi_target(
        surface=surface,
        targets=(t1, t2, t3),
        config=config,
    )

    assert len(result.stations) == 5
    assert str(result.summary["trajectory_type"]) == "Target sequence"
    assert str(result.summary["target_sequence"]) == "yes"
    assert int(result.summary["target_sequence_point_count"]) == 3
    assert float(result.summary["t3_exact_x_m"]) == pytest.approx(t3.x)
    assert float(result.summary["t3_exact_y_m"]) == pytest.approx(t3.y)
    assert float(result.summary["t3_exact_z_m"]) == pytest.approx(t3.z)
    assert float(result.summary["distance_t3_m"]) == pytest.approx(0.0)
    assert float(result.summary["max_dls_total_deg_per_30m"]) == pytest.approx(2.1)
    assert float(result.summary["md_total_m"]) == pytest.approx(430.0)
    assert result.azimuth_deg == pytest.approx(15.0)
    assert validation_call["final_target"] == t3
    assert float(validation_call["post_t1_start_md_m"]) == pytest.approx(100.0)
    assert float(result.summary["horizontal_length_m"]) == pytest.approx(
        np.linalg.norm(np.array([t2.x - t1.x, t2.y - t1.y, t2.z - t1.z]))
        + np.linalg.norm(np.array([t3.x - t2.x, t3.y - t2.y, t3.z - t2.z]))
    )


def test_same_direction_reference_case_solves_with_minimum_kop() -> None:
    config = _fast_config(kop_min_vertical_m=550.0, offer_j_profile=False)
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.pos_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.pos_tolerance_m
    assert (
        abs(float(result.summary["entry_inc_deg"]) - config.entry_inc_target_deg)
        <= config.entry_inc_tolerance_deg
    )
    assert float(result.summary["kop_md_m"]) == pytest.approx(550.0, abs=1e-6)
    assert float(result.summary["azimuth_turn_deg"]) == pytest.approx(0.0, abs=1e-6)
    assert (
        str(result.summary["trajectory_type"])
        == "Unified J Profile + Build + Azimuth Turn"
    )
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


def test_planner_wraps_minimum_curvature_output_errors_as_planning_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywp.planner as planner_module

    def _boom(*args: object, **kwargs: object):
        raise ValueError("dogleg angle is too close to 180 degrees")

    monkeypatch.setattr(planner_module, "compute_positions_min_curv", _boom)

    with pytest.raises(PlanningError, match="минимальной кривизны"):
        TrajectoryPlanner().plan(
            surface=Point3D(0.0, 0.0, 0.0),
            t1=Point3D(600.0, 800.0, 2400.0),
            t3=Point3D(1500.0, 2000.0, 2500.0),
            config=_fast_config(kop_min_vertical_m=550.0),
        )


def test_higher_min_vertical_pushes_kop_up_deterministically() -> None:
    config_low = _fast_config(kop_min_vertical_m=550.0, offer_j_profile=False)
    config_high = _fast_config(kop_min_vertical_m=900.0, offer_j_profile=False)
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
    assert float(result_high.summary["md_total_m"]) >= float(
        result_low.summary["md_total_m"]
    )


def test_higher_build_dls_reduces_total_md() -> None:
    planner = TrajectoryPlanner()
    config_soft = _fast_config(dls_build_max_deg_per_30m=3.0, offer_j_profile=False)
    config_hard = _fast_config(dls_build_max_deg_per_30m=6.0, offer_j_profile=False)

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

    assert float(result_hard.summary["md_total_m"]) <= float(
        result_soft.summary["md_total_m"]
    )


def test_deep_low_angle_turn_solution_survives_relaxed_build_limit() -> None:
    record = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_DEBUG_1.INC").read_text()
    )[0]
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    planner = TrajectoryPlanner()
    config_08_pi = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=2.4,
        turn_solver_max_restarts=0,
    )
    config_10_pi = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        turn_solver_max_restarts=0,
    )

    result_08_pi = planner.plan(surface=surface, t1=t1, t3=t3, config=config_08_pi)
    result_10_pi = planner.plan(surface=surface, t1=t1, t3=t3, config=config_10_pi)

    assert float(result_08_pi.summary["distance_t1_m"]) <= config_08_pi.pos_tolerance_m
    assert float(result_10_pi.summary["distance_t1_m"]) <= config_10_pi.pos_tolerance_m
    assert float(result_10_pi.summary["md_total_m"]) <= float(
        result_08_pi.summary["md_total_m"]
    )
    assert float(result_10_pi.summary["kop_vertical_m"]) == pytest.approx(
        550.0, abs=1e-6
    )
    assert (
        str(result_10_pi.summary["trajectory_type"])
        == "Unified J Profile + Build + Azimuth Turn"
    )
    assert str(result_10_pi.summary["trajectory_profile_family"]) == "unified"
    assert (
        float(result_10_pi.summary["max_dls_total_deg_per_30m"])
        <= float(config_10_pi.dls_build_max_deg_per_30m) + 1e-6
    )
    assert int(result_10_pi.summary["solver_turn_restarts_used"]) == 0


def test_j_profile_proposal_does_not_label_spatial_turn_as_j_profile() -> None:
    record = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_DEBUG_1.INC").read_text()
    )[0]
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    planner = TrajectoryPlanner()
    messages_enabled: list[str] = []
    messages_disabled: list[str] = []

    enabled = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=_fast_config(
            kop_min_vertical_m=550.0,
            dls_build_max_deg_per_30m=3.0,
            turn_solver_max_restarts=0,
            offer_j_profile=True,
        ),
        progress_callback=lambda message, _fraction: messages_enabled.append(message),
    )
    disabled = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=_fast_config(
            kop_min_vertical_m=550.0,
            dls_build_max_deg_per_30m=3.0,
            turn_solver_max_restarts=0,
            offer_j_profile=False,
        ),
        progress_callback=lambda message, _fraction: messages_disabled.append(message),
    )

    assert str(enabled.summary["trajectory_type"]) == (
        "Unified J Profile + Build + Azimuth Turn"
    )
    assert str(enabled.summary["trajectory_profile_family"]) == "unified"
    assert str(disabled.summary["trajectory_type"]) == (
        "Unified J Profile + Build + Azimuth Turn"
    )
    assert str(disabled.summary["trajectory_profile_family"]) == "unified"
    assert not any("J-образные кандидаты" in message for message in messages_enabled)
    assert any("fallback-поиск" in message for message in messages_enabled)
    assert any("fallback-поиск" in message for message in messages_disabled)


def test_classic_j_profile_uses_single_build_without_hold_or_second_build() -> None:
    from pywp.planner import (
        _evaluate_profile_candidate,
        _profile_single_build_j,
        _resolve_horizontal_dls,
        _solve_post_entry_section,
    )
    from pywp.planner_geometry import _build_section_geometry
    from pywp.planner_validation import _build_trajectory

    entry_inc_deg = 60.0
    build_dls_deg_per_30m = 3.0
    kop_vertical_m = 550.0
    surface, t1, t3 = _classic_j_reference_targets()
    config = _fast_config(
        kop_min_vertical_m=kop_vertical_m,
        entry_inc_target_deg=entry_inc_deg,
        max_inc_deg=70.0,
        dls_build_min_deg_per_30m=0.1,
        dls_build_max_deg_per_30m=build_dls_deg_per_30m,
        turn_solver_max_restarts=0,
    )

    geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
    post_entry = _solve_post_entry_section(
        ds_m=geometry.ds_13_m,
        dz_m=geometry.dz_13_m,
        inc_entry_deg=geometry.inc_entry_deg,
        dls_deg_per_30m=_resolve_horizontal_dls(config=config),
        max_inc_deg=float(config.max_inc_deg),
    )
    assert post_entry is not None

    profile = _profile_single_build_j(
        geometry=geometry,
        config=config,
        post_entry=post_entry,
        build_dls_lower_deg_per_30m=0.1,
        build_dls_upper_deg_per_30m=build_dls_deg_per_30m,
    )
    assert profile is not None
    trajectory = _build_trajectory(profile)

    assert str(profile.profile_family) == "j_profile"
    assert float(profile.hold_length_m) == pytest.approx(0.0, abs=1e-9)
    assert float(profile.build2_length_m) == pytest.approx(0.0, abs=1e-9)
    assert float(profile.dls_build2_deg_per_30m) == pytest.approx(
        0.0,
        abs=1e-9,
    )
    optimization_eval = _evaluate_profile_candidate(
        candidate=profile,
        target_point=np.array([t1.x, t1.y, t1.z], dtype=float),
        config=config,
    )
    assert optimization_eval.feasible
    assert float(optimization_eval.build2_margin_m) == pytest.approx(0.0, abs=1e-9)
    assert [segment.name for segment in trajectory.segments] == [
        "VERTICAL",
        "BUILD1",
        "HORIZONTAL",
    ]


def test_post_entry_solver_uses_horizontal_dls_limit_independent_from_build() -> None:
    import pywp.planner as planner_module

    config = _fast_config(
        dls_build_max_deg_per_30m=6.0,
        dls_horizontal_max_deg_per_30m=1.0,
    )
    assert planner_module._resolve_horizontal_dls(config) == pytest.approx(1.0)

    constrained = planner_module._solve_post_entry_section(
        ds_m=400.0,
        dz_m=100.0,
        inc_entry_deg=86.0,
        dls_deg_per_30m=planner_module._resolve_horizontal_dls(config),
        max_inc_deg=float(config.max_inc_deg),
    )
    relaxed = planner_module._solve_post_entry_section(
        ds_m=400.0,
        dz_m=100.0,
        inc_entry_deg=86.0,
        dls_deg_per_30m=planner_module._resolve_horizontal_dls(
            config.validated_copy(dls_horizontal_max_deg_per_30m=2.0)
        ),
        max_inc_deg=float(config.max_inc_deg),
    )

    assert constrained is None
    assert relaxed is not None
    assert relaxed.transition_dls_deg_per_30m == pytest.approx(2.0)


def test_planner_applies_horizontal_dls_limit_to_post_entry_section() -> None:
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(1000.0, 0.0, 2500.0)
    t3 = Point3D(1400.0, 0.0, 2600.0)
    constrained = _fast_config(
        dls_build_max_deg_per_30m=6.0,
        dls_horizontal_max_deg_per_30m=1.0,
        kop_min_vertical_m=550.0,
        turn_solver_max_restarts=0,
    )
    relaxed = constrained.validated_copy(dls_horizontal_max_deg_per_30m=2.0)

    with pytest.raises(PlanningError, match="HORIZONTAL DLS limit"):
        TrajectoryPlanner().plan(surface=surface, t1=t1, t3=t3, config=constrained)

    result = TrajectoryPlanner().plan(surface=surface, t1=t1, t3=t3, config=relaxed)

    assert (
        float(result.summary["max_dls_horizontal_deg_per_30m"])
        <= float(relaxed.dls_horizontal_max_deg_per_30m) + 1e-6
    )
    assert (
        float(result.summary["max_dls_total_deg_per_30m"])
        > float(relaxed.dls_horizontal_max_deg_per_30m) + 1e-6
    )


def test_variable_build_j_profile_handles_non_coplanar_entry_without_hold() -> None:
    import pywp.planner as planner_module
    from pywp.planner_geometry import _build_section_geometry, _horizontal_offset

    surface = Point3D(377899.9, 930000.4, -20.0)
    t1 = Point3D(377309.9, 929820.8, 3701.51)
    t3 = Point3D(376307.3, 929590.1, 3749.49)
    config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        optimization_mode="none",
        turn_solver_max_restarts=0,
        offer_j_profile=True,
    )

    result = TrajectoryPlanner().plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=config,
    )

    assert str(result.summary["trajectory_type"]) == "J-образная траектория"
    assert str(result.summary["trajectory_profile_family"]) == "j_profile"
    assert float(result.summary["distance_t1_m"]) <= 1e-6
    assert float(result.summary["hold_length_m"]) == pytest.approx(0.0, abs=1e-9)
    assert float(result.summary["build2_dls_selected_deg_per_30m"]) == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert list(result.stations["segment"].drop_duplicates()) == [
        "VERTICAL",
        "BUILD1",
        "HORIZONTAL",
    ]
    assert float(result.stations["DLS_deg_per_30m"].max()) <= 3.0 + 1e-6
    geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
    zero_azimuth_turn = bool(abs(float(geometry.t1_cross_m)) <= 1e-9)
    params, _optimization, *_rest = planner_module._solve_turn_with_restarts(
        surface=surface,
        t1=t1,
        t3=t3,
        geometry=geometry,
        horizontal_offset_t1_m=_horizontal_offset(surface, t1),
        config=config,
        optimization_context=None,
        zero_azimuth_turn=zero_azimuth_turn,
        progress_callback=None,
    )
    assert str(params.profile_family) == "j_profile"
    assert len(params.build1_controls) >= 2
    az_mid_deg = float(params.build1_controls[0][2])
    azimuth_excess_deg = planner_module._azimuth_shortest_arc_excess_deg(
        float(geometry.azimuth_surface_t1_deg),
        az_mid_deg,
        float(geometry.azimuth_entry_deg),
    )
    assert azimuth_excess_deg > 1e-6
    assert azimuth_excess_deg <= (
        planner_module.VARIABLE_J_AZIMUTH_EXCESS_TOLERANCE_DEG + 1e-6
    )


@pytest.mark.parametrize(
    "well_name",
    [
        "well_01",
        "well_02",
        "well_03",
        "well_07",
        "well_09",
        "well_11",
    ],
)
def test_reverse_variable_j_regression_cases_fall_back_to_unified_profile(
    well_name: str,
) -> None:
    import pywp.planner as planner_module
    from pywp.planner_geometry import _build_section_geometry, _horizontal_offset

    record = next(
        item
        for item in parse_welltrack_text(Path("tests/test_data/WELLTRACKS4.INC").read_text())
        if str(item.name) == well_name
    )
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        optimization_mode="none",
        turn_solver_max_restarts=0,
        offer_j_profile=True,
    )
    geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
    zero_azimuth_turn = bool(abs(float(geometry.t1_cross_m)) <= 1e-9)

    params, _optimization, *_rest = planner_module._solve_turn_with_restarts(
        surface=surface,
        t1=t1,
        t3=t3,
        geometry=geometry,
        horizontal_offset_t1_m=_horizontal_offset(surface, t1),
        config=config,
        optimization_context=None,
        zero_azimuth_turn=zero_azimuth_turn,
        progress_callback=None,
    )
    assert str(params.profile_family) == "unified"


def test_reverse_variable_j_candidate_falls_back_to_unified_when_relaxed_build_allows_it() -> None:
    record = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_DEBUG_1.INC").read_text()
    )[0]
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    planner = TrajectoryPlanner()
    messages: list[str] = []

    result = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=_fast_config(
            kop_min_vertical_m=550.0,
            dls_build_max_deg_per_30m=6.0,
            optimization_mode="none",
            turn_solver_max_restarts=0,
            offer_j_profile=True,
        ),
        progress_callback=lambda message, _fraction: messages.append(message),
    )
    assert str(result.summary["trajectory_type"]) == (
        "Unified J Profile + Build + Azimuth Turn"
    )
    assert str(result.summary["trajectory_profile_family"]) == "unified"
    assert any("сначала проверяем classic J-профиль" in item for item in messages)
    assert not any("classic J-профиль принят" in item for item in messages)


def test_minimize_md_keeps_shorter_unified_profile_over_feasible_j_profile() -> None:
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACK_ERD.INC").read_text(encoding="utf-8")
    )
    surface, t1, t3 = welltrack_points_to_targets(records[0].points)
    result = TrajectoryPlanner().plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=_fast_config(
            max_total_md_postcheck_m=9000.0,
            turn_solver_max_restarts=0,
            offer_j_profile=True,
        ),
    )

    assert str(result.summary["trajectory_profile_family"]) == "unified"
    assert 67.0 <= float(result.summary["hold_inc_deg"]) <= 73.0


def test_prefer_j_profile_policy_accepts_classic_j_reference_case() -> None:
    surface, t1, t3 = _classic_j_reference_targets()

    result = TrajectoryPlanner().plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=_fast_config(
            kop_min_vertical_m=550.0,
            dls_build_max_deg_per_30m=3.0,
            entry_inc_target_deg=60.0,
            max_inc_deg=70.0,
            turn_solver_max_restarts=0,
            j_profile_policy=J_PROFILE_POLICY_PREFER,
        ),
    )

    assert str(result.summary["optimization_mode"]) == OPTIMIZATION_MINIMIZE_MD
    assert str(result.summary["optimization_status"]) == "analytic_j_profile"
    assert str(result.summary["trajectory_type"]) == "J-образная траектория"
    assert str(result.summary["trajectory_profile_family"]) == "j_profile"
    assert list(result.stations["segment"].drop_duplicates()) == [
        "VERTICAL",
        "BUILD1",
        "HORIZONTAL",
    ]


def test_split_build_rescue_can_find_independent_build_candidate() -> None:
    import pywp.planner as planner_module
    from pywp.planner_geometry import _build_section_geometry
    from pywp.planner_validation import _estimate_t1_endpoint_for_profile

    record = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_DEBUG_1.INC").read_text()
    )[0]
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        turn_solver_max_restarts=0,
    )
    geometry = _build_section_geometry(
        surface=surface,
        t1=t1,
        t3=t3,
        config=config,
    )
    build_dls_upper = planner_module._resolve_build_dls_max(
        config=config,
        constrained_segments=("BUILD1", "BUILD2"),
    )
    build_dls_lower = planner_module._effective_build_dls_lower_bound(
        config=config,
        upper_dls_deg_per_30m=build_dls_upper,
    )
    post_entry = planner_module._solve_post_entry_section(
        ds_m=geometry.ds_13_m,
        dz_m=geometry.dz_13_m,
        inc_entry_deg=geometry.inc_entry_deg,
        dls_deg_per_30m=planner_module._resolve_horizontal_dls(config=config),
        max_inc_deg=float(config.max_inc_deg),
    )
    assert post_entry is not None

    bounds = (
        (0.0, 1.0),
        (
            float(max(config.kop_min_vertical_m, 0.0)),
            float(max(geometry.z1_m - 1e-9, config.kop_min_vertical_m + 1e-9)),
        ),
        (0.5, float(geometry.inc_entry_deg - 0.5)),
        (0.0, float(max(geometry.s1_m + geometry.z1_m, 1000.0))),
        (0.0, 360.0),
    )
    search_settings = planner_module._turn_search_settings(restart_index=0)
    base_seed_vectors = planner_module._turn_seed_vectors(
        geometry=geometry,
        build_dls_lower_deg_per_30m=build_dls_lower,
        build_dls_upper_deg_per_30m=build_dls_upper,
        bounds=bounds,
        search_settings=search_settings,
        zero_azimuth_turn=False,
    )
    target_point = np.array(
        [geometry.t1_east_m, geometry.t1_north_m, geometry.t1_tvd_m],
        dtype=float,
    )

    candidates, best = planner_module._collect_split_build_turn_candidates(
        geometry=geometry,
        build_dls_lower_deg_per_30m=build_dls_lower,
        build_dls_upper_deg_per_30m=build_dls_upper,
        bounds=bounds,
        base_seed_vectors=base_seed_vectors,
        post_entry=post_entry,
        target_point=target_point,
        config=config,
        search_settings=search_settings,
        zero_azimuth_turn=False,
    )

    assert candidates
    candidate = candidates[0]
    endpoint = np.array(_estimate_t1_endpoint_for_profile(candidate), dtype=float)
    _, _, _, lateral_m, vertical_m, _ = planner_module._target_miss_components(
        endpoint,
        target_point,
    )
    assert lateral_m <= config.lateral_tolerance_m
    assert vertical_m <= config.vertical_tolerance_m
    assert candidate.dls_build1_deg_per_30m != pytest.approx(
        candidate.dls_build2_deg_per_30m,
        abs=1e-3,
    )
    assert max(
        candidate.dls_build1_deg_per_30m,
        candidate.dls_build2_deg_per_30m,
    ) <= build_dls_upper + 1e-6
    assert best[0] < 1e-6


def test_planner_respects_explicit_separate_build2_limit() -> None:
    record = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_DEBUG_1.INC").read_text()
    )[0]
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        dls_build2_max_deg_per_30m=5.4,
        turn_solver_max_restarts=0,
    )

    result = TrajectoryPlanner().plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.lateral_tolerance_m
    assert float(result.summary["build1_dls_selected_deg_per_30m"]) <= 3.0 + 1e-6
    assert float(result.summary["build2_dls_selected_deg_per_30m"]) <= 5.4 + 1e-6
    assert float(result.summary["build2_dls_selected_deg_per_30m"]) > 3.0 + 1e-3


def test_split_build_md_search_keeps_later_shorter_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywp.planner as planner_module

    worse_candidate = ProfileParameters(
        kop_vertical_m=600.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=5.0,
        dls_build2_deg_per_30m=2.8,
        build1_length_m=300.0,
        hold_length_m=400.0,
        build2_length_m=250.0,
        horizontal_length_m=1100.0,
        horizontal_adjust_length_m=100.0,
        horizontal_hold_length_m=1000.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=1.5,
        azimuth_hold_deg=35.0,
        azimuth_entry_deg=20.0,
    )
    better_candidate = ProfileParameters(
        kop_vertical_m=600.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=48.0,
        dls_build1_deg_per_30m=4.8,
        dls_build2_deg_per_30m=2.6,
        build1_length_m=260.0,
        hold_length_m=260.0,
        build2_length_m=220.0,
        horizontal_length_m=1050.0,
        horizontal_adjust_length_m=90.0,
        horizontal_hold_length_m=960.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=1.5,
        azimuth_hold_deg=28.0,
        azimuth_entry_deg=20.0,
    )
    seed_vectors = [
        np.array([1.0, 0.8, 600.0, 52.0, 400.0, 35.0], dtype=float),
        np.array([0.6, 0.3, 600.0, 48.0, 260.0, 28.0], dtype=float),
    ]

    monkeypatch.setattr(
        planner_module,
        "_split_build_rescue_seed_vectors",
        lambda **kwargs: [seed.copy() for seed in seed_vectors],
    )
    monkeypatch.setattr(
        planner_module,
        "_turn_least_squares_probes",
        lambda **kwargs: [np.asarray(kwargs["seed_vector"], dtype=float)],
    )
    monkeypatch.setattr(
        planner_module,
        "_make_turn_profile_builder",
        lambda **kwargs: (
            lambda values: worse_candidate
            if float(np.asarray(values, dtype=float)[0]) > 0.9
            else better_candidate
        ),
    )
    monkeypatch.setattr(
        planner_module,
        "_estimate_t1_endpoint_for_profile",
        lambda candidate: (0.0, 0.0, 0.0),
    )

    config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=5.4,
        dls_build2_max_deg_per_30m=3.0,
        dls_horizontal_max_deg_per_30m=2.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        offer_j_profile=False,
        turn_solver_max_restarts=0,
    )
    geometry = planner_module.SectionGeometry(
        s1_m=1200.0,
        z1_m=2600.0,
        ds_13_m=1100.0,
        dz_13_m=120.0,
        azimuth_entry_deg=35.0,
        azimuth_surface_t1_deg=20.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        t1_cross_m=10.0,
        t3_cross_m=0.0,
        t1_east_m=0.0,
        t1_north_m=0.0,
        t1_tvd_m=0.0,
    )
    post_entry = planner_module.PostEntrySection(
        total_length_m=1200.0,
        transition_length_m=200.0,
        hold_length_m=1000.0,
        hold_inc_deg=85.0,
        transition_dls_deg_per_30m=1.5,
    )
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )

    candidates, _ = planner_module._collect_split_build_turn_candidates(
        geometry=geometry,
        build_dls_lower_deg_per_30m=0.1,
        build_dls_upper_deg_per_30m=5.4,
        build2_dls_upper_deg_per_30m=3.0,
        bounds=bounds,
        base_seed_vectors=[],
        post_entry=post_entry,
        target_point=np.zeros(3, dtype=float),
        config=config,
        search_settings=planner_module._turn_search_settings(0),
        zero_azimuth_turn=False,
    )

    assert candidates == [better_candidate, worse_candidate]


def test_planner_respects_build2_limit_when_build1_limit_is_higher() -> None:
    record = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_DEBUG_1.INC").read_text()
    )[0]
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    split_config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=5.4,
        dls_build2_max_deg_per_30m=3.0,
        turn_solver_max_restarts=0,
    )
    control_config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        turn_solver_max_restarts=0,
    )

    planner = TrajectoryPlanner()
    control = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=control_config,
    )
    result = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=split_config,
    )

    assert float(result.summary["distance_t1_m"]) <= split_config.lateral_tolerance_m
    assert float(result.summary["build1_dls_selected_deg_per_30m"]) <= 5.4 + 1e-6
    assert float(result.summary["build2_dls_selected_deg_per_30m"]) <= 3.0 + 1e-6
    assert float(result.summary["md_total_m"]) <= float(control.summary["md_total_m"]) + 1e-6


def test_split_build_limits_do_not_worsen_md_vs_base_build2_limit_case() -> None:
    records = parse_welltrack_text(Path("tests/test_data/WELLTRACKS4.INC").read_text())
    record = next(item for item in records if str(item.name) == "well_02")
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    common_kwargs = dict(
        kop_min_vertical_m=550.0,
        dls_horizontal_max_deg_per_30m=1.5,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        offer_j_profile=False,
        turn_solver_max_restarts=0,
    )
    control_config = _fast_config(
        dls_build_max_deg_per_30m=1.8,
        **common_kwargs,
    )
    split_config = _fast_config(
        dls_build_max_deg_per_30m=2.4,
        dls_build2_max_deg_per_30m=1.8,
        **common_kwargs,
    )

    planner = TrajectoryPlanner()
    control = planner.plan(surface=surface, t1=t1, t3=t3, config=control_config)
    split = planner.plan(surface=surface, t1=t1, t3=t3, config=split_config)

    assert float(split.summary["distance_t1_m"]) <= split_config.lateral_tolerance_m
    assert float(split.summary["distance_t3_m"]) <= split_config.lateral_tolerance_m
    assert float(split.summary["build1_dls_selected_deg_per_30m"]) <= 2.4 + 1e-6
    assert float(split.summary["build2_dls_selected_deg_per_30m"]) <= 1.8 + 1e-6
    assert float(split.summary["md_total_m"]) <= float(control.summary["md_total_m"]) + 1e-6


def test_split_build_limits_keep_equal_build_baseline_for_welltrack4_well_08() -> None:
    records = parse_welltrack_text(Path("tests/test_data/WELLTRACKS4.INC").read_text())
    record = next(item for item in records if str(item.name) == "well_08")
    surface, t1, t3 = welltrack_points_to_targets(record.points)
    common_kwargs = dict(
        md_step_m=10.0,
        md_step_control_m=2.0,
        lateral_tolerance_m=30.0,
        vertical_tolerance_m=2.0,
        entry_inc_target_deg=86.0,
        entry_inc_tolerance_deg=2.0,
        dls_horizontal_max_deg_per_30m=1.5,
        kop_min_vertical_m=550.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        offer_j_profile=False,
        turn_solver_max_restarts=0,
    )
    control_config = TrajectoryConfig(
        dls_build_max_deg_per_30m=1.8,
        **common_kwargs,
    )
    split_config = TrajectoryConfig(
        dls_build_max_deg_per_30m=2.4,
        dls_build2_max_deg_per_30m=1.8,
        **common_kwargs,
    )

    planner = TrajectoryPlanner()
    control = planner.plan(surface=surface, t1=t1, t3=t3, config=control_config)
    split = planner.plan(surface=surface, t1=t1, t3=t3, config=split_config)

    assert float(split.summary["distance_t1_m"]) <= split_config.lateral_tolerance_m
    assert float(split.summary["distance_t3_m"]) <= split_config.lateral_tolerance_m
    assert float(split.summary["build1_dls_selected_deg_per_30m"]) <= 2.4 + 1e-6
    assert float(split.summary["build2_dls_selected_deg_per_30m"]) <= 1.8 + 1e-6
    assert float(split.summary["md_total_m"]) <= float(control.summary["md_total_m"]) + 1e-6


def test_zero_azimuth_turn_does_not_trigger_split_build_rescue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywp.planner as planner_module

    def _boom(**kwargs: object):
        raise AssertionError("split-build rescue should not run for zero-azimuth turns")

    monkeypatch.setattr(planner_module, "_collect_split_build_turn_candidates", _boom)

    config = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        dls_build2_max_deg_per_30m=5.4,
        turn_solver_max_restarts=0,
        offer_j_profile=False,
    )
    result = TrajectoryPlanner().plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        config=config,
    )

    assert float(result.summary["distance_t1_m"]) <= config.lateral_tolerance_m


def test_zero_azimuth_turn_respects_independent_build2_limit_for_dev_fixture() -> None:
    parsed = parse_dev_target_file(
        Path("tests/test_data/dev_target_import/build_hold_build_equal_pi_with_horizontal_pi.dev")
    )
    surface, t1, t3 = parsed.record.points
    control_config = _fast_config(
        md_step_m=30.0,
        md_step_control_m=10.0,
        entry_inc_target_deg=float(parsed.summary.entry_inc_deg),
        max_inc_deg=90.0,
        kop_min_vertical_m=float(parsed.summary.kop_md_m),
        dls_build_max_deg_per_30m=1.8,
        dls_horizontal_max_deg_per_30m=1.5,
        turn_solver_max_restarts=0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        offer_j_profile=False,
    )
    split_config = _fast_config(
        md_step_m=30.0,
        md_step_control_m=10.0,
        entry_inc_target_deg=float(parsed.summary.entry_inc_deg),
        max_inc_deg=90.0,
        kop_min_vertical_m=float(parsed.summary.kop_md_m),
        dls_build_max_deg_per_30m=2.4,
        dls_build2_max_deg_per_30m=1.8,
        dls_horizontal_max_deg_per_30m=1.5,
        turn_solver_max_restarts=0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        offer_j_profile=False,
    )

    planner = TrajectoryPlanner()
    control = planner.plan(surface=surface, t1=t1, t3=t3, config=control_config)
    split = planner.plan(surface=surface, t1=t1, t3=t3, config=split_config)

    assert float(split.summary["distance_t1_m"]) <= split_config.lateral_tolerance_m
    assert float(split.summary["distance_t3_m"]) <= split_config.lateral_tolerance_m
    assert float(split.summary["build1_dls_selected_deg_per_30m"]) <= 2.4 + 1e-6
    assert float(split.summary["build2_dls_selected_deg_per_30m"]) <= 1.8 + 1e-6
    assert float(split.summary["build1_dls_selected_deg_per_30m"]) > float(
        split.summary["build2_dls_selected_deg_per_30m"]
    ) + 1e-3
    assert float(split.summary["md_total_m"]) < float(control.summary["md_total_m"]) - 1e-3


def test_zero_azimuth_turn_split_build_remains_feasible_with_fixed_kop() -> None:
    parsed = parse_dev_target_file(
        Path("tests/test_data/dev_target_import/build_hold_build_equal_pi_with_horizontal_pi.dev")
    )
    surface, t1, t3 = parsed.record.points
    config = _fast_config(
        md_step_m=30.0,
        md_step_control_m=10.0,
        entry_inc_target_deg=float(parsed.summary.entry_inc_deg),
        max_inc_deg=90.0,
        kop_min_vertical_m=float(parsed.summary.kop_md_m),
        use_fixed_kop=True,
        dls_build_max_deg_per_30m=2.4,
        dls_build2_max_deg_per_30m=1.8,
        dls_horizontal_max_deg_per_30m=1.5,
        turn_solver_max_restarts=0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        offer_j_profile=False,
    )

    result = TrajectoryPlanner().plan(surface=surface, t1=t1, t3=t3, config=config)

    assert float(result.summary["distance_t1_m"]) <= config.lateral_tolerance_m
    assert float(result.summary["distance_t3_m"]) <= config.lateral_tolerance_m
    assert float(result.summary["kop_md_m"]) == pytest.approx(
        float(parsed.summary.kop_md_m),
        abs=1e-6,
    )
    assert float(result.summary["build1_dls_selected_deg_per_30m"]) <= 2.4 + 1e-6
    assert float(result.summary["build2_dls_selected_deg_per_30m"]) <= 1.8 + 1e-6
    assert float(result.summary["build1_dls_selected_deg_per_30m"]) >= float(
        result.summary["build2_dls_selected_deg_per_30m"]
    ) - 1e-6


def test_post_entry_solver_keeps_boundary_case_within_numerical_tolerance() -> None:
    import pywp.planner as planner_module
    from pywp.mcm import minimum_curvature_increment

    post_entry = planner_module._solve_post_entry_section(
        ds_m=1000.0,
        dz_m=0.0,
        inc_entry_deg=86.0,
        dls_deg_per_30m=3.6,
        max_inc_deg=95.0,
    )

    assert post_entry is not None
    dn_arc, _, dz_arc = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=86.0,
        azi1_deg=0.0,
        md2_m=float(post_entry.transition_length_m),
        inc2_deg=float(post_entry.hold_inc_deg),
        azi2_deg=0.0,
    )
    hold_inc_rad = np.radians(float(post_entry.hold_inc_deg))
    ds_pred = float(dn_arc + post_entry.hold_length_m * np.sin(hold_inc_rad))
    dz_pred = float(dz_arc + post_entry.hold_length_m * np.cos(hold_inc_rad))

    assert ds_pred == pytest.approx(1000.0, abs=1e-6)
    assert dz_pred == pytest.approx(0.0, abs=1e-6)


def test_md_optimization_stops_within_theoretical_gap_when_solution_is_already_short() -> (
    None
):
    planner = TrajectoryPlanner()
    config_none = _fast_config(kop_min_vertical_m=200.0, offer_j_profile=False)
    config_md = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        offer_j_profile=False,
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
    assert (
        float(result_md.summary["md_total_m"])
        <= float(result_none.summary["md_total_m"]) + 1e-6
    )
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


def test_anti_collision_selection_prefers_clearance_improving_candidate(
    monkeypatch,
) -> None:
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
        if float(candidate.kop_vertical_m) == pytest.approx(
            improved_candidate.kop_vertical_m
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
        if float(candidate.dls_build1_deg_per_30m) == pytest.approx(
            improved_candidate.dls_build1_deg_per_30m
        ) and float(candidate.dls_build2_deg_per_30m) == pytest.approx(
            improved_candidate.dls_build2_deg_per_30m
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


def test_anti_collision_selection_caches_repeated_vector_evaluations(
    monkeypatch,
) -> None:
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
        offer_j_profile=False,
    )
    config_md = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        max_total_md_postcheck_m=25000.0,
        offer_j_profile=False,
    )

    result_none = planner.plan(surface=surface, t1=t1, t3=t3, config=config_none)
    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=config_md)

    assert float(result_md.summary["distance_t1_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["distance_t3_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["md_total_m"]) + 100.0 < float(
        result_none.summary["md_total_m"]
    )
    assert (
        str(result_md.summary["optimization_status"])
        == "within_md_theoretical_gap_after_boundary"
    )
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


def test_anti_collision_early_stage_uses_fast_seed_selection_without_local_slsqp(
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

    def _unexpected_minimize(*args, **kwargs):
        raise AssertionError("Early anti-collision stage should not enter local SLSQP.")

    monkeypatch.setattr(planner_module, "minimize", _unexpected_minimize)

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

    assert result.optimization.runs_used == 0
    assert result.optimization.status in {"seed_selected", "clearance_improved"}


def test_anti_collision_candidate_can_focus_late_build2_without_moving_early_profile(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module

    baseline_candidate = ProfileParameters(
        kop_vertical_m=700.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=50.0,
        dls_build1_deg_per_30m=4.2,
        dls_build2_deg_per_30m=2.6,
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
    improved_candidate = ProfileParameters(
        kop_vertical_m=700.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=49.0,
        dls_build1_deg_per_30m=4.2,
        dls_build2_deg_per_30m=5.4,
        build1_length_m=280.0,
        hold_length_m=360.0,
        build2_length_m=290.0,
        horizontal_length_m=1210.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1100.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=36.0,
        azimuth_entry_deg=67.0,
    )
    wild_candidate = ProfileParameters(
        kop_vertical_m=1650.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=58.0,
        dls_build1_deg_per_30m=5.1,
        dls_build2_deg_per_30m=5.1,
        build1_length_m=360.0,
        hold_length_m=240.0,
        build2_length_m=330.0,
        horizontal_length_m=1160.0,
        horizontal_adjust_length_m=90.0,
        horizontal_hold_length_m=1070.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=44.0,
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
        candidate_md_start_m=4000.0,
        candidate_md_end_m=4350.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
        prefer_keep_kop=True,
        prefer_keep_build1=True,
        prefer_adjust_build2=True,
    )
    _ = planner_module._candidate_to_search_vector(
        candidate=improved_candidate,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        split_build=True,
    )
    minimize_shapes: list[tuple[int, ...]] = []

    def fake_clearance(
        *,
        candidate: ProfileParameters,
        surface: Point3D,
        context: AntiCollisionOptimizationContext,
    ) -> AntiCollisionClearanceEvaluation:
        if candidate == wild_candidate:
            return AntiCollisionClearanceEvaluation(
                min_separation_factor=1.12,
                max_overlap_depth_m=0.0,
            )
        if float(candidate.dls_build2_deg_per_30m) >= 5.0:
            return AntiCollisionClearanceEvaluation(
                min_separation_factor=1.04,
                max_overlap_depth_m=0.0,
            )
        return AntiCollisionClearanceEvaluation(
            min_separation_factor=0.42,
            max_overlap_depth_m=32.0,
        )

    def fake_evaluate_candidate(
        *,
        values: np.ndarray,
        profile_builder,
        target_point: np.ndarray,
        config: TrajectoryConfig,
    ) -> CandidateOptimizationEvaluation:
        assert float(values[0]) == pytest.approx(
            planner_module._candidate_to_search_vector(
                candidate=baseline_candidate,
                zero_azimuth_turn=False,
                lower_dls_deg_per_30m=0.1,
                upper_dls_deg_per_30m=6.0,
                bounds=bounds,
                split_build=True,
            )[0],
            abs=1e-9,
        )
        assert float(values[2]) == pytest.approx(700.0, abs=1e-9)
        candidate = improved_candidate if float(values[1]) > 0.8 else baseline_candidate
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
        minimize_shapes.append(np.asarray(x0, dtype=float).shape)
        assert method == "SLSQP"
        return SimpleNamespace(
            success=True, x=np.asarray([1.0, x0[1], x0[2], x0[3]], dtype=float)
        )

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
        candidates=[baseline_candidate, wild_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        config=config,
        optimization_context=context,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        optimization_bounds=bounds,
        profile_builder=lambda values: baseline_candidate,
        optimization_profile_builder=lambda values: baseline_candidate,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
        split_build=True,
    )

    assert result.params == improved_candidate
    assert minimize_shapes
    assert all(shape == (4,) for shape in minimize_shapes)


def test_anti_collision_late_build2_stage_raises_clean_error_when_no_preserved_profile_candidate_exists(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module

    wild_candidate = ProfileParameters(
        kop_vertical_m=1650.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=58.0,
        dls_build1_deg_per_30m=5.1,
        dls_build2_deg_per_30m=5.1,
        build1_length_m=360.0,
        hold_length_m=240.0,
        build2_length_m=330.0,
        horizontal_length_m=1160.0,
        horizontal_adjust_length_m=90.0,
        horizontal_hold_length_m=1070.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=44.0,
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
        candidate_md_start_m=4000.0,
        candidate_md_end_m=4350.0,
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        references=(),
        prefer_keep_kop=True,
        prefer_keep_build1=True,
        prefer_adjust_build2=True,
        baseline_kop_vertical_m=700.0,
        baseline_build1_dls_deg_per_30m=4.2,
    )

    monkeypatch.setattr(
        planner_module,
        "evaluate_candidate_anti_collision_clearance",
        lambda **kwargs: AntiCollisionClearanceEvaluation(
            min_separation_factor=0.42,
            max_overlap_depth_m=32.0,
        ),
    )
    monkeypatch.setattr(
        planner_module,
        "_evaluate_candidate_for_optimization",
        lambda **kwargs: CandidateOptimizationEvaluation(
            candidate=None,
            t1_miss_m=1e9,
            md_total_m=1e9,
            kop_vertical_m=1e9,
            t1_margin_m=-1e9,
            build1_margin_m=-1e9,
            build2_margin_m=-1e9,
            max_inc_margin_deg=-1e9,
            horizontal_dls_margin_deg_per_30m=-1e9,
        ),
    )
    monkeypatch.setattr(
        planner_module,
        "minimize",
        lambda **kwargs: SimpleNamespace(
            success=False, x=np.asarray(kwargs["x0"], dtype=float)
        ),
    )

    with pytest.raises(
        PlanningError,
        match="Late anti-collision BUILD2/HOLD adjustment could not find a valid trajectory",
    ):
        planner_module._select_anti_collision_candidate(
            candidates=[wild_candidate],
            surface=Point3D(0.0, 0.0, 0.0),
            config=config,
            optimization_context=context,
            zero_azimuth_turn=False,
            lower_dls_deg_per_30m=0.1,
            upper_dls_deg_per_30m=6.0,
            bounds=bounds,
            optimization_bounds=bounds,
            profile_builder=lambda values: wild_candidate,
            optimization_profile_builder=lambda values: wild_candidate,
            target_point=np.zeros(3, dtype=float),
            search_settings=planner_module._turn_search_settings(0),
            split_build=True,
        )


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
        offer_j_profile=False,
    )
    config_md = _fast_config(
        kop_min_vertical_m=200.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        max_total_md_postcheck_m=25000.0,
        offer_j_profile=False,
    )

    result_none = planner.plan(surface=surface, t1=t1, t3=t3, config=config_none)
    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=config_md)

    assert float(result_md.summary["distance_t1_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["distance_t3_m"]) <= config_md.pos_tolerance_m
    assert float(result_md.summary["md_total_m"]) + 100.0 < float(
        result_none.summary["md_total_m"]
    )
    assert str(result_md.summary["optimization_status"]) in {
        "within_md_theoretical_gap_after_2d",
        "at_md_boundary_extremum",
    }
    assert int(result_md.summary["optimization_runs_used"]) >= 2


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


def test_recover_profile_from_build_and_kop_is_deterministic_for_reference_order() -> (
    None
):
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
    zero_azimuth_turn = geometry.is_zero_azimuth_turn(
        target_direction=planner_module.classify_trajectory_type(
            gv_m=float(geometry.z1_m),
            horizontal_offset_t1_m=planner_module._horizontal_offset(
                surface=surface, point=t1
            ),
        ),
        tolerance_m=config.pos_tolerance_m,
    )
    assert zero_azimuth_turn is False

    md_params = planner_module._solve_turn_profile(
        geometry=geometry,
        surface=surface,
        config=config,
        optimization_context=None,
        zero_azimuth_turn=zero_azimuth_turn,
        search_settings=planner_module._turn_search_settings(0),
    ).params
    kop_params = planner_module._solve_turn_profile(
        geometry=geometry,
        surface=surface,
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
        float(md_params.kop_vertical_m) + float(kop_params.kop_vertical_m)
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
    assert profile_a.azimuth_hold_deg == pytest.approx(
        profile_b.azimuth_hold_deg, abs=1e-6
    )


def test_solver_respects_split_tolerances_for_reverse_entry_geometry() -> None:
    config = _fast_config(
        lateral_tolerance_m=2.0,
        vertical_tolerance_m=2.0,
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

    assert float(result.summary["lateral_distance_t1_m"]) <= config.lateral_tolerance_m
    assert (
        float(result.summary["vertical_distance_t1_m"]) <= config.vertical_tolerance_m
    )
    assert float(result.summary["lateral_distance_t3_m"]) <= config.lateral_tolerance_m
    assert (
        float(result.summary["vertical_distance_t3_m"]) <= config.vertical_tolerance_m
    )
    assert (
        float(result.summary["build1_dls_selected_deg_per_30m"])
        <= float(config.dls_build_max_deg_per_30m) + 1e-6
    )
    assert (
        float(result.summary["build2_dls_selected_deg_per_30m"])
        <= float(config.dls_build_max_deg_per_30m) + 1e-6
    )
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

    assert (
        float(result_kop.summary["lateral_distance_t1_m"])
        <= config_kop.lateral_tolerance_m
    )
    assert (
        float(result_kop.summary["vertical_distance_t1_m"])
        <= config_kop.vertical_tolerance_m
    )
    assert (
        float(result_kop.summary["lateral_distance_t3_m"])
        <= config_kop.lateral_tolerance_m
    )
    assert (
        float(result_kop.summary["vertical_distance_t3_m"])
        <= config_kop.vertical_tolerance_m
    )
    assert float(result_kop.summary["kop_md_m"]) + 1.0 < float(
        result_none.summary["kop_md_m"]
    )
    assert str(result_kop.summary["optimization_mode"]) == OPTIMIZATION_MINIMIZE_KOP
    assert str(result_kop.summary["optimization_status"]) in {
        "refined",
        "at_min_kop_limit",
    }
    assert int(result_kop.summary["optimization_runs_used"]) > 0


def test_fixed_kop_mode_uses_configured_kop_instead_of_only_minimum_bound() -> None:
    planner = TrajectoryPlanner()
    config_free = _fast_config(
        dls_build_min_deg_per_30m=0.0,
        dls_build_max_deg_per_30m=6.0,
        kop_min_vertical_m=200.0,
        use_fixed_kop=False,
        optimization_mode="none",
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
    )
    config_fixed = _fast_config(
        dls_build_min_deg_per_30m=0.0,
        dls_build_max_deg_per_30m=6.0,
        kop_min_vertical_m=200.0,
        use_fixed_kop=True,
        optimization_mode="none",
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
    )
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(2500.0, 800.0, 2400.0)
    t3 = Point3D(1500.0, 2000.0, 2500.0)

    result_free = planner.plan(surface=surface, t1=t1, t3=t3, config=config_free)
    result_fixed = planner.plan(surface=surface, t1=t1, t3=t3, config=config_fixed)

    assert float(result_free.summary["kop_md_m"]) > 200.0 + 1.0
    assert float(result_fixed.summary["kop_md_m"]) == pytest.approx(200.0, abs=1e-6)
    assert float(result_fixed.summary["distance_t1_m"]) <= config_fixed.pos_tolerance_m
    assert float(result_fixed.summary["distance_t3_m"]) <= config_fixed.pos_tolerance_m


def test_turn_least_squares_probes_drop_fixed_kop_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywp.planner as planner_module

    recorded_sizes: list[int] = []

    def fake_least_squares(
        *,
        fun: object,
        x0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        method: str,
        jac: str,
        x_scale: str,
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
    ) -> SimpleNamespace:
        recorded_sizes.append(int(np.asarray(x0, dtype=float).size))
        return SimpleNamespace(success=False, x=np.asarray(x0, dtype=float))

    monkeypatch.setattr(planner_module, "least_squares", fake_least_squares)

    probes = planner_module._turn_least_squares_probes(
        seed_vector=np.array([1.0, 550.0, 20.0, 120.0, 45.0], dtype=float),
        fixed_components={1: 550.0},
        bounds=((0.0, 1.0), (549.0, 551.0), (0.5, 85.0), (0.0, 500.0), (0.0, 360.0)),
        profile_builder=lambda _values: None,
        target_point=np.zeros(3, dtype=float),
        config=_fast_config(),
        max_nfev=20,
    )

    assert recorded_sizes == [4]
    assert len(probes) == 1
    assert float(probes[0][1]) == pytest.approx(550.0)


def test_slsqp_search_probes_drop_fixed_kop_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywp.planner as planner_module

    recorded_sizes: list[int] = []

    def fake_minimize(
        *,
        fun: object,
        x0: np.ndarray,
        method: str,
        bounds: list[tuple[float, float]],
        constraints: list[dict[str, object]],
        options: dict[str, object],
    ) -> SimpleNamespace:
        recorded_sizes.append(int(np.asarray(x0, dtype=float).size))
        return SimpleNamespace(success=False, x=np.asarray(x0, dtype=float))

    monkeypatch.setattr(planner_module, "minimize", fake_minimize)

    probes = planner_module._slsqp_search_probes(
        seed_vector=np.array([1.0, 550.0], dtype=float),
        fixed_components={1: 550.0},
        bounds=((0.0, 1.0), (549.0, 551.0)),
        objective=lambda _values: 0.0,
        constraint_functions=(lambda _values: 1.0,),
        maxiter=20,
    )

    assert recorded_sizes == [1]
    assert len(probes) == 1
    assert float(probes[0][1]) == pytest.approx(550.0)


def test_recover_turn_hold_inc_bounds_keep_boundary_value() -> None:
    import pywp.planner as planner_module

    bounds = planner_module._recover_turn_hold_inc_bounds(
        min_hold_inc_deg=85.5,
        max_hold_inc_deg=85.5,
    )

    assert bounds is not None
    lower, upper = bounds
    assert lower == pytest.approx(85.5)
    assert upper == pytest.approx(85.5)


def test_min_hold_angle_constraint_raises_hold_inc_without_breaking_solution() -> None:
    planner = TrajectoryPlanner()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(400.0, 0.0, 2200.0)
    t3 = Point3D(2400.0, 0.0, 2300.0)
    base_config = _fast_config(
        optimization_mode="none",
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
        offer_j_profile=False,
    )
    constrained_config = base_config.validated_copy(min_hold_inc_deg=13.0)

    base_result = planner.plan(surface=surface, t1=t1, t3=t3, config=base_config)
    constrained_result = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=constrained_config,
    )

    assert float(base_result.summary["hold_inc_deg"]) < 13.0
    assert float(constrained_result.summary["hold_inc_deg"]) >= 13.0 - 1e-6
    assert (
        float(constrained_result.summary["distance_t1_m"])
        <= constrained_config.pos_tolerance_m
    )
    assert (
        float(constrained_result.summary["distance_t3_m"])
        <= constrained_config.pos_tolerance_m
    )
    assert float(constrained_result.summary["md_total_m"]) >= float(
        base_result.summary["md_total_m"]
    ) - 1e-6


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
        offer_j_profile=False,
    )
    config_kop = _fast_config(
        kop_min_vertical_m=550.0,
        dls_build_max_deg_per_30m=3.0,
        optimization_mode=OPTIMIZATION_MINIMIZE_KOP,
        turn_solver_max_restarts=0,
        max_total_md_postcheck_m=20000.0,
        offer_j_profile=False,
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
    assert (
        float(result_kop.summary["md_total_m"])
        <= float(result_md.summary["md_total_m"]) + 1e-6
    )
    assert str(result_kop.summary["optimization_status"]) == "at_min_kop_limit"
    assert int(result_kop.summary["optimization_runs_used"]) <= 4


def test_md_optimization_skips_full_slsqp_when_2d_stage_confirms_boundary_extremum() -> (
    None
):
    planner = TrajectoryPlanner()
    records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS3.INC").read_text(encoding="utf-8")
    )
    target_record = next(record for record in records if str(record.name) == "well_02")
    surface, t1, t3 = welltrack_points_to_targets(target_record.points)
    config_md = TrajectoryConfig(
        optimization_mode=OPTIMIZATION_MINIMIZE_MD,
        turn_solver_max_restarts=2,
        offer_j_profile=False,
    )
    config_kop = TrajectoryConfig(
        optimization_mode=OPTIMIZATION_MINIMIZE_KOP,
        turn_solver_max_restarts=2,
        offer_j_profile=False,
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


def test_md_boundary_extremum_without_improvement_keeps_seed_candidate(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module
    from pywp.planner_types import PostEntrySection, SectionGeometry

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
    worse_boundary_candidate = ProfileParameters(
        kop_vertical_m=820.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=360.0,
        hold_length_m=520.0,
        build2_length_m=240.0,
        horizontal_length_m=1260.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1150.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    geometry = SectionGeometry(
        s1_m=1000.0,
        z1_m=2400.0,
        ds_13_m=1500.0,
        dz_13_m=100.0,
        azimuth_entry_deg=67.0,
        azimuth_surface_t1_deg=42.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        t1_cross_m=0.0,
        t3_cross_m=0.0,
        t1_east_m=600.0,
        t1_north_m=800.0,
        t1_tvd_m=2400.0,
    )
    post_entry = PostEntrySection(
        total_length_m=1200.0,
        transition_length_m=110.0,
        hold_length_m=1090.0,
        hold_inc_deg=85.0,
        transition_dls_deg_per_30m=2.0,
    )
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )

    monkeypatch.setattr(
        planner_module,
        "_theoretical_objective_lower_bound",
        lambda **kwargs: 0.0,
    )
    monkeypatch.setattr(
        planner_module,
        "_optimization_target_reached",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        planner_module,
        "_make_turn_profile_builder",
        lambda **kwargs: (lambda values: seed_candidate),
    )
    monkeypatch.setattr(
        planner_module,
        "_collect_optimization_seed_vectors",
        lambda **kwargs: [np.array([0.5, 550.0, 50.0, 400.0, 30.0], dtype=float)],
    )
    monkeypatch.setattr(
        planner_module,
        "_boundary_refine_md_candidates",
        lambda **kwargs: ([worse_boundary_candidate], 1),
    )
    monkeypatch.setattr(
        planner_module,
        "_is_md_boundary_extremum_candidate",
        lambda **kwargs: True,
    )

    result = planner_module._select_feasible_candidate(
        candidates=[seed_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        geometry=geometry,
        post_entry=post_entry,
        config=_fast_config(optimization_mode=OPTIMIZATION_MINIMIZE_MD),
        optimization_context=None,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: seed_candidate,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
    )

    assert result.params == seed_candidate
    assert result.optimization.status == "seed_selected"
    assert result.optimization.objective_value == pytest.approx(
        float(seed_candidate.md_total_m)
    )


def test_md_2d_boundary_extremum_without_improvement_keeps_seed_candidate(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module
    from pywp.planner_types import PostEntrySection, SectionGeometry

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
    worse_boundary_candidate = ProfileParameters(
        kop_vertical_m=820.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=360.0,
        hold_length_m=520.0,
        build2_length_m=240.0,
        horizontal_length_m=1260.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1150.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    worse_2d_candidate = ProfileParameters(
        kop_vertical_m=810.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=350.0,
        hold_length_m=510.0,
        build2_length_m=235.0,
        horizontal_length_m=1250.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1140.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    geometry = SectionGeometry(
        s1_m=1000.0,
        z1_m=2400.0,
        ds_13_m=1500.0,
        dz_13_m=100.0,
        azimuth_entry_deg=67.0,
        azimuth_surface_t1_deg=42.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        t1_cross_m=0.0,
        t3_cross_m=0.0,
        t1_east_m=600.0,
        t1_north_m=800.0,
        t1_tvd_m=2400.0,
    )
    post_entry = PostEntrySection(
        total_length_m=1200.0,
        transition_length_m=110.0,
        hold_length_m=1090.0,
        hold_inc_deg=85.0,
        transition_dls_deg_per_30m=2.0,
    )
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    extremum_calls = {"count": 0}

    monkeypatch.setattr(
        planner_module,
        "_theoretical_objective_lower_bound",
        lambda **kwargs: 0.0,
    )
    monkeypatch.setattr(
        planner_module,
        "_optimization_target_reached",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        planner_module,
        "_make_turn_profile_builder",
        lambda **kwargs: (lambda values: seed_candidate),
    )
    monkeypatch.setattr(
        planner_module,
        "_collect_optimization_seed_vectors",
        lambda **kwargs: [np.array([0.5, 550.0, 50.0, 400.0, 30.0], dtype=float)],
    )
    monkeypatch.setattr(
        planner_module,
        "_boundary_refine_md_candidates",
        lambda **kwargs: ([worse_boundary_candidate], 1),
    )
    monkeypatch.setattr(
        planner_module,
        "_two_dimensional_md_refine_candidates",
        lambda **kwargs: ([worse_2d_candidate], 1),
    )

    def fake_is_boundary_extremum_candidate(**kwargs) -> bool:
        extremum_calls["count"] += 1
        return extremum_calls["count"] == 2

    monkeypatch.setattr(
        planner_module,
        "_is_md_boundary_extremum_candidate",
        fake_is_boundary_extremum_candidate,
    )

    result = planner_module._select_feasible_candidate(
        candidates=[seed_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        geometry=geometry,
        post_entry=post_entry,
        config=_fast_config(optimization_mode=OPTIMIZATION_MINIMIZE_MD),
        optimization_context=None,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: seed_candidate,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
    )

    assert result.params == seed_candidate
    assert result.optimization.status == "seed_selected"
    assert result.optimization.objective_value == pytest.approx(
        float(seed_candidate.md_total_m)
    )


def test_md_2d_refinement_runs_even_when_boundary_already_improved(
    monkeypatch,
) -> None:
    import pywp.planner as planner_module
    from pywp.planner_types import PostEntrySection, SectionGeometry

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
    improved_boundary_candidate = ProfileParameters(
        kop_vertical_m=730.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=300.0,
        hold_length_m=450.0,
        build2_length_m=190.0,
        horizontal_length_m=1180.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1070.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    best_2d_candidate = ProfileParameters(
        kop_vertical_m=700.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        inc_hold_deg=52.0,
        dls_build1_deg_per_30m=3.2,
        dls_build2_deg_per_30m=3.2,
        build1_length_m=280.0,
        hold_length_m=420.0,
        build2_length_m=170.0,
        horizontal_length_m=1160.0,
        horizontal_adjust_length_m=110.0,
        horizontal_hold_length_m=1050.0,
        horizontal_inc_deg=85.0,
        horizontal_dls_deg_per_30m=2.0,
        azimuth_hold_deg=42.0,
        azimuth_entry_deg=67.0,
    )
    geometry = SectionGeometry(
        s1_m=1000.0,
        z1_m=2400.0,
        ds_13_m=1500.0,
        dz_13_m=100.0,
        azimuth_entry_deg=67.0,
        azimuth_surface_t1_deg=42.0,
        inc_entry_deg=86.0,
        inc_required_t1_t3_deg=84.0,
        t1_cross_m=0.0,
        t3_cross_m=0.0,
        t1_east_m=600.0,
        t1_north_m=800.0,
        t1_tvd_m=2400.0,
    )
    post_entry = PostEntrySection(
        total_length_m=1200.0,
        transition_length_m=110.0,
        hold_length_m=1090.0,
        hold_inc_deg=85.0,
        transition_dls_deg_per_30m=2.0,
    )
    bounds = (
        (0.0, 1.0),
        (200.0, 1800.0),
        (0.5, 85.5),
        (0.0, 2500.0),
        (0.0, 360.0),
    )
    refinement_calls = {"count": 0}
    target_objective = 0.5 * (
        float(improved_boundary_candidate.md_total_m)
        + float(best_2d_candidate.md_total_m)
    )

    monkeypatch.setattr(
        planner_module,
        "_theoretical_objective_lower_bound",
        lambda **kwargs: 0.0,
    )
    monkeypatch.setattr(
        planner_module,
        "_optimization_target_reached",
        lambda objective_value, **kwargs: float(objective_value) <= target_objective,
    )
    monkeypatch.setattr(
        planner_module,
        "_make_turn_profile_builder",
        lambda **kwargs: (lambda values: seed_candidate),
    )
    monkeypatch.setattr(
        planner_module,
        "_collect_optimization_seed_vectors",
        lambda **kwargs: [np.array([0.5, 550.0, 50.0, 400.0, 30.0, 30.0], dtype=float)],
    )
    monkeypatch.setattr(
        planner_module,
        "_boundary_refine_md_candidates",
        lambda **kwargs: ([improved_boundary_candidate], 1),
    )

    def _fake_two_dimensional_md_refine_candidates(**kwargs):
        refinement_calls["count"] += 1
        return [best_2d_candidate], 1

    monkeypatch.setattr(
        planner_module,
        "_two_dimensional_md_refine_candidates",
        _fake_two_dimensional_md_refine_candidates,
    )
    monkeypatch.setattr(
        planner_module,
        "_is_md_boundary_extremum_candidate",
        lambda **kwargs: False,
    )

    result = planner_module._select_feasible_candidate(
        candidates=[seed_candidate],
        surface=Point3D(0.0, 0.0, 0.0),
        geometry=geometry,
        post_entry=post_entry,
        config=_fast_config(optimization_mode=OPTIMIZATION_MINIMIZE_MD),
        optimization_context=None,
        zero_azimuth_turn=False,
        lower_dls_deg_per_30m=0.1,
        upper_dls_deg_per_30m=6.0,
        bounds=bounds,
        profile_builder=lambda values: seed_candidate,
        target_point=np.zeros(3, dtype=float),
        search_settings=planner_module._turn_search_settings(0),
    )

    assert refinement_calls["count"] == 1
    assert result.params == best_2d_candidate
    assert result.optimization.status == "within_md_theoretical_gap_after_2d"
    assert result.optimization.objective_value == pytest.approx(
        float(best_2d_candidate.md_total_m)
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
    config = _fast_config(
        turn_solver_mode=TURN_SOLVER_LEAST_SQUARES, pos_tolerance_m=2.0
    )
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
    assert int(result.summary["solver_turn_max_restarts"]) == int(
        config.turn_solver_max_restarts
    )
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
    assert float(result.summary["horizontal_inc_deg"]) < float(
        result.summary["entry_inc_deg"]
    )
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
    assert (
        str(result.summary["trajectory_target_direction"])
        == "Цели в обратном направлении"
    )
    assert np.isfinite(float(result.summary["azimuth_turn_deg"]))


def test_turn_solver_retries_with_deeper_search_when_first_attempt_fails(
    monkeypatch,
) -> None:
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
        t1=Point3D(500.0, 500.0, 2200.0),
        t3=Point3D(1200.0, 1500.0, 2300.0),
        config=coarse,
    )
    result_fine = planner.plan(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(500.0, 500.0, 2200.0),
        t3=Point3D(1200.0, 1500.0, 2300.0),
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
    config = _fast_config(lateral_tolerance_m=1e-10, vertical_tolerance_m=1e-10)

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
