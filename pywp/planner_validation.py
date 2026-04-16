from __future__ import annotations

import numpy as np
import pandas as pd

from pywp.classification import (
    WellClassification,
    classify_well,
    complexity_label,
    trajectory_type_label,
)
from pywp.mcm import add_dls, compute_positions_min_curv, minimum_curvature_increment
from pywp.models import OPTIMIZATION_NONE, Point3D, TrajectoryConfig
from pywp.planner_geometry import (
    _distance_3d,
    _normalize_azimuth_deg,
    _shortest_azimuth_delta_deg,
)
from pywp.planner_types import (
    EndpointState,
    OptimizationOutcome,
    PlanningError,
    ProfileEndpointEvaluation,
    ProfileParameters,
    SectionGeometry,
    TurnSearchSettings,
)
from pywp.segments import BuildSegment, HoldSegment, HorizontalSegment, VerticalSegment
from pywp.constants import SMALL
from pywp.trajectory import WellTrajectory
DLS_VALIDATION_TOLERANCE_DEG_PER_30M = 0.01
TRAJECTORY_MODEL_LABEL = "Unified J Profile + Build + Azimuth Turn"


def _estimate_t1_endpoint_for_profile(profile: ProfileParameters) -> tuple[float, float, float]:
    t1_state = _evaluate_profile_endpoints(params=profile).t1
    return float(t1_state.east_m), float(t1_state.north_m), float(t1_state.tvd_m)


def _advance_endpoint_state(
    state: EndpointState,
    *,
    length_m: float,
    inc_to_deg: float,
    azi_to_deg: float,
) -> EndpointState:
    length = float(length_m)
    if length <= SMALL:
        return EndpointState(
            md_m=float(state.md_m),
            east_m=float(state.east_m),
            north_m=float(state.north_m),
            tvd_m=float(state.tvd_m),
            inc_deg=float(inc_to_deg),
            azi_deg=_normalize_azimuth_deg(float(azi_to_deg)),
        )

    dn, de, dz = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=float(state.inc_deg),
        azi1_deg=float(state.azi_deg),
        md2_m=length,
        inc2_deg=float(inc_to_deg),
        azi2_deg=float(azi_to_deg),
    )
    return EndpointState(
        md_m=float(state.md_m + length),
        east_m=float(state.east_m + de),
        north_m=float(state.north_m + dn),
        tvd_m=float(state.tvd_m + dz),
        inc_deg=float(inc_to_deg),
        azi_deg=_normalize_azimuth_deg(float(azi_to_deg)),
    )


def _offset_endpoint_state(
    *,
    state: EndpointState,
    surface: Point3D,
) -> EndpointState:
    return EndpointState(
        md_m=float(state.md_m),
        east_m=float(state.east_m + surface.x),
        north_m=float(state.north_m + surface.y),
        tvd_m=float(state.tvd_m + surface.z),
        inc_deg=float(state.inc_deg),
        azi_deg=float(state.azi_deg),
    )


def _offset_endpoint_evaluation(
    *,
    evaluation: ProfileEndpointEvaluation,
    surface: Point3D,
) -> ProfileEndpointEvaluation:
    return ProfileEndpointEvaluation(
        t1=_offset_endpoint_state(state=evaluation.t1, surface=surface),
        t3=_offset_endpoint_state(state=evaluation.t3, surface=surface),
    )


def _evaluate_profile_endpoints(
    *,
    params: ProfileParameters,
) -> ProfileEndpointEvaluation:
    state = EndpointState(
        md_m=0.0,
        east_m=0.0,
        north_m=0.0,
        tvd_m=0.0,
        inc_deg=0.0,
        azi_deg=_normalize_azimuth_deg(float(params.azimuth_hold_deg)),
    )
    state = _advance_endpoint_state(
        state,
        length_m=float(params.kop_vertical_m),
        inc_to_deg=0.0,
        azi_to_deg=float(params.azimuth_hold_deg),
    )
    state = _advance_endpoint_state(
        state,
        length_m=float(params.build1_length_m),
        inc_to_deg=float(params.inc_hold_deg),
        azi_to_deg=float(params.azimuth_hold_deg),
    )
    state = _advance_endpoint_state(
        state,
        length_m=float(params.hold_length_m),
        inc_to_deg=float(params.inc_hold_deg),
        azi_to_deg=float(params.azimuth_hold_deg),
    )
    t1_state = _advance_endpoint_state(
        state,
        length_m=float(params.build2_length_m),
        inc_to_deg=float(params.inc_entry_deg),
        azi_to_deg=float(params.azimuth_entry_deg),
    )
    t3_state = t1_state
    if (
        float(params.horizontal_adjust_length_m) > SMALL
        and abs(float(params.horizontal_inc_deg) - float(params.inc_entry_deg)) > 1e-6
    ):
        t3_state = _advance_endpoint_state(
            t3_state,
            length_m=float(params.horizontal_adjust_length_m),
            inc_to_deg=float(params.horizontal_inc_deg),
            azi_to_deg=float(params.azimuth_entry_deg),
        )
    t3_state = _advance_endpoint_state(
        t3_state,
        length_m=float(params.horizontal_hold_length_m),
        inc_to_deg=float(params.horizontal_inc_deg),
        azi_to_deg=float(params.azimuth_entry_deg),
    )
    return ProfileEndpointEvaluation(t1=t1_state, t3=t3_state)


def _target_delta_components(
    *,
    state: EndpointState,
    target: Point3D,
) -> tuple[float, float, float, float, float, float]:
    dx_m = float(state.east_m - target.x)
    dy_m = float(state.north_m - target.y)
    dz_m = float(state.tvd_m - target.z)
    lateral_distance_m = float(np.hypot(dx_m, dy_m))
    vertical_distance_m = float(abs(dz_m))
    distance_m = float(np.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m))
    return dx_m, dy_m, dz_m, lateral_distance_m, vertical_distance_m, distance_m


def _build_trajectory(
    params: ProfileParameters,
    interpolation_method: str = "rodrigues",
) -> WellTrajectory:
    segments = [
        VerticalSegment(
            length_m=params.kop_vertical_m,
            azi_deg=params.azimuth_hold_deg,
            name="VERTICAL",
        )
    ]
    if params.build1_length_m > SMALL:
        segments.append(
            BuildSegment(
                inc_from_deg=0.0,
                inc_to_deg=params.inc_hold_deg,
                dls_deg_per_30m=params.dls_build1_deg_per_30m,
                azi_deg=params.azimuth_hold_deg,
                name="BUILD1",
                interpolation_method=interpolation_method,
            )
        )
    if params.hold_length_m > SMALL:
        segments.append(
            HoldSegment(
                length_m=params.hold_length_m,
                inc_deg=params.inc_hold_deg,
                azi_deg=params.azimuth_hold_deg,
                name="HOLD",
            )
        )
    if params.build2_length_m > SMALL:
        segments.append(
            BuildSegment(
                inc_from_deg=params.inc_hold_deg,
                inc_to_deg=params.inc_entry_deg,
                dls_deg_per_30m=params.dls_build2_deg_per_30m,
                azi_deg=params.azimuth_hold_deg,
                azi_to_deg=params.azimuth_entry_deg,
                name="BUILD2",
                interpolation_method=interpolation_method,
            )
        )
    if (
        params.horizontal_adjust_length_m > SMALL
        and abs(params.horizontal_inc_deg - params.inc_entry_deg) > 1e-6
    ):
        segments.append(
            BuildSegment(
                inc_from_deg=params.inc_entry_deg,
                inc_to_deg=params.horizontal_inc_deg,
                dls_deg_per_30m=params.horizontal_dls_deg_per_30m,
                azi_deg=params.azimuth_entry_deg,
                name="HORIZONTAL",
                interpolation_method=interpolation_method,
            )
        )
    if params.horizontal_hold_length_m > SMALL:
        segments.append(
            HorizontalSegment(
                length_m=params.horizontal_hold_length_m,
                inc_deg=params.horizontal_inc_deg,
                azi_deg=params.azimuth_entry_deg,
                name="HORIZONTAL",
            )
        )
    return WellTrajectory(segments)


def _candidate_turn_deg(candidate: ProfileParameters) -> float:
    return float(
        abs(
            _shortest_azimuth_delta_deg(
                candidate.azimuth_hold_deg,
                candidate.azimuth_entry_deg,
            )
        )
    )


def _build_summary(
    df: pd.DataFrame,
    t1: Point3D,
    t3: Point3D,
    md_t1_m: float,
    params: ProfileParameters,
    horizontal_offset_t1_m: float,
    classification: WellClassification,
    config: TrajectoryConfig,
    endpoint_eval: ProfileEndpointEvaluation,
    optimization_outcome: OptimizationOutcome,
    turn_search_settings: TurnSearchSettings | None = None,
    turn_restarts_used: int = 0,
) -> dict[str, float | str]:
    t1_idx = int((df["MD_m"] - md_t1_m).abs().idxmin())
    t1_row = df.loc[t1_idx]
    t3_row = df.iloc[-1]

    (
        t1_dx_m,
        t1_dy_m,
        t1_dz_m,
        t1_lateral_m,
        t1_vertical_m,
        distance_t1,
    ) = _target_delta_components(
        state=endpoint_eval.t1,
        target=t1,
    )
    (
        t3_dx_m,
        t3_dy_m,
        t3_dz_m,
        t3_lateral_m,
        t3_vertical_m,
        distance_t3,
    ) = _target_delta_components(
        state=endpoint_eval.t3,
        target=t3,
    )
    distance_t1_control = _distance_3d(
        t1_row["X_m"],
        t1_row["Y_m"],
        t1_row["Z_m"],
        t1.x,
        t1.y,
        t1.z,
    )
    distance_t3_control = _distance_3d(
        t3_row["X_m"],
        t3_row["Y_m"],
        t3_row["Z_m"],
        t3.x,
        t3.y,
        t3.z,
    )
    control_gap_t1 = _distance_3d(
        t1_row["X_m"],
        t1_row["Y_m"],
        t1_row["Z_m"],
        endpoint_eval.t1.east_m,
        endpoint_eval.t1.north_m,
        endpoint_eval.t1.tvd_m,
    )
    control_gap_t3 = _distance_3d(
        t3_row["X_m"],
        t3_row["Y_m"],
        t3_row["Z_m"],
        endpoint_eval.t3.east_m,
        endpoint_eval.t3.north_m,
        endpoint_eval.t3.tvd_m,
    )

    max_dls = float(np.nanmax(df["DLS_deg_per_30m"].to_numpy()))
    max_inc_actual = float(np.nanmax(df["INC_deg"].to_numpy()))
    md_total_m = float(df["MD_m"].iloc[-1])
    md_postcheck_limit_m = float(config.max_total_md_postcheck_m)
    md_postcheck_excess_m = float(max(0.0, md_total_m - md_postcheck_limit_m))
    md_postcheck_exceeded = bool(md_postcheck_excess_m > 1e-6)

    summary: dict[str, float | str] = {
        "distance_t1_m": float(distance_t1),
        "distance_t3_m": float(distance_t3),
        "lateral_distance_t1_m": float(t1_lateral_m),
        "lateral_distance_t3_m": float(t3_lateral_m),
        "vertical_distance_t1_m": float(t1_vertical_m),
        "vertical_distance_t3_m": float(t3_vertical_m),
        "distance_t1_control_m": float(distance_t1_control),
        "distance_t3_control_m": float(distance_t3_control),
        "control_gap_t1_m": float(control_gap_t1),
        "control_gap_t3_m": float(control_gap_t3),
        "kop_vertical_m": float(params.kop_vertical_m),
        "kop_md_m": float(params.kop_vertical_m),
        "entry_inc_deg": float(endpoint_eval.t1.inc_deg),
        "entry_inc_control_deg": float(t1_row["INC_deg"]),
        "entry_inc_target_deg": float(config.entry_inc_target_deg),
        "entry_inc_tolerance_deg": float(config.entry_inc_tolerance_deg),
        "lateral_tolerance_m": float(config.lateral_tolerance_m),
        "vertical_tolerance_m": float(config.vertical_tolerance_m),
        "max_inc_deg": float(config.max_inc_deg),
        "max_inc_actual_deg": max_inc_actual,
        "inc_required_t1_t3_deg": float(params.inc_required_t1_t3_deg),
        "horizontal_adjust_length_m": float(params.horizontal_adjust_length_m),
        "horizontal_hold_length_m": float(params.horizontal_hold_length_m),
        "horizontal_inc_deg": float(params.horizontal_inc_deg),
        "hold_inc_deg": float(params.inc_hold_deg),
        "hold_length_m": float(params.hold_length_m),
        "build_dls_selected_deg_per_30m": float(params.dls_build1_deg_per_30m),
        "build1_dls_selected_deg_per_30m": float(params.dls_build1_deg_per_30m),
        "build2_dls_selected_deg_per_30m": float(params.dls_build2_deg_per_30m),
        "build_dls_max_config_deg_per_30m": float(config.dls_build_max_deg_per_30m),
        "build_dls_relaxed_from_max": (
            "yes"
            if (
                float(params.dls_build1_deg_per_30m)
                < float(config.dls_build_max_deg_per_30m) - 1e-6
                or float(params.dls_build2_deg_per_30m)
                < float(config.dls_build_max_deg_per_30m) - 1e-6
            )
            else "no"
        ),
        "build_dls_split_selected": (
            "yes"
            if abs(
                float(params.dls_build1_deg_per_30m)
                - float(params.dls_build2_deg_per_30m)
            )
            > 1e-6
            else "no"
        ),
        "max_dls_total_deg_per_30m": max_dls,
        "md_total_m": md_total_m,
        "max_total_md_postcheck_m": md_postcheck_limit_m,
        "md_postcheck_excess_m": md_postcheck_excess_m,
        "md_postcheck_exceeded": "yes" if md_postcheck_exceeded else "no",
        "t1_horizontal_offset_m": float(horizontal_offset_t1_m),
        "horizontal_length_m": float(params.horizontal_length_m),
        "trajectory_type": TRAJECTORY_MODEL_LABEL,
        "trajectory_target_direction": trajectory_type_label(
            classification.trajectory_type
        ),
        "well_complexity": complexity_label(classification.complexity),
        "well_complexity_by_offset": complexity_label(classification.complexity_by_offset),
        "well_complexity_by_hold": complexity_label(classification.complexity_by_hold),
        "hold_azimuth_deg": float(params.azimuth_hold_deg),
        "entry_azimuth_deg": float(params.azimuth_entry_deg),
        "t1_exact_x_m": float(endpoint_eval.t1.east_m),
        "t1_exact_y_m": float(endpoint_eval.t1.north_m),
        "t1_exact_z_m": float(endpoint_eval.t1.tvd_m),
        "t3_exact_x_m": float(endpoint_eval.t3.east_m),
        "t3_exact_y_m": float(endpoint_eval.t3.north_m),
        "t3_exact_z_m": float(endpoint_eval.t3.tvd_m),
        "t1_miss_dx_m": float(t1_dx_m),
        "t1_miss_dy_m": float(t1_dy_m),
        "t1_miss_dz_m": float(t1_dz_m),
        "t3_miss_dx_m": float(t3_dx_m),
        "t3_miss_dy_m": float(t3_dy_m),
        "t3_miss_dz_m": float(t3_dz_m),
        "azimuth_turn_deg": float(
            abs(
                _shortest_azimuth_delta_deg(
                    params.azimuth_hold_deg,
                    params.azimuth_entry_deg,
                )
            )
        ),
        "solver_turn_mode": str(config.turn_solver_mode),
        "solver_turn_max_restarts": int(config.turn_solver_max_restarts),
        "solver_turn_restarts_used": int(turn_restarts_used),
        "solver_turn_attempts_used": (
            int(turn_restarts_used + 1) if turn_search_settings is not None else 0
        ),
        "solver_turn_search_depth_scale": (
            float(turn_search_settings.search_depth_scale)
            if turn_search_settings is not None
            else 1.0
        ),
        "solver_turn_seed_lattice_points": (
            int(turn_search_settings.seed_lattice_points)
            if turn_search_settings is not None
            else 0
        ),
        "solver_turn_local_max_nfev": (
            int(turn_search_settings.local_max_nfev)
            if turn_search_settings is not None
            else 0
        ),
        "optimization_mode": str(optimization_outcome.mode),
        "optimization_status": str(optimization_outcome.status),
        "optimization_objective_value_m": float(optimization_outcome.objective_value),
        "optimization_theoretical_lower_bound_m": float(
            optimization_outcome.theoretical_lower_bound
        ),
        "optimization_absolute_gap_m": float(optimization_outcome.absolute_gap_value),
        "optimization_relative_gap_pct": float(optimization_outcome.relative_gap_pct),
        "optimization_seeds_used": int(optimization_outcome.seeds_used),
        "optimization_runs_used": int(optimization_outcome.runs_used),
        "solver_strategy": (
            "unified_azimuth_turn_feasibility_first"
            if str(optimization_outcome.mode) == OPTIMIZATION_NONE
            else f"unified_azimuth_turn_{optimization_outcome.mode}"
        ),
        "class_reverse_offset_min_m": float(classification.limits.reverse_min_m),
        "class_reverse_offset_max_m": float(classification.limits.reverse_max_m),
        "class_offset_ordinary_max_m": float(classification.limits.ordinary_offset_max_m),
        "class_offset_complex_max_m": float(classification.limits.complex_offset_max_m),
        "class_hold_ordinary_max_deg": float(classification.limits.hold_ordinary_max_deg),
        "class_hold_complex_max_deg": float(classification.limits.hold_complex_max_deg),
    }

    for segment, limit in config.dls_limits_deg_per_30m.items():
        seg_max = float(df.loc[df["segment"] == segment, "DLS_deg_per_30m"].max(skipna=True))
        if np.isnan(seg_max):
            seg_max = 0.0
        summary[f"max_dls_{segment.lower()}_deg_per_30m"] = seg_max
        summary[f"dls_limit_{segment.lower()}_deg_per_30m"] = float(limit)
    return summary


def _assert_solution_is_valid(
    summary: dict[str, float | str],
    config: TrajectoryConfig,
    *,
    position_tolerance_slack_m: float = 0.0,
) -> None:
    position_slack_m = max(
        float(position_tolerance_slack_m),
        0.0,
    )
    allowed_lateral_tolerance_m = float(config.lateral_tolerance_m) + position_slack_m
    allowed_vertical_tolerance_m = float(config.vertical_tolerance_m) + position_slack_m
    if (
        float(summary["lateral_distance_t1_m"]) > allowed_lateral_tolerance_m
        or float(summary["vertical_distance_t1_m"]) > allowed_vertical_tolerance_m
    ):
        raise PlanningError(
            "Failed to hit t1 within tolerance. "
            "Miss="
            f"lateral {float(summary['lateral_distance_t1_m']):.2f} m / "
            f"vertical {float(summary['vertical_distance_t1_m']):.2f} m, "
            "tolerances="
            f"{config.lateral_tolerance_m:.2f} / {config.vertical_tolerance_m:.2f} m. "
            "Analytical delta: "
            f"dX={float(summary['t1_miss_dx_m']):.2f} m, "
            f"dY={float(summary['t1_miss_dy_m']):.2f} m, "
            f"dZ={float(summary['t1_miss_dz_m']):.2f} m. "
            "Increase BUILD DLS limit, relax tolerance, or adjust target geometry."
        )
    if (
        float(summary["lateral_distance_t3_m"]) > allowed_lateral_tolerance_m
        or float(summary["vertical_distance_t3_m"]) > allowed_vertical_tolerance_m
    ):
        raise PlanningError(
            "Failed to hit t3 within tolerance. "
            "Miss="
            f"lateral {float(summary['lateral_distance_t3_m']):.2f} m / "
            f"vertical {float(summary['vertical_distance_t3_m']):.2f} m, "
            "tolerances="
            f"{config.lateral_tolerance_m:.2f} / {config.vertical_tolerance_m:.2f} m. "
            "Analytical delta: "
            f"dX={float(summary['t3_miss_dx_m']):.2f} m, "
            f"dY={float(summary['t3_miss_dy_m']):.2f} m, "
            f"dZ={float(summary['t3_miss_dz_m']):.2f} m. "
            "Increase HORIZONTAL DLS limit and/or max INC, or move t3 closer/deeper relative to t1."
        )
    if abs(float(summary["entry_inc_deg"]) - config.entry_inc_target_deg) > config.entry_inc_tolerance_deg + 1e-6:
        raise PlanningError("Entry inclination at t1 is outside required target range.")
    if float(summary["max_inc_actual_deg"]) > config.max_inc_deg + 1e-6:
        raise PlanningError(
            "Trajectory exceeds configured max INC. "
            f"max actual INC={float(summary['max_inc_actual_deg']):.2f} deg, "
            f"limit={config.max_inc_deg:.2f} deg."
        )
    for segment, limit in config.dls_limits_deg_per_30m.items():
        actual = float(summary.get(f"max_dls_{segment.lower()}_deg_per_30m", 0.0))
        if actual > limit + DLS_VALIDATION_TOLERANCE_DEG_PER_30M:
            raise PlanningError(f"DLS limit exceeded on segment {segment}: {actual:.2f} > {limit:.2f}")


def _is_candidate_feasible(candidate: ProfileParameters | None, config: TrajectoryConfig) -> bool:
    if candidate is None:
        return False
    horizontal_limit = config.dls_limits_deg_per_30m.get("HORIZONTAL")
    if horizontal_limit is not None and candidate.horizontal_dls_deg_per_30m > float(horizontal_limit) + SMALL:
        return False
    if candidate.horizontal_inc_deg > config.max_inc_deg + SMALL:
        return False
    return True


def _build_validated_control_and_summary(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    geometry: SectionGeometry,
    horizontal_offset_t1_m: float,
    params: ProfileParameters,
    optimization_outcome: OptimizationOutcome,
    config: TrajectoryConfig,
    turn_search_settings: TurnSearchSettings | None,
    turn_restarts_used: int,
) -> tuple[WellTrajectory, pd.DataFrame, dict[str, float | str]]:
    trajectory = _build_trajectory(params=params, interpolation_method=str(getattr(config, "interpolation_method", "rodrigues")))
    endpoint_eval = _offset_endpoint_evaluation(
        evaluation=_evaluate_profile_endpoints(params=params),
        surface=surface,
    )
    try:
        control = compute_positions_min_curv(
            trajectory.stations(md_step_m=config.md_step_control_m),
            start=surface,
        )
        control = add_dls(control)
    except ValueError as exc:
        raise PlanningError(
            "Не удалось сформировать контрольную инклинометрию методом минимальной кривизны. "
            f"Причина: {exc}"
        ) from exc
    classification = classify_well(
        gv_m=geometry.z1_m,
        horizontal_offset_t1_m=horizontal_offset_t1_m,
        hold_inc_deg=params.inc_hold_deg,
    )
    summary = _build_summary(
        df=control,
        t1=t1,
        t3=t3,
        md_t1_m=params.md_t1_m,
        params=params,
        horizontal_offset_t1_m=horizontal_offset_t1_m,
        classification=classification,
        config=config,
        endpoint_eval=endpoint_eval,
        optimization_outcome=optimization_outcome,
        turn_search_settings=turn_search_settings,
        turn_restarts_used=turn_restarts_used,
    )
    position_tolerance_slack_m = 0.0
    if str(optimization_outcome.mode) == "anti_collision_avoidance":
        position_tolerance_slack_m = 0.01
    _assert_solution_is_valid(
        summary=summary,
        config=config,
        position_tolerance_slack_m=position_tolerance_slack_m,
    )
    return trajectory, control, summary
