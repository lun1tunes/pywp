from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.constants import SMALL
from pywp.mcm import add_dls
from pywp.models import PlannerResult, Point3D, SummaryDict, TrajectoryConfig
from pywp.planner_types import PlanningError
from pywp.reference_trajectories import build_reference_trajectory_stations
from pywp.ui_utils import dls_to_pi


@dataclass(frozen=True)
class SidetrackStart:
    point: Point3D
    inc_deg: float
    azi_deg: float


class SidetrackPlanner:
    def plan(
        self,
        *,
        start: SidetrackStart,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
    ) -> PlannerResult:
        config.validate_for_planning()
        horizontal_inc_deg, horizontal_azi_deg = _angles_from_points(t1, t3)
        best: tuple[float, pd.DataFrame] | None = None
        best_dls_excess: tuple[float, float] | None = None
        last_problem = ""

        for lead_m, tail_m in _control_length_pairs(
            start=start,
            t1=t1,
            t3=t3,
            config=config,
        ):
            try:
                stations = _build_sidetrack_stations(
                    start=start,
                    t1=t1,
                    t3=t3,
                    horizontal_inc_deg=horizontal_inc_deg,
                    horizontal_azi_deg=horizontal_azi_deg,
                    lead_m=lead_m,
                    tail_m=tail_m,
                    config=config,
                )
            except (ValueError, PlanningError) as exc:
                last_problem = str(exc)
                continue
            max_dls = _finite_max(stations["DLS_deg_per_30m"])
            dls_limit = float(config.dls_build_max_deg_per_30m)
            dls_excess = max(0.0, max_dls - dls_limit)
            if dls_excess > 1e-6:
                last_problem = (
                    "ПИ бокового ствола превышает лимит расчетной модели: "
                    f"{dls_to_pi(max_dls):.2f} > {dls_to_pi(dls_limit):.2f} deg/10m."
                )
                if best_dls_excess is None or dls_excess < best_dls_excess[0]:
                    best_dls_excess = (dls_excess, max_dls)
                continue
            max_inc = _finite_max(stations["INC_deg"])
            if max_inc > float(config.max_inc_deg) + 1e-6:
                last_problem = (
                    "Боковой ствол превышает configured max INC: "
                    f"{max_inc:.2f} > {float(config.max_inc_deg):.2f} deg."
                )
                continue
            score = _candidate_score(stations=stations, config=config)
            if best is None or score < best[0]:
                best = (score, stations)

        fallback_used = False
        if best is None:
            for stations in _bend_fallback_station_candidates(
                start=start,
                t1=t1,
                t3=t3,
                horizontal_inc_deg=horizontal_inc_deg,
                horizontal_azi_deg=horizontal_azi_deg,
                config=config,
            ):
                max_dls = _finite_max(stations["DLS_deg_per_30m"])
                dls_limit = float(config.dls_build_max_deg_per_30m)
                dls_excess = max(0.0, max_dls - dls_limit)
                if dls_excess > 1e-6:
                    last_problem = (
                        "ПИ бокового ствола превышает лимит расчетной модели: "
                        f"{dls_to_pi(max_dls):.2f} > "
                        f"{dls_to_pi(dls_limit):.2f} deg/10m."
                    )
                    if best_dls_excess is None or dls_excess < best_dls_excess[0]:
                        best_dls_excess = (dls_excess, max_dls)
                    continue
                max_inc = _finite_max(stations["INC_deg"])
                if max_inc > float(config.max_inc_deg) + 1e-6:
                    last_problem = (
                        "Боковой ствол превышает configured max INC: "
                        f"{max_inc:.2f} > {float(config.max_inc_deg):.2f} deg."
                    )
                    continue
                score = _candidate_score(stations=stations, config=config)
                if best is None or score < best[0]:
                    best = (score, stations)
                    fallback_used = True
                    # Candidate order prefers the shortest/smallest bend.  A
                    # feasible fallback is already a last-resort path, so do
                    # not evaluate the rest of the comparatively expensive
                    # two-Bezier lattice for this window.
                    break

        if best is None:
            if best_dls_excess is not None:
                _, max_dls = best_dls_excess
                dls_limit = float(config.dls_build_max_deg_per_30m)
                last_problem = (
                    "ПИ бокового ствола превышает лимит расчетной модели: "
                    f"{dls_to_pi(max_dls):.2f} > {dls_to_pi(dls_limit):.2f} deg/10m."
                )
            suffix = f" Последняя причина: {last_problem}" if last_problem else ""
            raise PlanningError(
                "Не удалось построить боковой ствол от окна зарезки до t1/t3." + suffix
            )

        stations = best[1]
        md_t1_m = _md_at_t1(stations)
        summary = _build_summary(
            stations=stations,
            start=start,
            t1=t1,
            t3=t3,
            md_t1_m=md_t1_m,
            horizontal_inc_deg=horizontal_inc_deg,
            horizontal_azi_deg=horizontal_azi_deg,
            config=config,
            fallback_used=fallback_used,
        )
        _validate_target_miss(summary=summary, config=config)
        return PlannerResult(
            stations=stations,
            summary=summary,
            azimuth_deg=horizontal_azi_deg,
            md_t1_m=md_t1_m,
        )


def _control_length_pairs(
    *,
    start: SidetrackStart,
    t1: Point3D,
    t3: Point3D,
    config: TrajectoryConfig,
) -> list[tuple[float, float]]:
    chord_m = max(_distance(start.point, t1), float(config.md_step_m))
    horizontal_m = max(_distance(t1, t3), float(config.md_step_m))
    base_m = max(chord_m, 0.25 * horizontal_m)
    scales = (0.08, 0.14, 0.22, 0.34, 0.50, 0.75, 1.05, 1.40)
    return [
        (
            max(float(config.md_step_m), base_m * lead_scale),
            max(float(config.md_step_m), base_m * tail_scale),
        )
        for lead_scale in scales
        for tail_scale in scales
    ]


def _build_sidetrack_stations(
    *,
    start: SidetrackStart,
    t1: Point3D,
    t3: Point3D,
    horizontal_inc_deg: float,
    horizontal_azi_deg: float,
    lead_m: float,
    tail_m: float,
    config: TrajectoryConfig,
) -> pd.DataFrame:
    start_dir = _unit_vector_from_angles(float(start.inc_deg), float(start.azi_deg))
    end_dir = _unit_vector_from_angles(horizontal_inc_deg, horizontal_azi_deg)
    p0 = _point_array(start.point)
    p3 = _point_array(t1)
    p1 = p0 + start_dir * float(lead_m)
    p2 = p3 - end_dir * float(tail_m)

    build_xyz = _sample_cubic_bezier(
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        step_m=float(config.md_step_m),
    )
    build_stations = _stations_from_xyz(
        xyz=build_xyz,
        segment="BUILD1",
        start_md_m=0.0,
    )
    build_stations.loc[0, "INC_deg"] = float(start.inc_deg)
    build_stations.loc[0, "AZI_deg"] = _normalize_azimuth_deg(float(start.azi_deg))
    build_stations.loc[len(build_stations) - 1, "INC_deg"] = horizontal_inc_deg
    build_stations.loc[len(build_stations) - 1, "AZI_deg"] = horizontal_azi_deg
    build_stations = add_dls(build_stations)

    horizontal_stations = _straight_segment_stations(
        start=t1,
        end=t3,
        inc_deg=horizontal_inc_deg,
        azi_deg=horizontal_azi_deg,
        start_md_m=float(build_stations["MD_m"].iloc[-1]),
        step_m=float(config.md_step_m),
        segment="HORIZONTAL",
    )
    stations = pd.concat(
        [build_stations, horizontal_stations.iloc[1:].copy()],
        ignore_index=True,
    )
    stations = add_dls(stations)
    return stations


def _bend_fallback_station_candidates(
    *,
    start: SidetrackStart,
    t1: Point3D,
    t3: Point3D,
    horizontal_inc_deg: float,
    horizontal_azi_deg: float,
    config: TrajectoryConfig,
) -> Iterator[pd.DataFrame]:
    """Yield longer C1-continuous two-Bezier approaches to t1.

    The extra bend supplies trajectory length for difficult pose changes while
    preserving the exact window pose and the t1->t3 tangent.  The caller still
    enforces the configured DLS and INC limits for every yielded candidate.
    """

    p0 = _point_array(start.point)
    p6 = _point_array(t1)
    chord = p6 - p0
    chord_m = float(np.linalg.norm(chord))
    if chord_m <= SMALL:
        return
    chord_dir = chord / chord_m
    start_dir = _unit_vector_from_angles(float(start.inc_deg), float(start.azi_deg))
    end_dir = _unit_vector_from_angles(horizontal_inc_deg, horizontal_azi_deg)
    bend_basis_1, bend_basis_2 = _bend_basis_vectors(
        chord_dir=chord_dir,
        start_dir=start_dir,
        end_dir=end_dir,
    )
    bend_directions = (
        bend_basis_1,
        -bend_basis_1,
        bend_basis_2,
        -bend_basis_2,
    )
    horizontal_m = max(_distance(t1, t3), float(config.md_step_m))
    bend_base_m = max(chord_m, 0.25 * horizontal_m)

    for fraction in (0.50, 0.38, 0.62):
        chord_waypoint = p0 + chord * float(fraction)
        for offset_scale in (0.20, 0.40, 0.70, 1.00):
            for direction in bend_directions:
                waypoint = chord_waypoint + direction * bend_base_m * offset_scale
                incoming = waypoint - p0
                outgoing = p6 - waypoint
                incoming_m = float(np.linalg.norm(incoming))
                outgoing_m = float(np.linalg.norm(outgoing))
                if incoming_m <= SMALL or outgoing_m <= SMALL:
                    continue
                joint_dir = _normalized_vector(
                    incoming / incoming_m + outgoing / outgoing_m,
                    fallback=chord_dir,
                )
                for handle_scale in (0.22, 0.36, 0.52, 0.72):
                    try:
                        yield _build_bend_sidetrack_stations(
                            start=start,
                            t1=t1,
                            t3=t3,
                            horizontal_inc_deg=horizontal_inc_deg,
                            horizontal_azi_deg=horizontal_azi_deg,
                            waypoint=waypoint,
                            joint_dir=joint_dir,
                            start_handle_m=max(
                                float(config.md_step_m),
                                incoming_m * handle_scale,
                            ),
                            incoming_joint_handle_m=max(
                                float(config.md_step_m),
                                incoming_m * handle_scale,
                            ),
                            outgoing_joint_handle_m=max(
                                float(config.md_step_m),
                                outgoing_m * handle_scale,
                            ),
                            end_handle_m=max(
                                float(config.md_step_m),
                                outgoing_m * handle_scale,
                            ),
                            config=config,
                        )
                    except (ValueError, PlanningError):
                        continue


def _bend_basis_vectors(
    *,
    chord_dir: np.ndarray,
    start_dir: np.ndarray,
    end_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seed = start_dir + end_dir
    seed = seed - chord_dir * float(np.dot(seed, chord_dir))
    if float(np.linalg.norm(seed)) <= SMALL:
        axes = (
            np.asarray([0.0, 0.0, 1.0], dtype=float),
            np.asarray([1.0, 0.0, 0.0], dtype=float),
            np.asarray([0.0, 1.0, 0.0], dtype=float),
        )
        seed = max(
            (axis - chord_dir * float(np.dot(axis, chord_dir)) for axis in axes),
            key=lambda value: float(np.linalg.norm(value)),
        )
    first = _normalized_vector(seed, fallback=np.asarray([1.0, 0.0, 0.0]))
    second = _normalized_vector(
        np.cross(chord_dir, first),
        fallback=np.asarray([0.0, 1.0, 0.0]),
    )
    return first, second


def _normalized_vector(value: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm > SMALL:
        return value / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= SMALL:
        raise PlanningError("Не удалось определить направление загиба бокового ствола.")
    return fallback / fallback_norm


def _build_bend_sidetrack_stations(
    *,
    start: SidetrackStart,
    t1: Point3D,
    t3: Point3D,
    horizontal_inc_deg: float,
    horizontal_azi_deg: float,
    waypoint: np.ndarray,
    joint_dir: np.ndarray,
    start_handle_m: float,
    incoming_joint_handle_m: float,
    outgoing_joint_handle_m: float,
    end_handle_m: float,
    config: TrajectoryConfig,
) -> pd.DataFrame:
    p0 = _point_array(start.point)
    p6 = _point_array(t1)
    start_dir = _unit_vector_from_angles(float(start.inc_deg), float(start.azi_deg))
    end_dir = _unit_vector_from_angles(horizontal_inc_deg, horizontal_azi_deg)

    first_xyz = _sample_cubic_bezier(
        p0=p0,
        p1=p0 + start_dir * float(start_handle_m),
        p2=waypoint - joint_dir * float(incoming_joint_handle_m),
        p3=waypoint,
        step_m=float(config.md_step_m),
    )
    second_xyz = _sample_cubic_bezier(
        p0=waypoint,
        p1=waypoint + joint_dir * float(outgoing_joint_handle_m),
        p2=p6 - end_dir * float(end_handle_m),
        p3=p6,
        step_m=float(config.md_step_m),
    )
    build1 = _stations_from_xyz(xyz=first_xyz, segment="BUILD1", start_md_m=0.0)
    build2 = _stations_from_xyz(
        xyz=second_xyz,
        segment="BUILD2",
        start_md_m=float(build1["MD_m"].iloc[-1]),
    )
    joint_inc_deg, joint_azi_deg = _angles_from_vector(joint_dir)
    build1.loc[0, "INC_deg"] = float(start.inc_deg)
    build1.loc[0, "AZI_deg"] = _normalize_azimuth_deg(float(start.azi_deg))
    build1.loc[len(build1) - 1, "INC_deg"] = joint_inc_deg
    build1.loc[len(build1) - 1, "AZI_deg"] = joint_azi_deg
    build2.loc[0, "INC_deg"] = joint_inc_deg
    build2.loc[0, "AZI_deg"] = joint_azi_deg
    build2.loc[len(build2) - 1, "INC_deg"] = horizontal_inc_deg
    build2.loc[len(build2) - 1, "AZI_deg"] = horizontal_azi_deg
    build = pd.concat([build1, build2.iloc[1:].copy()], ignore_index=True)

    horizontal = _straight_segment_stations(
        start=t1,
        end=t3,
        inc_deg=horizontal_inc_deg,
        azi_deg=horizontal_azi_deg,
        start_md_m=float(build["MD_m"].iloc[-1]),
        step_m=float(config.md_step_m),
        segment="HORIZONTAL",
    )
    return add_dls(
        pd.concat([build, horizontal.iloc[1:].copy()], ignore_index=True)
    )


def _sample_cubic_bezier(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    step_m: float,
) -> np.ndarray:
    chord_m = float(np.linalg.norm(p3 - p0))
    control_m = float(
        np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
    )
    samples = max(int(math.ceil(max(chord_m, control_m) / max(step_m, 1.0))), 8)
    samples = min(samples, 1600)
    t = np.linspace(0.0, 1.0, samples + 1)
    omt = 1.0 - t
    xyz = (
        (omt**3)[:, None] * p0
        + (3.0 * omt * omt * t)[:, None] * p1
        + (3.0 * omt * t * t)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    return _deduplicate_xyz(xyz)


def _stations_from_xyz(
    *,
    xyz: np.ndarray,
    segment: str,
    start_md_m: float,
) -> pd.DataFrame:
    if len(xyz) < 2:
        raise PlanningError(
            "Недостаточно точек для построения участка бокового ствола."
        )
    distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if np.any(distances <= SMALL):
        raise PlanningError("Участок бокового ствола содержит совпадающие станции.")
    md = float(start_md_m) + np.concatenate([[0.0], np.cumsum(distances)])
    stations = build_reference_trajectory_stations(
        xs=[float(value) for value in xyz[:, 0]],
        ys=[float(value) for value in xyz[:, 1]],
        zs=[float(value) for value in xyz[:, 2]],
        mds=[float(value) for value in md],
    )
    stations["segment"] = str(segment)
    return stations


def _straight_segment_stations(
    *,
    start: Point3D,
    end: Point3D,
    inc_deg: float,
    azi_deg: float,
    start_md_m: float,
    step_m: float,
    segment: str,
) -> pd.DataFrame:
    p0 = _point_array(start)
    p1 = _point_array(end)
    length_m = float(np.linalg.norm(p1 - p0))
    if length_m <= SMALL:
        raise PlanningError("t3 совпадает с t1, продуктивный участок не задан.")
    samples = max(int(math.ceil(length_m / max(step_m, 1.0))), 1)
    fractions = np.linspace(0.0, 1.0, samples + 1)
    xyz = p0 + (p1 - p0) * fractions[:, None]
    md = float(start_md_m) + length_m * fractions
    stations = pd.DataFrame(
        {
            "MD_m": md,
            "INC_deg": [float(inc_deg)] * len(md),
            "AZI_deg": [_normalize_azimuth_deg(float(azi_deg))] * len(md),
            "X_m": xyz[:, 0],
            "Y_m": xyz[:, 1],
            "Z_m": xyz[:, 2],
            "segment": [str(segment)] * len(md),
        }
    )
    return add_dls(stations)


def _candidate_score(*, stations: pd.DataFrame, config: TrajectoryConfig) -> float:
    dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)
    dls_values = dls_values[np.isfinite(dls_values)]
    max_dls = float(np.max(dls_values)) if len(dls_values) else 0.0
    max_inc = float(np.nanmax(stations["INC_deg"].to_numpy(dtype=float)))
    md_total = float(stations["MD_m"].iloc[-1])
    dls_limit = max(float(config.dls_build_max_deg_per_30m), SMALL)
    inc_excess = max(0.0, max_inc - float(config.max_inc_deg))
    dls_excess = max(0.0, max_dls - dls_limit)
    return float(max_dls + 4.0 * dls_excess + 10.0 * inc_excess + 0.001 * md_total)


def _build_summary(
    *,
    stations: pd.DataFrame,
    start: SidetrackStart,
    t1: Point3D,
    t3: Point3D,
    md_t1_m: float,
    horizontal_inc_deg: float,
    horizontal_azi_deg: float,
    config: TrajectoryConfig,
    fallback_used: bool = False,
) -> SummaryDict:
    t1_row = stations.loc[int((stations["MD_m"] - float(md_t1_m)).abs().idxmin())]
    t3_row = stations.iloc[-1]
    t1_dx, t1_dy, t1_dz, t1_lateral, t1_vertical, t1_distance = _target_miss(
        row=t1_row, target=t1
    )
    t3_dx, t3_dy, t3_dz, t3_lateral, t3_vertical, t3_distance = _target_miss(
        row=t3_row, target=t3
    )
    max_dls = _finite_max(stations["DLS_deg_per_30m"])
    max_inc = float(np.nanmax(stations["INC_deg"].to_numpy(dtype=float)))
    md_total = float(stations["MD_m"].iloc[-1])
    build1_dls = _segment_dls_max(stations, "BUILD1")
    build2_dls = _segment_dls_max(stations, "BUILD2")
    build_dls = max(build1_dls, build2_dls)
    horizontal_dls = _segment_dls_max(stations, "HORIZONTAL")
    dls_limit = float(config.dls_build_max_deg_per_30m)
    summary: SummaryDict = {
        "distance_t1_m": t1_distance,
        "distance_t3_m": t3_distance,
        "lateral_distance_t1_m": t1_lateral,
        "lateral_distance_t3_m": t3_lateral,
        "vertical_distance_t1_m": t1_vertical,
        "vertical_distance_t3_m": t3_vertical,
        "distance_t1_control_m": t1_distance,
        "distance_t3_control_m": t3_distance,
        "control_gap_t1_m": 0.0,
        "control_gap_t3_m": 0.0,
        "kop_vertical_m": 0.0,
        "kop_md_m": 0.0,
        "entry_inc_deg": float(t1_row["INC_deg"]),
        "entry_inc_control_deg": float(t1_row["INC_deg"]),
        "entry_inc_target_deg": float(config.entry_inc_target_deg),
        "entry_inc_tolerance_deg": float(config.entry_inc_tolerance_deg),
        "lateral_tolerance_m": float(config.lateral_tolerance_m),
        "vertical_tolerance_m": float(config.vertical_tolerance_m),
        "max_inc_deg": float(config.max_inc_deg),
        "max_inc_actual_deg": max_inc,
        "inc_required_t1_t3_deg": float(horizontal_inc_deg),
        "horizontal_adjust_length_m": 0.0,
        "horizontal_hold_length_m": _distance(t1, t3),
        "horizontal_inc_deg": float(horizontal_inc_deg),
        "hold_inc_deg": float(horizontal_inc_deg),
        "hold_length_m": 0.0,
        "build_dls_selected_deg_per_30m": build_dls,
        "build1_dls_selected_deg_per_30m": build1_dls,
        "build2_dls_selected_deg_per_30m": build2_dls,
        "build_dls_max_config_deg_per_30m": dls_limit,
        "build_dls_relaxed_from_max": "no",
        "build_dls_split_selected": "yes" if fallback_used else "no",
        "max_dls_total_deg_per_30m": max_dls,
        "md_total_m": md_total,
        "max_total_md_postcheck_m": float(config.max_total_md_postcheck_m),
        "md_postcheck_excess_m": max(
            0.0, md_total - float(config.max_total_md_postcheck_m)
        ),
        "md_postcheck_exceeded": (
            "yes" if md_total > float(config.max_total_md_postcheck_m) + 1e-6 else "no"
        ),
        "dls_postcheck_excess_deg_per_30m": max(0.0, max_dls - dls_limit),
        "t1_horizontal_offset_m": _horizontal_offset(start.point, t1),
        "horizontal_length_m": _distance(t1, t3),
        "trajectory_type": (
            "Sidetrack Bend Fallback + Horizontal"
            if fallback_used
            else "Sidetrack Bezier + Horizontal"
        ),
        "trajectory_target_direction": "Боковой продуктивный ствол",
        "well_complexity": "Пилот + боковой ствол",
        "well_complexity_by_offset": "Пилот + боковой ствол",
        "well_complexity_by_hold": "Пилот + боковой ствол",
        "hold_azimuth_deg": float(horizontal_azi_deg),
        "entry_azimuth_deg": float(horizontal_azi_deg),
        "t1_exact_x_m": float(t1_row["X_m"]),
        "t1_exact_y_m": float(t1_row["Y_m"]),
        "t1_exact_z_m": float(t1_row["Z_m"]),
        "t3_exact_x_m": float(t3_row["X_m"]),
        "t3_exact_y_m": float(t3_row["Y_m"]),
        "t3_exact_z_m": float(t3_row["Z_m"]),
        "t1_miss_dx_m": t1_dx,
        "t1_miss_dy_m": t1_dy,
        "t1_miss_dz_m": t1_dz,
        "t3_miss_dx_m": t3_dx,
        "t3_miss_dy_m": t3_dy,
        "t3_miss_dz_m": t3_dz,
        "azimuth_turn_deg": abs(
            _shortest_azimuth_delta_deg(float(start.azi_deg), float(horizontal_azi_deg))
        ),
        "solver_turn_mode": (
            "sidetrack_bend_fallback" if fallback_used else "sidetrack_bezier"
        ),
        "solver_turn_max_restarts": int(config.turn_solver_max_restarts),
        "solver_turn_restarts_used": 0,
        "solver_turn_attempts_used": 1,
        "solver_turn_search_depth_scale": 1.0,
        "solver_turn_seed_lattice_points": 0,
        "solver_turn_local_max_nfev": 0,
        "optimization_mode": str(config.optimization_mode),
        "optimization_status": "sidetrack_geometric",
        "optimization_objective_value_m": md_total,
        "optimization_theoretical_lower_bound_m": _distance(start.point, t1)
        + _distance(t1, t3),
        "optimization_absolute_gap_m": 0.0,
        "optimization_relative_gap_pct": 0.0,
        "optimization_seeds_used": 0,
        "optimization_runs_used": 0,
        "solver_strategy": (
            "pilot_sidetrack_bend_fallback"
            if fallback_used
            else "pilot_sidetrack_bezier"
        ),
        "sidetrack_fallback_used": "yes" if fallback_used else "no",
        "max_dls_vertical_deg_per_30m": 0.0,
        "max_dls_build1_deg_per_30m": build1_dls,
        "max_dls_hold_deg_per_30m": 0.0,
        "max_dls_build2_deg_per_30m": build2_dls,
        "max_dls_horizontal_deg_per_30m": horizontal_dls,
    }
    for segment, limit in config.dls_limits_deg_per_30m.items():
        summary[f"dls_limit_{segment.lower()}_deg_per_30m"] = float(limit)
    return summary


def _validate_target_miss(
    *,
    summary: SummaryDict,
    config: TrajectoryConfig,
) -> None:
    if (
        float(summary["lateral_distance_t1_m"])
        > float(config.lateral_tolerance_m) + SMALL
        or float(summary["vertical_distance_t1_m"])
        > float(config.vertical_tolerance_m) + SMALL
    ):
        raise PlanningError(
            "Боковой ствол не попал в t1 в пределах допуска. "
            f"Miss lateral={float(summary['lateral_distance_t1_m']):.2f} м, "
            f"vertical={float(summary['vertical_distance_t1_m']):.2f} м."
        )
    if (
        float(summary["lateral_distance_t3_m"])
        > float(config.lateral_tolerance_m) + SMALL
        or float(summary["vertical_distance_t3_m"])
        > float(config.vertical_tolerance_m) + SMALL
    ):
        raise PlanningError(
            "Боковой ствол не попал в t3 в пределах допуска. "
            f"Miss lateral={float(summary['lateral_distance_t3_m']):.2f} м, "
            f"vertical={float(summary['vertical_distance_t3_m']):.2f} м."
        )
    if float(summary["max_inc_actual_deg"]) > float(config.max_inc_deg) + 1e-6:
        raise PlanningError(
            "Боковой ствол превышает configured max INC. "
            f"max actual INC={float(summary['max_inc_actual_deg']):.2f} deg, "
            f"limit={float(config.max_inc_deg):.2f} deg."
        )


def _md_at_t1(stations: pd.DataFrame) -> float:
    build_rows = stations.loc[stations["segment"].isin(("BUILD1", "BUILD2"))]
    if build_rows.empty:
        raise PlanningError("Не удалось определить MD входа в t1 для бокового ствола.")
    return float(build_rows["MD_m"].iloc[-1])


def _angles_from_points(start: Point3D, end: Point3D) -> tuple[float, float]:
    dx = float(end.x) - float(start.x)
    dy = float(end.y) - float(start.y)
    dz = float(end.z) - float(start.z)
    horizontal = float(math.hypot(dx, dy))
    if horizontal <= SMALL and abs(dz) <= SMALL:
        raise PlanningError("Невозможно определить направление для совпадающих точек.")
    inc_deg = float(math.degrees(math.atan2(horizontal, dz)))
    azi_deg = (
        0.0
        if horizontal <= SMALL
        else _normalize_azimuth_deg(math.degrees(math.atan2(dx, dy)))
    )
    return inc_deg, azi_deg


def _angles_from_vector(direction: np.ndarray) -> tuple[float, float]:
    vector = _normalized_vector(
        np.asarray(direction, dtype=float),
        fallback=np.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    horizontal = float(math.hypot(float(vector[0]), float(vector[1])))
    inc_deg = float(math.degrees(math.atan2(horizontal, float(vector[2]))))
    azi_deg = (
        0.0
        if horizontal <= SMALL
        else _normalize_azimuth_deg(
            math.degrees(math.atan2(float(vector[0]), float(vector[1])))
        )
    )
    return inc_deg, azi_deg


def _unit_vector_from_angles(inc_deg: float, azi_deg: float) -> np.ndarray:
    inc_rad = math.radians(float(inc_deg))
    azi_rad = math.radians(float(azi_deg))
    return np.asarray(
        [
            math.sin(inc_rad) * math.sin(azi_rad),
            math.sin(inc_rad) * math.cos(azi_rad),
            math.cos(inc_rad),
        ],
        dtype=float,
    )


def _deduplicate_xyz(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) <= 1:
        return xyz
    keep = [0]
    for index in range(1, len(xyz)):
        if float(np.linalg.norm(xyz[index] - xyz[keep[-1]])) > SMALL:
            keep.append(index)
    return xyz[keep]


def _target_miss(
    *,
    row: pd.Series,
    target: Point3D,
) -> tuple[float, float, float, float, float, float]:
    dx = float(row["X_m"]) - float(target.x)
    dy = float(row["Y_m"]) - float(target.y)
    dz = float(row["Z_m"]) - float(target.z)
    lateral = float(math.hypot(dx, dy))
    vertical = float(abs(dz))
    distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
    return dx, dy, dz, lateral, vertical, distance


def _segment_dls_max(stations: pd.DataFrame, segment: str) -> float:
    rows = stations.loc[stations["segment"] == str(segment), "DLS_deg_per_30m"]
    if rows.empty:
        return 0.0
    return _finite_max(rows)


def _finite_max(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return 0.0
    return float(np.max(array))


def _point_array(point: Point3D) -> np.ndarray:
    return np.asarray([float(point.x), float(point.y), float(point.z)], dtype=float)


def _distance(left: Point3D, right: Point3D) -> float:
    return float(np.linalg.norm(_point_array(right) - _point_array(left)))


def _horizontal_offset(left: Point3D, right: Point3D) -> float:
    return float(
        math.hypot(float(right.x) - float(left.x), float(right.y) - float(left.y))
    )


def _normalize_azimuth_deg(value: float) -> float:
    return float(value % 360.0)


def _shortest_azimuth_delta_deg(
    azimuth_from_deg: float, azimuth_to_deg: float
) -> float:
    delta = _normalize_azimuth_deg(float(azimuth_to_deg) - float(azimuth_from_deg))
    if delta > 180.0:
        delta -= 360.0
    return float(delta)
