from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.constants import RAD2DEG
from pywp.mcm import add_dls, dogleg_angle_rad
from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.planner_types import PlanningError
from pywp.ui_utils import dls_to_pi

SMALL = 1e-9
MAX_TRANSITION_MD_MULTIPLIER = 4.0


@dataclass(frozen=True)
class MultiHorizontalTransition:
    level_from: int
    level_to: int
    gap_m: float
    required_build_m: float
    excess_m: float
    max_feasible_delta_z_m: float


def extend_plan_with_multi_horizontal_targets(
    *,
    base_result: PlannerResult,
    target_pairs: tuple[tuple[Point3D, Point3D], ...],
    config: TrajectoryConfig,
) -> PlannerResult:
    if len(target_pairs) <= 1:
        return base_result

    stations = pd.DataFrame(base_result.stations).copy().reset_index(drop=True)
    if stations.empty:
        raise PlanningError("Многопластовая скважина: базовая траектория пуста.")
    required_columns = {"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"}
    missing_columns = required_columns.difference(stations.columns)
    if missing_columns:
        raise PlanningError(
            "Многопластовая скважина: базовая траектория не содержит колонки "
            f"{', '.join(sorted(missing_columns))}."
        )

    if "segment" in stations.columns:
        stations.loc[stations["segment"] == "HORIZONTAL", "segment"] = "HORIZONTAL1"

    rows: list[dict[str, object]] = []
    current = _row_state(stations.iloc[-1])
    transitions: list[MultiHorizontalTransition] = []
    horizontal_lengths = [_point_distance(*target_pairs[0])]
    final_azimuth_deg = float(base_result.azimuth_deg)

    for pair_index, (next_t1, next_t3) in enumerate(target_pairs[1:], start=2):
        next_hold = _direction_angles_between(next_t1, next_t3)
        final_azimuth_deg = float(next_hold[1])
        _validate_inc_limit(
            inc_deg=next_hold[0],
            config=config,
            context=f"горизонтальный участок {pair_index}_t1 → {pair_index}_t3",
        )
        horizontal_length = _point_distance(next_t1, next_t3)
        if horizontal_length <= SMALL:
            raise PlanningError(
                "Многопластовая скважина: горизонтальный участок "
                f"{pair_index}_t1 → {pair_index}_t3 имеет нулевую длину."
            )

        transition = _transition_feasibility(
            current=current,
            target=next_t1,
            target_inc_deg=next_hold[0],
            target_azi_deg=next_hold[1],
            config=config,
            level_from=pair_index - 1,
            level_to=pair_index,
        )
        if transition.excess_m > 1e-6:
            raise PlanningError(_transition_problem_text(transition))
        transitions.append(transition)

        transition_rows = _smooth_transition_rows(
            current=current,
            target=next_t1,
            target_inc_deg=next_hold[0],
            target_azi_deg=next_hold[1],
            segment_name=f"HORIZONTAL_BUILD{pair_index - 1}",
            config=config,
        )
        rows.extend(transition_rows)
        current = _row_state_from_payload(transition_rows[-1])

        horizontal_rows = _linear_hold_rows(
            current=current,
            target=next_t3,
            inc_deg=next_hold[0],
            azi_deg=next_hold[1],
            segment_name=f"HORIZONTAL{pair_index}",
            config=config,
        )
        rows.extend(horizontal_rows)
        current = _row_state_from_payload(horizontal_rows[-1])
        horizontal_lengths.append(horizontal_length)

    if rows:
        stations = pd.concat([stations, pd.DataFrame(rows)], ignore_index=True)
        stations = add_dls(stations)
    _validate_extended_stations(
        stations=stations,
        final_target=target_pairs[-1][1],
        config=config,
    )

    summary = dict(base_result.summary)
    final_target = target_pairs[-1][1]
    max_dls = float(np.nanmax(stations["DLS_deg_per_30m"].to_numpy(dtype=float)))
    md_total_m = float(stations["MD_m"].iloc[-1])
    md_postcheck_limit_m = float(config.max_total_md_postcheck_m)
    md_postcheck_excess_m = float(max(0.0, md_total_m - md_postcheck_limit_m))
    summary.update(
        {
            "trajectory_type": "Multi-horizontal",
            "multi_horizontal": "yes",
            "multi_horizontal_levels": int(len(target_pairs)),
            "multi_horizontal_transition_count": int(len(target_pairs) - 1),
            "multi_horizontal_min_transition_gap_m": float(
                min((item.gap_m for item in transitions), default=0.0)
            ),
            "multi_horizontal_max_transition_excess_m": 0.0,
            "horizontal_length_m": float(sum(horizontal_lengths)),
            "max_dls_total_deg_per_30m": max_dls,
            "md_total_m": md_total_m,
            "md_postcheck_excess_m": md_postcheck_excess_m,
            "md_postcheck_exceeded": "yes" if md_postcheck_excess_m > 1e-6 else "no",
            "distance_t3_m": 0.0,
            "lateral_distance_t3_m": 0.0,
            "vertical_distance_t3_m": 0.0,
            "distance_t3_control_m": 0.0,
            "t3_exact_x_m": float(final_target.x),
            "t3_exact_y_m": float(final_target.y),
            "t3_exact_z_m": float(final_target.z),
            "t3_miss_dx_m": 0.0,
            "t3_miss_dy_m": 0.0,
            "t3_miss_dz_m": 0.0,
        }
    )
    return PlannerResult(
        stations=stations,
        summary=summary,
        azimuth_deg=final_azimuth_deg,
        md_t1_m=float(base_result.md_t1_m),
    )


def _validate_extended_stations(
    *,
    stations: pd.DataFrame,
    final_target: Point3D,
    config: TrajectoryConfig,
) -> None:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if len(md_values) < 2 or np.any(~np.isfinite(md_values)):
        raise PlanningError("Многопластовая скважина: некорректная сетка MD.")
    if np.any(np.diff(md_values) <= 0.0):
        raise PlanningError("Многопластовая скважина: MD должен строго возрастать.")

    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    finite_inc = inc_values[np.isfinite(inc_values)]
    max_inc = float(np.max(finite_inc)) if len(finite_inc) else 0.0
    if max_inc > float(config.max_inc_deg) + 1e-6:
        raise PlanningError(
            "Многопластовая скважина: построенный HORIZONTAL_BUILD превышает "
            f"ограничение INC ({max_inc:.3f} > {float(config.max_inc_deg):.3f}°). "
            "Увеличьте горизонтальный зазор между уровнями или уменьшите ΔZ."
        )

    dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)
    finite_dls = dls_values[np.isfinite(dls_values)]
    max_dls = float(np.max(finite_dls)) if len(finite_dls) else 0.0
    if max_dls > float(config.dls_build_max_deg_per_30m) + 1e-6:
        raise PlanningError(
            "Многопластовая скважина: построенный HORIZONTAL_BUILD превышает "
            f"лимит ПИ ({dls_to_pi(max_dls):.3f} > "
            f"{dls_to_pi(float(config.dls_build_max_deg_per_30m)):.3f} deg/10m)."
        )

    final_row = stations.iloc[-1]
    miss_m = float(
        np.linalg.norm(
            np.array(
                [
                    float(final_row["X_m"]) - float(final_target.x),
                    float(final_row["Y_m"]) - float(final_target.y),
                    float(final_row["Z_m"]) - float(final_target.z),
                ],
                dtype=float,
            )
        )
    )
    if miss_m > 1e-6:
        raise PlanningError(
            "Многопластовая скважина: конечная станция не совпала с последней "
            f"точкой цели (промах {miss_m:.6f} м)."
        )


def _transition_feasibility(
    *,
    current: dict[str, float],
    target: Point3D,
    target_inc_deg: float,
    target_azi_deg: float,
    config: TrajectoryConfig,
    level_from: int,
    level_to: int,
) -> MultiHorizontalTransition:
    gap = _distance_from_current(current, target)
    if gap <= SMALL:
        raise PlanningError(
            "Многопластовая скважина: точки "
            f"{level_from}_t3 и {level_to}_t1 совпадают."
        )
    direct_inc, direct_azi = _direction_angles_from_current(current, target)
    _validate_inc_limit(
        inc_deg=direct_inc,
        config=config,
        context=f"переход {level_from}_t3 → {level_to}_t1",
    )
    dls_limit = float(config.dls_build_max_deg_per_30m)
    if dls_limit <= SMALL:
        raise PlanningError(
            "Многопластовая скважина: для HORIZONTAL_BUILD требуется положительный "
            "максимальный ПИ."
        )
    build_in_m = _dogleg_build_length_m(
        float(current["inc_deg"]),
        float(current["azi_deg"]),
        direct_inc,
        direct_azi,
        dls_limit,
    )
    build_out_m = _dogleg_build_length_m(
        direct_inc,
        direct_azi,
        target_inc_deg,
        target_azi_deg,
        dls_limit,
    )
    required = float(build_in_m + build_out_m)
    return MultiHorizontalTransition(
        level_from=int(level_from),
        level_to=int(level_to),
        gap_m=float(gap),
        required_build_m=required,
        excess_m=float(max(0.0, required - gap)),
        max_feasible_delta_z_m=_max_feasible_delta_z_m(
            current=current,
            target=target,
            target_inc_deg=target_inc_deg,
            target_azi_deg=target_azi_deg,
            dls_limit_deg_per_30m=dls_limit,
        ),
    )


def _smooth_transition_rows(
    *,
    current: dict[str, float],
    target: Point3D,
    target_inc_deg: float,
    target_azi_deg: float,
    segment_name: str,
    config: TrajectoryConfig,
) -> list[dict[str, object]]:
    gap = _distance_from_current(current, target)
    p0 = np.array([current["x"], current["y"], current["z"]], dtype=float)
    p3 = np.array([target.x, target.y, target.z], dtype=float)
    start_dir = _xyz_direction_vector(
        inc_deg=float(current["inc_deg"]),
        azi_deg=float(current["azi_deg"]),
    )
    end_dir = _xyz_direction_vector(
        inc_deg=float(target_inc_deg),
        azi_deg=float(target_azi_deg),
    )
    dls_limit = float(config.dls_build_max_deg_per_30m)
    min_control_m = float(
        max(
            float(config.min_structural_segment_m),
            float(config.md_step_m),
            1.0,
        )
    )
    control_lengths = _candidate_control_lengths(
        gap_m=gap,
        min_control_m=min_control_m,
    )

    best: pd.DataFrame | None = None
    best_score = float("inf")
    best_dls = float("inf")
    best_inc = float("inf")
    for lead_m in control_lengths:
        for tail_m in control_lengths:
            p1 = p0 + start_dir * float(lead_m)
            p2 = p3 - end_dir * float(tail_m)
            xyz = _sample_cubic_bezier(
                p0=p0,
                p1=p1,
                p2=p2,
                p3=p3,
                step_m=float(config.md_step_m),
            )
            try:
                candidate = _stations_from_xyz_path(
                    xyz=xyz,
                    segment_name=segment_name,
                    start_md_m=float(current["md_m"]),
                    start_inc_deg=float(current["inc_deg"]),
                    start_azi_deg=float(current["azi_deg"]),
                    end_inc_deg=float(target_inc_deg),
                    end_azi_deg=float(target_azi_deg),
                )
            except PlanningError:
                continue
            md_span = float(candidate["MD_m"].iloc[-1] - candidate["MD_m"].iloc[0])
            if md_span > max(float(gap) * MAX_TRANSITION_MD_MULTIPLIER, float(gap) + 500.0):
                continue
            max_dls = _max_finite(candidate["DLS_deg_per_30m"].to_numpy(dtype=float))
            max_inc = _max_finite(candidate["INC_deg"].to_numpy(dtype=float))
            dls_excess = max(0.0, max_dls - dls_limit)
            inc_excess = max(0.0, max_inc - float(config.max_inc_deg))
            score = float(
                1000.0 * dls_excess
                + 1000.0 * inc_excess
                + max_dls
                + 0.001 * md_span
            )
            if score < best_score:
                best = candidate
                best_score = score
                best_dls = max_dls
                best_inc = max_inc

    if best is None:
        raise PlanningError(
            "Многопластовая скважина: не удалось построить плавный "
            f"{segment_name} между уровнями без вырожденной геометрии. "
            "Увеличьте расстояние между горизонтальными участками или уменьшите ΔZ."
        )
    if best_dls > dls_limit + 1e-6:
        raise PlanningError(
            "Многопластовая скважина: плавный "
            f"{segment_name} требует ПИ {dls_to_pi(best_dls):.2f} deg/10m, что выше "
            f"лимита {dls_to_pi(dls_limit):.2f}. Увеличьте расстояние между уровнями, "
            "сократите соседние мини-горизонты или уменьшите ΔZ."
        )
    if best_inc > float(config.max_inc_deg) + 1e-6:
        raise PlanningError(
            "Многопластовая скважина: плавный "
            f"{segment_name} требует INC {best_inc:.2f}°, что выше "
            f"ограничения {float(config.max_inc_deg):.2f}°. "
            "Увеличьте горизонтальный зазор между уровнями или уменьшите ΔZ."
        )
    return [dict(row) for row in best.iloc[1:].to_dict("records")]


def _candidate_control_lengths(*, gap_m: float, min_control_m: float) -> tuple[float, ...]:
    values = {
        float(min_control_m),
        *(float(gap_m) * scale for scale in (0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50)),
    }
    return tuple(sorted(value for value in values if value > SMALL))


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
        np.linalg.norm(p1 - p0)
        + np.linalg.norm(p2 - p1)
        + np.linalg.norm(p3 - p2)
    )
    samples = max(int(np.ceil(max(chord_m, control_m) / max(float(step_m), 1.0))), 8)
    samples = min(samples, 2000)
    t = np.linspace(0.0, 1.0, samples + 1)
    omt = 1.0 - t
    xyz = (
        (omt**3)[:, None] * p0
        + (3.0 * omt * omt * t)[:, None] * p1
        + (3.0 * omt * t * t)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    return _deduplicate_xyz(xyz)


def _deduplicate_xyz(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=float)
    if len(points) <= 1:
        return points
    keep = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[keep[-1]])) > SMALL:
            keep.append(index)
    return points[keep]


def _stations_from_xyz_path(
    *,
    xyz: np.ndarray,
    segment_name: str,
    start_md_m: float,
    start_inc_deg: float,
    start_azi_deg: float,
    end_inc_deg: float,
    end_azi_deg: float,
) -> pd.DataFrame:
    points = np.asarray(xyz, dtype=float)
    if len(points) < 3:
        raise PlanningError("Многопластовая скважина: слишком мало точек перехода.")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(~np.isfinite(distances)) or np.any(distances <= SMALL):
        raise PlanningError(
            "Многопластовая скважина: переход содержит совпадающие станции."
        )
    md = float(start_md_m) + np.concatenate([[0.0], np.cumsum(distances)])
    tangent_x = np.gradient(points[:, 0], md, edge_order=2)
    tangent_y = np.gradient(points[:, 1], md, edge_order=2)
    tangent_z = np.gradient(points[:, 2], md, edge_order=2)
    horizontal = np.hypot(tangent_x, tangent_y)
    inc_deg = np.degrees(np.arctan2(horizontal, tangent_z))
    azi_deg = (np.degrees(np.arctan2(tangent_x, tangent_y)) + 360.0) % 360.0
    inc_deg[0] = float(start_inc_deg)
    azi_deg[0] = float(start_azi_deg) % 360.0
    inc_deg[-1] = float(end_inc_deg)
    azi_deg[-1] = float(end_azi_deg) % 360.0
    stations = pd.DataFrame(
        {
            "MD_m": md,
            "INC_deg": inc_deg,
            "AZI_deg": azi_deg,
            "segment": str(segment_name),
            "N_m": points[:, 1],
            "E_m": points[:, 0],
            "TVD_m": points[:, 2],
            "X_m": points[:, 0],
            "Y_m": points[:, 1],
            "Z_m": points[:, 2],
        }
    )
    return add_dls(stations)


def _max_finite(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if len(finite) else 0.0


def _linear_hold_rows(
    *,
    current: dict[str, float],
    target: Point3D,
    inc_deg: float,
    azi_deg: float,
    segment_name: str,
    config: TrajectoryConfig,
) -> list[dict[str, object]]:
    if _distance_from_current(current, target) <= SMALL:
        raise PlanningError(
            f"Многопластовая скважина: участок {segment_name} имеет нулевую длину."
        )
    return _straight_rows(
        current=current,
        target=target,
        segment_name=segment_name,
        md_step_m=float(config.md_step_m),
        inc_deg=float(inc_deg),
        azi_deg=float(azi_deg),
    )


def _straight_rows(
    *,
    current: dict[str, float],
    target: Point3D,
    segment_name: str,
    md_step_m: float,
    inc_deg: float,
    azi_deg: float,
) -> list[dict[str, object]]:
    start_xyz = np.array([current["x"], current["y"], current["z"]], dtype=float)
    end_xyz = np.array([target.x, target.y, target.z], dtype=float)
    delta = end_xyz - start_xyz
    length = float(np.linalg.norm(delta))
    if length <= SMALL:
        return []
    step = max(float(md_step_m), SMALL)
    distances = np.arange(step, length, step, dtype=float)
    if len(distances) == 0 or abs(float(distances[-1]) - length) > 1e-9:
        distances = np.append(distances, length)
    else:
        distances[-1] = length

    rows: list[dict[str, object]] = []
    for distance in distances:
        fraction = float(distance / length)
        xyz = start_xyz + delta * fraction
        rows.append(
            {
                "MD_m": float(current["md_m"] + distance),
                "INC_deg": float(inc_deg),
                "AZI_deg": float(azi_deg % 360.0),
                "segment": str(segment_name),
                "N_m": float(xyz[1]),
                "E_m": float(xyz[0]),
                "TVD_m": float(xyz[2]),
                "X_m": float(xyz[0]),
                "Y_m": float(xyz[1]),
                "Z_m": float(xyz[2]),
            }
        )
    return rows


def _row_state(row: pd.Series) -> dict[str, float]:
    return {
        "md_m": float(row["MD_m"]),
        "inc_deg": float(row["INC_deg"]),
        "azi_deg": float(row["AZI_deg"]),
        "x": float(row["X_m"]),
        "y": float(row["Y_m"]),
        "z": float(row["Z_m"]),
    }


def _row_state_from_payload(row: dict[str, object]) -> dict[str, float]:
    return {
        "md_m": float(row["MD_m"]),
        "inc_deg": float(row["INC_deg"]),
        "azi_deg": float(row["AZI_deg"]),
        "x": float(row["X_m"]),
        "y": float(row["Y_m"]),
        "z": float(row["Z_m"]),
    }


def _distance_from_current(current: dict[str, float], target: Point3D) -> float:
    return float(
        np.linalg.norm(
            np.array(
                [
                    float(target.x) - float(current["x"]),
                    float(target.y) - float(current["y"]),
                    float(target.z) - float(current["z"]),
                ],
                dtype=float,
            )
        )
    )


def _point_distance(point_a: Point3D, point_b: Point3D) -> float:
    return float(
        np.linalg.norm(
            np.array(
                [
                    float(point_b.x) - float(point_a.x),
                    float(point_b.y) - float(point_a.y),
                    float(point_b.z) - float(point_a.z),
                ],
                dtype=float,
            )
        )
    )


def _direction_angles_between(point_a: Point3D, point_b: Point3D) -> tuple[float, float]:
    dx = float(point_b.x) - float(point_a.x)
    dy = float(point_b.y) - float(point_a.y)
    dz = float(point_b.z) - float(point_a.z)
    return _direction_angles_from_delta(dx=dx, dy=dy, dz=dz)


def _direction_angles_from_current(
    current: dict[str, float],
    target: Point3D,
) -> tuple[float, float]:
    dx = float(target.x) - float(current["x"])
    dy = float(target.y) - float(current["y"])
    dz = float(target.z) - float(current["z"])
    return _direction_angles_from_delta(dx=dx, dy=dy, dz=dz)


def _direction_angles_from_delta(*, dx: float, dy: float, dz: float) -> tuple[float, float]:
    horizontal = float(np.hypot(dx, dy))
    if horizontal <= SMALL and abs(float(dz)) <= SMALL:
        raise PlanningError("Многопластовая скважина: нулевая длина участка.")
    inc_deg = float(np.degrees(np.arctan2(horizontal, float(dz))))
    azi_deg = float(np.degrees(np.arctan2(float(dx), float(dy))) % 360.0)
    return inc_deg, azi_deg


def _dogleg_build_length_m(
    inc_from_deg: float,
    azi_from_deg: float,
    inc_to_deg: float,
    azi_to_deg: float,
    dls_limit_deg_per_30m: float,
) -> float:
    dogleg_deg = float(
        dogleg_angle_rad(inc_from_deg, azi_from_deg, inc_to_deg, azi_to_deg) * RAD2DEG
    )
    if dogleg_deg <= 1e-9:
        return 0.0
    return float(dogleg_deg / float(dls_limit_deg_per_30m) * 30.0)


def _validate_inc_limit(
    *,
    inc_deg: float,
    config: TrajectoryConfig,
    context: str,
) -> None:
    if float(inc_deg) <= float(config.max_inc_deg) + 1e-9:
        return
    raise PlanningError(
        "Многопластовая скважина: "
        f"{context} требует INC {float(inc_deg):.2f}°, что выше "
        f"ограничения max_inc_deg={float(config.max_inc_deg):.2f}°. "
        "Увеличьте горизонтальный зазор между уровнями, уменьшите ΔZ или "
        "измените допустимый max_inc_deg."
    )


def _direction_vector(inc_deg: float, azi_deg: float) -> np.ndarray:
    inc_rad = np.radians(float(inc_deg))
    azi_rad = np.radians(float(azi_deg))
    return np.array(
        [
            np.sin(inc_rad) * np.cos(azi_rad),
            np.sin(inc_rad) * np.sin(azi_rad),
            np.cos(inc_rad),
        ],
        dtype=float,
    )


def _xyz_direction_vector(inc_deg: float, azi_deg: float) -> np.ndarray:
    north, east, down = _direction_vector(inc_deg=inc_deg, azi_deg=azi_deg)
    return np.array([east, north, down], dtype=float)


def _max_feasible_delta_z_m(
    *,
    current: dict[str, float],
    target: Point3D,
    target_inc_deg: float,
    target_azi_deg: float,
    dls_limit_deg_per_30m: float,
) -> float:
    dx = float(target.x) - float(current["x"])
    dy = float(target.y) - float(current["y"])
    actual_dz = float(target.z) - float(current["z"])
    horizontal = float(np.hypot(dx, dy))
    gap = float(np.hypot(horizontal, actual_dz))
    if horizontal <= SMALL or gap <= SMALL:
        return 0.0
    sign = 1.0 if actual_dz >= 0.0 else -1.0

    def required_for(abs_dz: float) -> float:
        direct_inc, direct_azi = _direction_angles_from_delta(
            dx=dx,
            dy=dy,
            dz=sign * float(abs_dz),
        )
        return _dogleg_build_length_m(
            float(current["inc_deg"]),
            float(current["azi_deg"]),
            direct_inc,
            direct_azi,
            dls_limit_deg_per_30m,
        ) + _dogleg_build_length_m(
            direct_inc,
            direct_azi,
            target_inc_deg,
            target_azi_deg,
            dls_limit_deg_per_30m,
        )

    lo = 0.0
    hi = abs(actual_dz)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if required_for(mid) <= float(np.hypot(horizontal, mid)) + 1e-9:
            lo = mid
        else:
            hi = mid
    return float(lo)


def _transition_problem_text(transition: MultiHorizontalTransition) -> str:
    trim_each = float(max(transition.excess_m, 0.0) / 2.0)
    return (
        "Многопластовая скважина: переход "
        f"HORIZONTAL_BUILD{transition.level_from} между "
        f"{transition.level_from}_t3 и {transition.level_to}_t1 слишком короткий "
        "для заданного максимального ПИ. "
        f"Текущий зазор {transition.gap_m:.1f} м, требуется примерно "
        f"{transition.required_build_m:.1f} м, не хватает {transition.excess_m:.1f} м. "
        "Рекомендации: сократить соседние мини-горизонты примерно на "
        f"{trim_each:.1f} м каждый, чтобы увеличить расстояние между пластами, "
        "или уменьшить вертикальную разницу между уровнями примерно до "
        f"{transition.max_feasible_delta_z_m:.1f} м."
    )
