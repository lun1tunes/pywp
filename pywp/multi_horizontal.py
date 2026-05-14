from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.constants import RAD2DEG
from pywp.mcm import add_dls, dogleg_angle_rad
from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.planner_types import PlanningError

SMALL = 1e-9


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

        transition_rows = _linear_transition_rows(
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
            f"ограничение INC ({max_inc:.3f} > {float(config.max_inc_deg):.3f}°)."
        )

    dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)
    finite_dls = dls_values[np.isfinite(dls_values)]
    max_dls = float(np.max(finite_dls)) if len(finite_dls) else 0.0
    if max_dls > float(config.dls_build_max_deg_per_30m) + 1e-6:
        raise PlanningError(
            "Многопластовая скважина: построенный HORIZONTAL_BUILD превышает "
            f"лимит ПИ ({max_dls:.3f} > {float(config.dls_build_max_deg_per_30m):.3f} "
            "deg/30m)."
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


def _linear_transition_rows(
    *,
    current: dict[str, float],
    target: Point3D,
    target_inc_deg: float,
    target_azi_deg: float,
    segment_name: str,
    config: TrajectoryConfig,
) -> list[dict[str, object]]:
    gap = _distance_from_current(current, target)
    direct_inc, direct_azi = _direction_angles_from_current(current, target)
    dls_limit = float(config.dls_build_max_deg_per_30m)
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

    def angles(distance_m: float) -> tuple[float, float]:
        remaining_m = float(gap - distance_m)
        if build_in_m > SMALL and distance_m < build_in_m:
            return _interpolate_angles(
                float(current["inc_deg"]),
                float(current["azi_deg"]),
                direct_inc,
                direct_azi,
                float(distance_m / build_in_m),
            )
        if build_out_m > SMALL and remaining_m < build_out_m:
            return _interpolate_angles(
                direct_inc,
                direct_azi,
                target_inc_deg,
                target_azi_deg,
                float(1.0 - remaining_m / build_out_m),
            )
        return direct_inc, direct_azi

    return _linear_rows(
        current=current,
        target=target,
        segment_name=segment_name,
        md_step_m=float(config.md_step_m),
        angles_at_distance=angles,
    )


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
    return _linear_rows(
        current=current,
        target=target,
        segment_name=segment_name,
        md_step_m=float(config.md_step_m),
        angles_at_distance=lambda _distance_m: (float(inc_deg), float(azi_deg)),
    )


def _linear_rows(
    *,
    current: dict[str, float],
    target: Point3D,
    segment_name: str,
    md_step_m: float,
    angles_at_distance: Callable[[float], tuple[float, float]],
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
        inc_deg, azi_deg = angles_at_distance(float(distance))
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


def _interpolate_angles(
    inc_from_deg: float,
    azi_from_deg: float,
    inc_to_deg: float,
    azi_to_deg: float,
    fraction: float,
) -> tuple[float, float]:
    start = _direction_vector(inc_from_deg, azi_from_deg)
    end = _direction_vector(inc_to_deg, azi_to_deg)
    vector = _slerp_direction(start, end, float(np.clip(fraction, 0.0, 1.0)))
    return _angles_from_direction(vector)


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


def _slerp_direction(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    start = start / max(float(np.linalg.norm(start)), SMALL)
    end = end / max(float(np.linalg.norm(end)), SMALL)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    theta = float(np.arccos(dot))
    if theta <= 1e-12:
        return start
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) <= SMALL:
        return start
    return (
        np.sin((1.0 - fraction) * theta) / sin_theta * start
        + np.sin(fraction * theta) / sin_theta * end
    )


def _angles_from_direction(vector: np.ndarray) -> tuple[float, float]:
    north = float(vector[0])
    east = float(vector[1])
    down = float(vector[2])
    horizontal = float(np.hypot(north, east))
    inc_deg = float(np.degrees(np.arctan2(horizontal, down)))
    azi_deg = float(np.degrees(np.arctan2(east, north)) % 360.0)
    return inc_deg, azi_deg


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
