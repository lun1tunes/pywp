from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.constants import RAD2DEG
from pywp.mcm import add_dls, dogleg_angle_rad, minimum_curvature_increment
from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.planner_types import PlanningError
from pywp.segments import BuildSegment, HoldSegment
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


@dataclass(frozen=True)
class _ConstantDlsTransitionCandidate:
    stations: pd.DataFrame
    dls_deg_per_30m: float
    endpoint_miss_m: float
    max_dls_deg_per_30m: float
    max_inc_deg: float
    md_span_m: float


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
        post_t1_start_md_m=float(base_result.md_t1_m),
    )

    summary = dict(base_result.summary)
    final_target = target_pairs[-1][1]
    max_dls = float(np.nanmax(stations["DLS_deg_per_30m"].to_numpy(dtype=float)))
    horizontal_limit = _horizontal_dls_limit(config)
    md_total_m = float(stations["MD_m"].iloc[-1])
    md_postcheck_limit_m = float(config.max_total_md_postcheck_m)
    md_postcheck_excess_m = float(max(0.0, md_total_m - md_postcheck_limit_m))
    summary.update(
        {
            "trajectory_type": "Multi-horizontal",
            "multi_horizontal": "yes",
            "multi_horizontal_levels": int(len(target_pairs)),
            "multi_horizontal_transition_count": int(len(target_pairs) - 1),
            "dls_limit_horizontal_deg_per_30m": float(horizontal_limit),
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
    post_t1_start_md_m: float,
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

    horizontal_limit = _horizontal_dls_limit(config)
    post_t1_mask = (
        stations["MD_m"].to_numpy(dtype=float) > float(post_t1_start_md_m) + 1e-6
    )
    dls_values = stations.loc[post_t1_mask, "DLS_deg_per_30m"].to_numpy(dtype=float)
    finite_dls = dls_values[np.isfinite(dls_values)]
    max_dls = float(np.max(finite_dls)) if len(finite_dls) else 0.0
    if max_dls > horizontal_limit + 1e-6:
        raise PlanningError(
            "Многопластовая скважина: построенный HORIZONTAL_BUILD превышает "
            f"лимит ПИ ({dls_to_pi(max_dls):.3f} > "
            f"{dls_to_pi(horizontal_limit):.3f} deg/10m)."
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
    dls_limit = _horizontal_dls_limit(config)
    if dls_limit <= SMALL:
        raise PlanningError(
            "Многопластовая скважина: для HORIZONTAL_BUILD требуется положительный "
            "максимальный ПИ HORIZONTAL."
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
    dls_limit = _horizontal_dls_limit(config)
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
            try:
                xyz = _sample_cubic_bezier(
                    p0=p0,
                    p1=p1,
                    p2=p2,
                    p3=p3,
                    step_m=float(config.md_step_m),
                )
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
            max_md_span = max(
                float(gap) * MAX_TRANSITION_MD_MULTIPLIER,
                float(gap) + 500.0,
            )
            if md_span > max_md_span:
                continue
            dls_values = candidate["DLS_deg_per_30m"].to_numpy(dtype=float)
            finite_dls = _finite_values(dls_values)
            max_dls = float(np.max(finite_dls)) if len(finite_dls) else 0.0
            p90_dls = float(np.quantile(finite_dls, 0.90)) if len(finite_dls) else 0.0
            mean_dls = float(np.mean(finite_dls)) if len(finite_dls) else 0.0
            max_inc = _max_finite(candidate["INC_deg"].to_numpy(dtype=float))
            dls_excess = max(0.0, max_dls - dls_limit)
            inc_excess = max(0.0, max_inc - float(config.max_inc_deg))
            dls_peak_headroom = max(0.0, dls_limit - max_dls)
            dls_work_headroom = max(0.0, dls_limit - p90_dls)
            score = float(
                1_000_000.0 * dls_excess
                + 1_000_000.0 * inc_excess
                + 2.0 * dls_work_headroom
                + 0.25 * dls_peak_headroom
                - 0.05 * mean_dls
                + 0.0001 * md_span
            )
            if score < best_score:
                best = candidate
                best_score = score
                best_dls = max_dls
                best_inc = max_inc

    fallback_error: PlanningError | None = None
    if (
        best is None
        or best_dls > dls_limit + 1e-6
        or best_inc > float(config.max_inc_deg) + 1e-6
    ):
        try:
            constant_dls_candidate = _constant_dls_transition_candidate(
                current=current,
                target=target,
                target_inc_deg=target_inc_deg,
                target_azi_deg=target_azi_deg,
                segment_name=segment_name,
                config=config,
            )
        except PlanningError as exc:
            fallback_error = exc
        else:
            return [
                dict(row)
                for row in constant_dls_candidate.stations.iloc[1:].to_dict("records")
            ]

    if best is None:
        message = (
            "Многопластовая скважина: не удалось построить плавный "
            f"{segment_name} между уровнями без вырожденной геометрии. "
            "Увеличьте расстояние между горизонтальными участками или уменьшите ΔZ."
        )
        if fallback_error is not None:
            message += f" Constant-DLS fallback: {fallback_error}"
        raise PlanningError(message)
    if best_dls > dls_limit + 1e-6:
        message = (
            "Многопластовая скважина: плавный "
            f"{segment_name} требует ПИ {dls_to_pi(best_dls):.2f} deg/10m, что выше "
            f"лимита {dls_to_pi(dls_limit):.2f}. Увеличьте расстояние между уровнями, "
            "сократите соседние мини-горизонты или уменьшите ΔZ."
        )
        if fallback_error is not None:
            message += f" Constant-DLS fallback: {fallback_error}"
        raise PlanningError(message)
    if best_inc > float(config.max_inc_deg) + 1e-6:
        message = (
            "Многопластовая скважина: плавный "
            f"{segment_name} требует INC {best_inc:.2f}°, что выше "
            f"ограничения {float(config.max_inc_deg):.2f}°. "
            "Увеличьте горизонтальный зазор между уровнями или уменьшите ΔZ."
        )
        if fallback_error is not None:
            message += f" Constant-DLS fallback: {fallback_error}"
        raise PlanningError(message)
    return [dict(row) for row in best.iloc[1:].to_dict("records")]


def _constant_dls_transition_candidate(
    *,
    current: dict[str, float],
    target: Point3D,
    target_inc_deg: float,
    target_azi_deg: float,
    segment_name: str,
    config: TrajectoryConfig,
) -> _ConstantDlsTransitionCandidate:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - scipy is a runtime dependency.
        raise PlanningError("scipy недоступен для Constant-DLS fallback.") from exc

    dls_limit = _horizontal_dls_limit(config)
    if dls_limit <= SMALL:
        raise PlanningError("лимит HORIZONTAL ПИ должен быть положительным.")

    target_delta = np.array(
        [
            float(target.x) - float(current["x"]),
            float(target.y) - float(current["y"]),
            float(target.z) - float(current["z"]),
        ],
        dtype=float,
    )
    gap_m = float(np.linalg.norm(target_delta))
    if gap_m <= SMALL:
        raise PlanningError("точки перехода совпадают.")

    start_inc = float(current["inc_deg"])
    start_azi = float(current["azi_deg"]) % 360.0
    end_inc = float(target_inc_deg)
    end_azi = float(target_azi_deg) % 360.0
    max_inc = float(config.max_inc_deg)
    max_hold_m = float(max(gap_m * MAX_TRANSITION_MD_MULTIPLIER, gap_m + 500.0))
    dls_lower = float(max(min(dls_limit * 0.02, 0.03), 1e-4))
    scale_m = float(max(gap_m, 1.0))

    def residual(values: np.ndarray) -> np.ndarray:
        inc_mid, azi_mid, dls_value, hold_length = [float(item) for item in values]
        try:
            delta, _, _ = _constant_dls_transition_delta_xyz(
                start_inc_deg=start_inc,
                start_azi_deg=start_azi,
                mid_inc_deg=inc_mid,
                mid_azi_deg=azi_mid,
                end_inc_deg=end_inc,
                end_azi_deg=end_azi,
                dls_deg_per_30m=dls_value,
                hold_length_m=hold_length,
            )
        except ValueError:
            return np.full(3, 1e6, dtype=float)
        if not np.all(np.isfinite(delta)):
            return np.full(3, 1e6, dtype=float)
        return (delta - target_delta) / scale_m

    best: _ConstantDlsTransitionCandidate | None = None
    best_score = float("inf")
    for seed in _constant_dls_transition_seeds(
        current=current,
        target=target,
        target_inc_deg=end_inc,
        target_azi_deg=end_azi,
        dls_limit_deg_per_30m=dls_limit,
        max_inc_deg=max_inc,
        max_hold_m=max_hold_m,
    ):
        result = least_squares(
            residual,
            np.asarray(seed, dtype=float),
            bounds=(
                np.array([0.0, 0.0, dls_lower, 0.0], dtype=float),
                np.array([max_inc, 360.0, dls_limit, max_hold_m], dtype=float),
            ),
            max_nfev=160,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        if not result.success and float(result.cost) > 1e-14:
            continue
        inc_mid, azi_mid, dls_value, hold_length = [
            float(item) for item in result.x
        ]
        try:
            stations = _build_constant_dls_transition_stations(
                current=current,
                mid_inc_deg=inc_mid,
                mid_azi_deg=azi_mid,
                target=target,
                target_inc_deg=end_inc,
                target_azi_deg=end_azi,
                dls_deg_per_30m=dls_value,
                hold_length_m=hold_length,
                segment_name=segment_name,
                config=config,
            )
        except ValueError:
            continue
        endpoint = stations.iloc[-1]
        endpoint_miss = float(
            np.linalg.norm(
                np.array(
                    [
                        float(endpoint["X_m"]) - float(target.x),
                        float(endpoint["Y_m"]) - float(target.y),
                        float(endpoint["Z_m"]) - float(target.z),
                    ],
                    dtype=float,
                )
            )
        )
        dls_values = _finite_values(stations["DLS_deg_per_30m"].to_numpy(dtype=float))
        max_dls = float(np.max(dls_values)) if len(dls_values) else 0.0
        inc_values = _finite_values(stations["INC_deg"].to_numpy(dtype=float))
        actual_max_inc = float(np.max(inc_values)) if len(inc_values) else 0.0
        md_span = float(stations["MD_m"].iloc[-1] - stations["MD_m"].iloc[0])
        if endpoint_miss > 1e-4:
            continue
        if max_dls > dls_limit + 1e-6:
            continue
        if actual_max_inc > max_inc + 1e-6:
            continue
        candidate = _ConstantDlsTransitionCandidate(
            stations=stations,
            dls_deg_per_30m=dls_value,
            endpoint_miss_m=endpoint_miss,
            max_dls_deg_per_30m=max_dls,
            max_inc_deg=actual_max_inc,
            md_span_m=md_span,
        )
        score = float(
            endpoint_miss * 1_000_000.0
            + 0.001 * md_span
            - 0.01 * dls_value
        )
        if score < best_score:
            best = candidate
            best_score = score

    if best is None:
        raise PlanningError(
            "не удалось подобрать Constant-DLS переход с ПИ не выше "
            f"{dls_to_pi(dls_limit):.2f} deg/10m."
        )
    return best


def _constant_dls_transition_seeds(
    *,
    current: dict[str, float],
    target: Point3D,
    target_inc_deg: float,
    target_azi_deg: float,
    dls_limit_deg_per_30m: float,
    max_inc_deg: float,
    max_hold_m: float,
) -> tuple[tuple[float, float, float, float], ...]:
    direct_inc, direct_azi = _direction_angles_from_current(current, target)
    start_inc = float(current["inc_deg"])
    start_azi = float(current["azi_deg"]) % 360.0
    end_inc = float(target_inc_deg)
    end_azi = float(target_azi_deg) % 360.0
    avg_inc = float(np.clip((start_inc + end_inc) * 0.5, 0.0, max_inc_deg))
    inc_candidates = _dedupe_float_candidates(
        np.clip(
            np.array(
                [
                    direct_inc,
                    avg_inc,
                    direct_inc - 5.0,
                    direct_inc + 5.0,
                ],
                dtype=float,
            ),
            0.0,
            max_inc_deg,
        )
    )
    avg_azi = _interpolate_azimuth_deg(start_azi, end_azi, 0.5)
    azi_candidates = _dedupe_float_candidates(
        np.mod(
            np.array(
                [
                    direct_azi,
                    avg_azi,
                    start_azi,
                    end_azi,
                ],
                dtype=float,
            ),
            360.0,
        )
    )
    dls_candidates = _dedupe_float_candidates(
        np.clip(
            np.array(
                [
                    dls_limit_deg_per_30m,
                    dls_limit_deg_per_30m * 0.75,
                    dls_limit_deg_per_30m * 0.50,
                ],
                dtype=float,
            ),
            1e-4,
            dls_limit_deg_per_30m,
        )
    )
    seeds: list[tuple[float, float, float, float]] = []
    gap_m = _distance_from_current(current, target)
    for inc_mid in inc_candidates:
        for azi_mid in azi_candidates:
            for dls_value in dls_candidates:
                try:
                    _, build1_m, build2_m = _constant_dls_transition_delta_xyz(
                        start_inc_deg=start_inc,
                        start_azi_deg=start_azi,
                        mid_inc_deg=inc_mid,
                        mid_azi_deg=azi_mid,
                        end_inc_deg=end_inc,
                        end_azi_deg=end_azi,
                        dls_deg_per_30m=dls_value,
                        hold_length_m=0.0,
                    )
                except ValueError:
                    continue
                if not np.isfinite(build1_m) or not np.isfinite(build2_m):
                    continue
                hold_seed = float(np.clip(gap_m - build1_m - build2_m, 0.0, max_hold_m))
                seeds.append((inc_mid, azi_mid, dls_value, hold_seed))
    return tuple(seeds)


def _build_constant_dls_transition_stations(
    *,
    current: dict[str, float],
    mid_inc_deg: float,
    mid_azi_deg: float,
    target: Point3D,
    target_inc_deg: float,
    target_azi_deg: float,
    dls_deg_per_30m: float,
    hold_length_m: float,
    segment_name: str,
    config: TrajectoryConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = [
        pd.DataFrame(
            [
                {
                    "MD_m": float(current["md_m"]),
                    "INC_deg": float(current["inc_deg"]),
                    "AZI_deg": float(current["azi_deg"]) % 360.0,
                    "segment": str(segment_name),
                }
            ]
        )
    ]
    md_start = float(current["md_m"])
    start_inc = float(current["inc_deg"])
    start_azi = float(current["azi_deg"]) % 360.0
    for frame in (
        BuildSegment(
            inc_from_deg=start_inc,
            inc_to_deg=float(mid_inc_deg),
            dls_deg_per_30m=float(dls_deg_per_30m),
            azi_deg=start_azi,
            azi_to_deg=float(mid_azi_deg),
            name=segment_name,
            interpolation_method=str(config.interpolation_method),
        ).generate(md_start, float(config.md_step_m)),
        HoldSegment(
            length_m=float(hold_length_m),
            inc_deg=float(mid_inc_deg),
            azi_deg=float(mid_azi_deg),
            name=segment_name,
        ).generate(
            md_start
            + _dogleg_build_length_m(
                start_inc,
                start_azi,
                float(mid_inc_deg),
                float(mid_azi_deg),
                float(dls_deg_per_30m),
            ),
            float(config.md_step_m),
        ),
    ):
        if len(frame) > 1:
            frames.append(frame.iloc[1:].copy())
    md_after_hold = float(frames[-1]["MD_m"].iloc[-1])
    build_out = BuildSegment(
        inc_from_deg=float(mid_inc_deg),
        inc_to_deg=float(target_inc_deg),
        dls_deg_per_30m=float(dls_deg_per_30m),
        azi_deg=float(mid_azi_deg),
        azi_to_deg=float(target_azi_deg),
        name=segment_name,
        interpolation_method=str(config.interpolation_method),
    ).generate(md_after_hold, float(config.md_step_m))
    if len(build_out) > 1:
        frames.append(build_out.iloc[1:].copy())
    stations = pd.concat(frames, ignore_index=True)
    return _attach_min_curve_positions(
        stations=stations,
        start_xyz=np.array(
            [float(current["x"]), float(current["y"]), float(current["z"])],
            dtype=float,
        ),
    )


def _attach_min_curve_positions(
    *,
    stations: pd.DataFrame,
    start_xyz: np.ndarray,
) -> pd.DataFrame:
    out = stations.copy().reset_index(drop=True)
    xs = [float(start_xyz[0])]
    ys = [float(start_xyz[1])]
    zs = [float(start_xyz[2])]
    for idx in range(1, len(out)):
        dn, de, dz = minimum_curvature_increment(
            md1_m=float(out.loc[idx - 1, "MD_m"]),
            inc1_deg=float(out.loc[idx - 1, "INC_deg"]),
            azi1_deg=float(out.loc[idx - 1, "AZI_deg"]),
            md2_m=float(out.loc[idx, "MD_m"]),
            inc2_deg=float(out.loc[idx, "INC_deg"]),
            azi2_deg=float(out.loc[idx, "AZI_deg"]),
        )
        xs.append(xs[-1] + float(de))
        ys.append(ys[-1] + float(dn))
        zs.append(zs[-1] + float(dz))
    out["N_m"] = ys
    out["E_m"] = xs
    out["TVD_m"] = zs
    out["X_m"] = xs
    out["Y_m"] = ys
    out["Z_m"] = zs
    return add_dls(out)


def _constant_dls_transition_delta_xyz(
    *,
    start_inc_deg: float,
    start_azi_deg: float,
    mid_inc_deg: float,
    mid_azi_deg: float,
    end_inc_deg: float,
    end_azi_deg: float,
    dls_deg_per_30m: float,
    hold_length_m: float,
) -> tuple[np.ndarray, float, float]:
    build1_m = _dogleg_build_length_m(
        start_inc_deg,
        start_azi_deg,
        mid_inc_deg,
        mid_azi_deg,
        dls_deg_per_30m,
    )
    build2_m = _dogleg_build_length_m(
        mid_inc_deg,
        mid_azi_deg,
        end_inc_deg,
        end_azi_deg,
        dls_deg_per_30m,
    )
    delta = np.zeros(3, dtype=float)
    if build1_m > SMALL:
        dn, de, dz = minimum_curvature_increment(
            0.0,
            start_inc_deg,
            start_azi_deg,
            build1_m,
            mid_inc_deg,
            mid_azi_deg,
        )
        delta += np.array([de, dn, dz], dtype=float)
    if float(hold_length_m) > SMALL:
        delta += _xyz_direction_vector(inc_deg=mid_inc_deg, azi_deg=mid_azi_deg) * float(
            hold_length_m
        )
    if build2_m > SMALL:
        dn, de, dz = minimum_curvature_increment(
            0.0,
            mid_inc_deg,
            mid_azi_deg,
            build2_m,
            end_inc_deg,
            end_azi_deg,
        )
        delta += np.array([de, dn, dz], dtype=float)
    return delta, float(build1_m), float(build2_m)


def _candidate_control_lengths(*, gap_m: float, min_control_m: float) -> tuple[float, ...]:
    gap = float(gap_m)
    values = {
        float(min_control_m),
        *(
            gap * scale
            for scale in (
                0.08,
                0.16,
                0.25,
                0.30,
                0.35,
                0.55,
                0.70,
                0.85,
                1.20,
            )
        ),
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
    finite = _finite_values(values)
    return float(np.max(finite)) if len(finite) else 0.0


def _finite_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    return finite[np.isfinite(finite)]


def _dedupe_float_candidates(values: np.ndarray, *, decimals: int = 8) -> tuple[float, ...]:
    result: list[float] = []
    seen: set[float] = set()
    for value in np.asarray(values, dtype=float):
        if not np.isfinite(value):
            continue
        key = round(float(value), int(decimals))
        if key in seen:
            continue
        seen.add(key)
        result.append(float(value))
    return tuple(result)


def _interpolate_azimuth_deg(start_deg: float, end_deg: float, fraction: float) -> float:
    start = float(start_deg) % 360.0
    delta = ((float(end_deg) - start + 540.0) % 360.0) - 180.0
    return float((start + delta * float(fraction)) % 360.0)


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


def _horizontal_dls_limit(config: TrajectoryConfig) -> float:
    return float(max(float(config.dls_horizontal_max_deg_per_30m), 0.0))


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
