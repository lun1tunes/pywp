from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from pywp.anticollision import analyze_anti_collision, build_anti_collision_well
from pywp.mcm import add_dls, wrap_azimuth_deg
from pywp.pydantic_base import FrozenArbitraryModel
from pywp.reference_trajectories import ImportedTrajectoryWell
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    PlanningUncertaintyModel,
)

HORIZONTAL_INC_THRESHOLD_DEG = 80.0
HORIZONTAL_MIN_INTERVAL_M = 100.0
HORIZONTAL_END_TOLERANCE_M = 60.0
KOP_INC_THRESHOLD_DEG = 1.0
ACTUAL_FUND_RESAMPLE_STEP_M = 10.0
HOLD_INC_TOLERANCE_DEG = 3.0
HOLD_MIN_INTERVAL_M = 90.0
HOLD_MAX_INC_DEG = 78.0
HOLD_MIN_INC_DEG = 8.0
HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M = 1.5
ROBUST_DLS_WINDOW_M = 60.0
MAX_REASONABLE_ACTUAL_DLS_DEG_PER_30M = 6.0
MIN_CUSTOM_ACTUAL_FUND_SCALE = 0.35

CALIBRATION_STATUS_READY = "ready"
CALIBRATION_STATUS_NO_CHANGE = "no_change"
CALIBRATION_STATUS_INSUFFICIENT = "insufficient"
CALIBRATION_STATUS_TOO_AGGRESSIVE = "too_aggressive"


class ActualFundWellMetrics(FrozenArbitraryModel):
    name: str
    family_name: str
    pad_group: str
    is_horizontal: bool
    md_total_m: float
    tvd_end_m: float
    lateral_departure_m: float
    kop_md_m: float | None
    kop_tvd_m: float | None
    horizontal_entry_md_m: float | None
    horizontal_length_m: float
    hold_inc_deg: float | None
    hold_azi_deg: float | None
    hold_length_m: float
    max_inc_deg: float
    max_dls_deg_per_30m: float | None
    max_build_dls_before_hold_deg_per_30m: float | None
    is_analysis_eligible: bool
    analysis_exclusion_reason: str | None = None


class ActualFundPadSummary(FrozenArbitraryModel):
    pad_group: str
    well_count: int
    horizontal_well_count: int
    median_kop_md_m: float | None
    median_hold_inc_deg: float | None
    median_hold_length_m: float | None
    max_pi_deg_per_30m: float | None
    median_horizontal_length_m: float | None


class ActualFundCalibrationResult(FrozenArbitraryModel):
    status: str
    base_preset: str
    scale_factor: float | None
    custom_model: PlanningUncertaintyModel | None = None
    actual_well_count: int
    horizontal_well_count: int
    analyzed_pair_count: int
    skipped_same_family_pair_count: int
    overlapping_pair_count_before: int
    overlapping_pair_count_after: int | None = None
    worst_separation_factor_before: float | None = None
    worst_separation_factor_after: float | None = None
    note: str = ""


def actual_well_family_name(name: object) -> str:
    label = str(name or "").strip()
    match = re.match(r"^(\d+)", label)
    if match:
        return match.group(1)
    head = re.split(r"[_\-/\s]+", label, maxsplit=1)[0].strip()
    return head or label


def actual_well_pad_group(name: object) -> str:
    family = actual_well_family_name(name)
    digits_match = re.match(r"^(\d+)", family)
    if not digits_match:
        return "—"
    digits = digits_match.group(1)
    if len(digits) > 2:
        return digits[:-2]
    return digits


def actual_well_is_horizontal(stations: pd.DataFrame) -> bool:
    survey = _reconstruct_actual_survey(stations)
    interval = _terminal_horizontal_interval(survey)
    max_inc = float(np.nanmax(survey["INC_deg"].to_numpy(dtype=float)))
    return bool(max_inc >= HORIZONTAL_INC_THRESHOLD_DEG and interval[2] >= HORIZONTAL_MIN_INTERVAL_M)


def build_actual_fund_well_metrics(
    actual_wells: Iterable[ImportedTrajectoryWell],
) -> tuple[ActualFundWellMetrics, ...]:
    metrics: list[ActualFundWellMetrics] = []
    for well in actual_wells:
        stations = _reconstruct_actual_survey(well.stations)
        source_md_values = well.stations["MD_m"].to_numpy(dtype=float)
        source_z_values = well.stations["Z_m"].to_numpy(dtype=float)
        md_values = stations["MD_m"].to_numpy(dtype=float)
        z_values = stations["Z_m"].to_numpy(dtype=float)
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        inc_values = stations["INC_deg"].to_numpy(dtype=float)
        kop_md = _first_kop_md_from_geometry(well.stations, threshold_deg=KOP_INC_THRESHOLD_DEG)
        horizontal_entry_md, _, horizontal_length = _terminal_horizontal_interval(stations)
        hold_start_md, hold_end_md, hold_inc, hold_azi = _detect_hold_interval(
            stations=stations,
            kop_md_m=kop_md,
            horizontal_entry_md_m=horizontal_entry_md,
        )
        hold_length = (
            0.0
            if hold_start_md is None or hold_end_md is None
            else float(max(hold_end_md - hold_start_md, 0.0))
        )
        build_limit_md = (
            float(hold_start_md)
            if hold_start_md is not None
            else float(horizontal_entry_md)
            if horizontal_entry_md is not None
            else float(md_values[-1])
        )
        build_mask = md_values <= build_limit_md + 1e-9
        is_horizontal = actual_well_is_horizontal(well.stations)
        max_dls = _robust_max_dls(stations)
        max_build_dls = _robust_max_dls(stations.loc[build_mask].reset_index(drop=True))
        exclusion_reason = _analysis_exclusion_reason(
            is_horizontal=is_horizontal,
            kop_md_m=kop_md,
            horizontal_entry_md_m=horizontal_entry_md,
            horizontal_length_m=float(horizontal_length),
            hold_inc_deg=hold_inc,
            hold_length_m=float(hold_length),
            max_dls_deg_per_30m=max_dls,
            max_build_dls_before_hold_deg_per_30m=max_build_dls,
        )
        metrics.append(
            ActualFundWellMetrics(
                name=str(well.name),
                family_name=actual_well_family_name(well.name),
                pad_group=actual_well_pad_group(well.name),
                is_horizontal=is_horizontal,
                md_total_m=float(md_values[-1]),
                tvd_end_m=float(z_values[-1]),
                lateral_departure_m=float(
                    np.hypot(x_values[-1] - x_values[0], y_values[-1] - y_values[0])
                ),
                kop_md_m=kop_md,
                kop_tvd_m=(
                    None
                    if kop_md is None
                    else _interp_1d(source_md_values, source_z_values, float(kop_md))
                ),
                horizontal_entry_md_m=horizontal_entry_md,
                horizontal_length_m=float(horizontal_length),
                hold_inc_deg=hold_inc,
                hold_azi_deg=hold_azi,
                hold_length_m=float(hold_length),
                max_inc_deg=float(np.nanmax(inc_values)),
                max_dls_deg_per_30m=max_dls,
                max_build_dls_before_hold_deg_per_30m=max_build_dls,
                is_analysis_eligible=exclusion_reason is None,
                analysis_exclusion_reason=exclusion_reason,
            )
        )
    return tuple(metrics)


def actual_fund_metrics_rows(
    metrics: Iterable[ActualFundWellMetrics],
) -> list[dict[str, object]]:
    return [
        {
            "Куст": item.pad_group,
            "Скважина": item.name,
            "Семейство": item.family_name,
            "Горизонтальная": "Да" if item.is_horizontal else "Нет",
            "В анализе": "Да" if item.is_analysis_eligible else "Нет",
            "Причина исключения": item.analysis_exclusion_reason or "—",
            "MD, м": item.md_total_m,
            "KOP MD, м": item.kop_md_m,
            "KOP TVD, м": item.kop_tvd_m,
            "Вход в горизонталь, MD": item.horizontal_entry_md_m,
            "Горизонталь, м": item.horizontal_length_m,
            "Зенит HOLD, deg": item.hold_inc_deg,
            "Азимут HOLD, deg": item.hold_azi_deg,
            "HOLD, м": item.hold_length_m,
            "Макс INC, deg": item.max_inc_deg,
            "Макс ПИ, deg/30м": item.max_dls_deg_per_30m,
            "Макс ПИ до HOLD, deg/30м": item.max_build_dls_before_hold_deg_per_30m,
        }
        for item in metrics
    ]


def actual_fund_pad_rows(
    metrics: Iterable[ActualFundWellMetrics],
) -> list[dict[str, object]]:
    summaries = summarize_actual_fund_by_pad(metrics)
    return [
        {
            "Куст": item.pad_group,
            "Скважин": item.well_count,
            "Горизонтальных": item.horizontal_well_count,
            "Медианный KOP MD, м": item.median_kop_md_m,
            "Медианный HOLD INC, deg": item.median_hold_inc_deg,
            "Медианный HOLD, м": item.median_hold_length_m,
            "Макс ПИ, deg/30м": item.max_pi_deg_per_30m,
            "Медианная горизонталь, м": item.median_horizontal_length_m,
        }
        for item in summaries
    ]


def summarize_actual_fund_by_pad(
    metrics: Iterable[ActualFundWellMetrics],
) -> tuple[ActualFundPadSummary, ...]:
    grouped: dict[str, list[ActualFundWellMetrics]] = defaultdict(list)
    for item in metrics:
        if bool(item.is_analysis_eligible):
            grouped[str(item.pad_group)].append(item)
    summaries: list[ActualFundPadSummary] = []
    for pad_group in sorted(grouped):
        items = grouped[pad_group]
        horizontal_items = [item for item in items if bool(item.is_horizontal)]
        summaries.append(
            ActualFundPadSummary(
                pad_group=pad_group,
                well_count=len(items),
                horizontal_well_count=len(horizontal_items),
                median_kop_md_m=_median_or_none(item.kop_md_m for item in horizontal_items),
                median_hold_inc_deg=_median_or_none(item.hold_inc_deg for item in horizontal_items),
                median_hold_length_m=_median_or_none(item.hold_length_m for item in horizontal_items),
                max_pi_deg_per_30m=_max_or_none(item.max_dls_deg_per_30m for item in horizontal_items),
                median_horizontal_length_m=_median_or_none(
                    item.horizontal_length_m for item in horizontal_items
                ),
            )
        )
    return tuple(summaries)


def calibrate_uncertainty_from_actual_fund(
    *,
    actual_wells: Iterable[ImportedTrajectoryWell],
    base_model: PlanningUncertaintyModel,
    base_preset: str = DEFAULT_UNCERTAINTY_PRESET,
) -> ActualFundCalibrationResult:
    actual_well_list = list(actual_wells)
    reconstructed_by_name = {
        str(well.name): _reconstruct_actual_survey(well.stations)
        for well in actual_well_list
    }
    metrics = build_actual_fund_well_metrics(actual_well_list)
    horizontal_names = {
        item.name for item in metrics if bool(item.is_analysis_eligible)
    }
    eligible_wells = [well for well in actual_well_list if str(well.name) in horizontal_names]
    if len(eligible_wells) < 2:
        return ActualFundCalibrationResult(
            status=CALIBRATION_STATUS_INSUFFICIENT,
            base_preset=str(base_preset),
            scale_factor=None,
            actual_well_count=len(actual_well_list),
            horizontal_well_count=len(eligible_wells),
            analyzed_pair_count=0,
            skipped_same_family_pair_count=0,
            overlapping_pair_count_before=0,
            note=(
                "Для калибровки нужны минимум две фактические горизонтальные "
                "скважины без аномалий анализа."
            ),
        )

    family_by_name = {
        str(well.name): actual_well_family_name(well.name)
        for well in eligible_wells
    }
    pair_filter = lambda left, right: family_by_name[str(left.name)] != family_by_name[str(right.name)]
    skipped_same_family_pairs = sum(
        1
        for left_index in range(len(eligible_wells))
        for right_index in range(left_index + 1, len(eligible_wells))
        if family_by_name[str(eligible_wells[left_index].name)]
        == family_by_name[str(eligible_wells[right_index].name)]
    )
    analysis_before = analyze_anti_collision(
        [
            build_anti_collision_well(
                name=well.name,
                color="#6B7280",
                stations=reconstructed_by_name[str(well.name)],
                surface=well.surface,
                t1=None,
                t3=None,
                azimuth_deg=float(well.azimuth_deg),
                md_t1_m=None,
                md_t3_m=None,
                model=base_model,
                include_display_geometry=False,
                well_kind="actual",
                is_reference_only=False,
            )
            for well in eligible_wells
        ],
        build_overlap_geometry=False,
        pair_filter=pair_filter,
    )
    worst_sf_before = analysis_before.worst_separation_factor
    if worst_sf_before is None or float(worst_sf_before) >= 0.999:
        return ActualFundCalibrationResult(
            status=CALIBRATION_STATUS_NO_CHANGE,
            base_preset=str(base_preset),
            scale_factor=1.0,
            actual_well_count=len(actual_well_list),
            horizontal_well_count=len(eligible_wells),
            analyzed_pair_count=int(analysis_before.pair_count),
            skipped_same_family_pair_count=int(skipped_same_family_pairs),
            overlapping_pair_count_before=int(analysis_before.overlapping_pair_count),
            overlapping_pair_count_after=int(analysis_before.overlapping_pair_count),
            worst_separation_factor_before=worst_sf_before,
            worst_separation_factor_after=worst_sf_before,
            note=(
                "Текущая модель уже не даёт overlap по фактическому горизонтальному фонду. "
                "Дополнительная пользовательская калибровка не требуется."
            ),
        )

    scale_factor = float(max(0.0, min(1.0, 0.98 * float(worst_sf_before))))
    if scale_factor < MIN_CUSTOM_ACTUAL_FUND_SCALE:
        return ActualFundCalibrationResult(
            status=CALIBRATION_STATUS_TOO_AGGRESSIVE,
            base_preset=str(base_preset),
            scale_factor=scale_factor,
            actual_well_count=len(actual_well_list),
            horizontal_well_count=len(eligible_wells),
            analyzed_pair_count=int(analysis_before.pair_count),
            skipped_same_family_pair_count=int(skipped_same_family_pairs),
            overlapping_pair_count_before=int(analysis_before.overlapping_pair_count),
            worst_separation_factor_before=worst_sf_before,
            note=(
                "Для снятия overlap по фактическому фонду потребовалось бы слишком "
                "сильно уменьшить конусы одной глобальной шкалой. Пользовательская "
                "модель не создана автоматически."
            ),
        )

    custom_model = base_model.__class__(
        sigma_inc_deg=float(base_model.sigma_inc_deg) * scale_factor,
        sigma_azi_deg=float(base_model.sigma_azi_deg) * scale_factor,
        sigma_lateral_drift_m_per_1000m=float(base_model.sigma_lateral_drift_m_per_1000m)
        * scale_factor,
        confidence_scale=float(base_model.confidence_scale),
        sample_step_m=float(base_model.sample_step_m),
        max_display_ellipses=int(base_model.max_display_ellipses),
        ellipse_points=int(base_model.ellipse_points),
        min_display_radius_m=float(base_model.min_display_radius_m),
        near_vertical_isotropic_threshold_deg=float(
            base_model.near_vertical_isotropic_threshold_deg
        ),
        directional_refine_threshold_deg=float(
            base_model.directional_refine_threshold_deg
        ),
        min_refined_step_m=float(base_model.min_refined_step_m),
    )
    analysis_after = analyze_anti_collision(
        [
            build_anti_collision_well(
                name=well.name,
                color="#6B7280",
                stations=reconstructed_by_name[str(well.name)],
                surface=well.surface,
                t1=None,
                t3=None,
                azimuth_deg=float(well.azimuth_deg),
                md_t1_m=None,
                md_t3_m=None,
                model=custom_model,
                include_display_geometry=False,
                well_kind="actual",
                is_reference_only=False,
            )
            for well in eligible_wells
        ],
        build_overlap_geometry=False,
        pair_filter=pair_filter,
    )
    return ActualFundCalibrationResult(
        status=CALIBRATION_STATUS_READY,
        base_preset=str(base_preset),
        scale_factor=scale_factor,
        custom_model=custom_model,
        actual_well_count=len(actual_well_list),
        horizontal_well_count=len(eligible_wells),
        analyzed_pair_count=int(analysis_before.pair_count),
        skipped_same_family_pair_count=int(skipped_same_family_pairs),
        overlapping_pair_count_before=int(analysis_before.overlapping_pair_count),
        overlapping_pair_count_after=int(analysis_after.overlapping_pair_count),
        worst_separation_factor_before=worst_sf_before,
        worst_separation_factor_after=analysis_after.worst_separation_factor,
        note=(
            "Пользовательская модель построена как единая эмпирическая шкала "
            "относительно базового пресета. Это planning-level field fit по "
            "фактическому фонду, а не формальная ISCWSA toolcode calibration."
        ),
    )


def _terminal_horizontal_interval(stations: pd.DataFrame) -> tuple[float | None, float | None, float]:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    if len(md_values) == 0:
        return None, None, 0.0
    mask = inc_values >= HORIZONTAL_INC_THRESHOLD_DEG
    if not np.any(mask):
        return None, None, 0.0
    candidates = [
        (start_md, end_md, length_m)
        for start_md, end_md, length_m in _mask_intervals(md_values, mask)
        if end_md >= float(md_values[-1]) - HORIZONTAL_END_TOLERANCE_M
    ]
    if not candidates:
        return None, None, 0.0
    best_start, best_end, best_length = max(candidates, key=lambda item: (item[2], item[1]))
    return float(best_start), float(best_end), float(best_length)


def _first_threshold_crossing_md(
    *,
    md_values: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float | None:
    if len(md_values) == 0:
        return None
    if float(values[0]) >= threshold:
        return float(md_values[0])
    for index in range(1, len(md_values)):
        left_value = float(values[index - 1])
        right_value = float(values[index])
        if left_value < threshold <= right_value:
            delta = float(right_value - left_value)
            if abs(delta) <= 1e-9:
                return float(md_values[index])
            alpha = float((threshold - left_value) / delta)
            return float(md_values[index - 1] + alpha * (md_values[index] - md_values[index - 1]))
    return None


def _first_kop_md_from_geometry(stations: pd.DataFrame, *, threshold_deg: float) -> float | None:
    source = stations.sort_values("MD_m").reset_index(drop=True)
    md_values = source["MD_m"].to_numpy(dtype=float)
    x_values = source["X_m"].to_numpy(dtype=float)
    y_values = source["Y_m"].to_numpy(dtype=float)
    z_values = source["Z_m"].to_numpy(dtype=float)
    if len(md_values) < 2:
        return None
    dx = np.diff(x_values)
    dy = np.diff(y_values)
    dz = np.diff(z_values)
    lengths = np.sqrt(dx * dx + dy * dy + dz * dz)
    valid = lengths > 1e-9
    if not np.any(valid):
        return None
    inc_deg = np.degrees(np.arctan2(np.hypot(dx, dy), dz))
    for index, is_valid in enumerate(valid.tolist()):
        if is_valid and float(inc_deg[index]) >= float(threshold_deg):
            return float(md_values[index])
    return None


def _interp_1d(md_values: np.ndarray, values: np.ndarray, md_m: float) -> float:
    return float(np.interp(float(md_m), md_values, values))


def _circular_median_deg(values_deg: np.ndarray) -> float | None:
    if len(values_deg) == 0:
        return None
    radians = np.deg2rad(values_deg % 360.0)
    mean_angle = float(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float(np.rad2deg(mean_angle) % 360.0)


def _median_or_none(values: Iterable[float | None]) -> float | None:
    finite = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if len(finite) == 0:
        return None
    return float(np.median(finite))


def _max_or_none(values: Iterable[float | None]) -> float | None:
    finite = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if len(finite) == 0:
        return None
    return float(np.max(finite))


def _analysis_exclusion_reason(
    *,
    is_horizontal: bool,
    kop_md_m: float | None,
    horizontal_entry_md_m: float | None,
    horizontal_length_m: float,
    hold_inc_deg: float | None,
    hold_length_m: float,
    max_dls_deg_per_30m: float | None,
    max_build_dls_before_hold_deg_per_30m: float | None,
) -> str | None:
    if not bool(is_horizontal):
        return "Не горизонтальная"
    if kop_md_m is None:
        return "Не удалось определить KOP"
    if horizontal_entry_md_m is None or float(horizontal_length_m) < HORIZONTAL_MIN_INTERVAL_M:
        return "Не удалось выделить терминальный горизонтальный участок"
    if hold_inc_deg is None or float(hold_length_m) < HOLD_MIN_INTERVAL_M:
        return "Не удалось устойчиво выделить HOLD"
    if (
        max_dls_deg_per_30m is not None
        and float(max_dls_deg_per_30m) > MAX_REASONABLE_ACTUAL_DLS_DEG_PER_30M
    ):
        return "Аномально высокий устойчивый ПИ по стволу"
    if (
        max_build_dls_before_hold_deg_per_30m is not None
        and float(max_build_dls_before_hold_deg_per_30m) > MAX_REASONABLE_ACTUAL_DLS_DEG_PER_30M
    ):
        return "Аномально высокий устойчивый ПИ до HOLD"
    return None


def _reconstruct_actual_survey(
    stations: pd.DataFrame,
    *,
    resample_step_m: float | None = ACTUAL_FUND_RESAMPLE_STEP_M,
) -> pd.DataFrame:
    source = stations.sort_values("MD_m").reset_index(drop=True).copy()
    md_values = source["MD_m"].to_numpy(dtype=float)
    x_values = source["X_m"].to_numpy(dtype=float)
    y_values = source["Y_m"].to_numpy(dtype=float)
    z_values = source["Z_m"].to_numpy(dtype=float)
    if len(md_values) < 2:
        raise ValueError("actual-fund analysis requires at least two stations.")
    if not (
        np.isfinite(md_values).all()
        and np.isfinite(x_values).all()
        and np.isfinite(y_values).all()
        and np.isfinite(z_values).all()
    ):
        raise ValueError("actual-fund stations require finite MD/X/Y/Z values.")
    if np.any(np.diff(md_values) <= 0.0):
        raise ValueError("actual-fund stations require strictly increasing MD.")

    positions = np.column_stack([x_values, y_values, z_values]).astype(float)
    station_directions = _station_direction_vectors(md_values, positions)
    md_grid = (
        _regular_md_grid(md_values, float(resample_step_m))
        if resample_step_m is not None
        else np.asarray(md_values, dtype=float)
    )
    x_grid = np.interp(md_grid, md_values, x_values)
    y_grid = np.interp(md_grid, md_values, y_values)
    z_grid = np.interp(md_grid, md_values, z_values)
    direction_grid = np.column_stack(
        [
            np.interp(md_grid, md_values, station_directions[:, axis_index])
            for axis_index in range(3)
        ]
    )
    direction_grid = _normalize_vectors(direction_grid)

    horizontal = np.hypot(direction_grid[:, 0], direction_grid[:, 1])
    inc_deg = np.degrees(np.arctan2(horizontal, direction_grid[:, 2]))
    azi_deg = wrap_azimuth_deg(np.degrees(np.arctan2(direction_grid[:, 0], direction_grid[:, 1])))

    rebuilt = pd.DataFrame(
        {
            "MD_m": md_grid,
            "INC_deg": inc_deg,
            "AZI_deg": azi_deg,
            "X_m": x_grid,
            "Y_m": y_grid,
            "Z_m": z_grid,
            "segment": ["IMPORTED"] * len(md_grid),
        }
    )
    return add_dls(rebuilt)


def _station_direction_vectors(md_values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    delta_md = np.diff(md_values)
    segment_vectors = np.diff(positions, axis=0)
    segment_norms = np.linalg.norm(segment_vectors, axis=1)
    segment_directions = np.zeros_like(segment_vectors, dtype=float)
    valid = segment_norms > 1e-9
    if np.any(valid):
        segment_directions[valid] = segment_vectors[valid] / segment_norms[valid, None]
        last_valid = np.array([0.0, 0.0, 1.0], dtype=float)
        for index in range(len(segment_directions)):
            if valid[index]:
                last_valid = segment_directions[index]
            else:
                segment_directions[index] = last_valid
        next_valid = np.array([0.0, 0.0, 1.0], dtype=float)
        for index in range(len(segment_directions) - 1, -1, -1):
            if valid[index]:
                next_valid = segment_directions[index]
            elif np.linalg.norm(segment_directions[index]) <= 1e-9:
                segment_directions[index] = next_valid
    else:
        segment_directions[:] = np.array([0.0, 0.0, 1.0], dtype=float)
    station_directions = np.zeros((len(md_values), 3), dtype=float)
    station_directions[0] = segment_directions[0]
    station_directions[-1] = segment_directions[-1]
    for index in range(1, len(md_values) - 1):
        left_weight = float(delta_md[index - 1])
        right_weight = float(delta_md[index])
        weighted = left_weight * segment_directions[index - 1] + right_weight * segment_directions[index]
        norm = float(np.linalg.norm(weighted))
        if norm <= 1e-9:
            station_directions[index] = segment_directions[index]
        else:
            station_directions[index] = weighted / norm
    return _normalize_vectors(station_directions)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.array(vectors, dtype=float, copy=True)
    valid = norms > 1e-12
    normalized[valid] = normalized[valid] / norms[valid, None]
    normalized[~valid] = np.array([0.0, 0.0, 1.0], dtype=float)
    return normalized


def _regular_md_grid(md_values: np.ndarray, step_m: float) -> np.ndarray:
    start_md = float(md_values[0])
    end_md = float(md_values[-1])
    if end_md <= start_md + 1e-9:
        return np.asarray([start_md, end_md], dtype=float)
    grid = np.arange(start_md, end_md, float(step_m), dtype=float)
    if len(grid) == 0 or abs(grid[0] - start_md) > 1e-9:
        grid = np.concatenate([[start_md], grid])
    if abs(grid[-1] - end_md) > 1e-9:
        grid = np.concatenate([grid, [end_md]])
    return grid


def _mask_intervals(md_values: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, float]]:
    intervals: list[tuple[float, float, float]] = []
    current_start_index: int | None = None
    for index, is_active in enumerate(mask.tolist()):
        if is_active and current_start_index is None:
            current_start_index = index
        is_last = index == len(mask) - 1
        if current_start_index is not None and ((not is_active) or is_last):
            end_index = index if is_active and is_last else index - 1
            start_md = float(md_values[current_start_index])
            end_md = float(md_values[end_index])
            intervals.append((start_md, end_md, float(max(end_md - start_md, 0.0))))
            current_start_index = None
    return intervals


def _rolling_median(values: np.ndarray, window_size: int) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=float)
    if window_size <= 1:
        return np.asarray(values, dtype=float)
    series = pd.Series(np.asarray(values, dtype=float))
    return (
        series.rolling(window=window_size, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _robust_max_dls(stations: pd.DataFrame) -> float | None:
    if len(stations) < 2:
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)
    finite_mask = np.isfinite(dls_values)
    if not np.any(finite_mask):
        return None
    step_m = float(np.median(np.diff(md_values))) if len(md_values) > 1 else ACTUAL_FUND_RESAMPLE_STEP_M
    window_size = max(3, int(round(ROBUST_DLS_WINDOW_M / max(step_m, 1.0))))
    smoothed = _rolling_median(np.where(finite_mask, dls_values, 0.0), window_size=window_size)
    finite_smoothed = smoothed[np.isfinite(smoothed)]
    if len(finite_smoothed) == 0:
        return None
    return float(np.max(finite_smoothed))


def _detect_hold_interval(
    *,
    stations: pd.DataFrame,
    kop_md_m: float | None,
    horizontal_entry_md_m: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    if kop_md_m is None:
        return None, None, None, None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    azi_values = stations["AZI_deg"].to_numpy(dtype=float)
    dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)

    analysis_end_md = (
        float(horizontal_entry_md_m)
        if horizontal_entry_md_m is not None
        else float(md_values[-1])
    )
    candidate_mask = (
        (md_values >= float(kop_md_m) + 10.0)
        & (md_values <= analysis_end_md - 10.0)
        & np.isfinite(inc_values)
        & np.isfinite(dls_values)
        & (inc_values >= HOLD_MIN_INC_DEG)
        & (inc_values <= HOLD_MAX_INC_DEG)
    )
    if not np.any(candidate_mask):
        return None, None, None, None

    step_m = float(np.median(np.diff(md_values))) if len(md_values) > 1 else ACTUAL_FUND_RESAMPLE_STEP_M
    window_size = max(3, int(round(ROBUST_DLS_WINDOW_M / max(step_m, 1.0))))
    stable_dls = _rolling_median(np.where(np.isfinite(dls_values), dls_values, 0.0), window_size=window_size)
    weighted_mode_inc = _weighted_modal_inclination(
        md_values=md_values,
        inc_values=inc_values,
        mask=candidate_mask & (stable_dls <= HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M),
    )
    if weighted_mode_inc is None:
        return None, None, None, None

    hold_mask = (
        candidate_mask
        & (np.abs(inc_values - float(weighted_mode_inc)) <= HOLD_INC_TOLERANCE_DEG)
        & (stable_dls <= HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M)
    )
    intervals = [
        (start_md, end_md, length_m)
        for start_md, end_md, length_m in _mask_intervals(md_values, hold_mask)
        if length_m >= HOLD_MIN_INTERVAL_M
    ]
    if not intervals:
        return None, None, None, None
    start_md, end_md, _ = min(intervals, key=lambda item: (item[0], -item[2]))
    interval_mask = (md_values >= start_md - 1e-9) & (md_values <= end_md + 1e-9)
    hold_inc = float(np.median(inc_values[interval_mask]))
    hold_azi = _circular_median_deg(azi_values[interval_mask])
    return float(start_md), float(end_md), hold_inc, hold_azi


def _weighted_modal_inclination(
    *,
    md_values: np.ndarray,
    inc_values: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if not np.any(mask):
        return None
    rounded = np.round(inc_values).astype(int)
    weights = np.zeros(len(md_values), dtype=float)
    if len(md_values) > 1:
        weights[:-1] = np.diff(md_values)
        weights[-1] = weights[-2]
    totals: dict[int, float] = defaultdict(float)
    for index, is_active in enumerate(mask.tolist()):
        if is_active:
            totals[int(rounded[index])] += float(weights[index])
    if not totals:
        return None
    best_bin, _ = max(totals.items(), key=lambda item: (item[1], -abs(item[0] - 45)))
    return float(best_bin)
