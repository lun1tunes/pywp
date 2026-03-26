from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from pywp.anticollision import analyze_anti_collision, build_anti_collision_well
from pywp.pydantic_base import FrozenArbitraryModel
from pywp.reference_trajectories import ImportedTrajectoryWell
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    PlanningUncertaintyModel,
)

HORIZONTAL_INC_THRESHOLD_DEG = 80.0
HORIZONTAL_MIN_INTERVAL_M = 150.0
KOP_INC_THRESHOLD_DEG = 5.0
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
    max_inc_deg: float
    max_dls_deg_per_30m: float | None
    max_build_dls_before_hold_deg_per_30m: float | None


class ActualFundPadSummary(FrozenArbitraryModel):
    pad_group: str
    well_count: int
    horizontal_well_count: int
    median_kop_md_m: float | None
    median_hold_inc_deg: float | None
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
    interval = _dominant_horizontal_interval(stations)
    max_inc = float(np.nanmax(stations["INC_deg"].to_numpy(dtype=float)))
    return bool(max_inc >= HORIZONTAL_INC_THRESHOLD_DEG and interval[2] >= HORIZONTAL_MIN_INTERVAL_M)


def build_actual_fund_well_metrics(
    actual_wells: Iterable[ImportedTrajectoryWell],
) -> tuple[ActualFundWellMetrics, ...]:
    metrics: list[ActualFundWellMetrics] = []
    for well in actual_wells:
        stations = well.stations.copy()
        md_values = stations["MD_m"].to_numpy(dtype=float)
        z_values = stations["Z_m"].to_numpy(dtype=float)
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        inc_values = stations["INC_deg"].to_numpy(dtype=float)
        dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)
        kop_md = _first_threshold_crossing_md(
            md_values=md_values,
            values=inc_values,
            threshold=KOP_INC_THRESHOLD_DEG,
        )
        horizontal_entry_md, _, horizontal_length = _dominant_horizontal_interval(stations)
        hold_mask = (
            (md_values >= float(horizontal_entry_md) - 1e-9)
            if horizontal_entry_md is not None
            else np.zeros(len(md_values), dtype=bool)
        )
        hold_inc = (
            float(np.median(inc_values[hold_mask]))
            if np.any(hold_mask)
            else None
        )
        hold_azi = (
            _circular_median_deg(stations.loc[hold_mask, "AZI_deg"].to_numpy(dtype=float))
            if np.any(hold_mask)
            else None
        )
        build_mask = (
            (md_values <= float(horizontal_entry_md) + 1e-9)
            if horizontal_entry_md is not None
            else np.ones(len(md_values), dtype=bool)
        )
        build_dls_values = dls_values[build_mask]
        finite_build_dls = build_dls_values[np.isfinite(build_dls_values)]
        finite_dls = dls_values[np.isfinite(dls_values)]
        metrics.append(
            ActualFundWellMetrics(
                name=str(well.name),
                family_name=actual_well_family_name(well.name),
                pad_group=actual_well_pad_group(well.name),
                is_horizontal=actual_well_is_horizontal(stations),
                md_total_m=float(md_values[-1]),
                tvd_end_m=float(z_values[-1]),
                lateral_departure_m=float(
                    np.hypot(x_values[-1] - x_values[0], y_values[-1] - y_values[0])
                ),
                kop_md_m=kop_md,
                kop_tvd_m=(
                    None if kop_md is None else _interp_1d(md_values, z_values, float(kop_md))
                ),
                horizontal_entry_md_m=horizontal_entry_md,
                horizontal_length_m=float(horizontal_length),
                hold_inc_deg=hold_inc,
                hold_azi_deg=hold_azi,
                max_inc_deg=float(np.nanmax(inc_values)),
                max_dls_deg_per_30m=(
                    None if len(finite_dls) == 0 else float(np.max(finite_dls))
                ),
                max_build_dls_before_hold_deg_per_30m=(
                    None if len(finite_build_dls) == 0 else float(np.max(finite_build_dls))
                ),
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
            "MD, м": item.md_total_m,
            "KOP MD, м": item.kop_md_m,
            "KOP TVD, м": item.kop_tvd_m,
            "Вход в горизонталь, MD": item.horizontal_entry_md_m,
            "Горизонталь, м": item.horizontal_length_m,
            "Зенит HOLD, deg": item.hold_inc_deg,
            "Азимут HOLD, deg": item.hold_azi_deg,
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
    metrics = build_actual_fund_well_metrics(actual_well_list)
    horizontal_names = {
        item.name for item in metrics if bool(item.is_horizontal)
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
                "Для калибровки нужны минимум две фактические горизонтальные скважины."
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
                stations=well.stations,
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
                stations=well.stations,
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


def _dominant_horizontal_interval(stations: pd.DataFrame) -> tuple[float | None, float | None, float]:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    if len(md_values) == 0:
        return None, None, 0.0
    mask = inc_values >= HORIZONTAL_INC_THRESHOLD_DEG
    if not np.any(mask):
        return None, None, 0.0
    best_start: float | None = None
    best_end: float | None = None
    best_length = 0.0
    current_start_index: int | None = None
    for index, is_high_angle in enumerate(mask.tolist()):
        if is_high_angle and current_start_index is None:
            current_start_index = index
        if (not is_high_angle or index == len(mask) - 1) and current_start_index is not None:
            end_index = index if is_high_angle and index == len(mask) - 1 else index - 1
            start_md = float(md_values[current_start_index])
            end_md = float(md_values[end_index])
            length = float(max(end_md - start_md, 0.0))
            if length > best_length:
                best_start, best_end, best_length = start_md, end_md, length
            current_start_index = None
    return best_start, best_end, best_length


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
