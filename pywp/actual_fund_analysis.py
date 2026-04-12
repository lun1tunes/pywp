from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from pywp.anticollision import (
    AntiCollisionAnalysis,
    analyze_anti_collision,
    anti_collision_report_events,
    build_anti_collision_well,
)
from pywp.constants import SMALL
from pywp.mcm import add_dls, wrap_azimuth_deg
from pywp.pydantic_base import FrozenArbitraryModel
from pywp.reference_trajectories import ImportedTrajectoryWell
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    PlanningUncertaintyModel,
    fitted_uncertainty_model,
)

HORIZONTAL_INC_THRESHOLD_DEG = 80.0
HORIZONTAL_MIN_INTERVAL_M = 100.0
HORIZONTAL_END_TOLERANCE_M = 60.0
ACTUAL_FUND_RESAMPLE_STEP_M = 10.0
HOLD_INC_TOLERANCE_DEG = 3.0
HOLD_MIN_INTERVAL_M = 90.0
HOLD_MAX_INC_DEG = 78.0
HOLD_MIN_INC_DEG = 8.0
HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M = 1.5
ROBUST_DLS_WINDOW_M = 60.0
MAX_REASONABLE_ACTUAL_DLS_DEG_PER_30M = 6.0
MIN_CUSTOM_ACTUAL_FUND_SCALE = 0.35
MIN_CUSTOM_ACTUAL_FUND_SEARCH_SCALE = 0.12
KOP_BUILD_RATE_THRESHOLD_DEG_PER_30M = 0.55
KOP_MIN_BUILD_INTERVAL_M = 60.0
KOP_VERTICAL_BASELINE_MAX_INC_DEG = 10.0
KOP_INC_BUFFER_DEG = 3.0
MAX_IGNORED_CLOSE_CALIBRATION_PAIRS = 2
IGNORABLE_CLOSE_PAIR_SF_THRESHOLD = 0.35
IGNORABLE_CLOSE_PAIR_OVERLAP_M = 25.0

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
    horizontal_entry_tvd_m: float | None
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


class ActualFundDepthClusterSummary(FrozenArbitraryModel):
    cluster_id: str
    well_count: int
    depth_from_tvd_m: float
    depth_to_tvd_m: float
    median_horizontal_entry_tvd_m: float
    median_kop_md_m: float
    anchor_horizontal_entry_tvd_m: float
    anchor_kop_md_m: float
    well_names: tuple[str, ...]


class ActualFundKopDepthFunction(FrozenArbitraryModel):
    mode: str
    cluster_count: int
    anchor_depths_tvd_m: tuple[float, ...]
    anchor_kop_md_m: tuple[float, ...]
    note: str

    def evaluate(self, horizontal_tvd_m: float) -> float:
        depth = float(horizontal_tvd_m)
        if not self.anchor_depths_tvd_m or not self.anchor_kop_md_m:
            raise ValueError("KOP depth function has no anchors.")
        if len(self.anchor_depths_tvd_m) == 1:
            return float(self.anchor_kop_md_m[0])
        return float(
            np.interp(
                depth,
                np.asarray(self.anchor_depths_tvd_m, dtype=float),
                np.asarray(self.anchor_kop_md_m, dtype=float),
            )
        )


class ActualFundZoneSummary(FrozenArbitraryModel):
    zone_key: str
    zone_label: str
    md_from_m: float
    md_to_m: float
    md_length_m: float
    inc_min_deg: float | None
    inc_max_deg: float | None
    inc_mean_deg: float | None
    dls_min_deg_per_30m: float | None
    dls_max_deg_per_30m: float | None
    dls_mean_deg_per_30m: float | None


class ActualFundWellAnalysis(FrozenArbitraryModel):
    name: str
    metrics: ActualFundWellMetrics
    survey: pd.DataFrame
    zone_summaries: tuple[ActualFundZoneSummary, ...]


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
    excluded_pilot_well_count: int = 0
    ignored_close_pair_count: int = 0
    ignored_close_pairs: tuple[str, ...] = ()
    note: str = ""


ZONE_VERTICAL = "vertical"
ZONE_BUILD1 = "build1"
ZONE_HOLD = "hold"
ZONE_BUILD2 = "build2"
ZONE_HORIZONTAL = "horizontal"

ZONE_LABELS: dict[str, str] = {
    ZONE_VERTICAL: "Вертикальный",
    ZONE_BUILD1: "BUILD 1",
    ZONE_HOLD: "HOLD",
    ZONE_BUILD2: "BUILD 2",
    ZONE_HORIZONTAL: "Горизонтальный",
}

HORIZONTAL_ENTRY_MIN_INC_DEG = 70.0
DEPTH_CLUSTER_REL_TOLERANCE = 0.08
DEPTH_CLUSTER_OUTLIER_IQR_FACTOR = 2.5


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


def actual_well_is_pilot_name(name: object) -> bool:
    label = str(name or "").strip().upper()
    return bool(label.endswith("_PL"))


def actual_well_is_horizontal(stations: pd.DataFrame) -> bool:
    survey = _reconstruct_actual_survey(stations)
    interval = _terminal_horizontal_interval(survey)
    max_inc = float(np.nanmax(survey["INC_deg"].to_numpy(dtype=float)))
    return bool(
        max_inc >= HORIZONTAL_INC_THRESHOLD_DEG
        and interval[2] >= HORIZONTAL_MIN_INTERVAL_M
    )


def build_actual_fund_well_analysis(
    actual_well: ImportedTrajectoryWell,
) -> ActualFundWellAnalysis:
    return _analyze_actual_well(actual_well)


def build_actual_fund_well_analyses(
    actual_wells: Iterable[ImportedTrajectoryWell],
) -> tuple[ActualFundWellAnalysis, ...]:
    return tuple(_analyze_actual_well(well) for well in actual_wells)


def build_actual_fund_well_metrics(
    actual_wells: Iterable[ImportedTrajectoryWell],
) -> tuple[ActualFundWellMetrics, ...]:
    return tuple(item.metrics for item in build_actual_fund_well_analyses(actual_wells))


def _analyze_actual_well(actual_well: ImportedTrajectoryWell) -> ActualFundWellAnalysis:
    stations = _reconstruct_actual_survey(actual_well.stations)
    source_md_values = actual_well.stations["MD_m"].to_numpy(dtype=float)
    source_z_values = actual_well.stations["Z_m"].to_numpy(dtype=float)
    md_values = stations["MD_m"].to_numpy(dtype=float)
    z_values = stations["Z_m"].to_numpy(dtype=float)
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    terminal_horizontal_start_md, _, terminal_horizontal_length_m = (
        _terminal_horizontal_interval(stations)
    )
    provisional_hold_start_md, provisional_hold_end_md, _, _ = _detect_hold_interval(
        stations=stations,
        kop_md_m=None,
        horizontal_entry_md_m=terminal_horizontal_start_md,
    )
    kop_md = _detect_kop_md(
        stations=stations,
        hold_start_md_m=provisional_hold_start_md,
    )
    horizontal_entry_md = _detect_horizontal_entry_md(
        stations=stations,
        kop_md_m=kop_md,
        hold_end_md_m=provisional_hold_end_md,
        terminal_horizontal_start_md_m=terminal_horizontal_start_md,
    )
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
    is_horizontal = bool(
        terminal_horizontal_start_md is not None
        and float(terminal_horizontal_length_m) >= HORIZONTAL_MIN_INTERVAL_M
        and float(np.nanmax(inc_values)) >= HORIZONTAL_INC_THRESHOLD_DEG
    )
    if not is_horizontal:
        horizontal_entry_md = None
    horizontal_length = (
        0.0
        if horizontal_entry_md is None
        else float(max(md_values[-1] - float(horizontal_entry_md), 0.0))
    )
    build_limit_md = (
        float(hold_start_md)
        if hold_start_md is not None
        else (
            float(horizontal_entry_md)
            if horizontal_entry_md is not None
            else float(md_values[-1])
        )
    )
    build_mask = md_values <= build_limit_md + SMALL
    max_dls = _robust_max_dls(stations)
    max_build_dls = _robust_max_dls(stations.loc[build_mask].reset_index(drop=True))
    exclusion_reason = _analysis_exclusion_reason(
        is_horizontal=is_horizontal,
        kop_md_m=kop_md,
        horizontal_entry_md_m=horizontal_entry_md,
        horizontal_entry_tvd_m=(
            None
            if horizontal_entry_md is None
            else _interp_1d(md_values, z_values, float(horizontal_entry_md))
        ),
        horizontal_length_m=float(horizontal_length),
        hold_inc_deg=hold_inc,
        hold_length_m=float(hold_length),
        max_dls_deg_per_30m=max_dls,
        max_build_dls_before_hold_deg_per_30m=max_build_dls,
    )
    metrics = ActualFundWellMetrics(
        name=str(actual_well.name),
        family_name=actual_well_family_name(actual_well.name),
        pad_group=actual_well_pad_group(actual_well.name),
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
        horizontal_entry_tvd_m=(
            None
            if horizontal_entry_md is None
            else _interp_1d(md_values, z_values, float(horizontal_entry_md))
        ),
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
    zone_summaries = _build_zone_summaries(
        stations=stations,
        kop_md_m=kop_md,
        hold_start_md_m=hold_start_md,
        hold_end_md_m=hold_end_md,
        horizontal_entry_md_m=horizontal_entry_md,
    )
    return ActualFundWellAnalysis(
        name=str(actual_well.name),
        metrics=metrics,
        survey=_annotate_actual_fund_survey(
            stations=stations,
            zone_summaries=zone_summaries,
        ),
        zone_summaries=zone_summaries,
    )


def _build_zone_summaries(
    *,
    stations: pd.DataFrame,
    kop_md_m: float | None,
    hold_start_md_m: float | None,
    hold_end_md_m: float | None,
    horizontal_entry_md_m: float | None,
) -> tuple[ActualFundZoneSummary, ...]:
    md_start = float(stations["MD_m"].iloc[0])
    md_end = float(stations["MD_m"].iloc[-1])
    intervals: list[tuple[str, float, float]] = []

    def add_interval(zone_key: str, start_m: float | None, end_m: float | None) -> None:
        if start_m is None or end_m is None:
            return
        start_value = float(start_m)
        end_value = float(end_m)
        if end_value <= start_value + SMALL:
            return
        intervals.append((zone_key, start_value, end_value))

    if kop_md_m is None:
        add_interval(ZONE_VERTICAL, md_start, md_end)
    else:
        kop_md = float(min(max(kop_md_m, md_start), md_end))
        add_interval(ZONE_VERTICAL, md_start, kop_md)
        if hold_start_md_m is not None and hold_start_md_m > kop_md + SMALL:
            add_interval(ZONE_BUILD1, kop_md, hold_start_md_m)
        elif (
            horizontal_entry_md_m is not None and horizontal_entry_md_m > kop_md + SMALL
        ):
            add_interval(ZONE_BUILD1, kop_md, horizontal_entry_md_m)
        else:
            add_interval(ZONE_BUILD1, kop_md, md_end)

        if hold_start_md_m is not None and hold_end_md_m is not None:
            add_interval(ZONE_HOLD, hold_start_md_m, hold_end_md_m)
            if (
                horizontal_entry_md_m is not None
                and horizontal_entry_md_m > hold_end_md_m + SMALL
            ):
                add_interval(ZONE_BUILD2, hold_end_md_m, horizontal_entry_md_m)
            elif hold_end_md_m < md_end - SMALL and horizontal_entry_md_m is None:
                add_interval(ZONE_BUILD2, hold_end_md_m, md_end)

        if horizontal_entry_md_m is not None:
            add_interval(ZONE_HORIZONTAL, horizontal_entry_md_m, md_end)

    zone_summaries: list[ActualFundZoneSummary] = []
    for zone_key, zone_start_md, zone_end_md in intervals:
        interval_df = _survey_interval(stations, zone_start_md, zone_end_md)
        inc_stats = _series_stats(interval_df["INC_deg"])
        dls_stats = _series_stats(interval_df["DLS_deg_per_30m"])
        zone_summaries.append(
            ActualFundZoneSummary(
                zone_key=zone_key,
                zone_label=ZONE_LABELS.get(zone_key, zone_key),
                md_from_m=float(zone_start_md),
                md_to_m=float(zone_end_md),
                md_length_m=float(max(zone_end_md - zone_start_md, 0.0)),
                inc_min_deg=inc_stats[0],
                inc_max_deg=inc_stats[1],
                inc_mean_deg=inc_stats[2],
                dls_min_deg_per_30m=dls_stats[0],
                dls_max_deg_per_30m=dls_stats[1],
                dls_mean_deg_per_30m=dls_stats[2],
            )
        )
    return tuple(zone_summaries)


def _annotate_actual_fund_survey(
    *,
    stations: pd.DataFrame,
    zone_summaries: tuple[ActualFundZoneSummary, ...],
) -> pd.DataFrame:
    annotated = stations.copy()
    x_values = annotated["X_m"].to_numpy(dtype=float)
    y_values = annotated["Y_m"].to_numpy(dtype=float)
    annotated["Lateral_m"] = np.hypot(
        x_values - float(x_values[0]),
        y_values - float(y_values[0]),
    )
    zone_keys: list[str] = []
    zone_labels: list[str] = []
    md_values = annotated["MD_m"].to_numpy(dtype=float)
    for md_value in md_values.tolist():
        summary = next(
            (
                item
                for item in zone_summaries
                if float(item.md_from_m) - SMALL
                <= float(md_value)
                <= float(item.md_to_m) + SMALL
            ),
            None,
        )
        if summary is None:
            zone_keys.append(ZONE_BUILD1)
            zone_labels.append(ZONE_LABELS[ZONE_BUILD1])
        else:
            zone_keys.append(str(summary.zone_key))
            zone_labels.append(str(summary.zone_label))
    annotated["AnalysisZoneKey"] = zone_keys
    annotated["AnalysisZoneLabel"] = zone_labels
    return annotated


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
            "Вход в горизонталь, TVD": item.horizontal_entry_tvd_m,
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


def summarize_actual_fund_by_depth(
    metrics: Iterable[ActualFundWellMetrics],
    *,
    relative_tolerance: float = DEPTH_CLUSTER_REL_TOLERANCE,
) -> tuple[ActualFundDepthClusterSummary, ...]:
    eligible = [
        item
        for item in metrics
        if bool(item.is_analysis_eligible)
        and item.horizontal_entry_tvd_m is not None
        and item.kop_md_m is not None
    ]
    if not eligible:
        return ()
    ordered = sorted(eligible, key=lambda item: float(item.horizontal_entry_tvd_m))
    clusters: list[list[ActualFundWellMetrics]] = []
    current: list[ActualFundWellMetrics] = []
    for item in ordered:
        depth_tvd = float(item.horizontal_entry_tvd_m)
        if not current:
            current = [item]
            continue
        current_depths = np.asarray(
            [float(candidate.horizontal_entry_tvd_m) for candidate in current],
            dtype=float,
        )
        cluster_center = float(np.median(current_depths))
        max_allowed_gap = max(80.0, abs(cluster_center) * float(relative_tolerance))
        if abs(depth_tvd - cluster_center) <= max_allowed_gap:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    if current:
        clusters.append(current)

    summaries: list[ActualFundDepthClusterSummary] = []
    for index, cluster_items in enumerate(clusters, start=1):
        depths = np.asarray(
            [float(item.horizontal_entry_tvd_m) for item in cluster_items],
            dtype=float,
        )
        kops = np.asarray(
            [float(item.kop_md_m) for item in cluster_items],
            dtype=float,
        )
        filtered_depths = _filter_depth_cluster_outliers(depths)
        filtered_kops = _filter_depth_cluster_outliers(kops)
        summaries.append(
            ActualFundDepthClusterSummary(
                cluster_id=f"DEPTH-{index:02d}",
                well_count=len(cluster_items),
                depth_from_tvd_m=float(np.min(depths)),
                depth_to_tvd_m=float(np.max(depths)),
                median_horizontal_entry_tvd_m=float(np.median(depths)),
                median_kop_md_m=float(np.median(kops)),
                anchor_horizontal_entry_tvd_m=_cluster_anchor_value(filtered_depths),
                anchor_kop_md_m=_cluster_anchor_value(filtered_kops),
                well_names=tuple(str(item.name) for item in cluster_items),
            )
        )
    return tuple(summaries)


def actual_fund_depth_rows(
    metrics: Iterable[ActualFundWellMetrics],
    *,
    relative_tolerance: float = DEPTH_CLUSTER_REL_TOLERANCE,
) -> list[dict[str, object]]:
    return [
        {
            "Глубинный кластер": item.cluster_id,
            "Скважин": item.well_count,
            "TVD диапазон, м": f"{float(item.depth_from_tvd_m):.0f} - {float(item.depth_to_tvd_m):.0f}",
            "Якорный TVD входа, м": float(item.anchor_horizontal_entry_tvd_m),
            "Якорный KOP MD, м": float(item.anchor_kop_md_m),
            "Скважины": ", ".join(item.well_names),
        }
        for item in summarize_actual_fund_by_depth(
            metrics,
            relative_tolerance=relative_tolerance,
        )
    ]


def build_actual_fund_kop_depth_function(
    metrics: Iterable[ActualFundWellMetrics],
    *,
    relative_tolerance: float = DEPTH_CLUSTER_REL_TOLERANCE,
) -> ActualFundKopDepthFunction | None:
    clusters = summarize_actual_fund_by_depth(
        metrics,
        relative_tolerance=relative_tolerance,
    )
    if not clusters:
        return None
    depths = tuple(float(item.anchor_horizontal_entry_tvd_m) for item in clusters)
    kops = tuple(float(item.anchor_kop_md_m) for item in clusters)
    if len(clusters) == 1:
        return ActualFundKopDepthFunction(
            mode="constant",
            cluster_count=1,
            anchor_depths_tvd_m=depths,
            anchor_kop_md_m=kops,
            note=(
                "Один глубинный кластер: функция вырождается в константу "
                "по якорю min + 1σ после отсечения явных выбросов."
            ),
        )
    return ActualFundKopDepthFunction(
        mode="piecewise_linear",
        cluster_count=len(clusters),
        anchor_depths_tvd_m=depths,
        anchor_kop_md_m=kops,
        note=(
            "KOP(TVD) задан кусочно-линейно по якорям глубинных кластеров: "
            "min + 1σ после отсечения явных выбросов."
        ),
    )


def _filter_depth_cluster_outliers(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 4:
        return array
    q1, q3 = np.percentile(array, [25.0, 75.0])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= SMALL:
        return array
    lower = float(q1 - DEPTH_CLUSTER_OUTLIER_IQR_FACTOR * iqr)
    upper = float(q3 + DEPTH_CLUSTER_OUTLIER_IQR_FACTOR * iqr)
    filtered = array[(array >= lower) & (array <= upper)]
    return filtered if filtered.size >= 2 else array


def _cluster_anchor_value(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cluster anchor cannot be built from empty values.")
    min_value = float(np.min(array))
    std_value = float(np.std(array, ddof=0)) if array.size > 1 else 0.0
    anchor = min_value + max(std_value, 0.0)
    return float(np.clip(anchor, min_value, float(np.max(array))))


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
                median_kop_md_m=_median_or_none(
                    item.kop_md_m for item in horizontal_items
                ),
                median_hold_inc_deg=_median_or_none(
                    item.hold_inc_deg for item in horizontal_items
                ),
                median_hold_length_m=_median_or_none(
                    item.hold_length_m for item in horizontal_items
                ),
                max_pi_deg_per_30m=_max_or_none(
                    item.max_dls_deg_per_30m for item in horizontal_items
                ),
                median_horizontal_length_m=_median_or_none(
                    item.horizontal_length_m for item in horizontal_items
                ),
            )
        )
    return tuple(summaries)


def _calibration_pair_key(well_a: str, well_b: str) -> tuple[str, str]:
    return tuple(sorted((str(well_a), str(well_b))))


def _calibration_pair_label(pair_key: tuple[str, str]) -> str:
    return f"{pair_key[0]} ↔ {pair_key[1]}"


def _calibration_pair_filter(
    *,
    family_by_name: Mapping[str, str],
    ignored_pair_keys: set[tuple[str, str]],
):
    def pair_filter(left, right) -> bool:
        left_name = str(left.name)
        right_name = str(right.name)
        if family_by_name[left_name] == family_by_name[right_name]:
            return False
        if _calibration_pair_key(left_name, right_name) in ignored_pair_keys:
            return False
        return True

    return pair_filter


def _build_calibration_analysis(
    *,
    eligible_wells: list[ImportedTrajectoryWell],
    reconstructed_by_name: Mapping[str, pd.DataFrame],
    model: PlanningUncertaintyModel,
    ignored_pair_keys: set[tuple[str, str]],
    family_by_name: Mapping[str, str],
) -> "AntiCollisionAnalysis":
    return analyze_anti_collision(
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
                model=model,
                include_display_geometry=False,
                well_kind="actual",
                is_reference_only=False,
            )
            for well in eligible_wells
        ],
        build_overlap_geometry=False,
        pair_filter=_calibration_pair_filter(
            family_by_name=family_by_name,
            ignored_pair_keys=ignored_pair_keys,
        ),
    )


def _retained_uncertainty_score(
    *,
    base_model: PlanningUncertaintyModel,
    candidate_model: PlanningUncertaintyModel,
) -> float:
    return float(
        np.mean(
            [
                float(candidate_model.sigma_inc_deg) / float(base_model.sigma_inc_deg),
                float(candidate_model.sigma_azi_deg) / float(base_model.sigma_azi_deg),
                float(candidate_model.sigma_lateral_drift_m_per_1000m)
                / max(float(base_model.sigma_lateral_drift_m_per_1000m), SMALL),
                float(candidate_model.confidence_scale)
                / float(base_model.confidence_scale),
            ]
        )
    )


def _calibration_candidate_model(
    *,
    base_model: PlanningUncertaintyModel,
    shape_key: str,
    alpha: float,
) -> PlanningUncertaintyModel:
    alpha_value = float(alpha)
    if shape_key == "uniform":
        return fitted_uncertainty_model(
            base_model,
            sigma_inc_scale=alpha_value,
            sigma_azi_scale=alpha_value,
            sigma_lateral_drift_scale=alpha_value,
            confidence_scale_factor=max(alpha_value, 0.10),
        )
    if shape_key == "azi_drift_relaxed":
        return fitted_uncertainty_model(
            base_model,
            sigma_inc_scale=max(alpha_value**0.90, 0.05),
            sigma_azi_scale=max(alpha_value**1.25, 0.05),
            sigma_lateral_drift_scale=max(alpha_value**1.35, 0.05),
            confidence_scale_factor=max(alpha_value**1.08, 0.10),
        )
    if shape_key == "angular_relaxed":
        return fitted_uncertainty_model(
            base_model,
            sigma_inc_scale=max(alpha_value**1.05, 0.05),
            sigma_azi_scale=max(alpha_value**1.20, 0.05),
            sigma_lateral_drift_scale=max(alpha_value**1.10, 0.05),
            confidence_scale_factor=max(alpha_value**1.05, 0.10),
        )
    if shape_key == "drift_relaxed":
        return fitted_uncertainty_model(
            base_model,
            sigma_inc_scale=max(alpha_value**0.95, 0.05),
            sigma_azi_scale=max(alpha_value**1.05, 0.05),
            sigma_lateral_drift_scale=max(alpha_value**1.55, 0.05),
            confidence_scale_factor=max(alpha_value**1.03, 0.10),
        )
    raise ValueError(f"Unsupported calibration shape: {shape_key}")


def _ignorable_close_pair_keys(
    analysis: "AntiCollisionAnalysis",
) -> tuple[tuple[str, str], ...]:
    candidates: list[tuple[tuple[str, str], float, float]] = []
    for event in anti_collision_report_events(analysis):
        min_sf = float(event.min_separation_factor)
        max_overlap = float(event.max_overlap_depth_m)
        if (
            min_sf <= IGNORABLE_CLOSE_PAIR_SF_THRESHOLD
            or max_overlap >= IGNORABLE_CLOSE_PAIR_OVERLAP_M
        ):
            candidates.append(
                (
                    _calibration_pair_key(str(event.well_a), str(event.well_b)),
                    min_sf,
                    max_overlap,
                )
            )
    candidates = sorted(
        candidates,
        key=lambda item: (item[1], -item[2], item[0][0], item[0][1]),
    )
    unique_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair_key, _, _ in candidates:
        if pair_key in seen:
            continue
        seen.add(pair_key)
        unique_keys.append(pair_key)
        if len(unique_keys) >= MAX_IGNORED_CLOSE_CALIBRATION_PAIRS:
            break
    return tuple(unique_keys)


def calibrate_uncertainty_from_actual_fund(
    *,
    actual_wells: Iterable[ImportedTrajectoryWell],
    base_model: PlanningUncertaintyModel,
    base_preset: str = DEFAULT_UNCERTAINTY_PRESET,
    analyses: tuple[ActualFundWellAnalysis, ...] | None = None,
) -> ActualFundCalibrationResult:
    actual_well_list = list(actual_wells)
    if analyses is None:
        analyses = build_actual_fund_well_analyses(actual_well_list)
    reconstructed_by_name = {str(item.name): item.survey for item in analyses}
    metrics = tuple(item.metrics for item in analyses)
    eligible_metric_names = {
        item.name for item in metrics if bool(item.is_analysis_eligible)
    }
    excluded_pilot_metric_names = {
        item.name
        for item in metrics
        if bool(item.is_analysis_eligible) and actual_well_is_pilot_name(item.name)
    }
    eligible_wells = [
        well
        for well in actual_well_list
        if str(well.name) in eligible_metric_names
        and str(well.name) not in excluded_pilot_metric_names
    ]
    excluded_pilot_well_count = len(excluded_pilot_metric_names)
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
            excluded_pilot_well_count=excluded_pilot_well_count,
            note=(
                "Для калибровки нужны минимум две фактические горизонтальные "
                "скважины без аномалий анализа и без пилотов `_PL`."
            ),
        )

    family_by_name = {
        str(well.name): actual_well_family_name(well.name) for well in eligible_wells
    }
    skipped_same_family_pairs = sum(
        1
        for left_index in range(len(eligible_wells))
        for right_index in range(left_index + 1, len(eligible_wells))
        if family_by_name[str(eligible_wells[left_index].name)]
        == family_by_name[str(eligible_wells[right_index].name)]
    )
    analysis_before = _build_calibration_analysis(
        eligible_wells=eligible_wells,
        reconstructed_by_name=reconstructed_by_name,
        model=base_model,
        ignored_pair_keys=set(),
        family_by_name=family_by_name,
    )
    worst_sf_before = analysis_before.worst_separation_factor
    if worst_sf_before is None or float(worst_sf_before) >= 0.999:
        note = (
            "Текущая модель уже не даёт overlap по фактическому горизонтальному фонду. "
            "Дополнительная пользовательская калибровка не требуется."
        )
        if excluded_pilot_well_count:
            note += f" Из fit исключены пилоты `_PL`: {excluded_pilot_well_count}."
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
            excluded_pilot_well_count=excluded_pilot_well_count,
            note=note,
        )

    ignored_candidates = _ignorable_close_pair_keys(analysis_before)
    ignore_options: list[tuple[tuple[str, str], ...]] = [()]
    for count in range(1, len(ignored_candidates) + 1):
        ignore_options.append(tuple(ignored_candidates[:count]))

    search_alphas = np.linspace(1.0, MIN_CUSTOM_ACTUAL_FUND_SEARCH_SCALE, num=18)
    shape_keys = (
        "uniform",
        "azi_drift_relaxed",
        "angular_relaxed",
        "drift_relaxed",
    )
    best_candidate: (
        tuple[
            float,
            PlanningUncertaintyModel,
            "AntiCollisionAnalysis",
            tuple[tuple[str, str], ...],
            str,
            float,
        ]
        | None
    ) = None
    for ignored_pair_keys in ignore_options:
        ignored_pair_key_set = set(ignored_pair_keys)
        for shape_key in shape_keys:
            for alpha in search_alphas.tolist():
                candidate_model = _calibration_candidate_model(
                    base_model=base_model,
                    shape_key=shape_key,
                    alpha=float(alpha),
                )
                analysis_after = _build_calibration_analysis(
                    eligible_wells=eligible_wells,
                    reconstructed_by_name=reconstructed_by_name,
                    model=candidate_model,
                    ignored_pair_keys=ignored_pair_key_set,
                    family_by_name=family_by_name,
                )
                if int(analysis_after.overlapping_pair_count) != 0:
                    continue
                retained_score = _retained_uncertainty_score(
                    base_model=base_model,
                    candidate_model=candidate_model,
                )
                score = retained_score - 0.12 * len(ignored_pair_keys)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (
                        score,
                        candidate_model,
                        analysis_after,
                        ignored_pair_keys,
                        shape_key,
                        float(alpha),
                    )

    if best_candidate is None:
        note = (
            "Автоматическая пользовательская функция не построена: даже после "
            "адаптивного ослабления INC/AZI/drift и исключения пилотов `_PL` "
            "ложные overlap по фактическому фонду остаются слишком жёсткими."
        )
        if ignored_candidates:
            note += (
                " Пробовали игнорировать до двух экстремально близких пар: "
                + ", ".join(
                    _calibration_pair_label(pair_key) for pair_key in ignored_candidates
                )
                + "."
            )
        return ActualFundCalibrationResult(
            status=CALIBRATION_STATUS_TOO_AGGRESSIVE,
            base_preset=str(base_preset),
            scale_factor=None,
            actual_well_count=len(actual_well_list),
            horizontal_well_count=len(eligible_wells),
            analyzed_pair_count=int(analysis_before.pair_count),
            skipped_same_family_pair_count=int(skipped_same_family_pairs),
            overlapping_pair_count_before=int(analysis_before.overlapping_pair_count),
            worst_separation_factor_before=worst_sf_before,
            excluded_pilot_well_count=excluded_pilot_well_count,
            note=note,
        )

    _, custom_model, analysis_after, ignored_pair_keys, shape_key, scale_factor = (
        best_candidate
    )
    note_parts = [
        "Пользовательская модель построена как empirical field-fit относительно "
        f"базового пресета '{base_preset}'.",
        f"Форма подгонки: {shape_key}, базовый scale={scale_factor:.2f}.",
        "Алгоритм умеет независимо ослаблять INC/AZI/drift, а не только одной "
        "глобальной шкалой.",
    ]
    if excluded_pilot_well_count:
        note_parts.append(
            f"Из fit исключены пилоты `_PL`: {excluded_pilot_well_count}."
        )
    if ignored_pair_keys:
        note_parts.append(
            "Сознательно проигнорированы экстремально близкие пары: "
            + ", ".join(
                _calibration_pair_label(pair_key) for pair_key in ignored_pair_keys
            )
            + "."
        )
    note_parts.append(
        "Это planning-level fit по фактическому фонду, а не формальная ISCWSA toolcode calibration."
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
        excluded_pilot_well_count=excluded_pilot_well_count,
        ignored_close_pair_count=len(ignored_pair_keys),
        ignored_close_pairs=tuple(
            _calibration_pair_label(pair_key) for pair_key in ignored_pair_keys
        ),
        note=" ".join(note_parts),
    )


def _terminal_horizontal_interval(
    stations: pd.DataFrame,
) -> tuple[float | None, float | None, float]:
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
    best_start, best_end, best_length = max(
        candidates, key=lambda item: (item[2], item[1])
    )
    return float(best_start), float(best_end), float(best_length)


def _detect_horizontal_entry_md(
    *,
    stations: pd.DataFrame,
    kop_md_m: float | None,
    hold_end_md_m: float | None,
    terminal_horizontal_start_md_m: float | None,
) -> float | None:
    if terminal_horizontal_start_md_m is None:
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)
    step_m = (
        float(np.median(np.diff(md_values)))
        if len(md_values) > 1
        else ACTUAL_FUND_RESAMPLE_STEP_M
    )
    window_size = max(3, int(round(ROBUST_DLS_WINDOW_M / max(step_m, 1.0))))
    stable_dls = _rolling_median(
        np.where(np.isfinite(dls_values), dls_values, 0.0), window_size=window_size
    )
    tail_length_m = max(HORIZONTAL_MIN_INTERVAL_M, 120.0)
    tail_mask = md_values >= float(md_values[-1]) - tail_length_m
    tail_dls = stable_dls[tail_mask & np.isfinite(stable_dls)]
    terminal_dls_baseline = (
        float(np.median(tail_dls))
        if len(tail_dls) > 0
        else HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M
    )
    horizontal_dls_threshold = float(
        min(
            HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M,
            max(0.6, terminal_dls_baseline + 0.4),
        )
    )
    search_start_md = (
        float(hold_end_md_m)
        if hold_end_md_m is not None
        else float(kop_md_m) if kop_md_m is not None else float(md_values[0])
    )
    horizontal_mask = (
        (md_values >= search_start_md - SMALL)
        & (stable_dls <= horizontal_dls_threshold)
        & (inc_values >= HORIZONTAL_ENTRY_MIN_INC_DEG)
    )
    candidate_intervals = [
        (start_md, end_md, length_m)
        for start_md, end_md, length_m in _mask_intervals(md_values, horizontal_mask)
        if end_md >= float(md_values[-1]) - HORIZONTAL_END_TOLERANCE_M
        and length_m >= HORIZONTAL_MIN_INTERVAL_M
    ]
    if candidate_intervals:
        best_start_md = min(float(item[0]) for item in candidate_intervals)
        return float(best_start_md)
    return float(terminal_horizontal_start_md_m)


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
            if abs(delta) <= SMALL:
                return float(md_values[index])
            alpha = float((threshold - left_value) / delta)
            return float(
                md_values[index - 1] + alpha * (md_values[index] - md_values[index - 1])
            )
    return None


def _detect_kop_md(
    *,
    stations: pd.DataFrame,
    hold_start_md_m: float | None,
) -> float | None:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if len(md_values) < 2:
        return None

    stable_inc = _stable_inc_values(stations)
    build_rate = _stable_inc_build_rate_deg_per_30m(
        stations=stations,
        stable_inc=stable_inc,
    )
    search_limit_md = (
        float(hold_start_md_m) if hold_start_md_m is not None else float(md_values[-1])
    )
    pre_hold_mask = md_values <= search_limit_md + SMALL
    if not np.any(pre_hold_mask):
        return None

    baseline_candidates = stable_inc[
        pre_hold_mask & (stable_inc <= KOP_VERTICAL_BASELINE_MAX_INC_DEG)
    ]
    if len(baseline_candidates) == 0:
        baseline_candidates = stable_inc[pre_hold_mask]
    if len(baseline_candidates) == 0:
        return None
    vertical_baseline_inc = float(np.median(baseline_candidates))
    kop_inc_threshold = float(
        max(
            HOLD_MIN_INC_DEG - 2.0,
            vertical_baseline_inc + KOP_INC_BUFFER_DEG,
        )
    )
    build_mask = (
        pre_hold_mask
        & np.isfinite(stable_inc)
        & np.isfinite(build_rate)
        & (stable_inc >= kop_inc_threshold)
        & (build_rate >= KOP_BUILD_RATE_THRESHOLD_DEG_PER_30M)
    )
    build_intervals = [
        (start_md, end_md, length_m)
        for start_md, end_md, length_m in _mask_intervals(md_values, build_mask)
        if length_m >= KOP_MIN_BUILD_INTERVAL_M
    ]
    if build_intervals:
        return float(min(float(item[0]) for item in build_intervals))
    return _first_threshold_crossing_md(
        md_values=md_values[pre_hold_mask],
        values=stable_inc[pre_hold_mask],
        threshold=kop_inc_threshold,
    )


def _interp_1d(md_values: np.ndarray, values: np.ndarray, md_m: float) -> float:
    return float(np.interp(float(md_m), md_values, values))


def _interpolate_row(stations: pd.DataFrame, md_m: float) -> dict[str, float]:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    return {
        column: float(
            np.interp(float(md_m), md_values, stations[column].to_numpy(dtype=float))
        )
        for column in (
            "MD_m",
            "INC_deg",
            "AZI_deg",
            "X_m",
            "Y_m",
            "Z_m",
            "DLS_deg_per_30m",
        )
    }


def _survey_interval(
    stations: pd.DataFrame, start_md_m: float, end_md_m: float
) -> pd.DataFrame:
    if end_md_m <= start_md_m + SMALL:
        return pd.DataFrame(columns=stations.columns)
    interval = stations.loc[
        (stations["MD_m"] >= float(start_md_m) - SMALL)
        & (stations["MD_m"] <= float(end_md_m) + SMALL)
    ].copy()
    if (
        interval.empty
        or abs(float(interval["MD_m"].iloc[0]) - float(start_md_m)) > SMALL
    ):
        interval = pd.concat(
            [pd.DataFrame([_interpolate_row(stations, float(start_md_m))]), interval],
            ignore_index=True,
        )
    if abs(float(interval["MD_m"].iloc[-1]) - float(end_md_m)) > SMALL:
        interval = pd.concat(
            [interval, pd.DataFrame([_interpolate_row(stations, float(end_md_m))])],
            ignore_index=True,
        )
    interval = interval.sort_values("MD_m").reset_index(drop=True)
    return interval


def _series_stats(series: pd.Series) -> tuple[float | None, float | None, float | None]:
    values = series.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None, None, None
    return float(np.min(finite)), float(np.max(finite)), float(np.mean(finite))


def _circular_median_deg(values_deg: np.ndarray) -> float | None:
    if len(values_deg) == 0:
        return None
    radians = np.deg2rad(values_deg % 360.0)
    mean_angle = float(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float(np.rad2deg(mean_angle) % 360.0)


def _median_or_none(values: Iterable[float | None]) -> float | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    if len(finite) == 0:
        return None
    return float(np.median(finite))


def _max_or_none(values: Iterable[float | None]) -> float | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    if len(finite) == 0:
        return None
    return float(np.max(finite))


def _analysis_exclusion_reason(
    *,
    is_horizontal: bool,
    kop_md_m: float | None,
    horizontal_entry_md_m: float | None,
    horizontal_entry_tvd_m: float | None,
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
    if (
        horizontal_entry_md_m is None
        or horizontal_entry_tvd_m is None
        or float(horizontal_length_m) < HORIZONTAL_MIN_INTERVAL_M
    ):
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
        and float(max_build_dls_before_hold_deg_per_30m)
        > MAX_REASONABLE_ACTUAL_DLS_DEG_PER_30M
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
    azi_deg = wrap_azimuth_deg(
        np.degrees(np.arctan2(direction_grid[:, 0], direction_grid[:, 1]))
    )

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


def _station_direction_vectors(
    md_values: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    delta_md = np.diff(md_values)
    segment_vectors = np.diff(positions, axis=0)
    segment_norms = np.linalg.norm(segment_vectors, axis=1)
    segment_directions = np.zeros_like(segment_vectors, dtype=float)
    valid = segment_norms > SMALL
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
            elif np.linalg.norm(segment_directions[index]) <= SMALL:
                segment_directions[index] = next_valid
    else:
        segment_directions[:] = np.array([0.0, 0.0, 1.0], dtype=float)
    station_directions = np.zeros((len(md_values), 3), dtype=float)
    station_directions[0] = segment_directions[0]
    station_directions[-1] = segment_directions[-1]
    for index in range(1, len(md_values) - 1):
        left_weight = float(delta_md[index - 1])
        right_weight = float(delta_md[index])
        weighted = (
            left_weight * segment_directions[index - 1]
            + right_weight * segment_directions[index]
        )
        norm = float(np.linalg.norm(weighted))
        if norm <= SMALL:
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
    if end_md <= start_md + SMALL:
        return np.asarray([start_md, end_md], dtype=float)
    grid = np.arange(start_md, end_md, float(step_m), dtype=float)
    if len(grid) == 0 or abs(grid[0] - start_md) > SMALL:
        grid = np.concatenate([[start_md], grid])
    if abs(grid[-1] - end_md) > SMALL:
        grid = np.concatenate([grid, [end_md]])
    return grid


def _mask_intervals(
    md_values: np.ndarray, mask: np.ndarray
) -> list[tuple[float, float, float]]:
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
    step_m = (
        float(np.median(np.diff(md_values)))
        if len(md_values) > 1
        else ACTUAL_FUND_RESAMPLE_STEP_M
    )
    window_size = max(3, int(round(ROBUST_DLS_WINDOW_M / max(step_m, 1.0))))
    smoothed = _rolling_median(
        np.where(finite_mask, dls_values, 0.0), window_size=window_size
    )
    finite_smoothed = smoothed[np.isfinite(smoothed)]
    if len(finite_smoothed) == 0:
        return None
    return float(np.max(finite_smoothed))


def _stable_inc_values(stations: pd.DataFrame) -> np.ndarray:
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if len(inc_values) == 0:
        return np.asarray([], dtype=float)
    step_m = (
        float(np.median(np.diff(md_values)))
        if len(md_values) > 1
        else ACTUAL_FUND_RESAMPLE_STEP_M
    )
    window_size = max(3, int(round(ROBUST_DLS_WINDOW_M / max(step_m, 1.0))))
    return _rolling_median(inc_values, window_size=window_size)


def _stable_inc_build_rate_deg_per_30m(
    *,
    stations: pd.DataFrame,
    stable_inc: np.ndarray | None = None,
) -> np.ndarray:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if len(md_values) == 0:
        return np.asarray([], dtype=float)
    if stable_inc is None:
        stable_inc = _stable_inc_values(stations)
    if len(md_values) == 1:
        return np.zeros(1, dtype=float)
    delta_md = np.diff(md_values)
    delta_inc = np.diff(np.asarray(stable_inc, dtype=float))
    segment_rate = np.zeros(len(delta_md), dtype=float)
    valid = delta_md > SMALL
    segment_rate[valid] = np.maximum(delta_inc[valid], 0.0) / delta_md[valid] * 30.0
    station_rate = np.zeros(len(md_values), dtype=float)
    station_rate[0] = segment_rate[0]
    station_rate[-1] = segment_rate[-1]
    for index in range(1, len(md_values) - 1):
        station_rate[index] = 0.5 * float(segment_rate[index - 1] + segment_rate[index])
    return station_rate


def _detect_hold_interval(
    *,
    stations: pd.DataFrame,
    kop_md_m: float | None,
    horizontal_entry_md_m: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    azi_values = stations["AZI_deg"].to_numpy(dtype=float)
    dls_values = stations["DLS_deg_per_30m"].to_numpy(dtype=float)

    analysis_end_md = (
        float(horizontal_entry_md_m)
        if horizontal_entry_md_m is not None
        else float(md_values[-1])
    )
    search_start_md = (
        float(kop_md_m) + 10.0 if kop_md_m is not None else float(md_values[0]) + 10.0
    )
    stable_inc = _stable_inc_values(stations)
    candidate_mask = (
        (md_values >= search_start_md)
        & (md_values <= analysis_end_md - 10.0)
        & np.isfinite(stable_inc)
        & np.isfinite(dls_values)
        & (stable_inc >= HOLD_MIN_INC_DEG)
        & (stable_inc <= HOLD_MAX_INC_DEG)
    )
    if not np.any(candidate_mask):
        return None, None, None, None

    step_m = (
        float(np.median(np.diff(md_values)))
        if len(md_values) > 1
        else ACTUAL_FUND_RESAMPLE_STEP_M
    )
    window_size = max(3, int(round(ROBUST_DLS_WINDOW_M / max(step_m, 1.0))))
    stable_dls = _rolling_median(
        np.where(np.isfinite(dls_values), dls_values, 0.0), window_size=window_size
    )
    weighted_mode_inc = _weighted_modal_inclination(
        md_values=md_values,
        inc_values=stable_inc,
        mask=candidate_mask & (stable_dls <= HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M),
    )
    if weighted_mode_inc is None:
        return None, None, None, None

    hold_mask = (
        candidate_mask
        & (np.abs(stable_inc - float(weighted_mode_inc)) <= HOLD_INC_TOLERANCE_DEG)
        & (stable_dls <= HOLD_STABLE_DLS_THRESHOLD_DEG_PER_30M)
    )
    intervals = [
        (start_md, end_md, length_m)
        for start_md, end_md, length_m in _mask_intervals(md_values, hold_mask)
        if length_m >= HOLD_MIN_INTERVAL_M
    ]
    if not intervals:
        return None, None, None, None
    start_md, end_md, _ = max(intervals, key=lambda item: (item[2], -item[0]))
    interval_mask = (md_values >= start_md - SMALL) & (md_values <= end_md + SMALL)
    hold_inc = float(np.median(stable_inc[interval_mask]))
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
