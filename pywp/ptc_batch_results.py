from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from pywp.coordinate_integration import (
    DEFAULT_CRS,
    csv_export_crs,
    get_crs_display_suffix,
    transform_stations_to_crs,
)
from pywp.coordinate_systems import CoordinateSystem
from pywp.ui_utils import dls_to_pi
from pywp.ui_well_panels import survey_export_dataframe
from pywp.welltrack_batch import SuccessfulWellPlan

__all__ = [
    "BATCH_SUMMARY_DISPLAY_ORDER",
    "BATCH_SUMMARY_RENAME_COLUMNS",
    "BatchSummaryStatusCounts",
    "batch_summary_display_df",
    "batch_summary_status_counts",
    "build_batch_survey_csv",
    "find_selected_success",
    "has_md_postcheck_warning",
]

BATCH_SUMMARY_RENAME_COLUMNS: dict[str, str] = {
    "Рестарты решателя": "Рестарты",
    "Классификация целей": "Цели",
    "Длина ГС, м": "ГС, м",
}
BATCH_SUMMARY_DISPLAY_ORDER: tuple[str, ...] = (
    "Скважина",
    "Точек",
    "Цели",
    "Сложность",
    "Отход t1, м",
    "Мин VERTICAL до KOP, м",
    "KOP MD, м",
    "ГС, м",
    "INC в t1, deg",
    "ЗУ HOLD, deg",
    "Макс ПИ, deg/10m",
    "Макс MD, м",
    "Рестарты",
    "Статус",
    "Проблема",
    "Модель траектории",
)

SurveyExportFrameFunc = Callable[..., pd.DataFrame]
DlsToPiFunc = Callable[[object], object]


@dataclass(frozen=True)
class BatchSummaryStatusCounts:
    ok_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    not_run_count: int = 0


def build_batch_survey_csv(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    csv_export_crs_func: Callable[..., CoordinateSystem] = csv_export_crs,
    transform_stations_func: Callable[..., pd.DataFrame] = transform_stations_to_crs,
    crs_display_suffix_func: Callable[[CoordinateSystem], str] = get_crs_display_suffix,
    survey_export_dataframe_func: SurveyExportFrameFunc = survey_export_dataframe,
    dls_to_pi_func: DlsToPiFunc = dls_to_pi,
) -> bytes:
    if not successes:
        return b""
    export_crs = csv_export_crs_func(
        target_crs,
        source_crs,
        auto_convert=auto_convert,
    )
    should_transform = (
        bool(auto_convert) and target_crs != source_crs and export_crs == target_crs
    )
    should_label_export_crs = bool(auto_convert) and target_crs != source_crs
    frames: list[pd.DataFrame] = []
    for success in successes:
        stations = success.stations.copy()
        if stations.empty:
            continue
        if should_transform:
            stations = transform_stations_func(
                stations,
                target_crs,
                source_crs,
                rename_columns=False,
            )
        stations.insert(0, "well_name", str(success.name))
        if "DLS_deg_per_30m" in stations.columns:
            stations["PI_deg_per_10m"] = dls_to_pi_func(
                stations["DLS_deg_per_30m"].to_numpy(dtype=float)
            )
            stations = stations.drop(columns=["DLS_deg_per_30m"])
        if should_label_export_crs:
            stations = survey_export_dataframe_func(
                stations,
                xy_label_suffix=crs_display_suffix_func(export_crs),
                xy_unit="deg" if export_crs.is_geographic() else "м",
            )
        frames.append(stations)
    if not frames:
        return b""
    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False).encode("utf-8")


def batch_summary_display_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    display_df = summary_df.rename(columns=BATCH_SUMMARY_RENAME_COLUMNS).copy()
    ordered = [
        column for column in BATCH_SUMMARY_DISPLAY_ORDER if column in display_df.columns
    ]
    trailing = [column for column in display_df.columns if column not in ordered]
    return display_df[ordered + trailing]


def batch_summary_status_counts(
    summary_df: pd.DataFrame,
) -> BatchSummaryStatusCounts:
    if summary_df.empty or not {"Статус", "Проблема"}.issubset(summary_df.columns):
        return BatchSummaryStatusCounts()
    ok_count = 0
    warning_count = 0
    error_count = 0
    not_run_count = 0
    for _, row in summary_df.iterrows():
        status = str(row["Статус"]).strip()
        if status == "OK":
            if _has_problem_text(row["Проблема"]):
                warning_count += 1
            else:
                ok_count += 1
            continue
        if status == "Не рассчитана":
            not_run_count += 1
        else:
            error_count += 1
    return BatchSummaryStatusCounts(
        ok_count=ok_count,
        warning_count=warning_count,
        error_count=error_count,
        not_run_count=not_run_count,
    )


def has_md_postcheck_warning(summary_df: pd.DataFrame) -> bool:
    if summary_df.empty or "Проблема" not in summary_df.columns:
        return False
    return bool(
        summary_df["Проблема"]
        .astype(str)
        .str.contains("Превышен лимит итоговой MD", regex=False)
        .any()
    )


def _has_problem_text(value: object) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text and text not in {"—", "ОК", "OK", "nan", "NaN", "None", "<NA>"})


def find_selected_success(
    *,
    selected_name: str,
    successes: list[SuccessfulWellPlan],
) -> SuccessfulWellPlan:
    return next(item for item in successes if str(item.name) == str(selected_name))
