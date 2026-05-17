from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import py7zr

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
    "build_batch_survey_dev_7z",
    "build_batch_survey_dev_file",
    "build_batch_survey_dev_zip",
    "dev_export_file_name",
    "build_batch_survey_welltrack",
    "find_selected_success",
    "has_md_postcheck_warning",
    "pilot_sidetrack_summary_df",
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
_SURVEY_EXPORT_REQUIRED_COLUMNS = ("MD_m", "X_m", "Y_m", "Z_m")


@dataclass(frozen=True)
class BatchSummaryStatusCounts:
    ok_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    not_run_count: int = 0


@dataclass(frozen=True)
class _SurveyExportContext:
    export_crs: CoordinateSystem
    should_transform: bool
    should_label_export_crs: bool


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
    export_context = _survey_export_context(
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs_func,
    )
    frames: list[pd.DataFrame] = []
    for success in successes:
        stations = success.stations.copy()
        if stations.empty:
            continue
        if export_context.should_transform:
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
        if export_context.should_label_export_crs:
            stations = survey_export_dataframe_func(
                stations,
                xy_label_suffix=crs_display_suffix_func(export_context.export_crs),
                xy_unit="deg" if export_context.export_crs.is_geographic() else "м",
            )
        frames.append(stations)
    if not frames:
        return b""
    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False, sep="\t").encode("utf-8")


def build_batch_survey_welltrack(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    csv_export_crs_func: Callable[..., CoordinateSystem] = csv_export_crs,
    transform_stations_func: Callable[..., pd.DataFrame] = transform_stations_to_crs,
) -> bytes:
    blocks: list[str] = []
    for success, stations in _iter_prepared_success_stations(
        successes,
        target_crs=source_crs,
        auto_convert=False,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs_func,
        transform_stations_func=transform_stations_func,
    ):
        lines = [f"WELLTRACK {_welltrack_name_literal(str(success.name))}"]
        for _, row in stations.iterrows():
            lines.append(
                " ".join(
                    [
                        _format_export_number(row["X_m"]),
                        _format_export_number(row["Y_m"]),
                        _format_export_number(row["Z_m"]),
                        _format_export_number(row["MD_m"]),
                    ]
                )
            )
        lines.append("/")
        blocks.append("\n".join(lines))
    if not blocks:
        return b""
    return ("\n\n".join(blocks) + "\n").encode("utf-8")


def build_batch_survey_dev_7z(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    csv_export_crs_func: Callable[..., CoordinateSystem] = csv_export_crs,
    transform_stations_func: Callable[..., pd.DataFrame] = transform_stations_to_crs,
) -> bytes:
    buffer = BytesIO()
    used_names: set[str] = set()
    file_count = 0
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        with py7zr.SevenZipFile(buffer, mode="w") as archive:
            for index, (success, stations) in enumerate(
                _iter_prepared_success_stations(
                    successes,
                    target_crs=source_crs,
                    auto_convert=False,
                    source_crs=source_crs,
                    csv_export_crs_func=csv_export_crs_func,
                    transform_stations_func=transform_stations_func,
                ),
                start=1,
            ):
                if len(stations.index) < 2:
                    continue
                file_name = _unique_export_file_name(
                    str(success.name),
                    index=index,
                    extension=".dev",
                    used_names=used_names,
                )
                source_path = temp_root / file_name
                source_path.write_text(
                    _dev_export_text(success=success, stations=stations),
                    encoding="utf-8",
                )
                archive.write(source_path, file_name)
                file_count += 1
    if file_count <= 0:
        return b""
    return buffer.getvalue()


def build_batch_survey_dev_zip(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    csv_export_crs_func: Callable[..., CoordinateSystem] = csv_export_crs,
    transform_stations_func: Callable[..., pd.DataFrame] = transform_stations_to_crs,
) -> bytes:
    return build_batch_survey_dev_7z(
        successes,
        target_crs=source_crs,
        auto_convert=False,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs_func,
        transform_stations_func=transform_stations_func,
    )


def build_batch_survey_dev_file(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    csv_export_crs_func: Callable[..., CoordinateSystem] = csv_export_crs,
    transform_stations_func: Callable[..., pd.DataFrame] = transform_stations_to_crs,
) -> bytes:
    prepared = _iter_prepared_success_stations(
        successes,
        target_crs=source_crs,
        auto_convert=False,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs_func,
        transform_stations_func=transform_stations_func,
    )
    if len(prepared) != 1:
        return b""
    success, stations = prepared[0]
    if len(stations.index) < 2:
        return b""
    return _dev_export_text(success=success, stations=stations).encode("utf-8")


def dev_export_file_name(name: str, *, fallback_index: int = 1) -> str:
    used_names: set[str] = set()
    return _unique_export_file_name(
        str(name),
        index=int(fallback_index),
        extension=".dev",
        used_names=used_names,
    )


def _iter_prepared_success_stations(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem,
    auto_convert: bool,
    source_crs: CoordinateSystem,
    csv_export_crs_func: Callable[..., CoordinateSystem],
    transform_stations_func: Callable[..., pd.DataFrame],
) -> list[tuple[SuccessfulWellPlan, pd.DataFrame]]:
    if not successes:
        return []
    export_context = _survey_export_context(
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs_func,
    )
    prepared: list[tuple[SuccessfulWellPlan, pd.DataFrame]] = []
    for success in successes:
        stations = success.stations.copy()
        if stations.empty or not set(_SURVEY_EXPORT_REQUIRED_COLUMNS).issubset(
            stations.columns
        ):
            continue
        stations = _sanitize_export_stations(stations)
        if stations.empty:
            continue
        if export_context.should_transform:
            stations = transform_stations_func(
                stations,
                target_crs,
                source_crs,
                rename_columns=False,
            )
            stations = _sanitize_export_stations(stations)
            if stations.empty:
                continue
        prepared.append((success, stations))
    return prepared


def _sanitize_export_stations(stations: pd.DataFrame) -> pd.DataFrame:
    result = stations.copy()
    for column in _SURVEY_EXPORT_REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    finite_mask = pd.Series(True, index=result.index)
    for column in _SURVEY_EXPORT_REQUIRED_COLUMNS:
        finite_mask &= result[column].map(math.isfinite)
    result = result.loc[finite_mask].copy()
    if result.empty:
        return result
    result = result.sort_values("MD_m", kind="mergesort")
    result = result.drop_duplicates(subset=["MD_m"], keep="first")
    return result.reset_index(drop=True)


def _survey_export_context(
    *,
    target_crs: CoordinateSystem,
    auto_convert: bool,
    source_crs: CoordinateSystem,
    csv_export_crs_func: Callable[..., CoordinateSystem],
) -> _SurveyExportContext:
    export_crs = csv_export_crs_func(
        target_crs,
        source_crs,
        auto_convert=auto_convert,
    )
    should_transform = (
        bool(auto_convert) and target_crs != source_crs and export_crs == target_crs
    )
    return _SurveyExportContext(
        export_crs=export_crs,
        should_transform=should_transform,
        should_label_export_crs=bool(auto_convert) and target_crs != source_crs,
    )


def _dev_export_text(
    *,
    success: SuccessfulWellPlan,
    stations: pd.DataFrame,
) -> str:
    if stations.empty:
        return ""
    surface_x = float(stations["X_m"].iloc[0])
    surface_y = float(stations["Y_m"].iloc[0])
    surface_z = float(stations["Z_m"].iloc[0])
    lines = [
        "# SURVEY FROM PYWP",
        f"# WELL NAME:                {success.name}",
        "# TRAJECTORY TYPE:          SURVEY",
        f"# WELL HEAD X-COORDINATE:   {_format_export_number(surface_x)}",
        f"# WELL HEAD Y-COORDINATE:   {_format_export_number(surface_y)}",
        "# MD AND TVD ARE REFERENCED (=0) AT WELL DATUM AND INCREASE DOWNWARDS",
        "# ANGLES ARE GIVEN IN DEGREES",
        "#==============================================================================================================================================",
        "    MD            X            Y            Z           TVD           DX           DY         AZIM_TN        INCL        DLS        AZIM_GN",
        "#==============================================================================================================================================",
    ]
    for _, row in stations.iterrows():
        x_value = _station_float(row, "X_m")
        y_value = _station_float(row, "Y_m")
        z_value = _station_float(row, "Z_m")
        dev_z_value = -z_value
        md_value = _station_float(row, "MD_m")
        azi_value = _station_float(row, "AZI_deg")
        inc_value = _station_float(row, "INC_deg")
        dls_value = _station_float(row, "DLS_deg_per_30m")
        lines.append(
            " ".join(
                [
                    _format_export_number(md_value),
                    _format_export_number(x_value),
                    _format_export_number(y_value),
                    _format_export_number(dev_z_value),
                    _format_export_number(z_value - surface_z),
                    _format_export_number(x_value - surface_x),
                    _format_export_number(y_value - surface_y),
                    _format_export_number(azi_value),
                    _format_export_number(inc_value),
                    _format_export_number(dls_value),
                    _format_export_number(azi_value),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _station_float(row: pd.Series, column: str) -> float:
    if column not in row:
        return 0.0
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def _welltrack_name_literal(name: str) -> str:
    text = str(name).strip() or "WELL"
    if "'" not in text:
        return f"'{text}'"
    if '"' not in text:
        return f'"{text}"'
    return f"'{text.replace(chr(39), '_').replace(chr(34), '_')}'"


def _format_export_number(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = 0.0
    if not math.isfinite(numeric_value):
        numeric_value = 0.0
    if abs(numeric_value) < 5e-10:
        numeric_value = 0.0
    return f"{numeric_value:.6f}"


def _unique_export_file_name(
    name: str,
    *,
    index: int,
    extension: str,
    used_names: set[str],
) -> str:
    stem = _safe_export_stem(name) or f"well_{int(index):03d}"
    candidate = f"{stem}{extension}"
    suffix = 2
    while candidate.casefold() in used_names:
        candidate = f"{stem}_{suffix}{extension}"
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate


def _safe_export_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")


def batch_summary_display_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    display_df = summary_df.rename(columns=BATCH_SUMMARY_RENAME_COLUMNS).copy()
    ordered = [
        column for column in BATCH_SUMMARY_DISPLAY_ORDER if column in display_df.columns
    ]
    trailing = [column for column in display_df.columns if column not in ordered]
    return display_df[ordered + trailing]


def pilot_sidetrack_summary_df(
    successes: list[SuccessfulWellPlan],
) -> pd.DataFrame:
    success_by_name = {str(success.name): success for success in successes}
    rows: list[dict[str, object]] = []
    for success in successes:
        summary = dict(success.summary)
        if str(summary.get("trajectory_type", "")).strip() != "PILOT_SIDETRACK":
            continue
        pilot_name = str(summary.get("pilot_well_name", "")).strip()
        pilot_success = success_by_name.get(pilot_name)
        pilot_summary = dict(pilot_success.summary) if pilot_success is not None else {}
        pilot_target_count = _summary_float(
            pilot_summary.get("pilot_target_count"),
            fallback=_pilot_segment_count(pilot_success),
        )
        rows.append(
            {
                "Скважина": str(success.name),
                "Пилот": pilot_name or "—",
                "Плановых точек пилота": _optional_int_text(pilot_target_count),
                "BUILD+HOLD до точек пилота": _optional_int_text(pilot_target_count),
                "Окно MD, м": _summary_float(summary.get("sidetrack_window_md_m")),
                "Окно Z, м": _summary_float(summary.get("sidetrack_window_z_m")),
                "Окно INC, deg": _summary_float(
                    summary.get("sidetrack_window_inc_deg")
                ),
                "Окно AZI, deg": _summary_float(
                    summary.get("sidetrack_window_azi_deg")
                ),
                "MD пилота, м": _summary_float(pilot_summary.get("md_total_m")),
                "MD бокового ствола, м": _summary_float(
                    summary.get("sidetrack_lateral_md_m")
                ),
                "Макс ПИ пилота, deg/10m": dls_to_pi(
                    _summary_float(pilot_summary.get("max_dls_total_deg_per_30m"))
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Скважина",
            "Пилот",
            "Плановых точек пилота",
            "BUILD+HOLD до точек пилота",
            "Окно MD, м",
            "Окно Z, м",
            "Окно INC, deg",
            "Окно AZI, deg",
            "MD пилота, м",
            "MD бокового ствола, м",
            "Макс ПИ пилота, deg/10m",
        ],
    )


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


def _pilot_segment_count(success: SuccessfulWellPlan | None) -> float:
    if success is None or success.stations.empty or "segment" not in success.stations:
        return float("nan")
    names = {
        str(value).strip()
        for value in success.stations["segment"].dropna().tolist()
        if str(value).strip().startswith("PILOT")
    }
    return float(len(names)) if names else float("nan")


def _summary_float(value: object, *, fallback: float = float("nan")) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric_value):
        return fallback
    return numeric_value


def _optional_int_text(value: object) -> str:
    numeric_value = _summary_float(value)
    if not math.isfinite(numeric_value):
        return "—"
    return str(int(round(numeric_value)))


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
