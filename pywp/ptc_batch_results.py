from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import py7zr

from pywp import ptc_target_records
from pywp.coordinate_integration import (
    DEFAULT_CRS,
    csv_export_crs,
    get_crs_display_suffix,
    transform_stations_to_crs,
    transform_xy_to_crs,
)
from pywp.coordinate_systems import CoordinateSystem
from pywp.eclipse_welltrack import WelltrackRecord
from pywp.mcm import dogleg_angle_rad
from pywp.reference_trajectories import ImportedTrajectoryWell
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
    "build_batch_target_csv",
    "build_batch_target_dev_7z",
    "build_batch_target_dev_file",
    "build_batch_target_welltrack",
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
_DEV_EXPORT_COLUMNS = (
    "MD",
    "X",
    "Y",
    "Z",
    "TVD",
    "DX",
    "DY",
    "AZIM_TN",
    "INCL",
    "DLS",
    "AZIM_GN",
)
_DEV_EXPORT_REQUIRED_COLUMNS = ("MD", "X", "Y", "Z", "TVD", "DX", "DY")
_DEV_EXPORT_ANGLE_COLUMNS = ("AZIM_TN", "INCL", "DLS", "AZIM_GN")
TransformXyFunc = Callable[
    [float, float, CoordinateSystem, CoordinateSystem],
    tuple[float, float],
]


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


def build_batch_target_csv(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    csv_export_crs_func: Callable[..., CoordinateSystem] = csv_export_crs,
    transform_xy_func: TransformXyFunc = transform_xy_to_crs,
    crs_display_suffix_func: Callable[[CoordinateSystem], str] = get_crs_display_suffix,
    survey_export_dataframe_func: SurveyExportFrameFunc = survey_export_dataframe,
) -> bytes:
    prepared = _iter_prepared_target_rows(records)
    if not prepared:
        return b""

    export_context = _survey_export_context(
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs_func,
    )
    frames: list[pd.DataFrame] = []
    for record, rows in prepared:
        identity_df = pd.DataFrame(
            {
                "well_name": [str(record.name)] * len(rows.index),
                "point_name": rows["point_name"].astype(str),
                "point_md_m": rows["point_md_m"].astype(float),
            }
        )
        if export_context.should_transform:
            source_block = _target_coordinate_block(
                rows[["X_m", "Y_m", "Z_m"]],
                xy_label_suffix=crs_display_suffix_func(source_crs),
                xy_unit="deg" if source_crs.is_geographic() else "м",
                z_column_name="Z_input_m",
                survey_export_dataframe_func=survey_export_dataframe_func,
            )
            export_rows = rows.copy()
            transformed_xy = [
                transform_xy_func(
                    float(x_value),
                    float(y_value),
                    source_crs,
                    target_crs,
                )
                for x_value, y_value in zip(
                    export_rows["X_m"],
                    export_rows["Y_m"],
                    strict=False,
                )
            ]
            export_rows["X_m"] = [point[0] for point in transformed_xy]
            export_rows["Y_m"] = [point[1] for point in transformed_xy]
            export_block = _target_coordinate_block(
                export_rows[["X_m", "Y_m", "Z_m"]],
                xy_label_suffix=crs_display_suffix_func(export_context.export_crs),
                xy_unit="deg" if export_context.export_crs.is_geographic() else "м",
                z_column_name="Z_output_m",
                survey_export_dataframe_func=survey_export_dataframe_func,
            )
            frames.append(
                pd.concat(
                    [
                        identity_df.reset_index(drop=True),
                        source_block.reset_index(drop=True),
                        export_block.reset_index(drop=True),
                    ],
                    axis=1,
                )
            )
            continue

        source_block = _target_coordinate_block(
            rows[["X_m", "Y_m", "Z_m"]],
            xy_label_suffix="",
            xy_unit="deg" if source_crs.is_geographic() else "м",
            z_column_name="Z_m",
            survey_export_dataframe_func=survey_export_dataframe_func,
        )
        frames.append(
            pd.concat(
                [
                    identity_df.reset_index(drop=True),
                    source_block.reset_index(drop=True),
                ],
                axis=1,
            )
        )
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


def build_batch_target_welltrack(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    del target_crs, auto_convert, source_crs
    blocks: list[str] = []
    for record, rows in _iter_prepared_target_rows(records):
        lines = [f"WELLTRACK {_welltrack_name_literal(str(record.name))}"]
        for _, row in rows.iterrows():
            lines.append(
                " ".join(
                    [
                        _format_export_number(row["X_m"]),
                        _format_export_number(row["Y_m"]),
                        _format_export_number(row["Z_m"]),
                        _format_export_number(row["point_md_m"]),
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
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
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
                    _dev_export_text(
                        success=success,
                        stations=stations,
                        reference_wells=reference_wells,
                    ),
                    encoding="utf-8",
                )
                archive.write(source_path, file_name)
                file_count += 1
    if file_count <= 0:
        return b""
    return buffer.getvalue()


def build_batch_target_dev_7z(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    del target_crs, auto_convert, source_crs
    buffer = BytesIO()
    used_names: set[str] = set()
    file_count = 0
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        with py7zr.SevenZipFile(buffer, mode="w") as archive:
            for index, (record, rows) in enumerate(
                _iter_prepared_target_rows(records),
                start=1,
            ):
                if rows.empty:
                    continue
                file_name = _unique_export_file_name(
                    str(record.name),
                    index=index,
                    extension=".dev",
                    used_names=used_names,
                )
                source_path = temp_root / file_name
                source_path.write_text(
                    _target_dev_export_text(record=record, rows=rows),
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
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    csv_export_crs_func: Callable[..., CoordinateSystem] = csv_export_crs,
    transform_stations_func: Callable[..., pd.DataFrame] = transform_stations_to_crs,
) -> bytes:
    return build_batch_survey_dev_7z(
        successes,
        target_crs=source_crs,
        auto_convert=False,
        source_crs=source_crs,
        reference_wells=reference_wells,
        csv_export_crs_func=csv_export_crs_func,
        transform_stations_func=transform_stations_func,
    )


def build_batch_survey_dev_file(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
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
    return _dev_export_text(
        success=success,
        stations=stations,
        reference_wells=reference_wells,
    ).encode("utf-8")


def build_batch_target_dev_file(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    del target_crs, auto_convert, source_crs
    prepared = _iter_prepared_target_rows(records)
    if len(prepared) != 1:
        return b""
    record, rows = prepared[0]
    if rows.empty:
        return b""
    return _target_dev_export_text(record=record, rows=rows).encode("utf-8")


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


def _iter_prepared_target_rows(
    records: list[WelltrackRecord],
) -> list[tuple[WelltrackRecord, pd.DataFrame]]:
    prepared: list[tuple[WelltrackRecord, pd.DataFrame]] = []
    for record in records:
        rows = ptc_target_records.raw_records_dataframe([record]).rename(
            columns={
                "Скважина": "well_name",
                "Точка": "point_name",
                "X, м": "X_m",
                "Y, м": "Y_m",
                "Z, м": "Z_m",
            }
        )
        if rows.empty:
            continue
        rows.insert(
            2,
            "point_md_m",
            [float(point.md) for point in tuple(record.points)],
        )
        rows = _sanitize_target_rows(rows)
        if rows.empty:
            continue
        prepared.append((record, rows))
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


def _sanitize_target_rows(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    required_columns = ("point_md_m", "X_m", "Y_m", "Z_m")
    for column in required_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    finite_mask = pd.Series(True, index=result.index)
    for column in required_columns:
        finite_mask &= result[column].map(math.isfinite)
    result = result.loc[finite_mask].copy()
    if result.empty:
        return result
    result["well_name"] = result["well_name"].astype(str)
    result["point_name"] = result["point_name"].astype(str)
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


def _target_coordinate_block(
    rows: pd.DataFrame,
    *,
    xy_label_suffix: str,
    xy_unit: str,
    z_column_name: str,
    survey_export_dataframe_func: SurveyExportFrameFunc,
) -> pd.DataFrame:
    block = survey_export_dataframe_func(
        rows.copy(),
        xy_label_suffix=xy_label_suffix,
        xy_unit=xy_unit,
    )
    return block.rename(columns={"Z_m": z_column_name})


def _dev_export_text(
    *,
    success: SuccessfulWellPlan,
    stations: pd.DataFrame,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
) -> str:
    rows = _dev_export_rows(
        success=success,
        stations=stations,
        reference_wells=reference_wells,
    )
    if rows.empty:
        return ""
    surface_x = float(rows["X"].iloc[0])
    surface_y = float(rows["Y"].iloc[0])
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
    for _, row in rows.iterrows():
        lines.append(
            " ".join(
                [
                    _format_export_number(row["MD"]),
                    _format_export_number(row["X"]),
                    _format_export_number(row["Y"]),
                    _format_export_number(row["Z"]),
                    _format_export_number(row["TVD"]),
                    _format_export_number(row["DX"]),
                    _format_export_number(row["DY"]),
                    _format_export_number(row["AZIM_TN"]),
                    _format_export_number(row["INCL"]),
                    _format_export_number(row["DLS"]),
                    _format_export_number(row["AZIM_GN"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _dev_export_rows(
    *,
    success: SuccessfulWellPlan,
    stations: pd.DataFrame,
    reference_wells: tuple[ImportedTrajectoryWell, ...],
) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame(columns=_DEV_EXPORT_COLUMNS)
    sidetrack_rows = _sidetrack_dev_export_rows(
        success=success,
        stations=stations,
        reference_wells=reference_wells,
    )
    if sidetrack_rows is not None and not sidetrack_rows.empty:
        return sidetrack_rows
    return _dev_rows_from_stations(stations)


def _sidetrack_dev_export_rows(
    *,
    success: SuccessfulWellPlan,
    stations: pd.DataFrame,
    reference_wells: tuple[ImportedTrajectoryWell, ...],
) -> pd.DataFrame | None:
    summary = dict(getattr(success, "summary", {}) or {})
    window_md = _summary_optional_float(summary, "sidetrack_window_md_m")
    if window_md is None:
        return None
    prefix_rows = _sidetrack_parent_prefix_rows(
        success=success,
        window_md=window_md,
        summary=summary,
        reference_wells=reference_wells,
    )
    if prefix_rows is None or prefix_rows.empty:
        return None
    surface_override = (
        float(prefix_rows["X"].iloc[0]),
        float(prefix_rows["Y"].iloc[0]),
        -float(prefix_rows["Z"].iloc[0]),
    )
    branch_rows = _dev_rows_from_stations(stations, surface_xyz=surface_override)
    branch_rows = branch_rows.loc[
        branch_rows["MD"].to_numpy(dtype=float) > float(window_md) + 1e-6
    ].copy()
    combined = pd.concat([prefix_rows, branch_rows], ignore_index=True)
    return _sanitize_dev_export_rows(combined)


def _sidetrack_parent_prefix_rows(
    *,
    success: SuccessfulWellPlan,
    window_md: float,
    summary: dict[str, object],
    reference_wells: tuple[ImportedTrajectoryWell, ...],
) -> pd.DataFrame | None:
    parent_well = _sidetrack_parent_reference_well(
        summary=summary,
        reference_wells=reference_wells,
    )
    parent_stations = (
        parent_well.stations.copy()
        if parent_well is not None
        else _sidetrack_parent_attr_stations(success)
    )
    if parent_stations is None or parent_stations.empty:
        return None
    prefix_stations = _survey_prefix_stations(
        parent_stations,
        window_md=window_md,
        window_x_m=_summary_optional_float(summary, "sidetrack_window_x_m"),
        window_y_m=_summary_optional_float(summary, "sidetrack_window_y_m"),
        window_z_m=_summary_optional_float(summary, "sidetrack_window_z_m"),
        window_inc_deg=_summary_optional_float(summary, "sidetrack_window_inc_deg"),
        window_azi_deg=_summary_optional_float(summary, "sidetrack_window_azi_deg"),
    )
    if prefix_stations.empty:
        return None
    prefix_rows = _dev_rows_from_stations(prefix_stations)
    if parent_well is not None and parent_well.dev_export_rows is not None:
        prefix_rows = _overlay_original_dev_rows(
            derived_rows=prefix_rows,
            original_rows=parent_well.dev_export_rows,
            window_md=window_md,
        )
    return prefix_rows


def _sidetrack_parent_reference_well(
    *,
    summary: dict[str, object],
    reference_wells: tuple[ImportedTrajectoryWell, ...],
) -> ImportedTrajectoryWell | None:
    parent_name = (
        str(summary.get("actual_parent_well_name", "")).strip()
        or str(summary.get("sidetrack_parent_well_name", "")).strip()
        or str(summary.get("pilot_well_name", "")).strip()
    )
    if not parent_name:
        return None
    by_name = {
        _well_name_key(well.name): well
        for well in reference_wells
    }
    return by_name.get(_well_name_key(parent_name))


def _sidetrack_parent_attr_stations(success: SuccessfulWellPlan) -> pd.DataFrame | None:
    reference = getattr(success.stations, "attrs", {}).get(
        "uncertainty_reference_stations"
    )
    if not isinstance(reference, pd.DataFrame):
        return None
    return reference.copy()


def _survey_prefix_stations(
    stations: pd.DataFrame,
    *,
    window_md: float,
    window_x_m: float | None,
    window_y_m: float | None,
    window_z_m: float | None,
    window_inc_deg: float | None,
    window_azi_deg: float | None,
) -> pd.DataFrame:
    prepared = _sanitize_export_stations(stations)
    if prepared.empty:
        return prepared
    clamped_md = float(window_md)
    min_md = float(prepared["MD_m"].iloc[0])
    max_md = float(prepared["MD_m"].iloc[-1])
    if clamped_md < min_md:
        clamped_md = min_md
    elif (
        clamped_md > max_md
        and window_x_m is None
        and window_y_m is None
        and window_z_m is None
        and window_inc_deg is None
        and window_azi_deg is None
    ):
        clamped_md = max_md
    prefix = prepared.loc[
        prepared["MD_m"].to_numpy(dtype=float) <= clamped_md + 1e-6
    ].copy()
    if prefix.empty:
        return prefix
    if abs(float(prefix["MD_m"].iloc[-1]) - clamped_md) > 1e-6:
        prefix = pd.concat(
            [
                prefix,
                pd.DataFrame(
                    [
                        _interpolate_station_row(
                            prepared,
                            md_m=clamped_md,
                            x_m=window_x_m,
                            y_m=window_y_m,
                            z_m=window_z_m,
                            inc_deg=window_inc_deg,
                            azi_deg=window_azi_deg,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
    return _sanitize_prefix_stations(prefix)


def _sanitize_prefix_stations(stations: pd.DataFrame) -> pd.DataFrame:
    result = stations.copy()
    if "segment" in result.columns:
        result["segment"] = result["segment"].astype(str)
    result = result.sort_values("MD_m", kind="mergesort")
    result = result.drop_duplicates(subset=["MD_m"], keep="last")
    return result.reset_index(drop=True)


def _interpolate_station_row(
    stations: pd.DataFrame,
    *,
    md_m: float,
    x_m: float | None,
    y_m: float | None,
    z_m: float | None,
    inc_deg: float | None,
    azi_deg: float | None,
) -> dict[str, object]:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    result: dict[str, object] = {
        "MD_m": float(md_m),
        "X_m": (
            float(x_m)
            if x_m is not None
            else float(np.interp(float(md_m), md_values, stations["X_m"].to_numpy(dtype=float)))
        ),
        "Y_m": (
            float(y_m)
            if y_m is not None
            else float(np.interp(float(md_m), md_values, stations["Y_m"].to_numpy(dtype=float)))
        ),
        "Z_m": (
            float(z_m)
            if z_m is not None
            else float(np.interp(float(md_m), md_values, stations["Z_m"].to_numpy(dtype=float)))
        ),
        "INC_deg": (
            float(inc_deg)
            if inc_deg is not None
            else float(np.interp(float(md_m), md_values, stations["INC_deg"].to_numpy(dtype=float)))
        ),
        "AZI_deg": (
            float(azi_deg)
            if azi_deg is not None
            else _interp_azimuth_deg(
                md_values=md_values,
                azi_values_deg=stations["AZI_deg"].to_numpy(dtype=float),
                md_m=float(md_m),
            )
        ),
        "DLS_deg_per_30m": (
            float(
                np.interp(
                    float(md_m),
                    md_values,
                    stations["DLS_deg_per_30m"].to_numpy(dtype=float),
                )
            )
            if "DLS_deg_per_30m" in stations.columns
            else 0.0
        ),
    }
    if "segment" in stations.columns:
        result["segment"] = str(stations["segment"].iloc[-1])
    return result


def _interp_azimuth_deg(
    *,
    md_values: np.ndarray,
    azi_values_deg: np.ndarray,
    md_m: float,
) -> float:
    radians = np.deg2rad(azi_values_deg % 360.0)
    sin_value = float(np.interp(float(md_m), md_values, np.sin(radians)))
    cos_value = float(np.interp(float(md_m), md_values, np.cos(radians)))
    if abs(sin_value) < 1e-9 and abs(cos_value) < 1e-9:
        return float(azi_values_deg[-1] % 360.0)
    return float(np.degrees(np.arctan2(sin_value, cos_value)) % 360.0)


def _overlay_original_dev_rows(
    *,
    derived_rows: pd.DataFrame,
    original_rows: pd.DataFrame,
    window_md: float,
) -> pd.DataFrame:
    result = derived_rows.copy()
    source = _sanitize_dev_export_rows(original_rows)
    if source.empty:
        return result
    source = source.loc[source["MD"].to_numpy(dtype=float) <= float(window_md) + 1e-6]
    if source.empty:
        return result
    derived_md = result["MD"].to_numpy(dtype=float)
    for _, row in source.iterrows():
        matches = np.isclose(derived_md, float(row["MD"]), atol=1e-6)
        if not np.any(matches):
            continue
        result.loc[int(np.flatnonzero(matches)[0]), list(_DEV_EXPORT_COLUMNS)] = [
            row[column] for column in _DEV_EXPORT_COLUMNS
        ]
    return result


def _dev_rows_from_stations(
    stations: pd.DataFrame,
    *,
    surface_xyz: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame(columns=_DEV_EXPORT_COLUMNS)
    if surface_xyz is None:
        surface_x = float(stations["X_m"].iloc[0])
        surface_y = float(stations["Y_m"].iloc[0])
        surface_z = float(stations["Z_m"].iloc[0])
    else:
        surface_x, surface_y, surface_z = (
            float(surface_xyz[0]),
            float(surface_xyz[1]),
            float(surface_xyz[2]),
        )
    rows = pd.DataFrame(
        {
            "MD": stations["MD_m"].to_numpy(dtype=float),
            "X": stations["X_m"].to_numpy(dtype=float),
            "Y": stations["Y_m"].to_numpy(dtype=float),
            "Z": -stations["Z_m"].to_numpy(dtype=float),
            "TVD": stations["Z_m"].to_numpy(dtype=float) - float(surface_z),
            "DX": stations["X_m"].to_numpy(dtype=float) - float(surface_x),
            "DY": stations["Y_m"].to_numpy(dtype=float) - float(surface_y),
            "AZIM_TN": (
                stations["AZI_deg"].to_numpy(dtype=float)
                if "AZI_deg" in stations.columns
                else np.zeros(len(stations), dtype=float)
            ),
            "INCL": (
                stations["INC_deg"].to_numpy(dtype=float)
                if "INC_deg" in stations.columns
                else np.zeros(len(stations), dtype=float)
            ),
            "DLS": (
                stations["DLS_deg_per_30m"].to_numpy(dtype=float)
                if "DLS_deg_per_30m" in stations.columns
                else np.zeros(len(stations), dtype=float)
            ),
        }
    )
    rows["AZIM_GN"] = rows["AZIM_TN"].to_numpy(dtype=float)
    return _sanitize_dev_export_rows(rows)


def _sanitize_dev_export_rows(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    for column in _DEV_EXPORT_REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in _DEV_EXPORT_ANGLE_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.where(values.map(math.isfinite), 0.0)
        result[column] = values.fillna(0.0)
    finite_mask = pd.Series(True, index=result.index)
    for column in _DEV_EXPORT_REQUIRED_COLUMNS:
        finite_mask &= result[column].map(math.isfinite)
    result = result.loc[finite_mask].copy()
    if result.empty:
        return pd.DataFrame(columns=_DEV_EXPORT_COLUMNS)
    result = result.sort_values("MD", kind="mergesort")
    result = result.drop_duplicates(subset=["MD"], keep="first")
    return result.reset_index(drop=True)


def _well_name_key(name: object) -> str:
    return str(name or "").strip().casefold()


def _summary_optional_float(summary: dict[str, object], key: str) -> float | None:
    raw_value = summary.get(key)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _target_dev_export_text(
    *,
    record: WelltrackRecord,
    rows: pd.DataFrame,
) -> str:
    if rows.empty:
        return ""
    surface_x = float(rows["X_m"].iloc[0])
    surface_y = float(rows["Y_m"].iloc[0])
    surface_z = float(rows["Z_m"].iloc[0])
    lines = [
        "# TARGET POINTS FROM PYWP",
        f"# WELL NAME:                {record.name}",
        "# TRAJECTORY TYPE:          TARGET POINTS",
        "# MD IS GENERATED AS CUMULATIVE STRAIGHT-LINE DISTANCE THROUGH TARGETS",
        f"# START X-COORDINATE:       {_format_export_number(surface_x)}",
        f"# START Y-COORDINATE:       {_format_export_number(surface_y)}",
        "# MD AND TVD ARE REFERENCED (=0) AT THE FIRST TARGET POINT",
        "# ANGLES ARE GIVEN IN DEGREES",
        "#==============================================================================================================================================",
        "    MD            X            Y            Z           TVD           DX           DY         AZIM_TN        INCL        DLS        AZIM_GN",
        "#==============================================================================================================================================",
    ]

    cumulative_md = 0.0
    prev_inc_deg = 0.0
    prev_azi_deg = 0.0
    prev_x = surface_x
    prev_y = surface_y
    prev_z = surface_z
    for index, row in rows.iterrows():
        x_value = _station_float(row, "X_m")
        y_value = _station_float(row, "Y_m")
        z_value = _station_float(row, "Z_m")
        if int(index) == 0:
            inc_deg = 0.0
            azi_deg = 0.0
            dls_deg_per_30m = 0.0
        else:
            dx = x_value - prev_x
            dy = y_value - prev_y
            dz = z_value - prev_z
            segment_length_m = math.sqrt(dx * dx + dy * dy + dz * dz)
            cumulative_md += segment_length_m
            horizontal_offset_m = math.hypot(dx, dy)
            inc_deg = (
                math.degrees(math.atan2(horizontal_offset_m, dz))
                if segment_length_m > 1e-9
                else prev_inc_deg
            )
            azi_deg = (
                math.degrees(math.atan2(dx, dy)) % 360.0
                if horizontal_offset_m > 1e-9
                else prev_azi_deg
            )
            dogleg_deg = math.degrees(
                float(
                    dogleg_angle_rad(
                        prev_inc_deg,
                        prev_azi_deg,
                        inc_deg,
                        azi_deg,
                    )
                )
            )
            dls_deg_per_30m = (
                dogleg_deg * 30.0 / segment_length_m
                if segment_length_m > 1e-9
                else 0.0
            )
        lines.append(
            " ".join(
                [
                    _format_export_number(cumulative_md),
                    _format_export_number(x_value),
                    _format_export_number(y_value),
                    _format_export_number(-z_value),
                    _format_export_number(z_value - surface_z),
                    _format_export_number(x_value - surface_x),
                    _format_export_number(y_value - surface_y),
                    _format_export_number(azi_deg),
                    _format_export_number(inc_deg),
                    _format_export_number(dls_deg_per_30m),
                    _format_export_number(azi_deg),
                ]
            )
        )
        prev_x = x_value
        prev_y = y_value
        prev_z = z_value
        prev_inc_deg = inc_deg
        prev_azi_deg = azi_deg
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
