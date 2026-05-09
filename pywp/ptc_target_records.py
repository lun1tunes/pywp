from __future__ import annotations

import pandas as pd

from pywp.eclipse_welltrack import WelltrackRecord

__all__ = [
    "DEFAULT_WELLHEAD_Z_TOLERANCE_M",
    "raw_records_dataframe",
    "record_first_point_is_surface_like",
    "record_has_strictly_increasing_md",
    "record_has_surface_like_point",
    "record_import_problem_text",
    "record_is_ready_for_calc",
    "record_target_point_count",
    "records_overview_dataframe",
]

DEFAULT_WELLHEAD_Z_TOLERANCE_M = 100.0
_OVERVIEW_COLUMNS = ("Скважина", "Точек", "Статус", "Проблема")
_RAW_RECORD_COLUMNS = ("Скважина", "Точка", "X, м", "Y, м", "Z, м")


def records_overview_dataframe(
    records: list[WelltrackRecord],
    *,
    wellhead_z_tolerance_m: float = DEFAULT_WELLHEAD_Z_TOLERANCE_M,
) -> pd.DataFrame:
    """Build the target import status table for parsed WELLTRACK records."""

    return pd.DataFrame(
        [
            {
                "Скважина": record.name,
                "Точек": record_target_point_count(record),
                "Статус": (
                    "✅"
                    if record_is_ready_for_calc(
                        record,
                        wellhead_z_tolerance_m=wellhead_z_tolerance_m,
                    )
                    else "❌"
                ),
                "Проблема": record_import_problem_text(
                    record,
                    wellhead_z_tolerance_m=wellhead_z_tolerance_m,
                ),
            }
            for record in records
        ],
        columns=list(_OVERVIEW_COLUMNS),
    )


def raw_records_dataframe(records: list[WelltrackRecord]) -> pd.DataFrame:
    """Build the current S/t1/t3 coordinate table used by the PTC UI."""

    raw_rows: list[dict[str, object]] = []
    for record in records:
        for index, point in enumerate(record.points, start=1):
            raw_rows.append(
                {
                    "Скважина": record.name,
                    "Точка": _point_label(index),
                    "X, м": float(point.x),
                    "Y, м": float(point.y),
                    "Z, м": float(point.z),
                }
            )
    return pd.DataFrame(raw_rows, columns=list(_RAW_RECORD_COLUMNS))


def record_target_point_count(record: WelltrackRecord) -> int:
    return int(max(len(tuple(record.points)) - 1, 0))


def record_has_surface_like_point(
    record: WelltrackRecord,
    *,
    wellhead_z_tolerance_m: float = DEFAULT_WELLHEAD_Z_TOLERANCE_M,
) -> bool:
    return any(
        abs(float(point.z)) <= float(wellhead_z_tolerance_m)
        for point in tuple(record.points)
    )


def record_first_point_is_surface_like(
    record: WelltrackRecord,
    *,
    wellhead_z_tolerance_m: float = DEFAULT_WELLHEAD_Z_TOLERANCE_M,
) -> bool:
    points = tuple(record.points)
    if not points:
        return False
    return bool(abs(float(points[0].z)) <= float(wellhead_z_tolerance_m))


def record_has_strictly_increasing_md(record: WelltrackRecord) -> bool:
    md_values = [float(point.md) for point in tuple(record.points)]
    return all(
        left < right
        for left, right in zip(md_values, md_values[1:], strict=False)
    )


def record_import_problem_text(
    record: WelltrackRecord,
    *,
    wellhead_z_tolerance_m: float = DEFAULT_WELLHEAD_Z_TOLERANCE_M,
) -> str:
    points = tuple(record.points)
    problems: list[str] = []
    target_count = record_target_point_count(record)
    if not points:
        problems.append("Нет точек WELLTRACK.")
    else:
        has_surface_like_point = record_has_surface_like_point(
            record,
            wellhead_z_tolerance_m=wellhead_z_tolerance_m,
        )
        if not has_surface_like_point:
            problems.append(
                "Не найдена точка `S`: среди точек нет `Z` около поверхности (±100 м)."
            )
        elif not record_first_point_is_surface_like(
            record,
            wellhead_z_tolerance_m=wellhead_z_tolerance_m,
        ):
            problems.append("Первая точка не похожа на устье `S`.")
        if not record_has_strictly_increasing_md(record):
            problems.append("MD точек должны строго возрастать.")
    if target_count < 2:
        missing_count = 2 - target_count
        if missing_count == 2:
            problems.append("Не хватает точек `t1` и `t3`.")
        else:
            problems.append("Не хватает одной из точек `t1/t3`.")
    elif target_count > 2:
        problems.append("Лишние точки: ожидаются только `S`, `t1`, `t3`.")
    return "—" if not problems else " ".join(problems)


def record_is_ready_for_calc(
    record: WelltrackRecord,
    *,
    wellhead_z_tolerance_m: float = DEFAULT_WELLHEAD_Z_TOLERANCE_M,
) -> bool:
    return (
        record_import_problem_text(
            record,
            wellhead_z_tolerance_m=wellhead_z_tolerance_m,
        )
        == "—"
    )


def _point_label(index: int) -> str:
    if int(index) == 1:
        return "S"
    if int(index) == 2:
        return "t1"
    if int(index) == 3:
        return "t3"
    return f"p{int(index)}"
