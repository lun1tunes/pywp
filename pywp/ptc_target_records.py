from __future__ import annotations

import math

import pandas as pd

from pywp.eclipse_welltrack import WelltrackRecord
from pywp.pilot_wells import (
    is_pilot_record,
    parent_name_for_pilot,
    pilot_name_key_for_parent,
    pilot_record_problem_text,
    well_name_key,
)

__all__ = [
    "DEFAULT_WELLHEAD_Z_TOLERANCE_M",
    "raw_records_dataframe",
    "record_first_point_is_surface_like",
    "record_has_finite_points",
    "record_has_strictly_increasing_md",
    "record_has_surface_like_point",
    "record_import_problem_text",
    "record_is_ready_for_calc",
    "record_target_point_count",
    "records_overview_dataframe",
]

DEFAULT_WELLHEAD_Z_TOLERANCE_M = 100.0
_OVERVIEW_COLUMNS = (
    "Скважина",
    "Точек",
    "Отход t1, м",
    "Длина ГС, м",
    "Примечание",
    "Статус",
    "Проблема",
)
_RAW_RECORD_COLUMNS = ("Скважина", "Точка", "X, м", "Y, м", "Z, м")


def records_overview_dataframe(
    records: list[WelltrackRecord],
    *,
    wellhead_z_tolerance_m: float = DEFAULT_WELLHEAD_Z_TOLERANCE_M,
) -> pd.DataFrame:
    """Build the target import status table for parsed WELLTRACK records."""

    record_names = {well_name_key(record.name) for record in records}
    pilot_by_parent_key = {
        well_name_key(parent_name_for_pilot(record.name)): record
        for record in records
        if is_pilot_record(record)
    }
    visible_records = [record for record in records if not is_pilot_record(record)]
    return pd.DataFrame(
        [
            {
                "Скважина": record.name,
                "Точек": record_target_point_count(record),
                "Отход t1, м": _record_t1_offset_m(record),
                "Длина ГС, м": _record_t1_t3_length_m(record),
                "Примечание": (
                    "Есть пилот"
                    if pilot_name_key_for_parent(record.name) in record_names
                    else ""
                ),
                "Статус": (
                    "✅"
                    if _record_overview_problem_text(
                        record,
                        pilot=pilot_by_parent_key.get(well_name_key(record.name)),
                        wellhead_z_tolerance_m=wellhead_z_tolerance_m,
                    )
                    == "—"
                    else "❌"
                ),
                "Проблема": _record_overview_problem_text(
                    record,
                    pilot=pilot_by_parent_key.get(well_name_key(record.name)),
                    wellhead_z_tolerance_m=wellhead_z_tolerance_m,
                ),
            }
            for record in visible_records
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
                    "Точка": _point_label(index, is_pilot=is_pilot_record(record)),
                    "X, м": float(point.x),
                    "Y, м": float(point.y),
                    "Z, м": float(point.z),
                }
            )
    return pd.DataFrame(raw_rows, columns=list(_RAW_RECORD_COLUMNS))


def record_target_point_count(record: WelltrackRecord) -> int:
    return int(max(len(tuple(record.points)) - 1, 0))


def _record_t1_offset_m(record: WelltrackRecord) -> float | None:
    points = tuple(record.points)
    if len(points) < 2:
        return None
    surface, t1 = points[0], points[1]
    if not (_point_has_finite_xyz(surface) and _point_has_finite_xyz(t1)):
        return None
    return float(
        math.hypot(
            float(t1.x) - float(surface.x),
            float(t1.y) - float(surface.y),
        )
    )


def _record_t1_t3_length_m(record: WelltrackRecord) -> float | None:
    points = tuple(record.points)
    if len(points) < 3:
        return None
    t1, t3 = points[1], points[2]
    if not (_point_has_finite_xyz(t1) and _point_has_finite_xyz(t3)):
        return None
    return float(
        math.hypot(
            float(t3.x) - float(t1.x),
            float(t3.y) - float(t1.y),
            float(t3.z) - float(t1.z),
        )
    )


def record_has_finite_points(record: WelltrackRecord) -> bool:
    return all(
        _point_has_finite_xyz(point) and _value_is_finite(point.md)
        for point in tuple(record.points)
    )


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
        if not record_has_finite_points(record):
            problems.append("Координаты X/Y/Z и MD должны быть конечными числами.")
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
        if is_pilot_record(record) and target_count >= 1:
            pilot_problem = pilot_record_problem_text(record)
            if pilot_problem != "—":
                problems.append(pilot_problem)
            return "—" if not problems else " ".join(problems)
        missing_count = 2 - target_count
        if missing_count == 2:
            problems.append("Не хватает точек `t1` и `t3`.")
        else:
            problems.append("Не хватает одной из точек `t1/t3`.")
    elif target_count > 2 and not is_pilot_record(record):
        problems.append("Лишние точки: ожидаются только `S`, `t1`, `t3`.")
    if is_pilot_record(record):
        pilot_problem = pilot_record_problem_text(record)
        if pilot_problem != "—":
            problems.append(pilot_problem)
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


def _record_overview_problem_text(
    record: WelltrackRecord,
    *,
    pilot: WelltrackRecord | None,
    wellhead_z_tolerance_m: float,
) -> str:
    problems: list[str] = []
    own_problem = record_import_problem_text(
        record,
        wellhead_z_tolerance_m=wellhead_z_tolerance_m,
    )
    if own_problem != "—":
        problems.append(own_problem)
    if pilot is not None:
        pilot_problem = record_import_problem_text(
            pilot,
            wellhead_z_tolerance_m=wellhead_z_tolerance_m,
        )
        if pilot_problem != "—":
            problems.append(f"Пилот: {pilot_problem}")
    return "—" if not problems else " ".join(problems)


def _point_label(index: int, *, is_pilot: bool = False) -> str:
    if int(index) == 1:
        return "S"
    if is_pilot:
        return f"PL{int(index) - 1}"
    if int(index) == 2:
        return "t1"
    if int(index) == 3:
        return "t3"
    return f"p{int(index)}"


def _point_has_finite_xyz(point: object) -> bool:
    return (
        _value_is_finite(getattr(point, "x", None))
        and _value_is_finite(getattr(point, "y", None))
        and _value_is_finite(getattr(point, "z", None))
    )


def _value_is_finite(value: object) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
