from __future__ import annotations

import math

import pandas as pd

from pywp.eclipse_welltrack import (
    WelltrackRecord,
    welltrack_points_to_target_pairs,
)
from pywp.pilot_wells import (
    is_zbs_record,
    is_pilot_record,
    parent_name_for_pilot,
    parent_name_for_zbs,
    pilot_parent_key_for_record,
    pilot_name_key_for_record,
    pilot_record_problem_text,
    well_name_key,
    zbs_target_points_to_pairs,
    zbs_multi_horizontal_level_count,
)
from pywp.welltrack_targets import (
    record_is_ordinary_target_sequence,
    record_multi_horizontal_level_count,
    record_point_labels,
    target_sequence_points_from_record,
)

__all__ = [
    "DEFAULT_WELLHEAD_Z_TOLERANCE_M",
    "raw_records_dataframe",
    "record_first_point_is_surface_like",
    "record_horizontal_length_preprocess_skip_reason",
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
                "Примечание": _record_note(
                    record,
                    has_pilot=pilot_name_key_for_record(record) in record_names,
                ),
                "Статус": (
                    "✅"
                    if _record_overview_problem_text(
                        record,
                        pilot=pilot_by_parent_key.get(
                            pilot_parent_key_for_record(record)
                        ),
                        wellhead_z_tolerance_m=wellhead_z_tolerance_m,
                    )
                    == "—"
                    else "❌"
                ),
                "Проблема": _record_overview_problem_text(
                    record,
                    pilot=pilot_by_parent_key.get(
                        pilot_parent_key_for_record(record)
                    ),
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
        point_count = len(tuple(record.points))
        explicit_labels = tuple(
            str(label).strip()
            for label in (getattr(record, "point_labels", ()) or ())
        )
        use_explicit_labels = (
            bool(explicit_labels)
            and len(explicit_labels) == point_count
            and all(explicit_labels)
        )
        for index, point in enumerate(record.points, start=1):
            raw_rows.append(
                {
                    "Скважина": record.name,
                    "Точка": (
                        str(explicit_labels[index - 1])
                        if use_explicit_labels
                        else _point_label(
                            index,
                            is_pilot=is_pilot_record(record),
                            is_zbs=is_zbs_record(record),
                            multi_level_count=_record_multi_horizontal_level_count(record),
                        )
                    ),
                    "X, м": float(point.x),
                    "Y, м": float(point.y),
                    "Z, м": float(point.z),
                }
            )
    return pd.DataFrame(raw_rows, columns=list(_RAW_RECORD_COLUMNS))


def record_target_point_count(record: WelltrackRecord) -> int:
    if is_zbs_record(record):
        return int(len(tuple(record.points)))
    return int(max(len(tuple(record.points)) - 1, 0))


def _record_t1_offset_m(record: WelltrackRecord) -> float | None:
    if is_zbs_record(record):
        return None
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
    if is_zbs_record(record):
        if (
            len(points) < 2
            or len(points) % 2 != 0
            or not all(_point_has_finite_xyz(point) for point in points)
        ):
            return None
        length_m = 0.0
        for index in range(0, len(points), 2):
            length_m += math.dist(
                (
                    float(points[index].x),
                    float(points[index].y),
                    float(points[index].z),
                ),
                (
                    float(points[index + 1].x),
                    float(points[index + 1].y),
                    float(points[index + 1].z),
                ),
            )
        return float(length_m)
    if len(points) < 3:
        return None
    if not all(_point_has_finite_xyz(point) for point in points[1:]):
        return None
    sequence_points = target_sequence_points_from_record(record)
    if len(sequence_points) >= 2:
        return float(
            sum(
                math.dist(
                    (float(left.x), float(left.y), float(left.z)),
                    (float(right.x), float(right.y), float(right.z)),
                )
                for left, right in zip(sequence_points, sequence_points[1:], strict=False)
            )
        )
    try:
        _, target_pairs = welltrack_points_to_target_pairs(points)
    except ValueError:
        return None
    length_m = 0.0
    for t1, t3 in target_pairs:
        length_m += math.hypot(
            float(t3.x) - float(t1.x),
            float(t3.y) - float(t1.y),
            float(t3.z) - float(t1.z),
        )
    return float(length_m)


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
    if is_zbs_record(record):
        return bool(tuple(record.points))
    return any(
        abs(float(point.z)) <= float(wellhead_z_tolerance_m)
        for point in tuple(record.points)
    )


def record_first_point_is_surface_like(
    record: WelltrackRecord,
    *,
    wellhead_z_tolerance_m: float = DEFAULT_WELLHEAD_Z_TOLERANCE_M,
) -> bool:
    if is_zbs_record(record):
        return bool(tuple(record.points))
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
    if is_zbs_record(record):
        if not points:
            problems.append("Нет точек WELLTRACK.")
        else:
            if not record_has_finite_points(record):
                problems.append("Координаты X/Y/Z и MD должны быть конечными числами.")
            if not record_has_strictly_increasing_md(record):
                problems.append("MD точек должны строго возрастать.")
            try:
                zbs_target_points_to_pairs(points)
            except ValueError as exc:
                problems.append(str(exc))
        return "—" if not problems else " ".join(dict.fromkeys(problems))
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
    elif (
        target_count > 2
        and not is_pilot_record(record)
        and not record_is_ordinary_target_sequence(record)
    ):
        if target_count % 2 != 0:
            problems.append(
                "Для многопластовой скважины после `S` ожидаются полные пары "
                "`N_t1/N_t3`."
            )
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


def _point_label(
    index: int,
    *,
    is_pilot: bool = False,
    is_zbs: bool = False,
    multi_level_count: int = 0,
) -> str:
    if is_zbs:
        if int(multi_level_count) > 1:
            level = int((int(index) - 1) // 2) + 1
            suffix = "t1" if int(index) % 2 == 1 else "t3"
            return f"{level}_{suffix}"
        if int(index) == 1:
            return "t1"
        if int(index) == 2:
            return "t3"
        return f"p{int(index)}"
    if int(index) == 1:
        return "S"
    if is_pilot:
        return f"PL{int(index) - 1}"
    if int(multi_level_count) > 1:
        target_index = int(index) - 2
        level = int(target_index // 2) + 1
        suffix = "t1" if target_index % 2 == 0 else "t3"
        return f"{level}_{suffix}"
    if int(index) == 2:
        return "t1"
    if int(index) == 3:
        return "t3"
    return f"p{int(index)}"


def _record_note(record: WelltrackRecord, *, has_pilot: bool) -> str:
    notes: list[str] = []
    if is_zbs_record(record):
        notes.append(
            f"Боковой ствол от факта: нужна скважина {parent_name_for_zbs(record.name)}"
        )
    if has_pilot:
        notes.append("Есть пилот")
    level_count = _record_multi_horizontal_level_count(record)
    if level_count > 1:
        notes.append(f"Многопластовая: {level_count} уровней")
    return "; ".join(notes)


def _record_multi_horizontal_level_count(record: WelltrackRecord) -> int:
    if is_zbs_record(record):
        return zbs_multi_horizontal_level_count(tuple(record.points))
    return record_multi_horizontal_level_count(record)


def record_horizontal_length_preprocess_skip_reason(
    record: WelltrackRecord,
    *,
    has_pilot: bool = False,
) -> str:
    if is_pilot_record(record):
        return "пилот"
    if bool(has_pilot):
        return "есть пилот"
    if _record_multi_horizontal_level_count(record) > 1:
        return "многопластовая"
    return "—"


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
