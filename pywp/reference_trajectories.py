from __future__ import annotations

import math
import os
import re
from io import StringIO
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from pywp.constants import SMALL
from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackRecord,
    decode_welltrack_bytes,
    parse_welltrack_text,
)
from pywp.mcm import add_dls
from pywp.models import Point3D
from pywp.pydantic_base import FrozenArbitraryModel

REFERENCE_WELL_ACTUAL = "actual"
REFERENCE_WELL_APPROVED = "approved"

REFERENCE_WELL_KIND_LABELS: dict[str, str] = {
    REFERENCE_WELL_ACTUAL: "Фактическая",
    REFERENCE_WELL_APPROVED: "Проектная утвержденная",
}

REFERENCE_WELL_KIND_COLORS: dict[str, str] = {
    REFERENCE_WELL_ACTUAL: "#6B7280",
    REFERENCE_WELL_APPROVED: "#C62828",
}

_REFERENCE_KIND_ALIASES: dict[str, str] = {
    "actual": REFERENCE_WELL_ACTUAL,
    "factual": REFERENCE_WELL_ACTUAL,
    "fact": REFERENCE_WELL_ACTUAL,
    "actual well": REFERENCE_WELL_ACTUAL,
    "факт": REFERENCE_WELL_ACTUAL,
    "фактическая": REFERENCE_WELL_ACTUAL,
    "фактический": REFERENCE_WELL_ACTUAL,
    "approved": REFERENCE_WELL_APPROVED,
    "approved project": REFERENCE_WELL_APPROVED,
    "project": REFERENCE_WELL_APPROVED,
    "plan": REFERENCE_WELL_APPROVED,
    "project approved": REFERENCE_WELL_APPROVED,
    "утвержденная": REFERENCE_WELL_APPROVED,
    "утверждённая": REFERENCE_WELL_APPROVED,
    "проектная": REFERENCE_WELL_APPROVED,
    "проектная утвержденная": REFERENCE_WELL_APPROVED,
    "проектная утверждённая": REFERENCE_WELL_APPROVED,
}


class ImportedTrajectoryWell(FrozenArbitraryModel):
    name: str
    kind: str
    stations: pd.DataFrame
    surface: Point3D
    azimuth_deg: float


def parse_reference_trajectory_table(
    rows: Iterable[Mapping[str, object]],
    *,
    default_kind: str | None = None,
) -> list[ImportedTrajectoryWell]:
    normalized_default_kind = (
        normalize_reference_well_kind(default_kind)
        if default_kind is not None
        else None
    )
    grouped_rows: dict[
        tuple[str, str], list[tuple[int, float, float, float, float]]
    ] = {}
    group_order: list[tuple[str, str]] = []
    has_non_empty_row = False

    for row_no, raw_row in enumerate(rows, start=1):
        row = {str(key).strip().lower(): value for key, value in dict(raw_row).items()}
        name_raw = _table_row_value(row, "wellname", "well", "name")
        kind_raw = _table_row_value(row, "type", "kind", "welltype", "trajectorytype")
        x_raw = _table_row_value(row, "x")
        y_raw = _table_row_value(row, "y")
        z_raw = _table_row_value(row, "z")
        md_raw = _table_row_value(row, "md")

        if all(
            _is_blank_table_value(value)
            for value in (name_raw, kind_raw, x_raw, y_raw, z_raw, md_raw)
        ):
            continue

        has_non_empty_row = True
        well_name = str(name_raw).strip()
        if not well_name:
            raise WelltrackParseError(
                f"Таблица дополнительных скважин: пустое имя скважины в строке {row_no}."
            )
        if normalized_default_kind is not None and _is_blank_table_value(kind_raw):
            well_kind = normalized_default_kind
        else:
            well_kind = normalize_reference_well_kind(kind_raw, row_no=row_no)
        x = _coerce_table_float(x_raw, field_name="X", row_no=row_no)
        y = _coerce_table_float(y_raw, field_name="Y", row_no=row_no)
        z = _coerce_table_float(z_raw, field_name="Z", row_no=row_no)
        md = _coerce_table_float(md_raw, field_name="MD", row_no=row_no)

        key = (well_name, well_kind)
        if key not in grouped_rows:
            grouped_rows[key] = []
            group_order.append(key)
        grouped_rows[key].append((row_no, x, y, z, md))

    if not has_non_empty_row:
        raise WelltrackParseError(
            "Таблица дополнительных скважин пуста. "
            "Вставьте строки в формате Wellname / Type / X / Y / Z / MD."
        )

    wells: list[ImportedTrajectoryWell] = []
    for well_name, well_kind in group_order:
        point_rows = grouped_rows[(well_name, well_kind)]
        if len(point_rows) < 2:
            raise WelltrackParseError(
                "Таблица дополнительных скважин: для "
                f"'{well_name}' требуется минимум 2 точки траектории."
            )
        sorted_rows = sorted(point_rows, key=lambda item: float(item[4]))
        md_values = [float(item[4]) for item in sorted_rows]
        if any(
            md_values[index + 1] <= md_values[index] + SMALL
            for index in range(len(md_values) - 1)
        ):
            raise WelltrackParseError(
                f"Таблица дополнительных скважин: MD для '{well_name}' должны быть строго возрастающими."
            )
        stations = build_reference_trajectory_stations(
            xs=[float(item[1]) for item in sorted_rows],
            ys=[float(item[2]) for item in sorted_rows],
            zs=[float(item[3]) for item in sorted_rows],
            mds=[float(item[4]) for item in sorted_rows],
        )
        wells.append(
            ImportedTrajectoryWell(
                name=well_name,
                kind=well_kind,
                stations=stations,
                surface=Point3D(
                    x=float(stations["X_m"].iloc[0]),
                    y=float(stations["Y_m"].iloc[0]),
                    z=float(stations["Z_m"].iloc[0]),
                ),
                azimuth_deg=float(_infer_reference_azimuth_deg(stations)),
            )
        )
    return wells


def parse_reference_trajectory_text(text: str) -> list[ImportedTrajectoryWell]:
    return parse_reference_trajectory_text_with_kind(text)


def parse_reference_trajectory_text_with_kind(
    text: str,
    *,
    default_kind: str | None = None,
) -> list[ImportedTrajectoryWell]:
    source = str(text or "")
    if not source.strip():
        raise WelltrackParseError(
            "Текст дополнительных скважин пуст. "
            "Вставьте строки в формате Wellname / Type / X / Y / Z / MD."
        )
    normalized_default_kind = (
        normalize_reference_well_kind(default_kind)
        if default_kind is not None
        else None
    )
    normalized_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not normalized_lines:
        raise WelltrackParseError(
            "Текст дополнительных скважин пуст. "
            "Вставьте строки в формате Wellname / Type / X / Y / Z / MD."
        )
    first_tokens = _split_reference_text_line(normalized_lines[0])
    has_header = _looks_like_reference_header(first_tokens)
    has_header_without_type = (
        normalized_default_kind is not None
        and _looks_like_reference_header_without_type(first_tokens)
    )
    if has_header or has_header_without_type:
        frame = pd.read_csv(
            StringIO("\n".join(normalized_lines)),
            sep=r"[\t,; ]+",
            engine="python",
        )
        return parse_reference_trajectory_table(
            frame.to_dict(orient="records"),
            default_kind=normalized_default_kind,
        )

    rows: list[dict[str, object]] = []
    for row_no, line in enumerate(normalized_lines, start=1):
        tokens = _split_reference_text_line(line)
        if len(tokens) == 6:
            rows.append(
                {
                    "Wellname": tokens[0],
                    "Type": tokens[1],
                    "X": tokens[2],
                    "Y": tokens[3],
                    "Z": tokens[4],
                    "MD": tokens[5],
                }
            )
            continue
        if normalized_default_kind is not None and len(tokens) == 5:
            rows.append(
                {
                    "Wellname": tokens[0],
                    "Type": normalized_default_kind,
                    "X": tokens[1],
                    "Y": tokens[2],
                    "Z": tokens[3],
                    "MD": tokens[4],
                }
            )
            continue
        expected_label = (
            "5 или 6 полей `Wellname X Y Z MD` / `Wellname Type X Y Z MD`"
            if normalized_default_kind is not None
            else "6 полей `Wellname Type X Y Z MD`"
        )
        raise WelltrackParseError(
            "Текст дополнительных скважин: ожидается "
            f"{expected_label} в строке {row_no}, получено {len(tokens)}."
        )
    return parse_reference_trajectory_table(
        rows,
        default_kind=normalized_default_kind,
    )


def parse_reference_trajectory_welltrack_text(
    text: str,
    *,
    kind: str,
) -> list[ImportedTrajectoryWell]:
    source = str(text or "")
    if not source.strip():
        raise WelltrackParseError("Текст WELLTRACK для дополнительных скважин пуст.")
    return reference_welltrack_records_to_wells(
        parse_welltrack_text(source),
        kind=kind,
    )


def parse_reference_trajectory_dev_text(
    text: str,
    *,
    well_name: str,
    kind: str,
) -> ImportedTrajectoryWell:
    source = str(text or "")
    normalized_kind = normalize_reference_well_kind(kind)
    normalized_name = str(well_name or "").strip()
    if not normalized_name:
        raise WelltrackParseError(".dev: имя скважины не задано.")

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    mds: list[float] = []
    for line_no, raw_line in enumerate(source.splitlines(), start=1):
        line = str(raw_line).strip()
        if not line or line.startswith("#"):
            continue
        tokens = _split_dev_text_line(line)
        if not tokens:
            continue
        try:
            md = _parse_dev_float(tokens[0])
        except ValueError:
            continue
        if len(tokens) < 4:
            raise WelltrackParseError(
                f".dev '{normalized_name}': строка {line_no} содержит MD, "
                "но не содержит обязательные X Y Z."
            )
        try:
            x = _parse_dev_float(tokens[1])
            y = _parse_dev_float(tokens[2])
            z = -_parse_dev_float(tokens[3])
        except ValueError as exc:
            raise WelltrackParseError(
                f".dev '{normalized_name}': не удалось прочитать X Y Z "
                f"в строке {line_no}."
            ) from exc
        mds.append(md)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    if len(mds) < 2:
        raise WelltrackParseError(
            f".dev '{normalized_name}': требуется минимум 2 числовые "
            "станции."
        )
    stations = build_reference_trajectory_stations(
        xs=xs,
        ys=ys,
        zs=zs,
        mds=mds,
    )
    return ImportedTrajectoryWell(
        name=normalized_name,
        kind=normalized_kind,
        stations=stations,
        surface=Point3D(
            x=float(stations["X_m"].iloc[0]),
            y=float(stations["Y_m"].iloc[0]),
            z=float(stations["Z_m"].iloc[0]),
        ),
        azimuth_deg=float(_infer_reference_azimuth_deg(stations)),
    )


def parse_reference_trajectory_dev_file(
    path: str | Path,
    *,
    kind: str,
) -> ImportedTrajectoryWell:
    source_path = Path(path).expanduser()
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise WelltrackParseError(
            f"Не удалось прочитать .dev файл `{source_path}`: {exc}"
        ) from exc
    text, _encoding = decode_welltrack_bytes(raw)
    return parse_reference_trajectory_dev_text(
        text,
        well_name=source_path.stem,
        kind=kind,
    )


def parse_reference_trajectory_dev_directories(
    directories: Iterable[str | Path],
    *,
    kind: str,
) -> list[ImportedTrajectoryWell]:
    normalized_kind = normalize_reference_well_kind(kind)
    source_dirs = [
        Path(str(directory).strip()).expanduser()
        for directory in directories
        if str(directory).strip()
    ]
    if not source_dirs:
        raise WelltrackParseError(
            "Укажите путь хотя бы к одной папке "
            "с .dev файлами."
        )

    dev_files: list[Path] = []
    unique_source_dirs: list[Path] = []
    seen_dir_keys: set[str] = set()
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise WelltrackParseError(f"Папка .dev не найдена: `{source_dir}`.")
        if not source_dir.is_dir():
            raise WelltrackParseError(
                f"Путь .dev не является папкой: `{source_dir}`."
            )
        dir_key = os.path.normcase(str(source_dir.resolve()))
        if dir_key in seen_dir_keys:
            continue
        seen_dir_keys.add(dir_key)
        unique_source_dirs.append(source_dir)
        dev_files.extend(
            sorted(
                (
                    child
                    for child in source_dir.iterdir()
                    if child.is_file() and child.suffix.lower() == ".dev"
                ),
                key=_natural_path_sort_key,
            )
        )

    if not dev_files:
        joined_dirs = ", ".join(f"`{item}`" for item in unique_source_dirs)
        raise WelltrackParseError(
            f"В папках {joined_dirs} не найдено .dev файлов."
        )

    wells: list[ImportedTrajectoryWell] = []
    seen_names: dict[str, Path] = {}
    for dev_file in dev_files:
        well_name = dev_file.stem
        well_key = well_name.casefold()
        previous_path = seen_names.get(well_key)
        if previous_path is not None:
            raise WelltrackParseError(
                "Найдено несколько .dev файлов с одинаковым "
                "именем скважины "
                f"`{well_name}`: `{previous_path}` и `{dev_file}`."
            )
        seen_names[well_key] = dev_file
        wells.append(
            parse_reference_trajectory_dev_file(
                dev_file,
                kind=normalized_kind,
            )
        )
    return wells


def reference_welltrack_records_to_wells(
    records: Iterable[WelltrackRecord],
    *,
    kind: str,
) -> list[ImportedTrajectoryWell]:
    normalized_kind = normalize_reference_well_kind(kind)
    ordered_records = list(records)
    if not ordered_records:
        raise WelltrackParseError("WELLTRACK дополнительных скважин пуст.")

    wells: list[ImportedTrajectoryWell] = []
    for record in ordered_records:
        points = tuple(record.points)
        if len(points) < 2:
            raise WelltrackParseError(
                "WELLTRACK дополнительных скважин: для "
                f"'{record.name}' требуется минимум 2 точки траектории."
            )
        try:
            stations = build_reference_trajectory_stations(
                xs=[float(point.x) for point in points],
                ys=[float(point.y) for point in points],
                zs=[float(point.z) for point in points],
                mds=[float(point.md) for point in points],
            )
        except WelltrackParseError as exc:
            raise WelltrackParseError(
                f"WELLTRACK дополнительных скважин '{record.name}': {exc}"
            ) from exc
        wells.append(
            ImportedTrajectoryWell(
                name=str(record.name),
                kind=normalized_kind,
                stations=stations,
                surface=Point3D(
                    x=float(stations["X_m"].iloc[0]),
                    y=float(stations["Y_m"].iloc[0]),
                    z=float(stations["Z_m"].iloc[0]),
                ),
                azimuth_deg=float(_infer_reference_azimuth_deg(stations)),
            )
        )
    return wells


def build_reference_trajectory_stations(
    *,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    mds: list[float],
) -> pd.DataFrame:
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    z_values = np.asarray(zs, dtype=float)
    md_values = np.asarray(mds, dtype=float)
    if not (
        len(x_values) == len(y_values) == len(z_values) == len(md_values)
        and len(md_values) >= 2
    ):
        raise WelltrackParseError(
            "массивы X, Y, Z и MD должны быть одинаковой длины и содержать не менее двух точек"
        )
    if not (
        np.isfinite(x_values).all()
        and np.isfinite(y_values).all()
        and np.isfinite(z_values).all()
        and np.isfinite(md_values).all()
    ):
        raise WelltrackParseError(
            "все значения X, Y, Z и MD должны быть конечными числами"
        )
    if np.any(np.diff(md_values) <= SMALL):
        raise WelltrackParseError("MD должен строго возрастать по траектории")

    tangent_x = _local_derivative(md_values, x_values)
    tangent_y = _local_derivative(md_values, y_values)
    tangent_z = _local_derivative(md_values, z_values)
    horizontal = np.hypot(tangent_x, tangent_y)
    inc_deg = np.degrees(np.arctan2(horizontal, np.abs(tangent_z)))
    azi_deg = (np.degrees(np.arctan2(tangent_x, tangent_y)) + 360.0) % 360.0

    stations = pd.DataFrame(
        {
            "MD_m": md_values,
            "INC_deg": inc_deg,
            "AZI_deg": azi_deg,
            "X_m": x_values,
            "Y_m": y_values,
            "Z_m": z_values,
            "segment": ["IMPORTED"] * len(md_values),
        }
    )
    return add_dls(stations)


def normalize_reference_well_kind(value: object, *, row_no: int | None = None) -> str:
    if _is_blank_table_value(value):
        row_suffix = f" в строке {int(row_no)}" if row_no is not None else ""
        raise WelltrackParseError(
            f"Таблица дополнительных скважин: пустой Type{row_suffix}."
        )
    normalized = str(value).strip().lower()
    mapped = _REFERENCE_KIND_ALIASES.get(normalized)
    if mapped is not None:
        return mapped
    row_suffix = f" в строке {int(row_no)}" if row_no is not None else ""
    raise WelltrackParseError(
        "Таблица дополнительных скважин: unsupported Type="
        f"{value!r}{row_suffix}. Ожидается actual или approved."
    )


def reference_well_display_label(well: ImportedTrajectoryWell) -> str:
    return f"{str(well.name)} ({REFERENCE_WELL_KIND_LABELS.get(str(well.kind), str(well.kind))})"


def reference_well_duplicate_name_keys(
    wells: Iterable[ImportedTrajectoryWell],
) -> set[str]:
    name_counts: dict[str, int] = {}
    for well in wells:
        key = str(well.name).strip().casefold()
        if not key:
            continue
        name_counts[key] = int(name_counts.get(key, 0)) + 1
    return {key for key, count in name_counts.items() if int(count) > 1}


def reference_well_collision_name(
    well: ImportedTrajectoryWell,
    *,
    planned_names: Iterable[str] = (),
    duplicate_name_keys: set[str] | None = None,
) -> str:
    name = str(well.name)
    name_key = name.strip().casefold()
    planned_name_keys = {
        str(item).strip().casefold()
        for item in planned_names
        if str(item).strip()
    }
    duplicate_keys = set(duplicate_name_keys or set())
    if name_key in planned_name_keys or name_key in duplicate_keys:
        return reference_well_display_label(well)
    return name


def reference_wells_to_table_rows(
    wells: Iterable[ImportedTrajectoryWell],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for well in wells:
        stations = well.stations
        for _, row in stations.iterrows():
            rows.append(
                {
                    "Wellname": str(well.name),
                    "Type": str(well.kind),
                    "X": float(row["X_m"]),
                    "Y": float(row["Y_m"]),
                    "Z": float(row["Z_m"]),
                    "MD": float(row["MD_m"]),
                }
            )
    return rows


def _infer_reference_azimuth_deg(stations: pd.DataFrame) -> float:
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    if len(x_values) < 2:
        return 0.0
    dx = np.diff(x_values)
    dy = np.diff(y_values)
    lengths = np.hypot(dx, dy)
    valid = lengths > SMALL
    if not np.any(valid):
        return 0.0
    first_index = int(np.argmax(valid))
    return float(
        (np.degrees(np.arctan2(dx[first_index], dy[first_index])) + 360.0) % 360.0
    )


def _local_derivative(md_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    if len(values) == 2:
        delta_md = float(md_values[1] - md_values[0])
        slope = (
            0.0 if abs(delta_md) <= 1e-12 else float(values[1] - values[0]) / delta_md
        )
        result[:] = slope
        return result
    for index in range(len(values)):
        if index == 0:
            left = 0
            right = 1
        elif index == len(values) - 1:
            left = len(values) - 2
            right = len(values) - 1
        else:
            left = index - 1
            right = index + 1
        delta_md = float(md_values[right] - md_values[left])
        result[index] = (
            0.0
            if abs(delta_md) <= 1e-12
            else float(values[right] - values[left]) / delta_md
        )
    return result


def _split_reference_text_line(line: str) -> list[str]:
    return [token for token in re.split(r"[\t,; ]+", str(line).strip()) if token]


def _split_dev_text_line(line: str) -> list[str]:
    text = str(line).strip()
    tokens = [token for token in re.split(r"[\t; ]+", text) if token]
    if len(tokens) > 1:
        return tokens
    return [token for token in text.split(",") if token]


def _parse_dev_float(value: str) -> float:
    return float(str(value).strip().replace(",", "."))


def _natural_path_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", path.name.casefold())
    )


def _looks_like_reference_header(tokens: list[str]) -> bool:
    normalized = [str(token).strip().lower() for token in tokens]
    expected = ["wellname", "type", "x", "y", "z", "md"]
    if normalized[: len(expected)] == expected:
        return True
    return set(expected).issubset(set(normalized))


def _looks_like_reference_header_without_type(tokens: list[str]) -> bool:
    normalized = [str(token).strip().lower() for token in tokens]
    expected = ["wellname", "x", "y", "z", "md"]
    if normalized[: len(expected)] == expected:
        return True
    return set(expected).issubset(set(normalized))


def _table_row_value(row: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _is_blank_table_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(math.isnan(value)) if isinstance(value, float) else False


def _coerce_table_float(value: object, *, field_name: str, row_no: int) -> float:
    if _is_blank_table_value(value):
        raise WelltrackParseError(
            f"Таблица дополнительных скважин: пустое значение {field_name} в строке {row_no}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WelltrackParseError(
            f"Таблица дополнительных скважин: {field_name} в строке {row_no} не является числом."
        ) from exc
    if not math.isfinite(number):
        raise WelltrackParseError(
            f"Таблица дополнительных скважин: {field_name} в строке {row_no} должно быть конечным числом."
        )
    return number
