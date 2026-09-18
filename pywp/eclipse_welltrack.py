from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Literal, Mapping

from pydantic import field_validator, model_validator
from pywp.models import Point3D
from pywp.pydantic_base import FrozenModel, coerce_model_like
from pywp.well_names import (
    is_alt_branch_name,
    is_pilot_name,
    is_zbs_name,
    parent_name_for_pilot,
    parent_name_for_zbs,
    well_name_key,
)

_WELLTRACK_RE = re.compile(r"^\s*WELLTRACK\b(.*)$", flags=re.IGNORECASE)
DEFAULT_WELLTRACK_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1251", "latin-1")
_MD_EPS = 1e-9
_TABLE_POINT_ALIASES: dict[str, str] = {
    "s": "wellhead",
    "s1": "wellhead",
    "s_1": "wellhead",
    "surface": "wellhead",
    "wellhead": "wellhead",
    "well_head": "wellhead",
    "well head": "wellhead",
    "wh": "wellhead",
    "pl": "pl1",
    "t1": "t1",
    "entry": "t1",
    "entry point": "t1",
    "t3": "t3",
    "target": "t3",
    "end": "t3",
}
_TABLE_PILOT_POINT_RE = re.compile(
    r"^(?:pl|p)_?([1-9]\d*)$",
    flags=re.IGNORECASE,
)
_TABLE_TARGET_SEQUENCE_POINT_RE = re.compile(
    r"^t_?([1-9]\d*)$",
    flags=re.IGNORECASE,
)
_TABLE_MULTI_HORIZONTAL_POINT_RE = re.compile(
    r"^([1-9]\d*)_?t_?([13])$",
    flags=re.IGNORECASE,
)
_TABLE_POINT_ORDER: tuple[str, ...] = ("wellhead", "t1", "t3")
_TABLE_POINT_DISPLAY_LABELS: dict[str, str] = {
    "wellhead": "S",
    "t1": "t1",
    "t3": "t3",
}
_TABLE_VALIDATION_ERROR_LIMIT = 50


class WelltrackParseError(ValueError):
    pass


class WelltrackPoint(FrozenModel):
    x: float
    y: float
    z: float
    md: float


class WelltrackRecord(FrozenModel):
    name: str
    points: tuple[WelltrackPoint, ...]
    point_labels: tuple[str, ...] = ()

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(
        cls,
        value: object,
    ) -> tuple[WelltrackPoint, ...]:
        if value is None:
            raise ValueError("points are required for WelltrackRecord.")
        return tuple(
            coerce_model_like(point, WelltrackPoint)
            for point in tuple(value)
        )

    @field_validator("point_labels", mode="before")
    @classmethod
    def _coerce_point_labels(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        labels = tuple(str(item).strip() for item in tuple(value))
        if any(not label for label in labels):
            raise ValueError("point_labels entries must be non-empty strings.")
        return labels

    @model_validator(mode="after")
    def _validate_point_labels(self) -> "WelltrackRecord":
        if self.point_labels and len(self.point_labels) != len(self.points):
            raise ValueError(
                "point_labels length must match the number of points when provided."
            )
        return self


def decode_welltrack_bytes(
    raw: bytes,
    encodings: tuple[str, ...] = DEFAULT_WELLTRACK_ENCODINGS,
) -> tuple[str, str]:
    payload = bytes(raw)
    if not payload:
        return "", "utf-8"

    for encoding in encodings:
        try:
            return payload.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError:
            continue

    first_encoding = encodings[0] if encodings else "utf-8"
    return payload.decode(first_encoding, errors="replace"), f"{first_encoding}(replace)"


def parse_welltrack_text(text: str) -> list[WelltrackRecord]:
    records: list[WelltrackRecord] = []
    current_name: str | None = None
    numeric_tokens: list[str] = []

    def finalize_current(line_no: int) -> None:
        nonlocal current_name, numeric_tokens
        if current_name is None:
            return
        if len(numeric_tokens) % 4 != 0:
            raise WelltrackParseError(
                f"WELLTRACK '{current_name}': ожидались группы X Y Z MD по 4 значения, "
                f"получено {len(numeric_tokens)} значений в строке {line_no}."
            )
        points: list[WelltrackPoint] = []
        for index in range(0, len(numeric_tokens), 4):
            try:
                x = float(numeric_tokens[index + 0])
                y = float(numeric_tokens[index + 1])
                z = float(numeric_tokens[index + 2])
                md = float(numeric_tokens[index + 3])
            except ValueError as exc:
                raise WelltrackParseError(
                    f"WELLTRACK '{current_name}': не удалось разобрать число около строки {line_no}: {exc}"
                ) from exc
            points.append(WelltrackPoint(x=x, y=y, z=z, md=md))
        _validate_record_md(points=points, well_name=current_name)

        records.append(WelltrackRecord(name=current_name, points=tuple(points)))
        current_name = None
        numeric_tokens = []

    lines = text.splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.split("--", 1)[0].strip()
        if not line:
            continue

        welltrack_match = _WELLTRACK_RE.match(line)
        if welltrack_match is not None:
            if current_name is not None:
                finalize_current(line_no=line_no)
            rest = welltrack_match.group(1).strip()
            name, tail = _parse_well_name(rest=rest, line_no=line_no)
            current_name = name
            _consume_tail_tokens(
                tail=tail,
                numeric_tokens=numeric_tokens,
                finalize=lambda: finalize_current(line_no=line_no),
            )
            continue

        if current_name is None:
            continue

        _consume_tail_tokens(
            tail=line,
            numeric_tokens=numeric_tokens,
            finalize=lambda: finalize_current(line_no=line_no),
        )

    if current_name is not None:
        finalize_current(line_no=len(lines) if lines else 1)

    return records


def parse_welltrack_points_table(
    rows: Iterable[Mapping[str, object]],
) -> list[WelltrackRecord]:
    grouped_points: dict[str, dict[str, WelltrackPoint]] = {}
    point_row_numbers: dict[str, dict[str, int]] = {}
    well_names: dict[str, str] = {}
    well_row_numbers: dict[str, list[int]] = {}
    well_order: list[str] = []
    validation_errors: list[str] = []
    has_non_empty_row = False

    for row_no, raw_row in enumerate(rows, start=1):
        row = _normalize_table_row(dict(raw_row))

        name_raw = _table_row_value(row, "wellname", "well_name", "well", "name")
        point_raw = _table_row_value(row, "point", "pointname", "point_name")
        x_raw = _table_row_value(row, "x", "x_m", "east", "easting")
        y_raw = _table_row_value(row, "y", "y_m", "north", "northing")
        z_raw = _table_row_value(row, "z", "z_m", "tvd", "z_tvd", "z_tvd_m")

        if all(_is_blank_table_value(value) for value in (name_raw, point_raw, x_raw, y_raw, z_raw)):
            continue

        has_non_empty_row = True
        if _is_blank_table_value(name_raw):
            validation_errors.append(
                f"Строка {row_no}: поле Wellname пустое."
            )
            continue
        well_name = str(name_raw).strip()
        if not well_name:
            validation_errors.append(
                f"Строка {row_no}: поле Wellname пустое."
            )
            continue

        row_error_count = len(validation_errors)
        point_name: str | None = None
        try:
            point_name = _normalize_table_point_name(
                point_raw,
                row_no=row_no,
                well_name=well_name,
                allow_pilot_points=_is_table_pilot_well_name(well_name),
            )
        except WelltrackParseError as exc:
            validation_errors.append(str(exc))

        coordinates: dict[str, float] = {}
        for field_name, raw_value in (("X", x_raw), ("Y", y_raw), ("Z", z_raw)):
            try:
                coordinates[field_name] = _coerce_table_float(
                    raw_value,
                    field_name=field_name,
                    row_no=row_no,
                    well_name=well_name,
                    point_value=point_raw,
                )
            except WelltrackParseError as exc:
                validation_errors.append(str(exc))
        if len(validation_errors) != row_error_count or point_name is None:
            continue

        well_key = _table_well_name_key(well_name)
        if well_key not in grouped_points:
            grouped_points[well_key] = {}
            point_row_numbers[well_key] = {}
            well_names[well_key] = well_name
            well_row_numbers[well_key] = []
            well_order.append(well_key)
        well_row_numbers[well_key].append(row_no)

        if point_name in grouped_points[well_key]:
            first_row_no = point_row_numbers[well_key][point_name]
            validation_errors.append(
                f"{_table_row_context(row_no, well_name, point_raw)}: точка "
                f"'{_table_point_display_name(point_name)}' дублирует строку "
                f"{first_row_no} той же скважины "
                f"'{well_names[well_key]}'."
            )
            continue

        md_index = _table_point_md_index(point_name)
        grouped_points[well_key][point_name] = WelltrackPoint(
            x=coordinates["X"],
            y=coordinates["Y"],
            z=coordinates["Z"],
            md=md_index,
        )
        point_row_numbers[well_key][point_name] = row_no

    if validation_errors:
        _raise_table_validation_errors(validation_errors)

    if not has_non_empty_row:
        raise WelltrackParseError(
            "Табличный WELLTRACK пуст. Вставьте строки в формате "
            "Wellname / Point / X / Y / Z."
        )

    grouped_points_by_name = {
        well_names[well_key]: grouped_points[well_key]
        for well_key in well_order
    }
    pilot_surface_by_parent_key = _table_pilot_surface_by_parent_key(
        grouped_points_by_name
    )
    records: list[WelltrackRecord] = []
    structural_errors: list[str] = []
    for well_key in well_order:
        well_name = well_names[well_key]
        try:
            records.append(
                _welltrack_record_from_table_points(
                    grouped_points[well_key],
                    well_name=well_name,
                    pilot_surface_by_parent_key=pilot_surface_by_parent_key,
                )
            )
        except WelltrackParseError as exc:
            structural_errors.append(
                _table_error_with_source_rows(
                    str(exc),
                    row_numbers=well_row_numbers[well_key],
                )
            )

    if structural_errors:
        _raise_table_validation_errors(structural_errors)

    return records


def _welltrack_record_from_table_points(
    source_points_by_name: Mapping[str, WelltrackPoint],
    *,
    well_name: str,
    pilot_surface_by_parent_key: Mapping[str, WelltrackPoint],
) -> WelltrackRecord:
    points_by_name = _table_points_with_inferred_pilot_surface(
        source_points_by_name,
        well_name=well_name,
        pilot_surface_by_parent_key=pilot_surface_by_parent_key,
    )
    if _is_table_pilot_well_name(well_name):
        ordered_names = _ordered_table_pilot_point_names(
            points_by_name,
            well_name=well_name,
        )
    elif _is_table_zbs_well_name(well_name) and (
        not _is_table_alt_branch_well_name(well_name)
        or "wellhead" not in points_by_name
    ):
        ordered_names = _ordered_table_zbs_point_names(
            points_by_name,
            well_name=well_name,
        )
    else:
        target_sequence_names = _ordered_table_target_sequence_point_names(
            points_by_name,
            well_name=well_name,
        )
        if target_sequence_names is not None:
            ordered_names = target_sequence_names
        elif _has_multi_horizontal_table_points(points_by_name):
            ordered_names = _ordered_table_multi_horizontal_point_names(
                points_by_name,
                well_name=well_name,
            )
        else:
            missing = [
                name for name in _TABLE_POINT_ORDER if name not in points_by_name
            ]
            if missing:
                raise WelltrackParseError(
                    "Табличный WELLTRACK: для скважины "
                    f"'{well_name}' отсутствуют точки: "
                    f"{', '.join(_table_point_display_name(name) for name in missing)}."
                )
            ordered_names = _TABLE_POINT_ORDER

    ordered_points = _ordered_table_points(points_by_name, ordered_names)
    _validate_record_md(points=list(ordered_points), well_name=well_name)
    return WelltrackRecord(
        name=well_name,
        points=ordered_points,
        point_labels=tuple(
            _table_point_display_name(name) for name in ordered_names
        ),
    )


def _ordered_table_points(
    points_by_name: Mapping[str, WelltrackPoint],
    ordered_names: Iterable[str],
) -> tuple[WelltrackPoint, ...]:
    return tuple(
        WelltrackPoint(
            x=points_by_name[name].x,
            y=points_by_name[name].y,
            z=points_by_name[name].z,
            md=float(points_by_name[name].md),
        )
        for name in tuple(str(name) for name in ordered_names)
    )


def welltrack_points_to_targets(
    points: tuple[WelltrackPoint, ...],
    *,
    order_mode: Literal["strict_file_order", "sort_by_md"] = "strict_file_order",
) -> tuple[Point3D, Point3D, Point3D]:
    if len(points) != 3:
        raise ValueError(f"Expected exactly 3 points (S, t1, t3), got {len(points)}.")
    if order_mode not in {"strict_file_order", "sort_by_md"}:
        raise ValueError(
            f"Unsupported order_mode={order_mode!r}. "
            "Supported values: 'strict_file_order', 'sort_by_md'."
        )

    ordered_points = points
    if order_mode == "sort_by_md":
        ordered_points = tuple(sorted(points, key=lambda point: float(point.md)))

    md_values = [float(point.md) for point in ordered_points]
    if not all(math.isfinite(value) for value in md_values):
        raise ValueError("All MD values must be finite numbers for S, t1, t3 mapping.")
    if not (md_values[0] + _MD_EPS < md_values[1] and md_values[1] + _MD_EPS < md_values[2]):
        raise ValueError(
            "Expected strictly increasing MD for points S, t1, t3 "
            f"(got MD sequence: {md_values[0]:.3f}, {md_values[1]:.3f}, {md_values[2]:.3f})."
        )

    surface = Point3D(
        x=ordered_points[0].x,
        y=ordered_points[0].y,
        z=ordered_points[0].z,
    )
    t1 = Point3D(
        x=ordered_points[1].x,
        y=ordered_points[1].y,
        z=ordered_points[1].z,
    )
    t3 = Point3D(
        x=ordered_points[2].x,
        y=ordered_points[2].y,
        z=ordered_points[2].z,
    )
    return surface, t1, t3


def welltrack_points_to_target_pairs(
    points: tuple[WelltrackPoint, ...],
    *,
    order_mode: Literal["strict_file_order", "sort_by_md"] = "strict_file_order",
) -> tuple[Point3D, tuple[tuple[Point3D, Point3D], ...]]:
    if len(points) < 3:
        raise ValueError(
            f"Expected at least 3 points (S, t1, t3), got {len(points)}."
        )
    target_count = len(points) - 1
    if target_count % 2 != 0:
        raise ValueError(
            "Expected S plus complete t1/t3 pairs for multi-horizontal well "
            f"(got {target_count} target points)."
        )
    if order_mode not in {"strict_file_order", "sort_by_md"}:
        raise ValueError(
            f"Unsupported order_mode={order_mode!r}. "
            "Supported values: 'strict_file_order', 'sort_by_md'."
        )

    ordered_points = points
    if order_mode == "sort_by_md":
        ordered_points = tuple(sorted(points, key=lambda point: float(point.md)))

    md_values = [float(point.md) for point in ordered_points]
    if not all(math.isfinite(value) for value in md_values):
        raise ValueError("All MD values must be finite numbers for target mapping.")
    if not all(
        left + _MD_EPS < right
        for left, right in zip(md_values, md_values[1:], strict=False)
    ):
        raise ValueError(
            "Expected strictly increasing MD for points S and t1/t3 pairs "
            f"(got MD sequence: {', '.join(f'{value:.3f}' for value in md_values)})."
        )

    surface = Point3D(
        x=ordered_points[0].x,
        y=ordered_points[0].y,
        z=ordered_points[0].z,
    )
    pairs: list[tuple[Point3D, Point3D]] = []
    for index in range(1, len(ordered_points), 2):
        t1 = Point3D(
            x=ordered_points[index].x,
            y=ordered_points[index].y,
            z=ordered_points[index].z,
        )
        t3 = Point3D(
            x=ordered_points[index + 1].x,
            y=ordered_points[index + 1].y,
            z=ordered_points[index + 1].z,
        )
        pairs.append((t1, t3))
    return surface, tuple(pairs)


def welltrack_multi_horizontal_level_count(
    points: tuple[WelltrackPoint, ...],
) -> int:
    target_count = int(max(len(tuple(points)) - 1, 0))
    if target_count < 4 or target_count % 2 != 0:
        return 0
    return int(target_count // 2)


def _table_row_value(row: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _table_row_context(
    row_no: int,
    well_name: object = "",
    point_value: object = None,
) -> str:
    normalized_well_name = str(well_name).strip()
    if not normalized_well_name:
        return f"Строка {int(row_no)}"
    if _is_blank_table_value(point_value):
        return f"Строка {int(row_no)} (скважина '{normalized_well_name}')"
    return (
        f"Строка {int(row_no)} (скважина '{normalized_well_name}', "
        f"Point={_table_value_repr(point_value)})"
    )


def _table_value_repr(value: object, *, max_length: int = 80) -> str:
    rendered = repr(value)
    if len(rendered) <= int(max_length):
        return rendered
    return f"{rendered[: max(int(max_length) - 3, 0)]}..."


def _format_table_row_numbers(row_numbers: Iterable[int]) -> str:
    numbers = sorted({int(row_no) for row_no in row_numbers})
    if not numbers:
        return "—"
    ranges: list[str] = []
    range_start = numbers[0]
    range_end = numbers[0]
    for number in numbers[1:]:
        if number == range_end + 1:
            range_end = number
            continue
        ranges.append(
            str(range_start)
            if range_start == range_end
            else f"{range_start}-{range_end}"
        )
        range_start = range_end = number
    ranges.append(
        str(range_start)
        if range_start == range_end
        else f"{range_start}-{range_end}"
    )
    return ", ".join(ranges)


def _table_error_with_source_rows(
    message: str,
    *,
    row_numbers: Iterable[int],
) -> str:
    detail = str(message).strip()
    if detail.endswith("."):
        detail = detail[:-1]
    return (
        f"{detail}. Исходные строки этой скважины: "
        f"{_format_table_row_numbers(row_numbers)}."
    )


def _raise_table_validation_errors(errors: Iterable[str]) -> None:
    normalized_errors = [str(error).strip() for error in errors if str(error).strip()]
    if not normalized_errors:
        return
    shown_errors = normalized_errors[:_TABLE_VALIDATION_ERROR_LIMIT]
    lines = [
        f"Таблица точек содержит ошибки ({len(normalized_errors)}):",
        *(f"- {error}" for error in shown_errors),
    ]
    hidden_count = len(normalized_errors) - len(shown_errors)
    if hidden_count > 0:
        lines.append(f"- Ещё ошибок: {hidden_count}.")
    lines.append("Исправьте указанные строки и повторите импорт.")
    raise WelltrackParseError("\n".join(lines))


def _normalize_table_row(row: Mapping[object, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for raw_key, value in row.items():
        key = re.sub(r"[\s\-/(),.:]+", "_", str(raw_key).strip().lower()).strip("_")
        if not key:
            continue
        normalized[key] = value
    return normalized


def _is_blank_table_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    return bool(math.isnan(value)) if isinstance(value, float) else False


def normalize_welltrack_table_point_label(value: object) -> str | None:
    """Return the canonical display label for a supported target-table point."""

    point_name = _canonical_table_point_name(value)
    return None if point_name is None else _table_point_display_name(point_name)


def _canonical_table_point_name(value: object) -> str | None:
    if _is_blank_table_value(value):
        return None
    normalized = str(value).strip().casefold()
    point_name = _TABLE_POINT_ALIASES.get(normalized)
    if point_name is not None:
        return point_name
    pilot_match = _TABLE_PILOT_POINT_RE.fullmatch(normalized)
    if pilot_match is not None:
        return f"pl{int(pilot_match.group(1))}"
    target_sequence_match = _TABLE_TARGET_SEQUENCE_POINT_RE.fullmatch(normalized)
    if target_sequence_match is not None:
        return f"t{int(target_sequence_match.group(1))}"
    multi_match = _TABLE_MULTI_HORIZONTAL_POINT_RE.fullmatch(normalized)
    if multi_match is not None:
        return f"{int(multi_match.group(1))}_t{int(multi_match.group(2))}"
    return None


def _normalize_table_point_name(
    value: object,
    *,
    row_no: int,
    well_name: str = "",
    allow_pilot_points: bool = False,
) -> str:
    context = _table_row_context(row_no, well_name, value)
    if _is_blank_table_value(value):
        raise WelltrackParseError(
            f"{context}: поле Point пустое."
        )
    point_name = _canonical_table_point_name(value)
    if allow_pilot_points:
        if point_name == "wellhead" or (
            point_name is not None
            and _TABLE_PILOT_POINT_RE.fullmatch(point_name) is not None
        ):
            return point_name
        raise WelltrackParseError(
            f"{context}: метка точки не поддерживается для пилота. "
            "Ожидается S или S1, затем PL (то же, что PL1), PL2, ...; регистр и "
            "подчёркивание перед номером не учитываются."
        )

    if point_name is not None and _TABLE_PILOT_POINT_RE.fullmatch(point_name) is None:
        return point_name
    raise WelltrackParseError(
        f"{context}: метка точки не поддерживается. "
        "Ожидается S, t1, t2, t3, ... (S1 также допустимо); для "
        "многопластовой скважины используйте пары 1_t1/1_t3, "
        "2_t1/2_t3, ... . Метки PL/PL1, PL2, ... допустимы только для "
        "скважины с суффиксом _PL или PL. Регистр и необязательные "
        "подчёркивания не учитываются."
    )


def _table_point_display_name(point_name: str) -> str:
    pilot_match = _TABLE_PILOT_POINT_RE.match(str(point_name))
    if pilot_match is not None:
        return f"PL{int(pilot_match.group(1))}"
    target_sequence_match = _TABLE_TARGET_SEQUENCE_POINT_RE.match(str(point_name))
    if target_sequence_match is not None:
        return f"t{int(target_sequence_match.group(1))}"
    multi_match = _TABLE_MULTI_HORIZONTAL_POINT_RE.match(str(point_name))
    if multi_match is not None:
        return f"{int(multi_match.group(1))}_t{int(multi_match.group(2))}"
    return _TABLE_POINT_DISPLAY_LABELS.get(str(point_name), str(point_name))


def _table_point_md_index(point_name: str) -> float:
    pilot_match = _TABLE_PILOT_POINT_RE.match(str(point_name))
    if pilot_match is not None:
        return float(int(pilot_match.group(1)))
    target_sequence_match = _TABLE_TARGET_SEQUENCE_POINT_RE.match(str(point_name))
    if target_sequence_match is not None:
        return float(int(target_sequence_match.group(1)))
    multi_match = _TABLE_MULTI_HORIZONTAL_POINT_RE.match(str(point_name))
    if multi_match is not None:
        level = int(multi_match.group(1))
        point_kind = int(multi_match.group(2))
        return float(2 * (level - 1) + (1 if point_kind == 1 else 2))
    return float(_TABLE_POINT_ORDER.index(point_name))


def _is_table_pilot_well_name(well_name: object) -> bool:
    return is_pilot_name(well_name)


def _is_table_zbs_well_name(well_name: object) -> bool:
    return is_zbs_name(well_name) or is_alt_branch_name(well_name)


def _is_table_alt_branch_well_name(well_name: object) -> bool:
    return is_alt_branch_name(well_name)


def _table_well_name_key(well_name: object) -> str:
    return well_name_key(well_name)


def _table_parent_name_for_pilot_well_name(well_name: object) -> str:
    return parent_name_for_pilot(well_name)


def _table_pilot_parent_name_for_well_name(well_name: object) -> str:
    text = str(well_name).strip()
    if _is_table_pilot_well_name(text):
        return _table_parent_name_for_pilot_well_name(text)
    if _is_table_alt_branch_well_name(text):
        return parent_name_for_zbs(text)
    return text


def _table_pilot_surface_by_parent_key(
    grouped_points: Mapping[str, Mapping[str, WelltrackPoint]],
) -> dict[str, WelltrackPoint]:
    pilot_surface_by_parent_key: dict[str, WelltrackPoint] = {}
    for well_name, points_by_name in grouped_points.items():
        if not _is_table_pilot_well_name(well_name):
            continue
        surface = points_by_name.get("wellhead")
        if surface is None:
            continue
        pilot_surface_by_parent_key.setdefault(
            _table_well_name_key(_table_pilot_parent_name_for_well_name(well_name)),
            surface,
        )
    return pilot_surface_by_parent_key


def _table_points_with_inferred_pilot_surface(
    points_by_name: Mapping[str, WelltrackPoint],
    *,
    well_name: str,
    pilot_surface_by_parent_key: Mapping[str, WelltrackPoint],
) -> Mapping[str, WelltrackPoint]:
    if "wellhead" in points_by_name or _is_table_pilot_well_name(well_name):
        return points_by_name
    if _is_table_zbs_well_name(well_name) and not _is_table_alt_branch_well_name(
        well_name
    ):
        return points_by_name
    pilot_surface = pilot_surface_by_parent_key.get(
        _table_well_name_key(_table_pilot_parent_name_for_well_name(well_name))
    )
    if pilot_surface is None:
        return points_by_name
    return {"wellhead": pilot_surface, **points_by_name}


def _ordered_table_pilot_point_names(
    points_by_name: Mapping[str, WelltrackPoint],
    *,
    well_name: str,
) -> tuple[str, ...]:
    if "wellhead" not in points_by_name:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' отсутствуют точки: S."
        )
    pilot_indices = sorted(
        int(match.group(1))
        for point_name in points_by_name
        if (match := _TABLE_PILOT_POINT_RE.match(str(point_name))) is not None
    )
    if not pilot_indices:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' отсутствуют точки: PL1."
        )
    expected_indices = list(range(1, int(pilot_indices[-1]) + 1))
    missing = [index for index in expected_indices if index not in pilot_indices]
    if missing:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' отсутствуют точки: "
            f"{', '.join(f'PL{index}' for index in missing)}."
        )
    return ("wellhead", *(f"pl{index}" for index in expected_indices))


def _ordered_table_zbs_point_names(
    points_by_name: Mapping[str, WelltrackPoint],
    *,
    well_name: str,
) -> tuple[str, ...]:
    if _has_multi_horizontal_table_points(points_by_name):
        return _ordered_table_zbs_multi_horizontal_point_names(
            points_by_name,
            well_name=well_name,
        )

    missing = [name for name in ("t1", "t3") if name not in points_by_name]
    if missing:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для бокового ствола от факта "
            f"'{well_name}' отсутствуют точки: "
            f"{', '.join(_table_point_display_name(name) for name in missing)}."
        )
    extra = [
        _table_point_display_name(name)
        for name in points_by_name
        if name not in {"t1", "t3"}
    ]
    if extra:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для бокового ствола от факта "
            f"'{well_name}' используйте только точки t1 и t3 без S либо полные "
            "многопластовые пары 1_t1/1_t3, 2_t1/2_t3, ... без S. "
            f"Лишние точки: {', '.join(extra)}."
        )
    return ("t1", "t3")


def _ordered_table_zbs_multi_horizontal_point_names(
    points_by_name: Mapping[str, WelltrackPoint],
    *,
    well_name: str,
) -> tuple[str, ...]:
    forbidden = sorted(
        _table_point_display_name(point_name)
        for point_name in points_by_name
        if point_name == "wellhead"
        or _TABLE_MULTI_HORIZONTAL_POINT_RE.match(str(point_name)) is None
    )
    if forbidden:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для многопластового бокового ствола от факта "
            f"'{well_name}' используйте только пары 1_t1/1_t3, 2_t1/2_t3, ... "
            f"без S и обычных точек. Лишние точки: {', '.join(forbidden)}."
        )

    levels = sorted(
        int(match.group(1))
        for point_name in points_by_name
        if (match := _TABLE_MULTI_HORIZONTAL_POINT_RE.match(str(point_name))) is not None
    )
    if not levels:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для многопластового бокового ствола от факта "
            f"'{well_name}' отсутствует точка 1_t1."
        )
    max_level = int(max(levels))
    missing: list[str] = []
    ordered: list[str] = []
    for level in range(1, max_level + 1):
        for suffix in ("t1", "t3"):
            point_name = f"{level}_{suffix}"
            if point_name not in points_by_name:
                missing.append(point_name)
            else:
                ordered.append(point_name)
    if missing:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для многопластового бокового ствола от факта "
            f"'{well_name}' отсутствуют точки: {', '.join(missing)}."
        )
    return tuple(ordered)


def _has_multi_horizontal_table_points(
    points_by_name: Mapping[str, WelltrackPoint],
) -> bool:
    return any(
        _TABLE_MULTI_HORIZONTAL_POINT_RE.match(str(point_name)) is not None
        for point_name in points_by_name
    )


def _ordered_table_target_sequence_point_names(
    points_by_name: Mapping[str, WelltrackPoint],
    *,
    well_name: str,
) -> tuple[str, ...] | None:
    if "wellhead" not in points_by_name:
        return None
    target_indices = sorted(
        int(match.group(1))
        for point_name in points_by_name
        if (match := _TABLE_TARGET_SEQUENCE_POINT_RE.match(str(point_name))) is not None
    )
    if not target_indices:
        return None
    if _has_multi_horizontal_table_points(points_by_name):
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' нельзя смешивать последовательность t1/t2/t3/... "
            "c многопластовыми парами N_t1/N_t3."
        )
    if target_indices == [1, 3] and len(points_by_name) == 3:
        return None
    if len(target_indices) < 2:
        return None
    max_index = int(target_indices[-1])
    has_horizontal_start = 2 in target_indices
    if has_horizontal_start and max_index < 3:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' при наличии t2 обязательна точка t3."
        )
    expected_indices = (
        list(range(1, max_index + 1))
        if has_horizontal_start
        else [1, *range(3, max_index + 1)]
    )
    missing = [index for index in expected_indices if index not in target_indices]
    if missing:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' отсутствуют точки последовательности: "
            f"{', '.join(f't{index}' for index in missing)}."
        )
    allowed = {"wellhead", *(f"t{index}" for index in expected_indices)}
    extra = sorted(
        _table_point_display_name(point_name)
        for point_name in points_by_name
        if point_name not in allowed
    )
    if extra:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' используйте либо S/t1/t3, либо последовательность "
            "S/t1/t2/t3/... или S/t1/t3/t4/... без посторонних точек. "
            f"Лишние точки: {', '.join(extra)}."
        )
    return ("wellhead", *(f"t{index}" for index in expected_indices))


def _ordered_table_multi_horizontal_point_names(
    points_by_name: Mapping[str, WelltrackPoint],
    *,
    well_name: str,
) -> tuple[str, ...]:
    if "wellhead" not in points_by_name:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для скважины "
            f"'{well_name}' отсутствуют точки: S."
        )
    forbidden = sorted(
        _table_point_display_name(point_name)
        for point_name in points_by_name
        if point_name != "wellhead"
        and _TABLE_MULTI_HORIZONTAL_POINT_RE.match(str(point_name)) is None
    )
    if forbidden:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для многопластовой скважины "
            f"'{well_name}' используйте только пары 1_t1/1_t3, 2_t1/2_t3, ... "
            f"без других точек. Лишние точки: {', '.join(forbidden)}."
        )
    levels = sorted(
        int(match.group(1))
        for point_name in points_by_name
        if (match := _TABLE_MULTI_HORIZONTAL_POINT_RE.match(str(point_name))) is not None
    )
    if not levels:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для многопластовой скважины "
            f"'{well_name}' отсутствует точка 1_t1."
        )
    max_level = int(max(levels))
    missing: list[str] = []
    ordered = ["wellhead"]
    for level in range(1, max_level + 1):
        for suffix in ("t1", "t3"):
            point_name = f"{level}_{suffix}"
            if point_name not in points_by_name:
                missing.append(point_name)
            else:
                ordered.append(point_name)
    if missing:
        raise WelltrackParseError(
            "Табличный WELLTRACK: для многопластовой скважины "
            f"'{well_name}' отсутствуют точки: {', '.join(missing)}."
        )
    return tuple(ordered)


def _coerce_table_float(
    value: object,
    *,
    field_name: str,
    row_no: int,
    well_name: str = "",
    point_value: object = None,
) -> float:
    context = _table_row_context(row_no, well_name, point_value)
    if _is_blank_table_value(value):
        raise WelltrackParseError(
            f"{context}: поле {field_name} пустое."
        )
    try:
        number = float(_normalize_table_float_text(value))
    except (TypeError, ValueError) as exc:
        raise WelltrackParseError(
            f"{context}: поле {field_name} должно быть числом; "
            f"получено {_table_value_repr(value)}."
        ) from exc
    if not math.isfinite(number):
        raise WelltrackParseError(
            f"{context}: поле {field_name} должно быть конечным числом; "
            f"получено {_table_value_repr(value)}."
        )
    return number


def _normalize_table_float_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    text = (
        text.replace("\u00A0", "")
        .replace("\u202F", "")
        .replace(" ", "")
        .replace("'", "")
    )
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    return text


def _parse_well_name(rest: str, line_no: int) -> tuple[str, str]:
    tail = rest.strip()
    if not tail:
        raise WelltrackParseError(f"Missing well name after WELLTRACK at line {line_no}.")

    if tail[0] in {"'", '"'}:
        quote = tail[0]
        end_quote_idx = tail.find(quote, 1)
        if end_quote_idx < 0:
            raise WelltrackParseError(f"Unclosed quoted well name at line {line_no}.")
        name = tail[1:end_quote_idx].strip()
        remainder = tail[end_quote_idx + 1:].strip()
        if not name:
            raise WelltrackParseError(f"Empty quoted well name at line {line_no}.")
        return name, remainder

    parts = tail.split(maxsplit=1)
    name = parts[0].strip()
    if name in {"/", ";"}:
        raise WelltrackParseError(f"Missing well name after WELLTRACK at line {line_no}.")
    remainder = parts[1].strip() if len(parts) > 1 else ""
    return name, remainder


def _consume_tail_tokens(tail: str, numeric_tokens: list[str], finalize: Callable[[], None]) -> None:
    normalized = tail.replace("/", " / ").replace(";", " ; ")
    for token in normalized.split():
        if token in {"/", ";"}:
            finalize()
            continue
        numeric_tokens.append(token)


def _validate_record_md(points: list[WelltrackPoint], well_name: str) -> None:
    if not points:
        return
    for index, point in enumerate(points, start=1):
        if not math.isfinite(float(point.md)):
            raise WelltrackParseError(
                f"WELLTRACK '{well_name}': MD at point #{index} must be finite."
            )
        if index == 1:
            continue
        previous_md = float(points[index - 2].md)
        current_md = float(point.md)
        if current_md + _MD_EPS < previous_md:
            raise WelltrackParseError(
                f"WELLTRACK '{well_name}': MD must be non-decreasing by point order. "
                f"Found MD[{index - 1}]={previous_md:.3f} > MD[{index}]={current_md:.3f}."
            )
