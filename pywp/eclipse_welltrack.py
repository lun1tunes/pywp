from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Literal, Mapping

from pydantic import field_validator
from pywp.models import Point3D
from pywp.pydantic_base import FrozenModel, coerce_model_like

_WELLTRACK_RE = re.compile(r"^\s*WELLTRACK\b(.*)$", flags=re.IGNORECASE)
DEFAULT_WELLTRACK_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1251", "latin-1")
_MD_EPS = 1e-9
_TABLE_POINT_ALIASES: dict[str, str] = {
    "s": "wellhead",
    "surface": "wellhead",
    "wellhead": "wellhead",
    "well_head": "wellhead",
    "well head": "wellhead",
    "wh": "wellhead",
    "t1": "t1",
    "entry": "t1",
    "entry point": "t1",
    "t3": "t3",
    "target": "t3",
    "end": "t3",
}
_TABLE_PILOT_POINT_RE = re.compile(r"^(?:pl|p)([1-9]\d*)$", flags=re.IGNORECASE)
_TABLE_POINT_ORDER: tuple[str, ...] = ("wellhead", "t1", "t3")
_TABLE_POINT_DISPLAY_LABELS: dict[str, str] = {
    "wellhead": "S",
    "t1": "t1",
    "t3": "t3",
}


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
                f"WELLTRACK '{current_name}': expected X Y Z MD groups of 4 values, "
                f"got {len(numeric_tokens)} values at line {line_no}."
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
                    f"WELLTRACK '{current_name}': failed to parse numeric value near line {line_no}: {exc}"
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
    well_order: list[str] = []
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
        well_name = str(name_raw).strip()
        if not well_name:
            raise WelltrackParseError(
                f"Табличный WELLTRACK: пустое имя скважины в строке {row_no}."
            )

        point_name = _normalize_table_point_name(
            point_raw,
            row_no=row_no,
            allow_pilot_points=_is_table_pilot_well_name(well_name),
        )
        x = _coerce_table_float(x_raw, field_name="X", row_no=row_no)
        y = _coerce_table_float(y_raw, field_name="Y", row_no=row_no)
        z = _coerce_table_float(z_raw, field_name="Z", row_no=row_no)

        if well_name not in grouped_points:
            grouped_points[well_name] = {}
            well_order.append(well_name)

        if point_name in grouped_points[well_name]:
            raise WelltrackParseError(
                "Табличный WELLTRACK: дублирующаяся точка "
                f"'{point_name}' для скважины '{well_name}' в строке {row_no}."
            )

        md_index = _table_point_md_index(point_name)
        grouped_points[well_name][point_name] = WelltrackPoint(
            x=x,
            y=y,
            z=z,
            md=md_index,
        )

    if not has_non_empty_row:
        raise WelltrackParseError(
            "Табличный WELLTRACK пуст. Вставьте строки в формате "
            "Wellname / Point / X / Y / Z."
        )

    records: list[WelltrackRecord] = []
    for well_name in well_order:
        points_by_name = grouped_points[well_name]
        if _is_table_pilot_well_name(well_name):
            ordered_names = _ordered_table_pilot_point_names(
                points_by_name,
                well_name=well_name,
            )
            ordered_points = tuple(points_by_name[name] for name in ordered_names)
            _validate_record_md(points=list(ordered_points), well_name=well_name)
            records.append(WelltrackRecord(name=well_name, points=ordered_points))
            continue

        missing = [name for name in _TABLE_POINT_ORDER if name not in points_by_name]
        if missing:
            raise WelltrackParseError(
                "Табличный WELLTRACK: для скважины "
                f"'{well_name}' отсутствуют точки: "
                f"{', '.join(_table_point_display_name(name) for name in missing)}."
            )
        ordered_points = tuple(points_by_name[name] for name in _TABLE_POINT_ORDER)
        _validate_record_md(points=list(ordered_points), well_name=well_name)
        records.append(WelltrackRecord(name=well_name, points=ordered_points))

    return records


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


def _table_row_value(row: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return None


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
    return bool(math.isnan(value)) if isinstance(value, float) else False


def _normalize_table_point_name(
    value: object,
    *,
    row_no: int,
    allow_pilot_points: bool = False,
) -> str:
    if _is_blank_table_value(value):
        raise WelltrackParseError(
            f"Табличный WELLTRACK: пустое значение Point в строке {row_no}."
        )
    normalized = str(value).strip().lower()
    if allow_pilot_points:
        if _TABLE_POINT_ALIASES.get(normalized) == "wellhead":
            return "wellhead"
        pilot_match = _TABLE_PILOT_POINT_RE.match(normalized)
        if pilot_match is not None:
            return f"pl{int(pilot_match.group(1))}"
        raise WelltrackParseError(
            "Табличный WELLTRACK: unsupported Point="
            f"{value!r} в строке {row_no}. "
            "Для пилота ожидается S, PL1, PL2, ..."
        )

    point_name = _TABLE_POINT_ALIASES.get(normalized)
    if point_name is None:
        raise WelltrackParseError(
            "Табличный WELLTRACK: unsupported Point="
            f"{value!r} в строке {row_no}. "
            "Ожидается S, t1 или t3. Для пилота используйте имя wellname_PL "
            "и точки S, PL1, PL2, ..."
        )
    return point_name


def _table_point_display_name(point_name: str) -> str:
    pilot_match = _TABLE_PILOT_POINT_RE.match(str(point_name))
    if pilot_match is not None:
        return f"PL{int(pilot_match.group(1))}"
    return _TABLE_POINT_DISPLAY_LABELS.get(str(point_name), str(point_name))


def _table_point_md_index(point_name: str) -> float:
    pilot_match = _TABLE_PILOT_POINT_RE.match(str(point_name))
    if pilot_match is not None:
        return float(int(pilot_match.group(1)))
    return float(_TABLE_POINT_ORDER.index(point_name))


def _is_table_pilot_well_name(well_name: object) -> bool:
    return str(well_name).strip().upper().endswith("_PL")


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


def _coerce_table_float(value: object, *, field_name: str, row_no: int) -> float:
    if _is_blank_table_value(value):
        raise WelltrackParseError(
            f"Табличный WELLTRACK: пустое значение {field_name} в строке {row_no}."
        )
    try:
        number = float(_normalize_table_float_text(value))
    except (TypeError, ValueError) as exc:
        raise WelltrackParseError(
            f"Табличный WELLTRACK: {field_name} в строке {row_no} не является числом."
        ) from exc
    if not math.isfinite(number):
        raise WelltrackParseError(
            f"Табличный WELLTRACK: {field_name} в строке {row_no} должно быть конечным числом."
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
