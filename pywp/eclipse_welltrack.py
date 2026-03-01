from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Literal

from pywp.models import Point3D

_WELLTRACK_RE = re.compile(r"^\s*WELLTRACK\b(.*)$", flags=re.IGNORECASE)
DEFAULT_WELLTRACK_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1251", "latin-1")
_MD_EPS = 1e-9


class WelltrackParseError(ValueError):
    pass


@dataclass(frozen=True)
class WelltrackPoint:
    x: float
    y: float
    z: float
    md: float


@dataclass(frozen=True)
class WelltrackRecord:
    name: str
    points: tuple[WelltrackPoint, ...]


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
        remainder = tail[end_quote_idx + 1 :].strip()
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
