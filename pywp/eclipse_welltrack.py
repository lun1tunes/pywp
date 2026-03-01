from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from pywp.models import Point3D

_WELLTRACK_RE = re.compile(r"^\s*WELLTRACK\b(.*)$", flags=re.IGNORECASE)


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


def welltrack_points_to_targets(points: tuple[WelltrackPoint, ...]) -> tuple[Point3D, Point3D, Point3D]:
    if len(points) != 3:
        raise ValueError(f"Expected exactly 3 points (S, t1, t3), got {len(points)}.")
    # By project contract, WELLTRACK points are provided as S, t1, t3.
    # MD is parsed for compatibility, but not used for target mapping.
    surface = Point3D(x=points[0].x, y=points[0].y, z=points[0].z)
    t1 = Point3D(x=points[1].x, y=points[1].y, z=points[1].z)
    t3 = Point3D(x=points[2].x, y=points[2].y, z=points[2].z)
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
