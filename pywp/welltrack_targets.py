from __future__ import annotations

import re
from dataclasses import dataclass

from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackRecord,
    welltrack_multi_horizontal_level_count,
    welltrack_points_to_target_pairs,
)
from pywp.models import Point3D
from pywp.pilot_wells import (
    is_pilot_record,
    is_zbs_record,
    zbs_multi_horizontal_level_count,
)

_TARGET_SEQUENCE_POINT_RE = re.compile(
    r"^t_?([1-9]\d*)$",
    flags=re.IGNORECASE,
)
_MULTI_HORIZONTAL_POINT_RE = re.compile(
    r"^([1-9]\d*)_?t_?([13])$",
    flags=re.IGNORECASE,
)
_SURFACE_POINT_LABELS = {
    "s",
    "s1",
    "s_1",
    "surface",
    "wellhead",
    "well_head",
    "well head",
    "wh",
}


@dataclass(frozen=True)
class OrdinaryWelltrackTargetLayout:
    surface: Point3D
    t1: Point3D
    t3: Point3D
    final_target: Point3D
    target_pairs: tuple[tuple[Point3D, Point3D], ...]
    target_sequence: tuple[Point3D, ...]
    target_sequence_numbers: tuple[int, ...]
    target_sequence_has_horizontal_start: bool
    target_points: tuple[Point3D, ...]
    target_labels: tuple[str, ...]


def record_point_labels(record: WelltrackRecord) -> tuple[str, ...]:
    explicit = tuple(
        str(label).strip() for label in getattr(record, "point_labels", ()) or ()
    )
    if explicit and len(explicit) == len(tuple(record.points)):
        return explicit

    point_count = len(tuple(record.points))
    if point_count <= 0:
        return ()
    if is_pilot_record(record):
        return ("S", *(f"PL{index}" for index in range(1, point_count)))
    if is_zbs_record(record):
        if point_count == 2:
            return ("t1", "t3")
        if point_count >= 4 and point_count % 2 == 0:
            labels: list[str] = []
            for level_index in range(1, point_count // 2 + 1):
                labels.extend([f"{level_index}_t1", f"{level_index}_t3"])
            return tuple(labels)
        return tuple(f"P{index + 1}" for index in range(point_count))

    target_count = point_count - 1
    if target_count >= 4 and target_count % 2 == 0:
        labels = ["S"]
        for level_index in range(1, target_count // 2 + 1):
            labels.extend([f"{level_index}_t1", f"{level_index}_t3"])
        return tuple(labels)
    if point_count == 3:
        return ("S", "t1", "t3")
    return ("S", *(f"P{index}" for index in range(1, point_count)))


def record_is_ordinary_target_sequence(record: WelltrackRecord) -> bool:
    if is_pilot_record(record) or is_zbs_record(record):
        return False
    points = tuple(record.points)
    if len(points) < 3:
        return False
    labels = record_point_labels(record)
    if (
        len(labels) != len(points)
        or str(labels[0]).strip().casefold() not in _SURFACE_POINT_LABELS
    ):
        return False
    indices = _simple_target_sequence_indices(labels)
    if indices is None:
        return False
    if len(indices) >= 3 and indices == tuple(range(1, len(indices) + 1)):
        return True
    # t2 is optional.  When it is omitted, t3 remains the first productive
    # target and the following targets must still be contiguous.
    return len(indices) >= 3 and indices == (1, *range(3, indices[-1] + 1))


def record_multi_horizontal_level_count(record: WelltrackRecord) -> int:
    if is_zbs_record(record):
        return zbs_multi_horizontal_level_count(tuple(record.points))
    if record_is_ordinary_target_sequence(record):
        return 0
    return welltrack_multi_horizontal_level_count(tuple(record.points))


def ordinary_record_target_layout(
    record: WelltrackRecord,
) -> OrdinaryWelltrackTargetLayout:
    if is_pilot_record(record) or is_zbs_record(record):
        raise ValueError(
            "ordinary_record_target_layout expects a non-pilot, non-ZBS record."
        )

    points = tuple(record.points)
    if len(points) < 3:
        raise ValueError(f"Expected at least 3 points (S, t1, t3), got {len(points)}.")

    target_points = tuple(_point3d_from_welltrack_point(point) for point in points)
    target_labels = record_point_labels(record)
    surface = target_points[0]

    simple_indices = _simple_target_sequence_indices(target_labels)
    labels_are_explicit = bool(getattr(record, "point_labels", ()))
    if labels_are_explicit:
        valid_classic = simple_indices == (1, 3)
        valid_horizontal = (
            simple_indices is not None
            and len(simple_indices) >= 3
            and simple_indices == tuple(range(1, len(simple_indices) + 1))
        )
        valid_without_t2 = (
            simple_indices is not None
            and len(simple_indices) >= 3
            and simple_indices == (1, *range(3, simple_indices[-1] + 1))
        )
        valid_multi_horizontal = _explicit_multi_horizontal_labels_are_valid(
            target_labels
        )
        if not (
            valid_classic
            or valid_horizontal
            or valid_without_t2
            or valid_multi_horizontal
        ):
            raise ValueError(
                "Для последовательности целей используйте либо S/t1/t3, "
                "либо S/t1/t2/t3/... или S/t1/t3/t4/... без пропусков; "
                "для многопластовой скважины используйте полные пары "
                "S/1_t1/1_t3/2_t1/2_t3/...."
            )

    if record_is_ordinary_target_sequence(record):
        sequence = target_points[1:]
        sequence_labels = tuple(
            str(label).strip().casefold() for label in target_labels[1:]
        )
        sequence_numbers = tuple(
            int(match.group(1))
            for label in sequence_labels
            if (match := _TARGET_SEQUENCE_POINT_RE.match(label)) is not None
        )
        has_horizontal_start = len(sequence_labels) >= 2 and sequence_labels[:2] == (
            "t1",
            "t2",
        )
        return OrdinaryWelltrackTargetLayout(
            surface=surface,
            t1=sequence[0],
            t3=sequence[1],
            final_target=sequence[-1],
            target_pairs=(),
            target_sequence=sequence,
            target_sequence_numbers=sequence_numbers,
            target_sequence_has_horizontal_start=has_horizontal_start,
            target_points=target_points,
            target_labels=target_labels,
        )

    surface, target_pairs = welltrack_points_to_target_pairs(points)
    t1, t3 = target_pairs[0]
    return OrdinaryWelltrackTargetLayout(
        surface=surface,
        t1=t1,
        t3=t3,
        final_target=target_pairs[-1][1],
        target_pairs=target_pairs,
        target_sequence=(),
        target_sequence_numbers=(),
        target_sequence_has_horizontal_start=False,
        target_points=target_points,
        target_labels=target_labels,
    )


def target_sequence_points_from_record(
    record: WelltrackRecord,
) -> tuple[Point3D, ...]:
    if not record_is_ordinary_target_sequence(record):
        return ()
    return tuple(
        _point3d_from_welltrack_point(point) for point in tuple(record.points)[1:]
    )


def _point3d_from_welltrack_point(point: WelltrackPoint) -> Point3D:
    return Point3D(x=float(point.x), y=float(point.y), z=float(point.z))


def _simple_target_sequence_indices(
    labels: tuple[str, ...],
) -> tuple[int, ...] | None:
    if (
        len(labels) < 2
        or str(labels[0]).strip().casefold() not in _SURFACE_POINT_LABELS
    ):
        return None
    indices: list[int] = []
    for label in labels[1:]:
        match = _TARGET_SEQUENCE_POINT_RE.match(str(label).strip())
        if match is None:
            return None
        indices.append(int(match.group(1)))
    return tuple(indices)


def _explicit_multi_horizontal_labels_are_valid(labels: tuple[str, ...]) -> bool:
    """Recognize the only non-sequential explicit layout for ordinary wells."""

    if (
        len(labels) < 3
        or str(labels[0]).strip().casefold() not in _SURFACE_POINT_LABELS
    ):
        return False
    normalized = tuple(str(label).strip().casefold() for label in labels[1:])
    matches = [_MULTI_HORIZONTAL_POINT_RE.fullmatch(label) for label in normalized]
    if any(match is None for match in matches) or len(normalized) % 2 != 0:
        return False
    canonical = tuple(
        f"{int(match.group(1))}_t{int(match.group(2))}"
        for match in matches
        if match is not None
    )
    level_count = len(normalized) // 2
    expected = tuple(
        f"{level}_{suffix}"
        for level in range(1, level_count + 1)
        for suffix in ("t1", "t3")
    )
    return canonical == expected
