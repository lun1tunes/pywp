from __future__ import annotations

from dataclasses import dataclass

TRAJECTORY_SAME_DIRECTION = "same_direction"
TRAJECTORY_REVERSE_DIRECTION = "reverse_direction"

COMPLEXITY_ORDINARY = "ordinary"
COMPLEXITY_COMPLEX = "complex"
COMPLEXITY_VERY_COMPLEX = "very_complex"

_COMPLEXITY_RANK = {
    COMPLEXITY_ORDINARY: 0,
    COMPLEXITY_COMPLEX: 1,
    COMPLEXITY_VERY_COMPLEX: 2,
}


@dataclass(frozen=True)
class DepthClassificationAnchor:
    gv_m: float
    reverse_min_m: float
    reverse_max_m: float
    ordinary_offset_max_m: float
    complex_offset_max_m: float
    hold_ordinary_max_deg: float
    hold_complex_max_deg: float
    reverse_allowed: bool


@dataclass(frozen=True)
class InterpolatedLimits:
    gv_m: float
    reverse_min_m: float
    reverse_max_m: float
    ordinary_offset_max_m: float
    complex_offset_max_m: float
    hold_ordinary_max_deg: float
    hold_complex_max_deg: float
    reverse_allowed: bool


@dataclass(frozen=True)
class WellClassification:
    trajectory_type: str
    complexity: str
    complexity_by_offset: str
    complexity_by_hold: str
    limits: InterpolatedLimits


CLASSIFICATION_ANCHORS: tuple[DepthClassificationAnchor, ...] = (
    # GV=1000: explicit business rule says only same-direction targets are allowed.
    # Offset and HOLD complexity thresholds for GV=1000 are not specified in source rules.
    # We keep the same thresholds as GV=2000 to avoid artificial discontinuity in interpolation.
    DepthClassificationAnchor(
        gv_m=1000.0,
        reverse_min_m=0.0,
        reverse_max_m=0.0,
        ordinary_offset_max_m=1250.0,
        complex_offset_max_m=1900.0,
        hold_ordinary_max_deg=37.0,
        hold_complex_max_deg=55.0,
        reverse_allowed=False,
    ),
    DepthClassificationAnchor(
        gv_m=2000.0,
        reverse_min_m=120.0,
        reverse_max_m=550.0,
        ordinary_offset_max_m=1250.0,
        complex_offset_max_m=1900.0,
        hold_ordinary_max_deg=37.0,
        hold_complex_max_deg=55.0,
        reverse_allowed=True,
    ),
    DepthClassificationAnchor(
        gv_m=3000.0,
        reverse_min_m=0.0,
        reverse_max_m=700.0,
        ordinary_offset_max_m=2000.0,
        complex_offset_max_m=3200.0,
        hold_ordinary_max_deg=37.0,
        hold_complex_max_deg=55.0,
        reverse_allowed=True,
    ),
    DepthClassificationAnchor(
        gv_m=3600.0,
        reverse_min_m=0.0,
        reverse_max_m=700.0,
        ordinary_offset_max_m=2300.0,
        complex_offset_max_m=3500.0,
        hold_ordinary_max_deg=35.0,
        hold_complex_max_deg=50.0,
        reverse_allowed=True,
    ),
)


def _clamp_gv(gv_m: float) -> float:
    low = CLASSIFICATION_ANCHORS[0].gv_m
    high = CLASSIFICATION_ANCHORS[-1].gv_m
    return float(min(max(gv_m, low), high))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _find_bracket(gv_m: float) -> tuple[DepthClassificationAnchor, DepthClassificationAnchor, float]:
    gv = _clamp_gv(gv_m)
    for left, right in zip(CLASSIFICATION_ANCHORS[:-1], CLASSIFICATION_ANCHORS[1:]):
        if left.gv_m <= gv <= right.gv_m:
            span = right.gv_m - left.gv_m
            t = 0.0 if span <= 1e-9 else (gv - left.gv_m) / span
            return left, right, float(t)

    anchor = CLASSIFICATION_ANCHORS[-1]
    return anchor, anchor, 0.0


def interpolate_limits(gv_m: float) -> InterpolatedLimits:
    left, right, t = _find_bracket(gv_m)
    gv = _clamp_gv(gv_m)
    reverse_allowed = left.reverse_allowed or right.reverse_allowed
    if gv <= CLASSIFICATION_ANCHORS[0].gv_m:
        reverse_allowed = CLASSIFICATION_ANCHORS[0].reverse_allowed

    return InterpolatedLimits(
        gv_m=gv,
        reverse_min_m=_lerp(left.reverse_min_m, right.reverse_min_m, t),
        reverse_max_m=_lerp(left.reverse_max_m, right.reverse_max_m, t),
        ordinary_offset_max_m=_lerp(left.ordinary_offset_max_m, right.ordinary_offset_max_m, t),
        complex_offset_max_m=_lerp(left.complex_offset_max_m, right.complex_offset_max_m, t),
        hold_ordinary_max_deg=_lerp(left.hold_ordinary_max_deg, right.hold_ordinary_max_deg, t),
        hold_complex_max_deg=_lerp(left.hold_complex_max_deg, right.hold_complex_max_deg, t),
        reverse_allowed=reverse_allowed,
    )


def classify_trajectory_type(gv_m: float, horizontal_offset_t1_m: float) -> str:
    limits = interpolate_limits(gv_m=gv_m)
    if not limits.reverse_allowed:
        return TRAJECTORY_SAME_DIRECTION
    if limits.reverse_max_m <= limits.reverse_min_m:
        return TRAJECTORY_SAME_DIRECTION
    if limits.reverse_min_m <= horizontal_offset_t1_m <= limits.reverse_max_m:
        return TRAJECTORY_REVERSE_DIRECTION
    return TRAJECTORY_SAME_DIRECTION


def _complexity_from_offset(horizontal_offset_t1_m: float, limits: InterpolatedLimits) -> str:
    if horizontal_offset_t1_m <= limits.ordinary_offset_max_m:
        return COMPLEXITY_ORDINARY
    if horizontal_offset_t1_m <= limits.complex_offset_max_m:
        return COMPLEXITY_COMPLEX
    return COMPLEXITY_VERY_COMPLEX


def _complexity_from_hold(hold_inc_deg: float, limits: InterpolatedLimits) -> str:
    if hold_inc_deg <= limits.hold_ordinary_max_deg:
        return COMPLEXITY_ORDINARY
    if hold_inc_deg <= limits.hold_complex_max_deg:
        return COMPLEXITY_COMPLEX
    return COMPLEXITY_VERY_COMPLEX


def classify_well(gv_m: float, horizontal_offset_t1_m: float, hold_inc_deg: float) -> WellClassification:
    limits = interpolate_limits(gv_m=gv_m)
    trajectory_type = classify_trajectory_type(gv_m=gv_m, horizontal_offset_t1_m=horizontal_offset_t1_m)
    complexity_by_offset = _complexity_from_offset(horizontal_offset_t1_m=horizontal_offset_t1_m, limits=limits)
    complexity_by_hold = _complexity_from_hold(hold_inc_deg=hold_inc_deg, limits=limits)
    complexity = (
        complexity_by_offset
        if _COMPLEXITY_RANK[complexity_by_offset] >= _COMPLEXITY_RANK[complexity_by_hold]
        else complexity_by_hold
    )
    return WellClassification(
        trajectory_type=trajectory_type,
        complexity=complexity,
        complexity_by_offset=complexity_by_offset,
        complexity_by_hold=complexity_by_hold,
        limits=limits,
    )


def trajectory_type_label(code: str) -> str:
    if code == TRAJECTORY_REVERSE_DIRECTION:
        return "Цели в обратном направлении"
    return "Цели в одном направлении"


def complexity_label(code: str) -> str:
    if code == COMPLEXITY_COMPLEX:
        return "Сложная"
    if code == COMPLEXITY_VERY_COMPLEX:
        return "Очень сложная"
    return "Обычная"


def reference_table_rows() -> list[dict[str, str | float]]:
    return [
        {
            "ГВ, м": 1000.0,
            "Отход t1 для обратного направления, м": "Не допускается",
            "Отход t1: Обычная (до), м": "—",
            "Отход t1: Сложная (до), м": "—",
            "ЗУ HOLD: Обычная (до), deg": "—",
            "ЗУ HOLD: Сложная (до), deg": "—",
        },
        {
            "ГВ, м": 2000.0,
            "Отход t1 для обратного направления, м": "120-550",
            "Отход t1: Обычная (до), м": 1250.0,
            "Отход t1: Сложная (до), м": 1900.0,
            "ЗУ HOLD: Обычная (до), deg": 37.0,
            "ЗУ HOLD: Сложная (до), deg": 55.0,
        },
        {
            "ГВ, м": 3000.0,
            "Отход t1 для обратного направления, м": "0-700",
            "Отход t1: Обычная (до), м": 2000.0,
            "Отход t1: Сложная (до), м": 3200.0,
            "ЗУ HOLD: Обычная (до), deg": 37.0,
            "ЗУ HOLD: Сложная (до), deg": 55.0,
        },
        {
            "ГВ, м": 3600.0,
            "Отход t1 для обратного направления, м": "0-700",
            "Отход t1: Обычная (до), м": 2300.0,
            "Отход t1: Сложная (до), м": 3500.0,
            "ЗУ HOLD: Обычная (до), deg": 35.0,
            "ЗУ HOLD: Сложная (до), deg": 50.0,
        },
    ]
