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
class DepthClassificationRule:
    """Single source of truth for business classification rows."""

    gv_m: float
    reverse_min_m: float
    reverse_max_m: float
    reverse_allowed: bool
    ordinary_offset_max_m: float | None
    complex_offset_max_m: float | None
    hold_ordinary_max_deg: float | None
    hold_complex_max_deg: float | None


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


CLASSIFICATION_RULES: tuple[DepthClassificationRule, ...] = (
    DepthClassificationRule(
        gv_m=1000.0,
        reverse_min_m=0.0,
        reverse_max_m=0.0,
        reverse_allowed=False,
        ordinary_offset_max_m=None,
        complex_offset_max_m=None,
        hold_ordinary_max_deg=None,
        hold_complex_max_deg=None,
    ),
    DepthClassificationRule(
        gv_m=2000.0,
        reverse_min_m=120.0,
        reverse_max_m=550.0,
        reverse_allowed=True,
        ordinary_offset_max_m=1250.0,
        complex_offset_max_m=1900.0,
        hold_ordinary_max_deg=37.0,
        hold_complex_max_deg=55.0,
    ),
    DepthClassificationRule(
        gv_m=3000.0,
        reverse_min_m=0.0,
        reverse_max_m=700.0,
        reverse_allowed=True,
        ordinary_offset_max_m=2000.0,
        complex_offset_max_m=3200.0,
        hold_ordinary_max_deg=37.0,
        hold_complex_max_deg=55.0,
    ),
    DepthClassificationRule(
        gv_m=3600.0,
        reverse_min_m=0.0,
        reverse_max_m=700.0,
        reverse_allowed=True,
        ordinary_offset_max_m=2300.0,
        complex_offset_max_m=3500.0,
        hold_ordinary_max_deg=35.0,
        hold_complex_max_deg=50.0,
    ),
)


def _optional_rule_value(rule: DepthClassificationRule, field_name: str) -> float | None:
    return getattr(rule, field_name)


def _resolve_rule_column(field_name: str) -> list[float]:
    resolved: list[float] = []
    for idx, rule in enumerate(CLASSIFICATION_RULES):
        value = _optional_rule_value(rule, field_name)
        if value is not None:
            resolved.append(float(value))
            continue

        fallback: float | None = None
        for right in CLASSIFICATION_RULES[idx + 1 :]:
            candidate = _optional_rule_value(right, field_name)
            if candidate is not None:
                fallback = float(candidate)
                break
        if fallback is None:
            for left in reversed(CLASSIFICATION_RULES[:idx]):
                candidate = _optional_rule_value(left, field_name)
                if candidate is not None:
                    fallback = float(candidate)
                    break
        if fallback is None:
            raise ValueError(
                f"Field '{field_name}' has no defined value in CLASSIFICATION_RULES."
            )
        resolved.append(fallback)
    return resolved


def _build_classification_anchors() -> tuple[DepthClassificationAnchor, ...]:
    ordinary_offset_values = _resolve_rule_column("ordinary_offset_max_m")
    complex_offset_values = _resolve_rule_column("complex_offset_max_m")
    hold_ordinary_values = _resolve_rule_column("hold_ordinary_max_deg")
    hold_complex_values = _resolve_rule_column("hold_complex_max_deg")
    anchors: list[DepthClassificationAnchor] = []
    for idx, rule in enumerate(CLASSIFICATION_RULES):
        anchors.append(
            DepthClassificationAnchor(
                gv_m=float(rule.gv_m),
                reverse_min_m=float(rule.reverse_min_m),
                reverse_max_m=float(rule.reverse_max_m),
                ordinary_offset_max_m=float(ordinary_offset_values[idx]),
                complex_offset_max_m=float(complex_offset_values[idx]),
                hold_ordinary_max_deg=float(hold_ordinary_values[idx]),
                hold_complex_max_deg=float(hold_complex_values[idx]),
                reverse_allowed=bool(rule.reverse_allowed),
            )
        )
    return tuple(anchors)


CLASSIFICATION_ANCHORS: tuple[DepthClassificationAnchor, ...] = (
    _build_classification_anchors()
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
    def _display(value: float | None) -> str | float:
        if value is None:
            return "—"
        return float(value)

    rows: list[dict[str, str | float]] = []
    for rule in CLASSIFICATION_RULES:
        reverse_window = (
            "Не допускается"
            if not rule.reverse_allowed
            else f"{float(rule.reverse_min_m):.0f}-{float(rule.reverse_max_m):.0f}"
        )
        rows.append(
            {
                "ГВ, м": float(rule.gv_m),
                "Отход t1 для обратного направления, м": reverse_window,
                "Отход t1: Обычная (до), м": _display(rule.ordinary_offset_max_m),
                "Отход t1: Сложная (до), м": _display(rule.complex_offset_max_m),
                "ЗУ HOLD: Обычная (до), deg": _display(rule.hold_ordinary_max_deg),
                "ЗУ HOLD: Сложная (до), deg": _display(rule.hold_complex_max_deg),
            }
        )
    return rows
