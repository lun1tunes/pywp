from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionReportEvent,
    TARGET_T1,
    TARGET_T3,
    anti_collision_report_events,
)
from pywp.models import (
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    OPTIMIZATION_MINIMIZE_KOP,
    OPTIMIZATION_NONE,
)

RECOMMENDATION_TARGET_SPACING = "target_spacing"
RECOMMENDATION_REDUCE_KOP = "reduce_kop"
RECOMMENDATION_TRAJECTORY_REVIEW = "trajectory_review"


@dataclass(frozen=True)
class AntiCollisionWellContext:
    well_name: str
    kop_md_m: float | None
    kop_min_vertical_m: float | None
    md_total_m: float | None = None
    optimization_mode: str = OPTIMIZATION_NONE


@dataclass(frozen=True)
class AntiCollisionWellOverrideSuggestion:
    well_name: str
    config_updates: dict[str, object]
    reason: str


@dataclass(frozen=True)
class AntiCollisionRecommendation:
    recommendation_id: str
    well_a: str
    well_b: str
    priority_rank: int
    category: str
    summary: str
    detail: str
    action_label: str
    can_prepare_rerun: bool
    affected_wells: tuple[str, ...]
    override_suggestions: tuple[AntiCollisionWellOverrideSuggestion, ...]
    classification: str
    area_label: str
    md_a_start_m: float
    md_a_end_m: float
    md_b_start_m: float
    md_b_end_m: float
    min_separation_factor: float
    max_overlap_depth_m: float
    required_spacing_t1_m: float | None = None
    required_spacing_t3_m: float | None = None


def build_anti_collision_recommendations(
    analysis: AntiCollisionAnalysis,
    *,
    well_context_by_name: Mapping[str, AntiCollisionWellContext] | None = None,
) -> tuple[AntiCollisionRecommendation, ...]:
    contexts = {
        str(name): value for name, value in (well_context_by_name or {}).items()
    }
    recommendations: list[AntiCollisionRecommendation] = []
    for index, event in enumerate(anti_collision_report_events(analysis), start=1):
        recommendations.append(
            _build_single_recommendation(
                analysis=analysis,
                event=event,
                well_context_by_name=contexts,
                recommendation_id=f"ac-rec-{index:03d}",
            )
        )
    return tuple(recommendations)


def anti_collision_recommendation_rows(
    recommendations: tuple[AntiCollisionRecommendation, ...]
    | list[AntiCollisionRecommendation],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in recommendations:
        rows.append(
            {
                "Приоритет": _priority_label(item.priority_rank),
                "Скважина A": item.well_a,
                "Скважина B": item.well_b,
                "Тип действия": _recommendation_category_label(item.category),
                "Область": item.area_label,
                "Интервал A, м": _md_interval_label(item.md_a_start_m, item.md_a_end_m),
                "Интервал B, м": _md_interval_label(item.md_b_start_m, item.md_b_end_m),
                "SF min": float(item.min_separation_factor),
                "Overlap max, м": float(item.max_overlap_depth_m),
                "Spacing t1, м": _nullable_float(item.required_spacing_t1_m),
                "Spacing t3, м": _nullable_float(item.required_spacing_t3_m),
                "Рекомендация": item.summary,
                "Подготовка пересчета": item.action_label,
            }
        )
    return rows


def recommendation_display_label(recommendation: AntiCollisionRecommendation) -> str:
    return (
        f"{recommendation.well_a} ↔ {recommendation.well_b} · "
        f"{_recommendation_category_label(recommendation.category)} · "
        f"SF {float(recommendation.min_separation_factor):.2f}"
    )


def _build_single_recommendation(
    *,
    analysis: AntiCollisionAnalysis,
    event: AntiCollisionReportEvent,
    well_context_by_name: Mapping[str, AntiCollisionWellContext],
    recommendation_id: str,
) -> AntiCollisionRecommendation:
    if str(event.classification) != "trajectory":
        spacing_t1_m = (
            float(event.max_overlap_depth_m)
            if TARGET_T1 in {str(event.label_a), str(event.label_b)}
            else None
        )
        spacing_t3_m = (
            float(event.max_overlap_depth_m)
            if TARGET_T3 in {str(event.label_a), str(event.label_b)}
            else None
        )
        target_scope = _target_scope_label(str(event.label_a), str(event.label_b))
        summary = (
            f"Пересечение в области целей {target_scope}: для устранения текущего "
            f"2σ overlap увеличьте spacing минимум на {float(event.max_overlap_depth_m):.1f} м."
        )
        detail = (
            "Этот конфликт затрагивает t1/t3, поэтому сначала нужно корректировать "
            "взаимное положение целей, а не автоматически перестраивать траектории."
        )
        return AntiCollisionRecommendation(
            recommendation_id=recommendation_id,
            well_a=str(event.well_a),
            well_b=str(event.well_b),
            priority_rank=int(event.priority_rank),
            category=RECOMMENDATION_TARGET_SPACING,
            summary=summary,
            detail=detail,
            action_label="Только рекомендация по целям",
            can_prepare_rerun=False,
            affected_wells=(),
            override_suggestions=(),
            classification=str(event.classification),
            area_label=_event_area_label(event),
            md_a_start_m=float(event.md_a_start_m),
            md_a_end_m=float(event.md_a_end_m),
            md_b_start_m=float(event.md_b_start_m),
            md_b_end_m=float(event.md_b_end_m),
            min_separation_factor=float(event.min_separation_factor),
            max_overlap_depth_m=float(event.max_overlap_depth_m),
            required_spacing_t1_m=spacing_t1_m,
            required_spacing_t3_m=spacing_t3_m,
        )

    if _event_is_vertical_trajectory(event=event, analysis=analysis):
        suggestions: list[AntiCollisionWellOverrideSuggestion] = []
        detail_parts: list[str] = []
        for well_name in (str(event.well_a), str(event.well_b)):
            context = well_context_by_name.get(well_name)
            if context is None:
                continue
            current_kop = context.kop_md_m
            kop_floor = context.kop_min_vertical_m
            if current_kop is None or kop_floor is None:
                continue
            if float(current_kop) <= float(kop_floor) + 1e-3:
                detail_parts.append(
                    f"{well_name}: KOP уже у нижней границы ({float(kop_floor):.1f} м)."
                )
                continue
            suggestions.append(
                AntiCollisionWellOverrideSuggestion(
                    well_name=well_name,
                    config_updates={"optimization_mode": OPTIMIZATION_MINIMIZE_KOP},
                    reason=(
                        f"Vertical collision: опустить KOP с {float(current_kop):.1f} м "
                        f"к нижней границе {float(kop_floor):.1f} м."
                    ),
                )
            )
            detail_parts.append(
                f"{well_name}: текущий KOP {float(current_kop):.1f} м, "
                f"нижняя граница {float(kop_floor):.1f} м."
            )
        if suggestions:
            summary = (
                "Конфликт на vertical участке: сначала стоит проверить более ранний KOP "
                "для затронутых скважин."
            )
            detail = " ".join(detail_parts)
            action_label = "Подготовить пересчет с minimize_kop"
            can_prepare = True
            affected_wells = tuple(suggestion.well_name for suggestion in suggestions)
        else:
            summary = (
                "Конфликт на vertical участке, но KOP уже находится у нижней "
                "допустимой границы."
            )
            detail = (
                "В текущих ограничениях этот конфликт не устраняется только KOP-маневром. "
                "Нужно проверять spacing устьев или spacing целей."
            )
            action_label = "Только рекомендация"
            can_prepare = False
            affected_wells = ()
        return AntiCollisionRecommendation(
            recommendation_id=recommendation_id,
            well_a=str(event.well_a),
            well_b=str(event.well_b),
            priority_rank=int(event.priority_rank),
            category=RECOMMENDATION_REDUCE_KOP,
            summary=summary,
            detail=detail,
            action_label=action_label,
            can_prepare_rerun=can_prepare,
            affected_wells=affected_wells,
            override_suggestions=tuple(suggestions),
            classification=str(event.classification),
            area_label=_event_area_label(event),
            md_a_start_m=float(event.md_a_start_m),
            md_a_end_m=float(event.md_a_end_m),
            md_b_start_m=float(event.md_b_start_m),
            md_b_end_m=float(event.md_b_end_m),
            min_separation_factor=float(event.min_separation_factor),
            max_overlap_depth_m=float(event.max_overlap_depth_m),
        )

    movable_well = _select_movable_well_for_trajectory_event(
        event=event,
        well_context_by_name=well_context_by_name,
    )
    if movable_well is not None:
        reference_well = (
            str(event.well_b)
            if str(movable_well) == str(event.well_a)
            else str(event.well_a)
        )
        return AntiCollisionRecommendation(
            recommendation_id=recommendation_id,
            well_a=str(event.well_a),
            well_b=str(event.well_b),
            priority_rank=int(event.priority_rank),
            category=RECOMMENDATION_TRAJECTORY_REVIEW,
            summary=(
                f"Конфликт на криволинейном участке: подготовить pairwise anti-collision "
                f"пересчет для {movable_well} относительно {reference_well}."
            ),
            detail=(
                "Для первого этапа используется controlled rerun одной скважины "
                "с objective на улучшение separation factor на конфликтном окне."
            ),
            action_label="Подготовить anti-collision пересчет",
            can_prepare_rerun=True,
            affected_wells=(str(movable_well),),
            override_suggestions=(
                AntiCollisionWellOverrideSuggestion(
                    well_name=str(movable_well),
                    config_updates={
                        "optimization_mode": OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
                    },
                    reason=(
                        f"Trajectory collision against {reference_well}: подготовить "
                        "anti-collision avoidance rerun на конфликтном интервале."
                    ),
                ),
            ),
            classification=str(event.classification),
            area_label=_event_area_label(event),
            md_a_start_m=float(event.md_a_start_m),
            md_a_end_m=float(event.md_a_end_m),
            md_b_start_m=float(event.md_b_start_m),
            md_b_end_m=float(event.md_b_end_m),
            min_separation_factor=float(event.min_separation_factor),
            max_overlap_depth_m=float(event.max_overlap_depth_m),
        )

    return AntiCollisionRecommendation(
        recommendation_id=recommendation_id,
        well_a=str(event.well_a),
        well_b=str(event.well_b),
        priority_rank=int(event.priority_rank),
        category=RECOMMENDATION_TRAJECTORY_REVIEW,
        summary=(
            "Конфликт на build/hold/turn участке: нужен специальный anti-collision "
            "пересчет траектории с учетом SF-ограничений."
        ),
        detail=(
            "Автоматическая подготовка пересчета для этого случая пока не включена."
        ),
        action_label="Ожидает anti-collision optimization",
        can_prepare_rerun=False,
        affected_wells=(),
        override_suggestions=(),
        classification=str(event.classification),
        area_label=_event_area_label(event),
        md_a_start_m=float(event.md_a_start_m),
        md_a_end_m=float(event.md_a_end_m),
        md_b_start_m=float(event.md_b_start_m),
        md_b_end_m=float(event.md_b_end_m),
        min_separation_factor=float(event.min_separation_factor),
        max_overlap_depth_m=float(event.max_overlap_depth_m),
    )


def _event_is_vertical_trajectory(
    *,
    event: AntiCollisionReportEvent,
    analysis: AntiCollisionAnalysis,
) -> bool:
    well_by_name = {str(well.name): well for well in analysis.wells}
    well_a = well_by_name.get(str(event.well_a))
    well_b = well_by_name.get(str(event.well_b))
    if well_a is None or well_b is None:
        return False
    tolerance_m = float(max((well.overlay.model.sample_step_m for well in analysis.wells), default=100.0) * 1.05)
    vertical_limit_a = _well_vertical_end_md(well_a.stations)
    vertical_limit_b = _well_vertical_end_md(well_b.stations)
    return bool(
        float(event.md_a_end_m) <= vertical_limit_a + tolerance_m
        and float(event.md_b_end_m) <= vertical_limit_b + tolerance_m
    )


def _well_vertical_end_md(stations: pd.DataFrame) -> float:
    if len(stations) == 0 or "MD_m" not in stations.columns:
        return 0.0
    if "segment" not in stations.columns:
        return float(stations["MD_m"].iloc[0])
    vertical_rows = stations.loc[
        stations["segment"].astype(str).str.upper() == "VERTICAL",
        "MD_m",
    ]
    if len(vertical_rows) == 0:
        return float(stations["MD_m"].iloc[0])
    return float(vertical_rows.max())


def _priority_label(priority_rank: int) -> str:
    if int(priority_rank) == 0:
        return "Цели ↔ цели"
    if int(priority_rank) == 1:
        return "Цель ↔ траектория"
    return "Траектория ↔ траектория"


def _recommendation_category_label(category: str) -> str:
    if str(category) == RECOMMENDATION_TARGET_SPACING:
        return "Цели / spacing"
    if str(category) == RECOMMENDATION_REDUCE_KOP:
        return "VERTICAL / KOP"
    return "Траектория / review"


def _target_scope_label(label_a: str, label_b: str) -> str:
    labels = {str(label_a), str(label_b)}
    labels.discard("")
    if labels == {TARGET_T1}:
        return "t1"
    if labels == {TARGET_T3}:
        return "t3"
    if labels == {TARGET_T1, TARGET_T3}:
        return "t1/t3"
    return "целей"


def _event_area_label(event: AntiCollisionReportEvent) -> str:
    if str(event.label_a) and str(event.label_b):
        return f"{event.label_a} ↔ {event.label_b}"
    if str(event.label_a):
        return f"{event.label_a} ↔ траектория"
    if str(event.label_b):
        return f"траектория ↔ {event.label_b}"
    return "траектория ↔ траектория"


def _select_movable_well_for_trajectory_event(
    *,
    event: AntiCollisionReportEvent,
    well_context_by_name: Mapping[str, AntiCollisionWellContext],
) -> str | None:
    left = well_context_by_name.get(str(event.well_a))
    right = well_context_by_name.get(str(event.well_b))
    candidates = [item for item in (left, right) if item is not None]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.kop_md_m if item.kop_md_m is not None else 0.0),
            -float(item.md_total_m if item.md_total_m is not None else 0.0),
            str(item.well_name),
        ),
    )
    return str(ranked[0].well_name)


def _md_interval_label(md_start_m: float, md_end_m: float) -> str:
    start = float(md_start_m)
    end = float(md_end_m)
    if abs(end - start) <= 0.5:
        return f"{start:.0f}"
    return f"{start:.0f} - {end:.0f}"


def _nullable_float(value: float | None) -> float | str:
    if value is None:
        return "—"
    return float(value)
