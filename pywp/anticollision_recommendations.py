from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionReportEvent,
    TARGET_T1,
    TARGET_T3,
    anti_collision_report_events,
)
from pywp.anticollision_stage import (
    ANTI_COLLISION_STAGE_EARLY_KOP_BUILD1,
    ANTI_COLLISION_STAGE_LATE_TRAJECTORY,
)
from pywp.models import (
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    OPTIMIZATION_MINIMIZE_KOP,
    OPTIMIZATION_NONE,
)

RECOMMENDATION_TARGET_SPACING = "target_spacing"
RECOMMENDATION_REDUCE_KOP = "reduce_kop"
RECOMMENDATION_TRAJECTORY_REVIEW = "trajectory_review"

MANEUVER_TARGET_SPACING = "Сместить цели / увеличить spacing"
MANEUVER_EARLIER_KOP = "Раньше начать набор / уменьшить KOP"
MANEUVER_EARLY_KOP_BUILD1 = "Сместить ранний уход: KOP / BUILD1"
MANEUVER_PREENTRY_TURN = "Pre-entry azimuth turn / сдвиг HOLD до t1"
MANEUVER_BUILD2_ENTRY = "Скорректировать BUILD2 перед t1"
MANEUVER_ENTRY_WINDOW_TURN = "Сдвиг входа и turn в окне t1"
MANEUVER_POSTENTRY_TURN = "Сместить post-entry / HORIZONTAL"
MANEUVER_KOP_AND_TRAJECTORY = "Сначала уменьшить KOP, затем локально отвести ствол"
MANEUVER_CLUSTER_MIXED = "Комбинированный cluster-level маневр"

_PRE_KOP_SEGMENTS = {"VERTICAL", "BUILD1"}
_EVENT_SEGMENT_TOLERANCE_M = 25.0
_KOP_SATURATION_TOLERANCE_M = 1.0
_BUILD_DLS_SATURATION_TOLERANCE_DEG_PER_30M = 1e-3


@dataclass(frozen=True)
class AntiCollisionWellContext:
    well_name: str
    kop_md_m: float | None
    kop_min_vertical_m: float | None
    build1_dls_deg_per_30m: float | None = None
    build_dls_max_deg_per_30m: float | None = None
    md_t1_m: float | None = None
    md_total_m: float | None = None
    optimization_mode: str = OPTIMIZATION_NONE
    anti_collision_stage: str | None = None
    anti_collision_attempted_stages: tuple[str, ...] = ()


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
    expected_maneuver: str
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


@dataclass(frozen=True)
class AntiCollisionRecommendationCluster:
    cluster_id: str
    well_names: tuple[str, ...]
    recommendations: tuple[AntiCollisionRecommendation, ...]
    recommendation_count: int
    target_conflict_count: int
    vertical_conflict_count: int
    trajectory_conflict_count: int
    worst_separation_factor: float
    summary: str
    detail: str
    expected_maneuver: str
    blocking_advisory: str | None
    rerun_order_label: str
    first_rerun_well: str | None
    first_rerun_maneuver: str | None
    action_steps: tuple["AntiCollisionClusterActionStep", ...]
    can_prepare_rerun: bool
    affected_wells: tuple[str, ...]
    action_label: str


@dataclass(frozen=True)
class AntiCollisionClusterActionStep:
    order_rank: int
    well_name: str
    category: str
    optimization_mode: str
    expected_maneuver: str
    reason: str
    related_recommendation_count: int
    worst_separation_factor: float


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
                "Ожидаемый маневр": item.expected_maneuver,
                "Рекомендация": item.summary,
                "Подготовка пересчета": item.action_label,
            }
        )
    return rows


def build_anti_collision_recommendation_clusters(
    recommendations: tuple[AntiCollisionRecommendation, ...]
    | list[AntiCollisionRecommendation],
) -> tuple[AntiCollisionRecommendationCluster, ...]:
    items = tuple(recommendations)
    if not items:
        return ()
    by_well: dict[str, set[str]] = {}
    for item in items:
        by_well.setdefault(str(item.well_a), set()).add(str(item.well_b))
        by_well.setdefault(str(item.well_b), set()).add(str(item.well_a))

    visited: set[str] = set()
    components: list[tuple[str, ...]] = []
    for well_name in sorted(by_well):
        if well_name in visited:
            continue
        stack = [well_name]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(sorted(by_well.get(current, ())))
        components.append(tuple(sorted(component)))

    clusters: list[AntiCollisionRecommendationCluster] = []
    for index, component in enumerate(components, start=1):
        component_set = set(component)
        member_recommendations = tuple(
            item
            for item in items
            if str(item.well_a) in component_set and str(item.well_b) in component_set
        )
        if not member_recommendations:
            continue
        target_count = sum(
            1 for item in member_recommendations if item.category == RECOMMENDATION_TARGET_SPACING
        )
        vertical_count = sum(
            1 for item in member_recommendations if item.category == RECOMMENDATION_REDUCE_KOP
        )
        trajectory_count = sum(
            1 for item in member_recommendations if item.category == RECOMMENDATION_TRAJECTORY_REVIEW
        )
        affected_wells = tuple(
            sorted(
                {
                    str(well_name)
                    for item in member_recommendations
                    for well_name in item.affected_wells
                }
            )
        )
        has_unresolved_target_conflicts = bool(target_count > 0)
        can_prepare = bool(
            not has_unresolved_target_conflicts
            and any(bool(item.can_prepare_rerun) for item in member_recommendations)
        )
        worst_sf = min(float(item.min_separation_factor) for item in member_recommendations)
        action_steps = _cluster_action_steps(member_recommendations)
        blocking_advisory = (
            "Сначала решить spacing целей."
            if has_unresolved_target_conflicts
            else None
        )
        rerun_order_label = (
            " → ".join(step.well_name for step in action_steps)
            if action_steps
            else "—"
        )
        first_rerun_well = action_steps[0].well_name if action_steps else None
        first_rerun_maneuver = action_steps[0].expected_maneuver if action_steps else None
        expected_maneuver = _cluster_expected_maneuver(
            target_count=target_count,
            vertical_count=vertical_count,
            trajectory_count=trajectory_count,
        )
        summary = (
            f"Кластер {', '.join(component)}: событий {len(member_recommendations)}, "
            f"worst SF {worst_sf:.2f}."
        )
        if affected_wells:
            detail = (
                "К пересчету в текущем плане: " + ", ".join(affected_wells) + "."
            )
        else:
            detail = "Для этого кластера пока доступны только advisory-рекомендации."
        if blocking_advisory is not None:
            detail += " " + blocking_advisory
        if first_rerun_well is not None and first_rerun_maneuver is not None:
            detail += (
                f" Первый рекомендуемый маневр: {first_rerun_well} "
                f"({first_rerun_maneuver})."
            )
        clusters.append(
            AntiCollisionRecommendationCluster(
                cluster_id=f"ac-cluster-{index:03d}",
                well_names=component,
                recommendations=member_recommendations,
                recommendation_count=len(member_recommendations),
                target_conflict_count=target_count,
                vertical_conflict_count=vertical_count,
                trajectory_conflict_count=trajectory_count,
                worst_separation_factor=float(worst_sf),
                summary=summary,
                detail=detail,
                expected_maneuver=expected_maneuver,
                blocking_advisory=blocking_advisory,
                rerun_order_label=rerun_order_label,
                first_rerun_well=first_rerun_well,
                first_rerun_maneuver=first_rerun_maneuver,
                action_steps=action_steps,
                can_prepare_rerun=can_prepare,
                affected_wells=affected_wells,
                action_label=(
                    "Подготовить cluster-level пересчет"
                    if can_prepare
                    else "Только advisory"
                ),
            )
        )
    return tuple(
        sorted(
            clusters,
            key=lambda item: (
                float(item.worst_separation_factor),
                -int(item.recommendation_count),
                item.well_names,
            ),
        )
    )


def anti_collision_cluster_rows(
    clusters: tuple[AntiCollisionRecommendationCluster, ...]
    | list[AntiCollisionRecommendationCluster],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in clusters:
        rows.append(
            {
                "Кластер": item.cluster_id,
                "Скважины": ", ".join(item.well_names),
                "Событий": int(item.recommendation_count),
                "Цели": int(item.target_conflict_count),
                "VERTICAL": int(item.vertical_conflict_count),
                "Траектория": int(item.trajectory_conflict_count),
                "SF min": float(item.worst_separation_factor),
                "Ожидаемый маневр": item.expected_maneuver,
                "Стартовый шаг": (
                    f"{item.first_rerun_well}: {item.first_rerun_maneuver}"
                    if item.first_rerun_well and item.first_rerun_maneuver
                    else (item.blocking_advisory or "—")
                ),
                "Порядок": item.rerun_order_label,
                "К пересчету": ", ".join(item.affected_wells) if item.affected_wells else "—",
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


def cluster_display_label(cluster: AntiCollisionRecommendationCluster) -> str:
    wells_label = ", ".join(cluster.well_names)
    first_step = (
        f" · старт: {cluster.first_rerun_well}"
        if cluster.first_rerun_well is not None
        else ""
    )
    return (
        f"{cluster.cluster_id} · {wells_label} · "
        f"событий {int(cluster.recommendation_count)} · "
        f"SF {float(cluster.worst_separation_factor):.2f}{first_step}"
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
            expected_maneuver=MANEUVER_TARGET_SPACING,
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

    if _event_prefers_kop_adjustment(
        event=event,
        analysis=analysis,
        well_context_by_name=well_context_by_name,
    ):
        suggestions: list[AntiCollisionWellOverrideSuggestion] = []
        detail_parts: list[str] = []
        for well_name in (str(event.well_a), str(event.well_b)):
            context = well_context_by_name.get(well_name)
            if context is None:
                continue
            current_kop = context.kop_md_m
            kop_floor = context.kop_min_vertical_m
            current_build1 = context.build1_dls_deg_per_30m
            build_limit = context.build_dls_max_deg_per_30m
            can_adjust_early = _well_can_prepare_early_kop_build1(context)
            if current_kop is None or kop_floor is None:
                detail_parts.append(
                    f"{well_name}: нет полного KOP-контекста, но ранний cone-aware "
                    "пересчет всё равно подготовлен."
                )
            else:
                if not can_adjust_early:
                    detail_parts.append(
                        f"{well_name}: ранний уход уже исчерпан "
                        f"(KOP {float(current_kop):.1f} м, BUILD1 "
                        f"{float(current_build1 or 0.0):.2f} deg/30m)."
                    )
                elif float(current_kop) <= float(kop_floor) + 1e-3:
                    detail_parts.append(
                        f"{well_name}: KOP уже у нижней границы ({float(kop_floor):.1f} м), "
                        "поэтому ранний маневр пойдет через BUILD1."
                    )
                else:
                    detail_parts.append(
                        f"{well_name}: текущий KOP {float(current_kop):.1f} м, "
                        f"нижняя граница {float(kop_floor):.1f} м."
                    )
                if (
                    can_adjust_early
                    and current_build1 is not None
                    and build_limit is not None
                    and float(current_build1)
                    < float(build_limit) - _BUILD_DLS_SATURATION_TOLERANCE_DEG_PER_30M
                ):
                    detail_parts.append(
                        f"{well_name}: BUILD1 {float(current_build1):.2f} из "
                        f"{float(build_limit):.2f} deg/30m."
                    )
            if not can_adjust_early:
                continue
            suggestions.append(
                AntiCollisionWellOverrideSuggestion(
                    well_name=well_name,
                    config_updates={
                        "optimization_mode": OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
                    },
                    reason=(
                        "Early collision: cone-aware rerun по KOP/BUILD1 "
                        f"для {well_name}."
                    ),
                )
            )
        if suggestions:
            summary = (
                "Конфликт на раннем участке: нужен cone-aware пересчет по KOP/BUILD1 "
                "для затронутых скважин."
            )
            detail = " ".join(detail_parts)
            action_label = "Подготовить anti-collision пересчет (KOP/BUILD1)"
            can_prepare = True
            affected_wells = tuple(suggestion.well_name for suggestion in suggestions)
        else:
            summary = (
                "Конфликт на раннем участке, но ранний маневр KOP/BUILD1 "
                "для этой пары уже исчерпан."
            )
            detail = (
                "Нужно переходить к следующему маневру по траектории или "
                "проверять spacing устьев/целей."
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
            expected_maneuver=MANEUVER_EARLY_KOP_BUILD1,
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
    well_a_context = well_context_by_name.get(str(event.well_a))
    well_b_context = well_context_by_name.get(str(event.well_b))
    if (
        well_a_context is not None
        and well_b_context is not None
        and _well_has_late_trajectory_attempt(well_a_context)
        and _well_has_late_trajectory_attempt(well_b_context)
    ):
        preferred_moving_well = _select_movable_well_for_trajectory_event(
            event=event,
            well_context_by_name=well_context_by_name,
        ) or str(event.well_a)
        fixed_well = (
            str(event.well_b)
            if str(preferred_moving_well) == str(event.well_a)
            else str(event.well_a)
        )
        expected_maneuver = _expected_trajectory_maneuver(
            event=event,
            moving_well=str(preferred_moving_well),
            well_context_by_name=well_context_by_name,
            analysis=analysis,
        )
        return AntiCollisionRecommendation(
            recommendation_id=recommendation_id,
            well_a=str(event.well_a),
            well_b=str(event.well_b),
            priority_rank=int(event.priority_rank),
            category=RECOMMENDATION_TRAJECTORY_REVIEW,
            summary=(
                "Остаточный конфликт на криволинейном участке: automatic "
                "late BUILD2/HOLD anti-collision rerun для этой пары уже исчерпан. "
                f"Нужен дополнительный local spacing не меньше "
                f"{float(event.max_overlap_depth_m):.1f} м."
            ),
            detail=(
                "Обе проектные скважины пары уже пересчитаны в anti-collision mode, "
                "но overlap сохраняется. Это признак того, что в текущих пределах "
                "t1/t3, min KOP и DLS remaining late conflict требует ручной "
                "корректировки targets/spacing или ослабления ограничений. "
                f"Практически следующий ручной ход: разводить {preferred_moving_well} "
                f"от {fixed_well} в lateral-плоскости на участке "
                f"'{expected_maneuver}' минимум на {float(event.max_overlap_depth_m):.1f} м."
            ),
            expected_maneuver=str(expected_maneuver),
            action_label="Только рекомендация",
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
    if movable_well is not None:
        secondary_well = (
            str(event.well_b)
            if str(movable_well) == str(event.well_a)
            else str(event.well_a)
        )
        override_suggestions = (
            AntiCollisionWellOverrideSuggestion(
                well_name=str(movable_well),
                config_updates={
                    "optimization_mode": OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
                },
                reason=(
                    f"Trajectory collision: подготовить anti-collision avoidance rerun "
                    f"для {movable_well} на конфликтном интервале."
                ),
            ),
            AntiCollisionWellOverrideSuggestion(
                well_name=str(secondary_well),
                config_updates={
                    "optimization_mode": OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
                },
                reason=(
                    f"Trajectory collision: подготовить anti-collision avoidance rerun "
                    f"для {secondary_well} как второй стороны пары."
                ),
            ),
        )
        return AntiCollisionRecommendation(
            recommendation_id=recommendation_id,
            well_a=str(event.well_a),
            well_b=str(event.well_b),
            priority_rank=int(event.priority_rank),
            category=RECOMMENDATION_TRAJECTORY_REVIEW,
            summary=(
                "Конфликт на криволинейном участке: подготовить anti-collision "
                f"пересчет пары {movable_well} ↔ {secondary_well} с учетом "
                "остальных конусов куста."
            ),
            detail=(
                "Для trajectory-collision пересчет готовится сразу для обеих "
                "проектных скважин пары, а optimization context учитывает "
                "все актуальные reference-cones остальных скважин."
            ),
            expected_maneuver=_expected_trajectory_maneuver(
                event=event,
                moving_well=str(movable_well),
                well_context_by_name=well_context_by_name,
                analysis=analysis,
            ),
            action_label="Подготовить anti-collision пересчет",
            can_prepare_rerun=True,
            affected_wells=(str(movable_well), str(secondary_well)),
            override_suggestions=override_suggestions,
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
        expected_maneuver=MANEUVER_CLUSTER_MIXED,
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


def _event_prefers_kop_adjustment(
    *,
    event: AntiCollisionReportEvent,
    analysis: AntiCollisionAnalysis,
    well_context_by_name: Mapping[str, AntiCollisionWellContext],
) -> bool:
    well_by_name = {str(well.name): well for well in analysis.wells}
    well_a = well_by_name.get(str(event.well_a))
    well_b = well_by_name.get(str(event.well_b))
    if well_a is None or well_b is None:
        return False
    tolerance_m = float(
        max((well.overlay.model.sample_step_m for well in analysis.wells), default=100.0)
        * 1.05
    )
    left_context = well_context_by_name.get(str(event.well_a))
    right_context = well_context_by_name.get(str(event.well_b))
    return bool(
        _well_interval_prefers_kop(
            stations=well_a.stations,
            md_start_m=float(event.md_a_start_m),
            md_end_m=float(event.md_a_end_m),
            kop_md_m=None if left_context is None else left_context.kop_md_m,
            tolerance_m=tolerance_m,
        )
        and _well_interval_prefers_kop(
            stations=well_b.stations,
            md_start_m=float(event.md_b_start_m),
            md_end_m=float(event.md_b_end_m),
            kop_md_m=None if right_context is None else right_context.kop_md_m,
            tolerance_m=tolerance_m,
        )
    )


def _well_interval_prefers_kop(
    *,
    stations: pd.DataFrame,
    md_start_m: float,
    md_end_m: float,
    kop_md_m: float | None,
    tolerance_m: float,
) -> bool:
    dominant_segment = _well_dominant_segment_for_interval(
        stations=stations,
        md_start_m=float(md_start_m),
        md_end_m=float(md_end_m),
    )
    if dominant_segment in _PRE_KOP_SEGMENTS:
        return True
    if kop_md_m is None:
        return False
    midpoint_md = 0.5 * (float(md_start_m) + float(md_end_m))
    return bool(midpoint_md <= float(kop_md_m) + float(tolerance_m))


def _well_can_prepare_early_kop_build1(context: AntiCollisionWellContext) -> bool:
    if _well_has_attempted_stage(context, ANTI_COLLISION_STAGE_EARLY_KOP_BUILD1):
        return False
    current_kop = context.kop_md_m
    kop_floor = context.kop_min_vertical_m
    current_build1 = context.build1_dls_deg_per_30m
    build_limit = context.build_dls_max_deg_per_30m
    if current_kop is None or kop_floor is None:
        return True
    if float(current_kop) > float(kop_floor) + _KOP_SATURATION_TOLERANCE_M:
        return True
    if current_build1 is None or build_limit is None:
        return True
    return bool(
        float(current_build1)
        < float(build_limit) - _BUILD_DLS_SATURATION_TOLERANCE_DEG_PER_30M
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


def _well_dominant_segment_for_interval(
    *,
    stations: pd.DataFrame,
    md_start_m: float,
    md_end_m: float,
) -> str:
    if len(stations) == 0 or "MD_m" not in stations.columns or "segment" not in stations.columns:
        return ""
    md_values = stations["MD_m"].to_numpy(dtype=float)
    segment_values = stations["segment"].astype(str).to_numpy()
    start_md = float(min(md_start_m, md_end_m))
    end_md = float(max(md_start_m, md_end_m))
    window_mask = (
        (md_values >= start_md - _EVENT_SEGMENT_TOLERANCE_M)
        & (md_values <= end_md + _EVENT_SEGMENT_TOLERANCE_M)
    )
    if not np.any(window_mask):
        center_md = 0.5 * (start_md + end_md)
        nearest_index = int(np.argmin(np.abs(md_values - center_md)))
        return str(segment_values[nearest_index]).strip().upper()
    window_segments = [str(value).strip().upper() for value in segment_values[window_mask]]
    if not window_segments:
        return ""
    counts: dict[str, int] = {}
    order: list[str] = []
    for segment in window_segments:
        if segment not in counts:
            counts[segment] = 0
            order.append(segment)
        counts[segment] += 1
    return max(order, key=lambda segment: (int(counts[segment]), -order.index(segment)))


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
    not_yet_late = [
        item for item in candidates if not _well_has_late_trajectory_attempt(item)
    ]
    not_yet_optimized = [
        item
        for item in (not_yet_late or candidates)
        if str(item.optimization_mode).strip() != OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    ]
    ranked_pool = not_yet_optimized if not_yet_optimized else (not_yet_late or candidates)
    ranked = sorted(
        ranked_pool,
        key=lambda item: (
            -float(item.kop_md_m if item.kop_md_m is not None else 0.0),
            -float(item.md_total_m if item.md_total_m is not None else 0.0),
            str(item.well_name),
        ),
    )
    return str(ranked[0].well_name)


def _well_has_late_trajectory_attempt(context: AntiCollisionWellContext) -> bool:
    if _well_has_attempted_stage(context, ANTI_COLLISION_STAGE_LATE_TRAJECTORY):
        return True
    return str(context.optimization_mode).strip() == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE


def _well_has_attempted_stage(
    context: AntiCollisionWellContext,
    stage_name: str,
) -> bool:
    target_stage = str(stage_name).strip()
    if not target_stage:
        return False
    current_stage = str(context.anti_collision_stage or "").strip()
    if current_stage == target_stage:
        return True
    return target_stage in {
        str(stage).strip()
        for stage in tuple(context.anti_collision_attempted_stages)
        if str(stage).strip()
    }


def _expected_trajectory_maneuver(
    *,
    event: AntiCollisionReportEvent,
    moving_well: str,
    well_context_by_name: Mapping[str, AntiCollisionWellContext],
    analysis: AntiCollisionAnalysis,
) -> str:
    moving_context = well_context_by_name.get(str(moving_well))
    moving_is_a = str(moving_well) == str(event.well_a)
    md_start = float(event.md_a_start_m if moving_is_a else event.md_b_start_m)
    md_end = float(event.md_a_end_m if moving_is_a else event.md_b_end_m)
    dominant_segment = _event_interval_segment_for_well(
        analysis=analysis,
        well_name=str(moving_well),
        md_start_m=md_start,
        md_end_m=md_end,
    )
    if dominant_segment == "BUILD2":
        return MANEUVER_BUILD2_ENTRY
    if moving_context is None or moving_context.md_t1_m is None:
        return MANEUVER_CLUSTER_MIXED
    t1_md = float(moving_context.md_t1_m)
    tolerance_m = 75.0
    if md_end <= t1_md - tolerance_m:
        return MANEUVER_PREENTRY_TURN
    if md_start >= t1_md + tolerance_m:
        return MANEUVER_POSTENTRY_TURN
    return MANEUVER_ENTRY_WINDOW_TURN


def _event_interval_segment_for_well(
    *,
    analysis: AntiCollisionAnalysis,
    well_name: str,
    md_start_m: float,
    md_end_m: float,
) -> str:
    for well in analysis.wells:
        if str(well.name) != str(well_name):
            continue
        return _well_dominant_segment_for_interval(
            stations=well.stations,
            md_start_m=float(md_start_m),
            md_end_m=float(md_end_m),
        )
    return ""


def _cluster_expected_maneuver(
    *,
    target_count: int,
    vertical_count: int,
    trajectory_count: int,
) -> str:
    if target_count > 0 and vertical_count == 0 and trajectory_count == 0:
        return MANEUVER_TARGET_SPACING
    if target_count > 0 and (vertical_count > 0 or trajectory_count > 0):
        return "Сначала spacing целей, затем локальный пересчет"
    if vertical_count > 0 and trajectory_count == 0:
        return MANEUVER_EARLIER_KOP
    if trajectory_count > 0 and vertical_count == 0:
        return "Локальный anti-collision отвод стволов"
    return MANEUVER_CLUSTER_MIXED


def _cluster_action_steps(
    recommendations: tuple[AntiCollisionRecommendation, ...],
) -> tuple[AntiCollisionClusterActionStep, ...]:
    per_well: dict[str, list[AntiCollisionRecommendation]] = {}
    for recommendation in recommendations:
        for well_name in recommendation.affected_wells:
            per_well.setdefault(str(well_name), []).append(recommendation)
    ranked_items: list[tuple[str, tuple[AntiCollisionRecommendation, ...]]] = [
        (well_name, tuple(items)) for well_name, items in per_well.items() if items
    ]
    ranked_items.sort(
        key=lambda item: _cluster_action_sort_key(item[0], item[1]),
    )
    steps: list[AntiCollisionClusterActionStep] = []
    for order_rank, (well_name, items) in enumerate(ranked_items, start=1):
        actionable_items = tuple(
            item for item in items if bool(item.can_prepare_rerun)
        ) or items
        primary = min(
            actionable_items,
            key=lambda recommendation: float(recommendation.min_separation_factor),
        )
        has_trajectory = any(
            str(item.category) == RECOMMENDATION_TRAJECTORY_REVIEW
            for item in actionable_items
        )
        has_vertical = any(
            str(item.category) == RECOMMENDATION_REDUCE_KOP
            for item in actionable_items
        )
        steps.append(
            AntiCollisionClusterActionStep(
                order_rank=int(order_rank),
                well_name=str(well_name),
                category=(
                    "mixed"
                    if has_trajectory and has_vertical
                    else str(primary.category)
                ),
                optimization_mode=_override_mode_for_well(well_name, items),
                expected_maneuver=_step_expected_maneuver(actionable_items),
                reason=" | ".join(
                    dict.fromkeys(str(recommendation.summary) for recommendation in items)
                ),
                related_recommendation_count=len(items),
                worst_separation_factor=min(
                    float(recommendation.min_separation_factor) for recommendation in items
                ),
            )
        )
    return tuple(steps)


def _cluster_action_sort_key(
    well_name: str,
    recommendations: tuple[AntiCollisionRecommendation, ...],
) -> tuple[float, int, int, int, str]:
    actionable_items = tuple(
        item for item in recommendations if bool(item.can_prepare_rerun)
    ) or recommendations
    primary = min(
        actionable_items,
        key=lambda recommendation: float(recommendation.min_separation_factor),
    )
    category_rank = (
        0
        if str(primary.category) == RECOMMENDATION_REDUCE_KOP
        else (1 if str(primary.category) == RECOMMENDATION_TRAJECTORY_REVIEW else 2)
    )
    worst_sf = float(primary.min_separation_factor)
    trajectory_count = sum(
        1
        for item in actionable_items
        if str(item.category) == RECOMMENDATION_TRAJECTORY_REVIEW
    )
    vertical_count = sum(
        1 for item in actionable_items if str(item.category) == RECOMMENDATION_REDUCE_KOP
    )
    return (
        float(worst_sf),
        int(category_rank),
        -int(trajectory_count),
        -int(vertical_count),
        str(well_name),
    )


def _override_mode_for_well(
    well_name: str,
    recommendations: tuple[AntiCollisionRecommendation, ...],
) -> str:
    has_anti_collision = False
    has_minimize_kop = False
    for recommendation in recommendations:
        for suggestion in recommendation.override_suggestions:
            if str(suggestion.well_name) != str(well_name):
                continue
            optimization_mode = str(
                suggestion.config_updates.get("optimization_mode", "")
            ).strip()
            if optimization_mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE:
                has_anti_collision = True
            elif optimization_mode == OPTIMIZATION_MINIMIZE_KOP:
                has_minimize_kop = True
    if has_anti_collision:
        return OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    if has_minimize_kop:
        return OPTIMIZATION_MINIMIZE_KOP
    return OPTIMIZATION_NONE


def _step_expected_maneuver(
    recommendations: tuple[AntiCollisionRecommendation, ...],
) -> str:
    if not recommendations:
        return MANEUVER_CLUSTER_MIXED
    actionable_items = tuple(
        item for item in recommendations if bool(item.can_prepare_rerun)
    ) or recommendations
    has_trajectory = any(
        str(item.category) == RECOMMENDATION_TRAJECTORY_REVIEW
        for item in actionable_items
    )
    has_vertical = any(
        str(item.category) == RECOMMENDATION_REDUCE_KOP
        for item in actionable_items
    )
    if has_trajectory and has_vertical:
        primary = min(
            actionable_items,
            key=lambda recommendation: float(recommendation.min_separation_factor),
        )
        if str(primary.category) == RECOMMENDATION_REDUCE_KOP:
            return MANEUVER_KOP_AND_TRAJECTORY
    has_trajectory = any(
        str(item.category) == RECOMMENDATION_TRAJECTORY_REVIEW
        for item in actionable_items
    )
    if has_trajectory:
        primary_trajectory = min(
            (
                item
                for item in actionable_items
                if str(item.category) == RECOMMENDATION_TRAJECTORY_REVIEW
            ),
            key=lambda recommendation: float(recommendation.min_separation_factor),
        )
        return str(primary_trajectory.expected_maneuver)
    primary = min(
        actionable_items,
        key=lambda recommendation: float(recommendation.min_separation_factor),
    )
    return str(primary.expected_maneuver)


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
