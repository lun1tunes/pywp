from __future__ import annotations

import math

from pywp.anticollision import AntiCollisionAnalysis, _segment_types_for_interval
from pywp.anticollision_recommendations import (
    AntiCollisionRecommendation,
    AntiCollisionRecommendationCluster,
    cluster_display_label,
    recommendation_display_label,
)

__all__ = [
    "anticollision_focus_well_names",
    "cluster_snapshot",
    "clusters_touching_focus_pad",
    "format_overlap_value",
    "format_sf_value",
    "md_interval_label",
    "pad_scoped_cluster_focus_well_names",
    "pad_scoped_cluster_target_well_names",
    "recommendation_snapshot",
    "recommendations_for_clusters",
    "report_rows_from_recommendations",
    "sf_help_markdown",
]


def recommendation_snapshot(
    recommendation: AntiCollisionRecommendation,
) -> dict[str, object]:
    return {
        "kind": "recommendation",
        "recommendation_id": str(recommendation.recommendation_id),
        "well_a": str(recommendation.well_a),
        "well_b": str(recommendation.well_b),
        "classification": str(recommendation.classification),
        "category": str(recommendation.category),
        "area_label": str(recommendation.area_label),
        "summary": str(recommendation.summary),
        "detail": str(recommendation.detail),
        "expected_maneuver": str(recommendation.expected_maneuver),
        "before_sf": float(recommendation.min_separation_factor),
        "before_overlap_m": float(recommendation.max_overlap_depth_m),
        "md_a_start_m": float(recommendation.md_a_start_m),
        "md_a_end_m": float(recommendation.md_a_end_m),
        "md_b_start_m": float(recommendation.md_b_start_m),
        "md_b_end_m": float(recommendation.md_b_end_m),
        "affected_wells": tuple(str(name) for name in recommendation.affected_wells),
        "action_label": str(recommendation.action_label),
        "source_label": recommendation_display_label(recommendation),
    }


def cluster_snapshot(
    cluster: AntiCollisionRecommendationCluster,
    *,
    target_well_names: tuple[str, ...] = (),
    focus_well_names: tuple[str, ...] = (),
) -> dict[str, object]:
    items = tuple(recommendation_snapshot(item) for item in cluster.recommendations)
    actionable_before_sf = [
        float(item.min_separation_factor)
        for item in cluster.recommendations
        if bool(item.can_prepare_rerun)
    ]
    before_sf = (
        min(actionable_before_sf)
        if actionable_before_sf
        else float(cluster.worst_separation_factor)
    )
    return {
        "kind": "cluster",
        "cluster_id": str(cluster.cluster_id),
        "source_label": cluster_display_label(cluster),
        "summary": str(cluster.summary),
        "detail": str(cluster.detail),
        "expected_maneuver": str(cluster.expected_maneuver),
        "blocking_advisory": (
            None
            if cluster.blocking_advisory is None
            else str(cluster.blocking_advisory)
        ),
        "affected_wells": tuple(str(name) for name in cluster.affected_wells),
        "well_names": tuple(str(name) for name in cluster.well_names),
        "target_well_names": tuple(str(name) for name in target_well_names),
        "focus_well_names": tuple(str(name) for name in focus_well_names),
        "recommendation_count": int(cluster.recommendation_count),
        "before_sf": float(before_sf),
        "rerun_order_label": str(cluster.rerun_order_label),
        "first_rerun_well": (
            None if cluster.first_rerun_well is None else str(cluster.first_rerun_well)
        ),
        "first_rerun_maneuver": (
            None
            if cluster.first_rerun_maneuver is None
            else str(cluster.first_rerun_maneuver)
        ),
        "action_steps": tuple(
            {
                "order_rank": int(step.order_rank),
                "well_name": str(step.well_name),
                "category": str(step.category),
                "optimization_mode": str(step.optimization_mode),
                "expected_maneuver": str(step.expected_maneuver),
                "reason": str(step.reason),
                "related_recommendation_count": int(step.related_recommendation_count),
                "worst_separation_factor": float(step.worst_separation_factor),
            }
            for step in cluster.action_steps
        ),
        "items": items,
    }


def format_sf_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.2f}"


def format_overlap_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.2f}"


def md_interval_label(md_start_m: float, md_end_m: float) -> str:
    start = float(md_start_m)
    end = float(md_end_m)
    if abs(end - start) <= 1e-6:
        return f"{start:.0f}"
    return f"{start:.0f}-{end:.0f}"


def sf_help_markdown() -> str:
    return (
        "**Что такое SF**\n\n"
        "SF (`Separation Factor`) показывает запас расстояния между двумя "
        "скважинами с учетом суммарной неопределенности их конусов.\n\n"
        "- `SF < 1` — конусы неопределенности overlap, это collision-risk.\n"
        "- `SF ≈ 1` — граничное состояние, запас почти исчерпан.\n"
        "- `SF > 1` — есть запас по разнесению; чем больше число, тем комфортнее ситуация.\n\n"
        "В текущем WELLTRACK это planning-level индикатор для сравнения вариантов, "
        "а не абсолютная гарантия безопасности."
    )


def clusters_touching_focus_pad(
    *,
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
    focus_pad_well_names: tuple[str, ...],
) -> tuple[AntiCollisionRecommendationCluster, ...]:
    focus_set = {str(name) for name in focus_pad_well_names if str(name).strip()}
    if not focus_set:
        return tuple(clusters)
    return tuple(
        cluster
        for cluster in clusters
        if focus_set.intersection(str(name) for name in cluster.well_names)
    )


def recommendations_for_clusters(
    *,
    recommendations: tuple[AntiCollisionRecommendation, ...],
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
) -> tuple[AntiCollisionRecommendation, ...]:
    visible_ids = {
        str(item.recommendation_id)
        for cluster in clusters
        for item in cluster.recommendations
    }
    if not visible_ids:
        return ()
    return tuple(
        item for item in recommendations if str(item.recommendation_id) in visible_ids
    )


def report_rows_from_recommendations(
    recommendations: tuple[AntiCollisionRecommendation, ...],
    analysis: AntiCollisionAnalysis | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in recommendations:
        segment_a = "—"
        segment_b = "—"
        if analysis is not None:
            segment_a = _segment_types_for_interval(
                analysis,
                str(item.well_a),
                float(item.md_a_start_m),
                float(item.md_a_end_m),
            )
            segment_b = _segment_types_for_interval(
                analysis,
                str(item.well_b),
                float(item.md_b_start_m),
                float(item.md_b_end_m),
            )
        rows.append(
            {
                "Приоритет": str(item.priority_rank),
                "Скважина A": str(item.well_a),
                "Скважина B": str(item.well_b),
                "Участок A": segment_a,
                "Участок B": segment_b,
                "Интервал A, м": md_interval_label(
                    float(item.md_a_start_m),
                    float(item.md_a_end_m),
                ),
                "Интервал B, м": md_interval_label(
                    float(item.md_b_start_m),
                    float(item.md_b_end_m),
                ),
                "SF min": float(item.min_separation_factor),
                "Overlap max, м": float(item.max_overlap_depth_m),
                "Мин. расстояние, м": float(item.min_center_distance_m),
                "Рекомендация по устранению": str(item.summary),
            }
        )
    return rows


def pad_scoped_cluster_target_well_names(
    *,
    cluster: AntiCollisionRecommendationCluster,
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    focus_set = {str(name) for name in focus_pad_well_names if str(name).strip()}
    focused_cluster = tuple(
        str(name) for name in cluster.well_names if str(name) in focus_set
    )
    cluster_scope = (
        tuple(str(name) for name in cluster.well_names)
        if cluster.well_names
        else tuple(str(name) for name in cluster.affected_wells)
    )
    if not focused_cluster:
        return cluster_scope
    ordered_scope: list[str] = []
    for well_name in [*focused_cluster, *cluster_scope]:
        normalized = str(well_name).strip()
        if normalized and normalized not in ordered_scope:
            ordered_scope.append(normalized)
    return tuple(ordered_scope)


def pad_scoped_cluster_focus_well_names(
    *,
    cluster: AntiCollisionRecommendationCluster,
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    focus_set = {str(name) for name in focus_pad_well_names if str(name).strip()}
    if not focus_set:
        return ()
    focused_affected = tuple(
        str(name) for name in cluster.affected_wells if str(name) in focus_set
    )
    if focused_affected:
        return focused_affected
    return tuple(str(name) for name in cluster.well_names if str(name) in focus_set)


def anticollision_focus_well_names(
    *,
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    focus_set = {str(name) for name in focus_pad_well_names if str(name).strip()}
    if not focus_set:
        return ()
    related = set(focus_set)
    for cluster in clusters:
        if focus_set.intersection(str(name) for name in cluster.well_names):
            related.update(str(name) for name in cluster.well_names)
            related.update(str(name) for name in cluster.affected_wells)
    return tuple(sorted(related))
