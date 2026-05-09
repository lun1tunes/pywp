from __future__ import annotations

import math

from pywp import ptc_anticollision_view
from pywp.anticollision_recommendations import (
    AntiCollisionClusterActionStep,
    AntiCollisionRecommendation,
    AntiCollisionRecommendationCluster,
)


def _recommendation(
    recommendation_id: str = "ac-rec-001",
    *,
    well_a: str = "PAD1-A",
    well_b: str = "PAD2-A",
    category: str = "target_spacing",
    affected_wells: tuple[str, ...] = ("PAD1-A",),
    can_prepare_rerun: bool = True,
    min_separation_factor: float = 0.64,
) -> AntiCollisionRecommendation:
    return AntiCollisionRecommendation(
        recommendation_id=recommendation_id,
        well_a=well_a,
        well_b=well_b,
        priority_rank=1,
        category=category,
        summary="Увеличить расстояние между конфликтными участками минимум на 35.0 м.",
        detail="Детали",
        expected_maneuver="Маневр",
        action_label="Подготовить пересчет",
        can_prepare_rerun=can_prepare_rerun,
        affected_wells=affected_wells,
        override_suggestions=(),
        classification="target",
        area_label="t1",
        md_a_start_m=100.0,
        md_a_end_m=100.0,
        md_b_start_m=120.0,
        md_b_end_m=250.0,
        min_separation_factor=min_separation_factor,
        max_overlap_depth_m=34.9,
        min_center_distance_m=12.3,
        required_spacing_t1_m=34.9,
        required_spacing_t3_m=None,
    )


def _cluster(
    recommendations: tuple[AntiCollisionRecommendation, ...],
) -> AntiCollisionRecommendationCluster:
    return AntiCollisionRecommendationCluster(
        cluster_id="ac-cluster-001",
        well_names=("PAD1-A", "PAD2-A", "PAD2-B"),
        recommendations=recommendations,
        recommendation_count=len(recommendations),
        target_conflict_count=1,
        vertical_conflict_count=0,
        trajectory_conflict_count=0,
        worst_separation_factor=0.64,
        summary="Кластер",
        detail="Детали кластера",
        expected_maneuver="Маневр кластера",
        blocking_advisory=None,
        rerun_order_label="PAD1-A",
        first_rerun_well="PAD1-A",
        first_rerun_maneuver="Маневр",
        action_steps=(
            AntiCollisionClusterActionStep(
                order_rank=1,
                well_name="PAD1-A",
                category="target_spacing",
                optimization_mode="anti_collision_avoidance",
                expected_maneuver="Маневр",
                reason="Причина",
                related_recommendation_count=1,
                worst_separation_factor=0.64,
            ),
        ),
        can_prepare_rerun=True,
        affected_wells=("PAD1-A",),
        action_label="Подготовить пересчет",
    )


def test_numeric_formatters_hide_invalid_values() -> None:
    assert ptc_anticollision_view.format_sf_value(0.1234) == "0.12"
    assert ptc_anticollision_view.format_overlap_value(34.987) == "34.99"
    assert ptc_anticollision_view.format_sf_value(None) == "—"
    assert ptc_anticollision_view.format_sf_value(math.nan) == "—"
    assert ptc_anticollision_view.format_overlap_value(math.inf) == "—"


def test_md_interval_and_help_text_are_stable() -> None:
    assert ptc_anticollision_view.md_interval_label(100.0, 100.0) == "100"
    assert ptc_anticollision_view.md_interval_label(100.0, 250.0) == "100-250"

    text = ptc_anticollision_view.sf_help_markdown()
    assert "Separation Factor" in text
    assert "SF < 1" in text
    assert "SF > 1" in text


def test_cluster_scope_expands_focus_pad_to_related_neighbor_wells() -> None:
    rec = _recommendation()
    cluster = _cluster((rec,))

    visible = ptc_anticollision_view.clusters_touching_focus_pad(
        clusters=(cluster,),
        focus_pad_well_names=("PAD1-A",),
    )
    focus_names = ptc_anticollision_view.anticollision_focus_well_names(
        clusters=visible,
        focus_pad_well_names=("PAD1-A",),
    )

    assert visible == (cluster,)
    assert focus_names == ("PAD1-A", "PAD2-A", "PAD2-B")


def test_snapshots_and_report_rows_keep_ui_contract() -> None:
    rec = _recommendation()
    cluster = _cluster((rec,))

    rec_snapshot = ptc_anticollision_view.recommendation_snapshot(rec)
    cluster_snapshot = ptc_anticollision_view.cluster_snapshot(
        cluster,
        target_well_names=("PAD1-A", "PAD2-A"),
        focus_well_names=("PAD1-A",),
    )
    rows = ptc_anticollision_view.report_rows_from_recommendations((rec,))

    assert rec_snapshot["source_label"] == "PAD1-A ↔ PAD2-A · Цели / spacing · SF 0.64"
    assert cluster_snapshot["target_well_names"] == ("PAD1-A", "PAD2-A")
    assert cluster_snapshot["focus_well_names"] == ("PAD1-A",)
    assert cluster_snapshot["before_sf"] == 0.64
    assert rows == [
        {
            "Приоритет": "1",
            "Скважина A": "PAD1-A",
            "Скважина B": "PAD2-A",
            "Участок A": "—",
            "Участок B": "—",
            "Интервал A, м": "100",
            "Интервал B, м": "120-250",
            "SF min": 0.64,
            "Overlap max, м": 34.9,
            "Мин. расстояние, м": 12.3,
            "Рекомендация по устранению": (
                "Увеличить расстояние между конфликтными участками минимум на 35.0 м."
            ),
        }
    ]
