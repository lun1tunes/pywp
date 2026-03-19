from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionCorridor,
    TARGET_T1,
    TARGET_T3,
    anti_collision_report_events,
    anti_collision_report_rows,
    analyze_anti_collision,
    build_anti_collision_well,
    collision_corridor_plan_polygon,
    collision_corridor_tube_mesh,
    collision_zone_plan_polygon,
    collision_zone_sphere_mesh,
)
from pywp.anticollision_rerun import build_anti_collision_analysis_for_successes
from pywp.anticollision_recommendations import (
    MANEUVER_BUILD2_ENTRY,
    MANEUVER_POSTENTRY_TURN,
    RECOMMENDATION_REDUCE_KOP,
    RECOMMENDATION_TARGET_SPACING,
    RECOMMENDATION_TRAJECTORY_REVIEW,
    build_anti_collision_recommendation_clusters,
    anti_collision_recommendation_rows,
    build_anti_collision_recommendations,
    AntiCollisionWellContext,
)
from pywp.models import OPTIMIZATION_ANTI_COLLISION_AVOIDANCE, Point3D
from pywp.welltrack_batch import SuccessfulWellPlan


def _straight_stations(*, y_offset_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 90.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [0.0, 1000.0, 2000.0],
            "Y_m": [y_offset_m, y_offset_m, y_offset_m],
            "Z_m": [0.0, 0.0, 0.0],
        }
    )


def _vertical_build_stations(
    *,
    y_offset_m: float,
    lateral_y_t1_m: float,
    lateral_y_end_m: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "INC_deg": [0.0, 0.0, 0.0, 20.0, 50.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "X_m": [0.0, 0.0, 0.0, 50.0, 320.0],
            "Y_m": [y_offset_m, y_offset_m, y_offset_m, lateral_y_t1_m, lateral_y_end_m],
            "Z_m": [0.0, 500.0, 1000.0, 1450.0, 1775.0],
            "segment": ["VERTICAL", "VERTICAL", "VERTICAL", "BUILD1", "BUILD1"],
        }
    )


def test_anti_collision_analysis_prioritizes_target_overlaps() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )

    assert any(sample.target_label == TARGET_T1 for sample in well_a.samples)
    assert any(sample.target_label == TARGET_T3 for sample in well_a.samples)

    analysis = analyze_anti_collision([well_a, well_b])

    assert analysis.pair_count == 1
    assert analysis.overlapping_pair_count == 1
    assert analysis.target_overlap_pair_count == 1
    assert analysis.corridors
    assert analysis.well_segments
    assert analysis.zones
    assert analysis.zones[0].classification == "target-target"
    assert analysis.zones[0].separation_factor < 1.0

    rows = anti_collision_report_rows(analysis)
    assert rows
    assert rows[0]["Приоритет"] == "Цели ↔ цели"
    assert len(analysis.zones) <= 5
    target_mds = {
        (float(zone.md_a_m), float(zone.md_b_m))
        for zone in analysis.zones
        if zone.classification == "target-target"
    }
    assert (1000.0, 1000.0) in target_mds
    assert (2000.0, 2000.0) in target_mds


def test_collision_zone_geometry_helpers_return_closed_plan_polygon_and_sphere_grid() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    zone = analyze_anti_collision([well_a, well_b]).zones[0]

    polygon = collision_zone_plan_polygon(zone)
    sphere_x, sphere_y, sphere_z = collision_zone_sphere_mesh(zone)

    assert polygon.shape[1] == 2
    assert tuple(polygon[0]) == tuple(polygon[-1])
    assert sphere_x.shape == sphere_y.shape == sphere_z.shape
    assert sphere_x.ndim == 2


def test_collision_corridor_geometry_and_well_segments_are_built() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])

    corridor = analysis.corridors[0]
    polygon = collision_corridor_plan_polygon(corridor)
    tube_mesh = collision_corridor_tube_mesh(corridor)

    assert tuple(polygon[0]) == tuple(polygon[-1])
    assert tube_mesh is not None
    assert tube_mesh.vertices_xyz.shape[1] == 3
    assert {segment.well_name for segment in analysis.well_segments} == {"WELL-A", "WELL-B"}


def test_lightweight_runtime_analysis_matches_full_report_events_and_recommendations() -> None:
    stations_a = _vertical_build_stations(
        y_offset_m=0.0,
        lateral_y_t1_m=60.0,
        lateral_y_end_m=280.0,
    )
    stations_b = _vertical_build_stations(
        y_offset_m=5.0,
        lateral_y_t1_m=140.0,
        lateral_y_end_m=340.0,
    )
    success_a = SuccessfulWellPlan(
        name="WELL-A",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(50.0, 60.0, 1450.0),
        t3=Point3D(320.0, 280.0, 1775.0),
        stations=stations_a,
        summary={"kop_md_m": 820.0},
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    success_b = SuccessfulWellPlan(
        name="WELL-B",
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(50.0, 140.0, 1450.0),
        t3=Point3D(320.0, 340.0, 1775.0),
        stations=stations_b,
        summary={"kop_md_m": 780.0},
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        config={"optimization_mode": "none", "kop_min_vertical_m": 550.0},
    )
    full_analysis = analyze_anti_collision(
        [
            build_anti_collision_well(
                name="WELL-A",
                color="#0B6E4F",
                stations=stations_a,
                surface=success_a.surface,
                t1=success_a.t1,
                t3=success_a.t3,
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="WELL-B",
                color="#D1495B",
                stations=stations_b,
                surface=success_b.surface,
                t1=success_b.t1,
                t3=success_b.t3,
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
        ]
    )
    runtime_analysis = build_anti_collision_analysis_for_successes(
        [success_a, success_b],
        model=full_analysis.wells[0].overlay.model,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=820.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=780.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    assert runtime_analysis.zones
    assert all(len(corridor.overlap_rings_xyz) == 0 for corridor in runtime_analysis.corridors)
    runtime_events = anti_collision_report_events(runtime_analysis)
    full_events = anti_collision_report_events(full_analysis)
    assert len(runtime_events) == len(full_events)
    for runtime_event, full_event in zip(runtime_events, full_events):
        assert runtime_event.well_a == full_event.well_a
        assert runtime_event.well_b == full_event.well_b
        assert runtime_event.classification == full_event.classification
        assert runtime_event.priority_rank == full_event.priority_rank
        assert runtime_event.label_a == full_event.label_a
        assert runtime_event.label_b == full_event.label_b
        assert runtime_event.md_a_start_m == full_event.md_a_start_m
        assert runtime_event.md_a_end_m == full_event.md_a_end_m
        assert runtime_event.md_b_start_m == full_event.md_b_start_m
        assert runtime_event.md_b_end_m == full_event.md_b_end_m
        assert runtime_event.min_separation_factor == pytest.approx(
            full_event.min_separation_factor
        )
        assert runtime_event.max_overlap_depth_m == pytest.approx(
            full_event.max_overlap_depth_m
        )

    runtime_recommendations = build_anti_collision_recommendations(
        runtime_analysis,
        well_context_by_name=contexts,
    )
    full_recommendations = build_anti_collision_recommendations(
        full_analysis,
        well_context_by_name=contexts,
    )
    assert len(runtime_recommendations) == len(full_recommendations)
    for runtime_item, full_item in zip(runtime_recommendations, full_recommendations):
        assert runtime_item.category == full_item.category
        assert runtime_item.summary == full_item.summary
        assert runtime_item.detail == full_item.detail
        assert runtime_item.expected_maneuver == full_item.expected_maneuver
        assert runtime_item.action_label == full_item.action_label
        assert runtime_item.can_prepare_rerun == full_item.can_prepare_rerun
        assert runtime_item.affected_wells == full_item.affected_wells
        assert runtime_item.min_separation_factor == pytest.approx(
            full_item.min_separation_factor
        )
        assert runtime_item.max_overlap_depth_m == pytest.approx(
            full_item.max_overlap_depth_m
        )


def test_overlap_ring_is_not_reduced_to_uniform_circle_for_offset_wells() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])

    overlap_ring = np.asarray(analysis.corridors[0].overlap_rings_xyz[0], dtype=float)
    center = np.mean(overlap_ring, axis=0)
    radial_distances = np.linalg.norm(overlap_ring - center[None, :], axis=1)

    assert float(np.std(radial_distances)) > 0.2


def test_report_merges_adjacent_corridors_into_single_event() -> None:
    corridor_a = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4200.0,
        md_a_end_m=4250.0,
        md_b_start_m=4100.0,
        md_b_end_m=4150.0,
        md_a_values_m=np.array([4200.0, 4250.0], dtype=float),
        md_b_values_m=np.array([4100.0, 4150.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.72, 0.69], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    corridor_b = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4250.0,
        md_a_end_m=4300.0,
        md_b_start_m=4150.0,
        md_b_end_m=4200.0,
        md_a_values_m=np.array([4250.0, 4300.0], dtype=float),
        md_b_values_m=np.array([4150.0, 4200.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.68, 0.66], dtype=float),
        overlap_depth_values_m=np.array([10.0, 11.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor_a, corridor_b),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.66,
    )

    events = anti_collision_report_events(analysis)
    rows = anti_collision_report_rows(analysis)

    assert len(events) == 1
    assert events[0].md_a_start_m == 4200.0
    assert events[0].md_a_end_m == 4300.0
    assert events[0].md_b_start_m == 4100.0
    assert events[0].md_b_end_m == 4200.0
    assert events[0].merged_corridor_count == 2
    assert rows[0]["Интервал A, м"] == "4200 - 4300"
    assert rows[0]["Интервал B, м"] == "4100 - 4200"
    assert rows[0]["Смежных зон"] == 2


def test_recommendations_prioritize_target_spacing_for_target_overlap() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_straight_stations(y_offset_m=0.0),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1000.0, 0.0, 0.0),
        t3=Point3D(2000.0, 0.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_straight_stations(y_offset_m=5.0),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(1000.0, 5.0, 0.0),
        t3=Point3D(2000.0, 5.0, 0.0),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])

    recommendations = build_anti_collision_recommendations(analysis)
    rows = anti_collision_recommendation_rows(recommendations)

    assert recommendations
    assert recommendations[0].category == RECOMMENDATION_TARGET_SPACING
    assert recommendations[0].required_spacing_t1_m is not None
    assert recommendations[0].can_prepare_rerun is False
    assert rows[0]["Тип действия"] == "Цели / spacing"
    assert rows[0]["Подготовка пересчета"] == "Только рекомендация по целям"


def test_recommendations_prepare_anticollision_kop_build1_for_vertical_collision() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_vertical_build_stations(
            y_offset_m=0.0,
            lateral_y_t1_m=60.0,
            lateral_y_end_m=280.0,
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(50.0, 60.0, 1450.0),
        t3=Point3D(320.0, 280.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_vertical_build_stations(
            y_offset_m=5.0,
            lateral_y_t1_m=140.0,
            lateral_y_end_m=340.0,
        ),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(50.0, 140.0, 1450.0),
        t3=Point3D(320.0, 340.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
    )
    analysis = analyze_anti_collision([well_a, well_b])
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=820.0,
            kop_min_vertical_m=550.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=780.0,
            kop_min_vertical_m=550.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    vertical_recommendations = [
        item for item in recommendations if item.category == RECOMMENDATION_REDUCE_KOP
    ]

    assert vertical_recommendations
    recommendation = vertical_recommendations[0]
    assert recommendation.can_prepare_rerun is True
    assert recommendation.affected_wells == ("WELL-A", "WELL-B")
    assert all(
        suggestion.config_updates.get("optimization_mode") == "anti_collision_avoidance"
        for suggestion in recommendation.override_suggestions
    )
    assert recommendation.expected_maneuver == "Сместить ранний уход: KOP / BUILD1"
    assert recommendation.action_label == "Подготовить anti-collision пересчет (KOP/BUILD1)"


def test_pre_kop_trajectory_conflict_is_reclassified_to_kop_build1_anticollision() -> None:
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=_vertical_build_stations(
            y_offset_m=0.0,
            lateral_y_t1_m=60.0,
            lateral_y_end_m=280.0,
        ),
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(50.0, 60.0, 1450.0),
        t3=Point3D(320.0, 280.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=_vertical_build_stations(
            y_offset_m=5.0,
            lateral_y_t1_m=140.0,
            lateral_y_end_m=340.0,
        ),
        surface=Point3D(0.0, 5.0, 0.0),
        t1=Point3D(50.0, 140.0, 1450.0),
        t3=Point3D(320.0, 340.0, 1775.0),
        azimuth_deg=90.0,
        md_t1_m=1500.0,
        include_display_geometry=False,
    )
    corridor = AntiCollisionCorridor(
        well_a="WELL-A",
        well_b="WELL-B",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=100.0,
        md_a_end_m=1450.0,
        md_b_start_m=100.0,
        md_b_end_m=1450.0,
        md_a_values_m=np.array([100.0, 1450.0], dtype=float),
        md_b_values_m=np.array([100.0, 1450.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.6, 0.55], dtype=float),
        overlap_depth_values_m=np.array([10.0, 12.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(well_a, well_b),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.55,
    )
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=820.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=780.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    assert recommendations[0].category == RECOMMENDATION_REDUCE_KOP
    assert recommendations[0].action_label == "Подготовить anti-collision пересчет (KOP/BUILD1)"
    assert all(
        suggestion.config_updates.get("optimization_mode") == "anti_collision_avoidance"
        for suggestion in recommendations[0].override_suggestions
    )


def test_build2_trajectory_conflict_reports_build2_maneuver() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [3600.0, 4000.0, 4300.0, 4580.0, 4680.0],
            "INC_deg": [40.0, 50.0, 70.0, 86.0, 90.0],
            "AZI_deg": [150.0, 180.0, 220.0, 238.3, 238.3],
            "X_m": [0.0, 200.0, 600.0, 950.0, 1050.0],
            "Y_m": [0.0, 120.0, 360.0, 520.0, 580.0],
            "Z_m": [2200.0, 2500.0, 3000.0, 3400.0, 3420.0],
            "segment": ["BUILD2", "BUILD2", "BUILD2", "BUILD2", "HORIZONTAL"],
        }
    )
    well_a = build_anti_collision_well(
        name="WELL-A",
        color="#0B6E4F",
        stations=stations,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(950.0, 520.0, 3400.0),
        t3=Point3D(1050.0, 580.0, 3420.0),
        azimuth_deg=90.0,
        md_t1_m=4580.0,
        include_display_geometry=False,
    )
    well_b = build_anti_collision_well(
        name="WELL-B",
        color="#D1495B",
        stations=stations.assign(Y_m=stations["Y_m"] + 20.0),
        surface=Point3D(0.0, 20.0, 0.0),
        t1=Point3D(950.0, 540.0, 3400.0),
        t3=Point3D(1050.0, 600.0, 3420.0),
        azimuth_deg=90.0,
        md_t1_m=4580.0,
        include_display_geometry=False,
    )
    corridor = AntiCollisionCorridor(
        well_a="WELL-A",
        well_b="WELL-B",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4075.0,
        md_a_end_m=4275.0,
        md_b_start_m=3975.0,
        md_b_end_m=4225.0,
        md_a_values_m=np.array([4075.0, 4275.0], dtype=float),
        md_b_values_m=np.array([3975.0, 4225.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.7, 0.65], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(well_a, well_b),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.65,
    )
    contexts = {
        "WELL-A": AntiCollisionWellContext(
            well_name="WELL-A",
            kop_md_m=2000.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4580.0,
            md_total_m=5000.0,
            optimization_mode="none",
        ),
        "WELL-B": AntiCollisionWellContext(
            well_name="WELL-B",
            kop_md_m=1900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4580.0,
            md_total_m=5000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    assert recommendations[0].category == RECOMMENDATION_TRAJECTORY_REVIEW
    assert recommendations[0].expected_maneuver == MANEUVER_BUILD2_ENTRY


def test_recommendations_prepare_pairwise_anti_collision_rerun_for_trajectory_collision() -> None:
    corridor = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4200.0,
        md_a_end_m=4300.0,
        md_b_start_m=4100.0,
        md_b_end_m=4200.0,
        md_a_values_m=np.array([4200.0, 4300.0], dtype=float),
        md_b_values_m=np.array([4100.0, 4200.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.72, 0.69], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    contexts = {
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3500.0,
            md_total_m=5600.0,
            optimization_mode="none",
        ),
        "well_05": AntiCollisionWellContext(
            well_name="well_05",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.category == RECOMMENDATION_TRAJECTORY_REVIEW
    assert recommendation.can_prepare_rerun is True
    assert recommendation.affected_wells == ("well_05",)
    assert recommendation.override_suggestions[0].config_updates["optimization_mode"] == (
        OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    )
    assert recommendation.expected_maneuver == "Pre-entry azimuth turn / сдвиг HOLD до t1"
    rows = anti_collision_recommendation_rows(recommendations)
    assert rows[0]["Ожидаемый маневр"] == recommendation.expected_maneuver


def test_recommendations_switch_movable_well_after_one_side_already_used_anticollision() -> None:
    corridor = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_05",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=4200.0,
        md_a_end_m=4300.0,
        md_b_start_m=4100.0,
        md_b_end_m=4200.0,
        md_a_values_m=np.array([4200.0, 4300.0], dtype=float),
        md_b_values_m=np.array([4100.0, 4200.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
        overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
        separation_factor_values=np.array([0.72, 0.69], dtype=float),
        overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor,),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=1,
        target_overlap_pair_count=0,
        worst_separation_factor=0.69,
    )
    contexts = {
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=2200.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4500.0,
            md_total_m=5600.0,
            optimization_mode=OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
        ),
        "well_05": AntiCollisionWellContext(
            well_name="well_05",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="minimize_kop",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.category == RECOMMENDATION_TRAJECTORY_REVIEW
    assert recommendation.affected_wells == ("well_05",)
    assert recommendation.override_suggestions[0].well_name == "well_05"


def test_recommendation_clusters_merge_connected_pairs_into_single_cluster() -> None:
    corridor_ab = AntiCollisionCorridor(
        well_a="well_01",
        well_b="well_03",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=2200.0,
        md_a_end_m=2400.0,
        md_b_start_m=2100.0,
        md_b_end_m=2350.0,
        md_a_values_m=np.array([2200.0, 2400.0], dtype=float),
        md_b_values_m=np.array([2100.0, 2350.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
        overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
        separation_factor_values=np.array([0.82, 0.78], dtype=float),
        overlap_depth_values_m=np.array([4.0, 5.0], dtype=float),
    )
    corridor_bc = AntiCollisionCorridor(
        well_a="well_02",
        well_b="well_03",
        classification="trajectory",
        priority_rank=2,
        label_a="",
        label_b="",
        md_a_start_m=2600.0,
        md_a_end_m=2900.0,
        md_b_start_m=2550.0,
        md_b_end_m=2850.0,
        md_a_values_m=np.array([2600.0, 2900.0], dtype=float),
        md_b_values_m=np.array([2550.0, 2850.0], dtype=float),
        label_a_values=("", ""),
        label_b_values=("", ""),
        midpoint_xyz=np.array([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float),
        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
        overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
        separation_factor_values=np.array([0.76, 0.71], dtype=float),
        overlap_depth_values_m=np.array([6.0, 7.0], dtype=float),
    )
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(corridor_ab, corridor_bc),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.71,
    )
    contexts = {
        "well_01": AntiCollisionWellContext(
            well_name="well_01",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3500.0,
            md_total_m=5200.0,
            optimization_mode="none",
        ),
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3600.0,
            md_total_m=5400.0,
            optimization_mode="none",
        ),
        "well_03": AntiCollisionWellContext(
            well_name="well_03",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.well_names == ("well_01", "well_02", "well_03")
    assert cluster.recommendation_count == 2
    assert cluster.trajectory_conflict_count == 2
    assert cluster.can_prepare_rerun is True
    assert cluster.affected_wells == ("well_03",)
    assert cluster.first_rerun_well == "well_03"
    assert cluster.rerun_order_label == "well_03"
    assert cluster.action_steps[0].well_name == "well_03"
    assert cluster.action_steps[0].optimization_mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    rows = cluster.recommendations
    assert rows[0].expected_maneuver


def test_cluster_with_target_spacing_conflict_is_not_actionable_for_rerun() -> None:
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(
            AntiCollisionCorridor(
                well_a="well_01",
                well_b="well_02",
                classification="target-target",
                priority_rank=1,
                label_a=TARGET_T1,
                label_b=TARGET_T1,
                md_a_start_m=3500.0,
                md_a_end_m=3500.0,
                md_b_start_m=3600.0,
                md_b_end_m=3600.0,
                md_a_values_m=np.array([3500.0], dtype=float),
                md_b_values_m=np.array([3600.0], dtype=float),
                label_a_values=(TARGET_T1,),
                label_b_values=(TARGET_T1,),
                midpoint_xyz=np.array([[1000.0, 0.0, 2400.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([12.0], dtype=float),
                separation_factor_values=np.array([0.62], dtype=float),
                overlap_depth_values_m=np.array([18.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="well_02",
                well_b="well_03",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=4100.0,
                md_a_end_m=4550.0,
                md_b_start_m=4050.0,
                md_b_end_m=4500.0,
                md_a_values_m=np.array([4100.0, 4550.0], dtype=float),
                md_b_values_m=np.array([4050.0, 4500.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([6.0, 6.0], dtype=float),
                separation_factor_values=np.array([0.81, 0.76], dtype=float),
                overlap_depth_values_m=np.array([7.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=1,
        worst_separation_factor=0.62,
    )
    contexts = {
        "well_01": AntiCollisionWellContext(
            well_name="well_01",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3500.0,
            md_total_m=5200.0,
            optimization_mode="none",
        ),
        "well_02": AntiCollisionWellContext(
            well_name="well_02",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=3600.0,
            md_total_m=5400.0,
            optimization_mode="none",
        ),
        "well_03": AntiCollisionWellContext(
            well_name="well_03",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=4700.0,
            md_total_m=6100.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.target_conflict_count == 1
    assert cluster.trajectory_conflict_count == 1
    assert cluster.can_prepare_rerun is False
    assert cluster.action_label == "Только advisory"
    assert cluster.blocking_advisory == "Сначала решить spacing целей."


def test_cluster_action_steps_normalize_mixed_vertical_and_trajectory_for_same_well() -> None:
    analysis = AntiCollisionAnalysis(
        wells=(
            build_anti_collision_well(
                name="well_a",
                color="#0B6E4F",
                stations=_vertical_build_stations(
                    y_offset_m=0.0,
                    lateral_y_t1_m=30.0,
                    lateral_y_end_m=120.0,
                ),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(50.0, 30.0, 1450.0),
                t3=Point3D(320.0, 120.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_b",
                color="#3A86FF",
                stations=_vertical_build_stations(
                    y_offset_m=10.0,
                    lateral_y_t1_m=40.0,
                    lateral_y_end_m=150.0,
                ),
                surface=Point3D(0.0, 10.0, 0.0),
                t1=Point3D(50.0, 40.0, 1450.0),
                t3=Point3D(320.0, 150.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_c",
                color="#00798C",
                stations=_straight_stations(y_offset_m=0.0),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(1000.0, 0.0, 0.0),
                t3=Point3D(2000.0, 0.0, 0.0),
                azimuth_deg=90.0,
                md_t1_m=1000.0,
            ),
        ),
        corridors=(
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_b",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=400.0,
                md_a_end_m=700.0,
                md_b_start_m=410.0,
                md_b_end_m=710.0,
                md_a_values_m=np.array([400.0, 700.0], dtype=float),
                md_b_values_m=np.array([410.0, 710.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                separation_factor_values=np.array([0.82, 0.78], dtype=float),
                overlap_depth_values_m=np.array([5.0, 7.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_c",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1750.0,
                md_a_end_m=2050.0,
                md_b_start_m=1700.0,
                md_b_end_m=2000.0,
                md_a_values_m=np.array([1750.0, 2050.0], dtype=float),
                md_b_values_m=np.array([1700.0, 2000.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([5.0, 5.0], dtype=float),
                separation_factor_values=np.array([0.76, 0.73], dtype=float),
                overlap_depth_values_m=np.array([7.0, 9.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.73,
    )
    contexts = {
        "well_a": AntiCollisionWellContext(
            well_name="well_a",
            kop_md_m=700.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2600.0,
            optimization_mode="none",
        ),
        "well_b": AntiCollisionWellContext(
            well_name="well_b",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1500.0,
            md_total_m=2700.0,
            optimization_mode="none",
        ),
        "well_c": AntiCollisionWellContext(
            well_name="well_c",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            md_t1_m=1000.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    cluster = clusters[0]
    step = next(item for item in cluster.action_steps if item.well_name == "well_a")
    assert step.optimization_mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
    assert step.category == "mixed"
    assert step.expected_maneuver == MANEUVER_POSTENTRY_TURN


def test_saturated_early_kop_build1_conflict_becomes_advisory_and_shifts_cluster_to_trajectory() -> None:
    analysis = AntiCollisionAnalysis(
        wells=(
            build_anti_collision_well(
                name="well_a",
                color="#0B6E4F",
                stations=_vertical_build_stations(
                    y_offset_m=0.0,
                    lateral_y_t1_m=30.0,
                    lateral_y_end_m=120.0,
                ),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(50.0, 30.0, 1450.0),
                t3=Point3D(320.0, 120.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_b",
                color="#3A86FF",
                stations=_vertical_build_stations(
                    y_offset_m=10.0,
                    lateral_y_t1_m=40.0,
                    lateral_y_end_m=150.0,
                ),
                surface=Point3D(0.0, 10.0, 0.0),
                t1=Point3D(50.0, 40.0, 1450.0),
                t3=Point3D(320.0, 150.0, 1775.0),
                azimuth_deg=90.0,
                md_t1_m=1500.0,
            ),
            build_anti_collision_well(
                name="well_c",
                color="#00798C",
                stations=_straight_stations(y_offset_m=0.0),
                surface=Point3D(0.0, 0.0, 0.0),
                t1=Point3D(1000.0, 0.0, 0.0),
                t3=Point3D(2000.0, 0.0, 0.0),
                azimuth_deg=90.0,
                md_t1_m=1000.0,
            ),
        ),
        corridors=(
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_b",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=400.0,
                md_a_end_m=700.0,
                md_b_start_m=410.0,
                md_b_end_m=710.0,
                md_a_values_m=np.array([400.0, 700.0], dtype=float),
                md_b_values_m=np.array([410.0, 710.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 700.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([6.0, 7.0], dtype=float),
                separation_factor_values=np.array([0.55, 0.52], dtype=float),
                overlap_depth_values_m=np.array([8.0, 9.0], dtype=float),
            ),
            AntiCollisionCorridor(
                well_a="well_a",
                well_b="well_c",
                classification="trajectory",
                priority_rank=2,
                label_a="",
                label_b="",
                md_a_start_m=1750.0,
                md_a_end_m=2050.0,
                md_b_start_m=1700.0,
                md_b_end_m=2000.0,
                md_a_values_m=np.array([1750.0, 2050.0], dtype=float),
                md_b_values_m=np.array([1700.0, 2000.0], dtype=float),
                label_a_values=("", ""),
                label_b_values=("", ""),
                midpoint_xyz=np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=float),
                overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                overlap_core_radius_m=np.array([3.0, 3.0], dtype=float),
                separation_factor_values=np.array([0.82, 0.80], dtype=float),
                overlap_depth_values_m=np.array([4.0, 4.0], dtype=float),
            ),
        ),
        well_segments=(),
        zones=(),
        pair_count=2,
        overlapping_pair_count=2,
        target_overlap_pair_count=0,
        worst_separation_factor=0.52,
    )
    contexts = {
        "well_a": AntiCollisionWellContext(
            well_name="well_a",
            kop_md_m=550.0,
            kop_min_vertical_m=550.0,
            build1_dls_deg_per_30m=3.0,
            build_dls_max_deg_per_30m=3.0,
            md_t1_m=1500.0,
            md_total_m=2600.0,
            optimization_mode="anti_collision_avoidance",
        ),
        "well_b": AntiCollisionWellContext(
            well_name="well_b",
            kop_md_m=550.0,
            kop_min_vertical_m=550.0,
            build1_dls_deg_per_30m=3.0,
            build_dls_max_deg_per_30m=3.0,
            md_t1_m=1500.0,
            md_total_m=2700.0,
            optimization_mode="anti_collision_avoidance",
        ),
        "well_c": AntiCollisionWellContext(
            well_name="well_c",
            kop_md_m=650.0,
            kop_min_vertical_m=550.0,
            build1_dls_deg_per_30m=2.0,
            build_dls_max_deg_per_30m=3.0,
            md_t1_m=1000.0,
            md_total_m=2000.0,
            optimization_mode="none",
        ),
    }

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=contexts,
    )
    early = next(
        item
        for item in recommendations
        if {item.well_a, item.well_b} == {"well_a", "well_b"}
    )
    assert early.category == RECOMMENDATION_REDUCE_KOP
    assert early.can_prepare_rerun is False

    clusters = build_anti_collision_recommendation_clusters(recommendations)
    cluster = clusters[0]
    assert cluster.action_steps
    step = cluster.action_steps[0]
    assert step.category == RECOMMENDATION_TRAJECTORY_REVIEW


def test_cluster_action_steps_keep_worst_sf_first_and_preserve_order_label() -> None:
    recommendations = (
        build_anti_collision_recommendations(
            AntiCollisionAnalysis(
                wells=(),
                corridors=(
                    AntiCollisionCorridor(
                        well_a="well_a",
                        well_b="well_c",
                        classification="trajectory",
                        priority_rank=2,
                        label_a="",
                        label_b="",
                        md_a_start_m=2500.0,
                        md_a_end_m=2700.0,
                        md_b_start_m=2600.0,
                        md_b_end_m=2800.0,
                        md_a_values_m=np.array([2500.0, 2700.0], dtype=float),
                        md_b_values_m=np.array([2600.0, 2800.0], dtype=float),
                        label_a_values=("", ""),
                        label_b_values=("", ""),
                        midpoint_xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
                        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                        overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                        separation_factor_values=np.array([0.82, 0.75], dtype=float),
                        overlap_depth_values_m=np.array([5.0, 7.0], dtype=float),
                    ),
                    AntiCollisionCorridor(
                        well_a="well_a",
                        well_b="well_b",
                        classification="trajectory",
                        priority_rank=2,
                        label_a="",
                        label_b="",
                        md_a_start_m=600.0,
                        md_a_end_m=900.0,
                        md_b_start_m=620.0,
                        md_b_end_m=910.0,
                        md_a_values_m=np.array([600.0, 900.0], dtype=float),
                        md_b_values_m=np.array([620.0, 910.0], dtype=float),
                        label_a_values=("", ""),
                        label_b_values=("", ""),
                        midpoint_xyz=np.array([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=float),
                        overlap_rings_xyz=(np.zeros((16, 3), dtype=float), np.ones((16, 3), dtype=float)),
                        overlap_core_radius_m=np.array([4.0, 4.0], dtype=float),
                        separation_factor_values=np.array([0.79, 0.77], dtype=float),
                        overlap_depth_values_m=np.array([4.0, 5.0], dtype=float),
                    ),
                ),
                well_segments=(),
                zones=(),
                pair_count=2,
                overlapping_pair_count=2,
                target_overlap_pair_count=0,
                worst_separation_factor=0.75,
            ),
            well_context_by_name={
                "well_a": AntiCollisionWellContext(
                    well_name="well_a",
                    kop_md_m=900.0,
                    kop_min_vertical_m=550.0,
                    md_t1_m=3400.0,
                    md_total_m=6000.0,
                    optimization_mode="none",
                ),
                "well_b": AntiCollisionWellContext(
                    well_name="well_b",
                    kop_md_m=840.0,
                    kop_min_vertical_m=550.0,
                    md_t1_m=1800.0,
                    md_total_m=5300.0,
                    optimization_mode="none",
                ),
                "well_c": AntiCollisionWellContext(
                    well_name="well_c",
                    kop_md_m=950.0,
                    kop_min_vertical_m=550.0,
                    md_t1_m=4200.0,
                    md_total_m=6200.0,
                    optimization_mode="none",
                ),
            },
        )
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert tuple(step.well_name for step in cluster.action_steps) == ("well_c", "well_a")
    assert cluster.first_rerun_well == "well_c"
    assert cluster.rerun_order_label == "well_c → well_a"
