from __future__ import annotations

import numpy as np
import pandas as pd

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
from pywp.anticollision_recommendations import (
    RECOMMENDATION_REDUCE_KOP,
    RECOMMENDATION_TARGET_SPACING,
    RECOMMENDATION_TRAJECTORY_REVIEW,
    anti_collision_recommendation_rows,
    build_anti_collision_recommendations,
    AntiCollisionWellContext,
)
from pywp.models import OPTIMIZATION_ANTI_COLLISION_AVOIDANCE, Point3D


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


def test_recommendations_prepare_minimize_kop_for_vertical_collision() -> None:
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
        suggestion.config_updates.get("optimization_mode") == "minimize_kop"
        for suggestion in recommendation.override_suggestions
    )


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
            md_total_m=5600.0,
            optimization_mode="none",
        ),
        "well_05": AntiCollisionWellContext(
            well_name="well_05",
            kop_md_m=900.0,
            kop_min_vertical_m=550.0,
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
