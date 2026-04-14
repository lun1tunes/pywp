from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from pywp.constants import SMALL
from pywp.models import Point3D
from pywp.uncertainty import (
    DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    PlanningUncertaintyModel,
    UncertaintyEllipseSample,
    UncertaintyStationSample,
    UncertaintyTubeMesh,
    WellUncertaintyOverlay,
    build_uncertainty_overlay,
    build_uncertainty_station_samples,
)

TARGET_NONE = ""
TARGET_T1 = "t1"
TARGET_T3 = "t3"
_PAIR_XY_PREFILTER_DIAMETER_FACTOR = 1.5
_PAIR_TERMINAL_PREFILTER_DIAMETER_FACTOR = 1.35


@dataclass(frozen=True)
class AntiCollisionSample:
    md_m: float
    center_xyz: tuple[float, float, float]
    covariance_xyz: np.ndarray
    target_label: str = TARGET_NONE


@dataclass(frozen=True)
class AntiCollisionWell:
    name: str
    color: str
    overlay: WellUncertaintyOverlay
    samples: tuple[AntiCollisionSample, ...]
    stations: pd.DataFrame
    surface: Point3D
    t1: Point3D | None
    t3: Point3D | None
    md_t1_m: float | None
    md_t3_m: float | None
    well_kind: str = "project"
    is_reference_only: bool = False


@dataclass(frozen=True)
class AntiCollisionZone:
    well_a: str
    well_b: str
    classification: str
    priority_rank: int
    label_a: str
    label_b: str
    md_a_m: float
    md_b_m: float
    center_distance_m: float
    combined_radius_m: float
    overlap_depth_m: float
    separation_factor: float
    hotspot_xyz: tuple[float, float, float]
    display_radius_m: float


@dataclass(frozen=True)
class AntiCollisionCorridor:
    well_a: str
    well_b: str
    classification: str
    priority_rank: int
    label_a: str
    label_b: str
    md_a_start_m: float
    md_a_end_m: float
    md_b_start_m: float
    md_b_end_m: float
    md_a_values_m: np.ndarray
    md_b_values_m: np.ndarray
    label_a_values: tuple[str, ...]
    label_b_values: tuple[str, ...]
    midpoint_xyz: np.ndarray
    overlap_rings_xyz: tuple[np.ndarray, ...]
    overlap_core_radius_m: np.ndarray
    separation_factor_values: np.ndarray
    overlap_depth_values_m: np.ndarray


@dataclass(frozen=True)
class AntiCollisionWellSegment:
    well_name: str
    md_start_m: float
    md_end_m: float
    classification: str
    priority_rank: int


@dataclass(frozen=True)
class AntiCollisionReportEvent:
    well_a: str
    well_b: str
    classification: str
    priority_rank: int
    label_a: str
    label_b: str
    md_a_start_m: float
    md_a_end_m: float
    md_b_start_m: float
    md_b_end_m: float
    min_separation_factor: float
    max_overlap_depth_m: float
    merged_corridor_count: int


@dataclass(frozen=True)
class AntiCollisionAnalysis:
    wells: tuple[AntiCollisionWell, ...]
    corridors: tuple[AntiCollisionCorridor, ...]
    well_segments: tuple[AntiCollisionWellSegment, ...]
    zones: tuple[AntiCollisionZone, ...]
    pair_count: int
    overlapping_pair_count: int
    target_overlap_pair_count: int
    worst_separation_factor: float | None


@dataclass(frozen=True)
class _AntiCollisionLateralEnvelope:
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    max_lateral_radius_m: float
    surface_x_m: float
    surface_y_m: float
    terminal_x_m: float
    terminal_y_m: float
    terminal_z_m: float
    terminal_lateral_radius_m: float
    terminal_spatial_radius_m: float


def anti_collision_method_caption(
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> str:
    return (
        "Anti-collision scan выполнен по planning-level 2σ конусам неопределенности. "
        "Для каждой пары скважин считается расстояние между центрами сечений и "
        "combined directional 2σ radius по суммарной ковариации двух скважин "
        "(uncorrelated uncertainty assumption). Красный volume/polygon показывает "
        "приближенное общее пересечение двух конусов: в каждом конфликтном "
        "сечении строится polygon-intersection uncertainty-контуров в общей "
        "локальной плоскости. Красные участки стволов показывают MD-интервалы "
        "с overlap. Приоритет в отчете отдается пересечениям в t1/t3."
        f" Базовая модель uncertainty: INC/AZI = {float(model.sigma_inc_deg):.2f}°/"
        f"{float(model.sigma_azi_deg):.2f}° (1σ), lateral drift "
        f"{float(model.sigma_lateral_drift_m_per_1000m):.1f} м/1000м (1σ), "
        "взвешенный по латеральной экспозиции sin(INC)."
    )


def build_anti_collision_well(
    *,
    name: str,
    color: str,
    stations: pd.DataFrame,
    surface: Point3D,
    t1: Point3D | None,
    t3: Point3D | None,
    azimuth_deg: float,
    md_t1_m: float | None,
    md_t3_m: float | None = None,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    include_display_geometry: bool = True,
    well_kind: str = "project",
    is_reference_only: bool = False,
) -> AntiCollisionWell:
    md_t3_m = (
        float(md_t3_m)
        if md_t3_m is not None
        else (float(stations["MD_m"].iloc[-1]) if len(stations) else None)
    )
    required_md_values = tuple(
        float(value) for value in (md_t1_m, md_t3_m) if value is not None
    )
    if include_display_geometry:
        overlay = build_uncertainty_overlay(
            stations=stations,
            surface=surface,
            azimuth_deg=azimuth_deg,
            model=model,
            required_md_m=required_md_values,
        )
        samples = tuple(
            _build_collision_sample(
                sample=sample,
                confidence_scale=float(model.confidence_scale),
                md_t1_m=None if md_t1_m is None else float(md_t1_m),
                md_t3_m=None if md_t3_m is None else float(md_t3_m),
            )
            for sample in overlay.samples
        )
    else:
        overlay = WellUncertaintyOverlay(samples=(), model=model)
        samples = tuple(
            _build_collision_sample_from_station_sample(
                sample=sample,
                md_t1_m=_optional_float(md_t1_m),
                md_t3_m=_optional_float(md_t3_m),
            )
            for sample in build_uncertainty_station_samples(
                stations=stations,
                model=model,
                required_md_m=required_md_values,
            )
        )
    return AntiCollisionWell(
        name=str(name),
        color=str(color),
        overlay=overlay,
        samples=samples,
        stations=stations.copy(),
        surface=surface,
        t1=t1,
        t3=t3,
        md_t1_m=None if md_t1_m is None else float(md_t1_m),
        md_t3_m=None if md_t3_m is None else float(md_t3_m),
        well_kind=str(well_kind),
        is_reference_only=bool(is_reference_only),
    )


def analyze_anti_collision(
    wells: list[AntiCollisionWell] | tuple[AntiCollisionWell, ...],
    *,
    build_overlap_geometry: bool = True,
    pair_filter: Callable[[AntiCollisionWell, AntiCollisionWell], bool] | None = None,
) -> AntiCollisionAnalysis:
    ordered_wells = tuple(wells)
    if build_overlap_geometry:
        for well in ordered_wells:
            if well.samples and not well.overlay.samples:
                raise ValueError(
                    "build_overlap_geometry=True requires wells built with display geometry."
                )
    lateral_envelopes = {
        well.name: _lateral_envelope_for_prefilter(well) for well in ordered_wells
    }
    corridors: list[AntiCollisionCorridor] = []
    zones: list[AntiCollisionZone] = []
    pair_count = 0
    for left_index in range(len(ordered_wells)):
        for right_index in range(left_index + 1, len(ordered_wells)):
            well_a = ordered_wells[left_index]
            well_b = ordered_wells[right_index]
            if not _should_analyze_pair(
                well_a=well_a,
                well_b=well_b,
                pair_filter=pair_filter,
            ):
                continue
            pair_count += 1
            if _pair_prefilter_xy_far_apart(
                lateral_envelope_a=lateral_envelopes[well_a.name],
                lateral_envelope_b=lateral_envelopes[well_b.name],
            ):
                continue
            pair_corridors = _pair_overlap_corridors(
                well_a=well_a,
                well_b=well_b,
                build_overlap_geometry=build_overlap_geometry,
            )
            corridors.extend(pair_corridors)
            zones.extend(
                _corridor_summary_zone(corridor) for corridor in pair_corridors
            )

    zones = sorted(
        zones,
        key=lambda zone: (
            int(zone.priority_rank),
            float(zone.separation_factor),
            -float(zone.overlap_depth_m),
            str(zone.well_a),
            str(zone.well_b),
        ),
    )
    pair_keys = {
        tuple(sorted((corridor.well_a, corridor.well_b))) for corridor in corridors
    }
    target_pair_keys = {
        tuple(sorted((corridor.well_a, corridor.well_b)))
        for corridor in corridors
        if int(corridor.priority_rank) < 2
    }
    worst_sf = (
        None
        if not corridors
        else float(
            min(
                float(np.min(corridor.separation_factor_values))
                for corridor in corridors
            )
        )
    )
    return AntiCollisionAnalysis(
        wells=ordered_wells,
        corridors=tuple(corridors),
        well_segments=tuple(_collect_well_overlap_segments(corridors, ordered_wells)),
        zones=tuple(zones),
        pair_count=int(pair_count),
        overlapping_pair_count=int(len(pair_keys)),
        target_overlap_pair_count=int(len(target_pair_keys)),
        worst_separation_factor=worst_sf,
    )


def _should_analyze_pair(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    pair_filter: Callable[[AntiCollisionWell, AntiCollisionWell], bool] | None = None,
) -> bool:
    if bool(well_a.is_reference_only) and bool(well_b.is_reference_only):
        return False
    if pair_filter is not None and not bool(pair_filter(well_a, well_b)):
        return False
    return True


def _lateral_envelope_for_prefilter(
    well: AntiCollisionWell,
) -> _AntiCollisionLateralEnvelope:
    if {"X_m", "Y_m"}.issubset(well.stations.columns):
        x_values = well.stations["X_m"].to_numpy(dtype=float)
        y_values = well.stations["Y_m"].to_numpy(dtype=float)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]
    else:
        x_values = np.asarray(
            [sample.center_xyz[0] for sample in well.samples], dtype=float
        )
        y_values = np.asarray(
            [sample.center_xyz[1] for sample in well.samples], dtype=float
        )
    if x_values.size == 0 or y_values.size == 0:
        surface_x = float(well.surface.x)
        surface_y = float(well.surface.y)
        x_values = np.asarray([surface_x], dtype=float)
        y_values = np.asarray([surface_y], dtype=float)
    terminal_center_xyz = (
        tuple(float(value) for value in well.samples[-1].center_xyz)
        if well.samples
        else (
            float(x_values[-1]),
            float(y_values[-1]),
            float(
                well.stations["Z_m"].to_numpy(dtype=float)[-1]
                if {"Z_m"}.issubset(well.stations.columns) and len(well.stations)
                else float(well.surface.z)
            ),
        )
    )
    max_lateral_radius_m = max(
        (
            _sample_xy_confidence_radius_m(
                covariance_xyz=sample.covariance_xyz,
                confidence_scale=float(well.overlay.model.confidence_scale),
            )
            for sample in well.samples
        ),
        default=0.0,
    )
    terminal_lateral_radius_m = (
        _sample_xy_confidence_radius_m(
            covariance_xyz=well.samples[-1].covariance_xyz,
            confidence_scale=float(well.overlay.model.confidence_scale),
        )
        if well.samples
        else 0.0
    )
    terminal_spatial_radius_m = (
        _sample_3d_confidence_radius_m(
            covariance_xyz=well.samples[-1].covariance_xyz,
            confidence_scale=float(well.overlay.model.confidence_scale),
        )
        if well.samples
        else 0.0
    )
    return _AntiCollisionLateralEnvelope(
        min_x_m=float(np.min(x_values)),
        max_x_m=float(np.max(x_values)),
        min_y_m=float(np.min(y_values)),
        max_y_m=float(np.max(y_values)),
        max_lateral_radius_m=float(max_lateral_radius_m),
        surface_x_m=float(well.surface.x),
        surface_y_m=float(well.surface.y),
        terminal_x_m=float(terminal_center_xyz[0]),
        terminal_y_m=float(terminal_center_xyz[1]),
        terminal_z_m=float(terminal_center_xyz[2]),
        terminal_lateral_radius_m=float(terminal_lateral_radius_m),
        terminal_spatial_radius_m=float(terminal_spatial_radius_m),
    )


def _sample_xy_confidence_radius_m(
    *,
    covariance_xyz: np.ndarray,
    confidence_scale: float,
) -> float:
    covariance_xy = np.asarray(covariance_xyz, dtype=float)[:2, :2]
    if covariance_xy.shape != (2, 2) or not np.all(np.isfinite(covariance_xy)):
        return 0.0
    eigenvalues = np.linalg.eigvalsh(covariance_xy)
    principal_variance = float(max(np.max(eigenvalues), 0.0))
    return float(max(confidence_scale, 0.0)) * float(np.sqrt(principal_variance))


def _sample_3d_confidence_radius_m(
    *,
    covariance_xyz: np.ndarray,
    confidence_scale: float,
) -> float:
    covariance = np.asarray(covariance_xyz, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return 0.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    principal_variance = float(max(np.max(eigenvalues), 0.0))
    return float(max(confidence_scale, 0.0)) * float(np.sqrt(principal_variance))


def _pair_prefilter_xy_far_apart(
    *,
    lateral_envelope_a: _AntiCollisionLateralEnvelope,
    lateral_envelope_b: _AntiCollisionLateralEnvelope,
) -> bool:
    gap_x_m = max(
        0.0,
        float(lateral_envelope_a.min_x_m) - float(lateral_envelope_b.max_x_m),
        float(lateral_envelope_b.min_x_m) - float(lateral_envelope_a.max_x_m),
    )
    gap_y_m = max(
        0.0,
        float(lateral_envelope_a.min_y_m) - float(lateral_envelope_b.max_y_m),
        float(lateral_envelope_b.min_y_m) - float(lateral_envelope_a.max_y_m),
    )
    xy_gap_m = float(np.hypot(gap_x_m, gap_y_m))
    max_lateral_diameter_m = 2.0 * max(
        float(lateral_envelope_a.max_lateral_radius_m),
        float(lateral_envelope_b.max_lateral_radius_m),
    )
    threshold_m = float(_PAIR_XY_PREFILTER_DIAMETER_FACTOR) * max_lateral_diameter_m
    return bool(xy_gap_m > threshold_m)


def _pair_prefilter_terminal_far_apart(
    *,
    lateral_envelope_a: _AntiCollisionLateralEnvelope,
    lateral_envelope_b: _AntiCollisionLateralEnvelope,
) -> bool:
    surface_xy_distance_m = float(
        np.hypot(
            float(lateral_envelope_a.surface_x_m)
            - float(lateral_envelope_b.surface_x_m),
            float(lateral_envelope_a.surface_y_m)
            - float(lateral_envelope_b.surface_y_m),
        )
    )
    terminal_xy_distance_m = float(
        np.hypot(
            float(lateral_envelope_a.terminal_x_m)
            - float(lateral_envelope_b.terminal_x_m),
            float(lateral_envelope_a.terminal_y_m)
            - float(lateral_envelope_b.terminal_y_m),
        )
    )
    terminal_3d_distance_m = float(
        np.linalg.norm(
            np.asarray(
                [
                    float(lateral_envelope_a.terminal_x_m)
                    - float(lateral_envelope_b.terminal_x_m),
                    float(lateral_envelope_a.terminal_y_m)
                    - float(lateral_envelope_b.terminal_y_m),
                    float(lateral_envelope_a.terminal_z_m)
                    - float(lateral_envelope_b.terminal_z_m),
                ],
                dtype=float,
            )
        )
    )
    terminal_xy_diameter_m = 2.0 * max(
        float(lateral_envelope_a.terminal_lateral_radius_m),
        float(lateral_envelope_b.terminal_lateral_radius_m),
    )
    terminal_3d_diameter_m = 2.0 * max(
        float(lateral_envelope_a.terminal_spatial_radius_m),
        float(lateral_envelope_b.terminal_spatial_radius_m),
    )
    xy_cutoff_m = (
        float(_PAIR_TERMINAL_PREFILTER_DIAMETER_FACTOR) * terminal_xy_diameter_m
    )
    spatial_cutoff_m = (
        float(_PAIR_TERMINAL_PREFILTER_DIAMETER_FACTOR) * terminal_3d_diameter_m
    )
    if surface_xy_distance_m <= xy_cutoff_m:
        return False
    return bool(
        terminal_xy_distance_m > xy_cutoff_m
        and terminal_3d_distance_m > spatial_cutoff_m
    )


def _segment_types_for_interval(
    analysis: AntiCollisionAnalysis,
    well_name: str,
    md_start_m: float,
    md_end_m: float,
) -> str:
    """Extract segment type names (VERTICAL, HOLD, BUILD1, etc.) for given MD interval.

    Returns comma-separated unique segment types that overlap with the interval.
    Uses the stations dataframe from AntiCollisionWell to get actual segment types.
    """
    # Find the well in analysis
    well = None
    for w in analysis.wells:
        if w.name == well_name:
            well = w
            break
    if well is None or well.stations is None or well.stations.empty:
        # Fallback to well_segments if stations not available
        segments: list[str] = []
        for segment in analysis.well_segments:
            if segment.well_name != well_name:
                continue
            if segment.md_end_m < md_start_m or segment.md_start_m > md_end_m:
                continue
            segments.append(segment.classification)
        if not segments:
            return "—"
        seen: set[str] = set()
        unique_segments: list[str] = []
        for seg in segments:
            if seg not in seen:
                seen.add(seg)
                unique_segments.append(seg)
        return ", ".join(unique_segments)

    # Use stations dataframe to get segment types
    df = well.stations
    if "segment" not in df.columns:
        return "—"

    # Filter stations within the MD interval
    mask = (df["MD_m"] >= md_start_m) & (df["MD_m"] <= md_end_m)
    segment_names = df.loc[mask, "segment"].dropna().astype(str).tolist()

    if not segment_names:
        return "—"

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_segments: list[str] = []
    for seg in segment_names:
        seg_upper = seg.upper()
        if seg_upper not in seen:
            seen.add(seg_upper)
            unique_segments.append(seg_upper)
    return ", ".join(unique_segments)


def anti_collision_report_rows(
    analysis: AntiCollisionAnalysis,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in anti_collision_report_events(analysis):
        segment_a = _segment_types_for_interval(
            analysis,
            event.well_a,
            event.md_a_start_m,
            event.md_a_end_m,
        )
        segment_b = _segment_types_for_interval(
            analysis,
            event.well_b,
            event.md_b_start_m,
            event.md_b_end_m,
        )
        rows.append(
            {
                "Приоритет": _priority_label_from_parts(
                    classification=event.classification
                ),
                "Скважина A": event.well_a,
                "Скважина B": event.well_b,
                "Участок A": segment_a,
                "Участок B": segment_b,
                "Интервал A, м": _md_interval_label(
                    float(event.md_a_start_m),
                    float(event.md_a_end_m),
                ),
                "Интервал B, м": _md_interval_label(
                    float(event.md_b_start_m),
                    float(event.md_b_end_m),
                ),
                "SF min": float(event.min_separation_factor),
                "Overlap max, м": float(event.max_overlap_depth_m),
                "Смежных зон": int(event.merged_corridor_count),
            }
        )
    return rows


def anti_collision_report_events(
    analysis: AntiCollisionAnalysis,
) -> tuple[AntiCollisionReportEvent, ...]:
    if not analysis.corridors:
        return ()
    merge_tolerance_m = _corridor_merge_tolerance_m(analysis.wells)
    sorted_corridors = sorted(
        analysis.corridors,
        key=lambda corridor: (
            int(corridor.priority_rank),
            str(corridor.well_a),
            str(corridor.well_b),
            str(corridor.label_a),
            str(corridor.label_b),
            float(corridor.md_a_start_m),
            float(corridor.md_b_start_m),
        ),
    )
    events: list[AntiCollisionReportEvent] = []
    current = _corridor_to_report_event(sorted_corridors[0])
    for corridor in sorted_corridors[1:]:
        if _report_events_can_merge(
            current,
            corridor,
            tolerance_m=merge_tolerance_m,
        ):
            current = _merge_report_event_with_corridor(current, corridor)
            continue
        events.append(current)
        current = _corridor_to_report_event(corridor)
    events.append(current)
    return tuple(events)


def collision_zone_plan_polygon(
    zone: AntiCollisionZone,
    *,
    point_count: int = 40,
) -> np.ndarray:
    return _circle_polygon_xy(
        center_x=float(zone.hotspot_xyz[0]),
        center_y=float(zone.hotspot_xyz[1]),
        radius_m=float(zone.display_radius_m),
        point_count=int(point_count),
    )


def collision_corridor_plan_polygon(
    corridor: AntiCollisionCorridor,
) -> np.ndarray:
    rings_xyz = [np.asarray(ring, dtype=float) for ring in corridor.overlap_rings_xyz]
    if not rings_xyz:
        return np.empty((0, 2), dtype=float)
    rings_xy = [ring[:, :2] for ring in rings_xyz]
    if len(rings_xy) == 1:
        ring_xy = np.asarray(rings_xy[0], dtype=float)
        if len(ring_xy) == 0:
            return np.empty((0, 2), dtype=float)
        return np.vstack([ring_xy, ring_xy[0]])

    positive_side: list[np.ndarray] = []
    negative_side: list[np.ndarray] = []
    centers_xy = np.asarray(corridor.midpoint_xyz[:, :2], dtype=float)
    for index, ring_xy in enumerate(rings_xy):
        center_xy = np.asarray(centers_xy[index], dtype=float)
        tangent_xy = _centerline_tangent_xy(centers_xy, index)
        tangent_norm = float(np.linalg.norm(tangent_xy))
        if tangent_norm <= SMALL:
            tangent_xy = np.array([1.0, 0.0], dtype=float)
            tangent_norm = 1.0
        tangent_xy = tangent_xy / tangent_norm
        normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
        offsets = (np.asarray(ring_xy, dtype=float) - center_xy[None, :]) @ normal_xy
        positive_side.append(np.asarray(ring_xy[int(np.argmax(offsets))], dtype=float))
        negative_side.append(np.asarray(ring_xy[int(np.argmin(offsets))], dtype=float))
    polygon = np.vstack(
        [
            np.asarray(positive_side, dtype=float),
            np.asarray(negative_side[::-1], dtype=float),
        ]
    )
    return np.vstack([polygon, polygon[0]])


def collision_corridor_tube_mesh(
    corridor: AntiCollisionCorridor,
) -> UncertaintyTubeMesh | None:
    rings = [np.asarray(ring, dtype=float) for ring in corridor.overlap_rings_xyz]
    if not rings:
        return None

    points_per_ring = int(rings[0].shape[0])
    if points_per_ring < 3:
        return None
    vertices = np.vstack(rings)
    triangles_i: list[int] = []
    triangles_j: list[int] = []
    triangles_k: list[int] = []
    if len(rings) == 1:
        center_index = int(len(vertices))
        center = np.mean(np.asarray(rings[0], dtype=float), axis=0)
        vertices = np.vstack([vertices, center[None, :]])
        for point_index in range(points_per_ring):
            next_index = (point_index + 1) % points_per_ring
            triangles_i.append(center_index)
            triangles_j.append(point_index)
            triangles_k.append(next_index)
    else:
        for ring_index in range(len(rings) - 1):
            start_a = ring_index * points_per_ring
            start_b = (ring_index + 1) * points_per_ring
            for point_index in range(points_per_ring):
                next_index = (point_index + 1) % points_per_ring
                a0 = start_a + point_index
                a1 = start_a + next_index
                b0 = start_b + point_index
                b1 = start_b + next_index
                triangles_i.extend([a0, a0])
                triangles_j.extend([a1, b1])
                triangles_k.extend([b1, b0])

        start_cap_center_index = int(len(vertices))
        end_cap_center_index = start_cap_center_index + 1
        vertices = np.vstack(
            [
                vertices,
                np.mean(np.asarray(rings[0], dtype=float), axis=0)[None, :],
                np.mean(np.asarray(rings[-1], dtype=float), axis=0)[None, :],
            ]
        )
        for point_index in range(points_per_ring):
            next_index = (point_index + 1) % points_per_ring
            triangles_i.append(start_cap_center_index)
            triangles_j.append(next_index)
            triangles_k.append(point_index)
            last_ring_start = (len(rings) - 1) * points_per_ring
            triangles_i.append(end_cap_center_index)
            triangles_j.append(last_ring_start + point_index)
            triangles_k.append(last_ring_start + next_index)
    return UncertaintyTubeMesh(
        vertices_xyz=np.asarray(vertices, dtype=float),
        i=np.asarray(triangles_i, dtype=int),
        j=np.asarray(triangles_j, dtype=int),
        k=np.asarray(triangles_k, dtype=int),
    )


def collision_corridor_point_sphere_mesh(
    corridor: AntiCollisionCorridor,
    *,
    lat_steps: int = 10,
    lon_steps: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rings = [np.asarray(ring, dtype=float) for ring in corridor.overlap_rings_xyz]
    if not rings:
        return (
            np.empty((0, 0), dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty((0, 0), dtype=float),
        )
    ring = np.asarray(rings[0], dtype=float)
    center = np.mean(ring, axis=0)
    radius = float(max(np.max(np.linalg.norm(ring - center[None, :], axis=1)), 1.0))
    lat = np.linspace(0.0, np.pi, int(max(lat_steps, 4)))
    lon = np.linspace(0.0, 2.0 * np.pi, int(max(lon_steps, 8)))
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    x = center[0] + radius * np.sin(lat_grid) * np.cos(lon_grid)
    y = center[1] + radius * np.sin(lat_grid) * np.sin(lon_grid)
    z = center[2] + radius * np.cos(lat_grid)
    return x, y, z


def collision_zone_sphere_mesh(
    zone: AntiCollisionZone,
    *,
    lat_steps: int = 10,
    lon_steps: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = float(max(zone.display_radius_m, 1.0))
    center = np.asarray(zone.hotspot_xyz, dtype=float)
    lat = np.linspace(0.0, np.pi, int(max(lat_steps, 4)))
    lon = np.linspace(0.0, 2.0 * np.pi, int(max(lon_steps, 8)))
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    x = center[0] + radius * np.sin(lat_grid) * np.cos(lon_grid)
    y = center[1] + radius * np.sin(lat_grid) * np.sin(lon_grid)
    z = center[2] + radius * np.cos(lat_grid)
    return x, y, z


def _build_collision_sample(
    *,
    sample: UncertaintyEllipseSample,
    confidence_scale: float,
    md_t1_m: float | None,
    md_t3_m: float | None,
) -> AntiCollisionSample:
    return AntiCollisionSample(
        md_m=float(sample.md_m),
        center_xyz=tuple(float(value) for value in sample.center_xyz),
        covariance_xyz=_covariance_from_ring(
            ring_xyz=np.asarray(sample.ring_xyz, dtype=float),
            center_xyz=np.asarray(sample.center_xyz, dtype=float),
            confidence_scale=float(confidence_scale),
        ),
        target_label=_target_label(
            md_m=float(sample.md_m),
            md_t1_m=_optional_float(md_t1_m),
            md_t3_m=_optional_float(md_t3_m),
        ),
    )


def _build_collision_sample_from_station_sample(
    *,
    sample: UncertaintyStationSample,
    md_t1_m: float | None,
    md_t3_m: float | None,
) -> AntiCollisionSample:
    return AntiCollisionSample(
        md_m=float(sample.md_m),
        center_xyz=tuple(float(value) for value in sample.center_xyz),
        covariance_xyz=np.asarray(sample.covariance_xyz, dtype=float),
        target_label=_target_label(
            md_m=float(sample.md_m),
            md_t1_m=_optional_float(md_t1_m),
            md_t3_m=_optional_float(md_t3_m),
        ),
    )


def _covariance_from_ring(
    *,
    ring_xyz: np.ndarray,
    center_xyz: np.ndarray,
    confidence_scale: float,
) -> np.ndarray:
    ring_open = np.asarray(ring_xyz, dtype=float)
    if len(ring_open) >= 2 and np.allclose(ring_open[0], ring_open[-1], atol=SMALL):
        ring_open = ring_open[:-1]
    if ring_open.ndim != 2 or ring_open.shape[1] != 3 or len(ring_open) < 3:
        return np.zeros((3, 3), dtype=float)
    offsets = ring_open - np.asarray(center_xyz, dtype=float)[None, :]
    boundary_covariance = np.cov(offsets.T, bias=True)
    scale = float(max(confidence_scale, SMALL))
    return 2.0 * np.asarray(boundary_covariance, dtype=float) / (scale * scale)


def _target_label(*, md_m: float, md_t1_m: float | None, md_t3_m: float | None) -> str:
    if md_t1_m is not None and abs(float(md_m) - float(md_t1_m)) <= 1e-6:
        return TARGET_T1
    if md_t3_m is not None and abs(float(md_m) - float(md_t3_m)) <= 1e-6:
        return TARGET_T3
    return TARGET_NONE


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _pair_overlap_corridors(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    build_overlap_geometry: bool,
) -> list[AntiCollisionCorridor]:
    if not well_a.samples or not well_b.samples:
        return []

    centers_a = np.asarray(
        [sample.center_xyz for sample in well_a.samples], dtype=float
    )
    centers_b = np.asarray(
        [sample.center_xyz for sample in well_b.samples], dtype=float
    )
    covariances_a = np.asarray(
        [np.asarray(sample.covariance_xyz, dtype=float) for sample in well_a.samples],
        dtype=float,
    )
    covariances_b = np.asarray(
        [np.asarray(sample.covariance_xyz, dtype=float) for sample in well_b.samples],
        dtype=float,
    )
    delta = centers_a[:, None, :] - centers_b[None, :, :]
    distance = np.linalg.norm(delta, axis=2)

    direction = np.zeros_like(delta, dtype=float)
    np.divide(
        delta,
        distance[:, :, None],
        out=direction,
        where=distance[:, :, None] > SMALL,
    )

    zero_mask = distance <= SMALL
    if np.any(zero_mask):
        combined_covariance = (
            covariances_a[:, None, :, :] + covariances_b[None, :, :, :]
        )
        zero_covariance = combined_covariance[zero_mask]
        eigenvalues, eigenvectors = np.linalg.eigh(zero_covariance)
        principal_indices = np.argmax(eigenvalues, axis=1)
        principal_vectors = np.take_along_axis(
            eigenvectors,
            principal_indices[:, None, None],
            axis=2,
        )[:, :, 0]
        principal_norm = np.linalg.norm(principal_vectors, axis=1)
        degenerate_mask = (np.max(eigenvalues, axis=1) <= 1e-12) | (
            principal_norm <= 1e-12
        )
        if np.any(degenerate_mask):
            principal_vectors[degenerate_mask] = np.array([1.0, 0.0, 0.0], dtype=float)
            principal_norm[degenerate_mask] = 1.0
        principal_vectors = principal_vectors / principal_norm[:, None]
        direction[zero_mask] = principal_vectors

    combined_sigma2 = np.einsum(
        "abi,aij,abj->ab",
        direction,
        covariances_a,
        direction,
    ) + np.einsum(
        "abi,bij,abj->ab",
        direction,
        covariances_b,
        direction,
    )
    confidence_scale = float(max(well_a.overlay.model.confidence_scale, SMALL))
    combined_radius = confidence_scale * np.sqrt(np.clip(combined_sigma2, 0.0, None))
    overlap_mask = (combined_radius > SMALL) & (distance <= combined_radius)
    if not np.any(overlap_mask):
        return []

    score = np.divide(
        distance,
        np.maximum(combined_radius, SMALL),
        out=np.full_like(distance, np.inf, dtype=float),
        where=combined_radius > SMALL,
    )
    row_best_j = np.argmin(score, axis=1)
    col_best_i = np.argmin(score, axis=0)

    matched_pairs: set[tuple[int, int]] = set()
    for index_a, index_b in enumerate(row_best_j.tolist()):
        if overlap_mask[int(index_a), int(index_b)]:
            matched_pairs.add((int(index_a), int(index_b)))
    for index_b, index_a in enumerate(col_best_i.tolist()):
        if overlap_mask[int(index_a), int(index_b)]:
            matched_pairs.add((int(index_a), int(index_b)))

    overlap_indices = np.argwhere(overlap_mask)
    for index_a, index_b in overlap_indices.tolist():
        sample_a = well_a.samples[int(index_a)]
        sample_b = well_b.samples[int(index_b)]
        if sample_a.target_label or sample_b.target_label:
            matched_pairs.add((int(index_a), int(index_b)))

    if not matched_pairs:
        return []
    return _build_pair_corridors(
        well_a=well_a,
        well_b=well_b,
        pairs=sorted(matched_pairs),
        distance=distance,
        combined_radius=combined_radius,
        build_overlap_geometry=build_overlap_geometry,
    )


def _build_pair_corridors(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    pairs: list[tuple[int, int]],
    distance: np.ndarray,
    combined_radius: np.ndarray,
    build_overlap_geometry: bool,
) -> list[AntiCollisionCorridor]:
    corridors: list[AntiCollisionCorridor] = []
    current_pairs: list[tuple[int, int]] = []

    for index_a, index_b in sorted(
        pairs,
        key=lambda pair: (
            0.5 * (well_a.samples[pair[0]].md_m + well_b.samples[pair[1]].md_m),
            pair[0],
            pair[1],
        ),
    ):
        next_key = _pair_class_key(
            label_a=well_a.samples[int(index_a)].target_label,
            label_b=well_b.samples[int(index_b)].target_label,
        )
        if not current_pairs:
            current_pairs.append((int(index_a), int(index_b)))
            continue
        prev_a, prev_b = current_pairs[-1]
        previous_key = _pair_class_key(
            label_a=well_a.samples[int(prev_a)].target_label,
            label_b=well_b.samples[int(prev_b)].target_label,
        )
        if (
            _pairs_are_contiguous(
                previous_pair=(prev_a, prev_b),
                next_pair=(int(index_a), int(index_b)),
            )
            and previous_key == next_key
        ):
            current_pairs.append((int(index_a), int(index_b)))
            continue
        corridors.append(
            _build_single_corridor(
                well_a=well_a,
                well_b=well_b,
                pairs=current_pairs,
                distance=distance,
                combined_radius=combined_radius,
                build_overlap_geometry=build_overlap_geometry,
            )
        )
        current_pairs = [(int(index_a), int(index_b))]

    if current_pairs:
        corridors.append(
            _build_single_corridor(
                well_a=well_a,
                well_b=well_b,
                pairs=current_pairs,
                distance=distance,
                combined_radius=combined_radius,
                build_overlap_geometry=build_overlap_geometry,
            )
        )
    return corridors


def _pairs_are_contiguous(
    *,
    previous_pair: tuple[int, int],
    next_pair: tuple[int, int],
) -> bool:
    delta_a = int(next_pair[0]) - int(previous_pair[0])
    delta_b = int(next_pair[1]) - int(previous_pair[1])
    if delta_a < 0 or delta_b < 0:
        return False
    if delta_a > 2 or delta_b > 2:
        return False
    return abs(delta_a - delta_b) <= 2


def _pair_class_key(*, label_a: str, label_b: str) -> tuple[int, str, str]:
    classification, priority_rank = _classify_pair_labels(
        label_a=str(label_a),
        label_b=str(label_b),
    )
    return int(priority_rank), str(label_a), str(label_b)


def _build_single_corridor(
    *,
    well_a: AntiCollisionWell,
    well_b: AntiCollisionWell,
    pairs: list[tuple[int, int]],
    distance: np.ndarray,
    combined_radius: np.ndarray,
    build_overlap_geometry: bool,
) -> AntiCollisionCorridor:
    ordered_pairs = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    midpoint_points: list[np.ndarray] = []
    core_radii: list[float] = []
    sf_values: list[float] = []
    overlap_depth_values: list[float] = []
    point_meta: list[tuple[int, str, str]] = []
    md_a_values: list[float] = []
    md_b_values: list[float] = []
    label_a_values: list[str] = []
    label_b_values: list[str] = []
    overlap_rings_xyz: list[np.ndarray] = []

    for index_a, index_b in ordered_pairs:
        sample_a = well_a.samples[int(index_a)]
        sample_b = well_b.samples[int(index_b)]
        if bool(well_a.is_reference_only) or bool(well_b.is_reference_only):
            label_a = TARGET_NONE
            label_b = TARGET_NONE
        else:
            label_a = str(sample_a.target_label)
            label_b = str(sample_b.target_label)
        combined_radius_m = float(combined_radius[int(index_a), int(index_b)])
        center_distance_m = float(distance[int(index_a), int(index_b)])
        overlap_depth_m = float(max(combined_radius_m - center_distance_m, 0.0))
        sf = (
            float(center_distance_m / combined_radius_m)
            if combined_radius_m > SMALL
            else float("inf")
        )
        midpoint_points.append(
            0.5
            * (
                np.asarray(sample_a.center_xyz, dtype=float)
                + np.asarray(sample_b.center_xyz, dtype=float)
            )
        )
        core_radii.append(_overlap_core_radius_m(overlap_depth_m=overlap_depth_m))
        if build_overlap_geometry:
            overlap_rings_xyz.append(
                _overlap_ring_between_samples(
                    ring_a_xyz=np.asarray(
                        well_a.overlay.samples[int(index_a)].ring_xyz,
                        dtype=float,
                    ),
                    ring_b_xyz=np.asarray(
                        well_b.overlay.samples[int(index_b)].ring_xyz,
                        dtype=float,
                    ),
                    center_xyz=midpoint_points[-1],
                    fallback_radius_m=core_radii[-1],
                )
            )
        sf_values.append(sf)
        overlap_depth_values.append(overlap_depth_m)
        priority_rank = _classify_pair_labels(
            label_a=label_a,
            label_b=label_b,
        )[1]
        point_meta.append((int(priority_rank), label_a, label_b))
        md_a_values.append(float(sample_a.md_m))
        md_b_values.append(float(sample_b.md_m))
        label_a_values.append(label_a)
        label_b_values.append(label_b)

    best_meta = min(point_meta, key=lambda item: item[0])
    best_priority = int(best_meta[0])
    classification = {
        0: "target-target",
        1: "target-trajectory",
    }.get(best_priority, "trajectory")
    first_index_a = min(index_a for index_a, _ in ordered_pairs)
    last_index_a = max(index_a for index_a, _ in ordered_pairs)
    first_index_b = min(index_b for _, index_b in ordered_pairs)
    last_index_b = max(index_b for _, index_b in ordered_pairs)
    return AntiCollisionCorridor(
        well_a=well_a.name,
        well_b=well_b.name,
        classification=classification,
        priority_rank=best_priority,
        label_a=str(best_meta[1]),
        label_b=str(best_meta[2]),
        md_a_start_m=_sample_interval_start(well_a.samples, first_index_a),
        md_a_end_m=_sample_interval_end(well_a.samples, last_index_a),
        md_b_start_m=_sample_interval_start(well_b.samples, first_index_b),
        md_b_end_m=_sample_interval_end(well_b.samples, last_index_b),
        md_a_values_m=np.asarray(md_a_values, dtype=float),
        md_b_values_m=np.asarray(md_b_values, dtype=float),
        label_a_values=tuple(label_a_values),
        label_b_values=tuple(label_b_values),
        midpoint_xyz=np.asarray(midpoint_points, dtype=float),
        overlap_rings_xyz=tuple(
            np.asarray(ring, dtype=float) for ring in overlap_rings_xyz
        ),
        overlap_core_radius_m=np.asarray(core_radii, dtype=float),
        separation_factor_values=np.asarray(sf_values, dtype=float),
        overlap_depth_values_m=np.asarray(overlap_depth_values, dtype=float),
    )


def _sample_interval_start(
    samples: tuple[AntiCollisionSample, ...],
    index: int,
) -> float:
    current_md = float(samples[int(index)].md_m)
    if int(index) == 0:
        return current_md
    previous_md = float(samples[int(index) - 1].md_m)
    return 0.5 * (previous_md + current_md)


def _sample_interval_end(
    samples: tuple[AntiCollisionSample, ...],
    index: int,
) -> float:
    current_md = float(samples[int(index)].md_m)
    if int(index) >= len(samples) - 1:
        return current_md
    next_md = float(samples[int(index) + 1].md_m)
    return 0.5 * (current_md + next_md)


def _overlap_core_radius_m(*, overlap_depth_m: float) -> float:
    return float(max(3.0, 0.5 * float(overlap_depth_m)))


def _overlap_ring_between_samples(
    *,
    ring_a_xyz: np.ndarray,
    ring_b_xyz: np.ndarray,
    center_xyz: np.ndarray,
    fallback_radius_m: float,
    resample_points: int = 32,
) -> np.ndarray:
    ring_a = _open_ring(np.asarray(ring_a_xyz, dtype=float))
    ring_b = _open_ring(np.asarray(ring_b_xyz, dtype=float))
    if len(ring_a) < 3 or len(ring_b) < 3:
        return _fallback_circle_ring(
            center_xyz=np.asarray(center_xyz, dtype=float),
            radius_m=float(fallback_radius_m),
            point_count=int(resample_points),
        )

    basis_u, basis_v = _shared_overlap_plane_basis(ring_a=ring_a, ring_b=ring_b)
    center = np.asarray(center_xyz, dtype=float)
    polygon_a_2d = _project_ring_to_plane(ring_a, center, basis_u, basis_v)
    polygon_b_2d = _project_ring_to_plane(ring_b, center, basis_u, basis_v)
    polygon_a_2d = _ensure_ccw_convex_polygon(polygon_a_2d)
    polygon_b_2d = _ensure_ccw_convex_polygon(polygon_b_2d)
    overlap_polygon_2d = _convex_polygon_intersection(polygon_a_2d, polygon_b_2d)
    if len(overlap_polygon_2d) < 3:
        return _fallback_circle_ring(
            center_xyz=np.asarray(center_xyz, dtype=float),
            radius_m=float(fallback_radius_m),
            point_count=int(resample_points),
        )
    resampled_polygon_2d = _resample_closed_polygon_2d(
        overlap_polygon_2d,
        point_count=int(resample_points),
    )
    return (
        center[None, :]
        + resampled_polygon_2d[:, 0:1] * basis_u[None, :]
        + resampled_polygon_2d[:, 1:2] * basis_v[None, :]
    )


def _open_ring(ring_xyz: np.ndarray) -> np.ndarray:
    ring = np.asarray(ring_xyz, dtype=float)
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1], atol=SMALL):
        return ring[:-1]
    return ring


def _shared_overlap_plane_basis(
    *,
    ring_a: np.ndarray,
    ring_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center_a = np.mean(np.asarray(ring_a, dtype=float), axis=0)
    center_b = np.mean(np.asarray(ring_b, dtype=float), axis=0)
    normal_a = _ring_plane_normal(ring_a, center_a)
    normal_b = _ring_plane_normal(ring_b, center_b)
    if float(np.dot(normal_a, normal_b)) < 0.0:
        normal_b = -normal_b
    normal = normal_a + normal_b
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        normal = normal_a if float(np.linalg.norm(normal_a)) > SMALL else normal_b
        normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        normal = center_b - center_a
        normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        normal_norm = 1.0
    normal = normal / normal_norm
    return _stable_normal_plane_basis(normal)


def _ring_plane_normal(ring_xyz: np.ndarray, center_xyz: np.ndarray) -> np.ndarray:
    offsets = (
        np.asarray(ring_xyz, dtype=float) - np.asarray(center_xyz, dtype=float)[None, :]
    )
    if len(offsets) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    _, _, vh = np.linalg.svd(offsets, full_matrices=False)
    if vh.ndim != 2 or vh.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    normal = np.asarray(vh[-1], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= SMALL:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / normal_norm


def _project_ring_to_plane(
    ring_xyz: np.ndarray,
    center_xyz: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> np.ndarray:
    offsets = (
        np.asarray(ring_xyz, dtype=float) - np.asarray(center_xyz, dtype=float)[None, :]
    )
    return np.column_stack(
        [
            offsets @ np.asarray(basis_u, dtype=float),
            offsets @ np.asarray(basis_v, dtype=float),
        ]
    )


def _ensure_ccw_convex_polygon(polygon_2d: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_2d, dtype=float)
    if len(polygon) >= 2 and np.allclose(polygon[0], polygon[-1], atol=SMALL):
        polygon = polygon[:-1]
    if len(polygon) < 3:
        return polygon
    centroid = np.mean(polygon, axis=0)
    angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
    ordered = polygon[np.argsort(angles)]
    area2 = np.sum(
        ordered[:, 0] * np.roll(ordered[:, 1], -1)
        - ordered[:, 1] * np.roll(ordered[:, 0], -1)
    )
    if area2 < 0.0:
        ordered = ordered[::-1]
    return ordered


def _convex_polygon_intersection(
    subject_polygon: np.ndarray,
    clip_polygon: np.ndarray,
) -> np.ndarray:
    output = np.asarray(subject_polygon, dtype=float)
    clip = np.asarray(clip_polygon, dtype=float)
    if len(output) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=float)

    def inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
        edge = edge_end - edge_start
        rel = point - edge_start
        return float(edge[0] * rel[1] - edge[1] * rel[0]) >= -SMALL

    def intersection(
        start: np.ndarray,
        end: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
    ) -> np.ndarray:
        line = end - start
        edge = edge_end - edge_start
        denom = float(line[0] * edge[1] - line[1] * edge[0])
        if abs(denom) <= 1e-12:
            return np.asarray(end, dtype=float)
        diff = edge_start - start
        t = float((diff[0] * edge[1] - diff[1] * edge[0]) / denom)
        return start + t * line

    for edge_start, edge_end in zip(clip, np.roll(clip, -1, axis=0)):
        input_list = np.asarray(output, dtype=float)
        if len(input_list) == 0:
            break
        output_points: list[np.ndarray] = []
        prev_point = np.asarray(input_list[-1], dtype=float)
        prev_inside = inside(prev_point, edge_start, edge_end)
        for current_point in input_list:
            current = np.asarray(current_point, dtype=float)
            current_inside = inside(current, edge_start, edge_end)
            if current_inside:
                if not prev_inside:
                    output_points.append(
                        intersection(prev_point, current, edge_start, edge_end)
                    )
                output_points.append(current)
            elif prev_inside:
                output_points.append(
                    intersection(prev_point, current, edge_start, edge_end)
                )
            prev_point = current
            prev_inside = current_inside
        output = np.asarray(output_points, dtype=float)
    if len(output) < 3:
        return np.empty((0, 2), dtype=float)
    return _ensure_ccw_convex_polygon(output)


def _resample_closed_polygon_2d(
    polygon_2d: np.ndarray, *, point_count: int
) -> np.ndarray:
    polygon = _ensure_ccw_convex_polygon(np.asarray(polygon_2d, dtype=float))
    if len(polygon) < 3:
        return polygon
    closed = np.vstack([polygon, polygon[0]])
    segment_vectors = np.diff(closed, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    perimeter = float(np.sum(segment_lengths))
    if perimeter <= SMALL:
        return polygon
    target_distances = np.linspace(
        0.0, perimeter, int(max(point_count, 12)), endpoint=False
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    resampled: list[np.ndarray] = []
    segment_index = 0
    for distance in target_distances:
        while (
            segment_index < len(segment_lengths) - 1
            and cumulative[segment_index + 1] < distance - SMALL
        ):
            segment_index += 1
        segment_start = closed[segment_index]
        segment_end = closed[segment_index + 1]
        segment_length = float(segment_lengths[segment_index])
        if segment_length <= 1e-12:
            resampled.append(np.asarray(segment_start, dtype=float))
            continue
        local_t = float((distance - cumulative[segment_index]) / segment_length)
        resampled.append(segment_start + local_t * (segment_end - segment_start))
    return np.asarray(resampled, dtype=float)


def _fallback_circle_ring(
    *,
    center_xyz: np.ndarray,
    radius_m: float,
    point_count: int,
) -> np.ndarray:
    basis_u, basis_v = _stable_normal_plane_basis(
        np.array([0.0, 0.0, 1.0], dtype=float)
    )
    angles = np.linspace(0.0, 2.0 * np.pi, int(max(point_count, 12)), endpoint=False)
    radius = float(max(radius_m, 1.0))
    return (
        np.asarray(center_xyz, dtype=float)[None, :]
        + np.cos(angles)[:, None] * (radius * basis_u[None, :])
        + np.sin(angles)[:, None] * (radius * basis_v[None, :])
    )


def _corridor_summary_zone(corridor: AntiCollisionCorridor) -> AntiCollisionZone:
    worst_index = _corridor_representative_index(corridor)
    midpoint_xyz = np.asarray(corridor.midpoint_xyz, dtype=float)
    overlap_depth_values = np.asarray(corridor.overlap_depth_values_m, dtype=float)
    radii = np.asarray(corridor.overlap_core_radius_m, dtype=float)
    overlap_depth_m = float(overlap_depth_values[worst_index])
    separation_factor = float(corridor.separation_factor_values[worst_index])
    combined_radius_m = (
        overlap_depth_m / max(1.0 - separation_factor, SMALL)
        if separation_factor < 1.0
        else overlap_depth_m
    )
    center_distance_m = max(combined_radius_m - overlap_depth_m, 0.0)
    return AntiCollisionZone(
        well_a=corridor.well_a,
        well_b=corridor.well_b,
        classification=corridor.classification,
        priority_rank=int(corridor.priority_rank),
        label_a=corridor.label_a,
        label_b=corridor.label_b,
        md_a_m=float(np.asarray(corridor.md_a_values_m, dtype=float)[worst_index]),
        md_b_m=float(np.asarray(corridor.md_b_values_m, dtype=float)[worst_index]),
        center_distance_m=float(center_distance_m),
        combined_radius_m=float(combined_radius_m),
        overlap_depth_m=float(overlap_depth_m),
        separation_factor=float(separation_factor),
        hotspot_xyz=(
            float(midpoint_xyz[worst_index, 0]),
            float(midpoint_xyz[worst_index, 1]),
            float(midpoint_xyz[worst_index, 2]),
        ),
        display_radius_m=float(radii[worst_index]),
    )


def _corridor_to_report_event(
    corridor: AntiCollisionCorridor,
) -> AntiCollisionReportEvent:
    return AntiCollisionReportEvent(
        well_a=str(corridor.well_a),
        well_b=str(corridor.well_b),
        classification=str(corridor.classification),
        priority_rank=int(corridor.priority_rank),
        label_a=str(corridor.label_a),
        label_b=str(corridor.label_b),
        md_a_start_m=float(corridor.md_a_start_m),
        md_a_end_m=float(corridor.md_a_end_m),
        md_b_start_m=float(corridor.md_b_start_m),
        md_b_end_m=float(corridor.md_b_end_m),
        min_separation_factor=float(np.min(corridor.separation_factor_values)),
        max_overlap_depth_m=float(np.max(corridor.overlap_depth_values_m)),
        merged_corridor_count=1,
    )


def _report_events_can_merge(
    event: AntiCollisionReportEvent,
    corridor: AntiCollisionCorridor,
    *,
    tolerance_m: float,
) -> bool:
    if (
        str(event.well_a) != str(corridor.well_a)
        or str(event.well_b) != str(corridor.well_b)
        or str(event.classification) != str(corridor.classification)
        or str(event.label_a) != str(corridor.label_a)
        or str(event.label_b) != str(corridor.label_b)
    ):
        return False
    overlap_or_touch_a = (
        float(corridor.md_a_start_m) <= float(event.md_a_end_m) + tolerance_m
    )
    overlap_or_touch_b = (
        float(corridor.md_b_start_m) <= float(event.md_b_end_m) + tolerance_m
    )
    return bool(overlap_or_touch_a and overlap_or_touch_b)


def _merge_report_event_with_corridor(
    event: AntiCollisionReportEvent,
    corridor: AntiCollisionCorridor,
) -> AntiCollisionReportEvent:
    return AntiCollisionReportEvent(
        well_a=event.well_a,
        well_b=event.well_b,
        classification=event.classification,
        priority_rank=int(event.priority_rank),
        label_a=event.label_a,
        label_b=event.label_b,
        md_a_start_m=min(float(event.md_a_start_m), float(corridor.md_a_start_m)),
        md_a_end_m=max(float(event.md_a_end_m), float(corridor.md_a_end_m)),
        md_b_start_m=min(float(event.md_b_start_m), float(corridor.md_b_start_m)),
        md_b_end_m=max(float(event.md_b_end_m), float(corridor.md_b_end_m)),
        min_separation_factor=min(
            float(event.min_separation_factor),
            float(np.min(corridor.separation_factor_values)),
        ),
        max_overlap_depth_m=max(
            float(event.max_overlap_depth_m),
            float(np.max(corridor.overlap_depth_values_m)),
        ),
        merged_corridor_count=int(event.merged_corridor_count) + 1,
    )


def _corridor_merge_tolerance_m(
    wells: tuple[AntiCollisionWell, ...],
) -> float:
    return float(
        max((well.overlay.model.sample_step_m for well in wells), default=100.0) * 1.05
    )


def _md_interval_label(md_start_m: float, md_end_m: float) -> str:
    start = float(md_start_m)
    end = float(md_end_m)
    if abs(end - start) <= 0.5:
        return f"{start:.0f}"
    return f"{start:.0f} - {end:.0f}"


def _corridor_representative_index(corridor: AntiCollisionCorridor) -> int:
    label_a_values = tuple(str(value) for value in corridor.label_a_values)
    label_b_values = tuple(str(value) for value in corridor.label_b_values)
    separation_values = np.asarray(corridor.separation_factor_values, dtype=float)
    ranked_indices = sorted(
        range(len(separation_values)),
        key=lambda index: (
            _classify_pair_labels(
                label_a=label_a_values[index],
                label_b=label_b_values[index],
            )[1],
            float(separation_values[index]),
            -float(np.asarray(corridor.overlap_depth_values_m, dtype=float)[index]),
        ),
    )
    return int(ranked_indices[0])


def _collect_well_overlap_segments(
    corridors: list[AntiCollisionCorridor],
    wells: tuple[AntiCollisionWell, ...],
) -> list[AntiCollisionWellSegment]:
    if not corridors:
        return []
    step_tolerance_m = float(
        max((well.overlay.model.sample_step_m for well in wells), default=100.0) * 1.05
    )
    raw_segments: list[AntiCollisionWellSegment] = []
    for corridor in corridors:
        raw_segments.append(
            AntiCollisionWellSegment(
                well_name=corridor.well_a,
                md_start_m=float(corridor.md_a_start_m),
                md_end_m=float(corridor.md_a_end_m),
                classification=corridor.classification,
                priority_rank=int(corridor.priority_rank),
            )
        )
        raw_segments.append(
            AntiCollisionWellSegment(
                well_name=corridor.well_b,
                md_start_m=float(corridor.md_b_start_m),
                md_end_m=float(corridor.md_b_end_m),
                classification=corridor.classification,
                priority_rank=int(corridor.priority_rank),
            )
        )
    merged: list[AntiCollisionWellSegment] = []
    for well_name in sorted({segment.well_name for segment in raw_segments}):
        well_segments = sorted(
            [segment for segment in raw_segments if segment.well_name == well_name],
            key=lambda segment: (float(segment.md_start_m), float(segment.md_end_m)),
        )
        current = well_segments[0]
        for segment in well_segments[1:]:
            if float(segment.md_start_m) <= float(current.md_end_m) + step_tolerance_m:
                current = AntiCollisionWellSegment(
                    well_name=current.well_name,
                    md_start_m=float(current.md_start_m),
                    md_end_m=max(float(current.md_end_m), float(segment.md_end_m)),
                    classification=(
                        current.classification
                        if int(current.priority_rank) <= int(segment.priority_rank)
                        else segment.classification
                    ),
                    priority_rank=min(
                        int(current.priority_rank), int(segment.priority_rank)
                    ),
                )
                continue
            merged.append(current)
            current = segment
        merged.append(current)
    return merged


def _classify_pair_labels(*, label_a: str, label_b: str) -> tuple[str, int]:
    if label_a and label_b:
        return "target-target", 0
    if label_a or label_b:
        return "target-trajectory", 1
    return "trajectory", 2


def _priority_label_from_parts(*, classification: str) -> str:
    if classification == "target-target":
        return "Цели ↔ цели"
    if classification == "target-trajectory":
        return "Цель ↔ траектория"
    return "Траектория ↔ траектория"


def _zone_location_label_from_parts(*, label_a: str, label_b: str) -> str:
    left = str(label_a) or "траектория"
    right = str(label_b) or "траектория"
    return f"{left} ↔ {right}"


def _circle_polygon_xy(
    *,
    center_x: float,
    center_y: float,
    radius_m: float,
    point_count: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(max(point_count, 12)), endpoint=False)
    polygon = np.column_stack(
        [
            float(center_x) + float(radius_m) * np.cos(angles),
            float(center_y) + float(radius_m) * np.sin(angles),
        ]
    )
    return np.vstack([polygon, polygon[0]])


def _centerline_tangent_xy(centers_xy: np.ndarray, index: int) -> np.ndarray:
    current = np.asarray(centers_xy[int(index)], dtype=float)
    if int(index) == 0:
        return np.asarray(centers_xy[1], dtype=float) - current
    if int(index) == len(centers_xy) - 1:
        return current - np.asarray(centers_xy[-2], dtype=float)
    return np.asarray(centers_xy[int(index) + 1], dtype=float) - np.asarray(
        centers_xy[int(index) - 1], dtype=float
    )


def _centerline_tangent_xyz(centers_xyz: np.ndarray, index: int) -> np.ndarray:
    current = np.asarray(centers_xyz[int(index)], dtype=float)
    if int(index) == 0:
        return np.asarray(centers_xyz[1], dtype=float) - current
    if int(index) == len(centers_xyz) - 1:
        return current - np.asarray(centers_xyz[-2], dtype=float)
    return np.asarray(centers_xyz[int(index) + 1], dtype=float) - np.asarray(
        centers_xyz[int(index) - 1], dtype=float
    )


def _stable_normal_plane_basis(
    tangent_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(tangent_xyz, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        tangent = np.array([0.0, 0.0, 1.0], dtype=float)
        tangent_norm = 1.0
    tangent = tangent / tangent_norm
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(tangent, reference))) > 0.98:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    basis_1 = np.cross(reference, tangent)
    basis_1_norm = float(np.linalg.norm(basis_1))
    if basis_1_norm <= 1e-12:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_1 = np.cross(reference, tangent)
        basis_1_norm = float(np.linalg.norm(basis_1))
    basis_1 = basis_1 / basis_1_norm
    basis_2 = np.cross(tangent, basis_1)
    basis_2 = basis_2 / float(np.linalg.norm(basis_2))
    return basis_1, basis_2


def _align_ring_for_continuity(
    *,
    ring_open_xyz: np.ndarray,
    previous_ring_open_xyz: np.ndarray | None,
) -> np.ndarray:
    current = np.asarray(ring_open_xyz, dtype=float)
    if previous_ring_open_xyz is None:
        return current
    previous = np.asarray(previous_ring_open_xyz, dtype=float)
    if current.shape != previous.shape or current.ndim != 2 or current.shape[0] < 3:
        return current

    best_ring = current
    best_cost = float(np.mean(np.linalg.norm(current - previous, axis=1)))
    for candidate_base in (current, current[::-1]):
        for shift in range(current.shape[0]):
            candidate = np.roll(candidate_base, shift=shift, axis=0)
            candidate_cost = float(
                np.mean(np.linalg.norm(candidate - previous, axis=1))
            )
            if candidate_cost + SMALL < best_cost:
                best_ring = candidate
                best_cost = candidate_cost
    return best_ring
