from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.mcm import minimum_curvature_increment
from pywp.models import INTERPOLATION_RODRIGUES, Point3D
from pywp.planner_types import ProfileParameters
from pywp.planner_validation import _build_trajectory
from pywp.constants import SMALL
from pywp.uncertainty import (
    DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    PlanningUncertaintyModel,
    station_uncertainty_covariance_xyz_many,
)
AntiCollisionSegment = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]


@dataclass(frozen=True)
class _SampledState:
    md_m: float
    x_m: float
    y_m: float
    z_m: float
    inc_deg: float
    azi_deg: float


@dataclass(frozen=True)
class AntiCollisionReferencePath:
    well_name: str
    md_start_m: float
    md_end_m: float
    sample_md_m: np.ndarray
    xyz_m: np.ndarray
    covariance_xyz: np.ndarray
    segments: tuple[AntiCollisionSegment, ...]


@dataclass(frozen=True)
class AntiCollisionOptimizationContext:
    candidate_md_start_m: float
    candidate_md_end_m: float
    sf_target: float
    sample_step_m: float
    uncertainty_model: PlanningUncertaintyModel
    references: tuple[AntiCollisionReferencePath, ...]
    prefer_lower_kop: bool = False
    prefer_higher_build1: bool = False
    prefer_keep_kop: bool = False
    prefer_keep_build1: bool = False
    prefer_adjust_build2: bool = False
    baseline_kop_vertical_m: float | None = None
    baseline_build1_dls_deg_per_30m: float | None = None
    interpolation_method: str = INTERPOLATION_RODRIGUES


@dataclass(frozen=True)
class AntiCollisionClearanceEvaluation:
    min_separation_factor: float
    max_overlap_depth_m: float

    @property
    def sf_margin(self) -> float:
        return float(self.min_separation_factor)


def build_anti_collision_reference_path(
    *,
    well_name: str,
    stations: pd.DataFrame,
    md_start_m: float,
    md_end_m: float,
    sample_step_m: float,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> AntiCollisionReferencePath:
    sampled = sample_stations_in_md_window(
        stations=stations,
        md_start_m=md_start_m,
        md_end_m=md_end_m,
        sample_step_m=sample_step_m,
    )
    xyz_m = sampled[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=float)
    covariance_xyz = station_uncertainty_covariance_xyz_many(
        md_m=sampled["MD_m"].to_numpy(dtype=float),
        inc_deg=sampled["INC_deg"].to_numpy(dtype=float),
        azi_deg=sampled["AZI_deg"].to_numpy(dtype=float),
        model=model,
    )
    return AntiCollisionReferencePath(
        well_name=str(well_name),
        md_start_m=float(sampled["MD_m"].iloc[0]),
        md_end_m=float(sampled["MD_m"].iloc[-1]),
        sample_md_m=sampled["MD_m"].to_numpy(dtype=float),
        xyz_m=np.asarray(xyz_m, dtype=float),
        covariance_xyz=np.asarray(covariance_xyz, dtype=float),
        segments=tuple(
            _polyline_segments(
                xyz=np.asarray(xyz_m, dtype=float),
                covariance=np.asarray(covariance_xyz, dtype=float),
            )
        ),
    )


def evaluate_candidate_anti_collision_clearance(
    *,
    candidate: ProfileParameters,
    surface: Point3D,
    context: AntiCollisionOptimizationContext,
) -> AntiCollisionClearanceEvaluation:
    sampled_candidate = sample_profile_stations_in_md_window(
        candidate=candidate,
        surface=surface,
        md_start_m=float(context.candidate_md_start_m),
        md_end_m=float(context.candidate_md_end_m),
        sample_step_m=float(context.sample_step_m),
        interpolation_method=str(context.interpolation_method),
    )
    return evaluate_stations_anti_collision_clearance(
        stations=sampled_candidate,
        context=context,
    )


def evaluate_stations_anti_collision_clearance(
    *,
    stations: pd.DataFrame,
    context: AntiCollisionOptimizationContext,
) -> AntiCollisionClearanceEvaluation:
    sampled_candidate = sample_stations_in_md_window(
        stations=stations,
        md_start_m=float(context.candidate_md_start_m),
        md_end_m=float(context.candidate_md_end_m),
        sample_step_m=float(context.sample_step_m),
    )
    candidate_xyz = sampled_candidate[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=float)
    candidate_covariance = station_uncertainty_covariance_xyz_many(
        md_m=sampled_candidate["MD_m"].to_numpy(dtype=float),
        inc_deg=sampled_candidate["INC_deg"].to_numpy(dtype=float),
        azi_deg=sampled_candidate["AZI_deg"].to_numpy(dtype=float),
        model=context.uncertainty_model,
    )
    confidence_scale = float(max(context.uncertainty_model.confidence_scale, SMALL))
    return _clearance_from_continuous_segment_pairs(
        candidate_xyz=candidate_xyz,
        candidate_covariance=candidate_covariance,
        references=context.references,
        confidence_scale=confidence_scale,
    )


def _clearance_from_continuous_segment_pairs(
    *,
    candidate_xyz: np.ndarray,
    candidate_covariance: np.ndarray,
    references: tuple[AntiCollisionReferencePath, ...],
    confidence_scale: float,
) -> AntiCollisionClearanceEvaluation:
    min_sf = float("inf")
    max_overlap = 0.0
    candidate_segments = _polyline_segments(
        xyz=candidate_xyz,
        covariance=candidate_covariance,
    )
    for reference in references:
        pair_evaluation = _continuous_polyline_clearance(
            candidate_segments=candidate_segments,
            reference_segments=list(reference.segments),
            confidence_scale=confidence_scale,
        )
        min_sf = min(min_sf, float(pair_evaluation.min_separation_factor))
        max_overlap = max(max_overlap, float(pair_evaluation.max_overlap_depth_m))
    if not np.isfinite(min_sf):
        min_sf = 1e6
    return AntiCollisionClearanceEvaluation(
        min_separation_factor=float(min_sf),
        max_overlap_depth_m=float(max_overlap),
    )


def _polyline_segments(
    *,
    xyz: np.ndarray,
    covariance: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    points = np.asarray(xyz, dtype=float)
    covariances = np.asarray(covariance, dtype=float)
    if len(points) == 0:
        return []
    if len(points) == 1:
        sigma2_upper = float(np.trace(covariances[0]))
        return [(points[0], points[0], covariances[0], covariances[0], sigma2_upper)]
    return [
        (
            points[index],
            points[index + 1],
            covariances[index],
            covariances[index + 1],
            float(
                max(
                    float(np.trace(covariances[index])),
                    float(np.trace(covariances[index + 1])),
                )
            ),
        )
        for index in range(len(points) - 1)
    ]


def _continuous_polyline_clearance(
    *,
    candidate_segments: list[AntiCollisionSegment],
    reference_segments: list[AntiCollisionSegment],
    confidence_scale: float,
) -> AntiCollisionClearanceEvaluation:
    min_sf = float("inf")
    max_overlap = 0.0
    for candidate_segment in candidate_segments:
        for reference_segment in reference_segments:
            (
                closest_candidate,
                closest_reference,
                centerline_distance_m,
            ) = _segment_pair_closest_probe(
                candidate_segment=candidate_segment,
                reference_segment=reference_segment,
            )
            combined_radius_upper_m = _segment_pair_combined_radius_upper_bound(
                candidate_segment=candidate_segment,
                reference_segment=reference_segment,
                confidence_scale=confidence_scale,
            )
            sf_lower_bound = float(
                centerline_distance_m / max(combined_radius_upper_m, SMALL)
            )
            overlap_upper_bound = float(
                max(combined_radius_upper_m - centerline_distance_m, 0.0)
            )
            if (
                sf_lower_bound >= float(min_sf)
                and overlap_upper_bound <= float(max_overlap) + SMALL
            ):
                continue
            for candidate_param, reference_param in _segment_pair_probe_parameters(
                candidate_segment=candidate_segment,
                reference_segment=reference_segment,
                closest_candidate=closest_candidate,
                closest_reference=closest_reference,
            ):
                separation_factor, overlap_depth_m = _evaluate_segment_pair_probe(
                    candidate_segment=candidate_segment,
                    reference_segment=reference_segment,
                    candidate_param=float(candidate_param),
                    reference_param=float(reference_param),
                    confidence_scale=confidence_scale,
                )
                min_sf = min(min_sf, float(separation_factor))
                max_overlap = max(max_overlap, float(overlap_depth_m))
    if not np.isfinite(min_sf):
        min_sf = 1e6
    return AntiCollisionClearanceEvaluation(
        min_separation_factor=float(min_sf),
        max_overlap_depth_m=float(max_overlap),
    )


def _segment_pair_probe_parameters(
    *,
    candidate_segment: AntiCollisionSegment,
    reference_segment: AntiCollisionSegment,
    closest_candidate: float,
    closest_reference: float,
) -> tuple[tuple[float, float], ...]:
    candidate_start, candidate_end, _, _, _ = candidate_segment
    reference_start, reference_end, _, _, _ = reference_segment
    probe_pairs = [
        (closest_candidate, closest_reference),
        (
            0.0,
            _closest_parameter_on_segment_to_point(
                point=candidate_start,
                segment_start=reference_start,
                segment_end=reference_end,
            ),
        ),
        (
            1.0,
            _closest_parameter_on_segment_to_point(
                point=candidate_end,
                segment_start=reference_start,
                segment_end=reference_end,
            ),
        ),
        (
            _closest_parameter_on_segment_to_point(
                point=reference_start,
                segment_start=candidate_start,
                segment_end=candidate_end,
            ),
            0.0,
        ),
        (
            _closest_parameter_on_segment_to_point(
                point=reference_end,
                segment_start=candidate_start,
                segment_end=candidate_end,
            ),
            1.0,
        ),
    ]
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for candidate_param, reference_param in probe_pairs:
        key = (
            round(float(np.clip(candidate_param, 0.0, 1.0)), 8),
            round(float(np.clip(reference_param, 0.0, 1.0)), 8),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return tuple(deduped)


def _evaluate_segment_pair_probe(
    *,
    candidate_segment: AntiCollisionSegment,
    reference_segment: AntiCollisionSegment,
    candidate_param: float,
    reference_param: float,
    confidence_scale: float,
) -> tuple[float, float]:
    candidate_start, candidate_end, candidate_cov_start, candidate_cov_end, _ = candidate_segment
    reference_start, reference_end, reference_cov_start, reference_cov_end, _ = reference_segment
    candidate_point = _interpolate_segment_point(
        start=candidate_start,
        end=candidate_end,
        parameter=candidate_param,
    )
    reference_point = _interpolate_segment_point(
        start=reference_start,
        end=reference_end,
        parameter=reference_param,
    )
    candidate_covariance = _interpolate_segment_covariance(
        start_covariance=candidate_cov_start,
        end_covariance=candidate_cov_end,
        parameter=candidate_param,
    )
    reference_covariance = _interpolate_segment_covariance(
        start_covariance=reference_cov_start,
        end_covariance=reference_cov_end,
        parameter=reference_param,
    )
    delta_xyz = candidate_point - reference_point
    distance_m = float(np.linalg.norm(delta_xyz))
    combined_covariance = candidate_covariance + reference_covariance
    if distance_m <= SMALL:
        max_sigma2 = float(np.max(np.linalg.eigvalsh(combined_covariance)))
        combined_radius_m = float(
            confidence_scale * np.sqrt(max(max_sigma2, 0.0))
        )
        return 0.0, float(max(combined_radius_m, 0.0))
    direction = delta_xyz / distance_m
    combined_sigma2 = float(direction @ combined_covariance @ direction)
    combined_radius_m = float(
        confidence_scale * np.sqrt(max(combined_sigma2, 0.0))
    )
    separation_factor = float(distance_m / max(combined_radius_m, SMALL))
    overlap_depth_m = float(max(combined_radius_m - distance_m, 0.0))
    return separation_factor, overlap_depth_m


def _segment_pair_closest_probe(
    *,
    candidate_segment: AntiCollisionSegment,
    reference_segment: AntiCollisionSegment,
) -> tuple[float, float, float]:
    candidate_start, candidate_end, _, _, _ = candidate_segment
    reference_start, reference_end, _, _, _ = reference_segment
    closest_candidate, closest_reference = _closest_parameters_on_segments(
        candidate_start,
        candidate_end,
        reference_start,
        reference_end,
    )
    candidate_point = _interpolate_segment_point(
        start=candidate_start,
        end=candidate_end,
        parameter=closest_candidate,
    )
    reference_point = _interpolate_segment_point(
        start=reference_start,
        end=reference_end,
        parameter=closest_reference,
    )
    distance_m = float(np.linalg.norm(candidate_point - reference_point))
    return float(closest_candidate), float(closest_reference), float(distance_m)


def _segment_pair_combined_radius_upper_bound(
    *,
    candidate_segment: AntiCollisionSegment,
    reference_segment: AntiCollisionSegment,
    confidence_scale: float,
) -> float:
    candidate_sigma2_upper = float(candidate_segment[4])
    reference_sigma2_upper = float(reference_segment[4])
    return float(
        confidence_scale
        * np.sqrt(max(candidate_sigma2_upper + reference_sigma2_upper, 0.0))
    )


def _closest_parameter_on_segment_to_point(
    *,
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    direction = np.asarray(segment_end, dtype=float) - np.asarray(segment_start, dtype=float)
    denom = float(direction @ direction)
    if denom <= SMALL:
        return 0.0
    parameter = float((np.asarray(point, dtype=float) - np.asarray(segment_start, dtype=float)) @ direction / denom)
    return float(np.clip(parameter, 0.0, 1.0))


def _closest_parameters_on_segments(
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
    reference_start: np.ndarray,
    reference_end: np.ndarray,
) -> tuple[float, float]:
    p0 = np.asarray(candidate_start, dtype=float)
    p1 = np.asarray(candidate_end, dtype=float)
    q0 = np.asarray(reference_start, dtype=float)
    q1 = np.asarray(reference_end, dtype=float)
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(u @ u)
    b = float(u @ v)
    c = float(v @ v)
    d = float(u @ w)
    e = float(v @ w)
    if a <= SMALL and c <= SMALL:
        return 0.0, 0.0
    if a <= SMALL:
        return 0.0, _closest_parameter_on_segment_to_point(
            point=p0,
            segment_start=q0,
            segment_end=q1,
        )
    if c <= SMALL:
        return _closest_parameter_on_segment_to_point(
            point=q0,
            segment_start=p0,
            segment_end=p1,
        ), 0.0

    denominator = a * c - b * b
    s_numerator = 0.0
    s_denominator = denominator
    t_numerator = 0.0
    t_denominator = denominator

    if denominator <= SMALL:
        s_numerator = 0.0
        s_denominator = 1.0
        t_numerator = e
        t_denominator = c
    else:
        s_numerator = b * e - c * d
        t_numerator = a * e - b * d
        if s_numerator < 0.0:
            s_numerator = 0.0
            t_numerator = e
            t_denominator = c
        elif s_numerator > s_denominator:
            s_numerator = s_denominator
            t_numerator = e + b
            t_denominator = c

    if t_numerator < 0.0:
        t_numerator = 0.0
        if -d < 0.0:
            s_numerator = 0.0
        elif -d > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d
            s_denominator = a
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if -d + b < 0.0:
            s_numerator = 0.0
        elif -d + b > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d + b
            s_denominator = a

    candidate_param = 0.0 if abs(s_numerator) <= SMALL else s_numerator / max(s_denominator, SMALL)
    reference_param = 0.0 if abs(t_numerator) <= SMALL else t_numerator / max(t_denominator, SMALL)
    return float(np.clip(candidate_param, 0.0, 1.0)), float(np.clip(reference_param, 0.0, 1.0))


def _interpolate_segment_point(
    *,
    start: np.ndarray,
    end: np.ndarray,
    parameter: float,
) -> np.ndarray:
    alpha = float(np.clip(parameter, 0.0, 1.0))
    return (1.0 - alpha) * np.asarray(start, dtype=float) + alpha * np.asarray(end, dtype=float)


def _interpolate_segment_covariance(
    *,
    start_covariance: np.ndarray,
    end_covariance: np.ndarray,
    parameter: float,
) -> np.ndarray:
    alpha = float(np.clip(parameter, 0.0, 1.0))
    return (
        (1.0 - alpha) * np.asarray(start_covariance, dtype=float)
        + alpha * np.asarray(end_covariance, dtype=float)
    )


def sample_stations_in_md_window(
    *,
    stations: pd.DataFrame,
    md_start_m: float,
    md_end_m: float,
    sample_step_m: float,
) -> pd.DataFrame:
    if len(stations) == 0:
        raise ValueError("stations are empty")
    required_columns = {"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"}
    missing = sorted(required_columns.difference(stations.columns))
    if missing:
        raise ValueError("stations are missing columns: " + ", ".join(missing))
    md_values = stations["MD_m"].to_numpy(dtype=float)
    start_md = float(np.clip(md_start_m, float(md_values[0]), float(md_values[-1])))
    end_md = float(np.clip(md_end_m, float(md_values[0]), float(md_values[-1])))
    if end_md <= start_md + 1e-6:
        end_md = start_md
    grid = _sample_md_grid(
        md_start_m=start_md,
        md_end_m=end_md,
        sample_step_m=sample_step_m,
    )
    return pd.DataFrame(
        {
            "MD_m": grid,
            "INC_deg": np.interp(grid, md_values, stations["INC_deg"].to_numpy(dtype=float)),
            "AZI_deg": _interp_azimuth_deg(
                grid=grid,
                md_values=md_values,
                azimuth_deg=stations["AZI_deg"].to_numpy(dtype=float),
            ),
            "X_m": np.interp(grid, md_values, stations["X_m"].to_numpy(dtype=float)),
            "Y_m": np.interp(grid, md_values, stations["Y_m"].to_numpy(dtype=float)),
            "Z_m": np.interp(grid, md_values, stations["Z_m"].to_numpy(dtype=float)),
        }
    )


def _interp_azimuth_deg(
    *,
    grid: np.ndarray,
    md_values: np.ndarray,
    azimuth_deg: np.ndarray,
) -> np.ndarray:
    source = np.asarray(azimuth_deg, dtype=float)
    if len(source) <= 1:
        return np.full(len(grid), float(source[0] % 360.0 if len(source) else 0.0), dtype=float)
    unwrapped_rad = np.unwrap(np.radians(source))
    interpolated_rad = np.interp(
        np.asarray(grid, dtype=float),
        np.asarray(md_values, dtype=float),
        unwrapped_rad,
    )
    return np.mod(np.degrees(interpolated_rad), 360.0)


def sample_profile_stations_in_md_window(
    *,
    candidate: ProfileParameters,
    surface: Point3D,
    md_start_m: float,
    md_end_m: float,
    sample_step_m: float,
    interpolation_method: str = INTERPOLATION_RODRIGUES,
) -> pd.DataFrame:
    start_md = float(max(md_start_m, 0.0))
    end_md = float(max(md_end_m, start_md))
    sample_step = float(max(sample_step_m, 1.0))
    segments = tuple(_profile_segment_specs(candidate=candidate, interpolation_method=interpolation_method))
    grid = _sample_md_grid(
        md_start_m=start_md,
        md_end_m=end_md,
        sample_step_m=sample_step,
    )
    surface_state = _SampledState(
        md_m=0.0,
        x_m=float(surface.x),
        y_m=float(surface.y),
        z_m=float(surface.z),
        inc_deg=0.0,
        azi_deg=float(candidate.azimuth_hold_deg % 360.0),
    )
    rows: list[dict[str, float | str]] = []
    current_state = surface_state
    segment_index = 0
    while (
        segment_index + 1 < len(segments)
        and float(segments[segment_index]["md_end_m"]) < start_md - 1e-6
    ):
        segment = segments[segment_index]
        current_state = _advance_sampled_state_along_segment(
            state=current_state,
            segment=segment,
            delta_start_m=0.0,
            delta_end_m=float(segment["length_m"]),
        )
        segment_index += 1

    for md_value in grid:
        while (
            segment_index + 1 < len(segments)
            and md_value > float(segments[segment_index]["md_end_m"]) + 1e-6
        ):
            segment = segments[segment_index]
            local_current_m = float(
                np.clip(current_state.md_m - float(segment["md_start_m"]), 0.0, float(segment["length_m"]))
            )
            current_state = _advance_sampled_state_along_segment(
                state=current_state,
                segment=segment,
                delta_start_m=local_current_m,
                delta_end_m=float(segment["length_m"]),
            )
            segment_index += 1

        segment = segments[min(segment_index, len(segments) - 1)]
        segment_start_md = float(segment["md_start_m"])
        local_current_m = float(
            np.clip(current_state.md_m - segment_start_md, 0.0, float(segment["length_m"]))
        )
        local_target_m = float(
            np.clip(md_value - segment_start_md, 0.0, float(segment["length_m"]))
        )
        current_state = _advance_sampled_state_along_segment(
            state=current_state,
            segment=segment,
            delta_start_m=local_current_m,
            delta_end_m=local_target_m,
        )
        rows.append(
            {
                "MD_m": float(current_state.md_m),
                "INC_deg": float(current_state.inc_deg),
                "AZI_deg": float(current_state.azi_deg),
                "X_m": float(current_state.x_m),
                "Y_m": float(current_state.y_m),
                "Z_m": float(current_state.z_m),
                "segment": str(segment["name"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            {
                "MD_m": [float(start_md)],
                "INC_deg": [float(surface_state.inc_deg)],
                "AZI_deg": [float(surface_state.azi_deg)],
                "X_m": [float(surface_state.x_m)],
                "Y_m": [float(surface_state.y_m)],
                "Z_m": [float(surface_state.z_m)],
                "segment": ["VERTICAL"],
            }
        )
    sampled = pd.DataFrame(rows)
    sampled = (
        sampled.drop_duplicates(subset=["MD_m"], keep="last")
        .sort_values("MD_m", kind="stable")
        .reset_index(drop=True)
    )
    return sampled


def _profile_segment_specs(
    *,
    candidate: ProfileParameters,
    interpolation_method: str = INTERPOLATION_RODRIGUES,
) -> list[dict[str, float | str]]:
    trajectory = _build_trajectory(params=candidate, interpolation_method=interpolation_method)
    specs: list[dict[str, float | str]] = []
    md_start = 0.0
    for segment in trajectory.segments:
        length_m = float(max(getattr(segment, "length_m", 0.0), 0.0))
        inc_from_deg = float(getattr(segment, "inc_from_deg", getattr(segment, "inc_deg", 0.0)))
        inc_to_deg = float(getattr(segment, "inc_to_deg", getattr(segment, "inc_deg", inc_from_deg)))
        azi_from_deg = float(getattr(segment, "azi_from_deg", getattr(segment, "azi_deg", 0.0)))
        azi_to_deg = float(getattr(segment, "azi_to_deg", getattr(segment, "azi_deg", azi_from_deg)))
        specs.append(
            {
                "name": str(getattr(segment, "name", "SEGMENT")),
                "md_start_m": float(md_start),
                "md_end_m": float(md_start + length_m),
                "length_m": float(length_m),
                "inc_from_deg": float(inc_from_deg),
                "inc_to_deg": float(inc_to_deg),
                "azi_from_deg": float(azi_from_deg % 360.0),
                "azi_to_deg": float(azi_to_deg % 360.0),
            }
        )
        md_start += length_m
    return specs


def _advance_sampled_state_along_segment(
    *,
    state: _SampledState,
    segment: dict[str, float | str],
    delta_start_m: float,
    delta_end_m: float,
) -> _SampledState:
    start_local_m = float(max(delta_start_m, 0.0))
    end_local_m = float(max(delta_end_m, start_local_m))
    if end_local_m <= start_local_m + 1e-9:
        return _SampledState(
            md_m=float(state.md_m),
            x_m=float(state.x_m),
            y_m=float(state.y_m),
            z_m=float(state.z_m),
            inc_deg=float(state.inc_deg),
            azi_deg=float(state.azi_deg),
        )
    length_m = float(end_local_m - start_local_m)
    start_inc_deg, start_azi_deg = _segment_orientation_at(
        segment=segment,
        local_md_m=start_local_m,
    )
    end_inc_deg, end_azi_deg = _segment_orientation_at(
        segment=segment,
        local_md_m=end_local_m,
    )
    dn_m, de_m, dz_m = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=float(start_inc_deg),
        azi1_deg=float(start_azi_deg),
        md2_m=float(length_m),
        inc2_deg=float(end_inc_deg),
        azi2_deg=float(end_azi_deg),
    )
    return _SampledState(
        md_m=float(state.md_m + length_m),
        x_m=float(state.x_m + de_m),
        y_m=float(state.y_m + dn_m),
        z_m=float(state.z_m + dz_m),
        inc_deg=float(end_inc_deg),
        azi_deg=float(end_azi_deg % 360.0),
    )


def _segment_orientation_at(
    *,
    segment: dict[str, float | str],
    local_md_m: float,
) -> tuple[float, float]:
    length_m = float(max(segment["length_m"], 0.0))
    inc_from_deg = float(segment["inc_from_deg"])
    inc_to_deg = float(segment["inc_to_deg"])
    azi_from_deg = float(segment["azi_from_deg"])
    azi_to_deg = float(segment["azi_to_deg"])
    if length_m <= 1e-9:
        return float(inc_to_deg), float(azi_to_deg % 360.0)
    alpha = float(np.clip(local_md_m / length_m, 0.0, 1.0))
    start_direction = _direction_vector(
        inc_deg=inc_from_deg,
        azi_deg=azi_from_deg,
    )
    end_direction = _direction_vector(
        inc_deg=inc_to_deg,
        azi_deg=azi_to_deg,
    )
    direction = _slerp_single_direction(
        direction_from=start_direction,
        direction_to=end_direction,
        t=alpha,
    )
    horizontal = float(np.hypot(direction[0], direction[1]))
    inc_deg = float(np.degrees(np.arctan2(horizontal, direction[2])))
    azi_deg = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0)
    return inc_deg, azi_deg


def _direction_vector(
    *,
    inc_deg: float,
    azi_deg: float,
) -> np.ndarray:
    inc_rad = np.radians(float(inc_deg))
    azi_rad = np.radians(float(azi_deg))
    return np.array(
        [
            np.sin(inc_rad) * np.cos(azi_rad),
            np.sin(inc_rad) * np.sin(azi_rad),
            np.cos(inc_rad),
        ],
        dtype=float,
    )


def _slerp_single_direction(
    *,
    direction_from: np.ndarray,
    direction_to: np.ndarray,
    t: float,
) -> np.ndarray:
    start = np.asarray(direction_from, dtype=float)
    end = np.asarray(direction_to, dtype=float)
    start = start / max(float(np.linalg.norm(start)), SMALL)
    end = end / max(float(np.linalg.norm(end)), SMALL)
    dot = float(np.clip(float(np.dot(start, end)), -1.0, 1.0))
    theta = float(np.arccos(dot))
    alpha = float(np.clip(t, 0.0, 1.0))
    if theta <= 1e-12:
        direction = start
    else:
        sin_theta = float(np.sin(theta))
        w0 = float(np.sin((1.0 - alpha) * theta) / sin_theta)
        w1 = float(np.sin(alpha * theta) / sin_theta)
        direction = w0 * start + w1 * end
    norm = max(float(np.linalg.norm(direction)), SMALL)
    return np.asarray(direction / norm, dtype=float)


def _sample_md_grid(
    *,
    md_start_m: float,
    md_end_m: float,
    sample_step_m: float,
) -> np.ndarray:
    start = float(md_start_m)
    end = float(md_end_m)
    step = float(max(sample_step_m, 1.0))
    if end <= start + 1e-6:
        return np.array([start], dtype=float)
    values = [start]
    current = start + step
    while current < end - 1e-6:
        values.append(float(current))
        current += step
    values.append(end)
    return np.asarray(sorted(set(round(float(value), 6) for value in values)), dtype=float)
