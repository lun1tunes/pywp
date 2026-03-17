from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.mcm import compute_positions_min_curv
from pywp.models import Point3D
from pywp.planner_types import ProfileParameters
from pywp.planner_validation import _build_trajectory
from pywp.uncertainty import (
    DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    PlanningUncertaintyModel,
    station_uncertainty_covariance_xyz,
)

SMALL = 1e-9


@dataclass(frozen=True)
class AntiCollisionReferencePath:
    well_name: str
    md_start_m: float
    md_end_m: float
    sample_md_m: np.ndarray
    xyz_m: np.ndarray
    covariance_xyz: np.ndarray


@dataclass(frozen=True)
class AntiCollisionOptimizationContext:
    candidate_md_start_m: float
    candidate_md_end_m: float
    sf_target: float
    sample_step_m: float
    uncertainty_model: PlanningUncertaintyModel
    references: tuple[AntiCollisionReferencePath, ...]
    prefer_lower_kop: bool = False


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
    covariance_xyz = np.stack(
        [
            station_uncertainty_covariance_xyz(
                md_m=float(row.MD_m),
                inc_deg=float(row.INC_deg),
                azi_deg=float(row.AZI_deg),
                model=model,
            )
            for row in sampled.itertuples(index=False)
        ],
        axis=0,
    )
    return AntiCollisionReferencePath(
        well_name=str(well_name),
        md_start_m=float(sampled["MD_m"].iloc[0]),
        md_end_m=float(sampled["MD_m"].iloc[-1]),
        sample_md_m=sampled["MD_m"].to_numpy(dtype=float),
        xyz_m=np.asarray(xyz_m, dtype=float),
        covariance_xyz=np.asarray(covariance_xyz, dtype=float),
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
    candidate_md_m = sampled_candidate["MD_m"].to_numpy(dtype=float)
    candidate_xyz = sampled_candidate[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=float)
    candidate_covariance = np.stack(
        [
            station_uncertainty_covariance_xyz(
                md_m=float(row.MD_m),
                inc_deg=float(row.INC_deg),
                azi_deg=float(row.AZI_deg),
                model=context.uncertainty_model,
            )
            for row in sampled_candidate.itertuples(index=False)
        ],
        axis=0,
    )
    confidence_scale = float(max(context.uncertainty_model.confidence_scale, SMALL))
    return _clearance_from_pairwise_samples(
        candidate_md_m=candidate_md_m,
        candidate_xyz=candidate_xyz,
        candidate_covariance=candidate_covariance,
        references=context.references,
        confidence_scale=confidence_scale,
    )


def _clearance_from_pairwise_samples(
    *,
    candidate_md_m: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_covariance: np.ndarray,
    references: tuple[AntiCollisionReferencePath, ...],
    confidence_scale: float,
) -> AntiCollisionClearanceEvaluation:
    min_sf = float("inf")
    max_overlap = 0.0
    for reference in references:
        sample_count = max(len(candidate_md_m), len(reference.sample_md_m))
        progress_grid = np.linspace(0.0, 1.0, num=max(sample_count, 1), dtype=float)
        candidate_progress = _progress_parameter(candidate_md_m)
        reference_progress = _progress_parameter(reference.sample_md_m)
        candidate_xyz_aligned = _interp_rows_by_progress(
            values=candidate_xyz,
            source_progress=candidate_progress,
            target_progress=progress_grid,
        )
        candidate_covariance_aligned = _interp_covariance_by_progress(
            values=candidate_covariance,
            source_progress=candidate_progress,
            target_progress=progress_grid,
        )
        reference_xyz_aligned = _interp_rows_by_progress(
            values=reference.xyz_m,
            source_progress=reference_progress,
            target_progress=progress_grid,
        )
        reference_covariance_aligned = _interp_covariance_by_progress(
            values=reference.covariance_xyz,
            source_progress=reference_progress,
            target_progress=progress_grid,
        )
        delta_xyz = candidate_xyz_aligned - reference_xyz_aligned
        distance = np.linalg.norm(delta_xyz, axis=1)
        direction = np.zeros((len(progress_grid), 3), dtype=float)
        np.divide(
            delta_xyz,
            distance[:, None],
            out=direction,
            where=distance[:, None] > SMALL,
        )
        zero_mask = distance <= SMALL
        if np.any(zero_mask):
            direction[zero_mask] = np.array([1.0, 0.0, 0.0], dtype=float)
        combined_sigma2 = np.einsum(
            "ai,aij,aj->a",
            direction,
            candidate_covariance_aligned,
            direction,
        ) + np.einsum(
            "ai,aij,aj->a",
            direction,
            reference_covariance_aligned,
            direction,
        )
        combined_radius = confidence_scale * np.sqrt(np.clip(combined_sigma2, 0.0, None))
        sf = np.divide(
            distance,
            np.maximum(combined_radius, SMALL),
            out=np.full_like(distance, np.inf, dtype=float),
            where=combined_radius > SMALL,
        )
        overlap_depth = np.maximum(combined_radius - distance, 0.0)
        min_sf = min(min_sf, float(np.min(sf)))
        max_overlap = max(max_overlap, float(np.max(overlap_depth)))
    if not np.isfinite(min_sf):
        min_sf = 1e6
    return AntiCollisionClearanceEvaluation(
        min_separation_factor=float(min_sf),
        max_overlap_depth_m=float(max_overlap),
    )


def _progress_parameter(sample_md_m: np.ndarray) -> np.ndarray:
    md_values = np.asarray(sample_md_m, dtype=float)
    if len(md_values) <= 1:
        return np.zeros(len(md_values), dtype=float)
    start = float(md_values[0])
    end = float(md_values[-1])
    span = max(end - start, SMALL)
    return np.clip((md_values - start) / span, 0.0, 1.0)


def _interp_rows_by_progress(
    *,
    values: np.ndarray,
    source_progress: np.ndarray,
    target_progress: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if len(source) == 0:
        return np.zeros((len(target_progress), 3), dtype=float)
    if len(source) == 1:
        return np.repeat(source, repeats=len(target_progress), axis=0)
    return np.column_stack(
        [
            np.interp(
                target_progress,
                source_progress,
                source[:, column_index],
            )
            for column_index in range(source.shape[1])
        ]
    )


def _interp_covariance_by_progress(
    *,
    values: np.ndarray,
    source_progress: np.ndarray,
    target_progress: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if len(source) == 0:
        return np.zeros((len(target_progress), 3, 3), dtype=float)
    if len(source) == 1:
        return np.repeat(source, repeats=len(target_progress), axis=0)
    result = np.empty((len(target_progress), source.shape[1], source.shape[2]), dtype=float)
    for row_index in range(source.shape[1]):
        for column_index in range(source.shape[2]):
            result[:, row_index, column_index] = np.interp(
                target_progress,
                source_progress,
                source[:, row_index, column_index],
            )
    return result


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
) -> pd.DataFrame:
    trajectory = _build_trajectory(params=candidate)
    stations = compute_positions_min_curv(
        trajectory.stations(md_step_m=float(sample_step_m)),
        start=surface,
    )
    return sample_stations_in_md_window(
        stations=stations,
        md_start_m=md_start_m,
        md_end_m=md_end_m,
        sample_step_m=sample_step_m,
    )


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
