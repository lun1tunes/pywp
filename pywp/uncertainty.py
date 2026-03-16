from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.models import Point3D

DEG2RAD = np.pi / 180.0
_REQUIRED_STATION_COLUMNS = frozenset({"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"})


@dataclass(frozen=True)
class PlanningUncertaintyModel:
    """Planning-level first-order uncertainty model.

    This is intentionally not a full ISCWSA tool-error model. It uses a
    first-order local-angle approximation: position uncertainty in the plane
    normal to the borehole grows with measured depth and assumed 1-sigma
    inclination/azimuth errors.
    """

    sigma_inc_deg: float = 0.15
    sigma_azi_deg: float = 0.15
    confidence_scale: float = 2.0
    sample_step_m: float = 250.0
    max_display_ellipses: int = 14
    ellipse_points: int = 36
    min_display_radius_m: float = 0.75

    def __post_init__(self) -> None:
        if float(self.sigma_inc_deg) <= 0.0:
            raise ValueError("sigma_inc_deg must be positive.")
        if float(self.sigma_azi_deg) <= 0.0:
            raise ValueError("sigma_azi_deg must be positive.")
        if float(self.confidence_scale) <= 0.0:
            raise ValueError("confidence_scale must be positive.")
        if float(self.sample_step_m) <= 0.0:
            raise ValueError("sample_step_m must be positive.")
        if int(self.max_display_ellipses) <= 0:
            raise ValueError("max_display_ellipses must be positive.")
        if int(self.ellipse_points) < 12:
            raise ValueError("ellipse_points must be >= 12.")
        if float(self.min_display_radius_m) < 0.0:
            raise ValueError("min_display_radius_m cannot be negative.")


DEFAULT_PLANNING_UNCERTAINTY_MODEL = PlanningUncertaintyModel()


@dataclass(frozen=True)
class UncertaintyEllipseSample:
    station_index: int
    md_m: float
    center_xyz: tuple[float, float, float]
    ring_xyz: np.ndarray
    ring_plan_xy: np.ndarray
    ring_section_xz: np.ndarray
    semi_axis_inc_m: float
    semi_axis_azi_m: float


@dataclass(frozen=True)
class WellUncertaintyOverlay:
    samples: tuple[UncertaintyEllipseSample, ...]
    model: PlanningUncertaintyModel


def uncertainty_model_caption(
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> str:
    return (
        "Показаны 2σ planning-level эллипсы неопределенности в плоскости, "
        "нормальной к стволу. Базовая модель: first-order по ошибкам "
        f"INC/AZI = {float(model.sigma_inc_deg):.2f}°/{float(model.sigma_azi_deg):.2f}°. "
        "Это визуализация для планирования, не полноценная ISCWSA tool model."
    )


def tangent_vector_xyz(inc_deg: float, azi_deg: float) -> np.ndarray:
    inc_rad = float(inc_deg) * DEG2RAD
    azi_rad = float(azi_deg) * DEG2RAD
    return np.array(
        [
            np.sin(inc_rad) * np.sin(azi_rad),
            np.sin(inc_rad) * np.cos(azi_rad),
            np.cos(inc_rad),
        ],
        dtype=float,
    )


def local_uncertainty_axes_xyz(
    inc_deg: float, azi_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inc_rad = float(inc_deg) * DEG2RAD
    azi_rad = float(azi_deg) * DEG2RAD
    tangent = tangent_vector_xyz(inc_deg=inc_deg, azi_deg=azi_deg)
    inc_axis = np.array(
        [
            np.cos(inc_rad) * np.sin(azi_rad),
            np.cos(inc_rad) * np.cos(azi_rad),
            -np.sin(inc_rad),
        ],
        dtype=float,
    )
    azi_axis = np.array(
        [
            np.cos(azi_rad),
            -np.sin(azi_rad),
            0.0,
        ],
        dtype=float,
    )
    return tangent, inc_axis, azi_axis


def station_uncertainty_axes_m(
    *,
    md_m: float,
    inc_deg: float,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> tuple[float, float]:
    md_value = max(float(md_m), 0.0)
    inc_rad = float(inc_deg) * DEG2RAD
    sigma_inc_m = md_value * float(model.sigma_inc_deg) * DEG2RAD
    sigma_azi_m = md_value * abs(np.sin(inc_rad)) * float(model.sigma_azi_deg) * DEG2RAD
    return (
        float(model.confidence_scale) * sigma_inc_m,
        float(model.confidence_scale) * sigma_azi_m,
    )


def station_uncertainty_covariance_xyz(
    *,
    md_m: float,
    inc_deg: float,
    azi_deg: float,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> np.ndarray:
    _, inc_axis, azi_axis = local_uncertainty_axes_xyz(inc_deg=inc_deg, azi_deg=azi_deg)
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=md_m,
        inc_deg=inc_deg,
        model=model,
    )
    scale = float(model.confidence_scale)
    sigma_inc_m = semi_inc_m / scale
    sigma_azi_m = semi_azi_m / scale
    return (
        sigma_inc_m * sigma_inc_m * np.outer(inc_axis, inc_axis)
        + sigma_azi_m * sigma_azi_m * np.outer(azi_axis, azi_axis)
    )


def build_uncertainty_overlay(
    *,
    stations: pd.DataFrame,
    surface: Point3D,
    azimuth_deg: float,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> WellUncertaintyOverlay:
    missing_cols = sorted(_REQUIRED_STATION_COLUMNS.difference(stations.columns))
    if missing_cols:
        raise ValueError(
            "stations are missing required columns for uncertainty overlay: "
            + ", ".join(missing_cols)
        )
    if len(stations) == 0:
        return WellUncertaintyOverlay(samples=(), model=model)

    md_values = stations["MD_m"].to_numpy(dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, int(model.ellipse_points), endpoint=False)
    samples: list[UncertaintyEllipseSample] = []
    for station_index in _display_station_indices(md_values=md_values, model=model):
        row = stations.iloc[int(station_index)]
        md_m = float(row["MD_m"])
        inc_deg = float(row["INC_deg"])
        azi_deg = float(row["AZI_deg"])
        semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
            md_m=md_m,
            inc_deg=inc_deg,
            model=model,
        )
        if max(semi_inc_m, semi_azi_m) < float(model.min_display_radius_m):
            continue

        _, inc_axis, azi_axis = local_uncertainty_axes_xyz(
            inc_deg=inc_deg,
            azi_deg=azi_deg,
        )
        center = np.array(
            [float(row["X_m"]), float(row["Y_m"]), float(row["Z_m"])], dtype=float
        )
        ring_xyz = (
            center[None, :]
            + np.cos(angles)[:, None] * (semi_inc_m * inc_axis[None, :])
            + np.sin(angles)[:, None] * (semi_azi_m * azi_axis[None, :])
        )
        ring_xyz = np.vstack([ring_xyz, ring_xyz[0]])
        ring_plan_xy = ring_xyz[:, :2].copy()
        ring_section_xz = np.column_stack(
            [
                _section_coordinate_xy(
                    x_values=ring_xyz[:, 0],
                    y_values=ring_xyz[:, 1],
                    surface=surface,
                    azimuth_deg=azimuth_deg,
                ),
                ring_xyz[:, 2],
            ]
        )
        samples.append(
            UncertaintyEllipseSample(
                station_index=int(station_index),
                md_m=md_m,
                center_xyz=(float(center[0]), float(center[1]), float(center[2])),
                ring_xyz=ring_xyz,
                ring_plan_xy=ring_plan_xy,
                ring_section_xz=ring_section_xz,
                semi_axis_inc_m=float(semi_inc_m),
                semi_axis_azi_m=float(semi_azi_m),
            )
        )
    return WellUncertaintyOverlay(samples=tuple(samples), model=model)


def _display_station_indices(
    *, md_values: np.ndarray, model: PlanningUncertaintyModel
) -> list[int]:
    if len(md_values) == 0:
        return []
    indices: list[int] = [0]
    next_sample_md = float(md_values[0]) + float(model.sample_step_m)
    for idx in range(1, len(md_values) - 1):
        md_value = float(md_values[idx])
        if md_value + 1e-9 >= next_sample_md:
            indices.append(int(idx))
            next_sample_md = md_value + float(model.sample_step_m)
    last_index = len(md_values) - 1
    if indices[-1] != last_index:
        indices.append(last_index)
    if len(indices) <= int(model.max_display_ellipses):
        return indices
    pick_positions = np.unique(
        np.linspace(0, len(indices) - 1, int(model.max_display_ellipses), dtype=int)
    )
    return [indices[int(pos)] for pos in pick_positions.tolist()]


def _section_coordinate_xy(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    surface: Point3D,
    azimuth_deg: float,
) -> np.ndarray:
    azimuth_rad = float(azimuth_deg) * DEG2RAD
    north = np.asarray(y_values, dtype=float) - float(surface.y)
    east = np.asarray(x_values, dtype=float) - float(surface.x)
    return north * np.cos(azimuth_rad) + east * np.sin(azimuth_rad)
