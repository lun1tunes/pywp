from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.constants import DEG2RAD
from pywp.models import Point3D

_REQUIRED_STATION_COLUMNS = frozenset({"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"})
UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC = "mwd_poor_magnetic"
UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC = "mwd_unknown_magnetic"
DEFAULT_UNCERTAINTY_PRESET = UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
UNCERTAINTY_PRESET_OPTIONS: dict[str, str] = {
    UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC: "MWD POOR magnetic (ISCWSA)",
    UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC: "MWD Unknown magnetic (ISCWSA)",
}


@dataclass(frozen=True)
class PlanningUncertaintyModel:
    """Planning-level first-order uncertainty model.

    This is intentionally not a full ISCWSA tool-error model. It uses a
    first-order local-angle approximation: position uncertainty in the plane
    normal to the borehole grows with measured depth and assumed 1-sigma
    inclination/azimuth errors.
    """

    sigma_inc_deg: float = 0.30
    sigma_azi_deg: float = 0.60
    sigma_lateral_drift_m_per_1000m: float = 12.0
    confidence_scale: float = 2.0
    sample_step_m: float = 100.0
    max_display_ellipses: int = 84
    ellipse_points: int = 36
    min_display_radius_m: float = 0.75
    near_vertical_isotropic_threshold_deg: float = 5.0
    directional_refine_threshold_deg: float = 5.0
    min_refined_step_m: float = 50.0

    def __post_init__(self) -> None:
        if float(self.sigma_inc_deg) <= 0.0:
            raise ValueError("sigma_inc_deg must be positive.")
        if float(self.sigma_azi_deg) <= 0.0:
            raise ValueError("sigma_azi_deg must be positive.")
        if float(self.confidence_scale) <= 0.0:
            raise ValueError("confidence_scale must be positive.")
        if float(self.sigma_lateral_drift_m_per_1000m) < 0.0:
            raise ValueError("sigma_lateral_drift_m_per_1000m cannot be negative.")
        if float(self.sample_step_m) <= 0.0:
            raise ValueError("sample_step_m must be positive.")
        if int(self.max_display_ellipses) <= 0:
            raise ValueError("max_display_ellipses must be positive.")
        if int(self.ellipse_points) < 12:
            raise ValueError("ellipse_points must be >= 12.")
        if float(self.min_display_radius_m) < 0.0:
            raise ValueError("min_display_radius_m cannot be negative.")
        if float(self.near_vertical_isotropic_threshold_deg) < 0.0:
            raise ValueError("near_vertical_isotropic_threshold_deg cannot be negative.")
        if float(self.directional_refine_threshold_deg) <= 0.0:
            raise ValueError("directional_refine_threshold_deg must be positive.")
        if float(self.min_refined_step_m) <= 0.0:
            raise ValueError("min_refined_step_m must be positive.")
        if float(self.min_refined_step_m) > float(self.sample_step_m):
            raise ValueError("min_refined_step_m must be <= sample_step_m.")


PLANNING_UNCERTAINTY_PRESET_MODELS: dict[str, PlanningUncertaintyModel] = {
    # ISCWSA MWD POOR magnetic: global geomagnetic model with high declination uncertainty
    # (~1.0-1.5° dec error, affecting azimuth significantly at higher inclinations)
    UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC: PlanningUncertaintyModel(
        sigma_inc_deg=0.35,
        sigma_azi_deg=1.20,
        sigma_lateral_drift_m_per_1000m=15.0,
        confidence_scale=1.0,
    ),
    # ISCWSA MWD Unknown magnetic: worst-case assumption for magnetic reference quality
    # (~1.5-2.5° dec error, very conservative for anti-collision planning)
    UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC: PlanningUncertaintyModel(
        sigma_inc_deg=0.40,
        sigma_azi_deg=1.80,
        sigma_lateral_drift_m_per_1000m=20.0,
        confidence_scale=1.0,
    ),
}
DEFAULT_PLANNING_UNCERTAINTY_MODEL = PLANNING_UNCERTAINTY_PRESET_MODELS[
    DEFAULT_UNCERTAINTY_PRESET
]


@dataclass(frozen=True)
class UncertaintyEllipseSample:
    station_index: int
    md_m: float
    center_xyz: tuple[float, float, float]
    center_plan_xy: tuple[float, float]
    center_section_xz: tuple[float, float]
    ring_xyz: np.ndarray
    ring_plan_xy: np.ndarray
    ring_section_xz: np.ndarray
    semi_axis_inc_m: float
    semi_axis_azi_m: float


@dataclass(frozen=True)
class WellUncertaintyOverlay:
    samples: tuple[UncertaintyEllipseSample, ...]
    model: PlanningUncertaintyModel


@dataclass(frozen=True)
class UncertaintyStationSample:
    station_index: int
    md_m: float
    inc_deg: float
    azi_deg: float
    center_xyz: tuple[float, float, float]
    covariance_xyz: np.ndarray


@dataclass(frozen=True)
class UncertaintyTubeMesh:
    vertices_xyz: np.ndarray
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray


def normalize_uncertainty_preset(
    preset: object,
) -> str:
    preset_key = str(preset or DEFAULT_UNCERTAINTY_PRESET).strip()
    if preset_key in PLANNING_UNCERTAINTY_PRESET_MODELS:
        return preset_key
    return DEFAULT_UNCERTAINTY_PRESET


def planning_uncertainty_model_for_preset(
    preset: object,
) -> PlanningUncertaintyModel:
    normalized = normalize_uncertainty_preset(preset)
    return PLANNING_UNCERTAINTY_PRESET_MODELS[normalized]


def uncertainty_preset_label(preset: object) -> str:
    preset_key = str(preset or DEFAULT_UNCERTAINTY_PRESET).strip()
    return UNCERTAINTY_PRESET_OPTIONS.get(
        preset_key,
        UNCERTAINTY_PRESET_OPTIONS[DEFAULT_UNCERTAINTY_PRESET],
    )


def uncertainty_model_caption(
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> str:
    return (
        "Показан 2σ planning-level конус неопределенности, построенный по "
        "эллиптическим сечениям в плоскости, нормальной к стволу. "
        "Базовая модель ordinary MWD proxy: first-order по ошибкам "
        f"INC/AZI = {float(model.sigma_inc_deg):.2f}°/{float(model.sigma_azi_deg):.2f}° "
        f"(1σ) + lateral drift {float(model.sigma_lateral_drift_m_per_1000m):.1f} м/1000м (1σ), "
        "взвешенный по латеральной экспозиции sin(INC). "
        "Это визуализация для планирования, не полноценная ISCWSA tool model."
    )


def tangent_vector_xyz(inc_deg: float, azi_deg: float) -> np.ndarray:
    inc_value = _validated_inclination_deg(inc_deg)
    azi_value = _normalized_azimuth_deg(azi_deg)
    inc_rad = inc_value * DEG2RAD
    azi_rad = azi_value * DEG2RAD
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
    inc_value = _validated_inclination_deg(inc_deg)
    azi_value = _normalized_azimuth_deg(azi_deg)
    inc_rad = inc_value * DEG2RAD
    azi_rad = azi_value * DEG2RAD
    tangent = tangent_vector_xyz(inc_deg=inc_value, azi_deg=azi_value)
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
    inc_value = _validated_inclination_deg(inc_deg)
    inc_rad = inc_value * DEG2RAD
    lateral_exposure = abs(float(np.sin(inc_rad)))
    sigma_drift_m = (
        (md_value / 1000.0)
        * float(model.sigma_lateral_drift_m_per_1000m)
        * lateral_exposure
    )
    sigma_inc_angle_m = md_value * float(model.sigma_inc_deg) * DEG2RAD
    sigma_inc_m = float(np.hypot(sigma_inc_angle_m, sigma_drift_m))
    if abs(float(inc_value)) < float(model.near_vertical_isotropic_threshold_deg):
        isotropic_radius_m = float(model.confidence_scale) * sigma_inc_m
        return isotropic_radius_m, isotropic_radius_m
    sigma_azi_angle_m = (
        md_value * abs(np.sin(inc_rad)) * float(model.sigma_azi_deg) * DEG2RAD
    )
    sigma_azi_m = float(np.hypot(sigma_azi_angle_m, sigma_drift_m))
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
    return np.asarray(
        station_uncertainty_covariance_xyz_many(
            md_m=np.asarray([float(md_m)], dtype=float),
            inc_deg=np.asarray([float(inc_deg)], dtype=float),
            azi_deg=np.asarray([float(azi_deg)], dtype=float),
            model=model,
        )[0],
        dtype=float,
    )


def station_uncertainty_covariance_xyz_many(
    *,
    md_m: np.ndarray,
    inc_deg: np.ndarray,
    azi_deg: np.ndarray,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> np.ndarray:
    md_values, inc_values, azi_values = np.broadcast_arrays(
        np.asarray(md_m, dtype=float),
        np.asarray(inc_deg, dtype=float),
        np.asarray(azi_deg, dtype=float),
    )
    md_nonnegative = np.maximum(md_values, 0.0)
    inc_rad = inc_values * DEG2RAD
    azi_rad = azi_values * DEG2RAD
    sin_inc = np.sin(inc_rad)
    cos_inc = np.cos(inc_rad)
    sin_azi = np.sin(azi_rad)
    cos_azi = np.cos(azi_rad)

    tangent = np.stack(
        [
            sin_inc * sin_azi,
            sin_inc * cos_azi,
            cos_inc,
        ],
        axis=-1,
    )
    inc_axis = np.stack(
        [
            cos_inc * sin_azi,
            cos_inc * cos_azi,
            -sin_inc,
        ],
        axis=-1,
    )
    azi_axis = np.stack(
        [
            cos_azi,
            -sin_azi,
            np.zeros_like(cos_azi),
        ],
        axis=-1,
    )

    lateral_exposure = np.abs(sin_inc)
    sigma_drift_m = (
        (md_nonnegative / 1000.0)
        * float(model.sigma_lateral_drift_m_per_1000m)
        * lateral_exposure
    )
    sigma_inc_angle_m = md_nonnegative * float(model.sigma_inc_deg) * DEG2RAD
    sigma_inc_m = np.hypot(sigma_inc_angle_m, sigma_drift_m)
    sigma_azi_angle_m = (
        md_nonnegative * lateral_exposure * float(model.sigma_azi_deg) * DEG2RAD
    )
    sigma_azi_m = np.hypot(sigma_azi_angle_m, sigma_drift_m)

    covariance = np.zeros(md_values.shape + (3, 3), dtype=float)
    sigma_inc2 = sigma_inc_m * sigma_inc_m
    sigma_azi2 = sigma_azi_m * sigma_azi_m

    near_vertical_mask = np.abs(inc_values) < float(
        model.near_vertical_isotropic_threshold_deg
    )
    if np.any(near_vertical_mask):
        projector = (
            np.eye(3, dtype=float)
            - tangent[..., :, None] * tangent[..., None, :]
        )
        covariance[near_vertical_mask] = (
            sigma_inc2[near_vertical_mask, None, None]
            * projector[near_vertical_mask]
        )
    if np.any(~near_vertical_mask):
        covariance[~near_vertical_mask] = (
            sigma_inc2[~near_vertical_mask, None, None]
            * (inc_axis[~near_vertical_mask, :, None] * inc_axis[~near_vertical_mask, None, :])
            + sigma_azi2[~near_vertical_mask, None, None]
            * (azi_axis[~near_vertical_mask, :, None] * azi_axis[~near_vertical_mask, None, :])
        )
    return covariance


def build_uncertainty_overlay(
    *,
    stations: pd.DataFrame,
    surface: Point3D,
    azimuth_deg: float,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    required_md_m: tuple[float, ...] = (),
) -> WellUncertaintyOverlay:
    missing_cols = sorted(_REQUIRED_STATION_COLUMNS.difference(stations.columns))
    if missing_cols:
        raise ValueError(
            "stations are missing required columns for uncertainty overlay: "
            + ", ".join(missing_cols)
        )
    if len(stations) == 0:
        return WellUncertaintyOverlay(samples=(), model=model)

    md_values, inc_values, azi_values_deg, x_values, y_values, z_values = (
        _validated_station_arrays(
            stations=stations,
            context="uncertainty overlay",
        )
    )
    azi_values_rad = np.unwrap(np.deg2rad(azi_values_deg))
    angles = np.linspace(0.0, 2.0 * np.pi, int(model.ellipse_points), endpoint=False)
    samples: list[UncertaintyEllipseSample] = []
    previous_ring_open_xyz: np.ndarray | None = None
    for md_m in _display_sample_md_values(
        md_values=md_values,
        inc_values=inc_values,
        azi_values_rad=azi_values_rad,
        model=model,
        required_md_m=required_md_m,
    ):
        state = _interpolate_station_state(
            md_values=md_values,
            inc_values=inc_values,
            azi_values_rad=azi_values_rad,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
            md_m=float(md_m),
        )
        station_index = int(np.argmin(np.abs(md_values - state["md_m"])))
        md_m = float(state["md_m"])
        inc_deg = float(state["inc_deg"])
        azi_deg = float(state["azi_deg"])
        semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
            md_m=md_m,
            inc_deg=inc_deg,
            model=model,
        )
        if max(semi_inc_m, semi_azi_m) < float(model.min_display_radius_m):
            continue

        tangent, inc_axis, azi_axis = local_uncertainty_axes_xyz(
            inc_deg=inc_deg,
            azi_deg=azi_deg,
        )
        if abs(float(inc_deg)) < float(model.near_vertical_isotropic_threshold_deg):
            inc_axis, azi_axis = _stable_normal_plane_basis(tangent)
        center = np.array(
            [float(state["x_m"]), float(state["y_m"]), float(state["z_m"])], dtype=float
        )
        ring_open_xyz = (
            center[None, :]
            + np.cos(angles)[:, None] * (semi_inc_m * inc_axis[None, :])
            + np.sin(angles)[:, None] * (semi_azi_m * azi_axis[None, :])
        )
        ring_open_xyz = _align_ring_for_continuity(
            ring_open_xyz=ring_open_xyz,
            previous_ring_open_xyz=previous_ring_open_xyz,
        )
        previous_ring_open_xyz = ring_open_xyz
        ring_xyz = ring_open_xyz
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
                center_plan_xy=(float(center[0]), float(center[1])),
                center_section_xz=(
                    float(
                        _section_coordinate_xy(
                            x_values=np.array([center[0]], dtype=float),
                            y_values=np.array([center[1]], dtype=float),
                            surface=surface,
                            azimuth_deg=azimuth_deg,
                        )[0]
                    ),
                    float(center[2]),
                ),
                ring_xyz=ring_xyz,
                ring_plan_xy=ring_plan_xy,
                ring_section_xz=ring_section_xz,
                semi_axis_inc_m=float(semi_inc_m),
                semi_axis_azi_m=float(semi_azi_m),
            )
        )
    return WellUncertaintyOverlay(samples=tuple(samples), model=model)


def build_uncertainty_station_samples(
    *,
    stations: pd.DataFrame,
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
    required_md_m: tuple[float, ...] = (),
) -> tuple[UncertaintyStationSample, ...]:
    missing_cols = sorted(_REQUIRED_STATION_COLUMNS.difference(stations.columns))
    if missing_cols:
        raise ValueError(
            "stations are missing required columns for uncertainty samples: "
            + ", ".join(missing_cols)
        )
    if len(stations) == 0:
        return ()

    md_values, inc_values, azi_values_deg, x_values, y_values, z_values = (
        _validated_station_arrays(
            stations=stations,
            context="uncertainty station samples",
        )
    )
    azi_values_rad = np.unwrap(np.deg2rad(azi_values_deg))

    sample_md_values = _display_sample_md_values(
        md_values=md_values,
        inc_values=inc_values,
        azi_values_rad=azi_values_rad,
        model=model,
        required_md_m=required_md_m,
    )
    if not sample_md_values:
        return ()

    sample_md = np.asarray(sample_md_values, dtype=float)
    sample_inc_deg = np.interp(sample_md, md_values, inc_values)
    sample_azi_rad = np.interp(sample_md, md_values, azi_values_rad)
    sample_azi_deg = np.rad2deg(sample_azi_rad) % 360.0
    sample_x = np.interp(sample_md, md_values, x_values)
    sample_y = np.interp(sample_md, md_values, y_values)
    sample_z = np.interp(sample_md, md_values, z_values)
    covariance_xyz = station_uncertainty_covariance_xyz_many(
        md_m=sample_md,
        inc_deg=sample_inc_deg,
        azi_deg=sample_azi_deg,
        model=model,
    )

    samples: list[UncertaintyStationSample] = []
    for index, md_m in enumerate(sample_md.tolist()):
        semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
            md_m=float(md_m),
            inc_deg=float(sample_inc_deg[index]),
            model=model,
        )
        station_index = int(np.argmin(np.abs(md_values - float(md_m))))
        samples.append(
            UncertaintyStationSample(
                station_index=station_index,
                md_m=float(md_m),
                inc_deg=float(sample_inc_deg[index]),
                azi_deg=float(sample_azi_deg[index]),
                center_xyz=(
                    float(sample_x[index]),
                    float(sample_y[index]),
                    float(sample_z[index]),
                ),
                covariance_xyz=np.asarray(covariance_xyz[index], dtype=float),
            )
        )
    return tuple(samples)


def _display_sample_md_values(
    *,
    md_values: np.ndarray,
    inc_values: np.ndarray,
    azi_values_rad: np.ndarray,
    model: PlanningUncertaintyModel,
    required_md_m: tuple[float, ...] = (),
) -> list[float]:
    if len(md_values) == 0:
        return []
    required_md_values = _required_md_values(
        md_values=md_values,
        required_md_m=required_md_m,
    )
    md_start = float(md_values[0])
    md_end = float(md_values[-1])
    sample_values: list[float] = [md_start]
    if md_end > md_start:
        step = float(model.sample_step_m)
        sample_md = md_start + step
        while sample_md < md_end - 1e-9:
            sample_values.append(float(sample_md))
            sample_md += step
    if sample_values[-1] != md_end:
        sample_values.append(md_end)
    for required_md in required_md_values:
        if all(abs(float(required_md) - existing) > 1e-6 for existing in sample_values):
            sample_values.append(float(required_md))
    sample_values = sorted(set(round(float(value), 6) for value in sample_values))
    sample_values = _refine_sample_md_values(
        md_values=md_values,
        inc_values=inc_values,
        azi_values_rad=azi_values_rad,
        sample_values=sample_values,
        model=model,
    )
    if len(sample_values) <= int(model.max_display_ellipses):
        return sample_values
    pinned = {round(float(md_start), 6), round(float(md_end), 6)}
    pinned.update(round(float(value), 6) for value in required_md_values)
    return _downsample_sample_md_values(
        sample_values=sample_values,
        pinned_values=pinned,
        max_display_ellipses=int(model.max_display_ellipses),
    )


def _required_md_values(
    *,
    md_values: np.ndarray,
    required_md_m: tuple[float, ...],
) -> list[float]:
    if len(md_values) == 0:
        return []
    md_start = float(md_values[0])
    md_end = float(md_values[-1])
    values: list[float] = []
    for md_marker in required_md_m:
        md_value = float(md_marker)
        if not np.isfinite(md_value):
            continue
        values.append(float(np.clip(md_value, md_start, md_end)))
    return sorted(set(round(float(value), 6) for value in values))


def _refine_sample_md_values(
    *,
    md_values: np.ndarray,
    inc_values: np.ndarray,
    azi_values_rad: np.ndarray,
    sample_values: list[float],
    model: PlanningUncertaintyModel,
) -> list[float]:
    if len(sample_values) < 2:
        return sample_values

    tangent_cache: dict[float, np.ndarray] = {}

    def tangent_at(md_m: float) -> np.ndarray:
        md_key = round(float(md_m), 6)
        if md_key in tangent_cache:
            return tangent_cache[md_key]
        inc_deg = float(np.interp(md_key, md_values, inc_values))
        azi_deg = float(np.rad2deg(np.interp(md_key, md_values, azi_values_rad)) % 360.0)
        tangent = tangent_vector_xyz(inc_deg=inc_deg, azi_deg=azi_deg)
        tangent_cache[md_key] = tangent
        return tangent

    def angular_change_deg(md_left: float, md_right: float) -> float:
        tangent_left = tangent_at(md_left)
        tangent_right = tangent_at(md_right)
        cosine = float(np.clip(np.dot(tangent_left, tangent_right), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    refined: list[float] = [float(sample_values[0])]

    def refine_interval(md_left: float, md_right: float) -> None:
        interval_m = float(md_right - md_left)
        if interval_m <= 1e-6:
            return
        if interval_m <= float(model.min_refined_step_m) + 1e-9:
            refined.append(float(md_right))
            return
        if angular_change_deg(md_left, md_right) <= float(model.directional_refine_threshold_deg):
            refined.append(float(md_right))
            return
        midpoint = round(float(0.5 * (md_left + md_right)), 6)
        if midpoint <= md_left + 1e-6 or midpoint >= md_right - 1e-6:
            refined.append(float(md_right))
            return
        refine_interval(md_left, midpoint)
        refine_interval(midpoint, md_right)

    for md_left, md_right in zip(sample_values, sample_values[1:]):
        refine_interval(float(md_left), float(md_right))
    return sorted(set(round(float(value), 6) for value in refined))


def _downsample_sample_md_values(
    *,
    sample_values: list[float],
    pinned_values: set[float],
    max_display_ellipses: int,
) -> list[float]:
    if len(sample_values) <= max_display_ellipses:
        return sample_values

    selected: set[float] = {
        round(float(value), 6)
        for value in sample_values
        if round(float(value), 6) in pinned_values
    }
    if len(selected) >= max_display_ellipses:
        return sorted(selected)[:max_display_ellipses]

    pick_positions = np.unique(
        np.linspace(0, len(sample_values) - 1, max_display_ellipses, dtype=int)
    )
    for pos in pick_positions.tolist():
        selected.add(round(float(sample_values[int(pos)]), 6))

    if len(selected) <= max_display_ellipses:
        return sorted(selected)

    sample_positions = {
        round(float(value), 6): idx for idx, value in enumerate(sample_values)
    }
    ranked = sorted(
        (float(value) for value in selected if float(value) not in pinned_values),
        key=lambda value: min(
            abs(int(sample_positions[round(float(value), 6)]) - int(pos))
            for pos in pick_positions.tolist()
        ),
        reverse=True,
    )
    while len(selected) > max_display_ellipses and ranked:
        selected.remove(round(float(ranked.pop(0)), 6))
    return sorted(selected)


def _interpolate_station_state(
    *,
    md_values: np.ndarray,
    inc_values: np.ndarray,
    azi_values_rad: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    md_m: float,
) -> dict[str, float]:
    _validate_md_values_for_interpolation(md_values, context="uncertainty interpolation")
    md_value = float(np.clip(md_m, float(md_values[0]), float(md_values[-1])))
    return {
        "md_m": md_value,
        "inc_deg": float(np.interp(md_value, md_values, inc_values)),
        "azi_deg": float(np.rad2deg(np.interp(md_value, md_values, azi_values_rad)) % 360.0),
        "x_m": float(np.interp(md_value, md_values, x_values)),
        "y_m": float(np.interp(md_value, md_values, y_values)),
        "z_m": float(np.interp(md_value, md_values, z_values)),
    }


def _validated_inclination_deg(inc_deg: float) -> float:
    inc_value = float(inc_deg)
    if not np.isfinite(inc_value):
        raise ValueError("inc_deg must be finite for uncertainty calculations.")
    if inc_value < 0.0 or inc_value > 180.0:
        raise ValueError("inc_deg must be within [0, 180] for uncertainty calculations.")
    return inc_value


def _normalized_azimuth_deg(azi_deg: float) -> float:
    azi_value = float(azi_deg)
    if not np.isfinite(azi_value):
        raise ValueError("azi_deg must be finite for uncertainty calculations.")
    return float(azi_value % 360.0)


def _validate_md_values_for_interpolation(
    md_values: np.ndarray,
    *,
    context: str,
) -> None:
    md_array = np.asarray(md_values, dtype=float)
    if md_array.ndim != 1 or len(md_array) == 0:
        raise ValueError(f"{context}: expected a non-empty 1D MD array.")
    if not np.all(np.isfinite(md_array)):
        raise ValueError(f"{context}: MD values must be finite.")
    if len(md_array) > 1 and np.any(np.diff(md_array) <= 0.0):
        raise ValueError(f"{context}: MD values must be strictly increasing.")


def _validated_station_arrays(
    *,
    stations: pd.DataFrame,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    azi_values_deg = stations["AZI_deg"].to_numpy(dtype=float)
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    z_values = stations["Z_m"].to_numpy(dtype=float)
    _validate_md_values_for_interpolation(md_values, context=context)
    if not (
        np.all(np.isfinite(inc_values))
        and np.all(np.isfinite(azi_values_deg))
        and np.all(np.isfinite(x_values))
        and np.all(np.isfinite(y_values))
        and np.all(np.isfinite(z_values))
    ):
        raise ValueError(f"{context}: station INC/AZI/X/Y/Z values must be finite.")
    if np.any((inc_values < 0.0) | (inc_values > 180.0)):
        raise ValueError(f"{context}: station INC values must be within [0, 180].")
    azi_values_deg = np.mod(azi_values_deg, 360.0)
    return md_values, inc_values, azi_values_deg, x_values, y_values, z_values


def uncertainty_ribbon_polygon(
    overlay: WellUncertaintyOverlay,
    *,
    projection: str,
) -> np.ndarray:
    if projection not in {"plan", "section"}:
        raise ValueError("projection must be 'plan' or 'section'.")
    if len(overlay.samples) < 2:
        return np.empty((0, 2), dtype=float)

    positive_side: list[np.ndarray] = []
    negative_side: list[np.ndarray] = []
    previous_positive_index: int | None = None
    previous_negative_index: int | None = None
    for sample_index, sample in enumerate(overlay.samples):
        center = (
            np.array(sample.center_plan_xy, dtype=float)
            if projection == "plan"
            else np.array(sample.center_section_xz, dtype=float)
        )
        ring_closed = (
            np.asarray(sample.ring_plan_xy, dtype=float)
            if projection == "plan"
            else np.asarray(sample.ring_section_xz, dtype=float)
        )
        ring = _open_closed_ring(ring_closed)
        if ring.shape[0] < 3:
            continue
        tangent = _projected_center_tangent(
            overlay=overlay,
            sample_index=sample_index,
            projection=projection,
        )
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            tangent = np.array([1.0, 0.0], dtype=float)
            tangent_norm = 1.0
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        offsets = (ring - center[None, :]) @ normal
        positive_index = _continuous_extreme_index(
            offsets=offsets,
            maximize=True,
            previous_index=previous_positive_index,
        )
        negative_index = _continuous_extreme_index(
            offsets=offsets,
            maximize=False,
            previous_index=previous_negative_index,
        )
        positive_side.append(ring[int(positive_index)])
        negative_side.append(ring[int(negative_index)])
        previous_positive_index = int(positive_index)
        previous_negative_index = int(negative_index)

    if len(positive_side) < 2:
        return np.empty((0, 2), dtype=float)
    polygon = np.vstack(
        [
            np.asarray(positive_side, dtype=float),
            np.asarray(negative_side[::-1], dtype=float),
        ]
    )
    return np.vstack([polygon, polygon[0]])


def build_uncertainty_tube_mesh(
    overlay: WellUncertaintyOverlay,
) -> UncertaintyTubeMesh | None:
    if len(overlay.samples) < 2:
        return None

    rings = [np.asarray(sample.ring_xyz[:-1], dtype=float) for sample in overlay.samples]
    points_per_ring = int(rings[0].shape[0])
    if points_per_ring < 3:
        return None

    vertices = np.vstack(rings)
    triangles_i: list[int] = []
    triangles_j: list[int] = []
    triangles_k: list[int] = []

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
            np.asarray(overlay.samples[0].center_xyz, dtype=float),
            np.asarray(overlay.samples[-1].center_xyz, dtype=float),
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


def _projected_center_tangent(
    *,
    overlay: WellUncertaintyOverlay,
    sample_index: int,
    projection: str,
) -> np.ndarray:
    centers = [
        np.array(
            sample.center_plan_xy if projection == "plan" else sample.center_section_xz,
            dtype=float,
        )
        for sample in overlay.samples
    ]
    current = centers[int(sample_index)]
    if int(sample_index) == 0:
        return centers[1] - current
    if int(sample_index) == len(centers) - 1:
        return current - centers[-2]
    return centers[int(sample_index) + 1] - centers[int(sample_index) - 1]


def _open_closed_ring(ring_xy: np.ndarray) -> np.ndarray:
    ring = np.asarray(ring_xy, dtype=float)
    if ring.ndim != 2 or ring.shape[0] == 0:
        return ring
    if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1], atol=1e-9):
        return ring[:-1]
    return ring


def _circular_index_distance(index_a: int, index_b: int, size: int) -> int:
    if size <= 0:
        return 0
    delta = abs(int(index_a) - int(index_b))
    return min(delta, size - delta)


def _continuous_extreme_index(
    *,
    offsets: np.ndarray,
    maximize: bool,
    previous_index: int | None,
) -> int:
    values = np.asarray(offsets, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        return 0

    best_value = float(np.max(values) if maximize else np.min(values))
    spread = float(np.max(values) - np.min(values))
    tolerance = max(spread * 0.03, 1e-9)
    if maximize:
        candidate_indices = np.flatnonzero(values >= best_value - tolerance)
        fallback_index = int(np.argmax(values))
    else:
        candidate_indices = np.flatnonzero(values <= best_value + tolerance)
        fallback_index = int(np.argmin(values))

    if len(candidate_indices) == 0:
        return fallback_index
    if previous_index is None:
        return fallback_index

    return int(
        min(
            candidate_indices.tolist(),
            key=lambda idx: (
                _circular_index_distance(int(idx), int(previous_index), len(values)),
                abs(float(values[int(idx)]) - best_value),
            ),
        )
    )


def _stable_normal_plane_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent_unit = np.asarray(tangent, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent_unit))
    if tangent_norm <= 1e-12:
        raise ValueError("tangent vector cannot be zero.")
    tangent_unit = tangent_unit / tangent_norm

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(tangent_unit, reference))) > 0.98:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)

    axis_1 = np.cross(reference, tangent_unit)
    axis_1_norm = float(np.linalg.norm(axis_1))
    if axis_1_norm <= 1e-12:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
        axis_1 = np.cross(reference, tangent_unit)
        axis_1_norm = float(np.linalg.norm(axis_1))
    axis_1 = axis_1 / axis_1_norm
    axis_2 = np.cross(tangent_unit, axis_1)
    axis_2 = axis_2 / float(np.linalg.norm(axis_2))
    return axis_1, axis_2


def _align_ring_for_continuity(
    *,
    ring_open_xyz: np.ndarray,
    previous_ring_open_xyz: np.ndarray | None,
) -> np.ndarray:
    if previous_ring_open_xyz is None:
        return np.asarray(ring_open_xyz, dtype=float)

    current = np.asarray(ring_open_xyz, dtype=float)
    previous = np.asarray(previous_ring_open_xyz, dtype=float)
    if current.shape != previous.shape or current.ndim != 2 or current.shape[0] < 3:
        return current

    best_ring = current
    best_cost = float(np.mean(np.linalg.norm(current - previous, axis=1)))

    for candidate_base in (current, current[::-1]):
        for shift in range(current.shape[0]):
            candidate = np.roll(candidate_base, shift=shift, axis=0)
            candidate_cost = float(np.mean(np.linalg.norm(candidate - previous, axis=1)))
            if candidate_cost + 1e-9 < best_cost:
                best_ring = candidate
                best_cost = candidate_cost
    return best_ring
