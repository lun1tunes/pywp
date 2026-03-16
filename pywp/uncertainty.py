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

    sigma_inc_deg: float = 0.30
    sigma_azi_deg: float = 0.60
    sigma_lateral_drift_m_per_1000m: float = 12.0
    confidence_scale: float = 2.0
    sample_step_m: float = 100.0
    max_display_ellipses: int = 36
    ellipse_points: int = 36
    min_display_radius_m: float = 0.75
    near_vertical_isotropic_threshold_deg: float = 5.0

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


DEFAULT_PLANNING_UNCERTAINTY_MODEL = PlanningUncertaintyModel()


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
class UncertaintyTubeMesh:
    vertices_xyz: np.ndarray
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray


def uncertainty_model_caption(
    model: PlanningUncertaintyModel = DEFAULT_PLANNING_UNCERTAINTY_MODEL,
) -> str:
    return (
        "Показан 2σ planning-level конус неопределенности, построенный по "
        "эллиптическим сечениям в плоскости, нормальной к стволу. "
        "Базовая модель ordinary MWD proxy: first-order по ошибкам "
        f"INC/AZI = {float(model.sigma_inc_deg):.2f}°/{float(model.sigma_azi_deg):.2f}° "
        f"(1σ) + lateral drift {float(model.sigma_lateral_drift_m_per_1000m):.1f} м/1000м (1σ). "
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
    sigma_drift_m = (md_value / 1000.0) * float(model.sigma_lateral_drift_m_per_1000m)
    sigma_inc_angle_m = md_value * float(model.sigma_inc_deg) * DEG2RAD
    sigma_inc_m = float(np.hypot(sigma_inc_angle_m, sigma_drift_m))
    if abs(float(inc_deg)) < float(model.near_vertical_isotropic_threshold_deg):
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
    tangent, inc_axis, azi_axis = local_uncertainty_axes_xyz(
        inc_deg=inc_deg,
        azi_deg=azi_deg,
    )
    semi_inc_m, semi_azi_m = station_uncertainty_axes_m(
        md_m=md_m,
        inc_deg=inc_deg,
        model=model,
    )
    scale = float(model.confidence_scale)
    sigma_inc_m = semi_inc_m / scale
    sigma_azi_m = semi_azi_m / scale
    if abs(float(inc_deg)) < float(model.near_vertical_isotropic_threshold_deg):
        axis_1, axis_2 = _stable_normal_plane_basis(tangent)
        return sigma_inc_m * sigma_inc_m * (
            np.outer(axis_1, axis_1) + np.outer(axis_2, axis_2)
        )
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

    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    azi_values_deg = stations["AZI_deg"].to_numpy(dtype=float)
    azi_values_rad = np.unwrap(np.deg2rad(azi_values_deg))
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    z_values = stations["Z_m"].to_numpy(dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, int(model.ellipse_points), endpoint=False)
    samples: list[UncertaintyEllipseSample] = []
    for md_m in _display_sample_md_values(
        md_values=md_values,
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


def _display_sample_md_values(
    *,
    md_values: np.ndarray,
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
    if len(sample_values) <= int(model.max_display_ellipses):
        return sample_values
    pinned = {round(float(md_start), 6), round(float(md_end), 6)}
    pinned.update(round(float(value), 6) for value in required_md_values)
    pick_positions = np.unique(
        np.linspace(0, len(sample_values) - 1, int(model.max_display_ellipses), dtype=int)
    )
    selected = [sample_values[int(pos)] for pos in pick_positions.tolist()]
    for pinned_value in pinned:
        if all(abs(float(pinned_value) - existing) > 1e-6 for existing in selected):
            selected.append(float(pinned_value))
    return sorted(set(round(float(value), 6) for value in selected))


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
    md_value = float(np.clip(md_m, float(md_values[0]), float(md_values[-1])))
    return {
        "md_m": md_value,
        "inc_deg": float(np.interp(md_value, md_values, inc_values)),
        "azi_deg": float(np.rad2deg(np.interp(md_value, md_values, azi_values_rad)) % 360.0),
        "x_m": float(np.interp(md_value, md_values, x_values)),
        "y_m": float(np.interp(md_value, md_values, y_values)),
        "z_m": float(np.interp(md_value, md_values, z_values)),
    }


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
    for sample_index, sample in enumerate(overlay.samples):
        center = (
            np.array(sample.center_plan_xy, dtype=float)
            if projection == "plan"
            else np.array(sample.center_section_xz, dtype=float)
        )
        ring = (
            np.asarray(sample.ring_plan_xy, dtype=float)
            if projection == "plan"
            else np.asarray(sample.ring_section_xz, dtype=float)
        )
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
        positive_side.append(ring[int(np.argmax(offsets))])
        negative_side.append(ring[int(np.argmin(offsets))])

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
