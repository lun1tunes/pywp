from __future__ import annotations

import numpy as np

from pywp.classification import TRAJECTORY_REVERSE_DIRECTION
from pywp.models import Point3D, TrajectoryConfig
from pywp.planner_types import PlanningError, SectionGeometry

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
SMALL = 1e-9


def _normalize_azimuth_deg(azimuth_deg: float) -> float:
    return float(np.mod(float(azimuth_deg), 360.0))


def _shortest_azimuth_delta_deg(azimuth_from_deg: float, azimuth_to_deg: float) -> float:
    delta = _normalize_azimuth_deg(azimuth_to_deg - azimuth_from_deg)
    if delta > 180.0:
        delta -= 360.0
    return float(delta)


def _mid_azimuth_deg(azimuth_from_deg: float, azimuth_to_deg: float) -> float:
    return _normalize_azimuth_deg(
        azimuth_from_deg
        + 0.5 * _shortest_azimuth_delta_deg(azimuth_from_deg, azimuth_to_deg)
    )


def _azimuth_deg_from_points(t1: Point3D, t3: Point3D) -> float:
    return _azimuth_deg_from_pair(surface=t1, target=t3)


def _azimuth_deg_from_pair(surface: Point3D, target: Point3D) -> float:
    dn = target.y - surface.y
    de = target.x - surface.x
    if np.isclose(dn, 0.0) and np.isclose(de, 0.0):
        raise PlanningError("Azimuth is undefined for overlapping plan coordinates.")
    azimuth_rad = np.arctan2(de, dn)
    return float(np.mod(azimuth_rad * RAD2DEG, 360.0))


def _project_to_section_axis(surface: Point3D, point: Point3D, azimuth_deg: float) -> tuple[float, float, float]:
    dn = point.y - surface.y
    de = point.x - surface.x
    az = azimuth_deg * DEG2RAD
    along = dn * np.cos(az) + de * np.sin(az)
    cross = -dn * np.sin(az) + de * np.cos(az)
    z = point.z - surface.z
    return float(along), float(cross), float(z)


def _inclination_from_displacement(ds_13: float, dz_13: float) -> float:
    return float(np.degrees(np.arctan2(ds_13, dz_13)))


def _dls_from_radius(radius_m: float) -> float:
    return float(30.0 * RAD2DEG / radius_m)


def _radius_from_dls(dls_deg_per_30m: float) -> float:
    return float(30.0 * RAD2DEG / dls_deg_per_30m)


def _required_dls_for_t1_reach(s1_m: float, z1_m: float, inc_entry_deg: float) -> float:
    if s1_m <= SMALL or z1_m <= SMALL:
        return float("nan")
    inc_entry_rad = inc_entry_deg * DEG2RAD
    one_minus_cos = max(1.0 - np.cos(inc_entry_rad), SMALL)
    sin_entry = max(np.sin(inc_entry_rad), SMALL)
    radius_limit_m = min(s1_m / one_minus_cos, z1_m / sin_entry)
    if radius_limit_m <= SMALL:
        return float("nan")
    return _dls_from_radius(radius_limit_m)


def _distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))


def _horizontal_offset(surface: Point3D, point: Point3D) -> float:
    return float(np.hypot(point.x - surface.x, point.y - surface.y))


def _build_section_geometry(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    config: TrajectoryConfig,
) -> SectionGeometry:
    azimuth_entry_deg = _azimuth_deg_from_points(t1=t1, t3=t3)
    azimuth_surface_t1_deg = _azimuth_deg_from_pair(surface=surface, target=t1)
    s1_m, c1_m, z1_m = _project_to_section_axis(surface=surface, point=t1, azimuth_deg=azimuth_entry_deg)
    s3_m, c3_m, z3_m = _project_to_section_axis(surface=surface, point=t3, azimuth_deg=azimuth_entry_deg)

    if z1_m <= 0.0:
        raise PlanningError("t1 must be below surface in TVD.")

    ds_13_m = s3_m - s1_m
    dz_13_m = z3_m - z1_m
    if ds_13_m <= 0.0:
        raise PlanningError("Invalid t1->t3 geometry: along-section offset must be positive.")

    inc_required_t1_t3_deg = _inclination_from_displacement(ds_13=ds_13_m, dz_13=dz_13_m)
    inc_entry_deg = float(config.entry_inc_target_deg)
    if inc_entry_deg > config.max_inc_deg + SMALL:
        raise PlanningError(
            "Entry INC target exceeds configured max INC. "
            f"entry_inc_target={inc_entry_deg:.2f} deg, max_inc={config.max_inc_deg:.2f} deg. "
            "Reduce entry_inc_target_deg or increase max_inc_deg."
        )
    if inc_required_t1_t3_deg > config.max_inc_deg + SMALL:
        raise PlanningError(
            "With current global max INC the t1->t3 geometry is infeasible without overbend. "
            f"Required straight INC is {inc_required_t1_t3_deg:.2f} deg, "
            f"max INC is {config.max_inc_deg:.2f} deg. "
            "To make target drillable, move t3 deeper and/or closer to t1 in horizontal projection, "
            "or increase max_inc_deg."
        )

    return SectionGeometry(
        s1_m=float(s1_m),
        z1_m=float(z1_m),
        ds_13_m=float(ds_13_m),
        dz_13_m=float(dz_13_m),
        azimuth_entry_deg=float(azimuth_entry_deg),
        azimuth_surface_t1_deg=float(azimuth_surface_t1_deg),
        inc_entry_deg=float(inc_entry_deg),
        inc_required_t1_t3_deg=float(inc_required_t1_t3_deg),
        t1_cross_m=float(c1_m),
        t3_cross_m=float(c3_m),
        t1_east_m=float(t1.x - surface.x),
        t1_north_m=float(t1.y - surface.y),
        t1_tvd_m=float(t1.z - surface.z),
    )


def _is_geometry_coplanar(geometry: SectionGeometry, tolerance_m: float) -> bool:
    return bool(abs(geometry.t1_cross_m) <= tolerance_m and abs(geometry.t3_cross_m) <= tolerance_m)


def _is_zero_azimuth_turn_geometry(
    geometry: SectionGeometry,
    target_direction: str,
    tolerance_m: float,
) -> bool:
    if str(target_direction) == TRAJECTORY_REVERSE_DIRECTION:
        return False
    return _is_geometry_coplanar(geometry=geometry, tolerance_m=tolerance_m)
