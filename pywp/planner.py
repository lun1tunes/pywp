from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pywp.mcm import add_dls, compute_positions_min_curv
from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.segments import BuildSegment, HoldSegment, VerticalSegment
from pywp.trajectory import WellTrajectory

DEG2RAD = np.pi / 180.0


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProfileParameters:
    inc_entry_deg: float
    dls_build_deg_per_30m: float
    vertical_length_m: float
    hold_length_m: float
    azimuth_deg: float

    @property
    def build_length_m(self) -> float:
        return abs(self.inc_entry_deg) / self.dls_build_deg_per_30m * 30.0


class TrajectoryPlanner:
    def plan(
        self,
        surface: Point3D,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
    ) -> PlannerResult:
        _validate_config(config)

        azimuth_deg = _azimuth_deg_from_points(t1=t1, t3=t3)
        s1, c1, z1 = _project_to_section_axis(surface=surface, point=t1, azimuth_deg=azimuth_deg)
        s3, c3, z3 = _project_to_section_axis(surface=surface, point=t3, azimuth_deg=azimuth_deg)

        if abs(c1) > config.pos_tolerance_m or abs(c3) > config.pos_tolerance_m:
            raise PlanningError(
                "Points are not coplanar for J-profile without TURN. "
                "Add TURN support or use coplanar points."
            )

        if s1 <= 0.0:
            raise PlanningError("t1 must be ahead of surface along section direction.")

        if s3 <= s1:
            raise PlanningError("t3 must be ahead of t1 along section direction.")

        ds_13 = s3 - s1
        dz_13 = z3 - z1

        if ds_13 <= 0.0:
            raise PlanningError("Invalid t1->t3 geometry: along-section offset must be positive.")

        inc_entry_deg = _inclination_from_displacement(ds_13=ds_13, dz_13=dz_13)
        if abs(inc_entry_deg - config.entry_inc_target_deg) > config.entry_inc_tolerance_deg:
            raise PlanningError(
                f"INC at t1 from target geometry is {inc_entry_deg:.2f} deg, "
                f"outside required {config.entry_inc_target_deg:.1f}±{config.entry_inc_tolerance_deg:.1f} deg."
            )

        params = _solve_j_profile(
            inc_entry_deg=inc_entry_deg,
            s1=s1,
            z1=z1,
            ds_13=ds_13,
            dz_13=dz_13,
            azimuth_deg=azimuth_deg,
            config=config,
        )

        trajectory = _build_trajectory(params)
        md_t1_m = params.vertical_length_m + params.build_length_m

        control = compute_positions_min_curv(
            trajectory.stations(md_step_m=config.md_step_control_m),
            start=surface,
        )
        control = add_dls(control)

        summary = _build_summary(df=control, t1=t1, t3=t3, md_t1_m=md_t1_m, config=config)
        _assert_solution_is_valid(summary=summary, config=config)

        output = compute_positions_min_curv(trajectory.stations(md_step_m=config.md_step_m), start=surface)
        output = add_dls(output)

        return PlannerResult(stations=output, summary=summary, azimuth_deg=azimuth_deg, md_t1_m=md_t1_m)


def _solve_j_profile(
    inc_entry_deg: float,
    s1: float,
    z1: float,
    ds_13: float,
    dz_13: float,
    azimuth_deg: float,
    config: TrajectoryConfig,
) -> ProfileParameters:
    i = inc_entry_deg * DEG2RAD
    one_minus_cos = 1.0 - np.cos(i)
    if one_minus_cos <= 1e-9:
        raise PlanningError("Entry inclination is too small for a valid J-profile build segment.")

    radius_build = s1 / one_minus_cos
    if radius_build <= 0.0:
        raise PlanningError("Invalid build radius computed for J-profile.")

    dls_build = _dls_from_radius(radius_build)
    if dls_build < config.dls_build_min_deg_per_30m - 1e-9 or dls_build > config.dls_build_max_deg_per_30m + 1e-9:
        raise PlanningError(
            f"Required BUILD DLS is {dls_build:.2f} deg/30m, "
            f"outside allowed range [{config.dls_build_min_deg_per_30m:.2f}, {config.dls_build_max_deg_per_30m:.2f}]."
        )

    vertical_length = z1 - radius_build * np.sin(i)
    if vertical_length < -1e-6:
        raise PlanningError("J-profile cannot reach t1 with non-negative vertical section for given geometry.")
    vertical_length = max(0.0, vertical_length)

    hold_length = float(np.hypot(ds_13, dz_13))
    if hold_length <= 0.0:
        raise PlanningError("t1 and t3 are identical; HOLD section length is zero.")

    return ProfileParameters(
        inc_entry_deg=float(inc_entry_deg),
        dls_build_deg_per_30m=float(dls_build),
        vertical_length_m=float(vertical_length),
        hold_length_m=hold_length,
        azimuth_deg=float(azimuth_deg),
    )


def _build_trajectory(params: ProfileParameters) -> WellTrajectory:
    segments = [
        VerticalSegment(length_m=params.vertical_length_m, azi_deg=params.azimuth_deg, name="VERTICAL"),
        BuildSegment(
            inc_from_deg=0.0,
            inc_to_deg=params.inc_entry_deg,
            dls_deg_per_30m=params.dls_build_deg_per_30m,
            azi_deg=params.azimuth_deg,
            name="BUILD",
        ),
        HoldSegment(
            length_m=params.hold_length_m,
            inc_deg=params.inc_entry_deg,
            azi_deg=params.azimuth_deg,
            name="HOLD",
        ),
    ]
    return WellTrajectory(segments=segments)


def _build_summary(
    df: pd.DataFrame,
    t1: Point3D,
    t3: Point3D,
    md_t1_m: float,
    config: TrajectoryConfig,
) -> dict[str, float]:
    t1_idx = int((df["MD_m"] - md_t1_m).abs().idxmin())
    t1_row = df.loc[t1_idx]
    t3_row = df.iloc[-1]

    distance_t1 = _distance_3d(t1_row["X_m"], t1_row["Y_m"], t1_row["Z_m"], t1.x, t1.y, t1.z)
    distance_t3 = _distance_3d(t3_row["X_m"], t3_row["Y_m"], t3_row["Z_m"], t3.x, t3.y, t3.z)

    max_dls = float(np.nanmax(df["DLS_deg_per_30m"].to_numpy()))

    summary: dict[str, float] = {
        "distance_t1_m": float(distance_t1),
        "distance_t3_m": float(distance_t3),
        "entry_inc_deg": float(t1_row["INC_deg"]),
        "entry_inc_target_deg": float(config.entry_inc_target_deg),
        "entry_inc_tolerance_deg": float(config.entry_inc_tolerance_deg),
        "max_dls_total_deg_per_30m": max_dls,
        "md_total_m": float(df["MD_m"].iloc[-1]),
    }

    for segment, limit in config.dls_limits_deg_per_30m.items():
        seg_max = float(df.loc[df["segment"] == segment, "DLS_deg_per_30m"].max(skipna=True))
        if np.isnan(seg_max):
            seg_max = 0.0
        summary[f"max_dls_{segment.lower()}_deg_per_30m"] = seg_max
        summary[f"dls_limit_{segment.lower()}_deg_per_30m"] = float(limit)

    return summary


def _assert_solution_is_valid(summary: dict[str, float], config: TrajectoryConfig) -> None:
    if summary["distance_t1_m"] > config.pos_tolerance_m:
        raise PlanningError("Failed to hit t1 within tolerance.")

    if summary["distance_t3_m"] > config.pos_tolerance_m:
        raise PlanningError("Failed to hit t3 within tolerance.")

    if abs(summary["entry_inc_deg"] - config.entry_inc_target_deg) > config.entry_inc_tolerance_deg + 1e-6:
        raise PlanningError("Entry inclination at t1 is outside required target range.")

    for segment, limit in config.dls_limits_deg_per_30m.items():
        actual = summary.get(f"max_dls_{segment.lower()}_deg_per_30m", 0.0)
        if actual > limit + 1e-6:
            raise PlanningError(f"DLS limit exceeded on segment {segment}: {actual:.2f} > {limit:.2f}")


def _validate_config(config: TrajectoryConfig) -> None:
    if config.md_step_m <= 0.0 or config.md_step_control_m <= 0.0:
        raise PlanningError("MD steps must be positive.")

    if config.pos_tolerance_m <= 0.0:
        raise PlanningError("Position tolerance must be positive.")

    if config.entry_inc_tolerance_deg < 0.0:
        raise PlanningError("entry_inc_tolerance_deg must be non-negative.")

    if config.dls_build_min_deg_per_30m > config.dls_build_max_deg_per_30m:
        raise PlanningError("dls_build_min_deg_per_30m cannot exceed dls_build_max_deg_per_30m.")

    for segment, limit in config.dls_limits_deg_per_30m.items():
        if limit < 0.0:
            raise PlanningError(f"DLS limit for segment {segment} cannot be negative.")


def _azimuth_deg_from_points(t1: Point3D, t3: Point3D) -> float:
    dn = t3.y - t1.y
    de = t3.x - t1.x
    if np.isclose(dn, 0.0) and np.isclose(de, 0.0):
        raise PlanningError("t1 and t3 overlap in plan; azimuth is undefined.")
    azimuth_rad = np.arctan2(de, dn)
    return float(np.mod(azimuth_rad * 180.0 / np.pi, 360.0))


def _project_to_section_axis(surface: Point3D, point: Point3D, azimuth_deg: float) -> tuple[float, float, float]:
    dn = point.y - surface.y
    de = point.x - surface.x
    az = azimuth_deg * DEG2RAD

    along = dn * np.cos(az) + de * np.sin(az)
    cross = -dn * np.sin(az) + de * np.cos(az)
    z = point.z - surface.z
    return float(along), float(cross), float(z)


def _inclination_from_displacement(ds_13: float, dz_13: float) -> float:
    # Inclination is measured from vertical: tan(INC) = horizontal / vertical.
    return float(np.degrees(np.arctan2(ds_13, dz_13)))


def _dls_from_radius(radius_m: float) -> float:
    return float(30.0 * 180.0 / (np.pi * radius_m))


def _distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))
