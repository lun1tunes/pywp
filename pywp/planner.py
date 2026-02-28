from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pywp.mcm import add_dls, compute_positions_min_curv
from pywp.models import (
    ALLOWED_OBJECTIVE_MODES,
    OBJECTIVE_MAXIMIZE_HOLD,
    OBJECTIVE_MINIMIZE_BUILD_DLS,
    PlannerResult,
    Point3D,
    TrajectoryConfig,
)
from pywp.segments import BuildSegment, HoldSegment, HorizontalSegment, VerticalSegment
from pywp.trajectory import WellTrajectory

# Directional-planning formulas use the standard circular-arc relations:
# arc length L = R * dI(rad), ds = R * (cos(I0)-cos(I1)), dTVD = R * (sin(I1)-sin(I0)).
# DLS/R conversion: DLS(deg/30m) = 30 * (180/pi) / R.
# See references used in this task:
# - IHS Petra help (minimum curvature / SPE 2262): https://onlinehelp.ihs.com/Energy/Petra/2021/Content/main_survey_computation.htm
# - SLB dogleg severity definition: https://glossary.slb.com/en/terms/d/dogleg_severity
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
SMALL = 1e-9


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class SectionGeometry:
    s1_m: float
    z1_m: float
    ds_13_m: float
    dz_13_m: float
    azimuth_deg: float
    inc_entry_deg: float


@dataclass(frozen=True)
class ProfileParameters:
    kop_vertical_m: float
    inc_entry_deg: float
    inc_hold_deg: float
    dls_build1_deg_per_30m: float
    dls_build2_deg_per_30m: float
    build1_length_m: float
    hold_length_m: float
    build2_length_m: float
    horizontal_length_m: float
    azimuth_deg: float

    @property
    def md_t1_m(self) -> float:
        return float(self.kop_vertical_m + self.build1_length_m + self.hold_length_m + self.build2_length_m)

    @property
    def md_total_m(self) -> float:
        return float(self.md_t1_m + self.horizontal_length_m)


class TrajectoryPlanner:
    def plan(
        self,
        surface: Point3D,
        t1: Point3D,
        t3: Point3D,
        config: TrajectoryConfig,
    ) -> PlannerResult:
        _validate_config(config)

        geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
        params = _solve_build_hold_build_horizontal_profile(geometry=geometry, config=config)
        trajectory = _build_trajectory(params=params)

        control = compute_positions_min_curv(
            trajectory.stations(md_step_m=config.md_step_control_m),
            start=surface,
        )
        control = add_dls(control)

        summary = _build_summary(
            df=control,
            t1=t1,
            t3=t3,
            md_t1_m=params.md_t1_m,
            kop_vertical_m=params.kop_vertical_m,
            config=config,
        )
        _assert_solution_is_valid(summary=summary, config=config)

        output = compute_positions_min_curv(trajectory.stations(md_step_m=config.md_step_m), start=surface)
        output = add_dls(output)

        return PlannerResult(
            stations=output,
            summary=summary,
            azimuth_deg=geometry.azimuth_deg,
            md_t1_m=params.md_t1_m,
        )


def _build_section_geometry(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    config: TrajectoryConfig,
) -> SectionGeometry:
    azimuth_deg = _azimuth_deg_from_points(t1=t1, t3=t3)
    s1_m, c1_m, z1_m = _project_to_section_axis(surface=surface, point=t1, azimuth_deg=azimuth_deg)
    s3_m, c3_m, z3_m = _project_to_section_axis(surface=surface, point=t3, azimuth_deg=azimuth_deg)

    if abs(c1_m) > config.pos_tolerance_m or abs(c3_m) > config.pos_tolerance_m:
        az_s_t1 = float(np.mod(np.degrees(np.arctan2(t1.x - surface.x, t1.y - surface.y)), 360.0))
        az_s_t3 = float(np.mod(np.degrees(np.arctan2(t3.x - surface.x, t3.y - surface.y)), 360.0))
        raise PlanningError(
            "Points are not coplanar for no-TURN profile. "
            f"S->t1 azimuth={az_s_t1:.2f} deg, S->t3 azimuth={az_s_t3:.2f} deg. "
            "Use coplanar targets (same azimuth from S) or add TURN support."
        )

    if s1_m <= 0.0:
        raise PlanningError("t1 must be ahead of surface along section direction.")
    if s3_m <= s1_m:
        raise PlanningError("t3 must be ahead of t1 along section direction.")

    ds_13_m = s3_m - s1_m
    dz_13_m = z3_m - z1_m
    if ds_13_m <= 0.0:
        raise PlanningError("Invalid t1->t3 geometry: along-section offset must be positive.")

    inc_entry_deg = _inclination_from_displacement(ds_13=ds_13_m, dz_13=dz_13_m)
    if abs(inc_entry_deg - config.entry_inc_target_deg) > config.entry_inc_tolerance_deg:
        raise PlanningError(
            f"INC at t1 from target geometry is {inc_entry_deg:.2f} deg, "
            f"outside required {config.entry_inc_target_deg:.1f}±{config.entry_inc_tolerance_deg:.1f} deg."
        )

    return SectionGeometry(
        s1_m=float(s1_m),
        z1_m=float(z1_m),
        ds_13_m=float(ds_13_m),
        dz_13_m=float(dz_13_m),
        azimuth_deg=float(azimuth_deg),
        inc_entry_deg=float(inc_entry_deg),
    )


def _solve_build_hold_build_horizontal_profile(
    geometry: SectionGeometry,
    config: TrajectoryConfig,
) -> ProfileParameters:
    lower = float(config.dls_build_min_deg_per_30m)
    upper = float(config.dls_build_max_deg_per_30m)

    kop_candidates = _iter_kop_candidates(geometry=geometry, config=config)
    best: ProfileParameters | None = None
    for kop_vertical_m in kop_candidates:
        candidate = _optimize_profile_for_objective(
            lower_dls=lower,
            upper_dls=upper,
            geometry=geometry,
            kop_vertical_m=kop_vertical_m,
            config=config,
        )
        if candidate is None:
            continue
        best = _select_better_candidate(current=best, candidate=candidate, objective_mode=config.objective_mode)

    if best is not None:
        return best

    available_z_for_builds = geometry.z1_m - float(config.kop_min_vertical_m)
    if available_z_for_builds <= SMALL:
        raise PlanningError(
            "No valid profile: minimum vertical before KOP is too deep for t1 TVD. "
            f"t1 TVD is {geometry.z1_m:.2f} m, requested minimum vertical is {config.kop_min_vertical_m:.2f} m."
        )

    required_dls = _required_dls_for_t1_reach(
        s1_m=geometry.s1_m,
        z1_m=available_z_for_builds,
        inc_entry_deg=geometry.inc_entry_deg,
    )
    raise PlanningError(
        "No valid VERTICAL->BUILD1->HOLD->BUILD2->HORIZONTAL solution found within configured limits. "
        f"Try increasing BUILD max DLS above {required_dls:.2f} deg/30m, reducing minimum vertical before KOP, "
        "or relaxing constraints."
    )


def _profile_for_dls(
    geometry: SectionGeometry,
    dls_build_deg_per_30m: float,
    kop_vertical_m: float,
) -> ProfileParameters | None:
    if dls_build_deg_per_30m <= 0.0:
        return None
    if kop_vertical_m < 0.0:
        return None

    horizontal_length_m = float(np.hypot(geometry.ds_13_m, geometry.dz_13_m))
    if horizontal_length_m <= SMALL:
        return None

    inc_entry_rad = geometry.inc_entry_deg * DEG2RAD
    if inc_entry_rad <= SMALL or inc_entry_rad >= (np.pi / 2.0) - SMALL:
        return None

    radius_m = _radius_from_dls(dls_build_deg_per_30m)
    z1_after_kop_m = geometry.z1_m - kop_vertical_m
    if z1_after_kop_m <= SMALL:
        return None

    # Aggregate build displacement to t1 for BUILD1 (0->Ih) + BUILD2 (Ih->Ientry):
    # ds_build_total = R*(1-cos(Ientry)), dz_build_total = R*sin(Ientry)
    a_m = geometry.s1_m - radius_m * (1.0 - np.cos(inc_entry_rad))
    b_m = z1_after_kop_m - radius_m * np.sin(inc_entry_rad)
    if a_m <= SMALL or b_m <= SMALL:
        return None

    inc_hold_rad = float(np.arctan2(a_m, b_m))
    if inc_hold_rad <= SMALL or inc_hold_rad >= inc_entry_rad - SMALL:
        return None

    hold_length_m = float(np.hypot(a_m, b_m))
    build1_length_m = float(radius_m * inc_hold_rad)
    build2_length_m = float(radius_m * (inc_entry_rad - inc_hold_rad))
    if hold_length_m <= SMALL or build1_length_m <= SMALL or build2_length_m <= SMALL:
        return None

    return ProfileParameters(
        kop_vertical_m=float(kop_vertical_m),
        inc_entry_deg=float(geometry.inc_entry_deg),
        inc_hold_deg=float(inc_hold_rad * RAD2DEG),
        dls_build1_deg_per_30m=float(dls_build_deg_per_30m),
        dls_build2_deg_per_30m=float(dls_build_deg_per_30m),
        build1_length_m=build1_length_m,
        hold_length_m=hold_length_m,
        build2_length_m=build2_length_m,
        horizontal_length_m=horizontal_length_m,
        azimuth_deg=float(geometry.azimuth_deg),
    )


def _iter_kop_candidates(geometry: SectionGeometry, config: TrajectoryConfig) -> np.ndarray:
    kop_min = float(max(config.kop_min_vertical_m, 0.0))
    kop_max = float(geometry.z1_m - SMALL)
    if kop_min >= kop_max:
        return np.array([], dtype=float)

    grid_size = int(max(config.kop_search_grid_size, 2))
    if grid_size == 2:
        return np.array([kop_min, kop_max], dtype=float)
    return np.linspace(kop_min, kop_max, grid_size, dtype=float)


def _candidate_rank(candidate: ProfileParameters, objective_mode: str) -> tuple[float, float, float]:
    if objective_mode == OBJECTIVE_MINIMIZE_BUILD_DLS:
        return (
            candidate.dls_build1_deg_per_30m,
            -candidate.hold_length_m,
            -candidate.kop_vertical_m,
        )

    return (
        -candidate.hold_length_m,
        candidate.dls_build1_deg_per_30m,
        -candidate.kop_vertical_m,
    )


def _select_better_candidate(
    current: ProfileParameters | None,
    candidate: ProfileParameters,
    objective_mode: str,
) -> ProfileParameters:
    if current is None:
        return candidate
    if _candidate_rank(candidate, objective_mode) < _candidate_rank(current, objective_mode):
        return candidate
    return current


def _optimize_profile_for_objective(
    lower_dls: float,
    upper_dls: float,
    geometry: SectionGeometry,
    kop_vertical_m: float,
    config: TrajectoryConfig,
) -> ProfileParameters | None:
    if config.objective_mode == OBJECTIVE_MINIMIZE_BUILD_DLS:
        return _optimize_minimize_build_dls(
            lower_dls=lower_dls,
            upper_dls=upper_dls,
            geometry=geometry,
            kop_vertical_m=kop_vertical_m,
            config=config,
        )

    return _optimize_maximize_hold(
        lower_dls=lower_dls,
        upper_dls=upper_dls,
        geometry=geometry,
        kop_vertical_m=kop_vertical_m,
        config=config,
    )


def _optimize_maximize_hold(
    lower_dls: float,
    upper_dls: float,
    geometry: SectionGeometry,
    kop_vertical_m: float,
    config: TrajectoryConfig,
) -> ProfileParameters | None:
    def objective(dls_build: float) -> float:
        candidate = _profile_for_dls(
            geometry=geometry,
            dls_build_deg_per_30m=dls_build,
            kop_vertical_m=kop_vertical_m,
        )
        if not _is_candidate_feasible(candidate=candidate, config=config):
            return 1e12 + abs(dls_build - lower_dls)
        return -candidate.hold_length_m

    optimum = minimize_scalar(objective, bounds=(lower_dls, upper_dls), method="bounded", options={"xatol": 1e-5})
    probes = [lower_dls, upper_dls, float(optimum.x), float(optimum.x) - 1e-3, float(optimum.x) + 1e-3]

    best: ProfileParameters | None = None
    best_hold = -1.0
    for dls in probes:
        dls_clamped = float(np.clip(dls, lower_dls, upper_dls))
        candidate = _profile_for_dls(
            geometry=geometry,
            dls_build_deg_per_30m=dls_clamped,
            kop_vertical_m=kop_vertical_m,
        )
        if not _is_candidate_feasible(candidate=candidate, config=config):
            continue
        if candidate.hold_length_m > best_hold:
            best = candidate
            best_hold = candidate.hold_length_m
    return best


def _optimize_minimize_build_dls(
    lower_dls: float,
    upper_dls: float,
    geometry: SectionGeometry,
    kop_vertical_m: float,
    config: TrajectoryConfig,
) -> ProfileParameters | None:
    lower_candidate = _profile_for_dls(
        geometry=geometry,
        dls_build_deg_per_30m=lower_dls,
        kop_vertical_m=kop_vertical_m,
    )
    if _is_candidate_feasible(candidate=lower_candidate, config=config):
        return lower_candidate

    grid = np.linspace(lower_dls, upper_dls, 400, dtype=float)
    candidates: list[ProfileParameters | None] = [
        _profile_for_dls(
            geometry=geometry,
            dls_build_deg_per_30m=float(dls),
            kop_vertical_m=kop_vertical_m,
        )
        for dls in grid
    ]
    feasible_mask = [bool(_is_candidate_feasible(candidate=cand, config=config)) for cand in candidates]
    try:
        first_feasible_idx = feasible_mask.index(True)
    except ValueError:
        return None

    if first_feasible_idx == 0:
        refined_candidate = candidates[0]
        return refined_candidate if isinstance(refined_candidate, ProfileParameters) else None

    lo = float(grid[first_feasible_idx - 1])
    hi = float(grid[first_feasible_idx])
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        mid_candidate = _profile_for_dls(
            geometry=geometry,
            dls_build_deg_per_30m=mid,
            kop_vertical_m=kop_vertical_m,
        )
        if _is_candidate_feasible(candidate=mid_candidate, config=config):
            hi = mid
        else:
            lo = mid

    candidate = _profile_for_dls(
        geometry=geometry,
        dls_build_deg_per_30m=hi,
        kop_vertical_m=kop_vertical_m,
    )
    if _is_candidate_feasible(candidate=candidate, config=config):
        return candidate
    return None


def _is_candidate_feasible(candidate: ProfileParameters | None, config: TrajectoryConfig) -> bool:
    if candidate is None:
        return False
    return bool(candidate.md_total_m <= config.max_total_md_m + SMALL)


def _required_dls_for_t1_reach(s1_m: float, z1_m: float, inc_entry_deg: float) -> float:
    inc_entry_rad = inc_entry_deg * DEG2RAD
    one_minus_cos = max(1.0 - np.cos(inc_entry_rad), SMALL)
    sin_entry = max(np.sin(inc_entry_rad), SMALL)
    radius_limit_m = min(s1_m / one_minus_cos, z1_m / sin_entry)
    return _dls_from_radius(radius_limit_m)


def _build_trajectory(params: ProfileParameters) -> WellTrajectory:
    segments = [
        VerticalSegment(
            length_m=params.kop_vertical_m,
            azi_deg=params.azimuth_deg,
            name="VERTICAL",
        ),
        BuildSegment(
            inc_from_deg=0.0,
            inc_to_deg=params.inc_hold_deg,
            dls_deg_per_30m=params.dls_build1_deg_per_30m,
            azi_deg=params.azimuth_deg,
            name="BUILD1",
        ),
        HoldSegment(
            length_m=params.hold_length_m,
            inc_deg=params.inc_hold_deg,
            azi_deg=params.azimuth_deg,
            name="HOLD",
        ),
        BuildSegment(
            inc_from_deg=params.inc_hold_deg,
            inc_to_deg=params.inc_entry_deg,
            dls_deg_per_30m=params.dls_build2_deg_per_30m,
            azi_deg=params.azimuth_deg,
            name="BUILD2",
        ),
        HorizontalSegment(
            length_m=params.horizontal_length_m,
            inc_deg=params.inc_entry_deg,
            azi_deg=params.azimuth_deg,
            name="HORIZONTAL",
        ),
    ]
    return WellTrajectory(segments=segments)


def _build_summary(
    df: pd.DataFrame,
    t1: Point3D,
    t3: Point3D,
    md_t1_m: float,
    kop_vertical_m: float,
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
        "kop_vertical_m": float(kop_vertical_m),
        "kop_md_m": float(kop_vertical_m),
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
    if config.kop_min_vertical_m < 0.0:
        raise PlanningError("kop_min_vertical_m must be non-negative.")
    if config.kop_search_grid_size < 2:
        raise PlanningError("kop_search_grid_size must be >= 2.")
    if config.entry_inc_tolerance_deg < 0.0:
        raise PlanningError("entry_inc_tolerance_deg must be non-negative.")
    if config.dls_build_min_deg_per_30m > config.dls_build_max_deg_per_30m:
        raise PlanningError("dls_build_min_deg_per_30m cannot exceed dls_build_max_deg_per_30m.")
    if config.max_total_md_m <= 0.0:
        raise PlanningError("max_total_md_m must be positive.")
    if config.objective_mode not in ALLOWED_OBJECTIVE_MODES:
        allowed = ", ".join(ALLOWED_OBJECTIVE_MODES)
        raise PlanningError(f"objective_mode must be one of: {allowed}.")
    for segment, limit in config.dls_limits_deg_per_30m.items():
        if limit < 0.0:
            raise PlanningError(f"DLS limit for segment {segment} cannot be negative.")


def _azimuth_deg_from_points(t1: Point3D, t3: Point3D) -> float:
    dn = t3.y - t1.y
    de = t3.x - t1.x
    if np.isclose(dn, 0.0) and np.isclose(de, 0.0):
        raise PlanningError("t1 and t3 overlap in plan; azimuth is undefined.")
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
    # Inclination is measured from vertical: tan(INC) = horizontal / vertical.
    return float(np.degrees(np.arctan2(ds_13, dz_13)))


def _dls_from_radius(radius_m: float) -> float:
    return float(30.0 * RAD2DEG / radius_m)


def _radius_from_dls(dls_deg_per_30m: float) -> float:
    return float(30.0 * RAD2DEG / dls_deg_per_30m)


def _distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))
