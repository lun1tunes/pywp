from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pywp.mcm import add_dls, compute_positions_min_curv
from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.segments import BuildSegment, HoldSegment, HorizontalSegment, VerticalSegment
from pywp.trajectory import WellTrajectory

DEG2RAD = np.pi / 180.0


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProfileParameters:
    inc_entry_deg: float
    dls_build1_deg_per_30m: float
    dls_build2_deg_per_30m: float
    vertical_length_m: float
    hold1_length_m: float
    hold2_length_m: float
    horizontal_length_m: float
    azimuth_deg: float

    @property
    def build1_length_m(self) -> float:
        return abs(self.inc_entry_deg) / self.dls_build1_deg_per_30m * 30.0

    @property
    def build2_length_m(self) -> float:
        return abs(90.0 - self.inc_entry_deg) / self.dls_build2_deg_per_30m * 30.0


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
                "Точки не лежат в одной вертикальной плоскости относительно устья. "
                "Для такого набора нужен TURN-участок или доп. контрольные точки."
            )

        opt_result = self._optimize_profile(
            s1=s1,
            z1=z1,
            s3=s3,
            z3=z3,
            cross1=c1,
            cross3=c3,
            config=config,
        )

        params = _profile_from_vector(opt_result.x, azimuth_deg=azimuth_deg)
        trajectory = _build_trajectory(params)
        md_t1_m = params.vertical_length_m + params.build1_length_m + params.hold1_length_m

        control = compute_positions_min_curv(
            trajectory.stations(md_step_m=config.md_step_control_m), start=surface
        )
        control = add_dls(control)

        summary = _build_summary(
            df=control,
            t1=t1,
            t3=t3,
            md_t1_m=md_t1_m,
            config=config,
        )
        _assert_solution_is_valid(summary=summary, config=config)

        output = compute_positions_min_curv(trajectory.stations(md_step_m=config.md_step_m), start=surface)
        output = add_dls(output)

        return PlannerResult(stations=output, summary=summary, azimuth_deg=azimuth_deg, md_t1_m=md_t1_m)

    def _optimize_profile(
        self,
        s1: float,
        z1: float,
        s3: float,
        z3: float,
        cross1: float,
        cross3: float,
        config: TrajectoryConfig,
    ):
        if s3 <= s1:
            raise PlanningError("t3 должен лежать дальше t1 по направлению горизонтального участка.")

        x0 = _initial_guess(s1=s1, z1=z1, s3=s3, z3=z3, config=config)

        lower = np.array(
            [
                config.inc_entry_min_deg,
                config.dls_build1_min_deg_per_30m,
                config.dls_build2_min_deg_per_30m,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        upper = np.array(
            [
                config.inc_entry_max_deg,
                config.dls_build1_max_deg_per_30m,
                config.dls_build2_max_deg_per_30m,
                config.max_total_md_m,
                config.max_total_md_m,
                config.max_total_md_m,
                config.max_total_md_m,
            ],
            dtype=float,
        )

        def objective(x: np.ndarray) -> np.ndarray:
            pred = _predict_geometry(x)
            pos_scale = max(config.pos_tolerance_m, 1e-6)
            md_scale = max(config.max_total_md_m, 1.0)

            residuals = [
                (pred["s_t1"] - s1) / pos_scale,
                (pred["z_t1"] - z1) / pos_scale,
                (pred["s_t3"] - s3) / pos_scale,
                (pred["z_t3"] - z3) / pos_scale,
                cross1 / pos_scale,
                cross3 / pos_scale,
                (x[3] + x[4] + x[5] + x[6]) / md_scale,
            ]
            return np.asarray(residuals, dtype=float)

        try:
            result = least_squares(objective, x0=x0, bounds=(lower, upper), method="trf")
        except ValueError as exc:
            raise PlanningError(f"Некорректные параметры оптимизации: {exc}") from exc
        if not result.success:
            raise PlanningError(f"Оптимизатор не сошёлся: {result.message}")

        final_residual = np.linalg.norm(objective(result.x)[:4])
        if final_residual > 3.0:
            raise PlanningError("Не удалось получить траекторию в заданных допусках.")

        return result


def _initial_guess(s1: float, z1: float, s3: float, z3: float, config: TrajectoryConfig) -> np.ndarray:
    inc = (config.inc_entry_min_deg + config.inc_entry_max_deg) / 2.0
    dls1 = (config.dls_build1_min_deg_per_30m + config.dls_build1_max_deg_per_30m) / 2.0
    dls2 = (config.dls_build2_min_deg_per_30m + config.dls_build2_max_deg_per_30m) / 2.0

    i = inc * DEG2RAD
    r1 = _radius_from_dls(dls1)
    r2 = _radius_from_dls(dls2)

    ds_b1 = r1 * (1.0 - np.cos(i))
    dz_b1 = r1 * np.sin(i)
    hold1 = max((s1 - ds_b1) / max(np.sin(i), 1e-6), 0.0)
    vertical = max(z1 - dz_b1 - hold1 * np.cos(i), 0.0)

    ds_b2 = r2 * np.cos(i)
    dz_b2 = r2 * (1.0 - np.sin(i))
    hold2 = max((z3 - z1 - dz_b2) / max(np.cos(i), 1e-6), 0.0)
    horizontal = max((s3 - s1 - hold2 * np.sin(i) - ds_b2), 0.0)

    return np.array([inc, dls1, dls2, vertical, hold1, hold2, horizontal], dtype=float)


def _profile_from_vector(x: np.ndarray, azimuth_deg: float) -> ProfileParameters:
    return ProfileParameters(
        inc_entry_deg=float(x[0]),
        dls_build1_deg_per_30m=float(x[1]),
        dls_build2_deg_per_30m=float(x[2]),
        vertical_length_m=float(x[3]),
        hold1_length_m=float(x[4]),
        hold2_length_m=float(x[5]),
        horizontal_length_m=float(x[6]),
        azimuth_deg=float(azimuth_deg),
    )


def _build_trajectory(params: ProfileParameters) -> WellTrajectory:
    segments = [
        VerticalSegment(length_m=params.vertical_length_m, azi_deg=params.azimuth_deg, name="VERTICAL"),
        BuildSegment(
            inc_from_deg=0.0,
            inc_to_deg=params.inc_entry_deg,
            dls_deg_per_30m=params.dls_build1_deg_per_30m,
            azi_deg=params.azimuth_deg,
            name="BUILD1",
        ),
        HoldSegment(
            length_m=params.hold1_length_m,
            inc_deg=params.inc_entry_deg,
            azi_deg=params.azimuth_deg,
            name="HOLD1",
        ),
        HoldSegment(
            length_m=params.hold2_length_m,
            inc_deg=params.inc_entry_deg,
            azi_deg=params.azimuth_deg,
            name="HOLD2",
        ),
        BuildSegment(
            inc_from_deg=params.inc_entry_deg,
            inc_to_deg=90.0,
            dls_deg_per_30m=params.dls_build2_deg_per_30m,
            azi_deg=params.azimuth_deg,
            name="BUILD2",
        ),
        HorizontalSegment(length_m=params.horizontal_length_m, azi_deg=params.azimuth_deg, name="HORIZONTAL"),
    ]
    return WellTrajectory(segments=segments)


def _predict_geometry(x: np.ndarray) -> dict[str, float]:
    inc_entry_deg, dls_b1, dls_b2, vertical, hold1, hold2, horizontal = x
    i = inc_entry_deg * DEG2RAD

    r1 = _radius_from_dls(dls_b1)
    r2 = _radius_from_dls(dls_b2)

    ds_b1 = r1 * (1.0 - np.cos(i))
    dz_b1 = r1 * np.sin(i)

    s_t1 = ds_b1 + hold1 * np.sin(i)
    z_t1 = vertical + dz_b1 + hold1 * np.cos(i)

    ds_b2 = r2 * np.cos(i)
    dz_b2 = r2 * (1.0 - np.sin(i))

    s_t3 = s_t1 + hold2 * np.sin(i) + ds_b2 + horizontal
    z_t3 = z_t1 + hold2 * np.cos(i) + dz_b2

    return {
        "s_t1": float(s_t1),
        "z_t1": float(z_t1),
        "s_t3": float(s_t3),
        "z_t3": float(z_t3),
    }


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
        raise PlanningError("Не удалось попасть в t1 в заданном допуске.")

    if summary["distance_t3_m"] > config.pos_tolerance_m:
        raise PlanningError("Не удалось попасть в t3 в заданном допуске.")

    if summary["entry_inc_deg"] > config.inc_entry_max_deg + config.angle_tolerance_deg:
        raise PlanningError("Нарушено ограничение на угол входа в пласт.")

    for segment, limit in config.dls_limits_deg_per_30m.items():
        actual = summary.get(f"max_dls_{segment.lower()}_deg_per_30m", 0.0)
        if actual > limit + 1e-6:
            raise PlanningError(f"Превышен DLS лимит на участке {segment}: {actual:.2f} > {limit:.2f}")


def _azimuth_deg_from_points(t1: Point3D, t3: Point3D) -> float:
    dn = t3.y - t1.y
    de = t3.x - t1.x
    if np.isclose(dn, 0.0) and np.isclose(de, 0.0):
        raise PlanningError("t1 и t3 совпадают в плане, невозможно определить азимут латерали.")
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


def _radius_from_dls(dls_deg_per_30m: float) -> float:
    dls = float(max(dls_deg_per_30m, 1e-9))
    return 30.0 * 180.0 / (np.pi * dls)


def _distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))


def _validate_config(config: TrajectoryConfig) -> None:
    if config.md_step_m <= 0.0 or config.md_step_control_m <= 0.0:
        raise PlanningError("Шаги MD должны быть положительными.")

    if config.pos_tolerance_m <= 0.0:
        raise PlanningError("Допуск по позиции должен быть положительным.")

    if config.inc_entry_min_deg > config.inc_entry_max_deg:
        raise PlanningError("inc_entry_min_deg не может быть больше inc_entry_max_deg.")

    if config.dls_build1_min_deg_per_30m > config.dls_build1_max_deg_per_30m:
        raise PlanningError("dls_build1_min_deg_per_30m не может быть больше dls_build1_max_deg_per_30m.")

    if config.dls_build2_min_deg_per_30m > config.dls_build2_max_deg_per_30m:
        raise PlanningError("dls_build2_min_deg_per_30m не может быть больше dls_build2_max_deg_per_30m.")

    for segment, limit in config.dls_limits_deg_per_30m.items():
        if limit < 0.0:
            raise PlanningError(f"DLS limit для участка {segment} не может быть отрицательным.")
