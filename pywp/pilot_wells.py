from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    evaluate_stations_anti_collision_clearance,
)
from pywp.constants import SMALL
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.mcm import (
    add_dls,
    compute_positions_min_curv,
    dls_deg_per_30m,
    dogleg_angle_rad,
    minimum_curvature_increment,
)
from pywp.models import PlannerResult, Point3D, SummaryDict, TrajectoryConfig
from pywp.planner_types import PlanningError
from pywp.pydantic_base import FrozenArbitraryModel
from pywp.segments import BuildSegment, HoldSegment
from pywp.sidetrack_solver import SidetrackPlanner, SidetrackStart
from pywp.trajectory import WellTrajectory
from pywp.ui_utils import dls_to_pi

PILOT_SUFFIX = "_PL"
SIDETRACK_WINDOW_ABOVE_FIRST_TARGET_MIN_M = 50.0
SIDETRACK_WINDOW_ABOVE_FIRST_TARGET_MAX_M = 100.0


class PilotWindow(FrozenArbitraryModel):
    pilot_name: str
    parent_name: str
    md_m: float
    point: Point3D
    inc_deg: float
    azi_deg: float

    @classmethod
    def from_station(
        cls,
        *,
        pilot_name: str,
        parent_name: str,
        row: pd.Series,
    ) -> "PilotWindow":
        return cls(
            pilot_name=str(pilot_name),
            parent_name=str(parent_name),
            md_m=float(row["MD_m"]),
            point=Point3D(
                x=float(row["X_m"]),
                y=float(row["Y_m"]),
                z=float(row["Z_m"]),
            ),
            inc_deg=float(row["INC_deg"]),
            azi_deg=float(row["AZI_deg"]),
        )


@dataclass(frozen=True)
class SidetrackWindowOverride:
    kind: Literal["md", "z"]
    value_m: float

    def __post_init__(self) -> None:
        normalized_kind = str(self.kind).strip().lower()
        if normalized_kind not in {"md", "z"}:
            raise ValueError("Тип ручного окна зарезки должен быть 'md' или 'z'.")
        if not math.isfinite(float(self.value_m)):
            raise ValueError("Значение ручного окна зарезки должно быть конечным.")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "value_m", float(self.value_m))


@dataclass(frozen=True)
class PilotBuildResult:
    stations: pd.DataFrame
    surface: Point3D
    first_target: Point3D
    final_target: Point3D
    md_first_target_m: float
    md_total_m: float
    azimuth_deg: float
    summary: SummaryDict


@dataclass(frozen=True)
class SidetrackPlan:
    result: PlannerResult
    window: PilotWindow
    stations: pd.DataFrame
    summary: SummaryDict
    md_t1_m: float
    azimuth_deg: float


def is_pilot_name(name: object) -> bool:
    return str(name).strip().upper().endswith(PILOT_SUFFIX)


def parent_name_for_pilot(name: object) -> str:
    text = str(name).strip()
    if not is_pilot_name(text):
        return text
    return text[: -len(PILOT_SUFFIX)]


def pilot_name_for_parent(name: object) -> str:
    return f"{str(name).strip()}{PILOT_SUFFIX}"


def well_name_key(name: object) -> str:
    return str(name).strip().casefold()


def pilot_name_key_for_parent(name: object) -> str:
    return well_name_key(pilot_name_for_parent(name))


def is_pilot_record(record: WelltrackRecord) -> bool:
    return is_pilot_name(record.name)


def visible_well_records(
    records: Iterable[WelltrackRecord],
) -> list[WelltrackRecord]:
    return [record for record in records if not is_pilot_record(record)]


def visible_well_names(records: Iterable[WelltrackRecord]) -> list[str]:
    return [str(record.name) for record in visible_well_records(records)]


def sync_pilot_surfaces_to_parents(
    records: Iterable[WelltrackRecord],
) -> list[WelltrackRecord]:
    record_list = list(records)
    parent_by_key = {
        well_name_key(record.name): record
        for record in record_list
        if not is_pilot_record(record)
    }
    synced: list[WelltrackRecord] = []
    for record in record_list:
        if not is_pilot_record(record) or not record.points:
            synced.append(record)
            continue
        parent = parent_by_key.get(well_name_key(parent_name_for_pilot(record.name)))
        if parent is None or not parent.points:
            synced.append(record)
            continue
        parent_surface = parent.points[0]
        pilot_surface = record.points[0]
        synced_surface = WelltrackPoint(
            x=float(parent_surface.x),
            y=float(parent_surface.y),
            z=float(parent_surface.z),
            md=float(pilot_surface.md),
        )
        if synced_surface == pilot_surface:
            synced.append(record)
            continue
        synced.append(
            WelltrackRecord(
                name=record.name,
                points=(synced_surface, *tuple(record.points[1:])),
            )
        )
    return synced


def order_records_with_pilots_first(
    records: Iterable[WelltrackRecord],
) -> list[WelltrackRecord]:
    ordered = list(records)
    by_name = {well_name_key(record.name): record for record in ordered}
    result: list[WelltrackRecord] = []
    seen: set[str] = set()

    for record in ordered:
        name = str(record.name)
        name_key = well_name_key(name)
        if name_key in seen:
            continue
        if is_pilot_record(record):
            result.append(record)
            seen.add(name_key)
            continue
        pilot = by_name.get(pilot_name_key_for_parent(name))
        if pilot is not None and well_name_key(pilot.name) not in seen:
            result.append(pilot)
            seen.add(well_name_key(pilot.name))
        result.append(record)
        seen.add(name_key)

    return result


def pilot_record_problem_text(record: WelltrackRecord) -> str:
    points = tuple(record.points)
    if len(points) < 2:
        return "Для пилота требуется `S` и минимум одна точка изучения."
    if not _points_have_finite_xyz(points):
        return "Координаты X/Y/Z и MD должны быть конечными числами."
    if _has_zero_length_leg(points):
        return "Пилот содержит совпадающие соседние точки."
    return "—"


def build_pilot_trajectory(
    record: WelltrackRecord,
    *,
    config: TrajectoryConfig,
) -> PilotBuildResult:
    problem = pilot_record_problem_text(record)
    if problem != "—":
        raise ValueError(problem)

    points = tuple(record.points)
    surface_point = points[0]
    study_points = points[1:]
    kop_point = _pilot_kop_point(
        surface=surface_point,
        first_target=study_points[0],
        config=config,
    )
    vertical_stations = _vertical_pilot_stations(
        surface=surface_point,
        kop=kop_point,
        config=config,
    )
    directional_stations, md_first_target_m = _directional_pilot_stations(
        kop=kop_point,
        study_points=study_points,
        start_md_m=float(vertical_stations["MD_m"].iloc[-1]),
        config=config,
    )
    stations = pd.concat(
        [vertical_stations, directional_stations.iloc[1:].copy()],
        ignore_index=True,
    )
    stations = add_dls(stations)
    md_values = stations["MD_m"].to_numpy(dtype=float)
    max_dls = _finite_max(stations.get("DLS_deg_per_30m", pd.Series(dtype=float)))
    surface = Point3D(x=surface_point.x, y=surface_point.y, z=surface_point.z)
    first_target = Point3D(
        x=study_points[0].x, y=study_points[0].y, z=study_points[0].z
    )
    final_target = Point3D(x=points[-1].x, y=points[-1].y, z=points[-1].z)
    md_total_m = float(md_values[-1])
    summary: SummaryDict = {
        "trajectory_type": "PILOT",
        "trajectory_target_direction": "Пилотный ствол",
        "well_complexity": "Пилот",
        "horizontal_length_m": 0.0,
        "entry_inc_deg": 0.0,
        "hold_inc_deg": 0.0,
        "max_dls_total_deg_per_30m": max_dls,
        "md_total_m": md_total_m,
        "max_total_md_postcheck_m": float(config.max_total_md_postcheck_m),
        "md_postcheck_excess_m": max(
            0.0,
            md_total_m - float(config.max_total_md_postcheck_m),
        ),
        "dls_postcheck_excess_deg_per_30m": max(
            0.0,
            max_dls - float(config.dls_build_max_deg_per_30m),
        ),
        "solver_turn_restarts_used": 0.0,
        "solver_turn_max_restarts": float(config.turn_solver_max_restarts),
        "pilot_target_count": float(max(len(points) - 1, 0)),
        "kop_md_m": float(vertical_stations["MD_m"].iloc[-1]),
    }
    return PilotBuildResult(
        stations=stations,
        surface=surface,
        first_target=first_target,
        final_target=final_target,
        md_first_target_m=float(md_first_target_m),
        md_total_m=md_total_m,
        azimuth_deg=_first_valid_azimuth_deg(stations),
        summary=summary,
    )


def select_sidetrack_window(
    *,
    pilot_name: str,
    parent_name: str,
    pilot_stations: pd.DataFrame,
    parent_t1: Point3D,
    parent_t3: Point3D,
    config: TrajectoryConfig,
    planner: object,
    optimization_context: AntiCollisionOptimizationContext | None = None,
    window_override: SidetrackWindowOverride | None = None,
) -> tuple[PilotWindow, PlannerResult]:
    del planner
    sidetrack_planner = SidetrackPlanner()
    if window_override is not None:
        window = _manual_sidetrack_window(
            pilot_name=pilot_name,
            parent_name=parent_name,
            pilot_stations=pilot_stations,
            override=window_override,
        )
        try:
            return window, sidetrack_planner.plan(
                start=SidetrackStart(
                    point=window.point,
                    inc_deg=float(window.inc_deg),
                    azi_deg=float(window.azi_deg),
                ),
                t1=parent_t1,
                t3=parent_t3,
                config=config,
            )
        except (ValueError, PlanningError) as exc:
            coordinate_label = "MD" if window_override.kind == "md" else "Z"
            raise ValueError(
                f"Ручное окно зарезки {parent_name} по {coordinate_label}="
                f"{window_override.value_m:.2f} м не дало расчет бокового ствола: {exc}"
            ) from exc

    candidates = _sidetrack_window_candidates(
        pilot_name=pilot_name,
        parent_name=parent_name,
        pilot_stations=pilot_stations,
        parent_t1=parent_t1,
        config=config,
    )
    last_problem = ""
    best: tuple[float, float, PilotWindow, PlannerResult] | None = None
    for window in candidates:
        try:
            result = sidetrack_planner.plan(
                start=SidetrackStart(
                    point=window.point,
                    inc_deg=float(window.inc_deg),
                    azi_deg=float(window.azi_deg),
                ),
                t1=parent_t1,
                t3=parent_t3,
                config=config,
            )
        except (ValueError, PlanningError) as exc:
            last_problem = str(exc)
            continue
        score = _sidetrack_window_score(
            window=window,
            result=result,
            optimization_context=optimization_context,
        )
        if best is None or (score, -float(window.md_m)) < (best[0], best[1]):
            best = (score, -float(window.md_m), window, result)

    if best is not None:
        _, _, window, result = best
        return window, result

    suffix = f" Последняя причина: {last_problem}" if last_problem else ""
    raise ValueError(
        "Не удалось подобрать окно зарезки на пилоте: ни одна станция пилота "
        "не дала расчет продуктивного ствола до t1/t3." + suffix
    )


def combine_pilot_and_sidetrack(
    *,
    pilot_stations: pd.DataFrame,
    sidetrack_result: PlannerResult,
    window: PilotWindow,
    config: TrajectoryConfig,
) -> SidetrackPlan:
    window_md = float(window.md_m)
    pilot_upper = pilot_stations.loc[
        pilot_stations["MD_m"].to_numpy(dtype=float) <= window_md + SMALL
    ].copy()
    if pilot_upper.empty:
        raise ValueError("Не удалось собрать общий участок пилота до окна зарезки.")

    sidetrack_stations = sidetrack_result.stations.copy()
    if sidetrack_stations.empty:
        raise ValueError("Расчет бокового ствола вернул пустую инклинометрию.")
    sidetrack_stations["MD_m"] = (
        sidetrack_stations["MD_m"].to_numpy(dtype=float) + window_md
    )
    sidetrack_stations["segment"] = [
        _sidetrack_segment_label(value)
        for value in sidetrack_stations.get(
            "segment", pd.Series(["SIDETRACK"] * len(sidetrack_stations))
        )
    ]
    stations = sidetrack_stations.sort_values("MD_m").reset_index(drop=True)
    stations = add_dls(stations)
    stations.attrs["uncertainty_reference_stations"] = pilot_upper.copy()
    summary = dict(sidetrack_result.summary)
    md_total_m = float(stations["MD_m"].iloc[-1])
    max_dls = max(
        float(summary.get("max_dls_total_deg_per_30m", 0.0)),
        _finite_max(stations.get("DLS_deg_per_30m", pd.Series(dtype=float))),
    )
    summary.update(
        {
            "trajectory_type": "PILOT_SIDETRACK",
            "pilot_well_name": str(window.pilot_name),
            "sidetrack_window_md_m": window_md,
            "sidetrack_window_x_m": float(window.point.x),
            "sidetrack_window_y_m": float(window.point.y),
            "sidetrack_window_z_m": float(window.point.z),
            "sidetrack_window_inc_deg": float(window.inc_deg),
            "sidetrack_window_azi_deg": float(window.azi_deg),
            "sidetrack_lateral_md_m": float(
                sidetrack_result.summary.get("md_total_m", 0.0)
            ),
            "md_total_m": md_total_m,
            "max_total_md_postcheck_m": float(config.max_total_md_postcheck_m),
            "md_postcheck_excess_m": max(
                0.0,
                md_total_m - float(config.max_total_md_postcheck_m),
            ),
            "max_dls_total_deg_per_30m": max_dls,
            "dls_postcheck_excess_deg_per_30m": max(
                0.0,
                max_dls - float(config.dls_build_max_deg_per_30m),
            ),
            "kop_md_m": window_md + float(summary.get("kop_md_m", 0.0)),
        }
    )
    return SidetrackPlan(
        result=sidetrack_result,
        window=window,
        stations=stations,
        summary=summary,
        md_t1_m=window_md + float(sidetrack_result.md_t1_m),
        azimuth_deg=float(sidetrack_result.azimuth_deg),
    )


def paired_pilot_parent_names(name_a: object, name_b: object) -> bool:
    left = str(name_a).strip()
    right = str(name_b).strip()
    return (
        is_pilot_name(left)
        and well_name_key(parent_name_for_pilot(left)) == well_name_key(right)
        or is_pilot_name(right)
        and well_name_key(parent_name_for_pilot(right)) == well_name_key(left)
    )


def _pilot_kop_point(
    *,
    surface: WelltrackPoint,
    first_target: WelltrackPoint,
    config: TrajectoryConfig,
) -> Point3D:
    vertical_room = float(first_target.z) - float(surface.z)
    min_directional_room = max(
        float(config.min_structural_segment_m),
        float(config.md_step_m),
    )
    if vertical_room <= min_directional_room + SMALL:
        raise ValueError(
            "Для пилота первая точка изучения должна быть ниже устья "
            "с запасом под VERTICAL и BUILD."
        )
    requested_kop_m = max(
        float(config.kop_min_vertical_m),
        float(config.min_structural_segment_m),
    )
    kop_vertical_m = min(requested_kop_m, vertical_room - min_directional_room)
    if kop_vertical_m <= SMALL:
        raise ValueError("Не удалось выделить вертикальный участок пилота до KOP.")
    return Point3D(
        x=float(surface.x),
        y=float(surface.y),
        z=float(surface.z) + float(kop_vertical_m),
    )


def _vertical_pilot_stations(
    *,
    surface: WelltrackPoint,
    kop: Point3D,
    config: TrajectoryConfig,
) -> pd.DataFrame:
    length_m = float(kop.z) - float(surface.z)
    if length_m <= SMALL:
        raise ValueError("Не удалось выделить вертикальный участок пилота до KOP.")
    samples = max(int(math.ceil(length_m / max(float(config.md_step_m), 1.0))), 1)
    fractions = np.linspace(0.0, 1.0, samples + 1)
    stations = pd.DataFrame(
        {
            "MD_m": length_m * fractions,
            "INC_deg": [0.0] * len(fractions),
            "AZI_deg": [0.0] * len(fractions),
            "X_m": [float(surface.x)] * len(fractions),
            "Y_m": [float(surface.y)] * len(fractions),
            "Z_m": float(surface.z) + length_m * fractions,
            "segment": ["VERTICAL"] * len(fractions),
        }
    )
    return add_dls(stations)


def _directional_pilot_stations(
    *,
    kop: Point3D,
    study_points: tuple[WelltrackPoint, ...],
    start_md_m: float,
    config: TrajectoryConfig,
) -> tuple[pd.DataFrame, float]:
    parts: list[pd.DataFrame] = []
    current_point = kop
    current_md = float(start_md_m)
    current_inc = 0.0
    current_azi = _azimuth_between(kop, _point_from_welltrack(study_points[0]))
    md_first_target_m = float("nan")

    for index, point in enumerate(study_points, start=1):
        target = _point_from_welltrack(point)
        leg = _pilot_build_hold_leg_to_target(
            start=current_point,
            target=target,
            start_md_m=current_md,
            start_inc_deg=current_inc,
            start_azi_deg=current_azi,
            segment_index=index,
            config=config,
        )
        append_leg = leg if not parts else leg.iloc[1:].copy()
        parts.append(append_leg)
        current_point = target
        current_md = float(leg["MD_m"].iloc[-1])
        current_inc = float(leg["INC_deg"].iloc[-1])
        current_azi = float(leg["AZI_deg"].iloc[-1])
        if index == 1:
            md_first_target_m = current_md

    stations = pd.concat(parts, ignore_index=True)
    return add_dls(stations), md_first_target_m


def _pilot_build_hold_leg_to_target(
    *,
    start: Point3D,
    target: Point3D,
    start_md_m: float,
    start_inc_deg: float,
    start_azi_deg: float,
    segment_index: int,
    config: TrajectoryConfig,
) -> pd.DataFrame:
    target_vector = _point_array(target) - _point_array(start)
    target_distance = float(np.linalg.norm(target_vector))
    if target_distance <= SMALL:
        raise ValueError("Пилот содержит совпадающие соседние точки.")

    target_inc_deg, target_azi_deg = _angles_from_delta(target_vector)
    dls_limit = float(config.dls_build_max_deg_per_30m)
    if dls_limit <= SMALL:
        raise ValueError("Для пилота dls_build_max_deg_per_30m должен быть > 0.")

    def residual(values: np.ndarray) -> np.ndarray:
        inc_to_deg = float(values[0])
        azi_to_deg = float(values[1]) % 360.0
        hold_length_m = float(values[2])
        build_length_m = _build_length_m(
            inc_from_deg=start_inc_deg,
            azi_from_deg=start_azi_deg,
            inc_to_deg=inc_to_deg,
            azi_to_deg=azi_to_deg,
            dls_deg_per_30m=dls_limit,
        )
        build_delta = _minimum_curvature_delta_xyz(
            length_m=build_length_m,
            inc_from_deg=start_inc_deg,
            azi_from_deg=start_azi_deg,
            inc_to_deg=inc_to_deg,
            azi_to_deg=azi_to_deg,
        )
        hold_delta = hold_length_m * _unit_vector_xyz(
            inc_deg=inc_to_deg,
            azi_deg=azi_to_deg,
        )
        return build_delta + hold_delta - target_vector

    initial_hold = max(target_distance, 0.0)
    result = least_squares(
        residual,
        x0=np.asarray([target_inc_deg, target_azi_deg, initial_hold], dtype=float),
        bounds=(
            np.asarray([0.0, 0.0, 0.0], dtype=float),
            np.asarray(
                [
                    float(config.max_inc_deg),
                    360.0,
                    max(target_distance * 2.0, target_distance + 2000.0),
                ],
                dtype=float,
            ),
        ),
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=300,
    )
    miss_m = float(np.linalg.norm(residual(result.x)))
    tolerance_m = min(
        float(config.lateral_tolerance_m), float(config.vertical_tolerance_m)
    )
    if (not bool(result.success)) or miss_m > max(tolerance_m, 0.25):
        raise ValueError(
            "Не удалось построить буримый пилотный участок BUILD+HOLD до точки "
            f"p{int(segment_index)} при ПИ <= {dls_to_pi(dls_limit):.2f} deg/10m. "
            f"Остаточное отклонение {miss_m:.2f} м."
        )

    inc_to_deg = float(result.x[0])
    azi_to_deg = float(result.x[1]) % 360.0
    hold_length_m = float(max(result.x[2], 0.0))
    build = BuildSegment(
        inc_from_deg=float(start_inc_deg),
        inc_to_deg=inc_to_deg,
        dls_deg_per_30m=dls_limit,
        azi_deg=float(start_azi_deg),
        azi_to_deg=azi_to_deg,
        name=f"PILOT_BUILD_{int(segment_index)}",
        interpolation_method=str(config.interpolation_method),
    )
    hold = HoldSegment(
        length_m=hold_length_m,
        inc_deg=inc_to_deg,
        azi_deg=azi_to_deg,
        name=f"PILOT_HOLD_{int(segment_index)}",
    )
    survey = WellTrajectory([build, hold]).stations(md_step_m=float(config.md_step_m))
    survey["MD_m"] = survey["MD_m"].to_numpy(dtype=float) + float(start_md_m)
    stations = compute_positions_min_curv(survey, start=start)
    stations = add_dls(stations)
    final_index = stations.index[-1]
    stations.loc[final_index, "X_m"] = float(target.x)
    stations.loc[final_index, "Y_m"] = float(target.y)
    stations.loc[final_index, "Z_m"] = float(target.z)
    return stations


def _point_from_welltrack(point: WelltrackPoint) -> Point3D:
    return Point3D(x=float(point.x), y=float(point.y), z=float(point.z))


def _point_array(point: Point3D) -> np.ndarray:
    return np.asarray([float(point.x), float(point.y), float(point.z)], dtype=float)


def _angles_from_delta(delta_xyz: np.ndarray) -> tuple[float, float]:
    dx, dy, dz = (float(value) for value in delta_xyz)
    horizontal = float(math.hypot(dx, dy))
    if horizontal <= SMALL and abs(dz) <= SMALL:
        raise ValueError("Невозможно определить направление для совпадающих точек.")
    inc_deg = float(math.degrees(math.atan2(horizontal, dz)))
    azi_deg = (
        0.0
        if horizontal <= SMALL
        else _normalize_azimuth_deg(math.degrees(math.atan2(dx, dy)))
    )
    return inc_deg, azi_deg


def _build_length_m(
    *,
    inc_from_deg: float,
    azi_from_deg: float,
    inc_to_deg: float,
    azi_to_deg: float,
    dls_deg_per_30m: float,
) -> float:
    if float(dls_deg_per_30m) <= SMALL:
        return 0.0
    dogleg_deg = float(
        math.degrees(
            float(
                dogleg_angle_rad(
                    float(inc_from_deg),
                    float(azi_from_deg),
                    float(inc_to_deg),
                    float(azi_to_deg),
                )
            )
        )
    )
    return float(dogleg_deg / float(dls_deg_per_30m) * 30.0)


def _minimum_curvature_delta_xyz(
    *,
    length_m: float,
    inc_from_deg: float,
    azi_from_deg: float,
    inc_to_deg: float,
    azi_to_deg: float,
) -> np.ndarray:
    if float(length_m) <= SMALL:
        return np.zeros(3, dtype=float)
    north_m, east_m, z_m = minimum_curvature_increment(
        md1_m=0.0,
        inc1_deg=float(inc_from_deg),
        azi1_deg=float(azi_from_deg),
        md2_m=float(length_m),
        inc2_deg=float(inc_to_deg),
        azi2_deg=float(azi_to_deg),
    )
    return np.asarray([east_m, north_m, z_m], dtype=float)


def _unit_vector_xyz(*, inc_deg: float, azi_deg: float) -> np.ndarray:
    inc_rad = math.radians(float(inc_deg))
    azi_rad = math.radians(float(azi_deg))
    return np.asarray(
        [
            math.sin(inc_rad) * math.sin(azi_rad),
            math.sin(inc_rad) * math.cos(azi_rad),
            math.cos(inc_rad),
        ],
        dtype=float,
    )


def _normalize_azimuth_deg(value: float) -> float:
    return float(value % 360.0)


def _azimuth_between(start: Point3D, end: Point3D) -> float:
    dx = float(end.x) - float(start.x)
    dy = float(end.y) - float(start.y)
    if math.hypot(dx, dy) <= SMALL:
        return 0.0
    return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)


def _sidetrack_window_candidates(
    *,
    pilot_name: str,
    parent_name: str,
    pilot_stations: pd.DataFrame,
    parent_t1: Point3D,
    config: TrajectoryConfig,
) -> list[PilotWindow]:
    if pilot_stations.empty:
        return []
    stations = pilot_stations.copy()
    preferred = _preferred_sidetrack_window_rows(
        stations=stations,
        parent_t1=parent_t1,
        config=config,
    )
    if not preferred.empty:
        return _pilot_windows_from_rows(
            preferred,
            pilot_name=pilot_name,
            parent_name=parent_name,
        )

    vertical_room = float(parent_t1.z) - stations["Z_m"].to_numpy(dtype=float)
    min_room = max(
        float(config.kop_min_vertical_m), float(config.min_structural_segment_m)
    )
    min_window_md = max(
        float(config.kop_min_vertical_m), float(config.min_structural_segment_m)
    )
    eligible = stations.loc[
        (vertical_room >= min_room)
        & (stations["MD_m"].to_numpy(dtype=float) >= min_window_md)
    ].copy()
    if eligible.empty:
        fallback_min_md = max(
            float(config.min_structural_segment_m),
            float(config.md_step_m),
        )
        eligible = (
            stations.loc[stations["MD_m"].to_numpy(dtype=float) >= fallback_min_md]
            .iloc[:-1]
            .copy()
        )
    if eligible.empty:
        return []
    eligible = eligible.sort_values("MD_m", ascending=True)
    spacing_m = max(150.0, float(config.md_step_m) * 10.0)
    selected_rows = []
    last_md: float | None = None
    for _, row in eligible.iterrows():
        md = float(row["MD_m"])
        if last_md is None or abs(last_md - md) >= spacing_m:
            selected_rows.append(row)
            last_md = md
    if len(selected_rows) > 18:
        indices = np.linspace(0, len(selected_rows) - 1, 18, dtype=int)
        selected_rows = [selected_rows[int(index)] for index in indices]
    return _pilot_windows_from_rows(
        selected_rows,
        pilot_name=pilot_name,
        parent_name=parent_name,
    )


def _preferred_sidetrack_window_rows(
    *,
    stations: pd.DataFrame,
    parent_t1: Point3D,
    config: TrajectoryConfig,
) -> pd.DataFrame:
    first_target_md = _first_pilot_target_md_m(stations)
    if first_target_md is None:
        return pd.DataFrame()
    min_room = max(
        float(config.kop_min_vertical_m), float(config.min_structural_segment_m)
    )
    md_values = stations["MD_m"].to_numpy(dtype=float)
    vertical_room = float(parent_t1.z) - stations["Z_m"].to_numpy(dtype=float)
    window_min_md = first_target_md - SIDETRACK_WINDOW_ABOVE_FIRST_TARGET_MAX_M
    window_max_md = first_target_md - SIDETRACK_WINDOW_ABOVE_FIRST_TARGET_MIN_M
    rows = stations.loc[
        (md_values >= window_min_md - SMALL)
        & (md_values <= window_max_md + SMALL)
        & (vertical_room >= min_room)
    ].copy()
    return rows.sort_values("MD_m", ascending=True)


def _first_pilot_target_md_m(stations: pd.DataFrame) -> float | None:
    if (
        stations.empty
        or "MD_m" not in stations.columns
        or "segment" not in stations.columns
    ):
        return None
    segments = stations["segment"].fillna("").astype(str).str.upper()
    for segment_name in ("PILOT_HOLD_1", "PILOT_BUILD_1"):
        rows = stations.loc[segments == segment_name]
        if not rows.empty:
            return float(rows["MD_m"].max())
    return None


def _pilot_windows_from_rows(
    rows: pd.DataFrame | list[pd.Series],
    *,
    pilot_name: str,
    parent_name: str,
) -> list[PilotWindow]:
    if isinstance(rows, pd.DataFrame):
        iterable = [row for _, row in rows.iterrows()]
    else:
        iterable = rows
    return [
        PilotWindow.from_station(
            pilot_name=pilot_name,
            parent_name=parent_name,
            row=row,
        )
        for row in iterable
    ]


def _manual_sidetrack_window(
    *,
    pilot_name: str,
    parent_name: str,
    pilot_stations: pd.DataFrame,
    override: SidetrackWindowOverride,
) -> PilotWindow:
    if pilot_stations.empty:
        raise ValueError("Ручное окно зарезки невозможно: инклинометрия пилота пуста.")
    stations = pilot_stations.copy()
    required = {"MD_m", "X_m", "Y_m", "Z_m", "INC_deg", "AZI_deg"}
    if not required.issubset(stations.columns):
        raise ValueError(
            "Ручное окно зарезки невозможно: в инклинометрии пилота нет "
            "MD/X/Y/Z/INC/AZI."
        )
    finite_mask = np.isfinite(stations[list(required)].to_numpy(dtype=float)).all(axis=1)
    stations = stations.loc[finite_mask].copy()
    stations = (
        stations.sort_values("MD_m", ascending=True)
        .drop_duplicates(subset=["MD_m"], keep="last")
        .reset_index(drop=True)
    )
    if len(stations) < 2:
        raise ValueError(
            "Ручное окно зарезки невозможно: у пилота меньше двух станций."
        )
    if override.kind == "md":
        row = _interpolate_station_by_md(stations, float(override.value_m))
    else:
        row = _interpolate_station_by_z(stations, float(override.value_m))
    return PilotWindow.from_station(
        pilot_name=pilot_name,
        parent_name=parent_name,
        row=row,
    )


def _interpolate_station_by_md(stations: pd.DataFrame, target_md_m: float) -> pd.Series:
    md_values = stations["MD_m"].to_numpy(dtype=float)
    md_min = float(np.nanmin(md_values))
    md_max = float(np.nanmax(md_values))
    if target_md_m < md_min - SMALL or target_md_m > md_max + SMALL:
        raise ValueError(
            f"Ручное окно зарезки по MD={target_md_m:.2f} м вне диапазона "
            f"пилота {md_min:.2f}–{md_max:.2f} м."
        )
    for index, row in stations.iterrows():
        if abs(float(row["MD_m"]) - target_md_m) <= SMALL:
            return row.copy()
    for index in range(len(stations) - 1):
        start = stations.iloc[index]
        end = stations.iloc[index + 1]
        start_md = float(start["MD_m"])
        end_md = float(end["MD_m"])
        if abs(end_md - start_md) <= SMALL:
            continue
        if start_md - SMALL <= target_md_m <= end_md + SMALL:
            fraction = (target_md_m - start_md) / (end_md - start_md)
            return _interpolated_station_row(
                start=start,
                end=end,
                fraction=float(max(0.0, min(1.0, fraction))),
                md_m=target_md_m,
            )
    raise ValueError(
        f"Не удалось интерполировать ручное окно зарезки по MD={target_md_m:.2f} м."
    )


def _interpolate_station_by_z(stations: pd.DataFrame, target_z_m: float) -> pd.Series:
    z_values = stations["Z_m"].to_numpy(dtype=float)
    z_min = float(np.nanmin(z_values))
    z_max = float(np.nanmax(z_values))
    if target_z_m < z_min - SMALL or target_z_m > z_max + SMALL:
        raise ValueError(
            f"Ручное окно зарезки по Z={target_z_m:.2f} м вне диапазона "
            f"пилота {z_min:.2f}–{z_max:.2f} м."
        )
    for _, row in stations.iterrows():
        if abs(float(row["Z_m"]) - target_z_m) <= SMALL:
            return row.copy()
    for index in range(len(stations) - 1):
        start = stations.iloc[index]
        end = stations.iloc[index + 1]
        start_z = float(start["Z_m"])
        end_z = float(end["Z_m"])
        if abs(end_z - start_z) <= SMALL:
            continue
        lower = min(start_z, end_z) - SMALL
        upper = max(start_z, end_z) + SMALL
        if lower <= target_z_m <= upper:
            fraction = (target_z_m - start_z) / (end_z - start_z)
            start_md = float(start["MD_m"])
            end_md = float(end["MD_m"])
            md_m = start_md + (end_md - start_md) * fraction
            return _interpolated_station_row(
                start=start,
                end=end,
                fraction=float(max(0.0, min(1.0, fraction))),
                md_m=float(md_m),
            )
    raise ValueError(
        f"Не удалось интерполировать ручное окно зарезки по Z={target_z_m:.2f} м."
    )


def _interpolated_station_row(
    *,
    start: pd.Series,
    end: pd.Series,
    fraction: float,
    md_m: float,
) -> pd.Series:
    fraction = float(max(0.0, min(1.0, fraction)))
    start_azi = float(start["AZI_deg"])
    end_azi = float(end["AZI_deg"])
    azi_delta = ((end_azi - start_azi + 180.0) % 360.0) - 180.0
    segment = end.get("segment", start.get("segment", ""))
    return pd.Series(
        {
            "MD_m": float(md_m),
            "X_m": _lerp(float(start["X_m"]), float(end["X_m"]), fraction),
            "Y_m": _lerp(float(start["Y_m"]), float(end["Y_m"]), fraction),
            "Z_m": _lerp(float(start["Z_m"]), float(end["Z_m"]), fraction),
            "INC_deg": _lerp(float(start["INC_deg"]), float(end["INC_deg"]), fraction),
            "AZI_deg": _normalize_azimuth_deg(start_azi + azi_delta * fraction),
            "segment": segment,
        }
    )


def _lerp(start: float, end: float, fraction: float) -> float:
    return float(start + (end - start) * fraction)


def _sidetrack_window_score(
    *,
    window: PilotWindow,
    result: PlannerResult,
    optimization_context: AntiCollisionOptimizationContext | None = None,
) -> float:
    if result.stations.empty or len(result.stations) < 2:
        return float("inf")
    first_tail = result.stations.iloc[1]
    window_md = float(window.md_m)
    tail_md = window_md + float(first_tail["MD_m"])
    if tail_md <= window_md + SMALL:
        return float("inf")
    junction_dls = float(
        dls_deg_per_30m(
            window_md,
            float(window.inc_deg),
            float(window.azi_deg),
            tail_md,
            float(first_tail["INC_deg"]),
            float(first_tail["AZI_deg"]),
        )[()]
    )
    planned_dls = max(
        junction_dls,
        float(result.summary.get("max_dls_total_deg_per_30m", 0.0)),
    )
    sidetrack_md_m = float(result.summary.get("md_total_m", 0.0))
    dls_limit = float(result.summary.get("build_dls_max_config_deg_per_30m", 0.0))
    dls_excess = max(0.0, planned_dls - dls_limit) if dls_limit > SMALL else 0.0
    score = sidetrack_md_m + 300.0 * planned_dls + 100_000.0 * dls_excess
    if optimization_context is not None:
        score += _sidetrack_anticollision_penalty(
            result=result,
            window=window,
            optimization_context=optimization_context,
        )
    return score


def _sidetrack_anticollision_penalty(
    *,
    result: PlannerResult,
    window: PilotWindow,
    optimization_context: AntiCollisionOptimizationContext,
) -> float:
    shifted_stations = result.stations.copy()
    shifted_stations["MD_m"] = shifted_stations["MD_m"].to_numpy(dtype=float) + float(
        window.md_m
    )
    try:
        clearance = evaluate_stations_anti_collision_clearance(
            stations=shifted_stations,
            context=optimization_context,
        )
    except ValueError:
        return 10_000.0
    sf_deficit = max(
        0.0,
        float(optimization_context.sf_target) - float(clearance.min_separation_factor),
    )
    return float(sf_deficit * 1000.0 + float(clearance.max_overlap_depth_m) * 0.1)


def _sidetrack_segment_label(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "SIDETRACK"
    if text == "HORIZONTAL":
        return text
    return f"SIDETRACK_{text}"


def _points_have_finite_xyz(points: tuple[WelltrackPoint, ...]) -> bool:
    for point in points:
        for value in (point.x, point.y, point.z, point.md):
            if not math.isfinite(float(value)):
                return False
    return True


def _has_zero_length_leg(points: tuple[WelltrackPoint, ...]) -> bool:
    for start, end in zip(points, points[1:], strict=False):
        if (
            math.dist(
                (float(start.x), float(start.y), float(start.z)),
                (float(end.x), float(end.y), float(end.z)),
            )
            <= SMALL
        ):
            return True
    return False


def _finite_max(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return 0.0
    return float(np.max(array))


def _first_valid_azimuth_deg(stations: pd.DataFrame) -> float:
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    if len(x_values) < 2:
        return 0.0
    dx = np.diff(x_values)
    dy = np.diff(y_values)
    lengths = np.hypot(dx, dy)
    valid = lengths > SMALL
    if not np.any(valid):
        return 0.0
    index = int(np.argmax(valid))
    return float((math.degrees(math.atan2(dx[index], dy[index])) + 360.0) % 360.0)
