from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

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
from pywp.planner import TrajectoryPlanner
from pywp.planner_types import PlanningError
from pywp.pydantic_base import FrozenArbitraryModel
from pywp.segments import BuildSegment, HoldSegment
from pywp.sidetrack_solver import SidetrackPlanner, SidetrackStart
from pywp.trajectory import WellTrajectory
from pywp.ui_utils import dls_to_pi

PILOT_SUFFIX = "_PL"
ZBS_SUFFIX = "_ZBS"
ALT_BRANCH_SUFFIX = "_2"
SIDETRACK_WINDOW_ABOVE_FIRST_TARGET_MIN_M = 50.0
SIDETRACK_WINDOW_ABOVE_FIRST_TARGET_MAX_M = 100.0
_SURFACE_POINT_LABELS = {"s", "surface", "wellhead", "well_head", "well head", "wh"}
_ZBS_MULTI_HORIZONTAL_LABEL_RE = re.compile(r"^[1-9]\d*_t[13]$", flags=re.IGNORECASE)


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


SidetrackCandidateValidator = Callable[
    [pd.DataFrame, PilotWindow, PlannerResult],
    PlannerResult | None,
]


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


@dataclass(frozen=True)
class _SidetrackGeometrySeed:
    source: str
    azimuth_deg: float
    inc_deg: float
    md_total_m: float


@dataclass(frozen=True)
class ReorientedPilotSidetrackPlan:
    """Joint fallback result for a replanned pilot and its sidetrack.

    ``total_drilled_md_m`` deliberately includes the complete pilot down to
    its last PL point and the lateral length from the selected window.  The
    pilot below the window remains a separately drilled bore and must not be
    dropped from the optimization objective.
    """

    pilot: PilotBuildResult
    window: PilotWindow
    sidetrack_result: PlannerResult
    target_azimuth_deg: float
    total_drilled_md_m: float
    geometry_seed_source: str = "legacy"
    geometry_seed_azimuth_deg: float = 0.0
    geometry_seed_inc_deg: float = 0.0
    geometry_seed_md_total_m: float = 0.0
    sidetrack_lateral_md_m: float = 0.0


def is_pilot_name(name: object) -> bool:
    return str(name).strip().upper().endswith(PILOT_SUFFIX)


def is_zbs_name(name: object) -> bool:
    return str(name).strip().upper().endswith(ZBS_SUFFIX)


def is_alt_branch_name(name: object) -> bool:
    return str(name).strip().upper().endswith(ALT_BRANCH_SUFFIX)


def _record_point_labels(record: WelltrackRecord) -> tuple[str, ...]:
    labels = tuple(
        str(label).strip() for label in getattr(record, "point_labels", ()) or ()
    )
    if len(labels) != len(tuple(record.points)):
        return ()
    return labels


def _is_surface_point_label(label: str) -> bool:
    return str(label).strip().casefold() in _SURFACE_POINT_LABELS


def _is_zbs_target_point_label(label: str) -> bool:
    normalized = str(label).strip()
    return normalized.casefold() in {"t1", "t3"} or (
        _ZBS_MULTI_HORIZONTAL_LABEL_RE.match(normalized) is not None
    )


def parent_name_for_pilot(name: object) -> str:
    text = str(name).strip()
    if not is_pilot_name(text):
        return text
    return text[: -len(PILOT_SUFFIX)]


def pilot_parent_name_for_record(record: object) -> str:
    raw_name = getattr(record, "name", record)
    text = str(raw_name).strip()
    if is_pilot_name(text):
        return parent_name_for_pilot(text)
    if is_alt_branch_name(text):
        if hasattr(record, "points") and is_zbs_record(record):
            return text
        return text[: -len(ALT_BRANCH_SUFFIX)]
    return text


def parent_name_for_zbs(name: object) -> str:
    text = str(name).strip()
    if is_zbs_name(text):
        return text[: -len(ZBS_SUFFIX)]
    if is_alt_branch_name(text):
        return text[: -len(ALT_BRANCH_SUFFIX)]
    return text


def pilot_name_for_parent(name: object) -> str:
    return f"{str(name).strip()}{PILOT_SUFFIX}"


def well_name_key(name: object) -> str:
    return str(name).strip().casefold()


def pilot_parent_key_for_record(name: object) -> str:
    return well_name_key(pilot_parent_name_for_record(name))


def pilot_name_key_for_parent(name: object) -> str:
    return well_name_key(pilot_name_for_parent(name))


def pilot_name_key_for_record(record: object) -> str:
    return pilot_name_key_for_parent(pilot_parent_name_for_record(record))


def is_pilot_record(record: WelltrackRecord) -> bool:
    return is_pilot_name(record.name)


def is_zbs_record(record: WelltrackRecord) -> bool:
    if is_zbs_name(record.name):
        return True
    if not is_alt_branch_name(record.name):
        return False
    labels = _record_point_labels(record)
    if labels:
        if any(_is_surface_point_label(label) for label in labels):
            return False
        if all(_is_zbs_target_point_label(label) for label in labels):
            return True
        return False
    points = tuple(record.points)
    if len(points) < 2 or len(points) % 2 != 0:
        return False
    # Unlabelled `_2` imports are only distinguishable structurally:
    # target-only sidetracks contain target pairs only, while surface-bearing
    # branches include an extra wellhead point and therefore stay odd-sized.
    return True


def visible_well_records(
    records: Iterable[WelltrackRecord],
    *,
    include_zbs: bool = True,
) -> list[WelltrackRecord]:
    return [
        record
        for record in records
        if not is_pilot_record(record) and (include_zbs or not is_zbs_record(record))
    ]


def visible_well_names(records: Iterable[WelltrackRecord]) -> list[str]:
    return [str(record.name) for record in visible_well_records(records)]


def zbs_target_points_to_pair(
    points: tuple[WelltrackPoint, ...],
) -> tuple[Point3D, Point3D]:
    pairs = zbs_target_points_to_pairs(points)
    if len(pairs) != 1:
        raise ValueError(
            "Для бокового ствола от фактической скважины ожидаются ровно "
            "две точки `t1` и `t3` без `S`; для многопластового ZBS "
            "используйте полные пары `1_t1/1_t3`, `2_t1/2_t3`, ... без `S`."
        )
    return pairs[0]


def zbs_target_points_to_pairs(
    points: tuple[WelltrackPoint, ...],
) -> tuple[tuple[Point3D, Point3D], ...]:
    if len(points) < 2 or len(points) % 2 != 0:
        raise ValueError(
            "Для бокового ствола от фактической скважины ожидаются две точки "
            "`t1` и `t3` без `S` либо полные пары `1_t1/1_t3`, "
            "`2_t1/2_t3`, ... без `S` для многопластового ZBS."
        )
    if not _points_have_finite_xyz(tuple(points)):
        raise ValueError("Координаты X/Y/Z и MD должны быть конечными числами.")
    md_values = [float(point.md) for point in points]
    if not all(
        left_md + SMALL < right_md
        for left_md, right_md in zip(md_values, md_values[1:], strict=False)
    ):
        raise ValueError(
            "MD точек ZBS должны строго возрастать: `t1` затем `t3`, "
            "для многопластового ZBS - `1_t1`, `1_t3`, `2_t1`, `2_t3`, ..."
        )

    pairs: list[tuple[Point3D, Point3D]] = []
    for index in range(0, len(points), 2):
        left = points[index]
        right = points[index + 1]
        if (
            math.dist(
                (float(left.x), float(left.y), float(left.z)),
                (float(right.x), float(right.y), float(right.z)),
            )
            <= SMALL
        ):
            level_label = "" if len(points) == 2 else f" уровня {index // 2 + 1}"
            raise ValueError(f"Точки `t1` и `t3` ZBS{level_label} совпадают.")
        pairs.append((_point_from_welltrack(left), _point_from_welltrack(right)))
    return tuple(pairs)


def zbs_multi_horizontal_level_count(points: tuple[WelltrackPoint, ...]) -> int:
    point_count = int(len(tuple(points)))
    if point_count < 4 or point_count % 2 != 0:
        return 0
    return int(point_count // 2)


def sync_pilot_surfaces_to_parents(
    records: Iterable[WelltrackRecord],
    *,
    only_parent_keys: Iterable[str] | None = None,
) -> list[WelltrackRecord]:
    record_list = list(records)
    scoped_parent_keys = (
        {str(key).strip().casefold() for key in only_parent_keys if str(key).strip()}
        if only_parent_keys is not None
        else None
    )
    parent_by_key: dict[str, WelltrackRecord] = {}
    for record in record_list:
        if is_pilot_record(record) or is_zbs_record(record):
            continue
        parent_by_key.setdefault(pilot_parent_key_for_record(record), record)
    synced: list[WelltrackRecord] = []
    for record in record_list:
        if not is_pilot_record(record) or not record.points:
            synced.append(record)
            continue
        parent_key = pilot_parent_key_for_record(record)
        if scoped_parent_keys is not None and parent_key.casefold() not in scoped_parent_keys:
            synced.append(record)
            continue
        parent = parent_by_key.get(parent_key)
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
                point_labels=getattr(record, "point_labels", ()),
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
        if not is_zbs_record(record):
            pilot = by_name.get(
                pilot_name_key_for_parent(pilot_parent_name_for_record(name))
            )
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
    candidate_validator: SidetrackCandidateValidator | None = None,
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
            if candidate_validator is not None:
                complete_result = candidate_validator(pilot_stations, window, result)
                if complete_result is None:
                    raise ValueError(
                        "Проверка полной траектории отклонила окно зарезки."
                    )
                _complete_sidetrack_lateral_md_m(
                    window=window,
                    result=complete_result,
                )
            return window, result
        except (
            ValueError,
            PlanningError,
            ArithmeticError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
        ) as exc:
            coordinate_label = "MD" if window_override.kind == "md" else "Z"
            raise ValueError(
                f"Ручное окно зарезки {parent_name} по {coordinate_label}="
                f"{window_override.value_m:.2f} м не дало расчет бокового ствола: {exc}"
            ) from exc

    candidate_groups = _sidetrack_window_candidate_groups(
        pilot_name=pilot_name,
        parent_name=parent_name,
        pilot_stations=pilot_stations,
        parent_t1=parent_t1,
        config=config,
    )
    last_problem = ""
    best: tuple[float, float, PilotWindow, PlannerResult] | None = None
    for candidates in candidate_groups:
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
                complete_result = None
                if candidate_validator is not None:
                    complete_result = candidate_validator(
                        pilot_stations,
                        window,
                        result,
                    )
                    if complete_result is None:
                        raise ValueError(
                            "Проверка полной траектории отклонила окно зарезки."
                        )
                score = _sidetrack_window_score(
                    window=window,
                    result=result,
                    complete_result=complete_result,
                    optimization_context=optimization_context,
                )
            except (
                ValueError,
                PlanningError,
                ArithmeticError,
                KeyError,
                TypeError,
                AttributeError,
                IndexError,
            ) as exc:
                last_problem = str(exc)
                continue
            if not math.isfinite(score):
                last_problem = (
                    "Расчет бокового ствола для окна "
                    f"MD={float(window.md_m):.2f} м вернул некорректную оценку."
                )
                continue
            if best is None or (score, -float(window.md_m)) < (best[0], best[1]):
                best = (score, -float(window.md_m), window, result)

    # The 50-100 m interval is evaluated first, but earlier valid windows above
    # PL1 still participate in the global complete-branch optimization.
    if best is not None:
        _, _, window, result = best
        return window, result

    suffix = f" Последняя причина: {last_problem}" if last_problem else ""
    raise ValueError(
        "Не удалось подобрать окно зарезки на пилоте: ни одна станция пилота "
        "не дала расчет продуктивного ствола до t1/t3." + suffix
    )


def _finite_float_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _planner_result_md_total_m(result: PlannerResult) -> float:
    """Read MD from stations first; summaries are metadata fallbacks only."""

    stations = result.stations
    if isinstance(stations, pd.DataFrame) and not stations.empty:
        if "MD_m" not in stations.columns:
            raise ValueError("Результат планировщика не содержит колонку MD_m.")
        try:
            md_values = stations["MD_m"].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Результат планировщика содержит нечисловой MD.") from exc
        if (
            np.any(~np.isfinite(md_values))
            or float(md_values[0]) < -SMALL
            or (len(md_values) > 1 and np.any(np.diff(md_values) <= SMALL))
        ):
            raise ValueError(
                "Результат планировщика содержит нечисловой, отрицательный "
                "или не возрастающий MD."
            )
        return float(md_values[-1])

    summary_md = _finite_float_or_none(result.summary.get("md_total_m"))
    if summary_md is not None and summary_md >= 0.0:
        return summary_md
    return 0.0


def _complete_sidetrack_lateral_md_m(
    *,
    window: PilotWindow,
    result: PlannerResult,
) -> float:
    """Validate a complete, globally-MD-referenced sidetrack candidate."""

    if not isinstance(result, PlannerResult):
        raise TypeError(
            "Проверка полного кандидата бокового ствола должна вернуть PlannerResult."
        )
    stations = result.stations
    if not isinstance(stations, pd.DataFrame) or len(stations) < 2:
        raise ValueError(
            "Полный кандидат бокового ствола содержит меньше двух станций."
        )
    required_columns = {"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"}
    missing_columns = required_columns.difference(stations.columns)
    if missing_columns:
        raise ValueError(
            "Полный кандидат бокового ствола не содержит колонки "
            f"{', '.join(sorted(missing_columns))}."
        )
    try:
        station_values = stations[
            ["MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"]
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Полный кандидат бокового ствола содержит нечисловую станцию."
        ) from exc
    if np.any(~np.isfinite(station_values)):
        raise ValueError("Полный кандидат бокового ствола содержит нечисловую станцию.")

    window_md_m = float(window.md_m)
    if abs(float(station_values[0, 0]) - window_md_m) > 1e-6:
        raise ValueError(
            "Глобальный MD полного кандидата бокового ствола должен начинаться "
            "в выбранном окне зарезки."
        )
    start_miss_m = float(
        np.linalg.norm(
            station_values[0, 3:6]
            - np.asarray(
                [float(window.point.x), float(window.point.y), float(window.point.z)],
                dtype=float,
            )
        )
    )
    if start_miss_m > 1e-3:
        raise ValueError(
            "Полный кандидат бокового ствола не начинается в выбранном окне зарезки."
        )
    if (
        abs(float(station_values[0, 1]) - float(window.inc_deg)) > 1e-3
        or _azimuth_difference_deg(
            float(station_values[0, 2]),
            float(window.azi_deg),
        )
        > 1e-3
    ):
        raise ValueError(
            "Ориентация полного кандидата бокового ствола не совпадает с окном зарезки."
        )

    lateral_md_m = _planner_result_md_total_m(result) - window_md_m
    if not math.isfinite(lateral_md_m) or lateral_md_m <= SMALL:
        raise ValueError(
            "Полный кандидат бокового ствола имеет некорректную длину от окна."
        )
    return float(lateral_md_m)


def _planner_result_entry_inc_deg(
    result: PlannerResult,
    *,
    fallback_inc_deg: float,
) -> float:
    md_t1_m = _finite_float_or_none(getattr(result, "md_t1_m", None))
    stations = result.stations
    if (
        md_t1_m is not None
        and isinstance(stations, pd.DataFrame)
        and {"MD_m", "INC_deg"}.issubset(stations.columns)
    ):
        finite = stations[["MD_m", "INC_deg"]].copy()
        finite = finite.loc[
            np.isfinite(finite[["MD_m", "INC_deg"]].to_numpy(dtype=float)).all(axis=1)
        ]
        finite = finite.sort_values("MD_m").drop_duplicates("MD_m", keep="last")
        if len(finite) >= 1:
            md_values = finite["MD_m"].to_numpy(dtype=float)
            inc_values = finite["INC_deg"].to_numpy(dtype=float)
            if len(finite) == 1 or md_t1_m <= float(md_values[0]):
                return float(inc_values[0])
            if md_t1_m >= float(md_values[-1]):
                return float(inc_values[-1])
            return float(np.interp(md_t1_m, md_values, inc_values))

    for key in ("entry_inc_deg", "inc_t1_deg", "hold_inc_deg"):
        summary_inc = _finite_float_or_none(result.summary.get(key))
        if summary_inc is not None:
            return summary_inc

    if isinstance(stations, pd.DataFrame) and "INC_deg" in stations.columns:
        inc_values = stations["INC_deg"].to_numpy(dtype=float)
        inc_values = inc_values[np.isfinite(inc_values)]
        if len(inc_values) > 0:
            return float(inc_values[-1])
    return float(fallback_inc_deg)


def _sidetrack_geometry_seed(
    *,
    pilot_target_points: tuple[Point3D, ...],
    parent_t1: Point3D,
    productive_direction_target: Point3D,
    sidetrack_config: TrajectoryConfig,
) -> _SidetrackGeometrySeed:
    productive_delta = _point_array(productive_direction_target) - _point_array(
        parent_t1
    )
    productive_inc_deg, productive_azimuth_deg = _angles_from_delta(productive_delta)
    fallback = _SidetrackGeometrySeed(
        source="productive_direction",
        azimuth_deg=float(productive_azimuth_deg),
        inc_deg=float(productive_inc_deg),
        md_total_m=0.0,
    )
    if len(pilot_target_points) == 0:
        return fallback

    try:
        standalone = TrajectoryPlanner().plan(
            surface=pilot_target_points[0],
            t1=parent_t1,
            t3=productive_direction_target,
            config=sidetrack_config,
        )
    except (ValueError, PlanningError, ArithmeticError, KeyError, TypeError):
        return fallback

    try:
        seed_azimuth_deg = _finite_float_or_none(
            getattr(standalone, "azimuth_deg", None)
        )
        stations = getattr(standalone, "stations", pd.DataFrame())
        if seed_azimuth_deg is None:
            if not isinstance(stations, pd.DataFrame) or not {"X_m", "Y_m"}.issubset(
                stations.columns
            ):
                return fallback
            seed_azimuth_deg = _first_valid_azimuth_deg(stations)
        seed_inc_deg = _planner_result_entry_inc_deg(
            standalone,
            fallback_inc_deg=productive_inc_deg,
        )
        seed_md_total_m = _planner_result_md_total_m(standalone)
    except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError):
        return fallback

    seed_azimuth_deg = _finite_float_or_none(seed_azimuth_deg)
    seed_inc_deg = _finite_float_or_none(seed_inc_deg)
    seed_md_total_m = _finite_float_or_none(seed_md_total_m)
    if seed_azimuth_deg is None or seed_inc_deg is None:
        return fallback
    if seed_md_total_m is None or seed_md_total_m < 0.0:
        seed_md_total_m = 0.0
    return _SidetrackGeometrySeed(
        source="standalone_sidetrack",
        azimuth_deg=_normalize_azimuth_deg(seed_azimuth_deg),
        inc_deg=float(seed_inc_deg),
        md_total_m=float(seed_md_total_m),
    )


def plan_reoriented_pilot_sidetrack_fallback(
    *,
    pilot_name: str,
    parent_name: str,
    pilot_target_points: tuple[Point3D, ...],
    parent_t1: Point3D,
    parent_t3: Point3D,
    productive_direction_target: Point3D,
    pilot_config: TrajectoryConfig,
    sidetrack_config: TrajectoryConfig,
    optimization_context: AntiCollisionOptimizationContext | None = None,
    candidate_validator: SidetrackCandidateValidator | None = None,
) -> ReorientedPilotSidetrackPlan:
    """Reorient the project pilot as a last-resort sidetrack fallback.

    The productive branch is first planned as a standalone well to extract a
    geometric approach hint.  The pilot is then rebuilt with PL1 terminal
    azimuth/inc candidates from that standalone geometry and from the direct
    t1->t2/t3 productive interval.  Every original PL point is still hit
    exactly.  All feasible pilot/window combinations are ranked by complete
    drilled MD rather than by ``window_md + lateral_md``.
    """

    if len(pilot_target_points) < 2:
        raise ValueError(
            "Fallback перестройки пилота требует устье и минимум одну PL-точку."
        )
    productive_delta = _point_array(productive_direction_target) - _point_array(
        parent_t1
    )
    if float(np.linalg.norm(productive_delta)) <= SMALL:
        raise ValueError(
            "Невозможно перестроить пилот: первая продуктивная секция имеет "
            "нулевую длину."
        )
    productive_inc_deg, productive_azimuth_deg = _angles_from_delta(productive_delta)
    geometry_seed = _sidetrack_geometry_seed(
        pilot_target_points=pilot_target_points,
        parent_t1=parent_t1,
        productive_direction_target=productive_direction_target,
        sidetrack_config=sidetrack_config,
    )
    azimuth_hints: list[tuple[float, float]] = []
    for azimuth_deg, inc_deg in (
        (geometry_seed.azimuth_deg, geometry_seed.inc_deg),
        (productive_azimuth_deg, productive_inc_deg),
    ):
        if not math.isfinite(float(azimuth_deg)) or not math.isfinite(float(inc_deg)):
            continue
        normalized_azimuth_deg = _normalize_azimuth_deg(float(azimuth_deg))
        if any(
            _azimuth_difference_deg(normalized_azimuth_deg, existing[0]) <= 1e-6
            for existing in azimuth_hints
        ):
            continue
        azimuth_hints.append((normalized_azimuth_deg, float(inc_deg)))
    if not azimuth_hints:
        azimuth_hints.append(
            (_normalize_azimuth_deg(productive_azimuth_deg), float(productive_inc_deg))
        )

    best: (
        tuple[
            tuple[float, float, float, float, float],
            PilotBuildResult,
            PilotWindow,
            PlannerResult,
            float,
        ]
        | None
    ) = None
    last_problem = ""
    for target_azimuth_deg, terminal_inc_hint_deg in azimuth_hints:
        try:
            pilot_candidates = _reoriented_pilot_candidates(
                target_points=pilot_target_points,
                target_azimuth_deg=target_azimuth_deg,
                terminal_inc_hint_deg=terminal_inc_hint_deg,
                productive_direction_target=productive_direction_target,
                parent_t1=parent_t1,
                config=pilot_config,
            )
        except (ValueError, PlanningError, ArithmeticError, KeyError, TypeError) as exc:
            last_problem = str(exc)
            continue
        for pilot in pilot_candidates:
            pilot_anti_collision_penalty = (
                _trajectory_anticollision_penalty(
                    stations=pilot.stations,
                    optimization_context=optimization_context,
                )
                if optimization_context is not None
                else 0.0
            )
            candidate_groups = _sidetrack_window_candidate_groups(
                pilot_name=pilot_name,
                parent_name=parent_name,
                pilot_stations=pilot.stations,
                parent_t1=parent_t1,
                config=sidetrack_config,
            )
            # Unlike the normal path, this is a global joint optimization.  The
            # preferred 50-100 m group is evaluated first but does not prevent a
            # shorter feasible total-drilling solution in the expanded group.
            windows = _bounded_reoriented_window_candidates(candidate_groups)
            for window in windows:
                try:
                    sidetrack_result = SidetrackPlanner().plan(
                        start=SidetrackStart(
                            point=window.point,
                            inc_deg=float(window.inc_deg),
                            azi_deg=float(window.azi_deg),
                        ),
                        t1=parent_t1,
                        t3=parent_t3,
                        config=sidetrack_config,
                    )
                    complete_result = (
                        candidate_validator(pilot.stations, window, sidetrack_result)
                        if candidate_validator is not None
                        else None
                    )
                    if candidate_validator is not None and complete_result is None:
                        raise ValueError(
                            "Проверка полной траектории отклонила окно зарезки."
                        )
                    lateral_md_m = (
                        _complete_sidetrack_lateral_md_m(
                            window=window,
                            result=complete_result,
                        )
                        if complete_result is not None
                        else _planner_result_md_total_m(sidetrack_result)
                    )
                    if not math.isfinite(lateral_md_m) or lateral_md_m <= SMALL:
                        raise ValueError(
                            "Расчет бокового ствола вернул нулевую длину от окна."
                        )
                except (
                    ValueError,
                    PlanningError,
                    ArithmeticError,
                    KeyError,
                    TypeError,
                    AttributeError,
                    IndexError,
                ) as exc:
                    last_problem = str(exc)
                    continue
                total_drilled_md_m = float(pilot.md_total_m) + lateral_md_m
                if optimization_context is None:
                    anti_collision_penalty = 0.0
                elif complete_result is not None:
                    # ``complete_result`` already uses global MD.  Applying
                    # the local-window shift here would double-count the
                    # pilot section in anti-collision evaluation.
                    anti_collision_penalty = _trajectory_anticollision_penalty(
                        stations=complete_result.stations,
                        optimization_context=optimization_context,
                    )
                else:
                    anti_collision_penalty = _sidetrack_anticollision_penalty(
                        result=sidetrack_result,
                        window=window,
                        optimization_context=optimization_context,
                    )
                pilot_summary_dls = (
                    _finite_float_or_none(
                        pilot.summary.get("max_dls_total_deg_per_30m")
                    )
                    or 0.0
                )
                pilot_station_dls = (
                    _finite_max(pilot.stations["DLS_deg_per_30m"])
                    if "DLS_deg_per_30m" in pilot.stations.columns
                    else 0.0
                )
                sidetrack_summary_dls = (
                    _finite_float_or_none(
                        (
                            complete_result.summary
                            if complete_result is not None
                            else sidetrack_result.summary
                        ).get("max_dls_total_deg_per_30m")
                    )
                    or 0.0
                )
                sidetrack_station_dls = (
                    _finite_max(
                        (
                            complete_result.stations
                            if complete_result is not None
                            else sidetrack_result.stations
                        )["DLS_deg_per_30m"]
                    )
                    if "DLS_deg_per_30m"
                    in (
                        complete_result.stations
                        if complete_result is not None
                        else sidetrack_result.stations
                    ).columns
                    else 0.0
                )
                max_dls = max(
                    pilot_summary_dls,
                    pilot_station_dls,
                    sidetrack_summary_dls,
                    sidetrack_station_dls,
                )
                key = (
                    total_drilled_md_m
                    + pilot_anti_collision_penalty
                    + anti_collision_penalty,
                    total_drilled_md_m,
                    max_dls,
                    lateral_md_m,
                    -float(window.md_m),
                )
                if best is None or key < best[0]:
                    best = (
                        key,
                        pilot,
                        window,
                        sidetrack_result,
                        float(target_azimuth_deg),
                    )

    if best is None:
        suffix = f" Последняя причина: {last_problem}" if last_problem else ""
        raise ValueError(
            "Не удалось выполнить fallback с перестройкой пилота под азимут "
            "продуктивной секции и подобрать допустимое окно зарезки." + suffix
        )

    key, pilot, window, sidetrack_result, selected_azimuth_deg = best
    return ReorientedPilotSidetrackPlan(
        pilot=pilot,
        window=window,
        sidetrack_result=sidetrack_result,
        target_azimuth_deg=float(selected_azimuth_deg),
        geometry_seed_source=str(geometry_seed.source),
        geometry_seed_azimuth_deg=float(geometry_seed.azimuth_deg),
        geometry_seed_inc_deg=float(geometry_seed.inc_deg),
        geometry_seed_md_total_m=float(geometry_seed.md_total_m),
        total_drilled_md_m=float(key[1]),
        sidetrack_lateral_md_m=float(key[3]),
    )


def combine_pilot_and_sidetrack(
    *,
    pilot_stations: pd.DataFrame,
    sidetrack_result: PlannerResult,
    window: PilotWindow,
    config: TrajectoryConfig,
) -> SidetrackPlan:
    window_md = float(window.md_m)
    if not math.isfinite(window_md) or window_md < -SMALL:
        raise ValueError("Окно зарезки должно иметь конечный неотрицательный MD.")
    window_inc_deg = float(window.inc_deg)
    window_azi_deg = float(window.azi_deg)
    if (
        not math.isfinite(window_inc_deg)
        or not math.isfinite(window_azi_deg)
        or window_inc_deg < -SMALL
        or window_inc_deg > 180.0 + SMALL
    ):
        raise ValueError("Ориентация окна зарезки должна быть конечной и допустимой.")
    required_pilot_columns = {"MD_m", "X_m", "Y_m", "Z_m", "INC_deg", "AZI_deg"}
    if not required_pilot_columns.issubset(pilot_stations.columns):
        raise ValueError(
            "Инклинометрия пилота не содержит MD/X/Y/Z/INC/AZI для сборки ЗБС."
        )
    pilot_stations = pilot_stations.copy().reset_index(drop=True)
    if len(pilot_stations) < 2:
        raise ValueError("Инклинометрия пилота содержит меньше двух станций.")
    try:
        pilot_numeric = pilot_stations[list(required_pilot_columns)].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Инклинометрия пилота содержит нечисловую станцию.") from exc
    pilot_md_values = pilot_stations["MD_m"].to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(pilot_numeric))
        or float(pilot_md_values[0]) < -SMALL
        or np.any(np.diff(pilot_md_values) <= SMALL)
    ):
        raise ValueError(
            "Инклинометрия пилота содержит нечисловой, отрицательный "
            "или не возрастающий MD."
        )
    pilot_total_md_m = float(pilot_stations["MD_m"].iloc[-1])
    if window_md > pilot_total_md_m + SMALL:
        raise ValueError("Окно зарезки находится ниже конечного MD пилота.")
    expected_window = _interpolate_station_by_md(pilot_stations, window_md)
    window_miss_m = float(
        np.linalg.norm(
            np.asarray(
                [
                    float(expected_window["X_m"]) - float(window.point.x),
                    float(expected_window["Y_m"]) - float(window.point.y),
                    float(expected_window["Z_m"]) - float(window.point.z),
                ],
                dtype=float,
            )
        )
    )
    if window_miss_m > 1e-3:
        raise ValueError(
            "Окно зарезки не лежит на траектории пилота "
            f"(расхождение {window_miss_m:.3f} м)."
        )
    window_inc_miss_deg = abs(float(expected_window["INC_deg"]) - float(window.inc_deg))
    window_azi_miss_deg = _azimuth_difference_deg(
        float(expected_window["AZI_deg"]), float(window.azi_deg)
    )
    if window_inc_miss_deg > 1e-3 or window_azi_miss_deg > 1e-3:
        raise ValueError("Ориентация окна зарезки не совпадает с траекторией пилота.")
    pilot_upper = pilot_stations.loc[
        pilot_stations["MD_m"].to_numpy(dtype=float) <= window_md + SMALL
    ].copy()
    if pilot_upper.empty:
        raise ValueError("Не удалось собрать общий участок пилота до окна зарезки.")
    if float(pilot_upper["MD_m"].iloc[-1]) < window_md - SMALL:
        window_row = {
            "MD_m": window_md,
            "INC_deg": float(window.inc_deg),
            "AZI_deg": float(window.azi_deg),
            "X_m": float(window.point.x),
            "Y_m": float(window.point.y),
            "Z_m": float(window.point.z),
            "N_m": float(window.point.y),
            "E_m": float(window.point.x),
            "TVD_m": float(window.point.z),
            "segment": "PILOT_WINDOW",
        }
        pilot_upper = pd.concat(
            [pilot_upper, pd.DataFrame([window_row])],
            ignore_index=True,
        )
        pilot_upper = add_dls(pilot_upper)

    sidetrack_stations = sidetrack_result.stations.copy()
    if len(sidetrack_stations) < 2:
        raise ValueError(
            "Расчет бокового ствола вернул меньше двух станций инклинометрии."
        )
    required_sidetrack_columns = (
        "MD_m",
        "INC_deg",
        "AZI_deg",
        "X_m",
        "Y_m",
        "Z_m",
    )
    if not set(required_sidetrack_columns).issubset(sidetrack_stations.columns):
        raise ValueError(
            "Расчет бокового ствола не вернул MD/INC/AZI/X/Y/Z для сборки ЗБС."
        )
    sidetrack_stations = sidetrack_stations.reset_index(drop=True)
    try:
        sidetrack_numeric = sidetrack_stations[
            list(required_sidetrack_columns)
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Инклинометрия бокового ствола содержит нечисловую станцию."
        ) from exc
    if np.any(~np.isfinite(sidetrack_numeric)):
        raise ValueError(
            "Инклинометрия бокового ствола содержит нечисловой, отрицательный "
            "или не возрастающий MD."
        )
    # PlannerResult does not require callers to preserve DataFrame row order.
    # Normalize local MD before locating the window station and offsetting it
    # into the combined well MD domain.  A stable sort keeps duplicate-MD rows
    # deterministic; duplicates are still rejected by the strict-MD check.
    sidetrack_stations["MD_m"] = sidetrack_numeric[:, 0]
    sidetrack_stations = sidetrack_stations.sort_values(
        "MD_m", kind="mergesort"
    ).reset_index(drop=True)
    sidetrack_md_values = sidetrack_stations["MD_m"].to_numpy(dtype=float)
    if (
        np.any(np.diff(sidetrack_md_values) <= SMALL)
        or float(sidetrack_md_values[0]) < -SMALL
    ):
        raise ValueError(
            "Инклинометрия бокового ствола содержит нечисловой, отрицательный "
            "или не возрастающий MD."
        )
    if abs(float(sidetrack_md_values[0])) > 1e-6:
        raise ValueError("Локальный MD бокового ствола должен начинаться с 0 м.")
    sidetrack_start = sidetrack_stations.iloc[0]
    sidetrack_start_miss_m = float(
        np.linalg.norm(
            np.asarray(
                [
                    float(sidetrack_start["X_m"]) - float(window.point.x),
                    float(sidetrack_start["Y_m"]) - float(window.point.y),
                    float(sidetrack_start["Z_m"]) - float(window.point.z),
                ],
                dtype=float,
            )
        )
    )
    if sidetrack_start_miss_m > 1e-3:
        raise ValueError(
            "Боковой ствол не начинается в выбранном окне зарезки "
            f"(расхождение {sidetrack_start_miss_m:.3f} м)."
        )
    sidetrack_start_inc_miss_deg = abs(
        float(sidetrack_start["INC_deg"]) - float(window.inc_deg)
    )
    sidetrack_start_azi_miss_deg = _azimuth_difference_deg(
        float(sidetrack_start["AZI_deg"]), float(window.azi_deg)
    )
    if sidetrack_start_inc_miss_deg > 1e-3 or sidetrack_start_azi_miss_deg > 1e-3:
        raise ValueError("Ориентация бокового ствола не совпадает с окном зарезки.")
    sidetrack_stations["MD_m"] = (
        sidetrack_stations["MD_m"].to_numpy(dtype=float) + window_md
    )
    sidetrack_stations["segment"] = [
        _sidetrack_segment_label(value)
        for value in sidetrack_stations.get(
            "segment", pd.Series(["SIDETRACK"] * len(sidetrack_stations))
        )
    ]
    stations = sidetrack_stations.reset_index(drop=True)
    stations = add_dls(stations)
    stations.attrs["uncertainty_reference_stations"] = pilot_upper.copy()
    summary = dict(sidetrack_result.summary)
    md_total_m = float(stations["MD_m"].iloc[-1])
    if not math.isfinite(md_total_m):
        md_total_m = 0.0
    # The input may legitimately have arrived in arbitrary row order.  Use
    # the normalized local survey rather than re-reading the unsorted source.
    sidetrack_lateral_md_m = float(sidetrack_md_values[-1])
    local_md_t1_m = _finite_float_or_none(sidetrack_result.md_t1_m)
    if (
        local_md_t1_m is None
        or local_md_t1_m < -SMALL
        or local_md_t1_m > sidetrack_lateral_md_m + SMALL
    ):
        raise ValueError("MD t1 бокового ствола находится вне локальной сетки MD.")
    sidetrack_azimuth_deg = _finite_float_or_none(sidetrack_result.azimuth_deg)
    if sidetrack_azimuth_deg is None:
        raise ValueError("Азимут бокового ствола должен быть конечным числом.")
    total_drilled_md_m = pilot_total_md_m + sidetrack_lateral_md_m
    sidetrack_summary_max_dls = (
        _finite_float_or_none(summary.get("max_dls_total_deg_per_30m")) or 0.0
    )
    sidetrack_kop_md_m = max(
        0.0,
        _finite_float_or_none(summary.get("kop_md_m")) or 0.0,
    )
    max_dls = max(
        sidetrack_summary_max_dls,
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
            "sidetrack_lateral_md_m": sidetrack_lateral_md_m,
            "sidetrack_complete_lateral_md_m": sidetrack_lateral_md_m,
            "pilot_total_md_m": pilot_total_md_m,
            "total_drilled_md_m": total_drilled_md_m,
            "sidetrack_window_optimization_objective_m": total_drilled_md_m,
            "md_total_m": md_total_m,
            "max_total_md_postcheck_m": float(config.max_total_md_postcheck_m),
            "md_postcheck_excess_m": max(
                0.0,
                total_drilled_md_m - float(config.max_total_md_postcheck_m),
            ),
            "md_postcheck_exceeded": (
                "yes"
                if total_drilled_md_m > float(config.max_total_md_postcheck_m) + 1e-6
                else "no"
            ),
            "max_dls_total_deg_per_30m": max_dls,
            "dls_postcheck_excess_deg_per_30m": max(
                0.0,
                max_dls - float(config.dls_build_max_deg_per_30m),
            ),
            "kop_md_m": window_md + sidetrack_kop_md_m,
        }
    )
    return SidetrackPlan(
        result=sidetrack_result,
        window=window,
        stations=stations,
        summary=summary,
        md_t1_m=window_md + local_md_t1_m,
        azimuth_deg=sidetrack_azimuth_deg,
    )


def paired_pilot_parent_names(name_a: object, name_b: object) -> bool:
    left = str(name_a).strip()
    right = str(name_b).strip()
    return (
        is_pilot_name(left)
        and pilot_parent_key_for_record(left) == pilot_parent_key_for_record(right)
        or is_pilot_name(right)
        and pilot_parent_key_for_record(right) == pilot_parent_key_for_record(left)
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


def _reoriented_pilot_candidates(
    *,
    target_points: tuple[Point3D, ...],
    target_azimuth_deg: float,
    terminal_inc_hint_deg: float,
    productive_direction_target: Point3D,
    parent_t1: Point3D,
    config: TrajectoryConfig,
) -> list[PilotBuildResult]:
    """Build feasible pilots whose PL1 tangent has a prescribed azimuth."""

    surface = target_points[0]
    study_points = target_points[1:]
    surface_wp = WelltrackPoint(
        x=float(surface.x), y=float(surface.y), z=float(surface.z), md=0.0
    )
    first_wp = WelltrackPoint(
        x=float(study_points[0].x),
        y=float(study_points[0].y),
        z=float(study_points[0].z),
        md=1.0,
    )
    kop = _pilot_kop_point(
        surface=surface_wp,
        first_target=first_wp,
        config=config,
    )
    vertical = _vertical_pilot_stations(
        surface=surface_wp,
        kop=kop,
        config=config,
    )
    p0 = _point_array(kop)
    p3 = _point_array(study_points[0])
    chord_m = float(np.linalg.norm(p3 - p0))
    if chord_m <= SMALL:
        return []

    geometric_inc_deg, _ = _angles_from_delta(p3 - p0)
    productive_inc_deg, _ = _angles_from_delta(
        _point_array(productive_direction_target) - _point_array(parent_t1)
    )
    to_t1_inc_deg, _ = _angles_from_delta(
        _point_array(parent_t1) - _point_array(study_points[0])
    )
    max_inc_deg = float(config.max_inc_deg)
    terminal_inc_hint = _finite_float_or_none(terminal_inc_hint_deg)
    if terminal_inc_hint is None:
        terminal_inc_hint = productive_inc_deg
    inc_candidates = _unique_floats(
        min(max_inc_deg, max(0.0, value))
        for value in (
            terminal_inc_hint,
            geometric_inc_deg,
            to_t1_inc_deg,
            productive_inc_deg,
            5.0,
            10.0,
            20.0,
            30.0,
            45.0,
            60.0,
            75.0,
            90.0,
        )
    )
    start_dir = _unit_vector_xyz(inc_deg=0.0, azi_deg=target_azimuth_deg)
    dls_limit = float(config.dls_build_max_deg_per_30m)
    first_legs_by_terminal_inc: dict[
        float, list[tuple[tuple[float, float, float], pd.DataFrame]]
    ] = {}

    for terminal_inc_deg in inc_candidates:
        end_dir = _unit_vector_xyz(
            inc_deg=terminal_inc_deg,
            azi_deg=target_azimuth_deg,
        )
        for lead_scale in (0.15, 0.30, 0.50, 0.75, 1.00, 1.35):
            for tail_scale in (0.12, 0.22, 0.36, 0.55, 0.80, 1.10):
                lead_m = max(float(config.md_step_m), chord_m * lead_scale)
                tail_m = max(float(config.md_step_m), chord_m * tail_scale)
                try:
                    first_leg = _pilot_cubic_leg(
                        p0=p0,
                        p1=p0 + start_dir * lead_m,
                        p2=p3 - end_dir * tail_m,
                        p3=p3,
                        start_md_m=float(vertical["MD_m"].iloc[-1]),
                        start_inc_deg=0.0,
                        start_azi_deg=target_azimuth_deg,
                        end_inc_deg=terminal_inc_deg,
                        end_azi_deg=target_azimuth_deg,
                        config=config,
                    )
                except (ValueError, PlanningError):
                    continue

                first_leg_max_dls = _finite_max(first_leg["DLS_deg_per_30m"])
                if first_leg_max_dls > dls_limit + 1e-6:
                    continue
                if _finite_max(first_leg["INC_deg"]) > max_inc_deg + 1e-6:
                    continue
                alignment_md = max(
                    float(vertical["MD_m"].iloc[-1]),
                    float(first_leg["MD_m"].iloc[-1]) - 75.0,
                )
                alignment_row = _interpolate_station_by_md(first_leg, alignment_md)
                alignment_error = _azimuth_difference_deg(
                    float(alignment_row["AZI_deg"]), target_azimuth_deg
                )
                key = (
                    alignment_error,
                    float(first_leg["MD_m"].iloc[-1]),
                    first_leg_max_dls,
                )
                bucket = first_legs_by_terminal_inc.setdefault(
                    round(terminal_inc_deg, 6), []
                )
                bucket.append((key, first_leg))

    # Retain geometric diversity without allowing the fallback lattice to
    # explode when each pilot is evaluated against all candidate windows.
    candidates: list[PilotBuildResult] = []
    for terminal_inc_key, bucket in first_legs_by_terminal_inc.items():
        bucket.sort(key=lambda item: item[0])
        for _key, first_leg in bucket[:1]:
            parts = [vertical, first_leg.iloc[1:].copy()]
            current_md = float(first_leg["MD_m"].iloc[-1])
            current_inc = float(first_leg["INC_deg"].iloc[-1])
            current_azi = float(first_leg["AZI_deg"].iloc[-1])
            current_point = study_points[0]
            try:
                for segment_index, target in enumerate(study_points[1:], start=2):
                    leg = _pilot_build_hold_leg_to_target(
                        start=current_point,
                        target=target,
                        start_md_m=current_md,
                        start_inc_deg=current_inc,
                        start_azi_deg=current_azi,
                        segment_index=segment_index,
                        config=config,
                    )
                    parts.append(leg.iloc[1:].copy())
                    current_point = target
                    current_md = float(leg["MD_m"].iloc[-1])
                    current_inc = float(leg["INC_deg"].iloc[-1])
                    current_azi = float(leg["AZI_deg"].iloc[-1])
                stations = add_dls(pd.concat(parts, ignore_index=True))
            except (ValueError, PlanningError):
                continue
            max_dls = _finite_max(stations["DLS_deg_per_30m"])
            actual_max_inc = _finite_max(stations["INC_deg"])
            md_values = stations["MD_m"].to_numpy(dtype=float)
            if (
                max_dls > dls_limit + 1e-6
                or actual_max_inc > max_inc_deg + 1e-6
                or np.any(np.diff(md_values) <= SMALL)
            ):
                continue
            candidates.append(
                _pilot_result_from_reoriented_stations(
                    stations=stations,
                    surface=surface,
                    study_points=study_points,
                    md_first_target_m=float(first_leg["MD_m"].iloc[-1]),
                    target_azimuth_deg=target_azimuth_deg,
                    terminal_inc_deg=float(terminal_inc_key),
                    config=config,
                )
            )
    candidates.sort(
        key=lambda pilot: (
            float(pilot.md_total_m),
            float(pilot.summary.get("max_dls_total_deg_per_30m", 0.0)),
        )
    )
    return _select_diverse_reoriented_pilot_candidates(candidates, limit=12)


def _select_diverse_reoriented_pilot_candidates(
    candidates: list[PilotBuildResult],
    *,
    limit: int,
) -> list[PilotBuildResult]:
    if len(candidates) <= limit:
        return candidates

    selected: list[PilotBuildResult] = []
    selected_ids: set[int] = set()

    def add_candidate(pilot: PilotBuildResult) -> None:
        if len(selected) >= limit:
            return
        pilot_id = id(pilot)
        if pilot_id in selected_ids:
            return
        selected.append(pilot)
        selected_ids.add(pilot_id)

    for pilot in candidates[: min(4, limit)]:
        add_candidate(pilot)

    by_inc = sorted(
        candidates,
        key=lambda pilot: float(
            pilot.summary.get("pilot_reorientation_pl1_inc_deg", 0.0)
        ),
    )
    for idx in np.linspace(0, len(by_inc) - 1, num=limit, dtype=int):
        add_candidate(by_inc[int(idx)])
        if len(selected) >= limit:
            break

    for pilot in candidates:
        add_candidate(pilot)
        if len(selected) >= limit:
            break
    return selected


def _bounded_reoriented_window_candidates(
    candidate_groups: list[list[PilotWindow]],
) -> list[PilotWindow]:
    """Keep the joint fallback search broad without an unbounded solver grid."""

    selected: list[PilotWindow] = []
    for group_index, group in enumerate(candidate_groups):
        limit = 6 if group_index == 0 else 8
        if len(group) <= limit:
            sampled = group
        else:
            indices = np.linspace(0, len(group) - 1, limit, dtype=int)
            sampled = [group[int(index)] for index in indices]
        selected.extend(sampled)
    by_md: dict[float, PilotWindow] = {}
    for window in selected:
        by_md.setdefault(round(float(window.md_m), 6), window)
    return list(by_md.values())


def _pilot_cubic_leg(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    start_md_m: float,
    start_inc_deg: float,
    start_azi_deg: float,
    end_inc_deg: float,
    end_azi_deg: float,
    config: TrajectoryConfig,
) -> pd.DataFrame:
    xyz = _sample_pilot_cubic_bezier(
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        # This fallback is exported as a minimum-curvature survey.  A coarse
        # cubic sampling grid can otherwise accumulate visible XYZ drift when
        # consumers reconstruct coordinates from MD/INC/AZI.
        step_m=min(float(config.md_step_m), 10.0),
    )
    distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if len(xyz) < 2 or np.any(distances <= SMALL):
        raise ValueError("Перестроенный участок пилота содержит совпадающие станции.")
    md = float(start_md_m) + np.concatenate([[0.0], np.cumsum(distances)])
    stations = _build_signed_trajectory_stations(
        xyz=xyz,
        md=md,
    )
    stations["segment"] = "PILOT_BUILD_1"
    stations.loc[0, "INC_deg"] = float(start_inc_deg)
    stations.loc[0, "AZI_deg"] = _normalize_azimuth_deg(start_azi_deg)
    stations.loc[len(stations) - 1, "INC_deg"] = float(end_inc_deg)
    stations.loc[len(stations) - 1, "AZI_deg"] = _normalize_azimuth_deg(end_azi_deg)
    return add_dls(stations)


def _sample_pilot_cubic_bezier(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    step_m: float,
) -> np.ndarray:
    chord_m = float(np.linalg.norm(p3 - p0))
    control_m = float(
        np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
    )
    samples = max(int(math.ceil(max(chord_m, control_m) / max(step_m, 1.0))), 8)
    samples = min(samples, 1600)
    t = np.linspace(0.0, 1.0, samples + 1)
    omt = 1.0 - t
    xyz = (
        (omt**3)[:, None] * p0
        + (3.0 * omt * omt * t)[:, None] * p1
        + (3.0 * omt * t * t)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    if len(xyz) < 2:
        raise ValueError("Недостаточно станций перестроенного пилота.")
    keep = np.ones(len(xyz), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1) > SMALL
    return xyz[keep]


def _build_signed_trajectory_stations(
    *,
    xyz: np.ndarray,
    md: np.ndarray,
) -> pd.DataFrame:
    xyz = np.asarray(xyz, dtype=float)
    md = np.asarray(md, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(md):
        raise ValueError("Некорректные станции перестроенного пилота.")
    if len(xyz) < 2:
        raise ValueError("Недостаточно станций перестроенного пилота.")
    if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(md)):
        raise ValueError("Перестроенный пилот содержит нечисловые станции.")
    if np.any(np.diff(md) <= SMALL):
        raise ValueError("MD перестроенного пилота должен строго возрастать.")

    tangents = _signed_station_tangent_vectors(xyz=xyz, md=md)
    horizontal = np.hypot(tangents[:, 0], tangents[:, 1])
    inc = np.degrees(np.arctan2(horizontal, tangents[:, 2]))
    azi = np.zeros(len(xyz), dtype=float)
    previous_azi = 0.0
    for idx, horizontal_component in enumerate(horizontal):
        if horizontal_component > SMALL:
            previous_azi = _normalize_azimuth_deg(
                math.degrees(
                    math.atan2(float(tangents[idx, 0]), float(tangents[idx, 1]))
                )
            )
        azi[idx] = previous_azi

    return pd.DataFrame(
        {
            "MD_m": md,
            "INC_deg": inc,
            "AZI_deg": azi,
            "X_m": xyz[:, 0],
            "Y_m": xyz[:, 1],
            "Z_m": xyz[:, 2],
        }
    )


def _signed_station_tangent_vectors(
    *,
    xyz: np.ndarray,
    md: np.ndarray,
) -> np.ndarray:
    segment_md = np.diff(md)
    segment_vectors = np.diff(xyz, axis=0) / segment_md[:, None]
    tangents = np.empty_like(xyz, dtype=float)
    tangents[0] = segment_vectors[0]
    tangents[-1] = segment_vectors[-1]
    if len(xyz) > 2:
        span_md = (md[2:] - md[:-2])[:, None]
        tangents[1:-1] = (xyz[2:] - xyz[:-2]) / span_md
    norms = np.linalg.norm(tangents, axis=1)
    if np.any(norms <= SMALL):
        raise ValueError("Перестроенный пилот содержит нулевую касательную.")
    return tangents / norms[:, None]


def _pilot_result_from_reoriented_stations(
    *,
    stations: pd.DataFrame,
    surface: Point3D,
    study_points: tuple[Point3D, ...],
    md_first_target_m: float,
    target_azimuth_deg: float,
    terminal_inc_deg: float,
    config: TrajectoryConfig,
) -> PilotBuildResult:
    md_total_m = float(stations["MD_m"].iloc[-1])
    max_dls = _finite_max(stations["DLS_deg_per_30m"])
    summary: SummaryDict = {
        "trajectory_type": "PILOT",
        "trajectory_target_direction": "Пилотный ствол",
        "well_complexity": "Пилот; перестроен под окно зарезки",
        "horizontal_length_m": 0.0,
        "entry_inc_deg": 0.0,
        "hold_inc_deg": 0.0,
        "max_dls_total_deg_per_30m": max_dls,
        "md_total_m": md_total_m,
        "max_total_md_postcheck_m": float(config.max_total_md_postcheck_m),
        "md_postcheck_excess_m": max(
            0.0, md_total_m - float(config.max_total_md_postcheck_m)
        ),
        "dls_postcheck_excess_deg_per_30m": max(
            0.0, max_dls - float(config.dls_build_max_deg_per_30m)
        ),
        "solver_turn_restarts_used": 0.0,
        "solver_turn_max_restarts": float(config.turn_solver_max_restarts),
        "pilot_target_count": float(len(study_points)),
        "kop_md_m": float(
            stations.loc[stations["segment"] == "VERTICAL", "MD_m"].max()
        ),
        "pilot_reorientation_fallback_used": "yes",
        "pilot_reorientation_target_azi_deg": float(target_azimuth_deg),
        "pilot_reorientation_pl1_inc_deg": float(terminal_inc_deg),
    }
    return PilotBuildResult(
        stations=stations,
        surface=surface,
        first_target=study_points[0],
        final_target=study_points[-1],
        md_first_target_m=float(md_first_target_m),
        md_total_m=md_total_m,
        azimuth_deg=_first_valid_azimuth_deg(stations),
        summary=summary,
    )


def _unique_floats(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if not math.isfinite(float(value)):
            continue
        if any(abs(float(value) - existing) <= 1e-6 for existing in result):
            continue
        result.append(float(value))
    return result


def _azimuth_difference_deg(left: float, right: float) -> float:
    return float(abs((float(left) - float(right) + 180.0) % 360.0 - 180.0))


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
    groups = _sidetrack_window_candidate_groups(
        pilot_name=pilot_name,
        parent_name=parent_name,
        pilot_stations=pilot_stations,
        parent_t1=parent_t1,
        config=config,
    )
    return [window for group in groups for window in group]


def _sidetrack_window_candidate_groups(
    *,
    pilot_name: str,
    parent_name: str,
    pilot_stations: pd.DataFrame,
    parent_t1: Point3D,
    config: TrajectoryConfig,
) -> list[list[PilotWindow]]:
    if pilot_stations.empty:
        return []
    required_columns = {"MD_m", "X_m", "Y_m", "Z_m", "INC_deg", "AZI_deg"}
    if not required_columns.issubset(pilot_stations.columns):
        raise ValueError(
            "Инклинометрия пилота не содержит MD/X/Y/Z/INC/AZI для поиска окна."
        )
    stations = pilot_stations.copy().reset_index(drop=True)
    try:
        numeric = stations[list(required_columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Инклинометрия пилота содержит нечисловую станцию.") from exc
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(numeric))
        or float(md_values[0]) < -SMALL
        or (len(md_values) > 1 and np.any(np.diff(md_values) <= SMALL))
    ):
        raise ValueError(
            "Инклинометрия пилота содержит нечисловой, отрицательный "
            "или не возрастающий MD."
        )
    if len(stations) < 2:
        return []
    preferred = _preferred_sidetrack_window_rows(
        stations=stations,
        parent_t1=parent_t1,
        config=config,
    )
    preferred_windows = (
        _pilot_windows_from_rows(
            preferred,
            pilot_name=pilot_name,
            parent_name=parent_name,
        )
        if not preferred.empty
        else []
    )

    vertical_room = float(parent_t1.z) - stations["Z_m"].to_numpy(dtype=float)
    min_room = max(
        float(config.kop_min_vertical_m), float(config.min_structural_segment_m)
    )
    min_window_md = max(
        float(config.kop_min_vertical_m), float(config.min_structural_segment_m)
    )
    first_target_md = _first_pilot_target_md_m(stations)
    before_first_target = np.ones(len(stations), dtype=bool)
    if first_target_md is not None:
        # Automatic windows must leave the engineering minimum distance above
        # PL1.  A closer or lower window remains possible only as an explicit
        # manual constraint.
        before_first_target = (
            stations["MD_m"].to_numpy(dtype=float)
            <= first_target_md - SIDETRACK_WINDOW_ABOVE_FIRST_TARGET_MIN_M + SMALL
        )
    eligible = stations.loc[
        (vertical_room >= min_room)
        & (stations["MD_m"].to_numpy(dtype=float) >= min_window_md)
        & before_first_target
    ].copy()
    if eligible.empty:
        fallback_min_md = max(
            float(config.min_structural_segment_m),
            float(config.md_step_m),
        )
        eligible = stations.loc[
            (stations["MD_m"].to_numpy(dtype=float) >= fallback_min_md)
            & before_first_target
        ].copy()
        if first_target_md is None:
            eligible = eligible.iloc[:-1].copy()
    if eligible.empty:
        return [preferred_windows] if preferred_windows else []
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
    expanded_windows = _pilot_windows_from_rows(
        selected_rows,
        pilot_name=pilot_name,
        parent_name=parent_name,
    )
    if preferred_windows:
        preferred_md = {round(float(window.md_m), 6) for window in preferred_windows}
        expanded_windows = [
            window
            for window in expanded_windows
            if round(float(window.md_m), 6) not in preferred_md
        ]
    return [group for group in (preferred_windows, expanded_windows) if group]


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
    try:
        numeric = stations[list(required)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Ручное окно зарезки невозможно: инклинометрия пилота содержит "
            "нечисловую станцию."
        ) from exc
    stations = stations.reset_index(drop=True)
    if len(stations) < 2:
        raise ValueError(
            "Ручное окно зарезки невозможно: у пилота меньше двух станций."
        )
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(numeric))
        or float(md_values[0]) < -SMALL
        or np.any(np.diff(md_values) <= SMALL)
    ):
        raise ValueError(
            "Ручное окно зарезки невозможно: MD пилота должен быть конечным, "
            "неотрицательным и строго возрастающим."
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
    complete_result: PlannerResult | None = None,
    optimization_context: AntiCollisionOptimizationContext | None = None,
) -> float:
    if result.stations.empty or len(result.stations) < 2:
        return float("inf")
    required_columns = {"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"}
    missing_columns = required_columns.difference(result.stations.columns)
    if missing_columns:
        raise ValueError(
            "Результат бокового ствола не содержит колонки "
            f"{', '.join(sorted(missing_columns))}."
        )
    try:
        station_values = result.stations[
            ["MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m"]
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Результат бокового ствола содержит нечисловую станцию."
        ) from exc
    if np.any(~np.isfinite(station_values)):
        raise ValueError("Результат бокового ствола содержит нечисловую станцию.")
    if abs(float(station_values[0, 0])) > 1e-6:
        raise ValueError("Локальный MD бокового ствола должен начинаться с 0 м.")
    start_miss_m = float(
        np.linalg.norm(
            station_values[0, 3:6]
            - np.asarray(
                [float(window.point.x), float(window.point.y), float(window.point.z)],
                dtype=float,
            )
        )
    )
    if start_miss_m > 1e-3:
        raise ValueError(
            "Результат бокового ствола не начинается в выбранном окне зарезки."
        )
    if (
        abs(float(station_values[0, 1]) - float(window.inc_deg)) > 1e-3
        or _azimuth_difference_deg(
            float(station_values[0, 2]),
            float(window.azi_deg),
        )
        > 1e-3
    ):
        raise ValueError(
            "Ориентация результата бокового ствола не совпадает с окном зарезки."
        )
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
    scoring_result = complete_result if complete_result is not None else result
    summary_dls = max(
        _finite_float_or_none(result.summary.get("max_dls_total_deg_per_30m")) or 0.0,
        _finite_float_or_none(scoring_result.summary.get("max_dls_total_deg_per_30m"))
        or 0.0,
    )
    station_dls = max(
        _finite_max(result.stations["DLS_deg_per_30m"])
        if "DLS_deg_per_30m" in result.stations.columns
        else 0.0,
        _finite_max(scoring_result.stations["DLS_deg_per_30m"])
        if "DLS_deg_per_30m" in scoring_result.stations.columns
        else 0.0,
    )
    planned_dls = max(junction_dls, summary_dls, station_dls)
    sidetrack_md_m = (
        _complete_sidetrack_lateral_md_m(window=window, result=complete_result)
        if complete_result is not None
        else _planner_result_md_total_m(result)
    )
    dls_limit = (
        _finite_float_or_none(
            scoring_result.summary.get("build_dls_max_config_deg_per_30m")
        )
        or _finite_float_or_none(result.summary.get("build_dls_max_config_deg_per_30m"))
        or 0.0
    )
    dls_excess = max(0.0, planned_dls - dls_limit) if dls_limit > SMALL else 0.0
    score = sidetrack_md_m + 300.0 * planned_dls + 100_000.0 * dls_excess
    if optimization_context is not None:
        score += (
            _trajectory_anticollision_penalty(
                stations=complete_result.stations,
                optimization_context=optimization_context,
            )
            if complete_result is not None
            else _sidetrack_anticollision_penalty(
                result=result,
                window=window,
                optimization_context=optimization_context,
            )
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
    return _trajectory_anticollision_penalty(
        stations=shifted_stations,
        optimization_context=optimization_context,
    )


def _trajectory_anticollision_penalty(
    *,
    stations: pd.DataFrame,
    optimization_context: AntiCollisionOptimizationContext | None,
) -> float:
    if optimization_context is None:
        return 0.0
    if stations.empty:
        return 10_000.0
    required_columns = {"MD_m", "X_m", "Y_m", "Z_m"}
    if not required_columns.issubset(stations.columns):
        return 10_000.0
    try:
        clearance = evaluate_stations_anti_collision_clearance(
            stations=stations,
            context=optimization_context,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return 10_000.0
    try:
        sf_target = float(optimization_context.sf_target)
        min_sf = float(clearance.min_separation_factor)
        max_overlap_m = float(clearance.max_overlap_depth_m)
    except (TypeError, ValueError):
        return 10_000.0
    if (
        not math.isfinite(sf_target)
        or not math.isfinite(min_sf)
        or not math.isfinite(max_overlap_m)
    ):
        return 10_000.0
    sf_deficit = max(0.0, sf_target - min_sf)
    overlap_penalty_m = max(0.0, max_overlap_m)
    return float(sf_deficit * 1000.0 + overlap_penalty_m * 0.1)


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
