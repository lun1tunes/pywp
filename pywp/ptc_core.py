from __future__ import annotations

import colorsys
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pywp import TrajectoryConfig, TrajectoryPlanner
from pywp.actual_fund_analysis import (
    ZONE_BUILD2,
    ZONE_HOLD,
    ZONE_HORIZONTAL,
    ActualFundKopDepthFunction,
    ActualFundWellAnalysis,
    actual_fund_depth_rows,
    actual_fund_metrics_rows,
    actual_fund_pad_rows,
    build_actual_fund_kop_depth_function,
    build_actual_fund_well_analyses,
    summarize_actual_fund_by_depth,
)
from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionZone,
    anti_collision_method_caption,
    anti_collision_report_rows,
    collision_corridor_plan_polygon,
    collision_corridor_point_sphere_mesh,
    collision_corridor_tube_mesh,
)
from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
    evaluate_stations_anti_collision_clearance,
)
from pywp.anticollision_recommendations import (
    AntiCollisionRecommendation,
    AntiCollisionRecommendationCluster,
    AntiCollisionWellContext,
    anti_collision_cluster_rows,
    anti_collision_recommendation_rows,
    build_anti_collision_recommendation_clusters,
    build_anti_collision_recommendations,
    cluster_display_label,
    recommendation_display_label,
)
from pywp.anticollision_rerun import (
    build_anti_collision_analysis_for_successes as build_anti_collision_analysis_for_successes_shared,
)
from pywp.anticollision_rerun import (
    build_anticollision_well_contexts as build_anticollision_well_contexts_shared,
)
from pywp.anticollision_rerun import (
    build_cluster_prepared_overrides as build_cluster_prepared_overrides_shared,
)
from pywp.anticollision_rerun import (
    build_prepared_optimization_context as build_prepared_optimization_context_shared,
)
from pywp.anticollision_rerun import (
    build_recommendation_prepared_overrides as build_recommendation_prepared_overrides_shared,
)
from pywp.anticollision_rerun import (
    recommendation_intervals_for_moving_well as recommendation_intervals_for_moving_well_shared,
)
from pywp.constants import SMALL
from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackPoint,
    WelltrackRecord,
    decode_welltrack_bytes,
    parse_welltrack_points_table,
    parse_welltrack_text,
    welltrack_points_to_targets,
)
from pywp.models import OPTIMIZATION_ANTI_COLLISION_AVOIDANCE, Point3D
from pywp.planner_config import optimization_display_label
from pywp.plot_axes import (
    equalized_axis_ranges,
    equalized_xy_ranges,
    linear_tick_values,
    nice_tick_step,
    reversed_axis_range,
)
from pywp.plotly_config import (
    DEFAULT_3D_CAMERA,
    trajectory_plotly_chart_config,
)
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    REFERENCE_WELL_KIND_COLORS,
    REFERENCE_WELL_KIND_LABELS,
    ImportedTrajectoryWell,
    parse_reference_trajectory_text_with_kind,
    parse_reference_trajectory_welltrack_text,
    reference_well_display_label,
)
from pywp.solver_diagnostics import summarize_problem_ru
from pywp.three_viewer import render_local_three_scene
from pywp.ui_calc_params import (
    CalcParamBinding,
    clear_kop_min_vertical_function,
    kop_min_vertical_function_from_state,
    kop_min_vertical_mode,
    set_kop_min_vertical_function,
)
from pywp.ui_theme import render_small_note
from pywp.ui_utils import (
    arrow_safe_text_dataframe,
    dls_to_pi,
    format_run_log_line,
)
from pywp.ui_well_panels import render_run_log_panel
from pywp.ui_well_result import (
    SingleWellResultView,
    render_key_metrics,
    render_result_plots,
    render_result_tables,
)
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    UNCERTAINTY_PRESET_OPTIONS,
    PlanningUncertaintyModel,
    build_uncertainty_tube_mesh,
    normalize_uncertainty_preset,
    planning_uncertainty_model_for_preset,
    uncertainty_preset_label,
    uncertainty_ribbon_polygon,
)
from pywp.well_pad import (
    PAD_SURFACE_ANCHOR_CENTER,
    PAD_SURFACE_ANCHOR_FIRST,
    PadLayoutPlan,
    PadWell,
    WellPad,
    apply_pad_layout,
    estimate_pad_nds_azimuth_deg,
    ordered_pad_wells,
)
from pywp.welltrack_batch import (
    DynamicClusterExecutionContext,
    SuccessfulWellPlan,
    WelltrackBatchPlanner,
    ensure_successful_plan_baseline,
    merge_batch_results,
    rebuild_optimization_context,
    recommended_batch_selection,
)
from pywp.welltrack_quality import (
    detect_t1_t3_order_issues,
    swap_t1_t3_for_wells,
)

DEFAULT_WELLTRACK_PATH = Path("tests/test_data/WELLTRACKS4.INC")
WT_UI_DEFAULTS_VERSION = 16
WT_LOG_COMPACT = "Краткий"
WT_LOG_VERBOSE = "Подробный"
WT_LOG_LEVEL_OPTIONS: tuple[str, ...] = (WT_LOG_COMPACT, WT_LOG_VERBOSE)
WT_T1T3_MIN_DELTA_M = 0.5
WT_3D_RENDER_AUTO = "Авто"
WT_3D_RENDER_DETAIL = "Детально"
WT_3D_RENDER_FAST = "Быстро"
WT_3D_BACKEND_PLOTLY = "Plotly"
WT_3D_BACKEND_THREE_LOCAL = "Three.js (локально, экспериментально)"
REFERENCE_LABEL_ACTUAL_COLOR = "#111111"
REFERENCE_LABEL_APPROVED_COLOR = "#C62828"
REFERENCE_PAD_LABEL_COLOR = "#334155"
REFERENCE_LABEL_HORIZONTAL_INC_THRESHOLD_DEG = 80.0
REFERENCE_LABEL_HORIZONTAL_MIN_INTERVAL_M = 100.0
REFERENCE_PAD_GROUP_DISTANCE_M = 300.0
WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M = 400.0
WT_3D_RENDER_OPTIONS: tuple[str, ...] = (
    WT_3D_RENDER_DETAIL,
    WT_3D_RENDER_FAST,
)
WT_3D_BACKEND_OPTIONS: tuple[str, ...] = (
    WT_3D_BACKEND_THREE_LOCAL,
    WT_3D_BACKEND_PLOTLY,
)
WT_3D_FAST_REFERENCE_WELL_THRESHOLD = 10
WT_3D_FAST_REFERENCE_POINT_THRESHOLD = 1200
WT_3D_FAST_CALC_WELL_THRESHOLD = 14
WT_3D_FAST_REFERENCE_TARGET_POINTS = 72
WT_3D_FAST_CALC_TARGET_POINTS = 180
WT_3D_FAST_REFERENCE_CONE_WELL_LIMIT = 6
WT_3D_REFERENCE_CONE_FOCUS_DISTANCE_M = 500.0
WT_THREE_MAX_HOVER_POINTS_PER_TRACE = 96
WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE = 24
WT_THREE_MAX_LABELS = 48
WT_THREE_MAX_REFERENCE_LABELS = 12
WT_PAD_FOCUS_ALL = "__all_pads__"
WT_IMPORT_WELLHEAD_Z_TOLERANCE_M = 100.0
_WT_LEGACY_KEY_ALIASES: dict[str, str] = {
    "wt_cfg_md_step_m": "wt_cfg_md_step",
    "wt_cfg_md_step_control_m": "wt_cfg_md_control",
    "wt_cfg_entry_inc_target_deg": "wt_cfg_entry_inc_target",
    "wt_cfg_entry_inc_tolerance_deg": "wt_cfg_entry_inc_tol",
    "wt_cfg_max_inc_deg": "wt_cfg_max_inc",
    "wt_cfg_max_total_md_postcheck_m": "wt_cfg_max_total_md_postcheck",
    "wt_cfg_kop_min_vertical_m": "wt_cfg_kop_min_vertical",
}
WT_CALC_PARAMS = CalcParamBinding(prefix="wt_cfg_")
DEFAULT_PAD_SPACING_M = 20.0
DEFAULT_PAD_SURFACE_ANCHOR_MODE = PAD_SURFACE_ANCHOR_CENTER
_BATCH_SUMMARY_RENAME_COLUMNS: dict[str, str] = {
    "Рестарты решателя": "Рестарты",
    "Классификация целей": "Цели",
    "Длина ГС, м": "ГС, м",
}
_BATCH_SUMMARY_DISPLAY_ORDER: tuple[str, ...] = (
    "Скважина",
    "Точек",
    "Цели",
    "Сложность",
    "Отход t1, м",
    "Мин VERTICAL до KOP, м",
    "KOP MD, м",
    "ГС, м",
    "INC в t1, deg",
    "ЗУ HOLD, deg",
    "Макс ПИ, deg/10m",
    "Макс MD, м",
    "Рестарты",
    "Статус",
    "Проблема",
    "Модель траектории",
)


@dataclass(frozen=True)
class _BatchRunRequest:
    selected_names: list[str]
    config: TrajectoryConfig
    run_clicked: bool


@dataclass(frozen=True)
class _WelltrackSourcePayload:
    mode: str
    source_text: str = ""
    table_rows: pd.DataFrame | None = None


@dataclass(frozen=True)
class _TargetOnlyWell:
    name: str
    surface: Point3D
    t1: Point3D
    t3: Point3D
    status: str
    problem: str


@dataclass(frozen=True)
class _DetectedPadUiMeta:
    source_surfaces_defined: bool
    inferred_spacing_m: float
    source_surface_x_m: float
    source_surface_y_m: float
    source_surface_z_m: float
    source_surface_count: int


def _build_well_color_palette() -> tuple[str, ...]:
    """Return a long palette with high local contrast for adjacent wells.

    The hue order intentionally jumps around the wheel instead of walking
    sequentially, so neighboring indices remain visually distinct. Reds,
    blacks and grays are excluded on purpose to keep collision overlays and
    reference wells semantically separate.
    """

    leading_colors: tuple[str, ...] = (
        "#15D562",
        "#1562D5",
        "#F59E0B",
        "#AF15D5",
        "#00A0A0",
        "#D54A9A",
        "#62D515",
        "#8A5CF6",
    )
    hue_degrees: tuple[float, ...] = (
        18.0,
        42.0,
        72.0,
        102.0,
        132.0,
        162.0,
        192.0,
        222.0,
        252.0,
        282.0,
        312.0,
    )
    hue_jump_order: tuple[int, ...] = (0, 5, 10, 4, 9, 3, 8, 2, 7, 1, 6)
    lightness_saturation_bands: tuple[tuple[float, float], ...] = (
        (0.47, 0.82),
        (0.61, 0.74),
        (0.39, 0.88),
        (0.55, 0.9),
    )
    ordered_hues = tuple(hue_degrees[index] for index in hue_jump_order)
    colors: list[str] = list(leading_colors)
    for lightness, saturation in lightness_saturation_bands:
        for hue_deg in ordered_hues:
            red, green, blue = colorsys.hls_to_rgb(
                hue_deg / 360.0,
                lightness,
                saturation,
            )
            candidate = "#{:02X}{:02X}{:02X}".format(
                int(round(red * 255.0)),
                int(round(green * 255.0)),
                int(round(blue * 255.0)),
            )
            if candidate not in colors:
                colors.append(candidate)
    return tuple(colors)


WELL_COLOR_PALETTE: tuple[str, ...] = _build_well_color_palette()


def _well_color(index: int) -> str:
    return WELL_COLOR_PALETTE[index % len(WELL_COLOR_PALETTE)]


def _well_color_map(records: list[WelltrackRecord]) -> dict[str, str]:
    return {
        str(record.name): _well_color(index)
        for index, record in enumerate(records)
    }


def _failed_target_only_wells(
    *,
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]],
) -> list[_TargetOnlyWell]:
    rows_by_name = {
        str(row.get("Скважина", "")).strip(): row
        for row in summary_rows
        if str(row.get("Скважина", "")).strip()
    }
    target_only_wells: list[_TargetOnlyWell] = []
    for record in records:
        row = rows_by_name.get(str(record.name))
        if row is None:
            continue
        status = str(row.get("Статус", "")).strip()
        if status in {"OK", "Не рассчитана"}:
            continue
        try:
            surface, t1, t3 = welltrack_points_to_targets(record.points)
        except ValueError:
            continue
        target_only_wells.append(
            _TargetOnlyWell(
                name=str(record.name),
                surface=surface,
                t1=t1,
                t3=t3,
                status=status or "Ошибка расчета",
                problem=str(row.get("Проблема", "")).strip(),
            )
        )
    return target_only_wells


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(90, 90, 90, {float(alpha):.3f})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {float(alpha):.3f})"


def _lighten_hex(hex_color: str, blend_with_white: float = 0.38) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return "#A0A0A0"
    blend = float(np.clip(blend_with_white, 0.0, 1.0))
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    red = int(round(red + (255 - red) * blend))
    green = int(round(green + (255 - green) * blend))
    blue = int(round(blue + (255 - blue) * blend))
    return f"#{red:02X}{green:02X}{blue:02X}"


def _reference_point_count(
    reference_wells: Iterable[ImportedTrajectoryWell],
) -> int:
    return int(
        sum(
            len(well.stations.index)
            for well in reference_wells
            if not well.stations.empty
        )
    )


def _resolve_3d_render_mode(
    *,
    requested_mode: str,
    calculated_well_count: int,
    reference_wells: Iterable[ImportedTrajectoryWell],
) -> str:
    normalized = str(requested_mode).strip()
    reference_wells_tuple = tuple(reference_wells)
    if normalized == WT_3D_RENDER_DETAIL:
        if (
            calculated_well_count >= WT_3D_FAST_CALC_WELL_THRESHOLD
            or len(reference_wells_tuple)
            >= WT_3D_FAST_REFERENCE_WELL_THRESHOLD
            or _reference_point_count(reference_wells_tuple)
            >= WT_3D_FAST_REFERENCE_POINT_THRESHOLD
        ):
            return WT_3D_RENDER_FAST
        return WT_3D_RENDER_DETAIL
    return WT_3D_RENDER_FAST


def _is_reference_trace_name(trace_name: str) -> bool:
    normalized = str(trace_name).strip()
    if not normalized:
        return False
    return bool(
        "(Фактическая)" in normalized
        or "(Проектная утвержденная)" in normalized
        or "Фактические скважины" in normalized
        or "Проектные утвержденные скважины" in normalized
    )


def _decimate_hover_payload(
    *,
    points: list[list[float]],
    hover_items: list[dict[str, object]],
    max_points: int,
) -> tuple[list[list[float]], list[dict[str, object]]]:
    if max_points <= 1 or len(points) <= max_points:
        return points, hover_items
    indices = np.unique(
        np.linspace(0, len(points) - 1, num=int(max_points), dtype=int)
    ).tolist()
    return (
        [points[index] for index in indices],
        [hover_items[index] for index in indices],
    )


def _decimated_station_frame(
    stations: pd.DataFrame, *, target_points: int
) -> pd.DataFrame:
    if target_points <= 2 or len(stations.index) <= target_points:
        return stations
    indices = np.linspace(
        0,
        len(stations.index) - 1,
        num=int(target_points),
        dtype=int,
    )
    unique_indices = np.unique(indices)
    return stations.iloc[unique_indices].reset_index(drop=True)


def _nan_separated_xyz_segments(
    polylines_xyz: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    for polyline_x, polyline_y, polyline_z in polylines_xyz:
        if len(polyline_x) == 0:
            continue
        x_values.extend(np.asarray(polyline_x, dtype=float).tolist())
        y_values.extend(np.asarray(polyline_y, dtype=float).tolist())
        z_values.extend(np.asarray(polyline_z, dtype=float).tolist())
        x_values.append(np.nan)
        y_values.append(np.nan)
        z_values.append(np.nan)
    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(z_values, dtype=float),
    )


def _combined_reference_trace_3d(
    *,
    reference_wells: Iterable[object],
    kind: str,
    target_points_per_well: int,
) -> go.Scatter3d | None:
    matching = [
        well
        for well in reference_wells
        if str(getattr(well, "kind", getattr(well, "well_kind", "")))
        == str(kind)
        and not getattr(well, "stations").empty
    ]
    if not matching:
        return None
    x_values, y_values, z_values = _nan_separated_xyz_segments(
        (
            tuple(
                _decimated_station_frame(
                    well.stations,
                    target_points=target_points_per_well,
                )[column].to_numpy(dtype=float)
                for column in ("X_m", "Y_m", "Z_m")
            )
            for well in matching
        )
    )
    kind_label = REFERENCE_WELL_KIND_LABELS.get(str(kind), str(kind))
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        name=f"{kind_label} (сводно)",
        showlegend=False,
        line={
            "width": 2.5,
            "color": REFERENCE_WELL_KIND_COLORS.get(str(kind), "#A0A0A0"),
        },
        hoverinfo="skip",
    )


def _anticollision_focus_reference_names(
    analysis: AntiCollisionAnalysis,
) -> set[str]:
    reference_names = {
        str(well.name)
        for well in analysis.wells
        if bool(well.is_reference_only)
    }
    focus_names: set[str] = set()
    for corridor in analysis.corridors:
        if str(corridor.well_a) in reference_names:
            focus_names.add(str(corridor.well_a))
        if str(corridor.well_b) in reference_names:
            focus_names.add(str(corridor.well_b))
    return focus_names


def _xy_bounds_from_stations(
    stations: pd.DataFrame,
) -> tuple[float, float, float, float] | None:
    required = {"X_m", "Y_m"}
    if stations.empty or not required.issubset(stations.columns):
        return None
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    if len(x_values) == 0 or len(y_values) == 0:
        return None
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not finite_mask.any():
        return None
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    return (
        float(np.min(x_values)),
        float(np.max(x_values)),
        float(np.min(y_values)),
        float(np.max(y_values)),
    )


def _xy_bounds_gap_m(
    bounds_a: tuple[float, float, float, float],
    bounds_b: tuple[float, float, float, float],
) -> float:
    min_x_a, max_x_a, min_y_a, max_y_a = bounds_a
    min_x_b, max_x_b, min_y_b, max_y_b = bounds_b
    dx = max(min_x_a - max_x_b, min_x_b - max_x_a, 0.0)
    dy = max(min_y_a - max_y_b, min_y_b - max_y_a, 0.0)
    return float(np.hypot(dx, dy))


def _anticollision_reference_cone_focus_names(
    analysis: AntiCollisionAnalysis,
) -> set[str]:
    focus_names = set(_anticollision_focus_reference_names(analysis))
    calculated_bounds = [
        bounds
        for well in analysis.wells
        if not bool(well.is_reference_only)
        for bounds in [_xy_bounds_from_stations(well.stations)]
        if bounds is not None
    ]
    if not calculated_bounds:
        return focus_names
    for well in analysis.wells:
        if not bool(well.is_reference_only):
            continue
        reference_bounds = _xy_bounds_from_stations(well.stations)
        if reference_bounds is None:
            continue
        if any(
            _xy_bounds_gap_m(reference_bounds, calculated_bounds_item)
            <= WT_3D_REFERENCE_CONE_FOCUS_DISTANCE_M
            for calculated_bounds_item in calculated_bounds
        ):
            focus_names.add(str(well.name))
    return focus_names


def _t1_name_trace_3d(
    *, well_name: str, t1: Point3D, color: str
) -> go.Scatter3d:
    return go.Scatter3d(
        x=[float(t1.x)],
        y=[float(t1.y)],
        z=[float(t1.z)],
        mode="text",
        text=[str(well_name)],
        textposition="top center",
        name=f"{well_name}: t1 label",
        showlegend=False,
        textfont={"color": str(color), "size": 12},
        hoverinfo="skip",
    )


def _t1_name_trace_2d(
    *,
    well_name: str,
    x_value: float,
    y_value: float,
    color: str,
) -> go.Scatter:
    return go.Scatter(
        x=[float(x_value)],
        y=[float(y_value)],
        mode="text",
        text=[str(well_name)],
        textposition="top center",
        name=f"{well_name}: t1 label",
        showlegend=False,
        textfont={"color": str(color), "size": 12},
        hoverinfo="skip",
    )


def _reference_label_color(kind: str) -> str:
    if str(kind) == REFERENCE_WELL_APPROVED:
        return REFERENCE_LABEL_APPROVED_COLOR
    return REFERENCE_LABEL_ACTUAL_COLOR


def _reference_legend_label(kind: str) -> str:
    if str(kind) == REFERENCE_WELL_APPROVED:
        return "Проектные утвержденные скважины"
    return "Фактические скважины"


def _reference_legend_color(kind: str) -> str:
    return str(
        REFERENCE_WELL_KIND_COLORS.get(
            str(kind),
            "#A0A0A0",
        )
    )


def _reference_kind_value(reference_well: object) -> str:
    return str(
        getattr(
            reference_well, "kind", getattr(reference_well, "well_kind", "")
        )
    )


def _reference_kinds_present(
    reference_wells: Iterable[object],
) -> tuple[str, ...]:
    kinds: list[str] = []
    seen: set[str] = set()
    for reference_well in reference_wells:
        kind = _reference_kind_value(reference_well)
        if not kind or kind in seen:
            continue
        seen.add(kind)
        kinds.append(kind)
    return tuple(kinds)


def _reference_legend_trace_3d(kind: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        mode="lines",
        name=_reference_legend_label(kind),
        line={"width": 4, "color": _reference_legend_color(kind)},
        hoverinfo="skip",
        legendgroup=f"reference_kind_{str(kind)}",
        visible="legendonly",
    )


def _reference_legend_trace_2d(kind: str) -> go.Scatter:
    return go.Scatter(
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        mode="lines",
        name=_reference_legend_label(kind),
        line={"width": 4, "color": _reference_legend_color(kind)},
        hoverinfo="skip",
        legendgroup=f"reference_kind_{str(kind)}",
        visible="legendonly",
    )


def _reference_label_anchor_point(
    reference_well: ImportedTrajectoryWell,
) -> tuple[float, float, float] | None:
    stations = reference_well.stations
    required = {"MD_m", "INC_deg", "X_m", "Y_m", "Z_m"}
    if stations.empty or not required.issubset(stations.columns):
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    z_values = stations["Z_m"].to_numpy(dtype=float)
    if len(md_values) == 0:
        return None
    high_angle_mask = inc_values >= float(
        REFERENCE_LABEL_HORIZONTAL_INC_THRESHOLD_DEG
    )
    if bool(high_angle_mask[-1]):
        start_index = len(high_angle_mask) - 1
        while start_index > 0 and bool(high_angle_mask[start_index - 1]):
            start_index -= 1
        if float(md_values[-1] - md_values[start_index]) >= float(
            REFERENCE_LABEL_HORIZONTAL_MIN_INTERVAL_M
        ):
            return (
                float(x_values[start_index]),
                float(y_values[start_index]),
                float(z_values[start_index]),
            )
    return (
        float(x_values[-1]),
        float(y_values[-1]),
        float(z_values[-1]),
    )


@dataclass(frozen=True)
class _ReferencePadLabel:
    label: str
    x: float
    y: float
    z: float


def _reference_pad_numeric_id(well_name: str) -> str | None:
    digits: list[str] = []
    for char in str(well_name).strip():
        if char.isdigit():
            digits.append(char)
            continue
        break
    if not digits:
        return None
    digit_text = "".join(digits)
    if len(digit_text) > 2:
        digit_text = digit_text[:-2]
    normalized = digit_text.lstrip("0")
    return normalized or "0"


def _reference_pad_labels(
    reference_wells: Iterable[ImportedTrajectoryWell],
) -> list[_ReferencePadLabel]:
    ordered_wells = [
        well
        for well in reference_wells
        if np.isfinite(float(well.surface.x))
        and np.isfinite(float(well.surface.y))
    ]
    if not ordered_wells:
        return []

    cell_size = float(REFERENCE_PAD_GROUP_DISTANCE_M)
    parents = list(range(len(ordered_wells)))
    ranks = [0] * len(ordered_wells)

    def _find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root == right_root:
            return
        if ranks[left_root] < ranks[right_root]:
            parents[left_root] = right_root
            return
        if ranks[left_root] > ranks[right_root]:
            parents[right_root] = left_root
            return
        parents[right_root] = left_root
        ranks[left_root] += 1

    cells: dict[tuple[int, int], list[int]] = {}
    for index, well in enumerate(ordered_wells):
        cell = (
            int(np.floor(float(well.surface.x) / cell_size)),
            int(np.floor(float(well.surface.y) / cell_size)),
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for candidate_index in cells.get(
                    (cell[0] + dx, cell[1] + dy), ()
                ):
                    candidate = ordered_wells[candidate_index]
                    if float(
                        np.hypot(
                            float(well.surface.x) - float(candidate.surface.x),
                            float(well.surface.y) - float(candidate.surface.y),
                        )
                    ) <= float(REFERENCE_PAD_GROUP_DISTANCE_M):
                        _union(index, candidate_index)
        cells.setdefault(cell, []).append(index)

    component_indices: dict[int, list[int]] = {}
    for index in range(len(ordered_wells)):
        component_indices.setdefault(_find(index), []).append(index)

    pad_labels: list[_ReferencePadLabel] = []
    fallback_counter = 1
    for component in component_indices.values():
        ordered_component = sorted(component)
        anchor_well = ordered_wells[ordered_component[0]]
        numeric_ids = [
            pad_id
            for index in ordered_component
            for pad_id in [
                _reference_pad_numeric_id(str(ordered_wells[index].name))
            ]
            if pad_id is not None
        ]
        if numeric_ids:
            counts = {
                pad_id: numeric_ids.count(pad_id)
                for pad_id in set(numeric_ids)
            }
            pad_id = sorted(
                counts.keys(),
                key=lambda value: (
                    -counts[value],
                    numeric_ids.index(value),
                    value,
                ),
            )[0]
        else:
            if len(ordered_component) <= 1:
                continue
            pad_id = str(fallback_counter)
            fallback_counter += 1
        pad_labels.append(
            _ReferencePadLabel(
                label=f"Куст {pad_id}",
                x=float(anchor_well.surface.x),
                y=float(anchor_well.surface.y),
                z=float(anchor_well.surface.z),
            )
        )
    return pad_labels


def _reference_pad_label_trace_3d(
    reference_wells: Iterable[ImportedTrajectoryWell],
) -> go.Scatter3d | None:
    pad_labels = _reference_pad_labels(reference_wells)
    if not pad_labels:
        return None
    return go.Scatter3d(
        x=[item.x for item in pad_labels],
        y=[item.y for item in pad_labels],
        z=[item.z for item in pad_labels],
        mode="text",
        text=[item.label for item in pad_labels],
        textposition="top center",
        name="Reference pads: кусты",
        showlegend=False,
        textfont={"color": REFERENCE_PAD_LABEL_COLOR, "size": 12},
        hoverinfo="skip",
    )


def _reference_pad_label_trace_2d(
    reference_wells: Iterable[ImportedTrajectoryWell],
) -> go.Scatter | None:
    pad_labels = _reference_pad_labels(reference_wells)
    if not pad_labels:
        return None
    return go.Scatter(
        x=[item.x for item in pad_labels],
        y=[item.y for item in pad_labels],
        mode="text",
        text=[item.label for item in pad_labels],
        textposition="top center",
        name="Reference pads: кусты",
        showlegend=False,
        textfont={"color": REFERENCE_PAD_LABEL_COLOR, "size": 12},
        hoverinfo="skip",
    )


def _analysis_reference_wells(
    analysis: AntiCollisionAnalysis,
) -> list[ImportedTrajectoryWell]:
    return [
        ImportedTrajectoryWell(
            name=str(well.name),
            kind=str(well.well_kind),
            stations=well.stations,
            surface=Point3D(
                x=float(well.surface.x),
                y=float(well.surface.y),
                z=float(well.surface.z),
            ),
            azimuth_deg=0.0,
        )
        for well in analysis.wells
        if bool(well.is_reference_only)
    ]


def _reference_name_trace_3d(
    reference_wells: Iterable[ImportedTrajectoryWell],
    *,
    kind: str,
) -> go.Scatter3d | None:
    points: list[tuple[float, float, float, str]] = []
    for well in reference_wells:
        if str(well.kind) != str(kind):
            continue
        anchor = _reference_label_anchor_point(well)
        if anchor is None:
            continue
        points.append(
            (
                float(anchor[0]),
                float(anchor[1]),
                float(anchor[2]),
                str(well.name),
            )
        )
    if not points:
        return None
    return go.Scatter3d(
        x=[item[0] for item in points],
        y=[item[1] for item in points],
        z=[item[2] for item in points],
        mode="text",
        text=[item[3] for item in points],
        textposition="top center",
        name=f"{REFERENCE_WELL_KIND_LABELS.get(str(kind), str(kind))}: подписи",
        showlegend=False,
        textfont={"color": _reference_label_color(str(kind)), "size": 11},
        hoverinfo="skip",
    )


def _reference_name_trace_2d(
    reference_wells: Iterable[ImportedTrajectoryWell],
    *,
    kind: str,
) -> go.Scatter | None:
    points: list[tuple[float, float, str]] = []
    for well in reference_wells:
        if str(well.kind) != str(kind):
            continue
        anchor = _reference_label_anchor_point(well)
        if anchor is None:
            continue
        points.append((float(anchor[0]), float(anchor[1]), str(well.name)))
    if not points:
        return None
    return go.Scatter(
        x=[item[0] for item in points],
        y=[item[1] for item in points],
        mode="text",
        text=[item[2] for item in points],
        textposition="top center",
        name=f"{REFERENCE_WELL_KIND_LABELS.get(str(kind), str(kind))}: подписи",
        showlegend=False,
        textfont={"color": _reference_label_color(str(kind)), "size": 11},
        hoverinfo="skip",
    )


def _trace_showlegend(trace: object) -> bool:
    showlegend = getattr(trace, "showlegend", None)
    if showlegend is None:
        return bool(str(getattr(trace, "name", "") or "").strip())
    return bool(showlegend)


def _trace_visibility_state(trace: object) -> str:
    visible = getattr(trace, "visible", True)
    if visible is False:
        return "hidden"
    if str(visible) == "legendonly":
        return "legendonly"
    return "visible"


def _plotly_color_and_opacity(
    color_value: object,
    *,
    fallback_opacity: float = 1.0,
) -> tuple[str, float]:
    color_text = str(color_value or "").strip()
    if not color_text:
        return "#94A3B8", float(np.clip(fallback_opacity, 0.0, 1.0))
    if color_text.startswith("#"):
        return color_text, float(np.clip(fallback_opacity, 0.0, 1.0))
    if color_text.startswith("rgba(") and color_text.endswith(")"):
        raw = color_text[5:-1]
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 4:
            try:
                red = int(float(parts[0]))
                green = int(float(parts[1]))
                blue = int(float(parts[2]))
                alpha = float(parts[3])
                return (
                    f"#{red:02X}{green:02X}{blue:02X}",
                    float(np.clip(alpha, 0.0, 1.0)),
                )
            except ValueError:
                pass
    if color_text.startswith("rgb(") and color_text.endswith(")"):
        raw = color_text[4:-1]
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 3:
            try:
                red = int(float(parts[0]))
                green = int(float(parts[1]))
                blue = int(float(parts[2]))
                return (
                    f"#{red:02X}{green:02X}{blue:02X}",
                    float(np.clip(fallback_opacity, 0.0, 1.0)),
                )
            except ValueError:
                pass
    return color_text, float(np.clip(fallback_opacity, 0.0, 1.0))


def _trace_extra_name(trace: object) -> str:
    hovertemplate = str(getattr(trace, "hovertemplate", "") or "")
    start = hovertemplate.find("<extra>")
    end = hovertemplate.find("</extra>")
    if start >= 0 and end > start:
        return hovertemplate[start + len("<extra>") : end].strip()
    return ""


def _customdata_row_to_hover_item(
    customdata_row: object,
    *,
    fallback_name: str,
) -> dict[str, object]:
    if isinstance(customdata_row, np.ndarray):
        values = customdata_row.tolist()
    elif isinstance(customdata_row, (list, tuple)):
        values = list(customdata_row)
    elif customdata_row is None:
        values = []
    else:
        values = [customdata_row]
    item: dict[str, object] = {"name": str(fallback_name).strip()}
    if values:
        first = values[0]
        try:
            first_float = float(first)
        except (TypeError, ValueError):
            first_float = None
        if first_float is not None and np.isfinite(first_float):
            item["md"] = float(first_float)
        elif str(first).strip():
            item["point"] = str(first).strip()
    if len(values) >= 2:
        try:
            second_float = float(values[1])
        except (TypeError, ValueError):
            second_float = None
        if second_float is not None and np.isfinite(second_float):
            item["dls"] = float(second_float)
    if len(values) >= 3:
        try:
            third_float = float(values[2])
        except (TypeError, ValueError):
            third_float = None
        if third_float is not None and np.isfinite(third_float):
            item["inc"] = float(third_float)
    if len(values) >= 4 and str(values[3]).strip():
        item["segment"] = str(values[3]).strip()
    return item


def _split_nan_separated_xyz_segments(
    *,
    x_values: Iterable[object],
    y_values: Iterable[object],
    z_values: Iterable[object],
) -> list[list[list[float]]]:
    x_array = np.asarray(list(x_values), dtype=float)
    y_array = np.asarray(list(y_values), dtype=float)
    z_array = np.asarray(list(z_values), dtype=float)
    if not (len(x_array) == len(y_array) == len(z_array)):
        return []
    segments: list[list[list[float]]] = []
    current_segment: list[list[float]] = []
    for x_value, y_value, z_value in zip(
        x_array, y_array, z_array, strict=False
    ):
        if not (
            np.isfinite(float(x_value))
            and np.isfinite(float(y_value))
            and np.isfinite(float(z_value))
        ):
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
            continue
        current_segment.append(
            [float(x_value), float(y_value), float(z_value)]
        )
    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def _scatter3d_trace_to_three_payload(
    trace: go.Scatter3d,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "lines": [],
        "points": [],
        "labels": [],
        "legend_items": [],
    }
    mode = str(trace.mode or "")
    trace_name = str(trace.name or "").strip()
    extra_name = _trace_extra_name(trace)
    hover_name = trace_name or extra_name
    trace_opacity = (
        1.0
        if getattr(trace, "opacity", None) is None
        else float(trace.opacity)
    )
    visibility_state = _trace_visibility_state(trace)
    is_reference_trace = _is_reference_trace_name(trace_name)
    legend_added = False
    if "lines" in mode:
        line = trace.line or {}
        color, opacity = _plotly_color_and_opacity(
            getattr(line, "color", None),
            fallback_opacity=trace_opacity,
        )
        if visibility_state == "visible":
            x_array = np.asarray(
                list(() if trace.x is None else trace.x), dtype=float
            )
            y_array = np.asarray(
                list(() if trace.y is None else trace.y), dtype=float
            )
            z_array = np.asarray(
                list(() if trace.z is None else trace.z), dtype=float
            )
            segments = _split_nan_separated_xyz_segments(
                x_values=x_array,
                y_values=y_array,
                z_values=z_array,
            )
            if segments:
                payload["lines"].append(
                    {
                        "name": trace_name,
                        "segments": segments,
                        "color": color,
                        "opacity": opacity,
                        "dash": str(getattr(line, "dash", "solid") or "solid"),
                        "role": (
                            "cone_tip"
                            if "граница конуса" in trace_name.lower()
                            else "line"
                        ),
                    }
                )
            customdata_rows = list(
                ()
                if getattr(trace, "customdata", None) is None
                else np.asarray(trace.customdata, dtype=object)
            )
            if customdata_rows:
                hover_points: list[list[float]] = []
                hover_items: list[dict[str, object]] = []
                for index, (x_value, y_value, z_value) in enumerate(
                    zip(x_array, y_array, z_array, strict=False)
                ):
                    if not (
                        np.isfinite(float(x_value))
                        and np.isfinite(float(y_value))
                        and np.isfinite(float(z_value))
                    ):
                        continue
                    hover_points.append(
                        [float(x_value), float(y_value), float(z_value)]
                    )
                    row = (
                        customdata_rows[index]
                        if index < len(customdata_rows)
                        else None
                    )
                    hover_items.append(
                        _customdata_row_to_hover_item(
                            row,
                            fallback_name=hover_name,
                        )
                    )
                if hover_points:
                    hover_points, hover_items = _decimate_hover_payload(
                        points=hover_points,
                        hover_items=hover_items,
                        max_points=(
                            WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
                            if is_reference_trace
                            else WT_THREE_MAX_HOVER_POINTS_PER_TRACE
                        ),
                    )
                    payload["points"].append(
                        {
                            "name": hover_name,
                            "points": hover_points,
                            "color": color,
                            "opacity": 0.001,
                            "size": 8.5,
                            "symbol": "circle",
                            "hover": hover_items,
                            "hover_only": True,
                            "role": (
                                "reference_hover"
                                if is_reference_trace
                                else "trajectory_hover"
                            ),
                        }
                    )
        if _trace_showlegend(trace) and trace_name:
            payload["legend_items"].append(
                {
                    "label": trace_name,
                    "color": color,
                    "opacity": opacity,
                }
            )
            legend_added = True
    if "markers" in mode:
        marker = trace.marker or {}
        color, opacity = _plotly_color_and_opacity(
            getattr(marker, "color", None),
            fallback_opacity=trace_opacity,
        )
        if not (trace_name or opacity > 0.01):
            return payload
        if visibility_state == "visible":
            x_array = np.asarray(
                list(() if trace.x is None else trace.x), dtype=float
            )
            y_array = np.asarray(
                list(() if trace.y is None else trace.y), dtype=float
            )
            z_array = np.asarray(
                list(() if trace.z is None else trace.z), dtype=float
            )
            customdata_rows = list(
                ()
                if getattr(trace, "customdata", None) is None
                else np.asarray(trace.customdata, dtype=object)
            )
            points: list[list[float]] = []
            hover_items: list[dict[str, object]] = []
            for index, (x_value, y_value, z_value) in enumerate(
                zip(x_array, y_array, z_array, strict=False)
            ):
                if not (
                    np.isfinite(float(x_value))
                    and np.isfinite(float(y_value))
                    and np.isfinite(float(z_value))
                ):
                    continue
                points.append([float(x_value), float(y_value), float(z_value)])
                row = (
                    customdata_rows[index]
                    if index < len(customdata_rows)
                    else None
                )
                hover_items.append(
                    _customdata_row_to_hover_item(
                        row,
                        fallback_name=hover_name,
                    )
                )
            if points:
                marker_size = getattr(marker, "size", 6)
                if isinstance(marker_size, (list, tuple, np.ndarray)):
                    marker_size = marker_size[0] if len(marker_size) else 6
                hover_only = opacity <= 0.01
                payload["points"].append(
                    {
                        "name": hover_name,
                        "points": points,
                        "color": color,
                        "opacity": opacity,
                        "size": (
                            max(float(marker_size), 8.5)
                            if hover_only
                            else float(marker_size)
                        ),
                        "symbol": str(
                            getattr(marker, "symbol", "circle") or "circle"
                        ),
                        "hover": hover_items,
                        "hover_only": hover_only,
                        "role": (
                            "reference_marker"
                            if is_reference_trace
                            else "marker"
                        ),
                    }
                )
        if _trace_showlegend(trace) and trace_name and not legend_added:
            payload["legend_items"].append(
                {
                    "label": trace_name,
                    "color": color,
                    "opacity": opacity,
                }
            )
    if "text" in mode and "lines" not in mode and "markers" not in mode:
        text_font = trace.textfont or {}
        color = str(getattr(text_font, "color", "#0F172A"))
        if visibility_state == "visible":
            x_array = np.asarray(
                list(() if trace.x is None else trace.x), dtype=float
            )
            y_array = np.asarray(
                list(() if trace.y is None else trace.y), dtype=float
            )
            z_array = np.asarray(
                list(() if trace.z is None else trace.z), dtype=float
            )
            text_values = list(() if trace.text is None else trace.text)
            for x_value, y_value, z_value, text_value in zip(
                x_array,
                y_array,
                z_array,
                text_values,
                strict=False,
            ):
                if not (
                    np.isfinite(float(x_value))
                    and np.isfinite(float(y_value))
                    and np.isfinite(float(z_value))
                ):
                    continue
                payload["labels"].append(
                    {
                        "text": str(text_value),
                        "position": [
                            float(x_value),
                            float(y_value),
                            float(z_value),
                        ],
                        "color": color,
                        "role": (
                            "reference_pad_label"
                            if "кусты" in trace_name.lower()
                            else (
                                "reference_label"
                                if "подписи" in trace_name.lower()
                                else (
                                    "well_label"
                                    if trace_name.endswith(": t1 label")
                                    else "label"
                                )
                            )
                        ),
                    }
                )
    return payload


def _mesh3d_trace_to_three_payload(
    trace: go.Mesh3d,
) -> dict[str, object] | None:
    if _trace_visibility_state(trace) != "visible":
        return None
    x_array = np.asarray(list(() if trace.x is None else trace.x), dtype=float)
    y_array = np.asarray(list(() if trace.y is None else trace.y), dtype=float)
    z_array = np.asarray(list(() if trace.z is None else trace.z), dtype=float)
    i_array = np.asarray(list(() if trace.i is None else trace.i), dtype=int)
    j_array = np.asarray(list(() if trace.j is None else trace.j), dtype=int)
    k_array = np.asarray(list(() if trace.k is None else trace.k), dtype=int)
    if (
        len(x_array) == 0
        or len(y_array) != len(x_array)
        or len(z_array) != len(x_array)
        or len(i_array) == 0
        or len(i_array) != len(j_array)
        or len(i_array) != len(k_array)
    ):
        return None
    color, opacity = _plotly_color_and_opacity(
        getattr(trace, "color", None),
        fallback_opacity=float(
            1.0 if getattr(trace, "opacity", None) is None else trace.opacity
        ),
    )
    payload = {
        "name": str(trace.name or "").strip(),
        "vertices": [
            [float(x_value), float(y_value), float(z_value)]
            for x_value, y_value, z_value in zip(
                x_array, y_array, z_array, strict=False
            )
        ],
        "faces": [
            [int(i_value), int(j_value), int(k_value)]
            for i_value, j_value, k_value in zip(
                i_array, j_array, k_array, strict=False
            )
        ],
        "color": color,
        "opacity": opacity,
        "role": (
            "cone"
            if "cone" in str(trace.name or "").lower()
            else (
                "overlap"
                if "overlap" in str(trace.name or "").lower()
                else "mesh"
            )
        ),
    }
    return payload


def _optimize_three_payload(payload: dict[str, object]) -> dict[str, object]:
    optimized = dict(payload)
    optimized["lines"] = _merge_three_line_payloads(payload.get("lines") or [])
    optimized["points"] = _merge_three_point_payloads(
        payload.get("points") or []
    )
    optimized["meshes"] = _merge_three_mesh_payloads(
        payload.get("meshes") or []
    )
    labels = list(payload.get("labels") or [])
    reference_labels = [
        item
        for item in labels
        if str(item.get("role") or "") == "reference_label"
    ]
    other_labels = [
        item
        for item in labels
        if str(item.get("role") or "") != "reference_label"
    ]
    if len(reference_labels) > WT_THREE_MAX_REFERENCE_LABELS:
        indices = np.unique(
            np.linspace(
                0,
                len(reference_labels) - 1,
                num=WT_THREE_MAX_REFERENCE_LABELS,
                dtype=int,
            )
        ).tolist()
        reference_labels = [reference_labels[index] for index in indices]
    labels = other_labels + reference_labels
    if len(labels) > WT_THREE_MAX_LABELS:
        labels = labels[:WT_THREE_MAX_LABELS]
    optimized["labels"] = labels
    optimized["legend"] = list(payload.get("legend") or [])
    return optimized


def _merge_three_line_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str, str], list[list[list[float]]]] = {}
    ordered_keys: list[tuple[str, float, str, str]] = []
    for item in items:
        color = str(item.get("color") or "#0F172A")
        opacity = float(item.get("opacity") or 1.0)
        dash = str(item.get("dash") or "solid")
        role = str(item.get("role") or "line")
        key = (color, opacity, dash, role)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].extend(
            [
                segment
                for segment in (item.get("segments") or [])
                if isinstance(segment, list) and len(segment) >= 2
            ]
        )
    merged: list[dict[str, object]] = []
    for color, opacity, dash, role in ordered_keys:
        segments = grouped[(color, opacity, dash, role)]
        if not segments:
            continue
        merged.append(
            {
                "segments": segments,
                "color": color,
                "opacity": float(opacity),
                "dash": dash,
                "role": role,
            }
        )
    return merged


def _merge_three_point_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, float, float, str, bool, str], dict[str, object]
    ] = {}
    ordered_keys: list[tuple[str, float, float, str, bool, str]] = []
    for item in items:
        color = str(item.get("color") or "#0F172A")
        opacity = float(item.get("opacity") or 1.0)
        size = float(item.get("size") or 6.0)
        symbol = str(item.get("symbol") or "circle")
        hover_only = bool(item.get("hover_only"))
        role = str(item.get("role") or "point")
        key = (color, opacity, size, symbol, hover_only, role)
        if key not in grouped:
            grouped[key] = {
                "points": [],
                "hover": [],
                "color": color,
                "opacity": float(opacity),
                "size": float(size),
                "symbol": symbol,
                "hover_only": hover_only,
                "role": role,
            }
            ordered_keys.append(key)
        valid_points = [
            point
            for point in (item.get("points") or [])
            if isinstance(point, list) and len(point) == 3
        ]
        grouped[key]["points"].extend(valid_points)
        raw_hover = list(item.get("hover") or [])
        grouped[key]["hover"].extend(raw_hover[: len(valid_points)])
    merged: list[dict[str, object]] = []
    for key in ordered_keys:
        entry = grouped[key]
        points = list(entry["points"])
        if not points:
            continue
        merged.append(
            {
                "points": points,
                "hover": list(entry["hover"]),
                "color": str(entry["color"]),
                "opacity": float(entry["opacity"]),
                "size": float(entry["size"]),
                "symbol": str(entry["symbol"]),
                "hover_only": bool(entry["hover_only"]),
                "role": str(entry["role"]),
            }
        )
    return merged


def _merge_three_mesh_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], dict[str, object]] = {}
    ordered_keys: list[tuple[str, float, str]] = []
    for item in items:
        color = str(item.get("color") or "#94A3B8")
        opacity = float(item.get("opacity") or 1.0)
        role = str(item.get("role") or "mesh")
        key = (color, opacity, role)
        if key not in grouped:
            grouped[key] = {
                "vertices": [],
                "faces": [],
                "color": color,
                "opacity": float(opacity),
                "role": role,
            }
            ordered_keys.append(key)
        merged = grouped[key]
        vertices = [
            vertex
            for vertex in (item.get("vertices") or [])
            if isinstance(vertex, list) and len(vertex) == 3
        ]
        faces = [
            face
            for face in (item.get("faces") or [])
            if isinstance(face, list) and len(face) == 3
        ]
        if not vertices or not faces:
            continue
        vertex_offset = len(merged["vertices"])
        merged["vertices"].extend(vertices)
        merged["faces"].extend(
            [
                [
                    int(face[0]) + vertex_offset,
                    int(face[1]) + vertex_offset,
                    int(face[2]) + vertex_offset,
                ]
                for face in faces
            ]
        )
    return [
        grouped[key]
        for key in ordered_keys
        if grouped[key]["vertices"] and grouped[key]["faces"]
    ]


def _raw_bounds_from_xyz_arrays(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> dict[str, list[float]] | None:
    if not (len(x_values) == len(y_values) == len(z_values)):
        return None
    finite_mask = (
        np.isfinite(x_values.astype(float, copy=False))
        & np.isfinite(y_values.astype(float, copy=False))
        & np.isfinite(z_values.astype(float, copy=False))
    )
    if not finite_mask.any():
        return None
    filtered_x = x_values[finite_mask].astype(float, copy=False)
    filtered_y = y_values[finite_mask].astype(float, copy=False)
    filtered_z = z_values[finite_mask].astype(float, copy=False)
    return {
        "min": [
            float(np.min(filtered_x)),
            float(np.min(filtered_y)),
            float(np.min(filtered_z)),
        ],
        "max": [
            float(np.max(filtered_x)),
            float(np.max(filtered_y)),
            float(np.max(filtered_z)),
        ],
    }


def _merge_raw_bounds(
    bounds_items: Iterable[dict[str, list[float]] | None],
) -> dict[str, list[float]] | None:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for item in bounds_items:
        if not item:
            continue
        mins.append(np.asarray(item["min"], dtype=float))
        maxs.append(np.asarray(item["max"], dtype=float))
    if not mins or not maxs:
        return None
    min_stack = np.vstack(mins)
    max_stack = np.vstack(maxs)
    return {
        "min": np.min(min_stack, axis=0).astype(float).tolist(),
        "max": np.max(max_stack, axis=0).astype(float).tolist(),
    }


def _successful_plan_raw_bounds(
    success: SuccessfulWellPlan,
) -> dict[str, list[float]] | None:
    stations = success.stations
    if stations.empty:
        return _raw_bounds_from_xyz_arrays(
            x_values=np.asarray(
                [success.surface.x, success.t1.x, success.t3.x], dtype=float
            ),
            y_values=np.asarray(
                [success.surface.y, success.t1.y, success.t3.y], dtype=float
            ),
            z_values=np.asarray(
                [success.surface.z, success.t1.z, success.t3.z], dtype=float
            ),
        )
    bounds = _raw_bounds_from_xyz_arrays(
        x_values=stations["X_m"].to_numpy(dtype=float),
        y_values=stations["Y_m"].to_numpy(dtype=float),
        z_values=stations["Z_m"].to_numpy(dtype=float),
    )
    return _merge_raw_bounds(
        (
            bounds,
            {
                "min": [
                    float(min(success.surface.x, success.t1.x, success.t3.x)),
                    float(min(success.surface.y, success.t1.y, success.t3.y)),
                    float(min(success.surface.z, success.t1.z, success.t3.z)),
                ],
                "max": [
                    float(max(success.surface.x, success.t1.x, success.t3.x)),
                    float(max(success.surface.y, success.t1.y, success.t3.y)),
                    float(max(success.surface.z, success.t1.z, success.t3.z)),
                ],
            },
        )
    )


def _target_only_raw_bounds(
    target_only: _TargetOnlyWell,
) -> dict[str, list[float]]:
    return {
        "min": [
            float(
                min(target_only.surface.x, target_only.t1.x, target_only.t3.x)
            ),
            float(
                min(target_only.surface.y, target_only.t1.y, target_only.t3.y)
            ),
            float(
                min(target_only.surface.z, target_only.t1.z, target_only.t3.z)
            ),
        ],
        "max": [
            float(
                max(target_only.surface.x, target_only.t1.x, target_only.t3.x)
            ),
            float(
                max(target_only.surface.y, target_only.t1.y, target_only.t3.y)
            ),
            float(
                max(target_only.surface.z, target_only.t1.z, target_only.t3.z)
            ),
        ],
    }


def _legend_pad_label(pad: WellPad) -> str:
    return f"Куст {str(pad.pad_id)}"


def _three_legend_tree_payload(
    *,
    records: list[WelltrackRecord],
    visible_well_names: Iterable[str],
    well_bounds_by_name: Mapping[str, dict[str, list[float]]],
    name_to_color: Mapping[str, str],
) -> tuple[
    list[dict[str, object]], dict[str, dict[str, list[float]]], set[str]
]:
    visible_set = {
        str(name) for name in visible_well_names if str(name).strip()
    }
    if not visible_set:
        return [], {}, set()
    pads, _, well_names_by_pad_id = _pad_membership(records)
    tree: list[dict[str, object]] = []
    focus_targets: dict[str, dict[str, list[float]]] = {}
    hidden_flat_legend_labels: set[str] = set()
    for pad in pads:
        pad_id = str(pad.pad_id)
        ordered_names = [
            str(name)
            for name in well_names_by_pad_id.get(pad_id, ())
            if str(name) in visible_set and str(name) in well_bounds_by_name
        ]
        if not ordered_names:
            continue
        child_nodes = []
        child_bounds = []
        for well_name in ordered_names:
            focus_id = f"well::{well_name}"
            child_nodes.append(
                {
                    "id": focus_id,
                    "label": str(well_name),
                    "color": str(name_to_color.get(str(well_name), "#64748b")),
                }
            )
            focus_targets[focus_id] = dict(well_bounds_by_name[well_name])
            child_bounds.append(well_bounds_by_name[well_name])
            hidden_flat_legend_labels.add(str(well_name))
        pad_focus_id = f"pad::{pad_id}"
        merged_pad_bounds = _merge_raw_bounds(child_bounds)
        if merged_pad_bounds is not None:
            focus_targets[pad_focus_id] = merged_pad_bounds
        tree.append(
            {
                "id": pad_focus_id,
                "label": _legend_pad_label(pad),
                "children": child_nodes,
            }
        )
    return tree, focus_targets, hidden_flat_legend_labels


def _augment_three_payload(
    *,
    payload: dict[str, object],
    legend_tree: list[dict[str, object]] | None = None,
    focus_targets: Mapping[str, dict[str, list[float]]] | None = None,
    hidden_flat_legend_labels: set[str] | None = None,
    collisions: list[dict[str, object]] | None = None,
    edit_wells: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    updated = dict(payload)
    if legend_tree:
        updated["legend_tree"] = list(legend_tree)
    if focus_targets:
        updated["focus_targets"] = {
            str(key): dict(value) for key, value in focus_targets.items()
        }
    hidden_labels = {
        str(label) for label in (hidden_flat_legend_labels or set())
    }
    if hidden_labels:
        updated["legend"] = [
            item
            for item in list(updated.get("legend") or [])
            if str(dict(item).get("label", "")).strip() not in hidden_labels
        ]
    if collisions is not None:
        updated["collisions"] = list(collisions)
    if edit_wells is not None:
        updated["edit_wells"] = list(edit_wells)
    return updated


def _plotly_3d_figure_to_three_payload(fig: go.Figure) -> dict[str, object]:
    scene = fig.layout.scene
    x_range = (
        [float(scene.xaxis.range[0]), float(scene.xaxis.range[1])]
        if scene.xaxis.range is not None
        else [0.0, 1000.0]
    )
    y_range = (
        [float(scene.yaxis.range[0]), float(scene.yaxis.range[1])]
        if scene.yaxis.range is not None
        else [0.0, 1000.0]
    )
    z_range = (
        [float(scene.zaxis.range[0]), float(scene.zaxis.range[1])]
        if scene.zaxis.range is not None
        else [0.0, 1000.0]
    )
    payload: dict[str, object] = {
        "background": "#FFFFFF",
        "title": str(
            getattr(getattr(fig.layout, "title", None), "text", "") or ""
        ),
        "bounds": {
            "min": [x_range[0], y_range[0], z_range[0]],
            "max": [x_range[1], y_range[1], z_range[1]],
        },
        "camera": (
            scene.camera.to_plotly_json()
            if getattr(scene, "camera", None) is not None
            else DEFAULT_3D_CAMERA
        ),
        "lines": [],
        "meshes": [],
        "points": [],
        "labels": [],
        "legend": [],
    }
    seen_legend_labels: set[str] = set()
    for trace in fig.data:
        if isinstance(trace, go.Scatter3d):
            trace_payload = _scatter3d_trace_to_three_payload(trace)
            payload["lines"].extend(trace_payload["lines"])
            payload["points"].extend(trace_payload["points"])
            payload["labels"].extend(trace_payload["labels"])
            for item in trace_payload["legend_items"]:
                label = str(item["label"])
                if not label or label in seen_legend_labels:
                    continue
                payload["legend"].append(item)
                seen_legend_labels.add(label)
            continue
        if isinstance(trace, go.Mesh3d):
            mesh_payload = _mesh3d_trace_to_three_payload(trace)
            if mesh_payload is None:
                continue
            payload["meshes"].append(mesh_payload)
            if _trace_showlegend(trace):
                label = str(mesh_payload["name"])
                if label and label not in seen_legend_labels:
                    payload["legend"].append(
                        {
                            "label": label,
                            "color": str(mesh_payload["color"]),
                            "opacity": float(mesh_payload["opacity"]),
                        }
                    )
                    seen_legend_labels.add(label)
    return _optimize_three_payload(payload)


def _render_plotly_or_three_3d(
    *,
    container: object,
    figure: go.Figure,
    backend: str,
    height: int,
    payload_overrides: dict[str, object] | None = None,
) -> None:
    if str(backend) == WT_3D_BACKEND_THREE_LOCAL:
        payload = _plotly_3d_figure_to_three_payload(figure)
        if payload_overrides:
            payload = _augment_three_payload(
                payload=payload,
                legend_tree=list(payload_overrides.get("legend_tree") or []),
                focus_targets=payload_overrides.get("focus_targets"),
                hidden_flat_legend_labels=set(
                    str(item)
                    for item in (
                        payload_overrides.get("hidden_flat_legend_labels")
                        or set()
                    )
                ),
                collisions=payload_overrides.get("collisions"),
                edit_wells=payload_overrides.get("edit_wells"),
            )
        with container:
            render_local_three_scene(
                payload,
                height=height,
                instance_token=int(
                    st.session_state.get("wt_three_viewer_nonce", 0)
                ),
            )
        return
    container.plotly_chart(
        figure,
        config=trajectory_plotly_chart_config(),
        width="stretch",
    )


def _build_edit_wells_payload(
    successes: list[SuccessfulWellPlan],
    name_to_color: Mapping[str, str],
) -> list[dict[str, object]]:
    edit_wells: list[dict[str, object]] = []
    for success in successes:
        config = success.config
        edit_wells.append(
            {
                "name": str(success.name),
                "surface": [
                    float(success.surface.x),
                    float(success.surface.y),
                    float(success.surface.z),
                ],
                "t1": [
                    float(success.t1.x),
                    float(success.t1.y),
                    float(success.t1.z),
                ],
                "t3": [
                    float(success.t3.x),
                    float(success.t3.y),
                    float(success.t3.z),
                ],
                "color": str(
                    name_to_color.get(str(success.name), "#2563eb")
                ),
                "config": {
                    "entry_inc_target_deg": float(config.entry_inc_target_deg),
                    "max_inc_deg": float(config.max_inc_deg),
                    "dls_build_max_deg_per_30m": float(
                        config.dls_build_max_deg_per_30m
                    ),
                    "kop_min_vertical_m": float(config.kop_min_vertical_m),
                },
            }
        )
    return edit_wells


def _trajectory_three_payload_overrides(
    *,
    records: list[WelltrackRecord],
    successes: list[SuccessfulWellPlan],
    target_only_wells: list[_TargetOnlyWell],
    name_to_color: Mapping[str, str],
) -> dict[str, object]:
    well_bounds_by_name: dict[str, dict[str, list[float]]] = {}
    for success in successes:
        bounds = _successful_plan_raw_bounds(success)
        if bounds is not None:
            well_bounds_by_name[str(success.name)] = bounds
    for target_only in target_only_wells:
        well_bounds_by_name[str(target_only.name)] = _target_only_raw_bounds(
            target_only
        )
    legend_tree, focus_targets, hidden_labels = _three_legend_tree_payload(
        records=records,
        visible_well_names=tuple(well_bounds_by_name.keys()),
        well_bounds_by_name=well_bounds_by_name,
        name_to_color=name_to_color,
    )
    return {
        "legend_tree": legend_tree,
        "focus_targets": focus_targets,
        "hidden_flat_legend_labels": hidden_labels,
        "edit_wells": _build_edit_wells_payload(successes, name_to_color),
    }


def _anticollision_three_payload_overrides(
    *,
    records: list[WelltrackRecord],
    analysis: AntiCollisionAnalysis,
) -> dict[str, object]:
    visible_names: list[str] = []
    well_bounds_by_name: dict[str, dict[str, list[float]]] = {}
    name_to_color: dict[str, str] = {}
    for well in analysis.wells:
        if bool(well.is_reference_only):
            continue
        visible_names.append(str(well.name))
        name_to_color[str(well.name)] = str(well.color)
        bounds = _raw_bounds_from_xyz_arrays(
            x_values=well.stations["X_m"].to_numpy(dtype=float),
            y_values=well.stations["Y_m"].to_numpy(dtype=float),
            z_values=well.stations["Z_m"].to_numpy(dtype=float),
        )
        extra_bounds = None
        if well.t1 is not None and well.t3 is not None:
            extra_bounds = {
                "min": [
                    float(min(well.surface.x, well.t1.x, well.t3.x)),
                    float(min(well.surface.y, well.t1.y, well.t3.y)),
                    float(min(well.surface.z, well.t1.z, well.t3.z)),
                ],
                "max": [
                    float(max(well.surface.x, well.t1.x, well.t3.x)),
                    float(max(well.surface.y, well.t1.y, well.t3.y)),
                    float(max(well.surface.z, well.t1.z, well.t3.z)),
                ],
            }
        merged_bounds = _merge_raw_bounds((bounds, extra_bounds))
        if merged_bounds is not None:
            well_bounds_by_name[str(well.name)] = merged_bounds
    legend_tree, focus_targets, hidden_labels = _three_legend_tree_payload(
        records=records,
        visible_well_names=visible_names,
        well_bounds_by_name=well_bounds_by_name,
        name_to_color=name_to_color,
    )
    from pywp.anticollision import (
        _segment_types_for_interval,
        anti_collision_report_events,
    )

    events = anti_collision_report_events(analysis)
    # Build a lookup: (well_a, well_b) -> list of zones for hotspot resolution
    zone_lookup: dict[tuple[str, str], list[AntiCollisionZone]] = {}
    for zone in analysis.zones:
        key = (str(zone.well_a), str(zone.well_b))
        zone_lookup.setdefault(key, []).append(zone)

    collisions = []
    for idx, event in enumerate(events):
        segment_a = _segment_types_for_interval(
            analysis,
            str(event.well_a),
            float(event.md_a_start_m),
            float(event.md_a_end_m),
        )
        segment_b = _segment_types_for_interval(
            analysis,
            str(event.well_b),
            float(event.md_b_start_m),
            float(event.md_b_end_m),
        )
        # Pick hotspot from the zone with worst SF within this event's MD range
        candidate_zones = zone_lookup.get(
            (str(event.well_a), str(event.well_b)), []
        )
        best_zone = None
        for zone in candidate_zones:
            if (
                float(zone.md_a_m) >= float(event.md_a_start_m) - 1.0
                and float(zone.md_a_m) <= float(event.md_a_end_m) + 1.0
                and float(zone.md_b_m) >= float(event.md_b_start_m) - 1.0
                and float(zone.md_b_m) <= float(event.md_b_end_m) + 1.0
            ):
                if best_zone is None or float(zone.separation_factor) < float(
                    best_zone.separation_factor
                ):
                    best_zone = zone
        if best_zone is None and candidate_zones:
            best_zone = min(
                candidate_zones, key=lambda z: float(z.separation_factor)
            )
        hotspot = (
            list(best_zone.hotspot_xyz) if best_zone is not None else [0.0, 0.0, 0.0]
        )
        collisions.append(
            {
                "id": f"collision::{event.well_a}::{event.well_b}::{idx}",
                "well_a": str(event.well_a),
                "well_b": str(event.well_b),
                "label": f"{event.well_a} ↔ {event.well_b}",
                "classification": str(event.classification),
                "priority_rank": int(event.priority_rank),
                "hotspot": hotspot,
                "separation_factor": float(event.min_separation_factor),
                "center_distance_m": float(event.min_center_distance_m),
                "segment_a": segment_a,
                "segment_b": segment_b,
            }
        )
    return {
        "legend_tree": legend_tree,
        "focus_targets": focus_targets,
        "hidden_flat_legend_labels": hidden_labels,
        "collisions": collisions,
    }


def _build_anti_collision_analysis(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
) -> AntiCollisionAnalysis:
    color_map = {
        str(item.name): (name_to_color or {}).get(
            str(item.name), _well_color(index)
        )
        for index, item in enumerate(successes)
    }
    return build_anti_collision_analysis_for_successes_shared(
        successes,
        model=model,
        name_to_color=color_map,
        reference_wells=reference_wells,
    )


def _anti_collision_cache_key(
    *,
    successes: list[SuccessfulWellPlan],
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None,
    reference_wells: tuple[ImportedTrajectoryWell, ...],
) -> str:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(
        np.asarray(
            [
                float(model.sigma_inc_deg),
                float(model.sigma_azi_deg),
                float(model.sigma_lateral_drift_m_per_1000m),
                float(model.confidence_scale),
                float(model.sample_step_m),
                float(model.min_refined_step_m),
                float(model.directional_refine_threshold_deg),
            ],
            dtype=np.float64,
        ).tobytes()
    )
    for well_name, color in sorted((name_to_color or {}).items()):
        digest.update(str(well_name).encode("utf-8"))
        digest.update(str(color).encode("utf-8"))
    ordered_successes = sorted(successes, key=lambda item: str(item.name))
    for success in ordered_successes:
        digest.update(str(success.name).encode("utf-8"))
        digest.update(
            np.asarray(
                [
                    float(success.surface.x),
                    float(success.surface.y),
                    float(success.surface.z),
                    float(success.t1.x),
                    float(success.t1.y),
                    float(success.t1.z),
                    float(success.t3.x),
                    float(success.t3.y),
                    float(success.t3.z),
                    float(success.azimuth_deg),
                    float(success.md_t1_m),
                ],
                dtype=np.float64,
            ).tobytes()
        )
        stations_subset = success.stations.loc[
            :,
            [
                column
                for column in (
                    "MD_m",
                    "INC_deg",
                    "AZI_deg",
                    "X_m",
                    "Y_m",
                    "Z_m",
                )
                if column in success.stations.columns
            ],
        ]
        digest.update(str(tuple(stations_subset.columns)).encode("utf-8"))
        digest.update(
            stations_subset.to_numpy(dtype=np.float64, copy=True).tobytes()
        )
    for reference_well in sorted(
        reference_wells,
        key=lambda item: (str(item.name), str(item.kind)),
    ):
        digest.update(str(reference_well.name).encode("utf-8"))
        digest.update(str(reference_well.kind).encode("utf-8"))
        digest.update(
            str(
                REFERENCE_WELL_KIND_COLORS.get(
                    str(reference_well.kind),
                    "#A0A0A0",
                )
            ).encode("utf-8")
        )
        stations_subset = reference_well.stations.loc[
            :,
            [
                column
                for column in (
                    "MD_m",
                    "INC_deg",
                    "AZI_deg",
                    "X_m",
                    "Y_m",
                    "Z_m",
                )
                if column in reference_well.stations.columns
            ],
        ]
        digest.update(str(tuple(stations_subset.columns)).encode("utf-8"))
        digest.update(
            stations_subset.to_numpy(dtype=np.float64, copy=True).tobytes()
        )
    return digest.hexdigest()


def _store_anticollision_failure_state(
    exc: Exception,
    *,
    started_at: float | None = None,
    log_lines: Iterable[str] = (),
) -> None:
    previous_state = st.session_state.get("wt_anticollision_last_run")
    previous_payload = (
        previous_state if isinstance(previous_state, Mapping) else {}
    )
    merged_log_lines = [
        str(item) for item in (previous_payload.get("log_lines") or ())
    ]
    for item in log_lines:
        text = str(item)
        if text:
            merged_log_lines.append(text)
    error_line = (
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"Ошибка anti-collision: {type(exc).__name__}: {exc}"
    )
    if not merged_log_lines or merged_log_lines[-1] != error_line:
        merged_log_lines.append(error_line)
    st.session_state["wt_anticollision_last_run"] = {
        "cached": False,
        "runtime_s": (
            float(perf_counter() - float(started_at))
            if started_at is not None
            else previous_payload.get("runtime_s")
        ),
        "log_lines": tuple(merged_log_lines),
        "pair_count": int(previous_payload.get("pair_count") or 0),
        "overlap_count": int(previous_payload.get("overlap_count") or 0),
        "recommendation_count": int(
            previous_payload.get("recommendation_count") or 0
        ),
        "cluster_count": int(previous_payload.get("cluster_count") or 0),
        "status": f"Ошибка: {type(exc).__name__}",
    }


def _cached_anti_collision_view_model(
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    records: list[WelltrackRecord],
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[
    AntiCollisionAnalysis,
    tuple[AntiCollisionRecommendation, ...],
    tuple[AntiCollisionRecommendationCluster, ...],
]:
    started_at = perf_counter()
    log_lines: list[str] = []

    def _emit(progress_value: int, message: str) -> None:
        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if progress_callback is not None:
            progress_callback(int(progress_value), message)

    color_map = _well_color_map(records) if records else {}
    _emit(8, "Подготовка входных данных anti-collision.")
    cache_key = _anti_collision_cache_key(
        successes=successes,
        model=uncertainty_model,
        name_to_color=color_map,
        reference_wells=reference_wells,
    )
    cache = dict(st.session_state.get("wt_anticollision_analysis_cache") or {})
    if str(cache.get("key", "")) == cache_key:
        analysis = cache.get("analysis")
        recommendations = cache.get("recommendations")
        clusters = cache.get("clusters")
        if (
            isinstance(analysis, AntiCollisionAnalysis)
            and isinstance(recommendations, tuple)
            and isinstance(clusters, tuple)
        ):
            _emit(100, "Использован кэш anti-collision анализа.")
            st.session_state["wt_anticollision_last_run"] = {
                "cached": True,
                "runtime_s": float(perf_counter() - started_at),
                "log_lines": tuple(log_lines),
                "pair_count": int(analysis.pair_count),
                "overlap_count": int(analysis.overlapping_pair_count),
                "recommendation_count": int(len(recommendations)),
                "cluster_count": int(len(clusters)),
            }
            return analysis, recommendations, clusters
    try:
        _emit(30, "Расчёт anti-collision модели.")
        analysis = _build_anti_collision_analysis(
            successes,
            model=uncertainty_model,
            name_to_color=color_map,
            reference_wells=reference_wells,
        )
        _emit(72, "Построение рекомендаций anti-collision.")
        recommendations = build_anti_collision_recommendations(
            analysis,
            well_context_by_name=_build_anticollision_well_contexts(successes),
        )
        _emit(88, "Кластеризация рекомендаций anti-collision.")
        clusters = build_anti_collision_recommendation_clusters(
            recommendations
        )
    except Exception as exc:
        _store_anticollision_failure_state(
            exc,
            started_at=started_at,
            log_lines=tuple(log_lines),
        )
        raise
    st.session_state["wt_anticollision_analysis_cache"] = {
        "key": cache_key,
        "analysis": analysis,
        "recommendations": recommendations,
        "clusters": clusters,
    }
    _emit(100, "Расчёт Anti-collision завершён.")
    st.session_state["wt_anticollision_last_run"] = {
        "cached": False,
        "runtime_s": float(perf_counter() - started_at),
        "log_lines": tuple(log_lines),
        "pair_count": int(analysis.pair_count),
        "overlap_count": int(analysis.overlapping_pair_count),
        "recommendation_count": int(len(recommendations)),
        "cluster_count": int(len(clusters)),
    }
    return analysis, recommendations, clusters


def _render_status_run_log(
    *,
    title: str,
    state_payload: Mapping[str, object] | None,
    empty_message: str,
) -> None:
    with st.expander(title, expanded=False):
        if not isinstance(state_payload, Mapping):
            st.caption(empty_message)
            return
        cached = bool(state_payload.get("cached"))
        runtime_s = state_payload.get("runtime_s")
        top_cols = st.columns(4, gap="small")
        top_cols[0].metric("Статус", "Кэш" if cached else "Выполнен")
        top_cols[1].metric(
            "Время, с",
            "—" if runtime_s is None else f"{float(runtime_s):.2f}",
        )
        if "pair_count" in state_payload:
            top_cols[2].metric(
                "Пар", f"{int(state_payload.get('pair_count') or 0)}"
            )
        elif "well_count" in state_payload:
            top_cols[2].metric(
                "Скважин", f"{int(state_payload.get('well_count') or 0)}"
            )
        if "recommendation_count" in state_payload:
            top_cols[3].metric(
                "Рекомендаций",
                f"{int(state_payload.get('recommendation_count') or 0)}",
            )
        elif "eligible_well_count" in state_payload:
            top_cols[3].metric(
                "В анализе",
                f"{int(state_payload.get('eligible_well_count') or 0)}",
            )
        if (
            "cluster_count" in state_payload
            or "overlap_count" in state_payload
        ):
            extra_cols = st.columns(2, gap="small")
            extra_cols[0].metric(
                "Overlap",
                f"{int(state_payload.get('overlap_count') or 0)}",
            )
            extra_cols[1].metric(
                "Кластеров",
                f"{int(state_payload.get('cluster_count') or 0)}",
            )
        if "status" in state_payload:
            st.caption(
                f"Статус результата: {str(state_payload.get('status'))}"
            )
        log_lines = tuple(state_payload.get("log_lines") or ())
        if log_lines:
            st.code(
                "\n".join(str(item) for item in log_lines), language="text"
            )


def _trajectory_interval_points(
    stations: pd.DataFrame,
    *,
    md_start_m: float,
    md_end_m: float,
) -> pd.DataFrame | None:
    if len(stations) == 0:
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if len(md_values) == 0:
        return None
    start_md = float(
        np.clip(md_start_m, float(md_values[0]), float(md_values[-1]))
    )
    end_md = float(
        np.clip(md_end_m, float(md_values[0]), float(md_values[-1]))
    )
    if end_md <= start_md + SMALL:
        return None

    interior_mask = (md_values > start_md + SMALL) & (
        md_values < end_md - SMALL
    )
    segment_md = np.concatenate(
        [
            np.array([start_md], dtype=float),
            md_values[interior_mask],
            np.array([end_md], dtype=float),
        ]
    )
    if len(segment_md) < 2:
        return None
    return pd.DataFrame(
        {
            "MD_m": segment_md,
            "X_m": np.interp(
                segment_md, md_values, stations["X_m"].to_numpy(dtype=float)
            ),
            "Y_m": np.interp(
                segment_md, md_values, stations["Y_m"].to_numpy(dtype=float)
            ),
            "Z_m": np.interp(
                segment_md, md_values, stations["Z_m"].to_numpy(dtype=float)
            ),
        }
    )


def _trajectory_hover_customdata(stations: pd.DataFrame) -> np.ndarray:
    dls_values = stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float)
    inc_values = stations["INC_deg"].fillna(0.0).to_numpy(dtype=float)
    segment_values = (
        stations["segment"].astype(str).to_numpy(dtype=object)
        if "segment" in stations.columns
        else np.full(len(stations), "—", dtype=object)
    )
    customdata = np.empty((len(stations), 4), dtype=object)
    customdata[:, 0] = stations["MD_m"].to_numpy(dtype=float)
    customdata[:, 1] = dls_values
    customdata[:, 2] = inc_values
    customdata[:, 3] = segment_values
    return customdata


def _hover_proxy_trace_3d(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    customdata: np.ndarray,
    hovertemplate: str,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="markers",
        showlegend=False,
        marker={
            "size": 7.5,
            "color": "rgba(0, 0, 0, 0.001)",
        },
        customdata=customdata,
        hovertemplate=hovertemplate,
        hoverlabel={"namelength": -1},
    )


@st.cache_data(show_spinner=False)
def _parse_welltrack_cached(text: str) -> list[WelltrackRecord]:
    return parse_welltrack_text(text)


def _empty_source_table_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Wellname": "",
                "Point": "",
                "X": np.nan,
                "Y": np.nan,
                "Z": np.nan,
            },
            {
                "Wellname": "",
                "Point": "",
                "X": np.nan,
                "Y": np.nan,
                "Z": np.nan,
            },
            {
                "Wellname": "",
                "Point": "",
                "X": np.nan,
                "Y": np.nan,
                "Z": np.nan,
            },
        ]
    )


def _normalize_source_table_df_for_ui(
    table_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if table_df is None:
        return _empty_source_table_df()
    normalized_df = _coerce_source_table_df_columns(
        pd.DataFrame(table_df).copy()
    )
    normalized_df = _expand_single_column_source_table_df(normalized_df)
    if "Point" in normalized_df.columns:
        normalized_df["Point"] = normalized_df["Point"].map(
            lambda value: (
                "S"
                if str(value).strip().lower() in {"wellhead", "s"}
                else value
            )
        )
    for column in ("Wellname", "Point", "X", "Y", "Z"):
        if column not in normalized_df.columns:
            normalized_df[column] = (
                "" if column in {"Wellname", "Point"} else np.nan
            )
    return normalized_df.loc[:, ["Wellname", "Point", "X", "Y", "Z"]]


def _coerce_source_table_df_columns(table_df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "wellname": "Wellname",
        "well_name": "Wellname",
        "well name": "Wellname",
        "well": "Wellname",
        "name": "Wellname",
        "point": "Point",
        "pointname": "Point",
        "point_name": "Point",
        "point name": "Point",
        "точка": "Point",
        "x": "X",
        "east": "X",
        "easting": "X",
        "x_m": "X",
        "y": "Y",
        "north": "Y",
        "northing": "Y",
        "y_m": "Y",
        "z": "Z",
        "tvd": "Z",
        "z_tvd": "Z",
        "z_m": "Z",
    }
    renamed: dict[object, str] = {}
    for raw_column in list(table_df.columns):
        column_text = str(raw_column).strip()
        normalized = re.sub(r"[\s\-/(),.:]+", "_", column_text.lower()).strip(
            "_"
        )
        if column_text.lower().startswith("unnamed"):
            continue
        if normalized in alias_map:
            renamed[raw_column] = alias_map[normalized]
            continue
        if column_text in {"Wellname", "Point", "X", "Y", "Z"}:
            renamed[raw_column] = column_text
    kept_columns = [column for column in table_df.columns if column in renamed]
    if not kept_columns and len(table_df.columns) == 1:
        return table_df
    if kept_columns:
        table_df = table_df.loc[:, kept_columns].rename(columns=renamed)
    return table_df


def _expand_single_column_source_table_df(
    table_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(table_df.columns) != 1:
        return table_df
    series = table_df.iloc[:, 0]
    non_blank_values = [
        str(value).strip()
        for value in series
        if not pd.isna(value) and str(value).strip()
    ]
    if not non_blank_values:
        return table_df
    if not any("\t" in value or ";" in value for value in non_blank_values):
        return table_df
    rows: list[dict[str, object]] = []
    for raw_value in non_blank_values:
        tokens = [
            token.strip()
            for token in re.split(r"[\t;]+", raw_value)
            if token.strip()
        ]
        if len(tokens) not in {5, 6}:
            return table_df
        rows.append(
            {
                "Wellname": tokens[0],
                "Point": tokens[1],
                "X": tokens[2],
                "Y": tokens[3],
                "Z": tokens[4],
            }
        )
    return pd.DataFrame(rows, columns=["Wellname", "Point", "X", "Y", "Z"])


def _empty_reference_trajectory_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Wellname": "",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": np.nan,
                "Y": np.nan,
                "Z": np.nan,
                "MD": np.nan,
            },
            {
                "Wellname": "",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": np.nan,
                "Y": np.nan,
                "Z": np.nan,
                "MD": np.nan,
            },
            {
                "Wellname": "",
                "Type": REFERENCE_WELL_ACTUAL,
                "X": np.nan,
                "Y": np.nan,
                "Z": np.nan,
                "MD": np.nan,
            },
        ]
    )


def _reference_wells_state_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_wells"


def _reference_source_mode_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_source_mode"


def _reference_source_text_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_source_text"


def _reference_welltrack_path_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_welltrack_path"


def _set_reference_wells_for_kind(
    *,
    kind: str,
    wells: Iterable[ImportedTrajectoryWell],
) -> None:
    normalized_kind = str(kind)
    key = _reference_wells_state_key(normalized_kind)
    st.session_state[key] = tuple(wells)
    actual_wells = tuple(
        st.session_state.get(_reference_wells_state_key(REFERENCE_WELL_ACTUAL))
        or ()
    )
    approved_wells = tuple(
        st.session_state.get(
            _reference_wells_state_key(REFERENCE_WELL_APPROVED)
        )
        or ()
    )
    st.session_state["wt_reference_wells"] = tuple(actual_wells) + tuple(
        approved_wells
    )


def _reference_wells_from_state() -> tuple[ImportedTrajectoryWell, ...]:
    actual_wells = tuple(
        st.session_state.get(_reference_wells_state_key(REFERENCE_WELL_ACTUAL))
        or ()
    )
    approved_wells = tuple(
        st.session_state.get(
            _reference_wells_state_key(REFERENCE_WELL_APPROVED)
        )
        or ()
    )
    combined = tuple(actual_wells) + tuple(approved_wells)
    if combined:
        st.session_state["wt_reference_wells"] = combined
        return combined

    legacy_combined = tuple(st.session_state.get("wt_reference_wells") or ())
    if legacy_combined:
        actual_legacy = tuple(
            item
            for item in legacy_combined
            if str(getattr(item, "kind", "")) == REFERENCE_WELL_ACTUAL
        )
        approved_legacy = tuple(
            item
            for item in legacy_combined
            if str(getattr(item, "kind", "")) == REFERENCE_WELL_APPROVED
        )
        st.session_state[_reference_wells_state_key(REFERENCE_WELL_ACTUAL)] = (
            actual_legacy
        )
        st.session_state[
            _reference_wells_state_key(REFERENCE_WELL_APPROVED)
        ] = approved_legacy
    return legacy_combined


def _reset_anticollision_view_state(*, clear_prepared: bool) -> None:
    st.session_state["wt_anticollision_analysis_cache"] = {}
    st.session_state["wt_anticollision_last_run"] = None
    if not clear_prepared:
        return
    st.session_state["wt_prepared_well_overrides"] = {}
    st.session_state["wt_prepared_override_message"] = ""
    st.session_state["wt_prepared_recommendation_id"] = ""
    st.session_state["wt_anticollision_prepared_cluster_id"] = ""
    st.session_state["wt_prepared_recommendation_snapshot"] = None
    st.session_state["wt_last_anticollision_resolution"] = None
    st.session_state["wt_last_anticollision_previous_successes"] = {}


def _init_state() -> None:
    st.session_state.setdefault("wt_source_mode", "Файл по пути")
    st.session_state.setdefault("wt_source_path", str(DEFAULT_WELLTRACK_PATH))
    st.session_state.setdefault("wt_source_inline", "")
    st.session_state.setdefault("wt_source_table_df", _empty_source_table_df())
    st.session_state.setdefault("wt_source_table_editor_nonce", 0)
    st.session_state.setdefault(
        _reference_wells_state_key(REFERENCE_WELL_ACTUAL),
        (),
    )
    st.session_state.setdefault(
        _reference_wells_state_key(REFERENCE_WELL_APPROVED),
        (),
    )
    st.session_state.setdefault(
        _reference_source_mode_key(REFERENCE_WELL_ACTUAL),
        "Вставить XYZ/MD текст",
    )
    st.session_state.setdefault(
        _reference_source_mode_key(REFERENCE_WELL_APPROVED),
        "Вставить XYZ/MD текст",
    )
    st.session_state.setdefault(
        _reference_source_text_key(REFERENCE_WELL_ACTUAL), ""
    )
    st.session_state.setdefault(
        _reference_source_text_key(REFERENCE_WELL_APPROVED), ""
    )
    st.session_state.setdefault(
        _reference_welltrack_path_key(REFERENCE_WELL_ACTUAL),
        "",
    )
    st.session_state.setdefault(
        _reference_welltrack_path_key(REFERENCE_WELL_APPROVED),
        "",
    )
    _apply_profile_defaults(force=False)
    st.session_state.setdefault("wt_ui_defaults_version", 0)

    if (
        int(st.session_state.get("wt_ui_defaults_version", 0))
        < WT_UI_DEFAULTS_VERSION
    ):
        _apply_profile_defaults(force=True)
        st.session_state["wt_ui_defaults_version"] = WT_UI_DEFAULTS_VERSION

    st.session_state.setdefault("wt_records", None)
    st.session_state.setdefault("wt_records_original", None)
    st.session_state.setdefault("wt_reference_wells", ())
    st.session_state.setdefault("wt_selected_names", [])
    st.session_state.setdefault("wt_pending_selected_names", None)
    st.session_state.setdefault("wt_loaded_at", "")
    st.session_state.setdefault("wt_pad_configs", {})
    st.session_state.setdefault("wt_pad_detected_meta", {})
    st.session_state.setdefault("wt_pad_selected_id", "")
    st.session_state.setdefault("wt_pad_last_applied_at", "")
    st.session_state.setdefault("wt_pad_auto_applied_on_import", False)

    st.session_state.setdefault("wt_summary_rows", None)
    st.session_state.setdefault("wt_successes", None)
    st.session_state.setdefault("wt_last_error", "")
    st.session_state.setdefault("wt_last_run_at", "")
    st.session_state.setdefault("wt_last_runtime_s", None)
    st.session_state.setdefault("wt_last_run_log_lines", [])
    st.session_state.setdefault("wt_log_verbosity", WT_LOG_COMPACT)
    st.session_state.setdefault("wt_results_view_mode", "Все скважины")
    st.session_state.setdefault("wt_results_all_view_mode", "Траектории")
    st.session_state.setdefault("wt_results_focus_pad_id", WT_PAD_FOCUS_ALL)
    st.session_state.setdefault("wt_batch_select_pad_id", "")
    st.session_state.setdefault("wt_3d_render_mode", WT_3D_RENDER_DETAIL)
    st.session_state.setdefault("wt_3d_backend", WT_3D_BACKEND_THREE_LOCAL)
    st.session_state.setdefault("wt_three_viewer_nonce", 0)
    st.session_state.setdefault(
        "wt_anticollision_uncertainty_preset", DEFAULT_UNCERTAINTY_PRESET
    )
    st.session_state.setdefault(
        "wt_actual_fund_analysis_view_mode", "По скважинам"
    )
    st.session_state.setdefault("wt_anticollision_last_run", None)
    st.session_state.setdefault("wt_t1_t3_last_resolution", None)
    st.session_state.setdefault("wt_t1_t3_acknowledged_well_names", ())
    if str(st.session_state.get("wt_results_view_mode", "")).strip() not in {
        "Отдельная скважина",
        "Все скважины",
    }:
        st.session_state["wt_results_view_mode"] = "Все скважины"
    if str(
        st.session_state.get("wt_results_all_view_mode", "")
    ).strip() not in {
        "Траектории",
        "Anti-collision",
    }:
        st.session_state["wt_results_all_view_mode"] = "Траектории"
    if str(st.session_state.get("wt_3d_render_mode", "")).strip() not in set(
        WT_3D_RENDER_OPTIONS
    ):
        st.session_state["wt_3d_render_mode"] = WT_3D_RENDER_DETAIL
    if str(st.session_state.get("wt_3d_backend", "")).strip() not in set(
        WT_3D_BACKEND_OPTIONS
    ):
        st.session_state["wt_3d_backend"] = WT_3D_BACKEND_THREE_LOCAL
    if str(
        st.session_state.get("wt_actual_fund_analysis_view_mode", "")
    ).strip() not in {
        "По скважинам",
        "По кустам",
        "По глубинам",
    }:
        st.session_state["wt_actual_fund_analysis_view_mode"] = "По скважинам"
    st.session_state["wt_anticollision_uncertainty_preset"] = (
        normalize_uncertainty_preset(
            st.session_state.get(
                "wt_anticollision_uncertainty_preset",
                DEFAULT_UNCERTAINTY_PRESET,
            ),
        )
    )
    st.session_state.setdefault("wt_prepared_well_overrides", {})
    st.session_state.setdefault("wt_prepared_override_message", "")
    st.session_state.setdefault("wt_prepared_recommendation_id", "")
    st.session_state.setdefault("wt_anticollision_prepared_cluster_id", "")
    st.session_state.setdefault("wt_prepared_recommendation_snapshot", None)
    st.session_state.setdefault("wt_last_anticollision_resolution", None)
    st.session_state.setdefault("wt_last_anticollision_previous_successes", {})
    st.session_state.setdefault("wt_anticollision_analysis_cache", {})
    _apply_edit_targets_from_query_params()


def _apply_edit_targets_from_query_params() -> None:
    import json as _json

    params = st.query_params
    raw = params.get("edit_targets", "")
    if not raw:
        return
    # Clear the query param so it doesn't persist
    try:
        del params["edit_targets"]
    except Exception:
        pass
    try:
        changes = _json.loads(raw)
    except (ValueError, TypeError):
        return
    if not isinstance(changes, list) or not changes:
        return
    records = st.session_state.get("wt_records")
    if not records:
        return
    change_map: dict[str, dict[str, list[float]]] = {}
    for entry in changes:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        t1 = entry.get("t1")
        t3 = entry.get("t3")
        if name and isinstance(t1, list) and isinstance(t3, list):
            change_map[name] = {"t1": t1, "t3": t3}
    if not change_map:
        return
    updated_records: list[WelltrackRecord] = []
    updated_names: list[str] = []
    for record in records:
        if str(record.name) not in change_map:
            updated_records.append(record)
            continue
        delta = change_map[str(record.name)]
        new_t1 = delta["t1"]
        new_t3 = delta["t3"]
        if len(record.points) != 3 or len(new_t1) < 3 or len(new_t3) < 3:
            updated_records.append(record)
            continue
        old_points = record.points
        new_points = (
            old_points[0],
            WelltrackPoint(
                x=float(new_t1[0]),
                y=float(new_t1[1]),
                z=float(new_t1[2]),
                md=float(old_points[1].md),
            ),
            WelltrackPoint(
                x=float(new_t3[0]),
                y=float(new_t3[1]),
                z=float(new_t3[2]),
                md=float(old_points[2].md),
            ),
        )
        updated_records.append(
            WelltrackRecord(name=record.name, points=new_points)
        )
        updated_names.append(str(record.name))
    if updated_names:
        st.session_state["wt_records"] = updated_records
        st.session_state["wt_edit_targets_applied"] = updated_names
        # Clear stale calculation results so old trajectories don't conflict
        st.session_state["wt_successes"] = None
        st.session_state["wt_summary_rows"] = None


def _clear_results() -> None:
    st.session_state["wt_summary_rows"] = None
    st.session_state["wt_successes"] = None
    st.session_state["wt_pending_selected_names"] = None
    st.session_state["wt_last_error"] = ""
    st.session_state["wt_last_run_at"] = ""
    st.session_state["wt_last_runtime_s"] = None
    st.session_state["wt_last_run_log_lines"] = []
    st.session_state["wt_results_view_mode"] = "Все скважины"
    st.session_state["wt_results_all_view_mode"] = "Траектории"
    st.session_state["wt_prepared_well_overrides"] = {}
    st.session_state["wt_prepared_override_message"] = ""
    st.session_state["wt_prepared_recommendation_id"] = ""
    st.session_state["wt_anticollision_prepared_cluster_id"] = ""
    st.session_state["wt_prepared_recommendation_snapshot"] = None
    st.session_state["wt_last_anticollision_resolution"] = None
    st.session_state["wt_last_anticollision_previous_successes"] = {}
    st.session_state["wt_anticollision_analysis_cache"] = {}


def _focus_all_wells_anticollision_results() -> None:
    st.session_state["wt_results_view_mode"] = "Все скважины"
    st.session_state["wt_results_all_view_mode"] = "Anti-collision"
    st.session_state["wt_3d_render_mode"] = WT_3D_RENDER_DETAIL
    st.session_state["wt_3d_backend"] = WT_3D_BACKEND_THREE_LOCAL


def _focus_all_wells_trajectory_results() -> None:
    st.session_state["wt_results_view_mode"] = "Все скважины"
    st.session_state["wt_results_all_view_mode"] = "Траектории"
    st.session_state["wt_3d_render_mode"] = WT_3D_RENDER_DETAIL
    st.session_state["wt_3d_backend"] = WT_3D_BACKEND_THREE_LOCAL


def _bump_three_viewer_nonce() -> None:
    st.session_state["wt_three_viewer_nonce"] = (
        int(st.session_state.get("wt_three_viewer_nonce", 0)) + 1
    )


def _clear_pad_state() -> None:
    st.session_state["wt_pad_configs"] = {}
    st.session_state["wt_pad_detected_meta"] = {}
    st.session_state["wt_pad_selected_id"] = ""
    st.session_state["wt_pad_last_applied_at"] = ""
    st.session_state["wt_pad_auto_applied_on_import"] = False
    for key in list(st.session_state.keys()):
        if str(key).startswith("wt_pad_cfg_"):
            del st.session_state[key]


def _apply_profile_defaults(force: bool) -> None:
    WT_CALC_PARAMS.preserve_state()
    legacy_found = _migrate_legacy_calc_param_keys()
    WT_CALC_PARAMS.apply_defaults(force=bool(force or legacy_found))


def _migrate_legacy_calc_param_keys() -> bool:
    legacy_found = False
    tolerance_legacy_keys = ("wt_cfg_pos_tolerance_m", "wt_cfg_pos_tol")
    tolerance_legacy_value: float | None = None
    for legacy_key in tolerance_legacy_keys:
        if legacy_key in st.session_state:
            if tolerance_legacy_value is None:
                try:
                    tolerance_legacy_value = float(
                        st.session_state[legacy_key]
                    )
                except (TypeError, ValueError):
                    tolerance_legacy_value = None
            del st.session_state[legacy_key]
            legacy_found = True
    if tolerance_legacy_value is not None:
        st.session_state.setdefault(
            "wt_cfg_lateral_tol", float(tolerance_legacy_value)
        )
        st.session_state.setdefault(
            "wt_cfg_vertical_tol", float(tolerance_legacy_value)
        )
    for legacy_key, new_key in _WT_LEGACY_KEY_ALIASES.items():
        if legacy_key in st.session_state:
            # Legacy widget keys are removed from active flow to avoid stale/corrupted
            # values being copied back into the new unified parameter keys.
            del st.session_state[legacy_key]
            legacy_found = True
        if new_key in st.session_state:
            st.session_state[new_key] = st.session_state[new_key]
    return legacy_found


def _decode_welltrack_payload(raw_payload: bytes, source_label: str) -> str:
    text, encoding = decode_welltrack_bytes(raw_payload)
    if encoding == "utf-8":
        return text
    if encoding.endswith("(replace)"):
        st.warning(
            f"{source_label}: не удалось надежно определить кодировку. "
            f"Текст декодирован как `{encoding}` с заменой поврежденных символов."
        )
        return text
    st.info(
        f"{source_label}: текст декодирован как `{encoding}` (fallback, не UTF-8). "
        "Проверьте корректность имен и комментариев."
    )
    return text


def _read_welltrack_file(path_text: str) -> str:
    file_path_raw = path_text.strip()
    if not file_path_raw:
        st.warning("Укажите путь к файлу WELLTRACK.")
        return ""

    file_path = Path(file_path_raw).expanduser()
    if not file_path.is_absolute():
        file_path = (Path.cwd() / file_path).resolve()

    if not file_path.exists():
        st.error(f"Файл не найден: {file_path}")
        return ""
    if not file_path.is_file():
        st.error(f"Путь не является файлом: {file_path}")
        return ""

    try:
        payload = file_path.read_bytes()
        return _decode_welltrack_payload(
            payload, source_label=f"Файл `{file_path}`"
        )
    except OSError as exc:
        st.error(f"Не удалось прочитать файл `{file_path}`: {exc}")
        return ""


def _all_wells_3d_figure(
    successes: list[SuccessfulWellPlan],
    *,
    target_only_wells: list[_TargetOnlyWell] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    name_to_color: dict[str, str] | None = None,
    focus_well_names: tuple[str, ...] = (),
    render_mode: str = WT_3D_RENDER_FAST,
    height: int = 620,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
    z_focus_arrays: list[np.ndarray] = []
    resolved_render_mode = _resolve_3d_render_mode(
        requested_mode=render_mode,
        calculated_well_count=len(successes),
        reference_wells=reference_wells,
    )
    focus_set = {str(name) for name in focus_well_names if str(name).strip()}
    color_map = name_to_color or {
        str(item.name): _well_color(index)
        for index, item in enumerate(successes)
    }
    for index, item in enumerate(successes):
        line_color = color_map.get(str(item.name), _well_color(index))
        line_dash = "dash" if bool(item.md_postcheck_exceeded) else "solid"
        name = item.name
        stations = (
            _decimated_station_frame(
                item.stations,
                target_points=WT_3D_FAST_CALC_TARGET_POINTS,
            )
            if resolved_render_mode == WT_3D_RENDER_FAST
            else item.stations
        )
        x_arrays.append(stations["X_m"].to_numpy(dtype=float))
        y_arrays.append(stations["Y_m"].to_numpy(dtype=float))
        z_arrays.append(stations["Z_m"].to_numpy(dtype=float))
        if not focus_set or str(item.name) in focus_set:
            x_focus_arrays.append(stations["X_m"].to_numpy(dtype=float))
            y_focus_arrays.append(stations["Y_m"].to_numpy(dtype=float))
            z_focus_arrays.append(stations["Z_m"].to_numpy(dtype=float))
        fig.add_trace(
            go.Scatter3d(
                x=stations["X_m"],
                y=stations["Y_m"],
                z=stations["Z_m"],
                mode="lines",
                name=name,
                line={"width": 5, "color": line_color, "dash": line_dash},
                customdata=_trajectory_hover_customdata(stations),
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{z:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m<br>"
                    "DLS: %{customdata[1]:.2f} deg/30м<br>"
                    "INC: %{customdata[2]:.2f} deg<br>"
                    "Сегмент: %{customdata[3]}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        surface = item.surface
        t1 = item.t1
        t3 = item.t3
        x_arrays.append(np.array([surface.x, t1.x, t3.x], dtype=float))
        y_arrays.append(np.array([surface.y, t1.y, t3.y], dtype=float))
        z_arrays.append(np.array([surface.z, t1.z, t3.z], dtype=float))
        if not focus_set or str(item.name) in focus_set:
            x_focus_arrays.append(
                np.array([surface.x, t1.x, t3.x], dtype=float)
            )
            y_focus_arrays.append(
                np.array([surface.y, t1.y, t3.y], dtype=float)
            )
            z_focus_arrays.append(
                np.array([surface.z, t1.z, t3.z], dtype=float)
            )
        fig.add_trace(
            go.Scatter3d(
                x=[surface.x, t1.x, t3.x],
                y=[surface.y, t1.y, t3.y],
                z=[surface.z, t1.z, t3.z],
                mode="markers",
                name=f"{name}: S/t1/t3",
                marker={
                    "size": 5,
                    "color": line_color,
                    "line": {"width": 1, "color": "rgba(255,255,255,0.9)"},
                },
                showlegend=False,
                hovertemplate="X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z/TVD: %{z:.2f} m<extra>%{fullData.name}</extra>",
            )
        )
        fig.add_trace(
            _t1_name_trace_3d(
                well_name=str(name),
                t1=t1,
                color=line_color,
            )
        )

    if resolved_render_mode == WT_3D_RENDER_FAST:
        for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
            combined_trace = _combined_reference_trace_3d(
                reference_wells=reference_wells,
                kind=kind,
                target_points_per_well=WT_3D_FAST_REFERENCE_TARGET_POINTS,
            )
            if combined_trace is not None:
                fig.add_trace(combined_trace)
                x_arrays.append(np.asarray(combined_trace.x, dtype=float))
                y_arrays.append(np.asarray(combined_trace.y, dtype=float))
                z_arrays.append(np.asarray(combined_trace.z, dtype=float))
    else:
        for reference_well in reference_wells:
            stations = reference_well.stations
            if stations.empty:
                continue
            line_color = REFERENCE_WELL_KIND_COLORS.get(
                str(reference_well.kind),
                "#A0A0A0",
            )
            x_values = stations["X_m"].to_numpy(dtype=float)
            y_values = stations["Y_m"].to_numpy(dtype=float)
            z_values = stations["Z_m"].to_numpy(dtype=float)
            x_arrays.append(x_values)
            y_arrays.append(y_values)
            z_arrays.append(z_values)
            fig.add_trace(
                go.Scatter3d(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    mode="lines",
                    name=reference_well_display_label(reference_well),
                    showlegend=False,
                    line={"width": 4, "color": line_color},
                    customdata=np.column_stack(
                        [stations["MD_m"].to_numpy(dtype=float)]
                    ),
                    hovertemplate=(
                        "Тип: "
                        + REFERENCE_WELL_KIND_LABELS.get(
                            str(reference_well.kind),
                            str(reference_well.kind),
                        )
                        + "<br>"
                        "X: %{x:.2f} m<br>"
                        "Y: %{y:.2f} m<br>"
                        "Z/TVD: %{z:.2f} m<br>"
                        "MD: %{customdata[0]:.2f} m"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    for kind in _reference_kinds_present(reference_wells):
        fig.add_trace(_reference_legend_trace_3d(kind))
    if resolved_render_mode == WT_3D_RENDER_DETAIL:
        for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
            label_trace = _reference_name_trace_3d(reference_wells, kind=kind)
            if label_trace is not None:
                fig.add_trace(label_trace)
    pad_label_trace = _reference_pad_label_trace_3d(reference_wells)
    if pad_label_trace is not None:
        fig.add_trace(pad_label_trace)

    for target_only in target_only_wells or ():
        line_color = color_map.get(str(target_only.name), "#6B7280")
        marker_x = np.array(
            [target_only.surface.x, target_only.t1.x, target_only.t3.x],
            dtype=float,
        )
        marker_y = np.array(
            [target_only.surface.y, target_only.t1.y, target_only.t3.y],
            dtype=float,
        )
        marker_z = np.array(
            [target_only.surface.z, target_only.t1.z, target_only.t3.z],
            dtype=float,
        )
        x_arrays.append(marker_x)
        y_arrays.append(marker_y)
        z_arrays.append(marker_z)
        if not focus_set or str(target_only.name) in focus_set:
            x_focus_arrays.append(marker_x)
            y_focus_arrays.append(marker_y)
            z_focus_arrays.append(marker_z)
        customdata = np.array(
            [
                ["S", target_only.status, target_only.problem or "—"],
                ["t1", target_only.status, target_only.problem or "—"],
                ["t3", target_only.status, target_only.problem or "—"],
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter3d(
                x=marker_x,
                y=marker_y,
                z=marker_z,
                mode="markers",
                name=f"{target_only.name}: цели (без траектории)",
                legendgroup=str(target_only.name),
                marker={
                    "size": 10,
                    "symbol": "cross",
                    "color": line_color,
                    "line": {"width": 3, "color": line_color},
                },
                customdata=customdata,
                hovertemplate=(
                    "Точка: %{customdata[0]}<br>"
                    "Статус: %{customdata[1]}<br>"
                    "Проблема: %{customdata[2]}<br>"
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{z:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        fig.add_trace(
            _t1_name_trace_3d(
                well_name=str(target_only.name),
                t1=target_only.t1,
                color=line_color,
            )
        )

    x_values = (
        (
            np.concatenate(x_focus_arrays)
            if x_focus_arrays
            else np.concatenate(x_arrays)
        )
        if x_arrays
        else np.array([0.0], dtype=float)
    )
    y_values = (
        (
            np.concatenate(y_focus_arrays)
            if y_focus_arrays
            else np.concatenate(y_arrays)
        )
        if y_arrays
        else np.array([0.0], dtype=float)
    )
    z_values = (
        (
            np.concatenate(z_focus_arrays)
            if z_focus_arrays
            else np.concatenate(z_arrays)
        )
        if z_arrays
        else np.array([0.0], dtype=float)
    )
    x_range, y_range, z_range = equalized_axis_ranges(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )
    xy_span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    xy_dtick = nice_tick_step(xy_span, target_ticks=6)
    x_tickvals = linear_tick_values(axis_range=x_range, step=xy_dtick)
    y_tickvals = linear_tick_values(axis_range=y_range, step=xy_dtick)
    xy_axis_style = {
        "tickmode": "array",
        "tickformat": ".0f",
        "showexponent": "none",
        "exponentformat": "none",
        "showgrid": True,
        "gridcolor": "rgba(0, 0, 0, 0.15)",
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": "rgba(0, 0, 0, 0.65)",
        "zerolinewidth": 2,
        "showline": True,
        "linecolor": "rgba(0, 0, 0, 0.65)",
        "linewidth": 1.5,
    }

    fig.update_layout(
        title="Все рассчитанные скважины (3D)",
        scene={
            "xaxis_title": "X / Восток (м)",
            "yaxis_title": "Y / Север (м)",
            "zaxis_title": "Z / TVD (м)",
            "camera": DEFAULT_3D_CAMERA,
            "xaxis": {
                "range": x_range,
                "tickvals": x_tickvals,
                **xy_axis_style,
            },
            "yaxis": {
                "range": y_range,
                "tickvals": y_tickvals,
                **xy_axis_style,
            },
            "zaxis": {
                "range": z_range,
                "tickformat": ".0f",
                "showexponent": "none",
                "exponentformat": "none",
                "showgrid": True,
                "gridcolor": "rgba(0, 0, 0, 0.12)",
                "gridwidth": 1,
                "zeroline": True,
                "zerolinecolor": "rgba(0, 0, 0, 0.45)",
                "zerolinewidth": 1,
            },
            "aspectmode": "cube",
        },
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def _all_wells_plan_figure(
    successes: list[SuccessfulWellPlan],
    *,
    target_only_wells: list[_TargetOnlyWell] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    name_to_color: dict[str, str] | None = None,
    focus_well_names: tuple[str, ...] = (),
    height: int = 560,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
    focus_set = {str(name) for name in focus_well_names if str(name).strip()}
    color_map = name_to_color or {
        str(item.name): _well_color(index)
        for index, item in enumerate(successes)
    }
    for index, item in enumerate(successes):
        line_color = color_map.get(str(item.name), _well_color(index))
        line_dash = "dash" if bool(item.md_postcheck_exceeded) else "solid"
        name = item.name
        stations = item.stations
        x_arrays.append(stations["X_m"].to_numpy(dtype=float))
        y_arrays.append(stations["Y_m"].to_numpy(dtype=float))
        if not focus_set or str(item.name) in focus_set:
            x_focus_arrays.append(stations["X_m"].to_numpy(dtype=float))
            y_focus_arrays.append(stations["Y_m"].to_numpy(dtype=float))
        fig.add_trace(
            go.Scatter(
                x=stations["X_m"],
                y=stations["Y_m"],
                mode="lines",
                name=name,
                line={"width": 1.5, "color": line_color, "dash": line_dash},
                customdata=np.column_stack(
                    [
                        stations["Z_m"].to_numpy(dtype=float),
                        stations["MD_m"].to_numpy(dtype=float),
                        dls_to_pi(
                            stations["DLS_deg_per_30m"]
                            .fillna(0.0)
                            .to_numpy(dtype=float)
                        ),
                    ]
                ),
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{customdata[0]:.2f} m<br>"
                    "MD: %{customdata[1]:.2f} m<br>"
                    "ПИ: %{customdata[2]:.2f} deg/10m"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        surface = item.surface
        t1 = item.t1
        t3 = item.t3
        x_arrays.append(np.array([surface.x, t1.x, t3.x], dtype=float))
        y_arrays.append(np.array([surface.y, t1.y, t3.y], dtype=float))
        if not focus_set or str(item.name) in focus_set:
            x_focus_arrays.append(
                np.array([surface.x, t1.x, t3.x], dtype=float)
            )
            y_focus_arrays.append(
                np.array([surface.y, t1.y, t3.y], dtype=float)
            )
        fig.add_trace(
            go.Scatter(
                x=[surface.x, t1.x, t3.x],
                y=[surface.y, t1.y, t3.y],
                mode="markers",
                name=f"{name}: S/t1/t3",
                marker={
                    "size": 7,
                    "color": line_color,
                    "line": {"width": 1, "color": "rgba(255,255,255,0.9)"},
                },
                showlegend=False,
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        fig.add_trace(
            _t1_name_trace_2d(
                well_name=str(name),
                x_value=float(t1.x),
                y_value=float(t1.y),
                color=line_color,
            )
        )

    for reference_well in reference_wells:
        stations = reference_well.stations
        if stations.empty:
            continue
        line_color = REFERENCE_WELL_KIND_COLORS.get(
            str(reference_well.kind),
            "#A0A0A0",
        )
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        x_arrays.append(x_values)
        y_arrays.append(y_values)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=reference_well_display_label(reference_well),
                showlegend=False,
                line={"width": 2, "color": line_color},
                customdata=np.column_stack(
                    [stations["MD_m"].to_numpy(dtype=float)]
                ),
                hovertemplate=(
                    "Тип: "
                    + REFERENCE_WELL_KIND_LABELS.get(
                        str(reference_well.kind),
                        str(reference_well.kind),
                    )
                    + "<br>"
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    for kind in _reference_kinds_present(reference_wells):
        fig.add_trace(_reference_legend_trace_2d(kind))
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label_trace = _reference_name_trace_2d(reference_wells, kind=kind)
        if label_trace is not None:
            fig.add_trace(label_trace)
    pad_label_trace = _reference_pad_label_trace_2d(reference_wells)
    if pad_label_trace is not None:
        fig.add_trace(pad_label_trace)

    for target_only in target_only_wells or ():
        line_color = color_map.get(str(target_only.name), "#6B7280")
        marker_x = np.array(
            [target_only.surface.x, target_only.t1.x, target_only.t3.x],
            dtype=float,
        )
        marker_y = np.array(
            [target_only.surface.y, target_only.t1.y, target_only.t3.y],
            dtype=float,
        )
        x_arrays.append(marker_x)
        y_arrays.append(marker_y)
        if not focus_set or str(target_only.name) in focus_set:
            x_focus_arrays.append(marker_x)
            y_focus_arrays.append(marker_y)
        customdata = np.array(
            [
                ["S", target_only.status, target_only.problem or "—"],
                ["t1", target_only.status, target_only.problem or "—"],
                ["t3", target_only.status, target_only.problem or "—"],
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                name=f"{target_only.name}: цели (без траектории)",
                marker={
                    "size": 12,
                    "symbol": "x",
                    "color": line_color,
                    "line": {"width": 3, "color": line_color},
                },
                customdata=customdata,
                hovertemplate=(
                    "Точка: %{customdata[0]}<br>"
                    "Статус: %{customdata[1]}<br>"
                    "Проблема: %{customdata[2]}<br>"
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        fig.add_trace(
            _t1_name_trace_2d(
                well_name=str(target_only.name),
                x_value=float(target_only.t1.x),
                y_value=float(target_only.t1.y),
                color=line_color,
            )
        )
    x_values = (
        (
            np.concatenate(x_focus_arrays)
            if x_focus_arrays
            else np.concatenate(x_arrays)
        )
        if x_arrays
        else np.array([0.0], dtype=float)
    )
    y_values = (
        (
            np.concatenate(y_focus_arrays)
            if y_focus_arrays
            else np.concatenate(y_arrays)
        )
        if y_arrays
        else np.array([0.0], dtype=float)
    )
    x_range, y_range = equalized_xy_ranges(
        x_values=x_values, y_values=y_values
    )
    xy_dtick = nice_tick_step(
        max(x_range[1] - x_range[0], y_range[1] - y_range[0]), target_ticks=6
    )
    x_tickvals = linear_tick_values(axis_range=x_range, step=xy_dtick)
    y_tickvals = linear_tick_values(axis_range=y_range, step=xy_dtick)

    fig.update_layout(
        title="Все рассчитанные скважины (план E-N, X=Восток, Y=Север)",
        xaxis_title="X / Восток (м)",
        yaxis_title="Y / Север (м)",
        xaxis={
            "range": x_range,
            "tickmode": "array",
            "tickvals": x_tickvals,
            "tickformat": ".0f",
            "showexponent": "none",
            "exponentformat": "none",
        },
        yaxis={
            "range": y_range,
            "tickmode": "array",
            "tickvals": y_tickvals,
            "tickformat": ".0f",
            "showexponent": "none",
            "exponentformat": "none",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def _all_wells_anticollision_3d_figure(
    analysis: AntiCollisionAnalysis,
    *,
    previous_successes_by_name: Mapping[str, SuccessfulWellPlan] | None = None,
    focus_well_names: tuple[str, ...] = (),
    render_mode: str = WT_3D_RENDER_FAST,
    height: int = 660,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
    z_focus_arrays: list[np.ndarray] = []
    reference_wells = tuple(
        well for well in analysis.wells if bool(well.is_reference_only)
    )
    resolved_render_mode = _resolve_3d_render_mode(
        requested_mode=render_mode,
        calculated_well_count=len(
            [
                well
                for well in analysis.wells
                if not bool(well.is_reference_only)
            ]
        ),
        reference_wells=reference_wells,
    )
    focus_set = {str(name) for name in focus_well_names if str(name).strip()}
    focus_reference_names = (
        _anticollision_reference_cone_focus_names(analysis)
        if resolved_render_mode == WT_3D_RENDER_FAST
        else set()
    )
    well_lookup = {str(well.name): well for well in analysis.wells}
    aggregated_reference_wells: list = []

    for well in analysis.wells:
        is_reference_only = bool(well.is_reference_only)
        is_focus_reference = str(well.name) in focus_reference_names
        if (
            resolved_render_mode == WT_3D_RENDER_FAST
            and is_reference_only
            and not is_focus_reference
        ):
            aggregated_reference_wells.append(well)
            continue
        well_label = (
            f"{well.name} ({REFERENCE_WELL_KIND_LABELS.get(str(well.well_kind), str(well.well_kind))})"
            if is_reference_only
            else str(well.name)
        )
        overlay = well.overlay
        tube_mesh = (
            build_uncertainty_tube_mesh(overlay)
            if (
                resolved_render_mode != WT_3D_RENDER_FAST
                or not is_reference_only
                or is_focus_reference
            )
            else None
        )
        include_in_focus = (not focus_set and not is_reference_only) or (
            bool(focus_set) and str(well.name) in focus_set
        )
        if tube_mesh is not None:
            x_arrays.append(tube_mesh.vertices_xyz[:, 0])
            y_arrays.append(tube_mesh.vertices_xyz[:, 1])
            z_arrays.append(tube_mesh.vertices_xyz[:, 2])
            if include_in_focus:
                x_focus_arrays.append(tube_mesh.vertices_xyz[:, 0])
                y_focus_arrays.append(tube_mesh.vertices_xyz[:, 1])
                z_focus_arrays.append(tube_mesh.vertices_xyz[:, 2])
            fig.add_trace(
                go.Mesh3d(
                    x=tube_mesh.vertices_xyz[:, 0],
                    y=tube_mesh.vertices_xyz[:, 1],
                    z=tube_mesh.vertices_xyz[:, 2],
                    i=tube_mesh.i,
                    j=tube_mesh.j,
                    k=tube_mesh.k,
                    name=f"{well_label} cone",
                    legendgroup=str(well.name),
                    showlegend=False,
                    color=str(well.color),
                    opacity=0.12,
                    flatshading=True,
                    hoverinfo="skip",
                )
            )
        if overlay.samples and (
            resolved_render_mode != WT_3D_RENDER_FAST or not is_reference_only
        ):
            terminal_ring = np.asarray(
                overlay.samples[-1].ring_xyz, dtype=float
            )
            x_arrays.append(terminal_ring[:, 0])
            y_arrays.append(terminal_ring[:, 1])
            z_arrays.append(terminal_ring[:, 2])
            if include_in_focus:
                x_focus_arrays.append(terminal_ring[:, 0])
                y_focus_arrays.append(terminal_ring[:, 1])
                z_focus_arrays.append(terminal_ring[:, 2])
            fig.add_trace(
                go.Scatter3d(
                    x=terminal_ring[:, 0],
                    y=terminal_ring[:, 1],
                    z=terminal_ring[:, 2],
                    mode="lines",
                    name=f"{well_label}: граница конуса",
                    legendgroup=str(well.name),
                    showlegend=False,
                    line={
                        "width": 0.8,
                        "color": _lighten_hex(str(well.color), 0.55),
                    },
                    hoverinfo="skip",
                )
            )

        stations = (
            _decimated_station_frame(
                well.stations,
                target_points=(
                    WT_3D_FAST_REFERENCE_TARGET_POINTS
                    if is_reference_only
                    else WT_3D_FAST_CALC_TARGET_POINTS
                ),
            )
            if resolved_render_mode == WT_3D_RENDER_FAST
            else well.stations
        )
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        z_values = stations["Z_m"].to_numpy(dtype=float)

        x_arrays.append(x_values)
        y_arrays.append(y_values)
        z_arrays.append(z_values)
        if include_in_focus:
            x_focus_arrays.append(x_values)
            y_focus_arrays.append(y_values)
            z_focus_arrays.append(z_values)

        fig.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                name=well_label,
                legendgroup=str(well.name),
                showlegend=not is_reference_only,
                line={"width": 5, "color": str(well.color)},
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{z:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m<br>"
                    "DLS: %{customdata[1]:.2f} deg/30м<br>"
                    "INC: %{customdata[2]:.2f} deg<br>"
                    "Сегмент: %{customdata[3]}"
                    "<extra>%{fullData.name}</extra>"
                ),
                customdata=_trajectory_hover_customdata(stations),
            )
        )
        previous_success = (previous_successes_by_name or {}).get(
            str(well.name)
        )
        if (
            previous_success is not None
            and not previous_success.stations.empty
        ):
            previous_stations = previous_success.stations
            previous_x = previous_stations["X_m"].to_numpy(dtype=float)
            previous_y = previous_stations["Y_m"].to_numpy(dtype=float)
            previous_z = previous_stations["Z_m"].to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=previous_x,
                    y=previous_y,
                    z=previous_z,
                    mode="lines",
                    name=f"{well.name}: до anti-collision",
                    legendgroup=str(well.name),
                    showlegend=False,
                    line={
                        "width": 2.0,
                        "color": _hex_to_rgba(str(well.color), 0.55),
                        "dash": "dot",
                    },
                    customdata=np.column_stack(
                        [previous_stations["MD_m"].to_numpy(dtype=float)]
                    ),
                    hovertemplate=(
                        "X: %{x:.2f} m<br>"
                        "Y: %{y:.2f} m<br>"
                        "Z/TVD: %{z:.2f} m<br>"
                        "MD: %{customdata[0]:.2f} m"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            x_arrays.append(previous_x)
            y_arrays.append(previous_y)
            z_arrays.append(previous_z)
            if include_in_focus:
                x_focus_arrays.append(previous_x)
                y_focus_arrays.append(previous_y)
                z_focus_arrays.append(previous_z)
        if resolved_render_mode != WT_3D_RENDER_FAST or not is_reference_only:
            fig.add_trace(
                _hover_proxy_trace_3d(
                    x_values=x_values,
                    y_values=y_values,
                    z_values=z_values,
                    customdata=_trajectory_hover_customdata(stations),
                    hovertemplate=(
                        "X: %{x:.2f} m<br>"
                        "Y: %{y:.2f} m<br>"
                        "Z/TVD: %{z:.2f} m<br>"
                        "MD: %{customdata[0]:.2f} m<br>"
                        "DLS: %{customdata[1]:.2f} deg/30м<br>"
                        "INC: %{customdata[2]:.2f} deg<br>"
                        "Сегмент: %{customdata[3]}"
                        f"<extra>{well_label}</extra>"
                    ),
                )
            )
        if (
            (well.t1 is not None)
            and (well.t3 is not None)
            and not is_reference_only
        ):
            fig.add_trace(
                go.Scatter3d(
                    x=[well.surface.x, well.t1.x, well.t3.x],
                    y=[well.surface.y, well.t1.y, well.t3.y],
                    z=[well.surface.z, well.t1.z, well.t3.z],
                    mode="markers",
                    name=f"{well_label}: цели",
                    legendgroup=str(well.name),
                    showlegend=False,
                    marker={
                        "size": 5,
                        "color": str(well.color),
                        "line": {"width": 1, "color": "rgba(255,255,255,0.9)"},
                    },
                    customdata=np.array([["S"], ["t1"], ["t3"]], dtype=object),
                    hovertemplate=(
                        "Точка: %{customdata[0]}<br>"
                        "X: %{x:.2f} m<br>"
                        "Y: %{y:.2f} m<br>"
                        "Z/TVD: %{z:.2f} m<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            fig.add_trace(
                _t1_name_trace_3d(
                    well_name=str(well.name),
                    t1=well.t1,
                    color=str(well.color),
                )
            )
            x_arrays.append(
                np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float)
            )
            y_arrays.append(
                np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float)
            )
            z_arrays.append(
                np.array([well.surface.z, well.t1.z, well.t3.z], dtype=float)
            )
            if include_in_focus:
                x_focus_arrays.append(
                    np.array(
                        [well.surface.x, well.t1.x, well.t3.x], dtype=float
                    )
                )
                y_focus_arrays.append(
                    np.array(
                        [well.surface.y, well.t1.y, well.t3.y], dtype=float
                    )
                )
                z_focus_arrays.append(
                    np.array(
                        [well.surface.z, well.t1.z, well.t3.z], dtype=float
                    )
                )

    if aggregated_reference_wells:
        for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
            combined_trace = _combined_reference_trace_3d(
                reference_wells=[
                    well
                    for well in aggregated_reference_wells
                    if str(well.well_kind) == str(kind)
                ],
                kind=kind,
                target_points_per_well=WT_3D_FAST_REFERENCE_TARGET_POINTS,
            )
            if combined_trace is not None:
                fig.add_trace(combined_trace)
                x_arrays.append(np.asarray(combined_trace.x, dtype=float))
                y_arrays.append(np.asarray(combined_trace.y, dtype=float))
                z_arrays.append(np.asarray(combined_trace.z, dtype=float))
    for kind in _reference_kinds_present(reference_wells):
        fig.add_trace(_reference_legend_trace_3d(kind))
    analysis_reference_wells = _analysis_reference_wells(analysis)
    if resolved_render_mode == WT_3D_RENDER_DETAIL:
        for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
            label_trace = _reference_name_trace_3d(
                analysis_reference_wells,
                kind=kind,
            )
            if label_trace is not None:
                fig.add_trace(label_trace)
    pad_label_trace = _reference_pad_label_trace_3d(analysis_reference_wells)
    if pad_label_trace is not None:
        fig.add_trace(pad_label_trace)

    overlap_legend_added = False
    for corridor in analysis.corridors:
        corridor_in_focus = (not focus_set) or bool(
            {str(corridor.well_a), str(corridor.well_b)}.intersection(
                focus_set
            )
        )
        mesh = collision_corridor_tube_mesh(corridor)
        if mesh is not None:
            x_arrays.append(mesh.vertices_xyz[:, 0])
            y_arrays.append(mesh.vertices_xyz[:, 1])
            z_arrays.append(mesh.vertices_xyz[:, 2])
            if corridor_in_focus:
                x_focus_arrays.append(mesh.vertices_xyz[:, 0])
                y_focus_arrays.append(mesh.vertices_xyz[:, 1])
                z_focus_arrays.append(mesh.vertices_xyz[:, 2])
            fig.add_trace(
                go.Mesh3d(
                    x=mesh.vertices_xyz[:, 0],
                    y=mesh.vertices_xyz[:, 1],
                    z=mesh.vertices_xyz[:, 2],
                    i=mesh.i,
                    j=mesh.j,
                    k=mesh.k,
                    name="Общая зона overlap",
                    legendgroup="collision_overlap",
                    showlegend=not overlap_legend_added,
                    color="rgb(198, 40, 40)",
                    opacity=0.42,
                    flatshading=True,
                    hoverinfo="skip",
                    hovertemplate=None,
                )
            )
        else:
            sphere_x, sphere_y, sphere_z = (
                collision_corridor_point_sphere_mesh(corridor)
            )
            x_arrays.append(sphere_x.reshape(-1))
            y_arrays.append(sphere_y.reshape(-1))
            z_arrays.append(sphere_z.reshape(-1))
            if corridor_in_focus:
                x_focus_arrays.append(sphere_x.reshape(-1))
                y_focus_arrays.append(sphere_y.reshape(-1))
                z_focus_arrays.append(sphere_z.reshape(-1))
            fig.add_trace(
                go.Surface(
                    x=sphere_x,
                    y=sphere_y,
                    z=sphere_z,
                    surfacecolor=np.ones_like(sphere_x, dtype=float),
                    colorscale=[
                        [0.0, "rgb(198, 40, 40)"],
                        [1.0, "rgb(198, 40, 40)"],
                    ],
                    cmin=0.0,
                    cmax=1.0,
                    showscale=False,
                    opacity=0.42,
                    hoverinfo="skip",
                    hovertemplate=None,
                    name="Общая зона overlap",
                    showlegend=not overlap_legend_added,
                    legendgroup="collision_overlap",
                )
            )
        overlap_legend_added = True

    segment_legend_added = False
    for segment in analysis.well_segments:
        well = well_lookup.get(str(segment.well_name))
        if well is None:
            continue
        interval_points = _trajectory_interval_points(
            well.stations,
            md_start_m=float(segment.md_start_m),
            md_end_m=float(segment.md_end_m),
        )
        if interval_points is None:
            continue
        x_segment = interval_points["X_m"].to_numpy(dtype=float)
        y_segment = interval_points["Y_m"].to_numpy(dtype=float)
        z_segment = interval_points["Z_m"].to_numpy(dtype=float)
        md_segment = interval_points["MD_m"].to_numpy(dtype=float)
        dls_segment = np.interp(
            md_segment,
            well.stations["MD_m"].to_numpy(dtype=float),
            well.stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float),
        )
        inc_segment = np.interp(
            md_segment,
            well.stations["MD_m"].to_numpy(dtype=float),
            well.stations["INC_deg"].fillna(0.0).to_numpy(dtype=float),
        )
        conflict_customdata = np.column_stack(
            [
                md_segment,
                dls_segment,
                inc_segment,
                np.full(len(md_segment), "Конфликтный участок", dtype=object),
            ]
        )
        x_arrays.append(x_segment)
        y_arrays.append(y_segment)
        z_arrays.append(z_segment)
        if (not focus_set) or str(segment.well_name) in focus_set:
            x_focus_arrays.append(x_segment)
            y_focus_arrays.append(y_segment)
            z_focus_arrays.append(z_segment)
        fig.add_trace(
            go.Scatter3d(
                x=x_segment,
                y=y_segment,
                z=z_segment,
                mode="lines",
                name="Конфликтный участок ствола",
                legendgroup="collision_path",
                showlegend=not segment_legend_added,
                line={"width": 8, "color": "rgb(198, 40, 40)"},
                customdata=conflict_customdata,
                hovertemplate=(
                    f"{segment.well_name}<br>"
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{z:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} м<br>"
                    "DLS: %{customdata[1]:.2f} deg/30м<br>"
                    "INC: %{customdata[2]:.2f} deg"
                    "<extra>Конфликтный участок ствола</extra>"
                ),
            )
        )
        fig.add_trace(
            _hover_proxy_trace_3d(
                x_values=x_segment,
                y_values=y_segment,
                z_values=z_segment,
                customdata=conflict_customdata,
                hovertemplate=(
                    f"{segment.well_name}<br>"
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{z:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} м<br>"
                    "DLS: %{customdata[1]:.2f} deg/30м<br>"
                    "INC: %{customdata[2]:.2f} deg"
                    "<extra>Конфликтный участок ствола</extra>"
                ),
            )
        )
        segment_legend_added = True

    x_values = (
        (
            np.concatenate(x_focus_arrays)
            if x_focus_arrays
            else np.concatenate(x_arrays)
        )
        if x_arrays
        else np.array([0.0], dtype=float)
    )
    y_values = (
        (
            np.concatenate(y_focus_arrays)
            if y_focus_arrays
            else np.concatenate(y_arrays)
        )
        if y_arrays
        else np.array([0.0], dtype=float)
    )
    z_values = (
        (
            np.concatenate(z_focus_arrays)
            if z_focus_arrays
            else np.concatenate(z_arrays)
        )
        if z_arrays
        else np.array([0.0], dtype=float)
    )
    x_range, y_range, z_range = equalized_axis_ranges(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )
    xy_span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    xy_dtick = nice_tick_step(xy_span, target_ticks=6)
    x_tickvals = linear_tick_values(axis_range=x_range, step=xy_dtick)
    y_tickvals = linear_tick_values(axis_range=y_range, step=xy_dtick)
    xy_axis_style = {
        "tickmode": "array",
        "tickformat": ".0f",
        "showexponent": "none",
        "exponentformat": "none",
        "showgrid": True,
        "gridcolor": "rgba(0, 0, 0, 0.15)",
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": "rgba(0, 0, 0, 0.65)",
        "zerolinewidth": 2,
        "showline": True,
        "linecolor": "rgba(0, 0, 0, 0.65)",
        "linewidth": 1.5,
    }

    fig.update_layout(
        title="Anti-collision: 3D конусы неопределенности",
        scene={
            "xaxis_title": "X / Восток (м)",
            "yaxis_title": "Y / Север (м)",
            "zaxis_title": "Z / TVD (м)",
            "camera": DEFAULT_3D_CAMERA,
            "xaxis": {
                "range": x_range,
                "tickvals": x_tickvals,
                **xy_axis_style,
            },
            "yaxis": {
                "range": y_range,
                "tickvals": y_tickvals,
                **xy_axis_style,
            },
            "zaxis": {
                "range": z_range,
                "tickformat": ".0f",
                "showexponent": "none",
                "exponentformat": "none",
                "showgrid": True,
                "gridcolor": "rgba(0, 0, 0, 0.12)",
                "gridwidth": 1,
                "zeroline": True,
                "zerolinecolor": "rgba(0, 0, 0, 0.45)",
                "zerolinewidth": 1,
            },
            "aspectmode": "cube",
        },
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def _all_wells_anticollision_plan_figure(
    analysis: AntiCollisionAnalysis,
    *,
    previous_successes_by_name: Mapping[str, SuccessfulWellPlan] | None = None,
    focus_well_names: tuple[str, ...] = (),
    height: int = 620,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
    focus_set = {str(name) for name in focus_well_names if str(name).strip()}
    well_lookup = {str(well.name): well for well in analysis.wells}

    for well in analysis.wells:
        well_label = (
            f"{well.name} ({REFERENCE_WELL_KIND_LABELS.get(str(well.well_kind), str(well.well_kind))})"
            if bool(well.is_reference_only)
            else str(well.name)
        )
        overlay = well.overlay
        ribbon = uncertainty_ribbon_polygon(overlay, projection="plan")
        if len(ribbon) >= 3:
            fig.add_trace(
                go.Scatter(
                    x=ribbon[:, 0],
                    y=ribbon[:, 1],
                    mode="lines",
                    name=f"{well_label} cone",
                    legendgroup=str(well.name),
                    showlegend=False,
                    line={"width": 0.0, "color": "rgba(0, 0, 0, 0)"},
                    fill="toself",
                    fillcolor=_hex_to_rgba(well.color, 0.14),
                    hoverinfo="skip",
                )
            )
            x_arrays.append(ribbon[:, 0])
            y_arrays.append(ribbon[:, 1])
            if (not focus_set and not bool(well.is_reference_only)) or (
                bool(focus_set) and str(well.name) in focus_set
            ):
                x_focus_arrays.append(ribbon[:, 0])
                y_focus_arrays.append(ribbon[:, 1])

        stations = well.stations
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=well_label,
                legendgroup=str(well.name),
                showlegend=not bool(well.is_reference_only),
                line={"width": 1.5, "color": str(well.color)},
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
                customdata=np.column_stack([[]]),
            )
        )
        previous_success = (previous_successes_by_name or {}).get(
            str(well.name)
        )
        if (
            previous_success is not None
            and not previous_success.stations.empty
        ):
            previous_stations = previous_success.stations
            previous_x = previous_stations["X_m"].to_numpy(dtype=float)
            previous_y = previous_stations["Y_m"].to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=previous_x,
                    y=previous_y,
                    mode="lines",
                    name=f"{well.name}: до anti-collision",
                    legendgroup=str(well.name),
                    showlegend=False,
                    line={
                        "width": 2,
                        "color": _hex_to_rgba(str(well.color), 0.55),
                        "dash": "dot",
                    },
                    customdata=np.column_stack(
                        [previous_stations["MD_m"].to_numpy(dtype=float)]
                    ),
                    hovertemplate=(
                        "X: %{x:.2f} m<br>"
                        "Y: %{y:.2f} m<br>"
                        "MD: %{customdata[0]:.2f} m"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            x_arrays.append(previous_x)
            y_arrays.append(previous_y)
            if (not focus_set) or str(well.name) in focus_set:
                x_focus_arrays.append(previous_x)
                y_focus_arrays.append(previous_y)
        if (
            (well.t1 is not None)
            and (well.t3 is not None)
            and not bool(well.is_reference_only)
        ):
            fig.add_trace(
                go.Scatter(
                    x=[well.surface.x, well.t1.x, well.t3.x],
                    y=[well.surface.y, well.t1.y, well.t3.y],
                    mode="markers",
                    name=f"{well_label}: цели",
                    legendgroup=str(well.name),
                    showlegend=False,
                    marker={
                        "size": 7,
                        "color": str(well.color),
                        "line": {"width": 1, "color": "rgba(255,255,255,0.9)"},
                    },
                    hovertemplate="X: %{x:.2f} m<br>Y: %{y:.2f} m<extra>%{fullData.name}</extra>",
                )
            )
            fig.add_trace(
                _t1_name_trace_2d(
                    well_name=str(well.name),
                    x_value=float(well.t1.x),
                    y_value=float(well.t1.y),
                    color=str(well.color),
                )
            )
            x_arrays.append(
                np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float)
            )
            y_arrays.append(
                np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float)
            )
            if (not focus_set) or str(well.name) in focus_set:
                x_focus_arrays.append(
                    np.array(
                        [well.surface.x, well.t1.x, well.t3.x], dtype=float
                    )
                )
                y_focus_arrays.append(
                    np.array(
                        [well.surface.y, well.t1.y, well.t3.y], dtype=float
                    )
                )

    overlap_legend_added = False
    for corridor in analysis.corridors:
        polygon = collision_corridor_plan_polygon(corridor)
        if len(polygon) < 3:
            continue
        fig.add_trace(
            go.Scatter(
                x=polygon[:, 0],
                y=polygon[:, 1],
                mode="lines",
                name="Общая зона overlap",
                legendgroup="collision_overlap",
                showlegend=not overlap_legend_added,
                line={"width": 0.0, "color": "rgba(0, 0, 0, 0)"},
                fill="toself",
                fillcolor="rgba(198, 40, 40, 0.42)",
                hoverinfo="skip",
            )
        )
        overlap_legend_added = True
        x_arrays.append(polygon[:, 0])
        y_arrays.append(polygon[:, 1])
        if (not focus_set) or bool(
            {str(corridor.well_a), str(corridor.well_b)}.intersection(
                focus_set
            )
        ):
            x_focus_arrays.append(polygon[:, 0])
            y_focus_arrays.append(polygon[:, 1])

    segment_legend_added = False
    for segment in analysis.well_segments:
        well = well_lookup.get(str(segment.well_name))
        if well is None:
            continue
        interval_points = _trajectory_interval_points(
            well.stations,
            md_start_m=float(segment.md_start_m),
            md_end_m=float(segment.md_end_m),
        )
        if interval_points is None:
            continue
        x_segment = interval_points["X_m"].to_numpy(dtype=float)
        y_segment = interval_points["Y_m"].to_numpy(dtype=float)
        md_segment = interval_points["MD_m"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x_segment,
                y=y_segment,
                mode="lines",
                name="Конфликтный участок ствола",
                legendgroup="collision_path",
                showlegend=not segment_legend_added,
                line={"width": 7, "color": "rgb(198, 40, 40)"},
                customdata=np.column_stack([md_segment]),
                hovertemplate=(
                    f"{segment.well_name}<br>"
                    "MD: %{customdata[0]:.2f} м"
                    "<extra>Конфликтный участок ствола</extra>"
                ),
            )
        )
        x_arrays.append(x_segment)
        y_arrays.append(y_segment)
        if (not focus_set) or str(segment.well_name) in focus_set:
            x_focus_arrays.append(x_segment)
            y_focus_arrays.append(y_segment)
        segment_legend_added = True
    for kind in _reference_kinds_present(
        [well for well in analysis.wells if bool(well.is_reference_only)]
    ):
        fig.add_trace(_reference_legend_trace_2d(kind))
    analysis_reference_wells = _analysis_reference_wells(analysis)
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label_trace = _reference_name_trace_2d(
            analysis_reference_wells,
            kind=kind,
        )
        if label_trace is not None:
            fig.add_trace(label_trace)
    pad_label_trace = _reference_pad_label_trace_2d(analysis_reference_wells)
    if pad_label_trace is not None:
        fig.add_trace(pad_label_trace)

    x_values = (
        (
            np.concatenate(x_focus_arrays)
            if x_focus_arrays
            else np.concatenate(x_arrays)
        )
        if x_arrays
        else np.array([0.0], dtype=float)
    )
    y_values = (
        (
            np.concatenate(y_focus_arrays)
            if y_focus_arrays
            else np.concatenate(y_arrays)
        )
        if y_arrays
        else np.array([0.0], dtype=float)
    )
    x_range, y_range = equalized_xy_ranges(
        x_values=x_values, y_values=y_values
    )
    xy_dtick = nice_tick_step(
        max(x_range[1] - x_range[0], y_range[1] - y_range[0]), target_ticks=6
    )
    x_tickvals = linear_tick_values(axis_range=x_range, step=xy_dtick)
    y_tickvals = linear_tick_values(axis_range=y_range, step=xy_dtick)
    fig.update_layout(
        title="Anti-collision: план E-N с конусами неопределенности",
        xaxis_title="X / Восток (м)",
        yaxis_title="Y / Север (м)",
        xaxis={
            "range": x_range,
            "tickmode": "array",
            "tickvals": x_tickvals,
            "tickformat": ".0f",
            "showexponent": "none",
            "exponentformat": "none",
        },
        yaxis={
            "range": y_range,
            "tickmode": "array",
            "tickvals": y_tickvals,
            "tickformat": ".0f",
            "showexponent": "none",
            "exponentformat": "none",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def _render_anticollision_panel(
    successes: list[SuccessfulWellPlan],
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str,
) -> None:
    panel_started_at = perf_counter()
    reference_wells = _reference_wells_from_state()
    if len(successes) + len(reference_wells) < 2:
        st.info(
            "Для anti-collision нужно минимум две успешно рассчитанные скважины."
        )
        return

    preset_options = list(UNCERTAINTY_PRESET_OPTIONS.keys())
    normalized_preset = normalize_uncertainty_preset(
        st.session_state.get(
            "wt_anticollision_uncertainty_preset",
            DEFAULT_UNCERTAINTY_PRESET,
        ),
    )
    if normalized_preset not in preset_options:
        normalized_preset = DEFAULT_UNCERTAINTY_PRESET
    st.session_state["wt_anticollision_uncertainty_preset"] = normalized_preset
    selected_preset = st.selectbox(
        "Пресет неопределенности для anti-collision",
        options=preset_options,
        format_func=uncertainty_preset_label,
        key="wt_anticollision_uncertainty_preset",
        help=(
            "Определяет уровень консерватизма planning-level конусов неопределенности "
            "для batch anti-collision анализа."
        ),
    )
    uncertainty_model = planning_uncertainty_model_for_preset(selected_preset)
    anti_collision_progress = st.progress(
        8,
        text="Подготовка anti-collision анализа...",
    )

    def _anti_collision_progress_update(value: int, text: str) -> None:
        anti_collision_progress.progress(int(value), text=text)

    try:
        analysis, recommendations, clusters = (
            _cached_anti_collision_view_model(
                successes=successes,
                uncertainty_model=uncertainty_model,
                records=records,
                reference_wells=reference_wells,
                progress_callback=_anti_collision_progress_update,
            )
        )
    except Exception as exc:
        anti_collision_progress.empty()
        _store_anticollision_failure_state(exc, started_at=panel_started_at)
        st.error(
            "Не удалось построить anti-collision анализ. Проверьте лог расчёта ниже."
        )
        _render_status_run_log(
            title="Лог расчёта Anti-collision",
            state_payload=st.session_state.get("wt_anticollision_last_run"),
            empty_message="Anti-collision анализ ещё не запускался.",
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return
    anti_collision_progress.empty()
    focus_pad_well_names = _focus_pad_well_names(
        records=records,
        focus_pad_id=focus_pad_id,
    )
    visible_clusters = _clusters_touching_focus_pad(
        clusters=clusters,
        focus_pad_well_names=focus_pad_well_names,
    )
    visible_recommendations = _recommendations_for_clusters(
        recommendations=recommendations,
        clusters=visible_clusters,
    )
    focus_anticollision_well_names = _anticollision_focus_well_names(
        clusters=visible_clusters,
        focus_pad_well_names=focus_pad_well_names,
    )
    previous_successes_by_name = {
        str(name): value
        for name, value in (
            st.session_state.get("wt_last_anticollision_previous_successes")
            or {}
        ).items()
    }
    st.caption(
        f"Пресет: {uncertainty_preset_label(selected_preset)}. "
        f"{anti_collision_method_caption(uncertainty_model)}"
    )
    _render_status_run_log(
        title="Лог расчёта Anti-collision",
        state_payload=st.session_state.get("wt_anticollision_last_run"),
        empty_message="Anti-collision анализ ещё не запускался.",
    )
    selected_render_mode = st.selectbox(
        "3D-режим отображения",
        options=list(WT_3D_RENDER_OPTIONS),
        key="wt_3d_render_mode",
    )
    selected_3d_backend = st.selectbox(
        "3D backend",
        options=list(WT_3D_BACKEND_OPTIONS),
        key="wt_3d_backend",
        help=(
            "Plotly сохраняет привычные hover-подсказки. "
            "Локальный Three.js backend быстрее на тяжёлых кустах и хранит все файлы локально."
        ),
    )
    if str(selected_3d_backend) == WT_3D_BACKEND_THREE_LOCAL:
        if st.button("Пересоздать 3D viewer", key="wt_recreate_three_ac"):
            _bump_three_viewer_nonce()
            st.rerun()
    resolved_render_mode = _resolve_3d_render_mode(
        requested_mode=selected_render_mode,
        calculated_well_count=len(successes),
        reference_wells=(
            well for well in analysis.wells if bool(well.is_reference_only)
        ),
    )
    if resolved_render_mode == WT_3D_RENDER_FAST:
        st.caption(
            "Включён быстрый 3D-режим: reference-скважины частично объединяются и "
            "часть их cone-геометрии скрывается из 3D-рендера, но anti-collision "
            "анализ и расчёт рекомендаций остаются прежними."
        )
        if reference_wells:
            st.caption(
                "В fast anti-collision 3D reference-конусы показываются только для "
                "reference-скважин, чьи XY-границы подходят к расчётным ближе чем на "
                f"{int(WT_3D_REFERENCE_CONE_FOCUS_DISTANCE_M)} м."
            )
    if str(selected_3d_backend) == WT_3D_BACKEND_THREE_LOCAL:
        st.caption(
            "Активен локальный Three.js viewer: 3D-смысл сцены сохраняется, "
            "но детальные Plotly-hover подсказки доступны только в режиме Plotly."
        )

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Проверено пар", f"{int(analysis.pair_count)}")
    m2.metric("Пар с overlap", f"{int(analysis.overlapping_pair_count)}")
    m3.metric(
        "Пересечения в t1/t3", f"{int(analysis.target_overlap_pair_count)}"
    )
    worst_sf = analysis.worst_separation_factor
    m4.metric(
        "Минимальный SF", "—" if worst_sf is None else f"{float(worst_sf):.2f}"
    )
    with st.expander("Что такое SF?", expanded=False):
        st.markdown(_sf_help_markdown())

    chart_col1, chart_col2 = st.columns(2, gap="medium")
    try:
        anticollision_3d_figure = _all_wells_anticollision_3d_figure(
            analysis,
            previous_successes_by_name=previous_successes_by_name,
            focus_well_names=focus_anticollision_well_names
            or focus_pad_well_names,
            render_mode=selected_render_mode,
        )
        _render_plotly_or_three_3d(
            container=chart_col1,
            figure=anticollision_3d_figure,
            backend=selected_3d_backend,
            height=660,
            payload_overrides=_anticollision_three_payload_overrides(
                records=records,
                analysis=analysis,
            ),
        )
        chart_col2.plotly_chart(
            _all_wells_anticollision_plan_figure(
                analysis,
                previous_successes_by_name=previous_successes_by_name,
                focus_well_names=focus_anticollision_well_names
                or focus_pad_well_names,
            ),
            width="stretch",
        )
    except Exception as exc:
        _store_anticollision_failure_state(exc, started_at=panel_started_at)
        st.error(
            "Не удалось отрисовать anti-collision визуализацию. Проверьте лог расчёта ниже."
        )
        _render_status_run_log(
            title="Лог расчёта Anti-collision",
            state_payload=st.session_state.get("wt_anticollision_last_run"),
            empty_message="Anti-collision анализ ещё не запускался.",
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return
    _render_last_anticollision_resolution(current_preset=selected_preset)

    if not analysis.zones:
        st.success(
            "Пересечения 2σ конусов неопределенности не обнаружены для рассчитанного набора."
        )
        return

    target_zones = [
        zone for zone in analysis.zones if int(zone.priority_rank) < 2
    ]
    if target_zones:
        st.warning(
            "Найдены пересечения, затрагивающие точки целей t1/t3. Они вынесены "
            "в начало отчета и должны разбираться в первую очередь."
        )
    else:
        st.warning(
            "Найдены пересечения 2σ конусов неопределенности по траекториям."
        )

    if focus_pad_well_names:
        st.info(
            "Показаны только anti-collision события и кластеры, которые затрагивают "
            f"выбранный куст ({', '.join(focus_pad_well_names)}). Если в такой кластер "
            "входят рассчитываемые скважины других кустов, они будут автоматически "
            "учтены и при необходимости попадут в пересчет."
        )
    report_rows = (
        _report_rows_from_recommendations(visible_recommendations)
        if focus_pad_well_names
        else anti_collision_report_rows(analysis)
    )
    report_event_count = len(report_rows)
    report_df = arrow_safe_text_dataframe(pd.DataFrame(report_rows))
    st.markdown("### Отчет по anti-collision")
    st.caption(
        "Смежные и пересекающиеся corridor-интервалы одной и той же collision природы "
        f"в отчете объединяются в одно событие. Всего событий: {int(report_event_count)}."
    )
    st.dataframe(
        report_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Приоритет": st.column_config.TextColumn("Приоритет"),
            "Скважина A": st.column_config.TextColumn("Скважина A"),
            "Скважина B": st.column_config.TextColumn("Скважина B"),
            "Область": st.column_config.TextColumn("Область"),
            "Интервал A, м": st.column_config.TextColumn("Интервал A, м"),
            "Интервал B, м": st.column_config.TextColumn("Интервал B, м"),
            "SF min": st.column_config.NumberColumn("SF min", format="%.2f"),
            "Overlap max, м": st.column_config.NumberColumn(
                "Overlap max, м",
                format="%.2f",
            ),
            "Мин. расстояние, м": st.column_config.NumberColumn(
                "Мин. расстояние, м", format="%.2f"
            ),
        },
    )

    recommendation_df = arrow_safe_text_dataframe(
        pd.DataFrame(
            anti_collision_recommendation_rows(visible_recommendations)
        )
    )
    st.markdown("### Рекомендации")
    st.caption(
        "Рекомендации делят конфликты на три класса: цели, vertical-участки и "
        "прочие траекторные пересечения. Автоподготовка пересчета сейчас включена "
        "для vertical-кейсов и для trajectory-collision rerun обеих проектных "
        "скважин пары с учетом остальных reference-cones куста."
    )
    st.dataframe(
        recommendation_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Приоритет": st.column_config.TextColumn("Приоритет"),
            "Скважина A": st.column_config.TextColumn("Скважина A"),
            "Скважина B": st.column_config.TextColumn("Скважина B"),
            "Тип действия": st.column_config.TextColumn("Тип действия"),
            "Область": st.column_config.TextColumn("Область"),
            "Интервал A, м": st.column_config.TextColumn("Интервал A, м"),
            "Интервал B, м": st.column_config.TextColumn("Интервал B, м"),
            "SF min": st.column_config.NumberColumn("SF min", format="%.2f"),
            "Overlap max, м": st.column_config.NumberColumn(
                "Overlap max, м", format="%.2f"
            ),
            "Spacing t1, м": st.column_config.TextColumn("Spacing t1, м"),
            "Spacing t3, м": st.column_config.TextColumn("Spacing t3, м"),
            "Ожидаемый маневр": st.column_config.TextColumn(
                "Ожидаемый маневр"
            ),
            "Рекомендация": st.column_config.TextColumn("Рекомендация"),
            "Подготовка пересчета": st.column_config.TextColumn(
                "Подготовка пересчета"
            ),
        },
    )

    cluster_df = arrow_safe_text_dataframe(
        pd.DataFrame(anti_collision_cluster_rows(visible_clusters))
    )
    st.markdown("### Cluster-level пересчет")
    st.caption(
        "Кластер объединяет связанные anti-collision события по connected component "
        "графа скважин. Подготовка cluster-level пересчета агрегирует pairwise "
        "рекомендации в единый per-well rerun plan с multi-reference конфликтным окном. "
        "Поля 'Стартовый шаг' и 'Порядок' показывают рекомендуемую очередность маневров."
    )
    st.dataframe(
        cluster_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Кластер": st.column_config.TextColumn("Кластер"),
            "Скважины": st.column_config.TextColumn("Скважины"),
            "Событий": st.column_config.NumberColumn("Событий", format="%d"),
            "Цели": st.column_config.NumberColumn("Цели", format="%d"),
            "VERTICAL": st.column_config.NumberColumn("VERTICAL", format="%d"),
            "Траектория": st.column_config.NumberColumn(
                "Траектория", format="%d"
            ),
            "SF min": st.column_config.NumberColumn("SF min", format="%.2f"),
            "Ожидаемый маневр": st.column_config.TextColumn(
                "Ожидаемый маневр"
            ),
            "Стартовый шаг": st.column_config.TextColumn("Стартовый шаг"),
            "Порядок": st.column_config.TextColumn("Порядок"),
            "К пересчету": st.column_config.TextColumn("К пересчету"),
            "Подготовка пересчета": st.column_config.TextColumn(
                "Подготовка пересчета"
            ),
        },
    )

    st.info(
        "Как использовать подготовку пересчета: "
        "`Подготовить одно событие` применяйте для локального отдельного конфликта. "
        "`Подготовить весь кластер` применяйте, когда одна и та же скважина участвует "
        "в нескольких связанных конфликтах. В каждый момент активен только один "
        "подготовленный план: новая подготовка заменяет предыдущую."
    )

    actionable_clusters = [
        item for item in visible_clusters if bool(item.can_prepare_rerun)
    ]
    if actionable_clusters:
        cluster_ids = [item.cluster_id for item in actionable_clusters]
        if (
            str(
                st.session_state.get(
                    "wt_anticollision_prepared_cluster_id", ""
                )
            )
            not in cluster_ids
        ):
            st.session_state["wt_anticollision_prepared_cluster_id"] = (
                cluster_ids[0]
            )
        cluster_select_col, cluster_button_col = st.columns(
            [6.0, 1.8], gap="small"
        )
        with cluster_select_col:
            selected_cluster_id = st.selectbox(
                "Подготовить пересчет для всего связанного кластера",
                options=cluster_ids,
                format_func=lambda value: cluster_display_label(
                    next(
                        item
                        for item in actionable_clusters
                        if str(item.cluster_id) == str(value)
                    )
                ),
                key="wt_anticollision_prepared_cluster_id",
            )
        with cluster_button_col:
            if st.button(
                "Подготовить весь кластер",
                type="primary",
                icon=":material/hub:",
                width="stretch",
            ):
                selected_cluster = next(
                    item
                    for item in actionable_clusters
                    if str(item.cluster_id) == str(selected_cluster_id)
                )
                with st.spinner("Подготовка cluster-level плана пересчета..."):
                    _prepare_rerun_from_cluster(
                        selected_cluster,
                        successes=successes,
                        uncertainty_model=uncertainty_model,
                        target_well_names=_pad_scoped_cluster_target_well_names(
                            cluster=selected_cluster,
                            focus_pad_well_names=focus_pad_well_names,
                        ),
                        focus_well_names=_pad_scoped_cluster_focus_well_names(
                            cluster=selected_cluster,
                            focus_pad_well_names=focus_pad_well_names,
                        ),
                    )
                st.toast(
                    "Подготовлен план пересчета для всего связанного кластера. "
                    "Он заменил предыдущий prepared plan."
                )
                st.rerun()

    actionable_recommendations = [
        item
        for item in visible_recommendations
        if bool(item.can_prepare_rerun)
    ]
    if actionable_recommendations:
        actionable_ids = [
            item.recommendation_id for item in actionable_recommendations
        ]
        if (
            str(
                st.session_state.get(
                    "wt_anticollision_prepared_recommendation_id", ""
                )
            )
            not in actionable_ids
        ):
            st.session_state["wt_anticollision_prepared_recommendation_id"] = (
                actionable_ids[0]
            )
        action_col, button_col = st.columns([6.0, 1.8], gap="small")
        with action_col:
            selected_recommendation_id = st.selectbox(
                "Подготовить пересчет по одному anti-collision событию",
                options=actionable_ids,
                format_func=lambda value: recommendation_display_label(
                    next(
                        item
                        for item in actionable_recommendations
                        if str(item.recommendation_id) == str(value)
                    )
                ),
                key="wt_anticollision_prepared_recommendation_id",
            )
        with button_col:
            if st.button(
                "Подготовить одно событие",
                type="primary",
                icon=":material/build:",
                width="stretch",
            ):
                selected_recommendation = next(
                    item
                    for item in actionable_recommendations
                    if str(item.recommendation_id)
                    == str(selected_recommendation_id)
                )
                with st.spinner("Подготовка pairwise плана пересчета..."):
                    _prepare_rerun_from_recommendation(
                        selected_recommendation,
                        successes=successes,
                        uncertainty_model=uncertainty_model,
                    )
                st.toast(
                    "Подготовлен план пересчета по одному событию. "
                    "Он заменил предыдущий prepared plan."
                )
                st.rerun()
    else:
        st.caption(
            "Для текущего набора доступны только advisory-рекомендации: target spacing "
            "или ожидание dedicated anti-collision optimization mode."
        )


def _build_config_form(
    binding: CalcParamBinding = WT_CALC_PARAMS,
    *,
    title: str = "Параметры расчета",
) -> TrajectoryConfig:
    binding.render_block(title=title)
    return binding.build_config()


def _prepared_override_rows() -> list[dict[str, object]]:
    prepared = st.session_state.get("wt_prepared_well_overrides", {}) or {}
    recommendation_snapshot = dict(
        st.session_state.get("wt_prepared_recommendation_snapshot") or {}
    )
    order_by_well: dict[str, int] = {}
    maneuver_by_well: dict[str, str] = {}
    step_rows = tuple(recommendation_snapshot.get("action_steps", ()) or ())
    for raw_step in step_rows:
        step = dict(raw_step)
        well_name = str(step.get("well_name", "")).strip()
        if not well_name:
            continue
        order_by_well[well_name] = int(step.get("order_rank", 0) or 0)
        maneuver_by_well[well_name] = (
            str(step.get("expected_maneuver", "—")).strip() or "—"
        )
    if not order_by_well:
        affected_wells = tuple(
            str(name)
            for name in recommendation_snapshot.get("affected_wells", ()) or ()
        )
        default_maneuver = (
            str(recommendation_snapshot.get("expected_maneuver", "—")).strip()
            or "—"
        )
        for index, well_name in enumerate(affected_wells, start=1):
            order_by_well[well_name] = int(index)
            maneuver_by_well[well_name] = default_maneuver
    rows: list[dict[str, object]] = []
    ordered_names = sorted(
        (str(name) for name in prepared.keys()),
        key=lambda name: (
            int(order_by_well.get(name, 10_000)),
            str(name),
        ),
    )
    for well_name in ordered_names:
        payload = dict(prepared.get(well_name, {}))
        update_fields = dict(payload.get("update_fields", {}))
        optimization_mode = str(
            update_fields.get("optimization_mode", "")
        ).strip()
        optimization_label = optimization_display_label(optimization_mode)
        rows.append(
            {
                "Порядок": (
                    "—"
                    if int(order_by_well.get(well_name, 0)) <= 0
                    else str(int(order_by_well[well_name]))
                ),
                "Скважина": str(well_name),
                "Маневр": str(maneuver_by_well.get(well_name, "—")).strip()
                or "—",
                "Оптимизация": optimization_label,
                "SF до": _format_sf_value(
                    recommendation_snapshot.get("before_sf")
                ),
                "Источник": str(payload.get("source", "—")).strip() or "—",
                "Причина": str(payload.get("reason", "—")).strip() or "—",
            }
        )
    return rows


def _prepared_plan_kind_label(snapshot: Mapping[str, object] | None) -> str:
    kind = str((snapshot or {}).get("kind", "")).strip()
    if kind == "cluster":
        return "весь связанный anti-collision кластер"
    if kind == "recommendation":
        return "одно anti-collision событие"
    return "локальный anti-collision план"


def _format_prepared_override_scope(
    *,
    selected_names: Iterable[str],
) -> list[dict[str, object]]:
    prepared = st.session_state.get("wt_prepared_well_overrides", {}) or {}
    rows: list[dict[str, object]] = []
    for well_name in list(dict.fromkeys(str(name) for name in selected_names)):
        payload = dict(prepared.get(well_name, {}))
        update_fields = dict(payload.get("update_fields", {}))
        if not update_fields:
            continue
        optimization_mode = str(
            update_fields.get("optimization_mode", "")
        ).strip()
        rows.append(
            {
                "Скважина": str(well_name),
                "Локальный режим": optimization_display_label(
                    optimization_mode
                ),
                "Источник": str(payload.get("source", "—")).strip() or "—",
                "Маневр": "—",
            }
        )
    maneuver_by_well = {
        str(row.get("Скважина", ""))
        .strip(): str(row.get("Маневр", "—"))
        .strip()
        or "—"
        for row in _prepared_override_rows()
    }
    for row in rows:
        row["Маневр"] = maneuver_by_well.get(str(row["Скважина"]), "—")
    return rows


def _welltrack_record_entry_tvd_m(record: WelltrackRecord) -> float | None:
    try:
        _, t1, _ = welltrack_points_to_targets(record.points)
    except ValueError:
        return None
    return float(t1.z)


def _evaluated_kop_min_vertical_for_record(
    *,
    record: WelltrackRecord,
    base_config: TrajectoryConfig,
    kop_function: ActualFundKopDepthFunction,
) -> float | None:
    entry_tvd_m = _welltrack_record_entry_tvd_m(record)
    if entry_tvd_m is None:
        return None
    raw_kop_m = float(kop_function.evaluate(entry_tvd_m))
    max_physical_kop_m = max(
        0.0,
        float(entry_tvd_m)
        - max(
            float(base_config.min_structural_segment_m),
            float(base_config.vertical_tolerance_m),
        ),
    )
    return float(min(max(raw_kop_m, 0.0), max_physical_kop_m))


def _build_selected_override_configs(
    *,
    base_config: TrajectoryConfig,
    selected_names: set[str],
    records_by_name: Mapping[str, WelltrackRecord] | None = None,
) -> dict[str, TrajectoryConfig]:
    prepared = st.session_state.get("wt_prepared_well_overrides", {}) or {}
    kop_function = kop_min_vertical_function_from_state(
        prefix=WT_CALC_PARAMS.prefix
    )
    config_map: dict[str, TrajectoryConfig] = {}
    for well_name in sorted(str(name) for name in selected_names):
        payload = dict(prepared.get(well_name, {}))
        update_fields = dict(payload.get("update_fields", {}))
        config = (
            base_config.validated_copy(**update_fields)
            if update_fields
            else base_config
        )
        if kop_function is not None and records_by_name is not None:
            record = records_by_name.get(well_name)
            if record is not None:
                evaluated_kop_m = _evaluated_kop_min_vertical_for_record(
                    record=record,
                    base_config=config,
                    kop_function=kop_function,
                )
                if evaluated_kop_m is not None:
                    config = config.validated_copy(
                        kop_min_vertical_m=evaluated_kop_m
                    )
        if config == base_config and not update_fields:
            continue
        config_map[well_name] = config
    return config_map


def _build_selected_optimization_contexts(
    *,
    selected_names: set[str],
    current_successes: list[SuccessfulWellPlan] | None = None,
) -> dict[str, AntiCollisionOptimizationContext]:
    prepared = st.session_state.get("wt_prepared_well_overrides", {}) or {}
    success_by_name = {
        str(item.name): item for item in (current_successes or [])
    }
    context_map: dict[str, AntiCollisionOptimizationContext] = {}
    updated_prepared: dict[str, dict[str, object]] | None = None
    for well_name in sorted(str(name) for name in selected_names):
        payload = dict(prepared.get(well_name, {}))
        optimization_context = payload.get("optimization_context")
        if isinstance(optimization_context, AntiCollisionOptimizationContext):
            rebuilt_context = rebuild_optimization_context(
                context=optimization_context,
                reference_success_by_name=success_by_name,
                strict_missing_references=True,
            )
            if rebuilt_context is None:
                continue
            context_map[well_name] = rebuilt_context
            if rebuilt_context is not optimization_context:
                if updated_prepared is None:
                    updated_prepared = {
                        str(name): dict(value)
                        for name, value in prepared.items()
                    }
                updated_payload = dict(updated_prepared.get(well_name, {}))
                updated_payload["optimization_context"] = rebuilt_context
                updated_prepared[well_name] = updated_payload
    if updated_prepared is not None:
        st.session_state["wt_prepared_well_overrides"] = updated_prepared
    return context_map


def _selected_execution_order(selected_names: list[str]) -> list[str]:
    ordered_selected = list(
        dict.fromkeys(str(name) for name in selected_names)
    )
    snapshot = dict(
        st.session_state.get("wt_prepared_recommendation_snapshot") or {}
    )
    selected_set = set(ordered_selected)
    action_steps = tuple(snapshot.get("action_steps", ()) or ())
    prioritized = [
        str(dict(step).get("well_name", "")).strip()
        for step in action_steps
        if str(dict(step).get("well_name", "")).strip() in selected_set
    ]
    prioritized = list(dict.fromkeys(name for name in prioritized if name))
    remainder = [
        name for name in ordered_selected if name not in set(prioritized)
    ]
    return [*prioritized, *remainder]


def _build_anticollision_well_contexts(
    successes: list[SuccessfulWellPlan],
) -> dict[str, AntiCollisionWellContext]:
    return build_anticollision_well_contexts_shared(successes)


def _recommendation_snapshot(
    recommendation: AntiCollisionRecommendation,
) -> dict[str, object]:
    return {
        "kind": "recommendation",
        "recommendation_id": str(recommendation.recommendation_id),
        "well_a": str(recommendation.well_a),
        "well_b": str(recommendation.well_b),
        "classification": str(recommendation.classification),
        "category": str(recommendation.category),
        "area_label": str(recommendation.area_label),
        "summary": str(recommendation.summary),
        "detail": str(recommendation.detail),
        "expected_maneuver": str(recommendation.expected_maneuver),
        "before_sf": float(recommendation.min_separation_factor),
        "before_overlap_m": float(recommendation.max_overlap_depth_m),
        "md_a_start_m": float(recommendation.md_a_start_m),
        "md_a_end_m": float(recommendation.md_a_end_m),
        "md_b_start_m": float(recommendation.md_b_start_m),
        "md_b_end_m": float(recommendation.md_b_end_m),
        "affected_wells": tuple(
            str(name) for name in recommendation.affected_wells
        ),
        "action_label": str(recommendation.action_label),
        "source_label": recommendation_display_label(recommendation),
    }


def _cluster_snapshot(
    cluster: AntiCollisionRecommendationCluster,
    *,
    target_well_names: tuple[str, ...] = (),
    focus_well_names: tuple[str, ...] = (),
) -> dict[str, object]:
    items = tuple(
        _recommendation_snapshot(item) for item in cluster.recommendations
    )
    actionable_before_sf = [
        float(item.min_separation_factor)
        for item in cluster.recommendations
        if bool(item.can_prepare_rerun)
    ]
    before_sf = (
        min(actionable_before_sf)
        if actionable_before_sf
        else float(cluster.worst_separation_factor)
    )
    return {
        "kind": "cluster",
        "cluster_id": str(cluster.cluster_id),
        "source_label": cluster_display_label(cluster),
        "summary": str(cluster.summary),
        "detail": str(cluster.detail),
        "expected_maneuver": str(cluster.expected_maneuver),
        "blocking_advisory": (
            None
            if cluster.blocking_advisory is None
            else str(cluster.blocking_advisory)
        ),
        "affected_wells": tuple(str(name) for name in cluster.affected_wells),
        "well_names": tuple(str(name) for name in cluster.well_names),
        "target_well_names": tuple(str(name) for name in target_well_names),
        "focus_well_names": tuple(str(name) for name in focus_well_names),
        "recommendation_count": int(cluster.recommendation_count),
        "before_sf": float(before_sf),
        "rerun_order_label": str(cluster.rerun_order_label),
        "first_rerun_well": (
            None
            if cluster.first_rerun_well is None
            else str(cluster.first_rerun_well)
        ),
        "first_rerun_maneuver": (
            None
            if cluster.first_rerun_maneuver is None
            else str(cluster.first_rerun_maneuver)
        ),
        "action_steps": tuple(
            {
                "order_rank": int(step.order_rank),
                "well_name": str(step.well_name),
                "category": str(step.category),
                "optimization_mode": str(step.optimization_mode),
                "expected_maneuver": str(step.expected_maneuver),
                "reason": str(step.reason),
                "related_recommendation_count": int(
                    step.related_recommendation_count
                ),
                "worst_separation_factor": float(step.worst_separation_factor),
            }
            for step in cluster.action_steps
        ),
        "items": items,
    }


def _format_sf_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.2f}"


def _format_overlap_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.2f}"


def _md_interval_label(md_start_m: float, md_end_m: float) -> str:
    start = float(md_start_m)
    end = float(md_end_m)
    if abs(end - start) <= 1e-6:
        return f"{start:.0f}"
    return f"{start:.0f}-{end:.0f}"


def _sf_help_markdown() -> str:
    return (
        "**Что такое SF**\n\n"
        "SF (`Separation Factor`) показывает запас расстояния между двумя "
        "скважинами с учетом суммарной неопределенности их конусов.\n\n"
        "- `SF < 1` — конусы неопределенности overlap, это collision-risk.\n"
        "- `SF ≈ 1` — граничное состояние, запас почти исчерпан.\n"
        "- `SF > 1` — есть запас по разнесению; чем больше число, тем комфортнее ситуация.\n\n"
        "В текущем WELLTRACK это planning-level индикатор для сравнения вариантов, "
        "а не абсолютная гарантия безопасности."
    )


def _evaluate_pair_interval_clearance(
    *,
    success_a: SuccessfulWellPlan,
    success_b: SuccessfulWellPlan,
    md_a_start_m: float,
    md_a_end_m: float,
    md_b_start_m: float,
    md_b_end_m: float,
    uncertainty_model: PlanningUncertaintyModel,
    sample_step_m: float = 50.0,
) -> tuple[float, float] | None:
    try:
        context_a = AntiCollisionOptimizationContext(
            candidate_md_start_m=float(md_a_start_m),
            candidate_md_end_m=float(md_a_end_m),
            sf_target=1.0,
            sample_step_m=float(sample_step_m),
            uncertainty_model=uncertainty_model,
            references=(
                build_anti_collision_reference_path(
                    well_name=str(success_b.name),
                    stations=success_b.stations,
                    md_start_m=float(md_b_start_m),
                    md_end_m=float(md_b_end_m),
                    sample_step_m=float(sample_step_m),
                    model=uncertainty_model,
                ),
            ),
        )
        evaluation_a = evaluate_stations_anti_collision_clearance(
            stations=success_a.stations,
            context=context_a,
        )
        context_b = AntiCollisionOptimizationContext(
            candidate_md_start_m=float(md_b_start_m),
            candidate_md_end_m=float(md_b_end_m),
            sf_target=1.0,
            sample_step_m=float(sample_step_m),
            uncertainty_model=uncertainty_model,
            references=(
                build_anti_collision_reference_path(
                    well_name=str(success_a.name),
                    stations=success_a.stations,
                    md_start_m=float(md_a_start_m),
                    md_end_m=float(md_a_end_m),
                    sample_step_m=float(sample_step_m),
                    model=uncertainty_model,
                ),
            ),
        )
        evaluation_b = evaluate_stations_anti_collision_clearance(
            stations=success_b.stations,
            context=context_b,
        )
    except ValueError:
        return None
    return (
        float(
            min(
                evaluation_a.min_separation_factor,
                evaluation_b.min_separation_factor,
            )
        ),
        float(
            max(
                evaluation_a.max_overlap_depth_m,
                evaluation_b.max_overlap_depth_m,
            )
        ),
    )


def _anticollision_resolution_status(
    *,
    before_sf: float,
    after_sf: float,
    after_overlap_m: float,
) -> str:
    if after_sf >= 1.0 - 1e-6 and after_overlap_m <= 1e-6:
        return "Конфликт снят"
    if after_sf > before_sf + 1e-3:
        return "SF улучшен"
    if after_sf < before_sf - 1e-3:
        return "SF ухудшен"
    return "Без заметных изменений"


def _build_pair_anticollision_resolution_item(
    *,
    snapshot: dict[str, object],
    success_by_name: dict[str, SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> dict[str, object] | None:
    success_a = success_by_name.get(str(snapshot.get("well_a", "")))
    success_b = success_by_name.get(str(snapshot.get("well_b", "")))
    if success_a is None or success_b is None:
        return None
    clearance = _evaluate_pair_interval_clearance(
        success_a=success_a,
        success_b=success_b,
        md_a_start_m=float(snapshot.get("md_a_start_m", 0.0)),
        md_a_end_m=float(snapshot.get("md_a_end_m", 0.0)),
        md_b_start_m=float(snapshot.get("md_b_start_m", 0.0)),
        md_b_end_m=float(snapshot.get("md_b_end_m", 0.0)),
        uncertainty_model=uncertainty_model,
    )
    if clearance is None:
        return None
    after_sf, after_overlap_m = clearance
    before_sf = float(snapshot.get("before_sf", 0.0))
    item = dict(snapshot)
    item.update(
        {
            "after_sf": float(after_sf),
            "after_overlap_m": float(after_overlap_m),
            "delta_sf": float(after_sf - before_sf),
            "status": _anticollision_resolution_status(
                before_sf=before_sf,
                after_sf=float(after_sf),
                after_overlap_m=float(after_overlap_m),
            ),
        }
    )
    return item


def _build_last_anticollision_resolution(
    *,
    snapshot: dict[str, object] | None,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    uncertainty_preset: str,
) -> dict[str, object] | None:
    data = dict(snapshot or {})
    if not data:
        return None
    success_by_name = {str(item.name): item for item in successes}
    kind = str(data.get("kind", "recommendation")).strip() or "recommendation"
    if kind == "cluster":
        before_sf = float(data.get("before_sf", 0.0))
        before_overlap_m = (
            max(
                float(dict(item).get("before_overlap_m", 0.0))
                for item in tuple(data.get("items", ()) or ())
            )
            if tuple(data.get("items", ()) or ())
            else 0.0
        )
        target_wells = _resolution_snapshot_well_names(data)
        current_clusters = _clusters_touching_resolution_snapshot(
            target_wells=target_wells,
            successes=successes,
            uncertainty_model=uncertainty_model,
        )
        if current_clusters:
            after_sf = min(
                float(item.worst_separation_factor)
                for item in current_clusters
            )
            after_overlap_m = max(
                float(recommendation.max_overlap_depth_m)
                for cluster in current_clusters
                for recommendation in cluster.recommendations
            )
            active_items = tuple(
                {
                    "cluster_id": str(cluster.cluster_id),
                    "cluster_label": cluster_display_label(cluster),
                    "well_a": str(recommendation.well_a),
                    "well_b": str(recommendation.well_b),
                    "action_label": str(recommendation.action_label),
                    "expected_maneuver": str(recommendation.expected_maneuver),
                    "area_label": str(recommendation.area_label),
                    "md_a_start_m": float(recommendation.md_a_start_m),
                    "md_a_end_m": float(recommendation.md_a_end_m),
                    "md_b_start_m": float(recommendation.md_b_start_m),
                    "md_b_end_m": float(recommendation.md_b_end_m),
                    "current_sf": float(recommendation.min_separation_factor),
                    "current_overlap_m": float(
                        recommendation.max_overlap_depth_m
                    ),
                }
                for cluster in current_clusters
                for recommendation in cluster.recommendations
            )
        else:
            after_sf = 1.0
            after_overlap_m = 0.0
            active_items = ()
        resolved = dict(data)
        resolved.update(
            {
                "before_sf": float(before_sf),
                "after_sf": float(after_sf),
                "before_overlap_m": float(before_overlap_m),
                "after_overlap_m": float(after_overlap_m),
                "delta_sf": float(after_sf - before_sf),
                "status": _anticollision_resolution_status(
                    before_sf=float(before_sf),
                    after_sf=float(after_sf),
                    after_overlap_m=float(after_overlap_m),
                ),
                "current_cluster_count": int(len(current_clusters)),
                "current_cluster_labels": tuple(
                    cluster_display_label(cluster)
                    for cluster in current_clusters
                ),
                "items": active_items,
                "uncertainty_preset": str(uncertainty_preset),
            }
        )
        return resolved

    resolved_item = _build_pair_anticollision_resolution_item(
        snapshot=data,
        success_by_name=success_by_name,
        uncertainty_model=uncertainty_model,
    )
    if resolved_item is None:
        return None
    resolved_item["uncertainty_preset"] = str(uncertainty_preset)
    return resolved_item


def _resolution_snapshot_well_names(
    snapshot: dict[str, object],
) -> tuple[str, ...]:
    target_wells = tuple(
        str(name) for name in snapshot.get("target_well_names", ()) or ()
    )
    if target_wells:
        return target_wells
    explicit_wells = tuple(
        str(name) for name in snapshot.get("well_names", ()) or ()
    )
    if explicit_wells:
        return explicit_wells
    affected_wells = tuple(
        str(name) for name in snapshot.get("affected_wells", ()) or ()
    )
    if affected_wells:
        return affected_wells
    derived: list[str] = []
    for raw_item in tuple(snapshot.get("items", ()) or ()):
        item = dict(raw_item)
        for key in ("well_a", "well_b"):
            well_name = str(item.get(key, "")).strip()
            if well_name and well_name not in derived:
                derived.append(well_name)
    return tuple(derived)


def _clusters_touching_resolution_snapshot(
    *,
    target_wells: tuple[str, ...],
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> tuple[AntiCollisionRecommendationCluster, ...]:
    target_set = {str(name) for name in target_wells if str(name).strip()}
    reference_wells = _reference_wells_from_state()
    if not target_set or (len(successes) + len(reference_wells) < 2):
        return ()
    analysis = build_anti_collision_analysis_for_successes_shared(
        successes,
        model=uncertainty_model,
        reference_wells=reference_wells,
        include_display_geometry=False,
        build_overlap_geometry=False,
    )
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=_build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)
    relevant = [
        cluster
        for cluster in clusters
        if target_set.intersection(str(name) for name in cluster.well_names)
    ]
    return tuple(relevant)


def _render_last_anticollision_resolution(*, current_preset: str) -> None:
    resolution = dict(
        st.session_state.get("wt_last_anticollision_resolution") or {}
    )
    if not resolution:
        return
    resolution_kind = (
        str(resolution.get("kind", "recommendation")).strip()
        or "recommendation"
    )
    st.markdown("### Результат последнего anti-collision пересчета")
    if resolution_kind == "cluster":
        caption = (
            "После пересчета выполнен полный повторный anti-collision scan по текущему "
            "набору скважин. Метрика 'SF после' и статус считаются по всем актуальным "
            "cluster-level событиям, которые сейчас затрагивают исходные скважины "
            "подготовленного плана."
        )
    else:
        caption = (
            "Сравнение выполнено на исходном конфликтном окне по тем же pairwise "
            "интервалам и текущему planning-level пресету неопределенности."
        )
    used_preset = str(resolution.get("uncertainty_preset", "")).strip()
    if used_preset and used_preset != str(current_preset):
        caption += (
            f" Последний пересчет считался на пресете "
            f"'{uncertainty_preset_label(used_preset)}', сейчас выбран "
            f"'{uncertainty_preset_label(current_preset)}'."
        )
    st.caption(caption)
    if resolution_kind == "cluster":
        m1, m2, m3, m4 = st.columns(4, gap="small")
        m1.metric("Кластер", str(resolution.get("source_label", "—")))
        m2.metric("SF до", _format_sf_value(resolution.get("before_sf")))
        m3.metric(
            "SF после",
            _format_sf_value(resolution.get("after_sf")),
            delta=(
                None
                if resolution.get("delta_sf") is None
                else f"{float(resolution['delta_sf']):+.2f}"
            ),
        )
        m4.metric("Статус", str(resolution.get("status", "—")))
        current_cluster_labels = tuple(
            resolution.get("current_cluster_labels", ()) or ()
        )
        if current_cluster_labels:
            st.caption(
                "Актуальные кластеры после пересчета: "
                + "; ".join(str(label) for label in current_cluster_labels)
            )
        else:
            st.caption(
                "После полного повторного scan активных anti-collision кластеров "
                "для исходных скважин не осталось."
            )
        items = list(resolution.get("items", ()) or ())
        if not items:
            st.success(
                "Повторный cluster-level scan не обнаружил оставшихся collision-событий "
                "для затронутых скважин."
            )
            return
        st.dataframe(
            arrow_safe_text_dataframe(
                pd.DataFrame(
                    [
                        {
                            "Кластер после": str(item.get("cluster_id", "—")),
                            "Пара": f"{item.get('well_a', '—')} ↔ {item.get('well_b', '—')}",
                            "Тип": str(item.get("action_label", "—")),
                            "Ожидаемый маневр": str(
                                item.get("expected_maneuver", "—")
                            ),
                            "Область": str(item.get("area_label", "—")),
                            "Интервал A, м": _md_interval_label(
                                float(item.get("md_a_start_m", 0.0)),
                                float(item.get("md_a_end_m", 0.0)),
                            ),
                            "Интервал B, м": _md_interval_label(
                                float(item.get("md_b_start_m", 0.0)),
                                float(item.get("md_b_end_m", 0.0)),
                            ),
                            "SF сейчас": _format_sf_value(
                                item.get("current_sf")
                            ),
                            "Overlap сейчас, м": _format_overlap_value(
                                item.get("current_overlap_m")
                            ),
                        }
                        for item in items
                    ]
                )
            ),
            width="stretch",
            hide_index=True,
        )
        return

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric(
        "Пара",
        f"{resolution.get('well_a', '—')} ↔ {resolution.get('well_b', '—')}",
    )
    m2.metric("SF до", _format_sf_value(resolution.get("before_sf")))
    m3.metric(
        "SF после",
        _format_sf_value(resolution.get("after_sf")),
        delta=(
            None
            if resolution.get("delta_sf") is None
            else f"{float(resolution['delta_sf']):+.2f}"
        ),
    )
    m4.metric("Статус", str(resolution.get("status", "—")))
    st.dataframe(
        arrow_safe_text_dataframe(
            pd.DataFrame(
                [
                    {
                        "Источник": str(resolution.get("source_label", "—")),
                        "Тип": str(resolution.get("action_label", "—")),
                        "Область": str(resolution.get("area_label", "—")),
                        "Интервал A, м": _md_interval_label(
                            float(resolution.get("md_a_start_m", 0.0)),
                            float(resolution.get("md_a_end_m", 0.0)),
                        ),
                        "Интервал B, м": _md_interval_label(
                            float(resolution.get("md_b_start_m", 0.0)),
                            float(resolution.get("md_b_end_m", 0.0)),
                        ),
                        "Overlap до, м": _format_overlap_value(
                            resolution.get("before_overlap_m")
                        ),
                        "Overlap после, м": _format_overlap_value(
                            resolution.get("after_overlap_m")
                        ),
                    }
                ]
            )
        ),
        width="stretch",
        hide_index=True,
    )


def _prepare_rerun_from_recommendation(
    recommendation: AntiCollisionRecommendation,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> None:
    prepared, skipped_wells, action_steps = (
        _build_recommendation_prepared_overrides(
            recommendation,
            successes=successes,
            uncertainty_model=uncertainty_model,
        )
    )
    snapshot = dict(_recommendation_snapshot(recommendation))
    if action_steps:
        snapshot["action_steps"] = action_steps
        snapshot["affected_wells"] = tuple(
            str(dict(item).get("well_name", "")).strip()
            for item in action_steps
            if str(dict(item).get("well_name", "")).strip()
        )
    st.session_state["wt_prepared_well_overrides"] = prepared
    st.session_state["wt_prepared_recommendation_snapshot"] = (
        snapshot if prepared else None
    )
    if prepared:
        message = str(recommendation.summary)
        if skipped_wells:
            message += (
                " Не удалось подготовить anti-collision контекст для: "
                + ", ".join(sorted(skipped_wells))
                + "."
            )
        st.session_state["wt_prepared_override_message"] = message
        st.session_state["wt_prepared_recommendation_id"] = str(
            recommendation.recommendation_id
        )
        st.session_state["wt_pending_selected_names"] = list(
            dict.fromkeys(
                str(name)
                for name in recommendation.affected_wells
                if str(name) in prepared
            )
        ) or list(prepared.keys())
        return
    st.session_state["wt_prepared_override_message"] = (
        "Не удалось подготовить пересчет по выбранной anti-collision рекомендации: "
        "контекст конфликта недоступен."
    )
    st.session_state["wt_prepared_recommendation_id"] = str(
        recommendation.recommendation_id
    )
    st.session_state["wt_prepared_recommendation_snapshot"] = None
    st.session_state["wt_pending_selected_names"] = None


def _recommendation_intervals_for_moving_well(
    *,
    recommendation: AntiCollisionRecommendation,
    moving_well_name: str,
) -> tuple[str, float, float, float, float] | None:
    return recommendation_intervals_for_moving_well_shared(
        recommendation=recommendation,
        moving_well_name=moving_well_name,
    )


def _build_prepared_optimization_context(
    *,
    recommendation: AntiCollisionRecommendation,
    moving_success: SuccessfulWellPlan | None,
    reference_success: SuccessfulWellPlan | None,
    uncertainty_model: PlanningUncertaintyModel,
    all_successes: list[SuccessfulWellPlan] | None = None,
) -> AntiCollisionOptimizationContext | None:
    return build_prepared_optimization_context_shared(
        recommendation=recommendation,
        moving_success=moving_success,
        reference_success=reference_success,
        uncertainty_model=uncertainty_model,
        all_successes=all_successes,
        reference_wells=_reference_wells_from_state(),
    )


def _build_cluster_prepared_overrides(
    cluster: AntiCollisionRecommendationCluster,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    return build_cluster_prepared_overrides_shared(
        cluster,
        successes=successes,
        uncertainty_model=uncertainty_model,
        reference_wells=_reference_wells_from_state(),
    )


def _build_recommendation_prepared_overrides(
    recommendation: AntiCollisionRecommendation,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> tuple[
    dict[str, dict[str, object]], list[str], tuple[dict[str, object], ...]
]:
    return build_recommendation_prepared_overrides_shared(
        recommendation,
        successes=successes,
        uncertainty_model=uncertainty_model,
        reference_wells=_reference_wells_from_state(),
    )


def _prepare_rerun_from_cluster(
    cluster: AntiCollisionRecommendationCluster,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    target_well_names: tuple[str, ...] = (),
    focus_well_names: tuple[str, ...] = (),
) -> None:
    if (
        not bool(cluster.can_prepare_rerun)
    ) or cluster.blocking_advisory is not None:
        blocking_message = (
            str(cluster.blocking_advisory)
            if cluster.blocking_advisory is not None
            else "Для этого cluster-level пересчета доступны только advisory-рекомендации."
        )
        st.session_state["wt_prepared_well_overrides"] = {}
        st.session_state["wt_prepared_override_message"] = (
            "Cluster-level пересчет недоступен: " + blocking_message
        )
        st.session_state["wt_prepared_recommendation_id"] = ""
        st.session_state["wt_prepared_recommendation_snapshot"] = None
        st.session_state["wt_pending_selected_names"] = None
        return

    prepared, skipped_wells = _build_cluster_prepared_overrides(
        cluster,
        successes=successes,
        uncertainty_model=uncertainty_model,
    )
    snapshot = _cluster_snapshot(
        cluster,
        target_well_names=target_well_names,
        focus_well_names=focus_well_names,
    )
    st.session_state["wt_prepared_well_overrides"] = prepared
    st.session_state["wt_prepared_recommendation_snapshot"] = (
        snapshot if prepared else None
    )
    if prepared:
        message = str(cluster.summary)
        if focus_well_names:
            message += (
                " Фокус пересчета: "
                + ", ".join(str(name) for name in focus_well_names)
                + "."
            )
        if target_well_names:
            expanded_scope = [
                str(name)
                for name in target_well_names
                if str(name) not in set(str(name) for name in focus_well_names)
            ]
            message += (
                " Область cluster-level пересчета: "
                + ", ".join(str(name) for name in target_well_names)
                + "."
            )
            if expanded_scope:
                message += (
                    " Соседние расчетные скважины других кустов будут подключены "
                    "автоматически, потому что входят в тот же связанный anti-collision "
                    "кластер: " + ", ".join(expanded_scope) + "."
                )
        if cluster.blocking_advisory:
            message += " " + str(cluster.blocking_advisory)
        if skipped_wells:
            message += (
                " Не удалось подготовить anti-collision контекст для: "
                + ", ".join(sorted(skipped_wells))
                + "."
            )
        st.session_state["wt_prepared_override_message"] = message
        st.session_state["wt_prepared_recommendation_id"] = ""
        ordered_wells = [
            str(step.well_name)
            for step in cluster.action_steps
            if str(step.well_name)
            in set(target_well_names or tuple(prepared.keys()))
        ]
        pending_names = list(
            dict.fromkeys(
                [
                    *ordered_wells,
                    *(str(name) for name in target_well_names),
                    *(str(name) for name in prepared.keys()),
                ]
            )
        )
        st.session_state["wt_pending_selected_names"] = pending_names
        return
    st.session_state["wt_prepared_override_message"] = (
        "Не удалось подготовить cluster-level пересчет: контекст конфликта недоступен."
    )
    st.session_state["wt_prepared_recommendation_id"] = ""
    st.session_state["wt_prepared_recommendation_snapshot"] = None
    st.session_state["wt_pending_selected_names"] = None


def _render_source_input() -> _WelltrackSourcePayload:
    st.markdown("### Источник WELLTRACK")
    source_mode = st.radio(
        "Режим загрузки",
        options=[
            "Файл по пути",
            "Загрузить файл",
            "Вставить текст",
            "Вставить таблицу",
        ],
        horizontal=True,
        key="wt_source_mode",
    )

    if source_mode == "Файл по пути":
        source_path = st.text_input(
            "Путь к файлу WELLTRACK",
            key="wt_source_path",
            placeholder="tests/test_data/WELLTRACKS3.INC",
        )
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_text=_read_welltrack_file(source_path),
        )

    if source_mode == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Файл ECLIPSE/INC", type=["inc", "txt", "data", "ecl"]
        )
        if uploaded_file is None:
            return _WelltrackSourcePayload(mode=source_mode)
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_text=_decode_welltrack_payload(
                uploaded_file.getvalue(),
                source_label=f"Загруженный файл `{uploaded_file.name}`",
            ),
        )

    if source_mode == "Вставить текст":
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_text=st.text_area(
                "Текст WELLTRACK",
                key="wt_source_inline",
                height=220,
                placeholder="WELLTRACK 'WELL-1' ...",
            ),
        )

    with st.expander("Таблица точек WELLTRACK", expanded=True):
        note_col, clear_col = st.columns(
            [5.0, 1.2], gap="small", vertical_alignment="bottom"
        )
        with note_col:
            st.caption(
                "Вставьте таблицу в формате `Wellname`, `Point`, `X`, `Y`, `Z`. "
                "Поддерживается copy/paste из Excel, Google Sheets и похожих таблиц. "
                "Point принимает `S`, `t1`, `t3`."
            )
        with clear_col:
            if st.button(
                "Очистить",
                key="wt_source_table_clear",
                icon=":material/delete:",
                width="stretch",
            ):
                st.session_state["wt_source_table_df"] = (
                    _empty_source_table_df()
                )
                st.session_state["wt_source_table_editor_nonce"] = (
                    int(
                        st.session_state.get("wt_source_table_editor_nonce", 0)
                    )
                    + 1
                )
                st.rerun()
        source_table_df = _normalize_source_table_df_for_ui(
            st.session_state.get(
                "wt_source_table_df", _empty_source_table_df()
            )
        )
        edited_table = st.data_editor(
            source_table_df,
            key=f"wt_source_table_editor_{int(st.session_state.get('wt_source_table_editor_nonce', 0))}",
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Wellname": st.column_config.TextColumn("Wellname"),
                "Point": st.column_config.SelectboxColumn(
                    "Point",
                    options=["S", "t1", "t3"],
                ),
                "X": st.column_config.NumberColumn("X"),
                "Y": st.column_config.NumberColumn("Y"),
                "Z": st.column_config.NumberColumn("Z"),
            },
        )
        st.session_state["wt_source_table_df"] = (
            _normalize_source_table_df_for_ui(pd.DataFrame(edited_table))
        )

    return _WelltrackSourcePayload(
        mode=source_mode,
        table_rows=pd.DataFrame(st.session_state["wt_source_table_df"]),
    )


def _store_parsed_records(records: list[WelltrackRecord]) -> bool:
    all_names = [record.name for record in records]
    st.session_state["wt_records"] = list(records)
    st.session_state["wt_records_original"] = list(records)
    st.session_state["wt_loaded_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    _clear_t1_t3_order_resolution_state()
    _clear_pad_state()
    st.session_state["wt_last_error"] = ""
    _clear_results()
    auto_layout_applied = _auto_apply_pad_layout_if_shared_surface(
        records=list(records)
    )
    st.session_state["wt_selected_names"] = list(all_names)
    return auto_layout_applied


def _auto_apply_pad_layout_if_shared_surface(
    records: list[WelltrackRecord],
) -> bool:
    pads = _ensure_pad_configs(base_records=list(records))
    metadata = dict(st.session_state.get("wt_pad_detected_meta", {}))
    auto_layout_pad_ids = {
        str(pad.pad_id)
        for pad in pads
        if len(pad.wells) > 1
        and not bool(
            getattr(
                metadata.get(str(pad.pad_id)), "source_surfaces_defined", False
            )
        )
    }
    if not auto_layout_pad_ids:
        return False
    plan_map = _build_pad_plan_map(pads)
    if not any(str(pad_id) in auto_layout_pad_ids for pad_id in plan_map):
        return False
    updated_records = apply_pad_layout(
        records=list(records),
        pads=pads,
        plan_by_pad_id=plan_map,
    )
    if updated_records == list(records):
        return False
    st.session_state["wt_records"] = list(updated_records)
    st.session_state["wt_pad_last_applied_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.session_state["wt_pad_auto_applied_on_import"] = True
    return True


def _render_import_controls() -> (
    tuple[_WelltrackSourcePayload, bool, bool, bool]
):
    source_col, action_col = st.columns(
        [4.0, 1.2], gap="small", vertical_alignment="bottom"
    )
    with source_col:
        source_payload = _render_source_input()
    with action_col:
        render_small_note("Действия импорта")
        parse_clicked = st.button(
            "Прочитать WELLTRACK",
            type="primary",
            icon=":material/upload_file:",
            width="stretch",
        )
        clear_clicked = st.button(
            "Очистить импорт", icon=":material/delete:", width="stretch"
        )
        reset_params_clicked = st.button(
            "Сбросить параметры к рекомендованным",
            icon=":material/restart_alt:",
            width="stretch",
            help=(
                "Сбрасывает общие и отдельные параметры расчета/солвера к "
                "рекомендованным значениям. Импортированный WELLTRACK и выбранные "
                "скважины не удаляются."
            ),
        )
    return (
        source_payload,
        bool(parse_clicked),
        bool(clear_clicked),
        bool(reset_params_clicked),
    )


def _handle_import_actions(
    source_payload: _WelltrackSourcePayload,
    parse_clicked: bool,
    clear_clicked: bool,
    reset_params_clicked: bool,
) -> None:
    if reset_params_clicked:
        _apply_profile_defaults(force=True)
        st.toast("Параметры расчета сброшены к рекомендованным.")

    if clear_clicked:
        st.session_state["wt_records"] = None
        st.session_state["wt_records_original"] = None
        st.session_state["wt_reference_wells"] = ()
        st.session_state[
            _reference_wells_state_key(REFERENCE_WELL_ACTUAL)
        ] = ()
        st.session_state[
            _reference_wells_state_key(REFERENCE_WELL_APPROVED)
        ] = ()
        st.session_state["wt_selected_names"] = []
        st.session_state["wt_loaded_at"] = ""
        _clear_t1_t3_order_resolution_state()
        _clear_pad_state()
        _clear_results()
        st.rerun()

    if not parse_clicked:
        return
    if source_payload.mode == "Вставить таблицу":
        table_rows = source_payload.table_rows
        if table_rows is None:
            st.warning(
                "Таблица пуста. Вставьте строки в формате Wellname / Point / X / Y / Z."
            )
            return
        with st.status(
            "Чтение и преобразование таблицы точек...", expanded=True
        ) as status:
            started = perf_counter()
            try:
                status.write(
                    "Проверка строк таблицы и сборка точек S / t1 / t3."
                )
                records = parse_welltrack_points_table(
                    pd.DataFrame(table_rows).to_dict(orient="records")
                )
                auto_layout_applied = _store_parsed_records(records=records)
                status.write(f"Собрано скважин из таблицы: {len(records)}.")
                if auto_layout_applied:
                    status.write(
                        "Обнаружены кусты с общим исходным S: устья автоматически "
                        "разведены по параметрам блока 'Кусты и расчет устьев'. "
                        "При необходимости можно нажать 'Вернуть исходные устья'."
                    )
                elapsed = perf_counter() - started
                status.update(
                    label=f"Импорт таблицы завершен за {elapsed:.2f} с",
                    state="complete",
                    expanded=False,
                )
            except WelltrackParseError as exc:
                st.session_state["wt_records"] = None
                st.session_state["wt_records_original"] = None
                _clear_t1_t3_order_resolution_state()
                _clear_pad_state()
                st.session_state["wt_last_error"] = str(exc)
                status.write(str(exc))
                status.update(
                    label="Ошибка разбора табличного WELLTRACK",
                    state="error",
                    expanded=True,
                )
        return

    source_text = str(source_payload.source_text)
    if not source_text.strip():
        st.warning(
            "Источник пустой. Загрузите файл или вставьте текст WELLTRACK."
        )
        return
    with st.status("Чтение и парсинг WELLTRACK...", expanded=True) as status:
        started = perf_counter()
        try:
            status.write("Проверка структуры WELLTRACK-блоков.")
            records = _parse_welltrack_cached(source_text)
            auto_layout_applied = _store_parsed_records(records=records)
            status.write(f"Найдено блоков WELLTRACK: {len(records)}.")
            if auto_layout_applied:
                status.write(
                    "Обнаружены кусты с общим исходным S: устья автоматически "
                    "разведены по параметрам блока 'Кусты и расчет устьев'. "
                    "При необходимости можно нажать 'Вернуть исходные устья'."
                )
            elapsed = perf_counter() - started
            status.update(
                label=f"Импорт завершен за {elapsed:.2f} с",
                state="complete",
                expanded=False,
            )
        except WelltrackParseError as exc:
            st.session_state["wt_records"] = None
            st.session_state["wt_records_original"] = None
            _clear_t1_t3_order_resolution_state()
            _clear_pad_state()
            st.session_state["wt_last_error"] = str(exc)
            status.write(str(exc))
            status.update(
                label="Ошибка парсинга WELLTRACK", state="error", expanded=True
            )


def _render_records_overview(records: list[WelltrackRecord]) -> None:
    st.markdown("### Загруженные скважины")

    parsed_df = _records_overview_dataframe(records)
    ready_count = int(
        sum(str(item) == "✅" for item in parsed_df["Статус"].tolist())
    )
    problem_count = int(len(parsed_df) - ready_count)
    x1, x2, x3 = st.columns(3, gap="small")
    x1.metric("Скважин", f"{len(records)}")
    x2.metric("Готово", f"{ready_count}")
    x3.metric("Проблем", f"{problem_count}")

    st.dataframe(
        arrow_safe_text_dataframe(parsed_df),
        width="stretch",
        hide_index=True,
        column_config={
            "Скважина": st.column_config.TextColumn(
                "Скважина", width="medium"
            ),
            "Точек": st.column_config.NumberColumn(
                "Точек",
                format="%d",
                width="small",
                help="Считаются только целевые точки `t1/t3`, без устья `S`.",
            ),
            "Статус": st.column_config.TextColumn(
                "Статус",
                width="small",
                help="`✅` — импорт готов к расчёту; `❌` — проверьте колонку 'Проблема'.",
            ),
            "Проблема": st.column_config.TextColumn(
                "Проблема",
                width="large",
                help=(
                    "Показывает вероятные проблемы импорта: нет точки `S`, не хватает "
                    "t1/t3, неверный порядок точек или лишние точки."
                ),
            ),
        },
    )


def _records_overview_dataframe(
    records: list[WelltrackRecord],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Скважина": record.name,
                "Точек": _record_target_point_count(record),
                "Статус": "✅" if _record_is_ready_for_calc(record) else "❌",
                "Проблема": _record_import_problem_text(record),
            }
            for record in records
        ]
    )


def _record_target_point_count(record: WelltrackRecord) -> int:
    return int(max(len(tuple(record.points)) - 1, 0))


def _record_has_surface_like_point(record: WelltrackRecord) -> bool:
    return any(
        abs(float(point.z)) <= WT_IMPORT_WELLHEAD_Z_TOLERANCE_M
        for point in tuple(record.points)
    )


def _record_first_point_is_surface_like(record: WelltrackRecord) -> bool:
    points = tuple(record.points)
    if not points:
        return False
    return bool(abs(float(points[0].z)) <= WT_IMPORT_WELLHEAD_Z_TOLERANCE_M)


def _record_has_strictly_increasing_md(record: WelltrackRecord) -> bool:
    md_values = [float(point.md) for point in tuple(record.points)]
    return all(
        left < right
        for left, right in zip(md_values, md_values[1:], strict=False)
    )


def _record_import_problem_text(record: WelltrackRecord) -> str:
    points = tuple(record.points)
    problems: list[str] = []
    target_count = _record_target_point_count(record)
    if not points:
        problems.append("Нет точек WELLTRACK.")
    else:
        has_surface_like_point = _record_has_surface_like_point(record)
        if not has_surface_like_point:
            problems.append(
                "Не найдена точка `S`: среди точек нет `Z` около поверхности (±100 м)."
            )
        elif not _record_first_point_is_surface_like(record):
            problems.append("Первая точка не похожа на устье `S`.")
        if not _record_has_strictly_increasing_md(record):
            problems.append("MD точек должны строго возрастать.")
    if target_count < 2:
        missing_count = 2 - target_count
        if missing_count == 2:
            problems.append("Не хватает точек `t1` и `t3`.")
        else:
            problems.append("Не хватает одной из точек `t1/t3`.")
    elif target_count > 2:
        problems.append("Лишние точки: ожидаются только `S`, `t1`, `t3`.")
    return "—" if not problems else " ".join(problems)


def _record_is_ready_for_calc(record: WelltrackRecord) -> bool:
    return _record_import_problem_text(record) == "—"


def _reference_kind_title(kind: str) -> str:
    if str(kind) == REFERENCE_WELL_ACTUAL:
        return "Фактические скважины"
    return "Проектные утвержденные скважины"


def _reference_kind_help(kind: str) -> str:
    if str(kind) == REFERENCE_WELL_ACTUAL:
        return (
            "Фактические скважины считаются заданными: они участвуют в visual / "
            "anti-collision, но не перестраиваются."
        )
    return (
        "Утвержденные проектные скважины считаются заданными: они участвуют в visual / "
        "anti-collision, но не перестраиваются."
    )


def _reference_kind_wells(kind: str) -> tuple[ImportedTrajectoryWell, ...]:
    return tuple(st.session_state.get(_reference_wells_state_key(kind)) or ())


def _render_reference_kind_import_block(*, kind: str) -> None:
    title = _reference_kind_title(kind)
    current_wells = _reference_kind_wells(kind)
    mode = st.radio(
        f"Источник для {title.lower()}",
        options=[
            "Вставить XYZ/MD текст",
            "Загрузить XYZ/MD файл",
            "Путь к WELLTRACK",
            "Загрузить WELLTRACK",
        ],
        key=_reference_source_mode_key(kind),
        horizontal=True,
        label_visibility="collapsed",
    )
    # st.caption(_reference_kind_help(kind))

    uploaded_xyz_file = None
    uploaded_welltrack_file = None

    if mode == "Вставить XYZ/MD текст":
        st.text_area(
            "Текст траекторий",
            key=_reference_source_text_key(kind),
            height=220,
            placeholder=(
                "Wellname X Y Z MD\n"
                "WELL-1 0 25 0 0\n"
                "WELL-1 900 25 300 950\n"
                "WELL-1 1800 25 400 1900"
            ),
        )
        st.caption(
            "Формат bulk-вставки: `Wellname X Y Z MD`. "
            "Разделители: пробел, tab, `,` или `;`. Первая строка может быть header."
        )
    elif mode == "Загрузить XYZ/MD файл":
        uploaded_xyz_file = st.file_uploader(
            f"Файл XYZ/MD для {title.lower()}",
            type=["txt", "csv", "tsv", "dat"],
            key=f"wt_reference_{kind}_xyz_file",
        )
    elif mode == "Путь к WELLTRACK":
        st.text_input(
            "Путь к WELLTRACK",
            key=_reference_welltrack_path_key(kind),
            placeholder="tests/test_data/WELLTRACKS3.INC",
        )
        st.caption(
            "Можно загрузить сразу несколько скважин из WELLTRACK. "
            "Они будут помечены как заданные и не будут перестраиваться."
        )
    else:
        uploaded_welltrack_file = st.file_uploader(
            f"WELLTRACK файл для {title.lower()}",
            type=["inc", "txt", "data", "ecl"],
            key=f"wt_reference_{kind}_welltrack_file",
        )

    action_col, clear_col = st.columns(2, gap="small")
    import_clicked = action_col.button(
        f"Импортировать {title.lower()}",
        key=f"wt_reference_import_{kind}",
        type="primary",
        icon=":material/upload_file:",
        use_container_width=True,
    )
    clear_clicked = clear_col.button(
        f"Очистить {title.lower()}",
        key=f"wt_reference_clear_{kind}",
        icon=":material/delete:",
        use_container_width=True,
    )

    if import_clicked:
        with st.status(f"Импорт {title.lower()}...", expanded=True) as status:
            started = perf_counter()
            try:
                if mode == "Вставить XYZ/MD текст":
                    parsed = parse_reference_trajectory_text_with_kind(
                        str(
                            st.session_state.get(
                                _reference_source_text_key(kind), ""
                            )
                        ),
                        default_kind=kind,
                    )
                elif mode == "Загрузить XYZ/MD файл":
                    payload = (
                        b""
                        if uploaded_xyz_file is None
                        else uploaded_xyz_file.getvalue()
                    )
                    parsed = parse_reference_trajectory_text_with_kind(
                        _decode_welltrack_payload(
                            payload,
                            source_label=f"Файл XYZ/MD `{getattr(uploaded_xyz_file, 'name', 'uploaded')}`",
                        ),
                        default_kind=kind,
                    )
                elif mode == "Путь к WELLTRACK":
                    parsed = parse_reference_trajectory_welltrack_text(
                        _read_welltrack_file(
                            str(
                                st.session_state.get(
                                    _reference_welltrack_path_key(kind), ""
                                )
                            )
                        ),
                        kind=kind,
                    )
                else:
                    payload = (
                        b""
                        if uploaded_welltrack_file is None
                        else uploaded_welltrack_file.getvalue()
                    )
                    parsed = parse_reference_trajectory_welltrack_text(
                        _decode_welltrack_payload(
                            payload,
                            source_label=f"WELLTRACK `{getattr(uploaded_welltrack_file, 'name', 'uploaded')}`",
                        ),
                        kind=kind,
                    )
                _set_reference_wells_for_kind(kind=kind, wells=parsed)
                _reset_anticollision_view_state(clear_prepared=True)
                status.write(f"Загружено скважин: {len(parsed)}.")
                status.update(
                    label=f"{title} импортированы за {perf_counter() - started:.2f} с",
                    state="complete",
                    expanded=False,
                )
                st.rerun()
            except WelltrackParseError as exc:
                status.write(str(exc))
                status.update(
                    label=f"Ошибка импорта: {title.lower()}",
                    state="error",
                    expanded=True,
                )

    if clear_clicked:
        _set_reference_wells_for_kind(kind=kind, wells=())
        st.session_state[_reference_source_text_key(kind)] = ""
        st.session_state[_reference_welltrack_path_key(kind)] = ""
        _reset_anticollision_view_state(clear_prepared=True)
        st.rerun()

    if current_wells:
        st.caption(f"Загружено {len(current_wells)} скважин.")
    else:
        st.caption("Скважины этого типа не загружены.")


ACTUAL_FUND_ZONE_COLORS: dict[str, str] = {
    "vertical": "#2563EB",
    "build1": "#F59E0B",
    "hold": "#16A34A",
    "build2": "#8B5CF6",
    "horizontal": "#0F766E",
}


def _actual_fund_zone_color(zone_key: str) -> str:
    return ACTUAL_FUND_ZONE_COLORS.get(str(zone_key), "#475569")


def _actual_fund_interp_row(
    survey: pd.DataFrame, md_m: float
) -> dict[str, float]:
    md_values = survey["MD_m"].to_numpy(dtype=float)
    return {
        column: float(
            np.interp(
                float(md_m), md_values, survey[column].to_numpy(dtype=float)
            )
        )
        for column in (
            "MD_m",
            "X_m",
            "Y_m",
            "Z_m",
            "INC_deg",
            "DLS_deg_per_30m",
            "Lateral_m",
        )
    }


def _actual_fund_interval_df(
    survey: pd.DataFrame,
    start_md_m: float,
    end_md_m: float,
) -> pd.DataFrame:
    if end_md_m <= start_md_m + SMALL:
        return pd.DataFrame(columns=survey.columns)
    interval = survey.loc[
        (survey["MD_m"] >= float(start_md_m) - SMALL)
        & (survey["MD_m"] <= float(end_md_m) + SMALL)
    ].copy()
    if (
        interval.empty
        or abs(float(interval["MD_m"].iloc[0]) - float(start_md_m)) > SMALL
    ):
        interval = pd.concat(
            [
                pd.DataFrame(
                    [_actual_fund_interp_row(survey, float(start_md_m))]
                ),
                interval,
            ],
            ignore_index=True,
        )
    if abs(float(interval["MD_m"].iloc[-1]) - float(end_md_m)) > SMALL:
        interval = pd.concat(
            [
                interval,
                pd.DataFrame(
                    [_actual_fund_interp_row(survey, float(end_md_m))]
                ),
            ],
            ignore_index=True,
        )
    return interval.sort_values("MD_m").reset_index(drop=True)


def _actual_fund_kop_marker(
    detail: ActualFundWellAnalysis,
) -> dict[str, float] | None:
    if detail.metrics.kop_md_m is None:
        return None
    return _actual_fund_interp_row(
        detail.survey, float(detail.metrics.kop_md_m)
    )


def _actual_fund_horizontal_entry_marker(
    detail: ActualFundWellAnalysis,
) -> dict[str, float] | None:
    if detail.metrics.horizontal_entry_md_m is None:
        return None
    return _actual_fund_interp_row(
        detail.survey, float(detail.metrics.horizontal_entry_md_m)
    )


def _actual_fund_lateral_from_horizontal_entry_m(
    detail: ActualFundWellAnalysis,
) -> float | None:
    entry = _actual_fund_horizontal_entry_marker(detail)
    if entry is None:
        return None
    end_x = float(detail.survey["X_m"].iloc[-1])
    end_y = float(detail.survey["Y_m"].iloc[-1])
    return float(
        np.hypot(end_x - float(entry["X_m"]), end_y - float(entry["Y_m"]))
    )


def _xy_interval_azimuth_deg(interval: pd.DataFrame) -> float | None:
    if len(interval) < 2 or not {"X_m", "Y_m"}.issubset(interval.columns):
        return None
    x_values = interval["X_m"].to_numpy(dtype=float)
    y_values = interval["Y_m"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    if len(x_values) < 2:
        return None
    dx = float(x_values[-1] - x_values[0])
    dy = float(y_values[-1] - y_values[0])
    if abs(dx) <= SMALL and abs(dy) <= SMALL:
        return None
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def _reference_profile_azimuth_deg(detail: ActualFundWellAnalysis) -> float:
    zone_by_key = {str(item.zone_key): item for item in detail.zone_summaries}
    for preferred_zone_key in (ZONE_HORIZONTAL, ZONE_BUILD2, ZONE_HOLD):
        zone = zone_by_key.get(preferred_zone_key)
        if zone is None:
            continue
        interval = _actual_fund_interval_df(
            detail.survey,
            float(zone.md_from_m),
            float(zone.md_to_m),
        )
        xy_azimuth = _xy_interval_azimuth_deg(interval)
        if xy_azimuth is not None:
            return xy_azimuth

    azi_values = detail.survey["AZI_deg"].to_numpy(dtype=float)
    finite_azi = azi_values[np.isfinite(azi_values)]
    if len(finite_azi) == 0:
        return 0.0
    return float(finite_azi[-1]) % 360.0


def _actual_fund_section_coordinate(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    surface_x: float,
    surface_y: float,
    azimuth_deg: float,
) -> np.ndarray:
    angle_rad = np.deg2rad(float(azimuth_deg) % 360.0)
    ux = float(np.sin(angle_rad))
    uy = float(np.cos(angle_rad))
    return (np.asarray(x_values, dtype=float) - float(surface_x)) * ux + (
        np.asarray(y_values, dtype=float) - float(surface_y)
    ) * uy


def _actual_fund_zone_table_rows(
    detail: ActualFundWellAnalysis,
) -> list[dict[str, object]]:
    return [
        {
            "Участок": str(item.zone_label),
            "Интервал MD, м": f"{float(item.md_from_m):.0f} - {float(item.md_to_m):.0f}",
            "Длина MD, м": float(item.md_length_m),
            "INC min, deg": item.inc_min_deg,
            "INC avg, deg": item.inc_mean_deg,
            "INC max, deg": item.inc_max_deg,
            "DLS min, deg/30м": item.dls_min_deg_per_30m,
            "DLS avg, deg/30м": item.dls_mean_deg_per_30m,
            "DLS max, deg/30м": item.dls_max_deg_per_30m,
        }
        for item in detail.zone_summaries
    ]


def _actual_fund_analysis_signature(
    actual_wells: tuple[ImportedTrajectoryWell, ...],
) -> tuple[tuple[str, object, ...], ...]:
    signature_rows: list[tuple[str, object, ...]] = []
    for item in actual_wells:
        stations = item.stations
        last_md = (
            float(stations["MD_m"].iloc[-1])
            if not stations.empty and "MD_m" in stations.columns
            else 0.0
        )
        point_count = int(len(stations))
        last_xyz = (
            (
                float(stations["X_m"].iloc[-1])
                if point_count and "X_m" in stations.columns
                else 0.0
            ),
            (
                float(stations["Y_m"].iloc[-1])
                if point_count and "Y_m" in stations.columns
                else 0.0
            ),
            (
                float(stations["Z_m"].iloc[-1])
                if point_count and "Z_m" in stations.columns
                else 0.0
            ),
        )
        signature_rows.append(
            (
                str(item.name),
                str(item.kind),
                point_count,
                last_md,
                *last_xyz,
            )
        )
    return tuple(signature_rows)


def _actual_fund_analyses(
    actual_wells: tuple[ImportedTrajectoryWell, ...],
) -> tuple[ActualFundWellAnalysis, ...]:
    signature = _actual_fund_analysis_signature(actual_wells)
    cache_key = "wt_actual_fund_analysis_cache"
    cached = st.session_state.get(cache_key)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        analyses = cached.get("analyses")
        if isinstance(analyses, tuple):
            return analyses
    analyses = build_actual_fund_well_analyses(actual_wells)
    st.session_state[cache_key] = {
        "signature": signature,
        "analyses": analyses,
    }
    return analyses


def _actual_fund_depth_cluster_color(index: int) -> str:
    palette = (
        "#2563EB",
        "#059669",
        "#D97706",
        "#7C3AED",
        "#DC2626",
        "#0891B2",
    )
    return palette[index % len(palette)]


def _actual_fund_kop_depth_figure(
    metrics: tuple[object, ...],
) -> go.Figure | None:
    eligible_metrics = [
        item
        for item in metrics
        if getattr(item, "is_analysis_eligible", False)
        and getattr(item, "horizontal_entry_tvd_m", None) is not None
        and getattr(item, "kop_md_m", None) is not None
    ]
    if not eligible_metrics:
        return None

    clusters = summarize_actual_fund_by_depth(eligible_metrics)
    cluster_by_well: dict[str, str] = {}
    for cluster in clusters:
        for well_name in cluster.well_names:
            cluster_by_well[str(well_name)] = str(cluster.cluster_id)
    cluster_order = [str(cluster.cluster_id) for cluster in clusters]
    cluster_color_by_id = {
        cluster_id: _actual_fund_depth_cluster_color(index)
        for index, cluster_id in enumerate(cluster_order)
    }

    fig = go.Figure()
    for cluster_id in cluster_order:
        cluster_items = [
            item
            for item in eligible_metrics
            if cluster_by_well.get(str(getattr(item, "name", "")))
            == cluster_id
        ]
        if not cluster_items:
            continue
        fig.add_trace(
            go.Scatter(
                x=[
                    float(item.horizontal_entry_tvd_m)
                    for item in cluster_items
                ],
                y=[float(item.kop_md_m) for item in cluster_items],
                mode="markers",
                name=cluster_id,
                marker={
                    "size": 10,
                    "color": cluster_color_by_id[cluster_id],
                    "line": {"width": 1, "color": "#FFFFFF"},
                },
                customdata=[
                    [
                        str(item.name),
                        float(item.horizontal_entry_tvd_m),
                        float(item.kop_md_m),
                    ]
                    for item in cluster_items
                ],
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br>TVD входа=%{customdata[1]:.1f} м"
                    "<br>KOP=%{customdata[2]:.1f} м"
                    "<extra></extra>"
                ),
            )
        )

    kop_function = build_actual_fund_kop_depth_function(eligible_metrics)
    if kop_function is not None:
        anchor_depths = np.asarray(
            kop_function.anchor_depths_tvd_m, dtype=float
        )
        anchor_kops = np.asarray(kop_function.anchor_kop_md_m, dtype=float)
        line_depths = np.linspace(
            float(np.min(anchor_depths)),
            float(np.max(anchor_depths)),
            max(2, 60),
            dtype=float,
        )
        line_kops = np.asarray(
            [kop_function.evaluate(float(depth)) for depth in line_depths],
            dtype=float,
        )
        fig.add_trace(
            go.Scatter(
                x=line_depths,
                y=line_kops,
                mode="lines",
                name="KOP(TVD)",
                line={"color": "#0F172A", "width": 2},
                hovertemplate=(
                    "KOP(TVD)"
                    "<br>TVD=%{x:.1f} м"
                    "<br>KOP=%{y:.1f} м"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anchor_depths,
                y=anchor_kops,
                mode="markers",
                name="Опорные точки",
                marker={"size": 8, "symbol": "diamond", "color": "#111827"},
                hovertemplate=(
                    "Опорная точка"
                    "<br>TVD=%{x:.1f} м"
                    "<br>KOP=%{y:.1f} м"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="TVD входа в ГС, м",
        yaxis_title="KOP MD, м",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 24, "b": 20},
        template="plotly_white",
    )
    return fig


def _actual_fund_plan_figure(detail: ActualFundWellAnalysis) -> go.Figure:
    figure = go.Figure()
    x_arrays = [detail.survey["X_m"].to_numpy(dtype=float)]
    y_arrays = [detail.survey["Y_m"].to_numpy(dtype=float)]
    kop_marker = _actual_fund_kop_marker(detail)
    if kop_marker is not None:
        x_arrays.append(np.asarray([float(kop_marker["X_m"])], dtype=float))
        y_arrays.append(np.asarray([float(kop_marker["Y_m"])], dtype=float))
    horizontal_marker = _actual_fund_horizontal_entry_marker(detail)
    if horizontal_marker is not None:
        x_arrays.append(
            np.asarray([float(horizontal_marker["X_m"])], dtype=float)
        )
        y_arrays.append(
            np.asarray([float(horizontal_marker["Y_m"])], dtype=float)
        )
    x_range, y_range = equalized_xy_ranges(
        x_values=np.concatenate(x_arrays),
        y_values=np.concatenate(y_arrays),
    )
    shown_legend_keys: set[str] = set()
    for zone in detail.zone_summaries:
        interval = _actual_fund_interval_df(
            detail.survey,
            float(zone.md_from_m),
            float(zone.md_to_m),
        )
        if len(interval) < 2:
            continue
        zone_key = str(zone.zone_key)
        customdata = np.column_stack(
            [
                interval["MD_m"].to_numpy(dtype=float),
                interval["Z_m"].to_numpy(dtype=float),
                interval["DLS_deg_per_30m"].to_numpy(dtype=float),
                interval["INC_deg"].to_numpy(dtype=float),
            ]
        )
        figure.add_trace(
            go.Scatter(
                x=interval["X_m"],
                y=interval["Y_m"],
                mode="lines",
                name=str(zone.zone_label),
                showlegend=zone_key not in shown_legend_keys,
                legendgroup=zone_key,
                line={"color": _actual_fund_zone_color(zone_key), "width": 4},
                customdata=customdata,
                hovertemplate=(
                    f"{detail.name}<br>Участок: {zone.zone_label}"
                    "<br>MD=%{customdata[0]:.1f} м"
                    "<br>Z=%{customdata[1]:.1f} м"
                    "<br>DLS=%{customdata[2]:.2f} deg/30м"
                    "<br>INC=%{customdata[3]:.2f} deg"
                    "<extra></extra>"
                ),
            )
        )
        shown_legend_keys.add(zone_key)

    if kop_marker is not None:
        figure.add_trace(
            go.Scatter(
                x=[float(kop_marker["X_m"])],
                y=[float(kop_marker["Y_m"])],
                mode="markers+text",
                name="KOP",
                text=["KOP"],
                textposition="top center",
                marker={"color": "#111827", "symbol": "diamond", "size": 10},
                customdata=[
                    [
                        float(kop_marker["MD_m"]),
                        float(kop_marker["Z_m"]),
                        float(kop_marker["DLS_deg_per_30m"]),
                        float(kop_marker["INC_deg"]),
                    ]
                ],
                hovertemplate=(
                    "KOP"
                    "<br>MD=%{customdata[0]:.1f} м"
                    "<br>Z=%{customdata[1]:.1f} м"
                    "<br>DLS=%{customdata[2]:.2f} deg/30м"
                    "<br>INC=%{customdata[3]:.2f} deg"
                    "<extra></extra>"
                ),
            )
        )

    if horizontal_marker is not None:
        figure.add_trace(
            go.Scatter(
                x=[float(horizontal_marker["X_m"])],
                y=[float(horizontal_marker["Y_m"])],
                mode="markers+text",
                name="Старт горизонта",
                text=["ГС"],
                textposition="top center",
                marker={"color": "#B91C1C", "symbol": "square", "size": 9},
                customdata=[
                    [
                        float(horizontal_marker["MD_m"]),
                        float(horizontal_marker["Z_m"]),
                        float(horizontal_marker["DLS_deg_per_30m"]),
                        float(horizontal_marker["INC_deg"]),
                    ]
                ],
                hovertemplate=(
                    "Вход в ГС"
                    "<br>MD=%{customdata[0]:.1f} м"
                    "<br>Z=%{customdata[1]:.1f} м"
                    "<br>DLS=%{customdata[2]:.2f} deg/30м"
                    "<br>INC=%{customdata[3]:.2f} deg"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        xaxis_title="X / Восток (м)",
        yaxis_title="Y / Север (м)",
        xaxis={"range": x_range, "tickformat": ".0f"},
        yaxis={
            "range": y_range,
            "tickformat": ".0f",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        template="plotly_white",
    )
    return figure


def _actual_fund_vertical_profile_figure(
    detail: ActualFundWellAnalysis,
) -> go.Figure:
    figure = go.Figure()
    surface_x = float(detail.survey["X_m"].iloc[0])
    surface_y = float(detail.survey["Y_m"].iloc[0])
    profile_azimuth_deg = _reference_profile_azimuth_deg(detail)
    shown_legend_keys: set[str] = set()
    for zone in detail.zone_summaries:
        interval = _actual_fund_interval_df(
            detail.survey,
            float(zone.md_from_m),
            float(zone.md_to_m),
        )
        if len(interval) < 2:
            continue
        zone_key = str(zone.zone_key)
        section_x = _actual_fund_section_coordinate(
            interval["X_m"].to_numpy(dtype=float),
            interval["Y_m"].to_numpy(dtype=float),
            surface_x=surface_x,
            surface_y=surface_y,
            azimuth_deg=profile_azimuth_deg,
        )
        customdata = np.column_stack(
            [
                interval["MD_m"].to_numpy(dtype=float),
                interval["Z_m"].to_numpy(dtype=float),
                interval["DLS_deg_per_30m"].to_numpy(dtype=float),
                interval["INC_deg"].to_numpy(dtype=float),
            ]
        )
        figure.add_trace(
            go.Scatter(
                x=section_x,
                y=interval["Z_m"],
                mode="lines",
                name=str(zone.zone_label),
                showlegend=zone_key not in shown_legend_keys,
                legendgroup=zone_key,
                line={"color": _actual_fund_zone_color(zone_key), "width": 4},
                customdata=customdata,
                hovertemplate=(
                    f"{detail.name}<br>Участок: {zone.zone_label}"
                    "<br>MD=%{customdata[0]:.1f} м"
                    "<br>Z=%{customdata[1]:.1f} м"
                    "<br>DLS=%{customdata[2]:.2f} deg/30м"
                    "<br>INC=%{customdata[3]:.2f} deg"
                    "<extra></extra>"
                ),
            )
        )
        shown_legend_keys.add(zone_key)

    kop_marker = _actual_fund_kop_marker(detail)
    if kop_marker is not None:
        kop_section_x = float(
            _actual_fund_section_coordinate(
                np.asarray([float(kop_marker["X_m"])], dtype=float),
                np.asarray([float(kop_marker["Y_m"])], dtype=float),
                surface_x=surface_x,
                surface_y=surface_y,
                azimuth_deg=profile_azimuth_deg,
            )[0]
        )
        figure.add_trace(
            go.Scatter(
                x=[kop_section_x],
                y=[float(kop_marker["Z_m"])],
                mode="markers+text",
                name="KOP",
                text=["KOP"],
                textposition="top center",
                marker={"color": "#111827", "symbol": "diamond", "size": 10},
                customdata=[
                    [
                        float(kop_marker["MD_m"]),
                        float(kop_marker["Z_m"]),
                        float(kop_marker["DLS_deg_per_30m"]),
                        float(kop_marker["INC_deg"]),
                    ]
                ],
                hovertemplate=(
                    "KOP"
                    "<br>MD=%{customdata[0]:.1f} м"
                    "<br>Z=%{customdata[1]:.1f} м"
                    "<br>DLS=%{customdata[2]:.2f} deg/30м"
                    "<br>INC=%{customdata[3]:.2f} deg"
                    "<extra></extra>"
                ),
            )
        )

    horizontal_marker = _actual_fund_horizontal_entry_marker(detail)
    if horizontal_marker is not None:
        horizontal_section_x = float(
            _actual_fund_section_coordinate(
                np.asarray([float(horizontal_marker["X_m"])], dtype=float),
                np.asarray([float(horizontal_marker["Y_m"])], dtype=float),
                surface_x=surface_x,
                surface_y=surface_y,
                azimuth_deg=profile_azimuth_deg,
            )[0]
        )
        figure.add_trace(
            go.Scatter(
                x=[horizontal_section_x],
                y=[float(horizontal_marker["Z_m"])],
                mode="markers+text",
                name="Старт горизонта",
                text=["ГС"],
                textposition="top center",
                marker={"color": "#B91C1C", "symbol": "square", "size": 9},
                customdata=[
                    [
                        float(horizontal_marker["MD_m"]),
                        float(horizontal_marker["Z_m"]),
                        float(horizontal_marker["DLS_deg_per_30m"]),
                        float(horizontal_marker["INC_deg"]),
                    ]
                ],
                hovertemplate=(
                    "Вход в ГС"
                    "<br>MD=%{customdata[0]:.1f} м"
                    "<br>Z=%{customdata[1]:.1f} м"
                    "<br>DLS=%{customdata[2]:.2f} deg/30м"
                    "<br>INC=%{customdata[3]:.2f} deg"
                    "<extra></extra>"
                ),
            )
        )
    profile_y_values = np.concatenate(
        [
            np.asarray(trace.y, dtype=float)
            for trace in figure.data
            if getattr(trace, "y", None) is not None and len(trace.y) > 0
        ]
    )

    figure.update_layout(
        xaxis_title="Координата по разрезу (м)",
        yaxis_title="Z / TVD (м)",
        yaxis={"range": reversed_axis_range(profile_y_values)},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        template="plotly_white",
    )
    return figure


def _render_reference_well_detail(
    analyses: tuple[ActualFundWellAnalysis, ...],
    *,
    select_label: str,
    selected_key: str,
) -> None:
    if not analyses:
        return
    names = [str(item.name) for item in analyses]
    default_name = next(
        (
            str(item.name)
            for item in analyses
            if bool(item.metrics.is_analysis_eligible)
        ),
        names[0],
    )
    selected_name = st.selectbox(
        select_label,
        options=names,
        index=max(names.index(default_name), 0),
        key=selected_key,
    )
    detail = next(
        item for item in analyses if str(item.name) == str(selected_name)
    )
    metrics = detail.metrics
    lateral_from_horizontal_entry_m = (
        _actual_fund_lateral_from_horizontal_entry_m(detail)
    )

    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    m1.metric("Итоговая MD, м", f"{float(metrics.md_total_m):.0f}")
    m2.metric(
        "KOP MD, м",
        "—" if metrics.kop_md_m is None else f"{float(metrics.kop_md_m):.0f}",
    )
    m3.metric(
        "Вход в ГС, MD",
        (
            "—"
            if metrics.horizontal_entry_md_m is None
            else f"{float(metrics.horizontal_entry_md_m):.0f}"
        ),
    )
    m4.metric(
        "Длина ГС, м",
        (
            "—"
            if not metrics.is_horizontal
            else f"{float(metrics.horizontal_length_m):.0f}"
        ),
    )
    m5.metric(
        "Отход, м",
        (
            "—"
            if lateral_from_horizontal_entry_m is None
            else f"{float(lateral_from_horizontal_entry_m):.0f}"
        ),
    )
    if not bool(metrics.is_analysis_eligible):
        st.info(
            "Скважина исключена из общего анализа: "
            f"{metrics.analysis_exclusion_reason or 'без пояснения'}."
        )

    c1, c2 = st.columns(2, gap="medium")
    c1.markdown(f"**{detail.name}: план**")
    c1.plotly_chart(_actual_fund_plan_figure(detail), width="stretch")
    c2.markdown(f"**{detail.name}: профиль**")
    c2.plotly_chart(
        _actual_fund_vertical_profile_figure(detail), width="stretch"
    )
    st.dataframe(
        arrow_safe_text_dataframe(
            pd.DataFrame(_actual_fund_zone_table_rows(detail))
        ),
        width="stretch",
        hide_index=True,
    )


def _render_actual_fund_well_detail(
    analyses: tuple[ActualFundWellAnalysis, ...],
) -> None:
    _render_reference_well_detail(
        analyses,
        select_label="Просмотр фактической скважины",
        selected_key="wt_actual_fund_selected_well",
    )


def _render_actual_fund_analysis_panel(
    analyses: tuple[ActualFundWellAnalysis, ...] | None = None,
) -> None:
    actual_wells = _reference_kind_wells(REFERENCE_WELL_ACTUAL)
    if not actual_wells:
        return

    if analyses is None:
        try:
            analyses = _actual_fund_analyses(actual_wells)
        except Exception as exc:
            with st.expander("Анализ фактического фонда", expanded=False):
                st.error(
                    "Не удалось построить анализ фактического фонда для загруженных скважин."
                )
                st.caption(f"{type(exc).__name__}: {exc}")
            return
    metrics = tuple(item.metrics for item in analyses)
    eligible_metrics = [
        item for item in metrics if bool(item.is_analysis_eligible)
    ]
    excluded_horizontal_metrics = [
        item
        for item in metrics
        if bool(item.is_horizontal) and not bool(item.is_analysis_eligible)
    ]
    pad_count = len(
        {
            str(item.pad_group)
            for item in eligible_metrics
            if str(item.pad_group) != "—"
        }
    )
    depth_clusters = summarize_actual_fund_by_depth(eligible_metrics)
    kop_depth_function = build_actual_fund_kop_depth_function(eligible_metrics)
    eligible_kop_values = [
        float(item.kop_md_m)
        for item in eligible_metrics
        if item.kop_md_m is not None
    ]

    with st.expander("Анализ фактического фонда", expanded=False):
        st.caption(
            "Учитываются только горизонтальные скважины с читаемым профилем "
            "(KOP, HOLD, ГС и ПИ в норме). Скважины с аномалиями исключаются."
        )
        a1, a2, a3, a4, a5 = st.columns(5, gap="small")
        a1.metric("Фактических скважин", f"{len(actual_wells)}")
        a2.metric("В анализе", f"{len(eligible_metrics)}")
        a3.metric("Кустов", f"{pad_count}")
        a4.metric(
            "Медианный KOP, м",
            (
                "—"
                if not eligible_kop_values
                else f"{float(np.median(eligible_kop_values)):.0f}"
            ),
        )
        a5.metric("Глубинных кластеров", f"{len(depth_clusters)}")
        if excluded_horizontal_metrics:
            st.info(
                f"Исключено скважин: {len(excluded_horizontal_metrics)}. "
                "Причины — в таблице."
            )

        view_mode = st.radio(
            "Свод фактического фонда",
            options=["По скважинам", "По кустам", "По глубинам"],
            key="wt_actual_fund_analysis_view_mode",
            horizontal=True,
        )
        if str(view_mode) == "По кустам":
            df = pd.DataFrame(actual_fund_pad_rows(metrics))
        elif str(view_mode) == "По глубинам":
            df = pd.DataFrame(actual_fund_depth_rows(metrics))
        else:
            df = pd.DataFrame(actual_fund_metrics_rows(metrics))
        st.dataframe(
            arrow_safe_text_dataframe(df),
            width="stretch",
            hide_index=True,
        )

        if kop_depth_function is not None:
            st.markdown("#### KOP(TVD) по фактическому фонду")
            st.caption(
                "Скважины сгруппированы по TVD входа в ГС. "
                "Для каждого кластера опорный KOP = min + 1σ (без выбросов). "
                "Полученная зависимость применяется к проектным скважинам через TVD t1."
            )
            figure = _actual_fund_kop_depth_figure(metrics)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
            f1, f2 = st.columns([3.6, 1.4], gap="small")
            with f1:
                st.caption(str(kop_depth_function.note))
                if kop_depth_function.anchor_depths_tvd_m:
                    st.caption(
                        "Опорные точки TVD → KOP: "
                        + ", ".join(
                            f"{float(depth):.0f} → {float(kop):.0f}"
                            for depth, kop in zip(
                                kop_depth_function.anchor_depths_tvd_m,
                                kop_depth_function.anchor_kop_md_m,
                                strict=False,
                            )
                        )
                    )
            with f2:
                if st.button(
                    "Применить функцию KOP / TVD",
                    key="wt_apply_actual_fund_kop_depth_function",
                    icon=":material/timeline:",
                    width="stretch",
                ):
                    set_kop_min_vertical_function(
                        prefix=WT_CALC_PARAMS.prefix,
                        kop_function=kop_depth_function,
                    )
                    st.toast(
                        "Функция KOP / TVD применена к параметрам расчёта."
                    )
                    st.rerun()
                if kop_min_vertical_mode(WT_CALC_PARAMS.prefix) != "constant":
                    if st.button(
                        "Вернуть фиксированный KOP",
                        key="wt_clear_actual_fund_kop_depth_function",
                        icon=":material/looks_one:",
                        width="stretch",
                    ):
                        clear_kop_min_vertical_function(
                            prefix=WT_CALC_PARAMS.prefix
                        )
                        st.toast("Возвращён режим фиксированного KOP.")
                        st.rerun()
            if (
                kop_min_vertical_function_from_state(
                    prefix=WT_CALC_PARAMS.prefix
                )
                is not None
            ):
                st.success(
                    "В параметрах расчёта активна функция KOP / TVD. Для каждой "
                    "выбранной скважины `Мин VERTICAL до KOP` будет вычисляться по глубине `t1`."
                )

        st.markdown("#### Просмотр выбранной скважины")
        st.caption(
            "Профили строятся по реконструированной survey-линии из фактического XYZ/MD. "
            "Зоны `Vertical / BUILD1 / HOLD / BUILD2 / Horizontal` и точки `KOP / старт ГС` "
            "используют ту же сегментацию, что и aggregate-анализ."
        )
        _render_actual_fund_well_detail(analyses)


def _render_reference_trajectory_panel() -> None:
    current_wells = _reference_wells_from_state()

    with st.container(border=True):
        st.markdown("### Дополнительные скважины для visual / anti-collision")
        st.caption(
            "Фактические и утвержденные проектные скважины загружаются независимо. "
            "Они считаются заданными, отображаются на графиках, получают конусы "
            "неопределенности и участвуют в anti-collision как reference wells."
        )
        m1, m2, m3 = st.columns(3, gap="small")
        m1.metric("Доп. скважин", f"{len(current_wells)}")
        m2.metric(
            "Фактических",
            f"{sum(1 for item in current_wells if str(item.kind) == REFERENCE_WELL_ACTUAL)}",
        )
        m3.metric(
            "Проектных утвержденных",
            f"{sum(1 for item in current_wells if str(item.kind) == REFERENCE_WELL_APPROVED)}",
        )
        with st.expander("Фактические скважины", expanded=False):
            _render_reference_kind_import_block(kind=REFERENCE_WELL_ACTUAL)
        with st.expander("Проектные утвержденные скважины", expanded=False):
            _render_reference_kind_import_block(kind=REFERENCE_WELL_APPROVED)
        _render_actual_fund_analysis_panel()

        if current_wells:
            st.dataframe(
                arrow_safe_text_dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Скважина": reference_well_display_label(item),
                                "Точек": int(len(item.stations)),
                                "MD max, м": float(
                                    item.stations["MD_m"].iloc[-1]
                                ),
                            }
                            for item in current_wells
                        ]
                    )
                ),
                width="stretch",
                hide_index=True,
            )


def _render_raw_records_table(records: list[WelltrackRecord]) -> None:
    with st.expander(
        "Текущие точки скважин (используются в расчете, включая обновленные устья S)",
        expanded=False,
    ):
        raw_rows: list[dict[str, object]] = []
        for record in records:
            for idx, point in enumerate(record.points, start=1):
                if idx == 1:
                    point_label = "S"
                elif idx == 2:
                    point_label = "t1"
                elif idx == 3:
                    point_label = "t3"
                else:
                    point_label = f"p{idx}"
                raw_rows.append(
                    {
                        "Скважина": record.name,
                        "Порядок": idx,
                        "Точка": point_label,
                        "X, м": float(point.x),
                        "Y, м": float(point.y),
                        "Z/TVD, м": float(point.z),
                    }
                )
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(raw_rows)),
            width="stretch",
            hide_index=True,
        )


def _render_t1_t3_order_panel(records: list[WelltrackRecord]) -> None:
    with st.container(border=True):
        st.markdown("### Проверка порядка t1/t3")
        resolution_message = _t1_t3_order_resolution_message()
        if resolution_message is not None:
            level, message = resolution_message
            if level == "success":
                st.success(message)
            else:
                st.info(message)
        detected_issues = detect_t1_t3_order_issues(
            records, min_delta_m=WT_T1T3_MIN_DELTA_M
        )
        acknowledged_well_names = _t1_t3_order_acknowledged_well_names()
        issues = [
            item
            for item in detected_issues
            if str(item.well_name) not in acknowledged_well_names
        ]
        if not issues:
            if detected_issues:
                st.info(
                    "Активных предупреждений по порядку `t1/t3` нет. "
                    "Для отмеченных скважин текущий порядок оставлен без изменений."
                )
            else:
                st.success("Проверка порядка t1/t3 — OK.")
            return

        st.warning(
            "Найдены скважины, где отход до t1 больше, чем до t3. "
            "Вероятно, порядок точек t1/t3 перепутан."
        )
        header_cols = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 3.0], gap="small")
        header_cols[0].markdown(
            "<div style='text-align: center;'><strong>Исправить</strong></div>",
            unsafe_allow_html=True,
        )
        header_cols[1].markdown("**Скважина**")
        header_cols[2].markdown("**Отход S→t1, м**")
        header_cols[3].markdown("**Отход S→t3, м**")
        header_cols[4].markdown("**Δ (t1 - t3), м**")
        st.markdown(
            "<div style='height: 0.25rem;'></div>",
            unsafe_allow_html=True,
        )
        for item in issues:
            well_name = str(item.well_name)
            checkbox_key = f"wt_t1_t3_fix_{well_name}"
            st.session_state.setdefault(checkbox_key, True)
            row_cols = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 3.0], gap="small")
            checkbox_cols = row_cols[0].columns([1.0, 0.8, 1.0], gap="small")
            checkbox_cols[1].checkbox(
                f"Исправить {well_name}",
                key=checkbox_key,
                label_visibility="collapsed",
            )
            row_cols[1].markdown(f"`{well_name}`")
            row_cols[2].markdown(f"{float(item.t1_offset_m):.2f}")
            row_cols[3].markdown(f"{float(item.t3_offset_m):.2f}")
            row_cols[4].markdown(f"{float(item.delta_m):.2f}")

        fix_col, keep_col = st.columns(2, gap="small")
        if fix_col.button(
            "Исправить порядок для выбранных скважин",
            type="primary",
            icon=":material/swap_horiz:",
            width="stretch",
        ):
            target_names = {
                str(item.well_name)
                for item in issues
                if bool(
                    st.session_state.get(
                        f"wt_t1_t3_fix_{str(item.well_name)}", True
                    )
                )
            }
            if not target_names:
                st.warning(
                    "Выберите хотя бы одну скважину для исправления порядка t1/t3."
                )
                return
            st.session_state["wt_records"] = swap_t1_t3_for_wells(
                records=list(records),
                well_names=target_names,
            )
            original_records = st.session_state.get("wt_records_original")
            if original_records is None:
                original_records = list(records)
            st.session_state["wt_records_original"] = swap_t1_t3_for_wells(
                records=list(original_records),
                well_names=target_names,
            )
            _set_t1_t3_order_acknowledged_well_names(
                _t1_t3_order_acknowledged_well_names() - target_names
            )
            _set_t1_t3_order_resolution(
                action="fixed",
                well_names=target_names,
            )
            _clear_results()
            st.toast(
                f"Порядок t1/t3 исправлен для {len(target_names)} скважин."
            )
            st.rerun()
        if keep_col.button(
            "Оставить все точки без изменений",
            icon=":material/do_not_disturb:",
            width="stretch",
        ):
            kept_well_names = {str(item.well_name) for item in issues}
            _set_t1_t3_order_acknowledged_well_names(
                _t1_t3_order_acknowledged_well_names() | kept_well_names
            )
            _set_t1_t3_order_resolution(
                action="kept",
                well_names=kept_well_names,
            )
            st.toast("Текущий порядок t1/t3 оставлен без изменений.")
            st.rerun()
        st.caption(
            "Исправление меняет местами координаты `t1` и `t3`, но сохраняет MD "
            "во 2-й и 3-й позиции, чтобы не ломать порядок MD."
        )


def _clear_t1_t3_order_resolution_state() -> None:
    st.session_state["wt_t1_t3_last_resolution"] = None
    st.session_state["wt_t1_t3_acknowledged_well_names"] = ()
    for key in list(st.session_state.keys()):
        if str(key).startswith("wt_t1_t3_fix_"):
            del st.session_state[key]


def _set_t1_t3_order_acknowledged_well_names(
    well_names: set[str] | list[str] | tuple[str, ...],
) -> None:
    st.session_state["wt_t1_t3_acknowledged_well_names"] = tuple(
        sorted(str(name) for name in well_names if str(name).strip())
    )


def _t1_t3_order_acknowledged_well_names() -> set[str]:
    raw_value = st.session_state.get("wt_t1_t3_acknowledged_well_names")
    if not isinstance(raw_value, (tuple, list, set)):
        return set()
    return {str(item) for item in raw_value if str(item).strip()}


def _set_t1_t3_order_resolution(
    *,
    action: str,
    well_names: set[str] | list[str] | tuple[str, ...],
) -> None:
    st.session_state["wt_t1_t3_last_resolution"] = {
        "action": str(action),
        "well_names": tuple(sorted(str(name) for name in well_names)),
    }


def _t1_t3_order_resolution_message() -> tuple[str, str] | None:
    resolution = st.session_state.get("wt_t1_t3_last_resolution")
    if not isinstance(resolution, dict):
        return None
    action = str(resolution.get("action", "")).strip()
    well_names = tuple(
        str(item) for item in (resolution.get("well_names") or ())
    )
    if not well_names:
        return None
    joined_names = ", ".join(well_names)
    if action == "fixed":
        return (
            "success",
            f"Порядок t1/t3 изменился для скважин: {joined_names}.",
        )
    if action == "kept":
        return (
            "info",
            f"Порядок t1/t3 оставлен без изменений для скважин: {joined_names}.",
        )
    return None


def _pad_config_defaults(pad: WellPad) -> dict[str, float | str]:
    return {
        "spacing_m": float(DEFAULT_PAD_SPACING_M),
        "nds_azimuth_deg": float(
            estimate_pad_nds_azimuth_deg(
                wells=pad.wells,
                surface_x=float(pad.surface.x),
                surface_y=float(pad.surface.y),
                surface_anchor_mode=DEFAULT_PAD_SURFACE_ANCHOR_MODE,
            )
        ),
        "first_surface_x": float(pad.surface.x),
        "first_surface_y": float(pad.surface.y),
        "first_surface_z": float(pad.surface.z),
        "surface_anchor_mode": DEFAULT_PAD_SURFACE_ANCHOR_MODE,
    }


def _source_surface_xyz(
    record: WelltrackRecord,
) -> tuple[float, float, float] | None:
    points = tuple(record.points)
    if not points:
        return None
    surface = points[0]
    return float(surface.x), float(surface.y), float(surface.z)


def _record_midpoint_xyz(
    record: WelltrackRecord,
) -> tuple[float, float, float]:
    points = tuple(record.points)
    if len(points) >= 3:
        try:
            _, t1, t3 = welltrack_points_to_targets(tuple(points[:3]))
            return (
                float(0.5 * (t1.x + t3.x)),
                float(0.5 * (t1.y + t3.y)),
                float(0.5 * (t1.z + t3.z)),
            )
        except (TypeError, ValueError):
            pass
    surface_xyz = _source_surface_xyz(record)
    return surface_xyz or (0.0, 0.0, 0.0)


def _estimate_surface_pad_axis_deg(
    surface_xyzs: list[tuple[float, float, float]],
) -> float:
    if len(surface_xyzs) <= 1:
        return 0.0
    points = np.asarray(
        [(item[0], item[1]) for item in surface_xyzs], dtype=float
    )
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_index = int(np.argmax(np.asarray(eigenvalues, dtype=float)))
        direction = np.asarray(eigenvectors[:, principal_index], dtype=float)
    except np.linalg.LinAlgError:
        direction = np.asarray([1.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= SMALL:
        return 0.0
    unit = direction / norm
    return float(
        np.degrees(np.arctan2(float(unit[0]), float(unit[1]))) % 360.0
    )


def _inferred_surface_spacing_m(
    *,
    surface_xyzs: list[tuple[float, float, float]],
    nds_azimuth_deg: float,
) -> float:
    if len(surface_xyzs) <= 1:
        return 0.0
    angle_rad = np.deg2rad(float(nds_azimuth_deg) % 360.0)
    ux = float(np.sin(angle_rad))
    uy = float(np.cos(angle_rad))
    projections = sorted(
        float(x) * ux + float(y) * uy for x, y, _ in surface_xyzs
    )
    diffs = [
        float(right - left)
        for left, right in zip(projections, projections[1:], strict=False)
        if float(right - left) > 1e-6
    ]
    if not diffs:
        return 0.0
    return float(np.median(np.asarray(diffs, dtype=float)))


def _detect_ui_pads(
    records: list[WelltrackRecord],
) -> tuple[list[WellPad], dict[str, _DetectedPadUiMeta]]:
    indexed_records = [
        (index, record, _source_surface_xyz(record))
        for index, record in enumerate(records)
    ]
    indexed_records = [
        (index, record, surface_xyz)
        for index, record, surface_xyz in indexed_records
        if surface_xyz is not None
    ]
    if not indexed_records:
        return [], {}

    adjacency: dict[int, set[int]] = {
        index: set() for index, _, _ in indexed_records
    }
    for left_pos, (left_index, _, left_surface) in enumerate(indexed_records):
        for right_index, _, right_surface in indexed_records[left_pos + 1 :]:
            distance_xy = float(
                np.hypot(
                    float(left_surface[0]) - float(right_surface[0]),
                    float(left_surface[1]) - float(right_surface[1]),
                )
            )
            if distance_xy <= WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M + SMALL:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    by_index = {
        index: (index, record, surface_xyz)
        for index, record, surface_xyz in indexed_records
    }
    clusters: list[
        list[tuple[int, WelltrackRecord, tuple[float, float, float]]]
    ] = []
    visited: set[int] = set()
    for index, _, _ in indexed_records:
        if index in visited:
            continue
        queue = [index]
        visited.add(index)
        cluster: list[
            tuple[int, WelltrackRecord, tuple[float, float, float]]
        ] = []
        while queue:
            current = queue.pop()
            cluster.append(by_index[current])
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        clusters.append(cluster)

    prepared: list[
        tuple[
            float,
            float,
            float,
            list[tuple[int, WelltrackRecord, tuple[float, float, float]]],
        ]
    ] = []
    for cluster in clusters:
        surface_xyzs = [surface_xyz for _, _, surface_xyz in cluster]
        center_x = float(np.mean([item[0] for item in surface_xyzs]))
        center_y = float(np.mean([item[1] for item in surface_xyzs]))
        center_z = float(np.mean([item[2] for item in surface_xyzs]))
        prepared.append((center_x, center_y, center_z, cluster))
    prepared.sort(key=lambda item: (item[0], item[1], item[2]))

    pads: list[WellPad] = []
    metadata: dict[str, _DetectedPadUiMeta] = {}
    for index, (center_x, center_y, center_z, cluster) in enumerate(
        prepared, start=1
    ):
        wells: list[PadWell] = []
        surface_xyzs: list[tuple[float, float, float]] = []
        unique_surface_keys: set[tuple[int, int, int]] = set()
        for record_index, record, surface_xyz in cluster:
            midpoint_x, midpoint_y, midpoint_z = _record_midpoint_xyz(record)
            wells.append(
                PadWell(
                    name=str(record.name),
                    record_index=int(record_index),
                    midpoint_x=float(midpoint_x),
                    midpoint_y=float(midpoint_y),
                    midpoint_z=float(midpoint_z),
                )
            )
            surface_xyzs.append(surface_xyz)
            unique_surface_keys.add(
                (
                    round(float(surface_xyz[0]), 6),
                    round(float(surface_xyz[1]), 6),
                    round(float(surface_xyz[2]), 6),
                )
            )
        pad_id = f"PAD-{index:02d}"
        source_surfaces_defined = len(unique_surface_keys) > 1
        auto_nds = _estimate_surface_pad_axis_deg(surface_xyzs)
        if abs(float(auto_nds)) <= SMALL:
            auto_nds = estimate_pad_nds_azimuth_deg(
                wells=tuple(wells),
                surface_x=float(center_x),
                surface_y=float(center_y),
                surface_anchor_mode=DEFAULT_PAD_SURFACE_ANCHOR_MODE,
            )
        pads.append(
            WellPad(
                pad_id=pad_id,
                surface=Point3D(
                    x=float(center_x), y=float(center_y), z=float(center_z)
                ),
                wells=tuple(wells),
                auto_nds_azimuth_deg=float(auto_nds),
            )
        )
        metadata[pad_id] = _DetectedPadUiMeta(
            source_surfaces_defined=bool(source_surfaces_defined),
            inferred_spacing_m=_inferred_surface_spacing_m(
                surface_xyzs=surface_xyzs,
                nds_azimuth_deg=float(auto_nds),
            ),
            source_surface_x_m=float(center_x),
            source_surface_y_m=float(center_y),
            source_surface_z_m=float(center_z),
            source_surface_count=len(surface_xyzs),
        )
    return pads, metadata


def _ensure_pad_configs(base_records: list[WelltrackRecord]) -> list[WellPad]:
    pads, metadata = _detect_ui_pads(base_records)
    existing = st.session_state.get("wt_pad_configs", {})
    merged: dict[str, dict[str, float]] = {}
    for pad in pads:
        defaults = _pad_config_defaults(pad)
        pad_meta = metadata.get(str(pad.pad_id))
        if isinstance(pad_meta, _DetectedPadUiMeta) and bool(
            pad_meta.source_surfaces_defined
        ):
            defaults = {
                "spacing_m": float(max(pad_meta.inferred_spacing_m, 0.0)),
                "nds_azimuth_deg": float(pad.auto_nds_azimuth_deg) % 360.0,
                "first_surface_x": float(pad_meta.source_surface_x_m),
                "first_surface_y": float(pad_meta.source_surface_y_m),
                "first_surface_z": float(pad_meta.source_surface_z_m),
                "surface_anchor_mode": DEFAULT_PAD_SURFACE_ANCHOR_MODE,
            }
        current = existing.get(str(pad.pad_id), {})
        merged[str(pad.pad_id)] = {
            "spacing_m": float(
                current.get("spacing_m", defaults["spacing_m"])
            ),
            "nds_azimuth_deg": float(
                current.get("nds_azimuth_deg", defaults["nds_azimuth_deg"])
            )
            % 360.0,
            "first_surface_x": float(
                current.get("first_surface_x", defaults["first_surface_x"])
            ),
            "first_surface_y": float(
                current.get("first_surface_y", defaults["first_surface_y"])
            ),
            "first_surface_z": float(
                current.get("first_surface_z", defaults["first_surface_z"])
            ),
            "surface_anchor_mode": str(
                current.get(
                    "surface_anchor_mode", defaults["surface_anchor_mode"]
                )
            ),
        }
        if isinstance(pad_meta, _DetectedPadUiMeta) and bool(
            pad_meta.source_surfaces_defined
        ):
            merged[str(pad.pad_id)] = dict(defaults)
    st.session_state["wt_pad_configs"] = merged
    st.session_state["wt_pad_detected_meta"] = metadata

    pad_ids = [str(pad.pad_id) for pad in pads]
    if not pad_ids:
        st.session_state["wt_pad_selected_id"] = ""
        return pads
    if str(st.session_state.get("wt_pad_selected_id", "")) not in pad_ids:
        st.session_state["wt_pad_selected_id"] = pad_ids[0]
    return pads


def _build_pad_plan_map(pads: list[WellPad]) -> dict[str, PadLayoutPlan]:
    config_map = st.session_state.get("wt_pad_configs", {})
    metadata = dict(st.session_state.get("wt_pad_detected_meta", {}))
    plan_map: dict[str, PadLayoutPlan] = {}
    for pad in pads:
        pad_id = str(pad.pad_id)
        pad_meta = metadata.get(pad_id)
        if isinstance(pad_meta, _DetectedPadUiMeta) and bool(
            pad_meta.source_surfaces_defined
        ):
            continue
        cfg = config_map.get(pad_id, _pad_config_defaults(pad))
        plan_map[pad_id] = PadLayoutPlan(
            pad_id=pad_id,
            first_surface_x=float(cfg["first_surface_x"]),
            first_surface_y=float(cfg["first_surface_y"]),
            first_surface_z=float(cfg["first_surface_z"]),
            spacing_m=float(max(cfg["spacing_m"], 0.0)),
            nds_azimuth_deg=float(cfg["nds_azimuth_deg"]) % 360.0,
            surface_anchor_mode=str(
                cfg.get("surface_anchor_mode", DEFAULT_PAD_SURFACE_ANCHOR_MODE)
            ),
        )
    return plan_map


def _project_pads_for_ui(records: list[WelltrackRecord]) -> list[WellPad]:
    base_records = st.session_state.get("wt_records_original")
    source_records = (
        list(base_records)
        if isinstance(base_records, list) and base_records
        else list(records)
    )
    pads, metadata = _detect_ui_pads(source_records)
    st.session_state["wt_pad_detected_meta"] = metadata
    return pads


def _pad_display_label(pad: WellPad) -> str:
    return f"{str(pad.pad_id)} · {int(len(pad.wells))} скв."


def _pad_config_for_ui(pad: WellPad) -> dict[str, float | str]:
    defaults = _pad_config_defaults(pad)
    pad_meta = dict(st.session_state.get("wt_pad_detected_meta", {})).get(
        str(pad.pad_id)
    )
    if isinstance(pad_meta, _DetectedPadUiMeta) and bool(
        pad_meta.source_surfaces_defined
    ):
        defaults = {
            "spacing_m": float(max(pad_meta.inferred_spacing_m, 0.0)),
            "nds_azimuth_deg": float(pad.auto_nds_azimuth_deg) % 360.0,
            "first_surface_x": float(pad_meta.source_surface_x_m),
            "first_surface_y": float(pad_meta.source_surface_y_m),
            "first_surface_z": float(pad_meta.source_surface_z_m),
            "surface_anchor_mode": DEFAULT_PAD_SURFACE_ANCHOR_MODE,
        }
    current = dict(st.session_state.get("wt_pad_configs", {})).get(
        str(pad.pad_id),
        {},
    )
    if isinstance(pad_meta, _DetectedPadUiMeta) and bool(
        pad_meta.source_surfaces_defined
    ):
        current = {}
    return {
        "spacing_m": float(current.get("spacing_m", defaults["spacing_m"])),
        "nds_azimuth_deg": float(
            current.get("nds_azimuth_deg", defaults["nds_azimuth_deg"])
        )
        % 360.0,
        "first_surface_x": float(
            current.get("first_surface_x", defaults["first_surface_x"])
        ),
        "first_surface_y": float(
            current.get("first_surface_y", defaults["first_surface_y"])
        ),
        "first_surface_z": float(
            current.get("first_surface_z", defaults["first_surface_z"])
        ),
        "surface_anchor_mode": str(
            current.get("surface_anchor_mode", defaults["surface_anchor_mode"])
        ),
    }


def _pad_anchor_mode_label(mode: object) -> str:
    if str(mode) == PAD_SURFACE_ANCHOR_CENTER:
        return "Центр куста"
    return "S первой скважины"


def _pad_membership(
    records: list[WelltrackRecord],
) -> tuple[list[WellPad], dict[str, str], dict[str, tuple[str, ...]]]:
    pads = _project_pads_for_ui(records)
    name_to_pad_id: dict[str, str] = {}
    well_names_by_pad_id: dict[str, tuple[str, ...]] = {}
    for pad in pads:
        pad_id = str(pad.pad_id)
        ordered = ordered_pad_wells(
            pad=pad,
            nds_azimuth_deg=float(_pad_config_for_ui(pad)["nds_azimuth_deg"]),
        )
        ordered_names = tuple(str(item.name) for item in ordered)
        well_names_by_pad_id[pad_id] = ordered_names
        for well_name in ordered_names:
            name_to_pad_id[well_name] = pad_id
    return pads, name_to_pad_id, well_names_by_pad_id


def _normalize_focus_pad_id(
    *,
    records: list[WelltrackRecord],
    requested_pad_id: str | None,
) -> str:
    pads, _, _ = _pad_membership(records)
    valid_options = {WT_PAD_FOCUS_ALL, *(str(pad.pad_id) for pad in pads)}
    selected = str(requested_pad_id or "").strip()
    if not selected or selected not in valid_options:
        if len(pads) == 1:
            return str(pads[0].pad_id)
        return WT_PAD_FOCUS_ALL
    if selected == WT_PAD_FOCUS_ALL and len(pads) == 1:
        return str(pads[0].pad_id)
    return selected


def _focus_pad_well_names(
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str | None,
) -> tuple[str, ...]:
    normalized = _normalize_focus_pad_id(
        records=records, requested_pad_id=focus_pad_id
    )
    if normalized == WT_PAD_FOCUS_ALL:
        return ()
    _, _, well_names_by_pad_id = _pad_membership(records)
    return tuple(well_names_by_pad_id.get(str(normalized), ()))


def _clusters_touching_focus_pad(
    *,
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
    focus_pad_well_names: tuple[str, ...],
) -> tuple[AntiCollisionRecommendationCluster, ...]:
    focus_set = {
        str(name) for name in focus_pad_well_names if str(name).strip()
    }
    if not focus_set:
        return tuple(clusters)
    return tuple(
        cluster
        for cluster in clusters
        if focus_set.intersection(str(name) for name in cluster.well_names)
    )


def _recommendations_for_clusters(
    *,
    recommendations: tuple[AntiCollisionRecommendation, ...],
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
) -> tuple[AntiCollisionRecommendation, ...]:
    visible_ids = {
        str(item.recommendation_id)
        for cluster in clusters
        for item in cluster.recommendations
    }
    if not visible_ids:
        return ()
    return tuple(
        item
        for item in recommendations
        if str(item.recommendation_id) in visible_ids
    )


def _report_rows_from_recommendations(
    recommendations: tuple[AntiCollisionRecommendation, ...],
    analysis: AntiCollisionAnalysis | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in recommendations:
        # Extract segment types if analysis is provided
        segment_a = "—"
        segment_b = "—"
        if analysis is not None:
            from pywp.anticollision import _segment_types_for_interval

            segment_a = _segment_types_for_interval(
                analysis,
                str(item.well_a),
                float(item.md_a_start_m),
                float(item.md_a_end_m),
            )
            segment_b = _segment_types_for_interval(
                analysis,
                str(item.well_b),
                float(item.md_b_start_m),
                float(item.md_b_end_m),
            )
        rows.append(
            {
                "Приоритет": str(item.priority_rank),
                "Скважина A": str(item.well_a),
                "Скважина B": str(item.well_b),
                "Участок A": segment_a,
                "Участок B": segment_b,
                "Интервал A, м": _md_interval_label(
                    float(item.md_a_start_m),
                    float(item.md_a_end_m),
                ),
                "Интервал B, м": _md_interval_label(
                    float(item.md_b_start_m),
                    float(item.md_b_end_m),
                ),
                "SF min": float(item.min_separation_factor),
                "Overlap max, м": float(item.max_overlap_depth_m),
                "Мин. расстояние, м": float(item.min_center_distance_m),
            }
        )
    return rows


def _pad_scoped_cluster_target_well_names(
    *,
    cluster: AntiCollisionRecommendationCluster,
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    focus_set = {
        str(name) for name in focus_pad_well_names if str(name).strip()
    }
    focused_cluster = tuple(
        str(name) for name in cluster.well_names if str(name) in focus_set
    )
    cluster_scope = (
        tuple(str(name) for name in cluster.well_names)
        if cluster.well_names
        else tuple(str(name) for name in cluster.affected_wells)
    )
    if not focused_cluster:
        return cluster_scope
    ordered_scope: list[str] = []
    for well_name in [*focused_cluster, *cluster_scope]:
        normalized = str(well_name).strip()
        if normalized and normalized not in ordered_scope:
            ordered_scope.append(normalized)
    return tuple(ordered_scope)


def _pad_scoped_cluster_focus_well_names(
    *,
    cluster: AntiCollisionRecommendationCluster,
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    focus_set = {
        str(name) for name in focus_pad_well_names if str(name).strip()
    }
    if not focus_set:
        return ()
    focused_affected = tuple(
        str(name) for name in cluster.affected_wells if str(name) in focus_set
    )
    if focused_affected:
        return focused_affected
    return tuple(
        str(name) for name in cluster.well_names if str(name) in focus_set
    )


def _anticollision_focus_well_names(
    *,
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    focus_set = {
        str(name) for name in focus_pad_well_names if str(name).strip()
    }
    if not focus_set:
        return ()
    related = set(focus_set)
    for cluster in clusters:
        if focus_set.intersection(str(name) for name in cluster.well_names):
            related.update(str(name) for name in cluster.well_names)
            related.update(str(name) for name in cluster.affected_wells)
    return tuple(sorted(related))


def _render_pad_layout_panel(records: list[WelltrackRecord]) -> None:
    base_records = st.session_state.get("wt_records_original")
    if base_records is None:
        base_records = list(records)
    pads = _ensure_pad_configs(base_records=list(base_records))
    if not pads:
        return

    with st.container(border=True):
        st.markdown("### Кусты и расчет устьев")
        if bool(st.session_state.get("wt_pad_auto_applied_on_import", False)):
            st.info(
                "После импорта исходные устья скважин совпадали, "
                "поэтому текущие координаты устьев были автоматически скорректированы. "
                "Если нужно вернуться к исходным устьям, нажмите "
                "'Вернуть исходные устья'."
            )
        st.caption(
            f"Из исходных данных WELLTRACK / точек целей было определено кустов: {len(pads)}. "
            "Их параметры показаны в таблице ниже."
        )
        pad_metadata = dict(st.session_state.get("wt_pad_detected_meta", {}))
        pad_rows = [
            {
                "Куст": str(pad.pad_id),
                "Скважин": int(len(pad.wells)),
                "Устья заданы": (
                    "Да"
                    if bool(
                        getattr(
                            pad_metadata.get(str(pad.pad_id)),
                            "source_surfaces_defined",
                            False,
                        )
                    )
                    else "Нет"
                ),
                "Авто НДС, deg": float(
                    _pad_config_for_ui(pad)["nds_azimuth_deg"]
                ),
                "S X, м": float(_pad_config_for_ui(pad)["first_surface_x"]),
                "S Y, м": float(_pad_config_for_ui(pad)["first_surface_y"]),
                "S Z, м": float(_pad_config_for_ui(pad)["first_surface_z"]),
            }
            for pad in pads
        ]
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(pad_rows)),
            width="stretch",
            hide_index=True,
        )

        pad_ids = [str(pad.pad_id) for pad in pads]
        st.selectbox(
            "Выберите куст", options=pad_ids, key="wt_pad_selected_id"
        )
        selected_id = str(
            st.session_state.get("wt_pad_selected_id", pad_ids[0])
        )
        selected_pad = next(
            (pad for pad in pads if str(pad.pad_id) == selected_id), pads[0]
        )
        selected_pad_meta = pad_metadata.get(selected_id)
        source_surfaces_defined = bool(
            getattr(selected_pad_meta, "source_surfaces_defined", False)
        )
        config_map = st.session_state.get("wt_pad_configs", {})
        selected_cfg = dict(
            config_map.get(selected_id, _pad_config_defaults(selected_pad))
        )
        previous_anchor_mode = str(
            selected_cfg.get(
                "surface_anchor_mode", DEFAULT_PAD_SURFACE_ANCHOR_MODE
            )
        )

        widget_keys = {
            "spacing_m": f"wt_pad_cfg_spacing_m_{selected_id}",
            "nds_azimuth_deg": f"wt_pad_cfg_nds_azimuth_deg_{selected_id}",
            "first_surface_x": f"wt_pad_cfg_first_surface_x_{selected_id}",
            "first_surface_y": f"wt_pad_cfg_first_surface_y_{selected_id}",
            "first_surface_z": f"wt_pad_cfg_first_surface_z_{selected_id}",
            "surface_anchor_center": f"wt_pad_cfg_surface_anchor_center_{selected_id}",
        }
        for field, widget_key in widget_keys.items():
            if field == "surface_anchor_center":
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = (
                        previous_anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                    )
                continue
            if widget_key not in st.session_state:
                st.session_state[widget_key] = float(selected_cfg[field])

        if source_surfaces_defined:
            st.info(
                "Положения устьев были заданы в исходных данных. Для этого куста "
                "параметры ниже показаны справочно и не редактируются."
            )

        anchor_center = st.toggle(
            "Координата куста = центр расстановки",
            key=widget_keys["surface_anchor_center"],
            help=(
                "Включено: введённые координаты устьев принимаются как центр куста. "
                "Выключено: координаты устьев задают первую скважину на кусте."
            ),
            disabled=source_surfaces_defined,
        )
        anchor_mode = (
            PAD_SURFACE_ANCHOR_CENTER
            if bool(anchor_center)
            else PAD_SURFACE_ANCHOR_FIRST
        )
        if anchor_mode != previous_anchor_mode:
            previous_auto_nds = estimate_pad_nds_azimuth_deg(
                wells=selected_pad.wells,
                surface_x=float(selected_pad.surface.x),
                surface_y=float(selected_pad.surface.y),
                surface_anchor_mode=previous_anchor_mode,
            )
            current_nds = float(
                st.session_state.get(
                    widget_keys["nds_azimuth_deg"],
                    selected_cfg["nds_azimuth_deg"],
                )
            )
            if abs(current_nds - previous_auto_nds) <= 1e-6:
                st.session_state[widget_keys["nds_azimuth_deg"]] = float(
                    estimate_pad_nds_azimuth_deg(
                        wells=selected_pad.wells,
                        surface_x=float(selected_pad.surface.x),
                        surface_y=float(selected_pad.surface.y),
                        surface_anchor_mode=anchor_mode,
                    )
                )

        p1, p2, p3, p4, p5 = st.columns(5, gap="small")
        spacing_m = p1.number_input(
            "Расстояние между устьями, м",
            min_value=0.0,
            step=1.0,
            key=widget_keys["spacing_m"],
            help="Шаг по кусту между соседними устьями скважин.",
            disabled=source_surfaces_defined,
        )
        nds_azimuth_deg = p2.number_input(
            "НДС (азимут), deg",
            min_value=0.0,
            max_value=360.0,
            step=0.5,
            key=widget_keys["nds_azimuth_deg"],
            help="Направление движения станка по кусту.",
            disabled=source_surfaces_defined,
        )
        first_surface_x = p3.number_input(
            (
                "S куста X (East), м"
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                else "S1 X (East), м"
            ),
            step=10.0,
            key=widget_keys["first_surface_x"],
            disabled=source_surfaces_defined,
        )
        first_surface_y = p4.number_input(
            (
                "S куста Y (North), м"
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                else "S1 Y (North), м"
            ),
            step=10.0,
            key=widget_keys["first_surface_y"],
            disabled=source_surfaces_defined,
        )
        first_surface_z = p5.number_input(
            (
                "S куста Z (TVD), м"
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                else "S1 Z (TVD), м"
            ),
            step=10.0,
            key=widget_keys["first_surface_z"],
            disabled=source_surfaces_defined,
        )

        selected_cfg["spacing_m"] = float(max(spacing_m, 0.0))
        selected_cfg["nds_azimuth_deg"] = float(nds_azimuth_deg) % 360.0
        selected_cfg["first_surface_x"] = float(first_surface_x)
        selected_cfg["first_surface_y"] = float(first_surface_y)
        selected_cfg["first_surface_z"] = float(first_surface_z)
        selected_cfg["surface_anchor_mode"] = anchor_mode
        config_map[selected_id] = selected_cfg
        st.session_state["wt_pad_configs"] = config_map

        ordered_wells = ordered_pad_wells(
            pad=selected_pad,
            nds_azimuth_deg=float(selected_cfg["nds_azimuth_deg"]),
        )
        angle_rad = np.deg2rad(float(selected_cfg["nds_azimuth_deg"]))
        ux = float(np.sin(angle_rad))
        uy = float(np.cos(angle_rad))
        center_slot_index = 0.5 * float(max(len(ordered_wells) - 1, 0))
        preview_rows: list[dict[str, object]] = []
        for slot_index, well in enumerate(ordered_wells, start=1):
            row = {
                "Порядок": int(slot_index),
                "Скважина": str(well.name),
                "Середина t1-t3 X, м": float(well.midpoint_x),
                "Середина t1-t3 Y, м": float(well.midpoint_y),
                "Опора S": _pad_anchor_mode_label(anchor_mode),
            }
            if source_surfaces_defined:
                source_record = next(
                    (
                        item
                        for item in base_records
                        if str(item.name) == str(well.name)
                    ),
                    None,
                )
                source_surface = (
                    _source_surface_xyz(source_record)
                    if source_record is not None
                    else None
                )
                row["Текущее S X, м"] = (
                    None
                    if source_surface is None
                    else float(source_surface[0])
                )
                row["Текущее S Y, м"] = (
                    None
                    if source_surface is None
                    else float(source_surface[1])
                )
                row["Текущее S Z, м"] = (
                    None
                    if source_surface is None
                    else float(source_surface[2])
                )
            else:
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER:
                    shift_m = (
                        float(slot_index - 1) - center_slot_index
                    ) * float(selected_cfg["spacing_m"])
                else:
                    shift_m = float(slot_index - 1) * float(
                        selected_cfg["spacing_m"]
                    )
                row["Новое S X, м"] = float(
                    selected_cfg["first_surface_x"] + shift_m * ux
                )
                row["Новое S Y, м"] = float(
                    selected_cfg["first_surface_y"] + shift_m * uy
                )
                row["Новое S Z, м"] = float(selected_cfg["first_surface_z"])
            preview_rows.append(row)
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(preview_rows)),
            width="stretch",
            hide_index=True,
        )

        a1, a2 = st.columns(2, gap="small")
        apply_clicked = a1.button(
            "Рассчитать устья скважин",
            type="primary",
            icon=":material/tune:",
            width="stretch",
            help=(
                "Обновляет координаты первой точки S для скважин по выбранным "
                "параметрам кустов. Последующие расчеты будут использовать новые устья."
            ),
            disabled=source_surfaces_defined,
        )
        reset_clicked = a2.button(
            "Вернуть исходные устья",
            icon=":material/restart_alt:",
            width="stretch",
            disabled=source_surfaces_defined,
        )

        if apply_clicked:
            plan_map = _build_pad_plan_map(pads)
            updated_records = apply_pad_layout(
                records=list(base_records),
                pads=pads,
                plan_by_pad_id=plan_map,
            )
            st.session_state["wt_records"] = list(updated_records)
            st.session_state["wt_pad_last_applied_at"] = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            st.session_state["wt_pad_auto_applied_on_import"] = False
            _clear_results()
            st.toast("Координаты устьев обновлены по параметрам кустов.")
            st.rerun()

        if reset_clicked:
            st.session_state["wt_records"] = list(base_records)
            st.session_state["wt_pad_last_applied_at"] = ""
            st.session_state["wt_pad_auto_applied_on_import"] = False
            _clear_results()
            st.rerun()

        if str(st.session_state.get("wt_pad_last_applied_at", "")):
            st.caption(
                f"Последнее обновление устьев: {st.session_state['wt_pad_last_applied_at']}"
            )


def _sync_selection_state(
    records: list[WelltrackRecord],
) -> tuple[list[str], list[str]]:
    all_names = [record.name for record in records]
    recommended_names = recommended_batch_selection(
        records=records,
        summary_rows=st.session_state.get("wt_summary_rows"),
    )
    pending_general = st.session_state.pop("wt_pending_selected_names", None)
    if pending_general is not None:
        st.session_state["wt_selected_names"] = [
            name for name in pending_general if name in all_names
        ]

    def _sync_key(key: str) -> None:
        current = [
            name for name in st.session_state.get(key, []) if name in all_names
        ]
        if current != st.session_state.get(key, []):
            st.session_state[key] = list(current)
        if not current and recommended_names:
            st.session_state[key] = list(recommended_names)

    _sync_key("wt_selected_names")
    return all_names, recommended_names


def _render_batch_selection_status(
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]] | None,
) -> None:
    if summary_rows:
        st.caption(
            "Результаты по невыбранным скважинам сохраняются. Для следующего запуска "
            "по умолчанию выделяются нерассчитанные, ошибочные и warning-кейсы."
        )
        return

    all_names = [record.name for record in records]
    rows_by_name = {
        str(row.get("Скважина", "")).strip(): row
        for row in (summary_rows or [])
    }
    ok_count = 0
    warning_count = 0
    error_count = 0
    not_run_count = 0
    for name in all_names:
        row = rows_by_name.get(name)
        if row is None:
            not_run_count += 1
            continue
        status = str(row.get("Статус", "")).strip()
        problem_text = str(row.get("Проблема", "")).strip()
        if status == "OK":
            if problem_text:
                warning_count += 1
            else:
                ok_count += 1
            continue
        if status == "Не рассчитана":
            not_run_count += 1
        else:
            error_count += 1

    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Без замечаний", f"{ok_count}")
    c2.metric("С предупреждениями", f"{warning_count}")
    c3.metric("С ошибками", f"{error_count}")
    c4.metric("Не рассчитаны", f"{not_run_count}")
    st.caption(
        "До первого запуска предвыбраны все скважины. Затем страница будет "
        "автоматически фокусироваться на нерассчитанных и проблемных скважинах."
    )


def _render_batch_run_forms(
    *,
    records: list[WelltrackRecord],
    all_names: list[str],
) -> list[_BatchRunRequest]:
    st.markdown("### Пакетный расчет")
    summary_rows = st.session_state.get("wt_summary_rows")
    _render_batch_selection_status(records=records, summary_rows=summary_rows)
    prepared_rows = _prepared_override_rows()
    if prepared_rows:
        prepared_snapshot = dict(
            st.session_state.get("wt_prepared_recommendation_snapshot") or {}
        )
        prepared_kind_label = _prepared_plan_kind_label(prepared_snapshot)
        info_col, action_col = st.columns([6.0, 1.4], gap="small")
        with info_col:
            st.info(
                str(
                    st.session_state.get("wt_prepared_override_message", "")
                    or "Подготовлен пересчет по anti-collision рекомендации."
                )
            )
            st.caption(
                "Сейчас активен prepared plan: "
                f"{prepared_kind_label}. При запуске ниже он будет применен только "
                "к отмеченным скважинам из таблицы overrides."
            )
        with action_col:
            if st.button(
                "Очистить план",
                icon=":material/close:",
                width="stretch",
            ):
                st.session_state["wt_prepared_well_overrides"] = {}
                st.session_state["wt_prepared_override_message"] = ""
                st.session_state["wt_prepared_recommendation_id"] = ""
                st.session_state["wt_anticollision_prepared_cluster_id"] = ""
                st.session_state["wt_prepared_recommendation_snapshot"] = None
                st.rerun()
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(prepared_rows)),
            width="stretch",
            hide_index=True,
            column_config={
                "Порядок": st.column_config.TextColumn("Порядок"),
                "Скважина": st.column_config.TextColumn("Скважина"),
                "Маневр": st.column_config.TextColumn("Маневр"),
                "Оптимизация": st.column_config.TextColumn("Оптимизация"),
                "SF до": st.column_config.TextColumn("SF до"),
                "Источник": st.column_config.TextColumn("Источник"),
                "Причина": st.column_config.TextColumn("Причина"),
            },
        )
        st.caption(
            "Новая подготовка pairwise/cluster всегда заменяет текущий plan. "
            "Скважины вне этого списка пойдут по общим параметрам расчета без "
            "локальных anti-collision overrides."
        )
    st.radio(
        "Детализация лога расчета",
        options=list(WT_LOG_LEVEL_OPTIONS),
        key="wt_log_verbosity",
        horizontal=True,
        help=(
            "`Краткий` — только ключевые события по каждой скважине. "
            "`Подробный` — все стадии солвера в реальном времени."
        ),
    )

    requests: list[_BatchRunRequest] = []
    pads, _, well_names_by_pad_id = _pad_membership(records)
    pad_ids = [str(pad.pad_id) for pad in pads]
    if (
        pad_ids
        and str(st.session_state.get("wt_batch_select_pad_id", "")).strip()
        not in pad_ids
    ):
        st.session_state["wt_batch_select_pad_id"] = pad_ids[0]
    with st.form("welltrack_run_form", clear_on_submit=False):
        st.markdown("#### Запуск / пересчет выбранных скважин")
        select_col, pad_col, action_col, pad_add_col, pad_only_col = (
            st.columns(
                [5.0, 2.4, 1.2, 1.45, 1.45],
                gap="small",
                vertical_alignment="bottom",
            )
        )
        with select_col:
            st.multiselect(
                "Скважины для расчета",
                options=all_names,
                key="wt_selected_names",
                help="Для каждой скважины ожидается ровно 3 точки: S, t1, t3.",
            )
        with action_col:
            select_all_clicked = st.form_submit_button(
                "Выбрать все",
                icon=":material/done_all:",
                width="stretch",
            )
        with pad_col:
            if len(pad_ids) > 1:
                st.selectbox(
                    "Куст",
                    options=pad_ids,
                    format_func=lambda value: _pad_display_label(
                        next(
                            pad
                            for pad in pads
                            if str(pad.pad_id) == str(value)
                        )
                    ),
                    key="wt_batch_select_pad_id",
                )
        with pad_add_col:
            add_pad_clicked = (
                st.form_submit_button(
                    "Добавить куст",
                    icon=":material/filter_alt:",
                    width="stretch",
                )
                if len(pad_ids) > 1
                else False
            )
        with pad_only_col:
            replace_with_pad_clicked = (
                st.form_submit_button(
                    "Только куст",
                    icon=":material/rule:",
                    width="stretch",
                )
                if len(pad_ids) > 1
                else False
            )
        st.caption(
            "Используйте этот блок и для первого расчета набора, и для пересчета "
            "любой выбранной части скважин. Применяются параметры расчета ниже, "
            "а результаты остальных скважин не будут затронуты."
        )
        if len(pad_ids) > 1:
            st.caption(
                "При работе с несколькими кустами можно либо добавить весь куст "
                "в текущую выборку, либо сразу переключиться только на него. Это "
                "удобно и для обычного batch, и как стартовая точка перед "
                "anti-collision пересчетом только для интересующего куста."
            )
        selected_now = list(st.session_state.get("wt_selected_names", []))
        prepared_scope_rows = _format_prepared_override_scope(
            selected_names=selected_now,
        )
        if prepared_scope_rows:
            st.warning(
                "Для части выбранных скважин будут применены локальные anti-collision "
                "overrides поверх общих параметров ниже."
            )
            st.dataframe(
                arrow_safe_text_dataframe(pd.DataFrame(prepared_scope_rows)),
                width="stretch",
                hide_index=True,
                column_config={
                    "Скважина": st.column_config.TextColumn("Скважина"),
                    "Локальный режим": st.column_config.TextColumn(
                        "Локальный режим"
                    ),
                    "Маневр": st.column_config.TextColumn("Маневр"),
                    "Источник": st.column_config.TextColumn("Источник"),
                },
            )
            st.caption(
                "Если нужен обычный batch без локальных anti-collision настроек, "
                "сначала нажмите `Очистить план` выше."
            )
        config = _build_config_form(
            binding=WT_CALC_PARAMS,
            title="Общие параметры расчета",
        )
        run_clicked = st.form_submit_button(
            "Запустить / пересчитать выбранные скважины",
            type="primary",
            icon=":material/play_arrow:",
        )
    if select_all_clicked:
        st.session_state["wt_pending_selected_names"] = list(all_names)
        st.rerun()
    if add_pad_clicked:
        selected_pad_id = str(
            st.session_state.get("wt_batch_select_pad_id", "")
        ).strip()
        current_selected = [
            str(name) for name in st.session_state.get("wt_selected_names", [])
        ]
        st.session_state["wt_pending_selected_names"] = list(
            dict.fromkeys(
                [
                    *current_selected,
                    *well_names_by_pad_id.get(selected_pad_id, ()),
                ]
            )
        )
        st.rerun()
    if replace_with_pad_clicked:
        selected_pad_id = str(
            st.session_state.get("wt_batch_select_pad_id", "")
        ).strip()
        st.session_state["wt_pending_selected_names"] = list(
            well_names_by_pad_id.get(selected_pad_id, ())
        )
        st.rerun()
    requests.append(
        _BatchRunRequest(
            selected_names=list(st.session_state.get("wt_selected_names", [])),
            config=config,
            run_clicked=bool(run_clicked),
        )
    )
    return requests


def _store_merged_batch_results(
    *,
    records: list[WelltrackRecord],
    new_rows: list[dict[str, object]],
    new_successes: list[SuccessfulWellPlan],
) -> None:
    merged_rows, merged_successes = merge_batch_results(
        records=records,
        existing_rows=st.session_state.get("wt_summary_rows"),
        existing_successes=st.session_state.get("wt_successes"),
        new_rows=new_rows,
        new_successes=new_successes,
    )
    st.session_state["wt_summary_rows"] = merged_rows
    st.session_state["wt_successes"] = merged_successes
    recommended_names = recommended_batch_selection(
        records=records,
        summary_rows=merged_rows,
    )
    st.session_state["wt_pending_selected_names"] = list(recommended_names)


def _run_batch_if_clicked(
    requests: list[_BatchRunRequest], records: list[WelltrackRecord]
) -> None:
    request = next((item for item in requests if item.run_clicked), None)
    if request is None:
        return
    selected_names = [str(name) for name in request.selected_names]
    selected_set = set(selected_names)
    if not selected_set:
        st.warning("Выберите минимум одну скважину для расчета.")
        return
    selected_execution_order = _selected_execution_order(selected_names)

    records_for_run = list(records)
    pad_layout_active = bool(
        str(st.session_state.get("wt_pad_last_applied_at", ""))
    )
    if pad_layout_active:
        base_records = st.session_state.get("wt_records_original")
        if base_records:
            pads = _ensure_pad_configs(base_records=list(base_records))
            plan_map = _build_pad_plan_map(pads)
            records_for_run = apply_pad_layout(
                records=list(base_records),
                pads=pads,
                plan_by_pad_id=plan_map,
            )
            st.session_state["wt_records"] = list(records_for_run)

    batch = WelltrackBatchPlanner(planner=TrajectoryPlanner())
    log_verbosity = str(
        st.session_state.get("wt_log_verbosity", WT_LOG_COMPACT)
    )
    verbose_log_enabled = log_verbosity == WT_LOG_VERBOSE
    records_by_name = {str(record.name): record for record in records_for_run}
    config_by_name = _build_selected_override_configs(
        base_config=request.config,
        selected_names=selected_set,
        records_by_name=records_by_name,
    )
    optimization_context_by_name = _build_selected_optimization_contexts(
        selected_names=selected_set,
        current_successes=list(st.session_state.get("wt_successes") or ()),
    )
    current_success_by_name = {
        str(item.name): item
        for item in (st.session_state.get("wt_successes") or ())
    }
    prepared_snapshot = dict(
        st.session_state.get("wt_prepared_recommendation_snapshot") or {}
    )
    prepared_override_names = {
        str(name)
        for name in (
            st.session_state.get("wt_prepared_well_overrides") or {}
        ).keys()
    }
    previous_anticollision_successes = {
        str(name): current_success_by_name[str(name)]
        for name in sorted(
            prepared_override_names.intersection(current_success_by_name)
        )
    }
    dynamic_cluster_context = None
    if str(prepared_snapshot.get("kind", "")).strip() == "cluster":
        target_well_names = tuple(
            str(name)
            for name in prepared_snapshot.get("target_well_names", ()) or ()
            if str(name).strip()
        ) or _resolution_snapshot_well_names(prepared_snapshot)
        if target_well_names:
            dynamic_cluster_context = DynamicClusterExecutionContext(
                target_well_names=tuple(target_well_names),
                uncertainty_model=planning_uncertainty_model_for_preset(
                    normalize_uncertainty_preset(
                        st.session_state.get(
                            "wt_anticollision_uncertainty_preset",
                            DEFAULT_UNCERTAINTY_PRESET,
                        )
                    )
                ),
                initial_successes=tuple(
                    st.session_state.get("wt_successes") or ()
                ),
                reference_wells=_reference_wells_from_state(),
            )
    missing_anticollision_context = sorted(
        well_name
        for well_name, cfg in config_by_name.items()
        if str(cfg.optimization_mode) == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
        and well_name not in optimization_context_by_name
    )
    if missing_anticollision_context and dynamic_cluster_context is None:
        st.session_state["wt_last_error"] = (
            "Не удалось запустить anti-collision пересчет: отсутствует контекст "
            "конфликтного окна для скважин "
            + ", ".join(missing_anticollision_context)
            + ". Подготовьте рекомендацию повторно."
        )
        st.error(str(st.session_state["wt_last_error"]))
        return
    run_started_s = perf_counter()
    log_lines: list[str] = []
    progress = st.progress(0, text="Подготовка batch-расчета...")
    phase_placeholder = st.empty()
    live_log_placeholder = st.empty()

    def append_log(message: str, *, verbose_only: bool = False) -> None:
        if verbose_only and not verbose_log_enabled:
            return
        log_lines.append(format_run_log_line(run_started_s, message))
        live_log_placeholder.code("\n".join(log_lines[-240:]), language="text")

    def set_phase(message: str) -> None:
        phase_placeholder.caption(message)

    prepared_scope_rows = _format_prepared_override_scope(
        selected_names=selected_names,
    )

    try:
        with st.spinner(
            "Выполняется расчет WELLTRACK-набора...", show_time=True
        ):
            started = perf_counter()
            append_log(
                f"Старт batch-расчета. Выбрано скважин: {len(selected_set)}. "
                f"Детализация лога: {log_verbosity}."
            )
            if prepared_scope_rows:
                append_log(
                    "Активен prepared anti-collision plan ("
                    + _prepared_plan_kind_label(prepared_snapshot)
                    + "). Локальные overrides будут применены к "
                    + ", ".join(
                        f"{row['Скважина']} ({row['Локальный режим']})"
                        for row in prepared_scope_rows
                    )
                    + "."
                )
            if optimization_context_by_name:
                append_log(
                    "Для части выбранных скважин активирован anti-collision avoidance "
                    "mode на конфликтном окне."
                )
            active_kop_function = kop_min_vertical_function_from_state(
                prefix=WT_CALC_PARAMS.prefix
            )
            if active_kop_function is not None:
                append_log(
                    "Для выбранных скважин активна функция KOP / TVD: "
                    + str(active_kop_function.note).strip()
                )
            if dynamic_cluster_context is not None:
                append_log(
                    "Включена iterative cluster-aware execution policy: "
                    "порядок шагов и anti-collision overrides будут пересчитываться "
                    "после каждого успешного шага по текущей topology кластера."
                )
            elif (
                len(selected_execution_order) > 1
                and selected_execution_order != selected_names
            ):
                append_log(
                    "Cluster-aware execution order: "
                    + " -> ".join(selected_execution_order)
                    + ". Следующие скважины используют обновленные reference paths "
                    "уже пересчитанных шагов."
                )
            if pad_layout_active:
                append_log(
                    "Активна раскладка устьев по кустам: перед расчетом применены "
                    "текущие координаты S из блока 'Кусты и расчет устьев'."
                )
            set_phase(
                f"Старт расчета набора. Выбрано скважин: {len(selected_set)}."
            )
            progress_state: dict[str, int] = {"value": 0}
            last_stage_by_well: dict[str, str] = {}

            def update_progress(value: int, text: str) -> None:
                clamped = int(max(0, min(99, value)))
                clamped = max(int(progress_state["value"]), clamped)
                progress_state["value"] = clamped
                progress.progress(clamped, text=text)

            def on_progress(index: int, total: int, name: str) -> None:
                start_fraction = (float(index) - 1.0) / max(float(total), 1.0)
                update_progress(
                    int(round(start_fraction * 100.0)),
                    text=f"{index}/{total}: {name} · подготовка",
                )
                set_phase(f"Расчет скважины {index}/{total}: {name}")
                append_log(
                    f"Расчет скважины {index}/{total}: {name}.",
                )

            def on_solver_progress(
                index: int,
                total: int,
                name: str,
                stage_text: str,
                stage_fraction: float,
            ) -> None:
                local_fraction = float(max(0.0, min(1.0, stage_fraction)))
                overall = (float(index) - 1.0 + local_fraction) / max(
                    float(total), 1.0
                )
                update_progress(
                    int(round(overall * 100.0)),
                    text=f"{index}/{total}: {name} · {stage_text}",
                )
                set_phase(f"Скважина {index}/{total} {name}: {stage_text}")
                stage_key = f"{index}:{name}"
                stage_norm = str(stage_text)
                if last_stage_by_well.get(stage_key) == stage_norm:
                    return
                last_stage_by_well[stage_key] = stage_norm
                append_log(f"{name}: {stage_norm}", verbose_only=True)

            def on_record_done(
                index: int,
                total: int,
                name: str,
                row: dict[str, object],
            ) -> None:
                end_fraction = float(index) / max(float(total), 1.0)
                update_progress(
                    int(round(end_fraction * 100.0)),
                    text=f"{index}/{total}: {name} · завершено",
                )
                status = str(row.get("Статус", "—"))
                raw_problem_text = str(row.get("Проблема", "")).strip()
                problem_text = (
                    summarize_problem_ru(raw_problem_text)
                    if raw_problem_text
                    else ""
                )
                restart_count = 0
                try:
                    restart_count = int(float(row.get("Рестарты решателя", 0)))
                except (TypeError, ValueError):
                    restart_count = 0
                restart_suffix = (
                    f" Использовано рестартов решателя: {restart_count}."
                    if restart_count > 0
                    else ""
                )
                if status == "OK":
                    if problem_text and problem_text != "ОК":
                        append_log(
                            f"{name}: расчет завершен с предупреждением. {problem_text}"
                            f"{restart_suffix}"
                        )
                    else:
                        append_log(
                            f"{name}: расчет завершен успешно.{restart_suffix}"
                        )
                    return
                if problem_text and problem_text != "ОК":
                    append_log(f"{name}: {status}. {problem_text}")
                else:
                    append_log(f"{name}: {status}.")

            summary_rows, successes = batch.evaluate(
                records=records_for_run,
                selected_names=selected_set,
                selected_order=selected_execution_order,
                config=request.config,
                config_by_name=config_by_name,
                optimization_context_by_name=optimization_context_by_name,
                dynamic_cluster_context=dynamic_cluster_context,
                progress_callback=on_progress,
                solver_progress_callback=on_solver_progress,
                record_done_callback=on_record_done,
            )
            batch_metadata = batch.last_evaluation_metadata
            skipped_policy_count = int(
                len(batch_metadata.skipped_selected_names)
            )
            if dynamic_cluster_context is not None:
                skipped_names = tuple(
                    str(name)
                    for name in batch_metadata.skipped_selected_names
                    if str(name).strip()
                )
                if skipped_names:
                    if bool(batch_metadata.cluster_blocked):
                        blocking_reason = (
                            str(batch_metadata.cluster_blocking_reason).strip()
                            if batch_metadata.cluster_blocking_reason
                            else "cluster-level пересчет перешел в advisory-only режим."
                        )
                        append_log(
                            "Iterative cluster-aware execution остановлен: "
                            + blocking_reason
                            + " Без дополнительного пересчета оставлены: "
                            + ", ".join(skipped_names)
                            + "."
                        )
                    elif bool(batch_metadata.cluster_resolved_early):
                        append_log(
                            "Iterative cluster-aware execution завершился досрочно: "
                            "после очередного шага дополнительные пересчеты для "
                            "оставшихся скважин не потребовались. Без повторного "
                            "пересчета оставлены: "
                            + ", ".join(skipped_names)
                            + "."
                        )
            elapsed_s = perf_counter() - started
            progress.progress(100, text="Batch-расчет завершен.")
            _store_merged_batch_results(
                records=records_for_run,
                new_rows=summary_rows,
                new_successes=successes,
            )
            applied_affected_wells = {
                str(name)
                for name in prepared_snapshot.get("affected_wells", ())
            }
            applied_prepared_plan = bool(
                prepared_snapshot
                and applied_affected_wells
                and applied_affected_wells.issubset(selected_set)
            )
            if applied_prepared_plan:
                preset = normalize_uncertainty_preset(
                    st.session_state.get(
                        "wt_anticollision_uncertainty_preset",
                        DEFAULT_UNCERTAINTY_PRESET,
                    )
                )
                resolution = _build_last_anticollision_resolution(
                    snapshot=prepared_snapshot,
                    successes=list(st.session_state.get("wt_successes") or ()),
                    uncertainty_model=planning_uncertainty_model_for_preset(
                        preset
                    ),
                    uncertainty_preset=preset,
                )
                st.session_state["wt_last_anticollision_resolution"] = (
                    resolution
                )
                st.session_state[
                    "wt_last_anticollision_previous_successes"
                ] = previous_anticollision_successes
                _focus_all_wells_anticollision_results()
            else:
                st.session_state["wt_last_anticollision_resolution"] = None
                st.session_state[
                    "wt_last_anticollision_previous_successes"
                ] = {}
                _focus_all_wells_trajectory_results()
            st.session_state["wt_last_error"] = ""
            st.session_state["wt_last_run_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.session_state["wt_last_runtime_s"] = float(elapsed_s)
            st.session_state["wt_prepared_well_overrides"] = {}
            st.session_state["wt_prepared_override_message"] = ""
            st.session_state["wt_prepared_recommendation_id"] = ""
            st.session_state["wt_anticollision_prepared_cluster_id"] = ""
            st.session_state["wt_prepared_recommendation_snapshot"] = None
            append_log(
                f"Batch-расчет завершен. Успешно: {len(successes)}, "
                f"ошибок: {len(summary_rows) - len(successes)}"
                + (
                    f", без дополнительного пересчета оставлено: {skipped_policy_count}"
                    if skipped_policy_count > 0
                    else ""
                )
                + ". "
                f"Затраченное время: {elapsed_s:.2f} с.",
            )
            if successes:
                phase_placeholder.success(
                    f"Расчет завершен за {elapsed_s:.2f} с. Успешно: {len(successes)}"
                    + (
                        f", без дополнительного пересчета оставлено: {skipped_policy_count}"
                        if skipped_policy_count > 0
                        else ""
                    )
                )
            else:
                phase_placeholder.error(
                    f"Расчет завершен за {elapsed_s:.2f} с, но без успешных скважин."
                )
    except Exception as exc:  # noqa: BLE001
        st.session_state["wt_last_error"] = str(exc)
        append_log(
            f"Ошибка batch-расчета: {summarize_problem_ru(str(exc))}",
        )
        phase_placeholder.error("Batch-расчет завершился ошибкой")
    finally:
        st.session_state["wt_last_run_log_lines"] = log_lines
        progress.empty()
        live_log_placeholder.empty()


def _render_batch_log() -> None:
    render_run_log_panel(st.session_state.get("wt_last_run_log_lines"))


def _build_batch_survey_csv(
    successes: list[SuccessfulWellPlan],
) -> bytes:
    """Build combined CSV with inclinometry data for all successful wells."""
    if not successes:
        return b""
    frames: list[pd.DataFrame] = []
    for success in successes:
        stations = success.stations.copy()
        if stations.empty:
            continue
        stations.insert(0, "well_name", str(success.name))
        if "DLS_deg_per_30m" in stations.columns:
            from pywp.ui_utils import dls_to_pi
            stations["PI_deg_per_10m"] = dls_to_pi(
                stations["DLS_deg_per_30m"].to_numpy(dtype=float)
            )
            stations = stations.drop(columns=["DLS_deg_per_30m"])
        frames.append(stations)
    if not frames:
        return b""
    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False).encode("utf-8")


def _render_batch_summary(
    summary_rows: list[dict[str, object]],
) -> pd.DataFrame:
    summary_df = WelltrackBatchPlanner.summary_dataframe(summary_rows)
    if not summary_df.empty:
        summary_df = arrow_safe_text_dataframe(summary_df)

    ok_count = 0
    warning_count = 0
    err_count = 0
    not_run_count = 0
    if not summary_df.empty and {"Статус", "Проблема"}.issubset(
        summary_df.columns
    ):
        for _, row in summary_df.iterrows():
            status = str(row["Статус"]).strip()
            problem_text = str(row["Проблема"]).strip()
            if status == "OK":
                if problem_text and problem_text != "—":
                    warning_count += 1
                else:
                    ok_count += 1
            elif status == "Не рассчитана":
                not_run_count += 1
            else:
                err_count += 1

    p1, p2, p3, p4, p5 = st.columns(5, gap="small")
    p1.metric("Строк в отчете", f"{len(summary_df)}")
    p2.metric("Без замечаний", f"{ok_count}")
    p3.metric("С предупреждениями", f"{warning_count}")
    p4.metric("Ошибки", f"{err_count}")
    run_time = st.session_state.get("wt_last_runtime_s")
    p5.metric(
        "Время расчета",
        "—" if run_time is None else f"{float(run_time):.2f} с",
    )
    if not_run_count:
        st.caption(
            f"Не рассчитаны: {not_run_count}. Это нормально для partial batch-расчета: "
            "строки остаются в отчете до отдельного запуска по этим скважинам."
        )
    render_small_note(
        f"Последний запуск: {st.session_state.get('wt_last_run_at', '—')}"
    )
    if not summary_df.empty and "Проблема" in summary_df.columns:
        has_md_postcheck_warning = bool(
            summary_df["Проблема"]
            .astype(str)
            .str.contains("Превышен лимит итоговой MD", regex=False)
            .any()
        )
        if has_md_postcheck_warning:
            st.caption(
                "Скважины с превышением лимита итоговой MD отображаются пунктирной "
                "траекторией на графиках."
            )

    st.markdown("### Сводка расчета")
    display_df = _batch_summary_display_df(summary_df)
    display_payload: pd.DataFrame | pd.io.formats.style.Styler
    if display_df.empty:
        display_payload = display_df
    else:
        display_payload = display_df.style.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("font-size", "0.92rem")],
                },
                {
                    "selector": "td",
                    "props": [("font-size", "0.90rem")],
                },
            ]
        )
    st.dataframe(
        display_payload,
        width="stretch",
        hide_index=True,
        column_config={
            "Скважина": st.column_config.TextColumn("Скважина", width="small"),
            "Точек": st.column_config.NumberColumn(
                "Точек", format="%d", width="small"
            ),
            "Цели": st.column_config.TextColumn("Цели", width="small"),
            "Сложность": st.column_config.TextColumn(
                "Сложность", width="small"
            ),
            "Отход t1, м": st.column_config.NumberColumn(
                "Отход t1, м",
                format="%.2f",
                width="small",
            ),
            "\u0413\u0421, \u043c": st.column_config.NumberColumn(
                "\u0413\u0421, \u043c",
                format="%.2f",
                width="small",
            ),
            "INC в t1, deg": st.column_config.NumberColumn(
                "INC t1, deg", format="%.2f", width="small"
            ),
            "ЗУ HOLD, deg": st.column_config.NumberColumn(
                "ЗУ HOLD, deg", format="%.2f", width="small"
            ),
            "Макс ПИ, deg/10m": st.column_config.NumberColumn(
                "Макс ПИ, deg/10m",
                format="%.2f",
                width="small",
            ),
            "Макс MD, м": st.column_config.NumberColumn(
                "Макс MD, м",
                format="%.2f",
                width="small",
            ),
            "Рестарты": st.column_config.TextColumn("Рестарты", width="small"),
            "Статус": st.column_config.TextColumn("Статус", width="small"),
            "Проблема": st.column_config.TextColumn(
                "Проблема", width="medium"
            ),
            "Модель траектории": st.column_config.TextColumn(
                "Модель траектории",
                width="medium",
            ),
        },
    )
    btn_cols = st.columns([1, 1])
    with btn_cols[0]:
        st.download_button(
            "Скачать сводку (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="welltrack_summary.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )
    with btn_cols[1]:
        successes = st.session_state.get("wt_successes") or []
        survey_data = _build_batch_survey_csv(successes)
        st.download_button(
            "Скачать инклинометрию (CSV)",
            data=survey_data or b"",
            file_name="welltrack_survey_all.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
            disabled=not survey_data,
        )
    return summary_df


def _batch_summary_display_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    display_df = summary_df.rename(
        columns=_BATCH_SUMMARY_RENAME_COLUMNS
    ).copy()
    ordered = [
        column
        for column in _BATCH_SUMMARY_DISPLAY_ORDER
        if column in display_df.columns
    ]
    trailing = [
        column for column in display_df.columns if column not in ordered
    ]
    return display_df[ordered + trailing]


def _ensure_selected_success_baseline(
    *,
    selected_name: str,
    successes: list[SuccessfulWellPlan],
) -> SuccessfulWellPlan:
    selected = next(
        item for item in successes if str(item.name) == str(selected_name)
    )
    updated = ensure_successful_plan_baseline(success=selected)
    if updated is selected:
        return selected
    st.session_state["wt_successes"] = [
        updated if str(item.name) == str(selected_name) else item
        for item in successes
    ]
    return updated


def _render_success_tabs(
    *,
    successes: list[SuccessfulWellPlan],
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]],
) -> None:
    name_to_color = _well_color_map(records)
    reference_wells = _reference_wells_from_state()
    target_only_wells = _failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    view_mode = st.radio(
        "Режим просмотра результатов",
        options=["Отдельная скважина", "Все скважины"],
        key="wt_results_view_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    if str(view_mode) == "Отдельная скважина":
        selected_name = st.selectbox(
            "Скважина", options=[item.name for item in successes]
        )
        selected = _ensure_selected_success_baseline(
            selected_name=str(selected_name),
            successes=successes,
        )
        well_view = SingleWellResultView(
            well_name=str(selected.name),
            surface=selected.surface,
            t1=selected.t1,
            t3=selected.t3,
            stations=selected.stations,
            summary=selected.summary,
            config=selected.config,
            azimuth_deg=float(selected.azimuth_deg),
            md_t1_m=float(selected.md_t1_m),
            runtime_s=selected.runtime_s,
            baseline_summary=selected.baseline_summary,
            baseline_runtime_s=selected.baseline_runtime_s,
            issue_messages=(
                (str(selected.md_postcheck_message),)
                if str(selected.md_postcheck_message).strip()
                else ()
            ),
            trajectory_line_dash=(
                "dash" if bool(selected.md_postcheck_exceeded) else "solid"
            ),
        )
        t1_horizontal_offset_m = render_key_metrics(
            view=well_view,
            title="Ключевые показатели",
            border=True,
        )
        render_result_plots(
            view=well_view,
            title_trajectory=None,
            title_plan=None,
            border=True,
        )
        render_result_tables(
            view=well_view,
            t1_horizontal_offset_m=t1_horizontal_offset_m,
            summary_tab_label="Сводка",
            survey_tab_label="Инклинометрия",
            survey_file_name=f"{selected_name}_survey.csv",
        )
        return

    all_view_mode = st.radio(
        "Режим отображения всех скважин",
        options=["Траектории", "Anti-collision"],
        key="wt_results_all_view_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    pads, _, _ = _pad_membership(records)
    if len(pads) > 1:
        focus_options = [WT_PAD_FOCUS_ALL, *(str(pad.pad_id) for pad in pads)]
        normalized_focus_pad_id = _normalize_focus_pad_id(
            records=records,
            requested_pad_id=st.session_state.get("wt_results_focus_pad_id"),
        )
        if normalized_focus_pad_id != str(
            st.session_state.get("wt_results_focus_pad_id", "")
        ):
            st.session_state["wt_results_focus_pad_id"] = (
                normalized_focus_pad_id
            )
        st.selectbox(
            "Фокус камеры по кусту",
            options=focus_options,
            format_func=lambda value: (
                "Все кусты"
                if str(value) == WT_PAD_FOCUS_ALL
                else _pad_display_label(
                    next(pad for pad in pads if str(pad.pad_id) == str(value))
                )
            ),
            key="wt_results_focus_pad_id",
            help=(
                "Камера в 3D и 2D будет фокусироваться на выбранном кусте, "
                "но остальные скважины останутся на сцене. В anti-collision для "
                "выбранного куста будут показаны только затрагивающие его события "
                "и кластеры."
            ),
        )
    focus_pad_id = _normalize_focus_pad_id(
        records=records,
        requested_pad_id=st.session_state.get("wt_results_focus_pad_id"),
    )
    focus_pad_well_names = _focus_pad_well_names(
        records=records,
        focus_pad_id=focus_pad_id,
    )
    if str(all_view_mode) == "Траектории":
        selected_render_mode = st.selectbox(
            "3D-режим отображения",
            options=list(WT_3D_RENDER_OPTIONS),
            key="wt_3d_render_mode",
        )
        selected_3d_backend = st.selectbox(
            "3D backend",
            options=list(WT_3D_BACKEND_OPTIONS),
            key="wt_3d_backend",
            help=(
                "Plotly сохраняет привычные hover-подсказки. "
                "Локальный Three.js backend быстрее на тяжёлых кустах и хранит все файлы локально."
            ),
        )
        if str(selected_3d_backend) == WT_3D_BACKEND_THREE_LOCAL:
            if st.button(
                "Пересоздать 3D viewer", key="wt_recreate_three_traj"
            ):
                _bump_three_viewer_nonce()
                st.rerun()
        resolved_render_mode = _resolve_3d_render_mode(
            requested_mode=selected_render_mode,
            calculated_well_count=len(successes),
            reference_wells=reference_wells,
        )
        if target_only_wells:
            st.caption(
                "Для непростроенных скважин на обзорных графиках показаны только "
                "точки S/t1/t3, без траектории."
            )
        if reference_wells:
            st.caption(
                "Дополнительные фактические и утвержденные скважины показаны как "
                "reference-траектории: серые и красные линии без точек S/t1/t3."
            )
        if resolved_render_mode == WT_3D_RENDER_FAST:
            st.caption(
                "Включён быстрый 3D-режим: reference-скважины объединяются в "
                "сводные фоновые 3D-линии, чтобы не перегружать браузер. 2D-план "
                "и сами расчётные скважины остаются без смысловых изменений."
            )
        if str(selected_3d_backend) == WT_3D_BACKEND_THREE_LOCAL:
            st.caption(
                "Активен локальный Three.js viewer: 3D-смысл сцены сохраняется, "
                "но детальные Plotly-hover подсказки доступны только в режиме Plotly."
            )
        c1, c2 = st.columns(2, gap="medium")
        overview_3d_figure = _all_wells_3d_figure(
            successes,
            target_only_wells=target_only_wells,
            reference_wells=reference_wells,
            name_to_color=name_to_color,
            focus_well_names=focus_pad_well_names,
            render_mode=selected_render_mode,
        )
        _render_plotly_or_three_3d(
            container=c1,
            figure=overview_3d_figure,
            backend=selected_3d_backend,
            height=620,
            payload_overrides=_trajectory_three_payload_overrides(
                records=records,
                successes=successes,
                target_only_wells=target_only_wells,
                name_to_color=name_to_color,
            ),
        )
        c2.plotly_chart(
            _all_wells_plan_figure(
                successes,
                target_only_wells=target_only_wells,
                reference_wells=reference_wells,
                name_to_color=name_to_color,
                focus_well_names=focus_pad_well_names,
            ),
            width="stretch",
        )
        return

    _render_anticollision_panel(
        successes,
        records=records,
        focus_pad_id=focus_pad_id,
    )
