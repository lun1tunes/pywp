from __future__ import annotations

import colorsys
import hashlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Suppress noisy Streamlit warnings BEFORE importing streamlit
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import streamlit as st

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from pywp import TrajectoryConfig
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
    collision_corridor_plan_polygon,
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
    build_anti_collision_recommendation_clusters,
    build_anti_collision_recommendations,
    cluster_display_label,
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
from pywp.coordinate_integration import (
    DEFAULT_CRS,
    csv_export_crs,
    get_crs_display_suffix,
    transform_stations_to_crs,
)
from pywp.coordinate_systems import CoordinateSystem
from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackRecord,
    parse_welltrack_text,
    welltrack_points_to_targets,
)
from pywp.models import Point3D
from pywp.pilot_wells import (
    is_pilot_name,
    parent_name_for_pilot,
    pilot_name_key_for_parent,
    sync_pilot_surfaces_to_parents,
    visible_well_records,
    well_name_key,
)
from pywp.planner_config import optimization_display_label
from pywp.plot_axes import (
    equalized_axis_ranges,
    equalized_xy_ranges,
    linear_tick_values,
    nice_tick_step,
    reversed_axis_range,
)
from pywp import ptc_anticollision_view
from pywp import ptc_anticollision_params
from pywp import ptc_batch_run
from pywp import ptc_batch_results
from pywp import ptc_batch_summary_panel
from pywp import ptc_edit_targets
from pywp import ptc_pad_state
from pywp import ptc_reference_state
from pywp import ptc_target_import
from pywp import ptc_target_records
from pywp import ptc_three_builders
from pywp import ptc_three_overrides
from pywp import ptc_three_payload
from pywp import ptc_welltrack_io
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    REFERENCE_WELL_KIND_COLORS,
    REFERENCE_WELL_KIND_LABELS,
    ImportedTrajectoryWell,
    reference_well_display_label,
)
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
)
from pywp.ui_well_panels import survey_export_dataframe
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    PlanningUncertaintyModel,
    build_uncertainty_tube_mesh,
    normalize_uncertainty_preset,
    uncertainty_preset_label,
    uncertainty_ribbon_polygon,
)
from pywp.well_pad import (
    PAD_SURFACE_ANCHOR_CENTER,
    PAD_SURFACE_ANCHOR_FIRST,
    PadLayoutPlan,
    WellPad,
    apply_pad_layout,
    estimate_pad_nds_azimuth_deg,
    ordered_pad_wells,
)
from pywp.welltrack_batch import (
    SuccessfulWellPlan,
    WelltrackBatchPlanner,
    rebuild_optimization_context,
)
from pywp.welltrack_quality import (
    detect_t1_t3_order_issues,
    swap_t1_t3_for_wells,
)

DEFAULT_WELLTRACK_PATH = ptc_target_import.DEFAULT_WELLTRACK_PATH
WT_SOURCE_FORMAT_WELLTRACK = ptc_target_import.WT_SOURCE_FORMAT_WELLTRACK
WT_SOURCE_FORMAT_TARGET_TABLE = ptc_target_import.WT_SOURCE_FORMAT_TARGET_TABLE
WT_SOURCE_FORMAT_OPTIONS = ptc_target_import.WT_SOURCE_FORMAT_OPTIONS
WT_SOURCE_MODE_FILE_PATH = ptc_target_import.WT_SOURCE_MODE_FILE_PATH
WT_SOURCE_MODE_UPLOAD = ptc_target_import.WT_SOURCE_MODE_UPLOAD
WT_SOURCE_MODE_INLINE_TEXT = ptc_target_import.WT_SOURCE_MODE_INLINE_TEXT
WT_SOURCE_MODE_TARGET_TABLE = ptc_target_import.WT_SOURCE_MODE_TARGET_TABLE
WT_SOURCE_WELLTRACK_MODES = ptc_target_import.WT_SOURCE_WELLTRACK_MODES
WT_UI_DEFAULTS_VERSION = 16
WT_LOG_COMPACT = ptc_batch_run.LOG_COMPACT
WT_LOG_VERBOSE = ptc_batch_run.LOG_VERBOSE
WT_LOG_LEVEL_OPTIONS = ptc_batch_run.LOG_LEVEL_OPTIONS
WT_T1T3_MIN_DELTA_M = 0.5
WT_3D_RENDER_AUTO = "Авто"
WT_3D_RENDER_DETAIL = "Детально"
WT_3D_RENDER_FAST = "Быстро"
REFERENCE_LABEL_ACTUAL_COLOR = "#111111"
REFERENCE_LABEL_APPROVED_COLOR = "#C62828"
REFERENCE_PAD_LABEL_COLOR = "#334155"
REFERENCE_LABEL_HORIZONTAL_INC_THRESHOLD_DEG = 80.0
REFERENCE_LABEL_HORIZONTAL_MIN_INTERVAL_M = 100.0
REFERENCE_PAD_GROUP_DISTANCE_M = 300.0
WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M = (
    ptc_pad_state.WT_IMPORTED_PAD_SURFACE_CHAIN_DISTANCE_M
)
WT_3D_RENDER_OPTIONS: tuple[str, ...] = (
    WT_3D_RENDER_DETAIL,
    WT_3D_RENDER_FAST,
)
WT_3D_FAST_REFERENCE_WELL_THRESHOLD = 10
WT_3D_FAST_REFERENCE_POINT_THRESHOLD = 1200
WT_3D_FAST_CALC_WELL_THRESHOLD = 14
WT_3D_FAST_REFERENCE_TARGET_POINTS = 72
WT_3D_FAST_CALC_TARGET_POINTS = 180
WT_3D_FAST_REFERENCE_CONE_WELL_LIMIT = 6
WT_3D_REFERENCE_CONE_FOCUS_DISTANCE_M = 500.0
WT_THREE_MAX_HOVER_POINTS_PER_TRACE = (
    ptc_three_payload.WT_THREE_MAX_HOVER_POINTS_PER_TRACE
)
WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE = (
    ptc_three_payload.WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
)
WT_THREE_MAX_LABELS = ptc_three_payload.WT_THREE_MAX_LABELS
WT_THREE_MAX_REFERENCE_LABELS = ptc_three_payload.WT_THREE_MAX_REFERENCE_LABELS
WT_PAD_FOCUS_ALL = ptc_pad_state.WT_PAD_FOCUS_ALL
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
DEFAULT_PAD_SPACING_M = ptc_pad_state.DEFAULT_PAD_SPACING_M
DEFAULT_PAD_SURFACE_ANCHOR_MODE = ptc_pad_state.DEFAULT_PAD_SURFACE_ANCHOR_MODE


_BatchRunRequest = ptc_batch_run.BatchRunRequest


_WelltrackSourcePayload = ptc_target_import.WelltrackSourcePayload


@dataclass(frozen=True)
class _TargetOnlyWell:
    name: str
    surface: Point3D
    t1: Point3D
    t3: Point3D
    status: str
    problem: str


_DetectedPadUiMeta = ptc_pad_state.DetectedPadUiMeta


def _build_well_color_palette() -> tuple[str, ...]:
    """Return a long palette with high local contrast for adjacent wells.

    The hue order intentionally jumps around the wheel instead of walking
    sequentially, so neighboring indices remain visually distinct. Reds,
    blacks and grays are excluded on purpose to keep collision overlays and
    reference wells semantically separate.
    """

    leading_colors: tuple[str, ...] = (
        "#22C55E",
        "#8B5CF6",
        "#F59E0B",
        "#4338CA",
        "#A3E635",
        "#0891B2",
        "#8B5E34",
        "#38BDF8",
        "#B45309",
        "#2563EB",
        "#65A30D",
        "#0F766E",
    )
    hue_degrees: tuple[float, ...] = (
        116.0,
        236.0,
        58.0,
        146.0,
        36.0,
        266.0,
        206.0,
        86.0,
        176.0,
        292.0,
    )
    hue_jump_order: tuple[int, ...] = tuple(range(len(hue_degrees)))
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
    try:
        _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
            st.session_state,
            visible_well_records(records),
        )
    except Exception:
        well_names_by_pad_id = {}
    color_map: dict[str, str] = {}
    for ordered_names in well_names_by_pad_id.values():
        for index, name in enumerate(ordered_names):
            color_map.setdefault(str(name), _well_color(index))
    fallback_index = 0
    for record in records:
        name = str(record.name)
        if name in color_map:
            continue
        color_map[name] = _well_color(fallback_index)
        fallback_index += 1
    name_by_key = {well_name_key(record.name): str(record.name) for record in records}
    for record in records:
        name = str(record.name)
        if is_pilot_name(name):
            parent_name = name_by_key.get(well_name_key(parent_name_for_pilot(name)))
            if parent_name is not None and parent_name in color_map:
                color_map[name] = color_map[parent_name]
            continue
        pilot_name = name_by_key.get(pilot_name_key_for_parent(name))
        if pilot_name is not None:
            color_map[pilot_name] = color_map[name]
    return color_map


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
    pending_edit_names = set(_pending_edit_target_names())
    target_only_wells: list[_TargetOnlyWell] = []
    for record in records:
        row = rows_by_name.get(str(record.name))
        if row is None:
            continue
        status = str(row.get("Статус", "")).strip()
        if status == "OK" or (
            status == "Не рассчитана" and str(record.name) not in pending_edit_names
        ):
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
            or len(reference_wells_tuple) >= WT_3D_FAST_REFERENCE_WELL_THRESHOLD
            or _reference_point_count(reference_wells_tuple)
            >= WT_3D_FAST_REFERENCE_POINT_THRESHOLD
        ):
            return WT_3D_RENDER_FAST
        return WT_3D_RENDER_DETAIL
    return WT_3D_RENDER_FAST


def _decimate_hover_payload(
    *,
    points: list[list[float]],
    hover_items: list[dict[str, object]],
    max_points: int,
) -> tuple[list[list[float]], list[dict[str, object]]]:
    return ptc_three_payload.decimate_hover_payload(
        points=points,
        hover_items=hover_items,
        max_points=max_points,
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


def _anticollision_focus_reference_names(
    analysis: AntiCollisionAnalysis,
) -> set[str]:
    reference_names = {
        str(well.name) for well in analysis.wells if bool(well.is_reference_only)
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
        getattr(reference_well, "kind", getattr(reference_well, "well_kind", ""))
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
    high_angle_mask = inc_values >= float(REFERENCE_LABEL_HORIZONTAL_INC_THRESHOLD_DEG)
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
        if np.isfinite(float(well.surface.x)) and np.isfinite(float(well.surface.y))
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
                for candidate_index in cells.get((cell[0] + dx, cell[1] + dy), ()):
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
            for pad_id in [_reference_pad_numeric_id(str(ordered_wells[index].name))]
            if pad_id is not None
        ]
        if numeric_ids:
            counts = {pad_id: numeric_ids.count(pad_id) for pad_id in set(numeric_ids)}
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


def _optimize_three_payload(payload: dict[str, object]) -> dict[str, object]:
    return ptc_three_payload.optimize_three_payload(payload)


def _merge_three_line_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    return ptc_three_payload.merge_three_line_payloads(items)


def _merge_three_point_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    return ptc_three_payload.merge_three_point_payloads(items)


def _merge_three_mesh_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    return ptc_three_payload.merge_three_mesh_payloads(items)


def _raw_bounds_from_xyz_arrays(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> dict[str, list[float]] | None:
    return ptc_three_payload.raw_bounds_from_xyz_arrays(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )


def _merge_raw_bounds(
    bounds_items: Iterable[dict[str, list[float]] | None],
) -> dict[str, list[float]] | None:
    return ptc_three_payload.merge_raw_bounds(bounds_items)


def _successful_plan_raw_bounds(
    success: SuccessfulWellPlan,
) -> dict[str, list[float]] | None:
    return ptc_three_overrides.successful_plan_raw_bounds(success)


def _target_only_raw_bounds(
    target_only: _TargetOnlyWell,
) -> dict[str, list[float]]:
    return ptc_three_overrides.target_only_raw_bounds(target_only)


def _legend_pad_label(pad: WellPad) -> str:
    return ptc_three_overrides.legend_pad_label(pad)


def _three_legend_tree_payload(
    *,
    records: list[WelltrackRecord],
    visible_well_names: Iterable[str],
    well_bounds_by_name: Mapping[str, dict[str, list[float]]],
    name_to_color: Mapping[str, str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, list[float]]], set[str]]:
    return ptc_three_overrides.three_legend_tree_payload(
        st.session_state,
        records=records,
        visible_well_names=visible_well_names,
        well_bounds_by_name=well_bounds_by_name,
        name_to_color=name_to_color,
    )


def _augment_three_payload(
    *,
    payload: dict[str, object],
    legend_tree: list[dict[str, object]] | None = None,
    focus_targets: Mapping[str, dict[str, list[float]]] | None = None,
    hidden_flat_legend_labels: set[str] | None = None,
    collisions: list[dict[str, object]] | None = None,
    edit_wells: list[dict[str, object]] | None = None,
    extra_labels: list[dict[str, object]] | None = None,
    extra_meshes: list[dict[str, object]] | None = None,
    extra_legend_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return ptc_three_overrides.augment_three_payload(
        payload=payload,
        legend_tree=legend_tree,
        focus_targets=focus_targets,
        hidden_flat_legend_labels=hidden_flat_legend_labels,
        collisions=collisions,
        edit_wells=edit_wells,
        extra_labels=extra_labels,
        extra_meshes=extra_meshes,
        extra_legend_items=extra_legend_items,
    )


def _all_wells_three_payload(
    successes: list[SuccessfulWellPlan],
    *,
    target_only_wells: list[_TargetOnlyWell] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    name_to_color: dict[str, str] | None = None,
    focus_well_names: tuple[str, ...] = (),
    render_mode: str = WT_3D_RENDER_FAST,
) -> dict[str, object]:
    resolved_render_mode = _resolve_3d_render_mode(
        requested_mode=render_mode,
        calculated_well_count=len(successes),
        reference_wells=reference_wells,
    )
    return ptc_three_builders.all_wells_three_payload(
        successes,
        target_only_wells=target_only_wells,
        reference_wells=reference_wells,
        name_to_color=name_to_color,
        focus_well_names=focus_well_names,
        render_mode=resolved_render_mode,
        fallback_color=_well_color,
    )


def _all_wells_anticollision_three_payload(
    analysis: AntiCollisionAnalysis,
    *,
    previous_successes_by_name: Mapping[str, SuccessfulWellPlan] | None = None,
    focus_well_names: tuple[str, ...] = (),
    render_mode: str = WT_3D_RENDER_FAST,
) -> dict[str, object]:
    reference_wells = tuple(
        well for well in analysis.wells if bool(well.is_reference_only)
    )
    resolved_render_mode = _resolve_3d_render_mode(
        requested_mode=render_mode,
        calculated_well_count=len(
            [well for well in analysis.wells if not bool(well.is_reference_only)]
        ),
        reference_wells=reference_wells,
    )
    return ptc_three_builders.anticollision_three_payload(
        analysis,
        previous_successes_by_name=previous_successes_by_name,
        focus_well_names=focus_well_names,
        render_mode=resolved_render_mode,
    )


def _render_three_payload(
    *,
    container: object,
    payload: dict[str, object],
    height: int,
    payload_overrides: dict[str, object] | None = None,
) -> None:
    if payload_overrides:
        payload = _augment_three_payload(
            payload=payload,
            legend_tree=list(payload_overrides.get("legend_tree") or []),
            focus_targets=payload_overrides.get("focus_targets"),
            hidden_flat_legend_labels=set(
                str(item)
                for item in (
                    payload_overrides.get("hidden_flat_legend_labels") or set()
                )
            ),
            collisions=payload_overrides.get("collisions"),
            edit_wells=payload_overrides.get("edit_wells"),
            extra_labels=payload_overrides.get("extra_labels"),
            extra_meshes=payload_overrides.get("extra_meshes"),
            extra_legend_items=payload_overrides.get("extra_legend_items"),
        )
    with container:
        edit_event = render_local_three_scene(
            payload,
            height=height,
            instance_token=int(st.session_state.get("wt_three_viewer_nonce", 0)),
            key=str(
                (payload_overrides or {}).get("component_key")
                or payload.get("title")
                or f"three-{height}"
            ),
        )
    if _handle_three_edit_event(edit_event):
        st.rerun()


def _build_edit_wells_payload(
    successes: list[SuccessfulWellPlan],
    name_to_color: Mapping[str, str],
) -> list[dict[str, object]]:
    return ptc_three_overrides.build_edit_wells_payload(
        successes,
        name_to_color,
    )


def _trajectory_three_payload_overrides(
    *,
    records: list[WelltrackRecord],
    successes: list[SuccessfulWellPlan],
    target_only_wells: list[_TargetOnlyWell],
    name_to_color: Mapping[str, str],
) -> dict[str, object]:
    return ptc_three_overrides.trajectory_three_payload_overrides(
        st.session_state,
        records=records,
        successes=successes,
        target_only_wells=target_only_wells,
        name_to_color=name_to_color,
    )


def _anticollision_three_payload_overrides(
    *,
    records: list[WelltrackRecord],
    analysis: AntiCollisionAnalysis,
    successes: list[SuccessfulWellPlan] | None = None,
) -> dict[str, object]:
    return ptc_three_overrides.anticollision_three_payload_overrides(
        st.session_state,
        records=records,
        analysis=analysis,
        successes=successes,
    )


def _build_anti_collision_analysis(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
) -> AntiCollisionAnalysis:
    color_map = {
        str(item.name): (name_to_color or {}).get(str(item.name), _well_color(index))
        for index, item in enumerate(successes)
    }
    return build_anti_collision_analysis_for_successes_shared(
        successes,
        model=model,
        name_to_color=color_map,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
    )


def _reference_uncertainty_models_from_state(
    reference_wells: tuple[ImportedTrajectoryWell, ...],
) -> dict[str, PlanningUncertaintyModel]:
    return ptc_anticollision_params.reference_uncertainty_models_from_state(
        reference_wells
    )


def _anti_collision_cache_key(
    *,
    successes: list[SuccessfulWellPlan],
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None,
    reference_wells: tuple[ImportedTrajectoryWell, ...],
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
) -> str:
    digest = hashlib.blake2b(digest_size=20)
    _update_uncertainty_model_cache_digest(digest, model)
    for well_name, reference_model in sorted(
        (reference_uncertainty_models_by_name or {}).items()
    ):
        digest.update(str(well_name).encode("utf-8"))
        _update_uncertainty_model_cache_digest(digest, reference_model)
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
        digest.update(stations_subset.to_numpy(dtype=np.float64, copy=True).tobytes())
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
        digest.update(stations_subset.to_numpy(dtype=np.float64, copy=True).tobytes())
    return digest.hexdigest()


def _update_uncertainty_model_cache_digest(
    digest: Any,
    model: PlanningUncertaintyModel,
) -> None:
    digest.update(str(model.iscwsa_tool_code or "").encode("utf-8"))
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
                float(model.iscwsa_environment.gtot_mps2),
                float(model.iscwsa_environment.mtot_nt),
                float(model.iscwsa_environment.dip_deg),
                float(model.iscwsa_environment.declination_deg),
                float(model.iscwsa_environment.lateral_singularity_inc_deg),
            ],
            dtype=np.float64,
        ).tobytes()
    )


def _store_anticollision_failure_state(
    exc: Exception,
    *,
    started_at: float | None = None,
    log_lines: Iterable[str] = (),
) -> None:
    previous_state = st.session_state.get("wt_anticollision_last_run")
    previous_payload = previous_state if isinstance(previous_state, Mapping) else {}
    merged_log_lines = [str(item) for item in (previous_payload.get("log_lines") or ())]
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
        "recommendation_count": int(previous_payload.get("recommendation_count") or 0),
        "cluster_count": int(previous_payload.get("cluster_count") or 0),
        "status": f"Ошибка: {type(exc).__name__}",
    }


def _cached_anti_collision_view_model(
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    records: list[WelltrackRecord],
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
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
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
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
            reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        )
        _emit(72, "Построение рекомендаций anti-collision.")
        recommendations = build_anti_collision_recommendations(
            analysis,
            well_context_by_name=_build_anticollision_well_contexts(successes),
        )
        _emit(88, "Кластеризация рекомендаций anti-collision.")
        clusters = build_anti_collision_recommendation_clusters(recommendations)
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
            top_cols[2].metric("Пар", f"{int(state_payload.get('pair_count') or 0)}")
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
        if "cluster_count" in state_payload or "overlap_count" in state_payload:
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
            st.caption(f"Статус результата: {str(state_payload.get('status'))}")
        log_lines = tuple(state_payload.get("log_lines") or ())
        if log_lines:
            st.code("\n".join(str(item) for item in log_lines), language="text")


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
    start_md = float(np.clip(md_start_m, float(md_values[0]), float(md_values[-1])))
    end_md = float(np.clip(md_end_m, float(md_values[0]), float(md_values[-1])))
    if end_md <= start_md + SMALL:
        return None

    interior_mask = (md_values > start_md + SMALL) & (md_values < end_md - SMALL)
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


@st.cache_data(show_spinner=False)
def _parse_welltrack_cached(text: str) -> list[WelltrackRecord]:
    return parse_welltrack_text(text)


def _empty_source_table_df() -> pd.DataFrame:
    return ptc_target_import.empty_source_table_df()


def _normalize_source_table_df_for_ui(
    table_df: pd.DataFrame | None,
) -> pd.DataFrame:
    return ptc_target_import.normalize_source_table_df_for_ui(table_df)


def _coerce_source_table_df_columns(table_df: pd.DataFrame) -> pd.DataFrame:
    return ptc_target_import.coerce_source_table_df_columns(table_df)


def _expand_single_column_source_table_df(
    table_df: pd.DataFrame,
) -> pd.DataFrame:
    return ptc_target_import.expand_single_column_source_table_df(table_df)


def _reference_wells_state_key(kind: str) -> str:
    return ptc_reference_state.reference_wells_state_key(kind)


def _reference_source_mode_key(kind: str) -> str:
    return ptc_reference_state.reference_source_mode_key(kind)


def _reference_source_text_key(kind: str) -> str:
    return ptc_reference_state.reference_source_text_key(kind)


def _reference_welltrack_path_key(kind: str) -> str:
    return ptc_reference_state.reference_welltrack_path_key(kind)


def _reference_dev_folder_count_key(kind: str) -> str:
    return ptc_reference_state.reference_dev_folder_count_key(kind)


def _reference_dev_folder_path_key(kind: str, index: int) -> str:
    return ptc_reference_state.reference_dev_folder_path_key(kind, index)


def _reference_dev_folder_paths(kind: str) -> tuple[str, ...]:
    return ptc_reference_state.reference_dev_folder_paths(kind)


def _clear_reference_dev_folder_state(kind: str) -> None:
    ptc_reference_state.clear_reference_dev_folder_state(kind)


def _clear_reference_import_state(kind: str) -> None:
    ptc_reference_state.clear_reference_import_state(
        kind,
        on_clear=lambda: _reset_anticollision_view_state(clear_prepared=True),
    )


def _set_reference_wells_for_kind(
    *,
    kind: str,
    wells: Iterable[ImportedTrajectoryWell],
) -> None:
    ptc_reference_state.set_reference_wells_for_kind(kind=kind, wells=wells)


def _reference_wells_from_state() -> tuple[ImportedTrajectoryWell, ...]:
    return ptc_reference_state.reference_wells_from_state()


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
    ptc_target_import.init_target_source_state_defaults(st.session_state)
    ptc_reference_state.init_reference_state_defaults()
    _apply_profile_defaults(force=False)
    st.session_state.setdefault("wt_ui_defaults_version", 0)

    if int(st.session_state.get("wt_ui_defaults_version", 0)) < WT_UI_DEFAULTS_VERSION:
        _apply_profile_defaults(force=True)
        st.session_state["wt_ui_defaults_version"] = WT_UI_DEFAULTS_VERSION

    st.session_state.setdefault("wt_records", None)
    st.session_state.setdefault("wt_records_original", None)
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
    st.session_state.setdefault("wt_results_all_view_mode", "Anti-collision")
    st.session_state.setdefault("wt_results_focus_pad_id", WT_PAD_FOCUS_ALL)
    st.session_state.setdefault("wt_batch_select_pad_id", "")
    st.session_state.setdefault("wt_3d_render_mode", WT_3D_RENDER_DETAIL)
    st.session_state.pop("wt_3d_backend", None)
    st.session_state.setdefault("wt_three_viewer_nonce", 0)
    st.session_state.setdefault(
        "wt_anticollision_uncertainty_preset", DEFAULT_UNCERTAINTY_PRESET
    )
    st.session_state.setdefault("wt_actual_fund_analysis_view_mode", "По скважинам")
    st.session_state.setdefault("wt_anticollision_last_run", None)
    st.session_state.setdefault("wt_t1_t3_last_resolution", None)
    st.session_state.setdefault("wt_t1_t3_acknowledged_well_names", ())
    st.session_state.setdefault("wt_edit_targets_pending_names", [])
    st.session_state.setdefault("wt_pending_all_wells_results_focus", False)
    pending_all_wells_results_focus = bool(
        st.session_state.get("wt_pending_all_wells_results_focus", False)
    )
    st.session_state["wt_pending_all_wells_results_focus"] = False
    if pending_all_wells_results_focus or _pending_edit_target_names():
        _set_all_wells_results_focus_state()
    if str(st.session_state.get("wt_results_view_mode", "")).strip() not in {
        "Отдельная скважина",
        "Все скважины",
    }:
        st.session_state["wt_results_view_mode"] = "Все скважины"
    if (
        str(st.session_state.get("wt_results_all_view_mode", "")).strip()
        != "Anti-collision"
    ):
        st.session_state["wt_results_all_view_mode"] = "Anti-collision"
    if str(st.session_state.get("wt_3d_render_mode", "")).strip() not in set(
        WT_3D_RENDER_OPTIONS
    ):
        st.session_state["wt_3d_render_mode"] = WT_3D_RENDER_DETAIL
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
        from urllib.parse import unquote as _unquote

        try:
            changes = _json.loads(_unquote(str(raw)))
        except (ValueError, TypeError):
            return
    _apply_edit_targets_changes(changes, source="query")


def _edit_target_point(value: object) -> list[float] | None:
    return ptc_edit_targets.edit_target_point(value)


def _records_with_edit_targets(
    records: Iterable[WelltrackRecord],
    change_map: Mapping[str, Mapping[str, list[float]]],
) -> tuple[list[WelltrackRecord], list[str]]:
    return ptc_edit_targets.records_with_edit_targets(
        records=records,
        change_map=change_map,
    )


def _unique_well_names(names: Iterable[object]) -> list[str]:
    return ptc_edit_targets.unique_well_names(names)


def _pending_edit_target_names() -> list[str]:
    return ptc_edit_targets.pending_edit_target_names(st.session_state)


def _invalidate_results_for_edited_targets(
    *,
    records: Iterable[WelltrackRecord],
    edited_names: Iterable[str],
) -> None:
    ptc_edit_targets.invalidate_results_for_edited_targets(
        st.session_state,
        records=records,
        edited_names=edited_names,
        base_row_factory=WelltrackBatchPlanner._base_row,
    )


def _apply_edit_targets_changes(
    changes: object,
    *,
    source: str = "3d",
) -> list[str]:
    return ptc_edit_targets.apply_edit_targets_changes(
        st.session_state,
        changes,
        source=source,
        base_row_factory=WelltrackBatchPlanner._base_row,
    )


def _handle_three_edit_event(event: object) -> bool:
    return ptc_edit_targets.handle_three_edit_event(
        st.session_state,
        event,
        apply_changes=lambda changes, source: _apply_edit_targets_changes(
            changes,
            source=source,
        ),
        bump_three_viewer_nonce=_bump_three_viewer_nonce,
    )


def _clear_results() -> None:
    st.session_state["wt_summary_rows"] = None
    st.session_state["wt_successes"] = None
    st.session_state["wt_pending_selected_names"] = None
    st.session_state["wt_last_error"] = ""
    st.session_state["wt_last_run_at"] = ""
    st.session_state["wt_last_runtime_s"] = None
    st.session_state["wt_last_run_log_lines"] = []
    st.session_state["wt_results_view_mode"] = "Все скважины"
    st.session_state["wt_results_all_view_mode"] = "Anti-collision"
    st.session_state["wt_prepared_well_overrides"] = {}
    st.session_state["wt_prepared_override_message"] = ""
    st.session_state["wt_prepared_recommendation_id"] = ""
    st.session_state["wt_anticollision_prepared_cluster_id"] = ""
    st.session_state["wt_prepared_recommendation_snapshot"] = None
    st.session_state["wt_last_anticollision_resolution"] = None
    st.session_state["wt_last_anticollision_previous_successes"] = {}
    st.session_state["wt_anticollision_analysis_cache"] = {}
    st.session_state["wt_edit_targets_pending_names"] = []
    st.session_state["wt_edit_targets_highlight_names"] = []


def _set_all_wells_results_focus_state() -> None:
    st.session_state["wt_results_view_mode"] = "Все скважины"
    st.session_state["wt_results_all_view_mode"] = "Anti-collision"
    st.session_state["wt_3d_render_mode"] = WT_3D_RENDER_DETAIL
    st.session_state.pop("wt_3d_backend", None)


def _queue_all_wells_results_focus() -> None:
    ptc_edit_targets.queue_all_wells_results_focus(st.session_state)


def _focus_all_wells_anticollision_results() -> None:
    _set_all_wells_results_focus_state()


def _focus_all_wells_trajectory_results() -> None:
    _set_all_wells_results_focus_state()


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
                    tolerance_legacy_value = float(st.session_state[legacy_key])
                except (TypeError, ValueError):
                    tolerance_legacy_value = None
            del st.session_state[legacy_key]
            legacy_found = True
    if tolerance_legacy_value is not None:
        st.session_state.setdefault("wt_cfg_lateral_tol", float(tolerance_legacy_value))
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
    return ptc_welltrack_io.decode_welltrack_payload(
        raw_payload,
        source_label=source_label,
        info=st.info,
        warning=st.warning,
    )


def _read_welltrack_file(path_text: str) -> str:
    return ptc_welltrack_io.read_welltrack_file(
        path_text,
        info=st.info,
        warning=st.warning,
        error=st.error,
    )


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
        str(item.name): _well_color(index) for index, item in enumerate(successes)
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
                    "Z: %{customdata[0]:.2f} m<br>"
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
            x_focus_arrays.append(np.array([surface.x, t1.x, t3.x], dtype=float))
            y_focus_arrays.append(np.array([surface.y, t1.y, t3.y], dtype=float))
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
                customdata=np.column_stack([stations["MD_m"].to_numpy(dtype=float)]),
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
        (np.concatenate(x_focus_arrays) if x_focus_arrays else np.concatenate(x_arrays))
        if x_arrays
        else np.array([0.0], dtype=float)
    )
    y_values = (
        (np.concatenate(y_focus_arrays) if y_focus_arrays else np.concatenate(y_arrays))
        if y_arrays
        else np.array([0.0], dtype=float)
    )
    x_range, y_range = equalized_xy_ranges(x_values=x_values, y_values=y_values)
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
        previous_success = (previous_successes_by_name or {}).get(str(well.name))
        if previous_success is not None and not previous_success.stations.empty:
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
                    np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float)
                )
                y_focus_arrays.append(
                    np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float)
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
                name="Зоны пересечений",
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
            {str(corridor.well_a), str(corridor.well_b)}.intersection(focus_set)
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
                name="Конфликтные участки ствола",
                legendgroup="collision_path",
                showlegend=not segment_legend_added,
                line={"width": 7, "color": "rgb(198, 40, 40)"},
                customdata=np.column_stack([md_segment]),
                hovertemplate=(
                    f"{segment.well_name}<br>"
                    "MD: %{customdata[0]:.2f} м"
                    "<extra>Конфликтные участки ствола</extra>"
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
        (np.concatenate(x_focus_arrays) if x_focus_arrays else np.concatenate(x_arrays))
        if x_arrays
        else np.array([0.0], dtype=float)
    )
    y_values = (
        (np.concatenate(y_focus_arrays) if y_focus_arrays else np.concatenate(y_arrays))
        if y_arrays
        else np.array([0.0], dtype=float)
    )
    x_range, y_range = equalized_xy_ranges(x_values=x_values, y_values=y_values)
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


def _build_config_form(
    binding: CalcParamBinding = WT_CALC_PARAMS,
    *,
    title: str = "Параметры расчета",
    on_change: Callable[[], None] | None = None,
) -> TrajectoryConfig:
    binding.render_block(title=title, on_change=on_change)
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
            str(recommendation_snapshot.get("expected_maneuver", "—")).strip() or "—"
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
        optimization_mode = str(update_fields.get("optimization_mode", "")).strip()
        optimization_label = optimization_display_label(optimization_mode)
        rows.append(
            {
                "Порядок": (
                    "—"
                    if int(order_by_well.get(well_name, 0)) <= 0
                    else str(int(order_by_well[well_name]))
                ),
                "Скважина": str(well_name),
                "Маневр": str(maneuver_by_well.get(well_name, "—")).strip() or "—",
                "Оптимизация": optimization_label,
                "SF до": _format_sf_value(recommendation_snapshot.get("before_sf")),
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
        optimization_mode = str(update_fields.get("optimization_mode", "")).strip()
        rows.append(
            {
                "Скважина": str(well_name),
                "Локальный режим": optimization_display_label(optimization_mode),
                "Источник": str(payload.get("source", "—")).strip() or "—",
                "Маневр": "—",
            }
        )
    maneuver_by_well = {
        str(row.get("Скважина", "")).strip(): str(row.get("Маневр", "—")).strip() or "—"
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
    kop_function = kop_min_vertical_function_from_state(prefix=WT_CALC_PARAMS.prefix)
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
                    config = config.validated_copy(kop_min_vertical_m=evaluated_kop_m)
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
    success_by_name = {str(item.name): item for item in (current_successes or [])}
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
                        str(name): dict(value) for name, value in prepared.items()
                    }
                updated_payload = dict(updated_prepared.get(well_name, {}))
                updated_payload["optimization_context"] = rebuilt_context
                updated_prepared[well_name] = updated_payload
    if updated_prepared is not None:
        st.session_state["wt_prepared_well_overrides"] = updated_prepared
    return context_map


def _selected_execution_order(selected_names: list[str]) -> list[str]:
    ordered_selected = list(dict.fromkeys(str(name) for name in selected_names))
    snapshot = dict(st.session_state.get("wt_prepared_recommendation_snapshot") or {})
    selected_set = set(ordered_selected)
    action_steps = tuple(snapshot.get("action_steps", ()) or ())
    prioritized = [
        str(dict(step).get("well_name", "")).strip()
        for step in action_steps
        if str(dict(step).get("well_name", "")).strip() in selected_set
    ]
    prioritized = list(dict.fromkeys(name for name in prioritized if name))
    remainder = [name for name in ordered_selected if name not in set(prioritized)]
    return [*prioritized, *remainder]


def _build_anticollision_well_contexts(
    successes: list[SuccessfulWellPlan],
) -> dict[str, AntiCollisionWellContext]:
    return build_anticollision_well_contexts_shared(successes)


def _recommendation_snapshot(
    recommendation: AntiCollisionRecommendation,
) -> dict[str, object]:
    return ptc_anticollision_view.recommendation_snapshot(recommendation)


def _cluster_snapshot(
    cluster: AntiCollisionRecommendationCluster,
    *,
    target_well_names: tuple[str, ...] = (),
    focus_well_names: tuple[str, ...] = (),
) -> dict[str, object]:
    return ptc_anticollision_view.cluster_snapshot(
        cluster,
        target_well_names=target_well_names,
        focus_well_names=focus_well_names,
    )


def _format_sf_value(value: object) -> str:
    return ptc_anticollision_view.format_sf_value(value)


def _format_overlap_value(value: object) -> str:
    return ptc_anticollision_view.format_overlap_value(value)


def _md_interval_label(md_start_m: float, md_end_m: float) -> str:
    return ptc_anticollision_view.md_interval_label(md_start_m, md_end_m)


def _sf_help_markdown() -> str:
    return ptc_anticollision_view.sf_help_markdown()


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
                float(item.worst_separation_factor) for item in current_clusters
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
                    "current_overlap_m": float(recommendation.max_overlap_depth_m),
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
                    cluster_display_label(cluster) for cluster in current_clusters
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
    explicit_wells = tuple(str(name) for name in snapshot.get("well_names", ()) or ())
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
        reference_uncertainty_models_by_name=(
            _reference_uncertainty_models_from_state(reference_wells)
        ),
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
    resolution = dict(st.session_state.get("wt_last_anticollision_resolution") or {})
    if not resolution:
        return
    resolution_kind = (
        str(resolution.get("kind", "recommendation")).strip() or "recommendation"
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
                            "Ожидаемый маневр": str(item.get("expected_maneuver", "—")),
                            "Область": str(item.get("area_label", "—")),
                            "Интервал A, м": _md_interval_label(
                                float(item.get("md_a_start_m", 0.0)),
                                float(item.get("md_a_end_m", 0.0)),
                            ),
                            "Интервал B, м": _md_interval_label(
                                float(item.get("md_b_start_m", 0.0)),
                                float(item.get("md_b_end_m", 0.0)),
                            ),
                            "SF сейчас": _format_sf_value(item.get("current_sf")),
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
    prepared, skipped_wells, action_steps = _build_recommendation_prepared_overrides(
        recommendation,
        successes=successes,
        uncertainty_model=uncertainty_model,
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
    reference_wells = _reference_wells_from_state()
    return build_prepared_optimization_context_shared(
        recommendation=recommendation,
        moving_success=moving_success,
        reference_success=reference_success,
        uncertainty_model=uncertainty_model,
        all_successes=all_successes,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=(
            _reference_uncertainty_models_from_state(reference_wells)
        ),
    )


def _build_cluster_prepared_overrides(
    cluster: AntiCollisionRecommendationCluster,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    reference_wells = _reference_wells_from_state()
    return build_cluster_prepared_overrides_shared(
        cluster,
        successes=successes,
        uncertainty_model=uncertainty_model,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=(
            _reference_uncertainty_models_from_state(reference_wells)
        ),
    )


def _build_recommendation_prepared_overrides(
    recommendation: AntiCollisionRecommendation,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> tuple[dict[str, dict[str, object]], list[str], tuple[dict[str, object], ...]]:
    reference_wells = _reference_wells_from_state()
    return build_recommendation_prepared_overrides_shared(
        recommendation,
        successes=successes,
        uncertainty_model=uncertainty_model,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=(
            _reference_uncertainty_models_from_state(reference_wells)
        ),
    )


def _prepare_rerun_from_cluster(
    cluster: AntiCollisionRecommendationCluster,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    target_well_names: tuple[str, ...] = (),
    focus_well_names: tuple[str, ...] = (),
) -> None:
    if (not bool(cluster.can_prepare_rerun)) or cluster.blocking_advisory is not None:
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
            if str(step.well_name) in set(target_well_names or tuple(prepared.keys()))
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
    if str(st.session_state.get("wt_source_format", "")).strip() not in set(
        WT_SOURCE_FORMAT_OPTIONS
    ):
        st.session_state["wt_source_format"] = WT_SOURCE_FORMAT_WELLTRACK

    source_format = st.radio(
        "Формат импорта",
        options=list(WT_SOURCE_FORMAT_OPTIONS),
        horizontal=True,
        key="wt_source_format",
    )

    if source_format == WT_SOURCE_FORMAT_WELLTRACK:
        if str(st.session_state.get("wt_source_mode", "")).strip() not in set(
            WT_SOURCE_WELLTRACK_MODES
        ):
            st.session_state["wt_source_mode"] = WT_SOURCE_MODE_FILE_PATH
        source_mode = st.radio(
            "Способ загрузки WELLTRACK",
            options=list(WT_SOURCE_WELLTRACK_MODES),
            horizontal=True,
            key="wt_source_mode",
        )
    else:
        source_mode = WT_SOURCE_MODE_TARGET_TABLE
        st.session_state["wt_source_mode"] = WT_SOURCE_MODE_TARGET_TABLE

    if source_mode == WT_SOURCE_MODE_FILE_PATH:
        source_path = st.text_input(
            "Путь к файлу WELLTRACK",
            key="wt_source_path",
            placeholder="tests/test_data/WELLTRACKS3.INC",
        )
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_text=_read_welltrack_file(source_path),
        )

    if source_mode == WT_SOURCE_MODE_UPLOAD:
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

    if source_mode == WT_SOURCE_MODE_INLINE_TEXT:
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_text=st.text_area(
                "Текст WELLTRACK",
                key="wt_source_inline",
                height=220,
                placeholder="WELLTRACK 'WELL-1'\n457091 891257 -63.2 0\n457707 890374 1852 1\n/",
            ),
        )

    with st.expander("Таблица точек целей", expanded=True):
        note_col, clear_col = st.columns(
            [5.0, 1.2], gap="small", vertical_alignment="bottom"
        )
        with note_col:
            st.caption(
                "Вставьте таблицу в формате `Wellname`, `Point`, `X`, `Y`, `Z`. "
                "Поддерживается copy/paste из Excel. Для обычной скважины "
                "`Point` принимает `S`, `t1`, `t3`. Для пилота используйте имя "
                "`wellname_PL`: точки `S`, `PL1`, `PL2`, ...; часть `wellname` "
                "должна совпадать с основной скважиной."
            )
        with clear_col:
            if st.button(
                "Очистить",
                key="wt_source_table_clear",
                icon=":material/delete:",
                width="stretch",
            ):
                st.session_state["wt_source_table_df"] = _empty_source_table_df()
                st.session_state["wt_source_table_editor_nonce"] = (
                    int(st.session_state.get("wt_source_table_editor_nonce", 0)) + 1
                )
                st.rerun()
        source_table_df = _normalize_source_table_df_for_ui(
            st.session_state.get("wt_source_table_df", _empty_source_table_df())
        )
        edited_table = st.data_editor(
            source_table_df,
            key=f"wt_source_table_editor_{int(st.session_state.get('wt_source_table_editor_nonce', 0))}",
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Wellname": st.column_config.TextColumn("Wellname"),
                "Point": st.column_config.TextColumn("Point"),
                "X": st.column_config.NumberColumn("X"),
                "Y": st.column_config.NumberColumn("Y"),
                "Z": st.column_config.NumberColumn("Z"),
            },
        )
        st.session_state["wt_source_table_df"] = _normalize_source_table_df_for_ui(
            pd.DataFrame(edited_table)
        )

    return _WelltrackSourcePayload(
        mode=WT_SOURCE_MODE_TARGET_TABLE,
        table_rows=pd.DataFrame(st.session_state["wt_source_table_df"]),
    )


def _store_parsed_records(records: list[WelltrackRecord]) -> bool:
    result = ptc_target_import.store_imported_records(
        st.session_state,
        records=list(records),
        loaded_at_text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        clear_t1_t3_order_state=_clear_t1_t3_order_resolution_state,
        clear_pad_state=_clear_pad_state,
        clear_results=_clear_results,
        auto_apply_pad_layout=_auto_apply_pad_layout_if_shared_surface,
    )
    _set_import_pilot_fixed_slots(
        records=list(st.session_state.get("wt_records_original") or records)
    )
    return bool(result.auto_layout_applied)


def _auto_apply_pad_layout_if_shared_surface(
    records: list[WelltrackRecord],
) -> bool:
    pads = _ensure_pad_configs(base_records=list(records))
    _set_import_pilot_fixed_slots(records=list(records), pads=pads)
    metadata = dict(st.session_state.get("wt_pad_detected_meta", {}))
    auto_layout_pad_ids = {
        str(pad.pad_id)
        for pad in pads
        if len(pad.wells) > 1
        and not bool(
            getattr(metadata.get(str(pad.pad_id)), "source_surfaces_defined", False)
        )
    }
    if not auto_layout_pad_ids:
        return False
    plan_map = _build_pad_plan_map(pads)
    if not any(str(pad_id) in auto_layout_pad_ids for pad_id in plan_map):
        return False
    updated_records = sync_pilot_surfaces_to_parents(
        apply_pad_layout(
            records=list(records),
            pads=pads,
            plan_by_pad_id=plan_map,
        )
    )
    if updated_records == list(records):
        return False
    st.session_state["wt_records"] = list(updated_records)
    st.session_state["wt_pad_last_applied_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.session_state["wt_pad_auto_applied_on_import"] = True
    return True


def _set_import_pilot_fixed_slots(
    *,
    records: list[WelltrackRecord],
    pads: list[WellPad] | None = None,
) -> bool:
    pilot_parent_keys = {
        well_name_key(parent_name_for_pilot(record.name))
        for record in records
        if is_pilot_name(record.name)
    }
    if not pilot_parent_keys:
        return False
    resolved_pads = pads if pads is not None else _ensure_pad_configs(base_records=records)
    config_raw = st.session_state.get("wt_pad_configs", {})
    config_map = dict(config_raw if isinstance(config_raw, Mapping) else {})
    changed = False
    for pad in resolved_pads:
        pilot_parent_names = [
            str(well.name)
            for well in pad.wells
            if well_name_key(well.name) in pilot_parent_keys
        ]
        if not pilot_parent_names:
            continue
        preferred_name = pilot_parent_names[0]
        pad_id = str(pad.pad_id)
        current_raw = config_map.get(pad_id, _pad_config_defaults(pad))
        current = dict(current_raw if isinstance(current_raw, Mapping) else {})
        current_slots = _pad_fixed_slots_from_config(pad=pad, config=current)
        next_slots = (
            (1, preferred_name),
            *(
                (int(slot), str(name))
                for slot, name in current_slots
                if int(slot) != 1 and str(name) != preferred_name
            ),
        )
        if current_slots == next_slots:
            continue
        current["fixed_slots"] = next_slots
        config_map[pad_id] = current
        changed = True
    if changed:
        st.session_state["wt_pad_configs"] = config_map
    return changed


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
        ptc_target_import.clear_target_import_flow_state(
            st.session_state,
            reference_well_state_keys=(
                _reference_wells_state_key(REFERENCE_WELL_ACTUAL),
                _reference_wells_state_key(REFERENCE_WELL_APPROVED),
            ),
            clear_t1_t3_order_state=_clear_t1_t3_order_resolution_state,
            clear_pad_state=_clear_pad_state,
            clear_results=_clear_results,
        )
        st.rerun()

    if not parse_clicked:
        return

    try:
        operation = ptc_target_import.build_target_import_operation(
            source_payload,
            parse_welltrack_text_func=_parse_welltrack_cached,
        )
    except ptc_target_import.TargetImportEmptySourceError as exc:
        st.warning(str(exc))
        return

    with st.status(operation.status_label, expanded=True) as status:
        started = perf_counter()
        try:
            status.write(operation.progress_message)
            records = operation.parse_records()
            auto_layout_applied = _store_parsed_records(records=records)
            status.write(operation.count_message(len(records)))
            if auto_layout_applied:
                status.write(ptc_target_import.AUTO_LAYOUT_APPLIED_MESSAGE)
            elapsed = perf_counter() - started
            status.update(
                label=operation.success_label(elapsed),
                state="complete",
                expanded=False,
            )
        except WelltrackParseError as exc:
            ptc_target_import.reset_failed_import_state(
                st.session_state,
                error_message=str(exc),
                clear_t1_t3_order_state=_clear_t1_t3_order_resolution_state,
                clear_pad_state=_clear_pad_state,
            )
            status.write(str(exc))
            status.update(label=operation.error_label, state="error", expanded=True)


def _render_records_overview(records: list[WelltrackRecord]) -> None:
    parsed_df = _records_overview_dataframe(records)
    problem_count = int(sum(str(item) != "✅" for item in parsed_df["Статус"].tolist()))
    well_count = int(sum(not is_pilot_name(record.name) for record in records))
    pilot_count = int(sum(is_pilot_name(record.name) for record in records))
    x1, x2, x3 = st.columns(3, gap="small")
    x1.metric("Скважин", f"{well_count}")
    x2.metric("Пилотов", f"{pilot_count}")
    x3.metric("Проблем", f"{problem_count}")

    with st.expander(
        "Статус загрузки целей",
        expanded=bool(problem_count > 0),
    ):
        st.dataframe(
            arrow_safe_text_dataframe(parsed_df),
            width="stretch",
            hide_index=True,
            column_config={
                "Скважина": st.column_config.TextColumn("Скважина", width="medium"),
                "Точек": st.column_config.NumberColumn(
                    "Точек",
                    format="%d",
                    width="small",
                    help="Считаются только целевые точки `t1/t3`, без устья `S`.",
                ),
                "Отход t1, м": st.column_config.NumberColumn(
                    "Отход t1, м",
                    format="%.2f",
                    width="small",
                    help="Горизонтальное расстояние от устья `S` до точки `t1`.",
                ),
                "Длина ГС, м": st.column_config.NumberColumn(
                    "Длина ГС, м",
                    format="%.2f",
                    width="small",
                    help="Пространственное расстояние между точками `t1` и `t3`.",
                ),
                "Примечание": st.column_config.TextColumn(
                    "Примечание",
                    width="small",
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
    return ptc_target_records.records_overview_dataframe(
        records,
        wellhead_z_tolerance_m=WT_IMPORT_WELLHEAD_Z_TOLERANCE_M,
    )


def _record_target_point_count(record: WelltrackRecord) -> int:
    return ptc_target_records.record_target_point_count(record)


def _record_has_surface_like_point(record: WelltrackRecord) -> bool:
    return ptc_target_records.record_has_surface_like_point(
        record,
        wellhead_z_tolerance_m=WT_IMPORT_WELLHEAD_Z_TOLERANCE_M,
    )


def _record_first_point_is_surface_like(record: WelltrackRecord) -> bool:
    return ptc_target_records.record_first_point_is_surface_like(
        record,
        wellhead_z_tolerance_m=WT_IMPORT_WELLHEAD_Z_TOLERANCE_M,
    )


def _record_has_strictly_increasing_md(record: WelltrackRecord) -> bool:
    return ptc_target_records.record_has_strictly_increasing_md(record)


def _record_import_problem_text(record: WelltrackRecord) -> str:
    return ptc_target_records.record_import_problem_text(
        record,
        wellhead_z_tolerance_m=WT_IMPORT_WELLHEAD_Z_TOLERANCE_M,
    )


def _record_is_ready_for_calc(record: WelltrackRecord) -> bool:
    return ptc_target_records.record_is_ready_for_calc(
        record,
        wellhead_z_tolerance_m=WT_IMPORT_WELLHEAD_Z_TOLERANCE_M,
    )


def _reference_kind_title(kind: str) -> str:
    return ptc_reference_state.reference_kind_title(kind)


def _reference_kind_wells(kind: str) -> tuple[ImportedTrajectoryWell, ...]:
    return ptc_reference_state.reference_kind_wells(kind)


ACTUAL_FUND_ZONE_COLORS: dict[str, str] = {
    "vertical": "#2563EB",
    "build1": "#F59E0B",
    "hold": "#16A34A",
    "build2": "#8B5CF6",
    "horizontal": "#0F766E",
}


def _actual_fund_zone_color(zone_key: str) -> str:
    return ACTUAL_FUND_ZONE_COLORS.get(str(zone_key), "#475569")


def _actual_fund_interp_row(survey: pd.DataFrame, md_m: float) -> dict[str, float]:
    md_values = survey["MD_m"].to_numpy(dtype=float)
    return {
        column: float(
            np.interp(float(md_m), md_values, survey[column].to_numpy(dtype=float))
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
                pd.DataFrame([_actual_fund_interp_row(survey, float(start_md_m))]),
                interval,
            ],
            ignore_index=True,
        )
    if abs(float(interval["MD_m"].iloc[-1]) - float(end_md_m)) > SMALL:
        interval = pd.concat(
            [
                interval,
                pd.DataFrame([_actual_fund_interp_row(survey, float(end_md_m))]),
            ],
            ignore_index=True,
        )
    return interval.sort_values("MD_m").reset_index(drop=True)


def _actual_fund_kop_marker(
    detail: ActualFundWellAnalysis,
) -> dict[str, float] | None:
    if detail.metrics.kop_md_m is None:
        return None
    return _actual_fund_interp_row(detail.survey, float(detail.metrics.kop_md_m))


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
    return float(np.hypot(end_x - float(entry["X_m"]), end_y - float(entry["Y_m"])))


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
            if cluster_by_well.get(str(getattr(item, "name", ""))) == cluster_id
        ]
        if not cluster_items:
            continue
        fig.add_trace(
            go.Scatter(
                x=[float(item.horizontal_entry_tvd_m) for item in cluster_items],
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
        anchor_depths = np.asarray(kop_function.anchor_depths_tvd_m, dtype=float)
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
        x_arrays.append(np.asarray([float(horizontal_marker["X_m"])], dtype=float))
        y_arrays.append(np.asarray([float(horizontal_marker["Y_m"])], dtype=float))
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
    detail = next(item for item in analyses if str(item.name) == str(selected_name))
    metrics = detail.metrics
    lateral_from_horizontal_entry_m = _actual_fund_lateral_from_horizontal_entry_m(
        detail
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
    c2.plotly_chart(_actual_fund_vertical_profile_figure(detail), width="stretch")
    st.dataframe(
        arrow_safe_text_dataframe(pd.DataFrame(_actual_fund_zone_table_rows(detail))),
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
    eligible_metrics = [item for item in metrics if bool(item.is_analysis_eligible)]
    excluded_horizontal_metrics = [
        item
        for item in metrics
        if bool(item.is_horizontal) and not bool(item.is_analysis_eligible)
    ]
    pad_count = len(
        {str(item.pad_group) for item in eligible_metrics if str(item.pad_group) != "—"}
    )
    depth_clusters = summarize_actual_fund_by_depth(eligible_metrics)
    kop_depth_function = build_actual_fund_kop_depth_function(eligible_metrics)
    eligible_kop_values = [
        float(item.kop_md_m) for item in eligible_metrics if item.kop_md_m is not None
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
                    st.toast("Функция KOP / TVD применена к параметрам расчёта.")
                    st.rerun()
                if kop_min_vertical_mode(WT_CALC_PARAMS.prefix) != "constant":
                    if st.button(
                        "Вернуть фиксированный KOP",
                        key="wt_clear_actual_fund_kop_depth_function",
                        icon=":material/looks_one:",
                        width="stretch",
                    ):
                        clear_kop_min_vertical_function(prefix=WT_CALC_PARAMS.prefix)
                        st.toast("Возвращён режим фиксированного KOP.")
                        st.rerun()
            if (
                kop_min_vertical_function_from_state(prefix=WT_CALC_PARAMS.prefix)
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


def _render_raw_records_table(records: list[WelltrackRecord]) -> None:
    highlight_names = {
        str(name)
        for name in (st.session_state.get("wt_edit_targets_highlight_names") or [])
        if str(name).strip()
    }
    with st.expander(
        "Текущие точки скважин (используются в расчете, включая обновленные устья S)",
        expanded=bool(highlight_names),
    ):
        if highlight_names:
            st.success(
                "Изменённые в 3D-редакторе точки t1/t3 подсвечены. "
                "Запустите расчёт для обновления траекторий."
            )
        raw_df = arrow_safe_text_dataframe(
            ptc_target_records.raw_records_dataframe(records)
        )
        if highlight_names and not raw_df.empty:
            highlight_mask = raw_df["Скважина"].astype(str).isin(
                highlight_names
            ) & raw_df["Точка"].astype(str).isin({"t1", "t3"})

            def _highlight_edit_rows(row: pd.Series) -> list[str]:
                if bool(highlight_mask.loc[row.name]):
                    return [
                        "background-color: rgba(34, 197, 94, 0.14); "
                        "font-weight: 600;"
                    ] * len(row)
                return [""] * len(row)

            table_payload = raw_df.style.apply(_highlight_edit_rows, axis=1)
        else:
            table_payload = raw_df
        st.dataframe(
            table_payload,
            width="stretch",
            hide_index=True,
        )


def _render_t1_t3_order_panel(
    records: list[WelltrackRecord],
    *,
    border: bool = True,
) -> None:
    visible_records = visible_well_records(records)
    panel_context = st.container(border=True) if border else nullcontext()
    with panel_context:
        resolution_message = _t1_t3_order_resolution_message()
        if resolution_message is not None:
            level, message = resolution_message
            if level == "success":
                st.success(message)
            else:
                st.info(message)
        detected_issues = detect_t1_t3_order_issues(
            visible_records, min_delta_m=WT_T1T3_MIN_DELTA_M
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
                    st.session_state.get(f"wt_t1_t3_fix_{str(item.well_name)}", True)
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
            st.toast(f"Порядок t1/t3 исправлен для {len(target_names)} скважин.")
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
    well_names = tuple(str(item) for item in (resolution.get("well_names") or ()))
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


def _pad_config_defaults(pad: WellPad) -> dict[str, object]:
    return ptc_pad_state.pad_config_defaults(pad)


def _pad_fixed_slots_from_config(
    *,
    pad: WellPad,
    config: Mapping[str, object],
) -> tuple[tuple[int, str], ...]:
    return ptc_pad_state.pad_fixed_slots_from_config(pad=pad, config=config)


def _pad_fixed_slots_editor_rows(
    *,
    pad: WellPad,
    config: Mapping[str, object],
) -> pd.DataFrame:
    return ptc_pad_state.pad_fixed_slots_editor_rows(pad=pad, config=config)


def _pad_fixed_slots_from_editor(
    *,
    pad: WellPad,
    editor_value: object,
) -> tuple[tuple[tuple[int, str], ...], list[str]]:
    return ptc_pad_state.pad_fixed_slots_from_editor(
        pad=pad,
        editor_value=editor_value,
    )


def _source_surface_xyz(
    record: WelltrackRecord,
) -> tuple[float, float, float] | None:
    return ptc_pad_state.source_surface_xyz(record)


def _record_midpoint_xyz(
    record: WelltrackRecord,
) -> tuple[float, float, float]:
    return ptc_pad_state.record_midpoint_xyz(record)


def _estimate_surface_pad_axis_deg(
    surface_xyzs: list[tuple[float, float, float]],
) -> float:
    return ptc_pad_state.estimate_surface_pad_axis_deg(surface_xyzs)


def _inferred_surface_spacing_m(
    *,
    surface_xyzs: list[tuple[float, float, float]],
    nds_azimuth_deg: float,
) -> float:
    return ptc_pad_state.inferred_surface_spacing_m(
        surface_xyzs=surface_xyzs,
        nds_azimuth_deg=nds_azimuth_deg,
    )


def _detect_ui_pads(
    records: list[WelltrackRecord],
) -> tuple[list[WellPad], dict[str, _DetectedPadUiMeta]]:
    return ptc_pad_state.detect_ui_pads(records)


def _ensure_pad_configs(base_records: list[WelltrackRecord]) -> list[WellPad]:
    return ptc_pad_state.ensure_pad_configs(
        st.session_state,
        base_records=visible_well_records(base_records),
    )


def _build_pad_plan_map(pads: list[WellPad]) -> dict[str, PadLayoutPlan]:
    return ptc_pad_state.build_pad_plan_map(st.session_state, pads)


def _project_pads_for_ui(records: list[WelltrackRecord]) -> list[WellPad]:
    return ptc_pad_state.project_pads_for_ui(
        st.session_state, visible_well_records(records)
    )


def _pad_display_label(pad: WellPad) -> str:
    return ptc_pad_state.pad_display_label(pad)


def _pad_config_for_ui(pad: WellPad) -> dict[str, object]:
    return ptc_pad_state.pad_config_for_ui(st.session_state, pad)


def _pad_anchor_mode_label(mode: object) -> str:
    return ptc_pad_state.pad_anchor_mode_label(mode)


def _pad_membership(
    records: list[WelltrackRecord],
) -> tuple[list[WellPad], dict[str, str], dict[str, tuple[str, ...]]]:
    return ptc_pad_state.pad_membership(
        st.session_state, visible_well_records(records)
    )


def _normalize_focus_pad_id(
    *,
    records: list[WelltrackRecord],
    requested_pad_id: str | None,
) -> str:
    return ptc_pad_state.normalize_focus_pad_id(
        st.session_state,
        records=visible_well_records(records),
        requested_pad_id=requested_pad_id,
    )


def _focus_pad_well_names(
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str | None,
) -> tuple[str, ...]:
    return ptc_pad_state.focus_pad_well_names(
        st.session_state,
        records=visible_well_records(records),
        focus_pad_id=focus_pad_id,
    )


def _focus_pad_fixed_well_names(
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str | None,
) -> tuple[str, ...]:
    return ptc_pad_state.focus_pad_fixed_well_names(
        st.session_state,
        records=visible_well_records(records),
        focus_pad_id=focus_pad_id,
    )


def _clusters_touching_focus_pad(
    *,
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
    focus_pad_well_names: tuple[str, ...],
) -> tuple[AntiCollisionRecommendationCluster, ...]:
    return ptc_anticollision_view.clusters_touching_focus_pad(
        clusters=clusters,
        focus_pad_well_names=focus_pad_well_names,
    )


def _recommendations_for_clusters(
    *,
    recommendations: tuple[AntiCollisionRecommendation, ...],
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
) -> tuple[AntiCollisionRecommendation, ...]:
    return ptc_anticollision_view.recommendations_for_clusters(
        recommendations=recommendations,
        clusters=clusters,
    )


def _report_rows_from_recommendations(
    recommendations: tuple[AntiCollisionRecommendation, ...],
    analysis: AntiCollisionAnalysis | None = None,
) -> list[dict[str, object]]:
    return ptc_anticollision_view.report_rows_from_recommendations(
        recommendations,
        analysis=analysis,
    )


def _pad_scoped_cluster_target_well_names(
    *,
    cluster: AntiCollisionRecommendationCluster,
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    return ptc_anticollision_view.pad_scoped_cluster_target_well_names(
        cluster=cluster,
        focus_pad_well_names=focus_pad_well_names,
    )


def _pad_scoped_cluster_focus_well_names(
    *,
    cluster: AntiCollisionRecommendationCluster,
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    return ptc_anticollision_view.pad_scoped_cluster_focus_well_names(
        cluster=cluster,
        focus_pad_well_names=focus_pad_well_names,
    )


def _anticollision_focus_well_names(
    *,
    clusters: tuple[AntiCollisionRecommendationCluster, ...],
    focus_pad_well_names: tuple[str, ...],
) -> tuple[str, ...]:
    return ptc_anticollision_view.anticollision_focus_well_names(
        clusters=clusters,
        focus_pad_well_names=focus_pad_well_names,
    )


def _render_pad_layout_panel(records: list[WelltrackRecord]) -> None:
    base_records = st.session_state.get("wt_records_original")
    if base_records is None:
        base_records = list(records)
    pads = _ensure_pad_configs(base_records=list(base_records))
    if not pads:
        return

    with st.container(border=True):
        _render_t1_t3_order_panel(records=records, border=False)
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
        pad_rows = []
        for pad in pads:
            cfg = _pad_config_for_ui(pad)
            pad_rows.append(
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
                    "Авто НДС, deg": float(cfg["nds_azimuth_deg"]),
                    "S X, м": float(cfg["first_surface_x"]),
                    "S Y, м": float(cfg["first_surface_y"]),
                    "S Z, м": float(cfg["first_surface_z"]),
                }
            )
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(pad_rows)),
            width="stretch",
            hide_index=True,
        )

        pad_ids = [str(pad.pad_id) for pad in pads]
        st.selectbox("Выберите куст", options=pad_ids, key="wt_pad_selected_id")
        selected_id = str(st.session_state.get("wt_pad_selected_id", pad_ids[0]))
        selected_pad = next(
            (pad for pad in pads if str(pad.pad_id) == selected_id), pads[0]
        )
        selected_pad_meta = pad_metadata.get(selected_id)
        source_surfaces_defined = bool(
            getattr(selected_pad_meta, "source_surfaces_defined", False)
        )
        config_map = st.session_state.get("wt_pad_configs", {})
        selected_cfg: dict[str, object] = dict(
            config_map.get(selected_id, _pad_config_defaults(selected_pad))
        )
        previous_anchor_mode = str(
            selected_cfg.get("surface_anchor_mode", DEFAULT_PAD_SURFACE_ANCHOR_MODE)
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
                "координаты устьев ниже показаны справочно и не редактируются. "
                "Фиксированный порядок можно задавать отдельно для anti-collision "
                "оптимизации."
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
            step=5.0,
            key=widget_keys["spacing_m"],
            help="Шаг по кусту между соседними устьями скважин.",
            disabled=source_surfaces_defined,
        )
        nds_azimuth_deg = p2.number_input(
            "НДС (азимут), deg",
            min_value=0.0,
            max_value=360.0,
            step=10.0,
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

        fixed_slots = _pad_fixed_slots_from_config(
            pad=selected_pad,
            config=selected_cfg,
        )
        fixed_editor_revision_key = f"wt_pad_fixed_slots_editor_revision_{selected_id}"
        try:
            fixed_editor_revision = int(
                st.session_state.get(fixed_editor_revision_key, 0)
            )
        except (TypeError, ValueError):
            fixed_editor_revision = 0
        fixed_editor_key = (
            f"wt_pad_fixed_slots_editor_{selected_id}_{fixed_editor_revision}"
        )
        st.markdown("**Фиксированный порядок скважин**")
        fixed_editor_df = st.data_editor(
            _pad_fixed_slots_editor_rows(
                pad=selected_pad,
                config=selected_cfg,
            ),
            key=fixed_editor_key,
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            disabled=len(selected_pad.wells) < 2,
            column_config={
                "Позиция": st.column_config.SelectboxColumn(
                    "Позиция",
                    options=list(range(1, int(len(selected_pad.wells)) + 1)),
                    required=False,
                    help="Номер слота на кусте, начиная с 1.",
                ),
                "Скважина": st.column_config.SelectboxColumn(
                    "Скважина",
                    options=sorted(str(well.name) for well in selected_pad.wells),
                    required=False,
                    help="Скважина, которую нужно закрепить в этой позиции.",
                ),
            },
        )
        fixed_slots, fixed_warnings = _pad_fixed_slots_from_editor(
            pad=selected_pad,
            editor_value=fixed_editor_df,
        )
        if fixed_warnings:
            st.warning(" ".join(dict.fromkeys(fixed_warnings)))
        clear_fixed_clicked = st.button(
            "Очистить фиксацию порядка",
            icon=":material/lock_open:",
            width="content",
            disabled=not fixed_slots or len(selected_pad.wells) < 2,
        )
        if clear_fixed_clicked:
            fixed_slots = ()
            selected_cfg["fixed_slots"] = ()
            config_map[selected_id] = selected_cfg
            st.session_state["wt_pad_configs"] = config_map
            st.session_state[fixed_editor_revision_key] = fixed_editor_revision + 1
            st.rerun()
        selected_cfg["fixed_slots"] = fixed_slots
        config_map[selected_id] = selected_cfg
        st.session_state["wt_pad_configs"] = config_map

        ordered_wells = ordered_pad_wells(
            pad=selected_pad,
            nds_azimuth_deg=float(selected_cfg["nds_azimuth_deg"]),
            fixed_slots=fixed_slots,
        )
        angle_rad = np.deg2rad(float(selected_cfg["nds_azimuth_deg"]))
        ux = float(np.sin(angle_rad))
        uy = float(np.cos(angle_rad))
        center_slot_index = 0.5 * float(max(len(ordered_wells) - 1, 0))
        fixed_slot_by_name = {str(name): int(slot) for slot, name in fixed_slots}
        preview_rows: list[dict[str, object]] = []
        for slot_index, well in enumerate(ordered_wells, start=1):
            row = {
                "Порядок": int(slot_index),
                "Скважина": str(well.name),
                "Фиксация": (
                    "Да"
                    if fixed_slot_by_name.get(str(well.name)) == int(slot_index)
                    else "Авто"
                ),
                "Середина t1-t3 X, м": float(well.midpoint_x),
                "Середина t1-t3 Y, м": float(well.midpoint_y),
                "Опора S": _pad_anchor_mode_label(anchor_mode),
            }
            if source_surfaces_defined:
                source_record = next(
                    (item for item in base_records if str(item.name) == str(well.name)),
                    None,
                )
                source_surface = (
                    _source_surface_xyz(source_record)
                    if source_record is not None
                    else None
                )
                row["Текущее S X, м"] = (
                    None if source_surface is None else float(source_surface[0])
                )
                row["Текущее S Y, м"] = (
                    None if source_surface is None else float(source_surface[1])
                )
                row["Текущее S Z, м"] = (
                    None if source_surface is None else float(source_surface[2])
                )
            else:
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER:
                    shift_m = (float(slot_index - 1) - center_slot_index) * float(
                        selected_cfg["spacing_m"]
                    )
                else:
                    shift_m = float(slot_index - 1) * float(selected_cfg["spacing_m"])
                row["Новое S X, м"] = float(
                    selected_cfg["first_surface_x"] + shift_m * ux
                )
                row["Новое S Y, м"] = float(
                    selected_cfg["first_surface_y"] + shift_m * uy
                )
                row["Новое S Z, м"] = float(selected_cfg["first_surface_z"])
            preview_rows.append(row)
        with st.expander(
            "Порядок бурения и координаты устьев на кусте",
            expanded=True,
        ):
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
            updated_records = sync_pilot_surfaces_to_parents(
                apply_pad_layout(
                    records=list(base_records),
                    pads=pads,
                    plan_by_pad_id=plan_map,
                )
            )
            st.session_state["wt_records"] = list(updated_records)
            st.session_state["wt_pad_last_applied_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
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

        # if str(st.session_state.get("wt_pad_last_applied_at", "")):
        #     st.caption(
        #         f"Последнее обновление устьев: {st.session_state['wt_pad_last_applied_at']}"
        #     )


def _sync_selection_state(
    records: list[WelltrackRecord],
) -> tuple[list[str], list[str]]:
    return ptc_batch_run.sync_selection_state(
        st.session_state,
        records=records,
    )


def _render_batch_selection_status(
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]] | None,
) -> None:
    status = ptc_batch_run.batch_selection_status(
        records=records,
        summary_rows=summary_rows,
    )
    if status.has_summary_rows:
        st.caption(
            "Результаты по невыбранным скважинам сохраняются. Для следующего запуска "
            "по умолчанию выделяются нерассчитанные, ошибочные и warning-кейсы."
        )
        return

    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Без замечаний", f"{status.ok_count}")
    c2.metric("С предупреждениями", f"{status.warning_count}")
    c3.metric("С ошибками", f"{status.error_count}")
    c4.metric("Не рассчитаны", f"{status.not_run_count}")
    st.caption(
        "Для первого запуска расчёта траекторий выбраны все скважины. "
        "После запуска автоматически выбираются нерассчитанные и проблемные скважины."
    )


def _store_merged_batch_results(
    *,
    records: list[WelltrackRecord],
    new_rows: list[dict[str, object]],
    new_successes: list[SuccessfulWellPlan],
) -> None:
    ptc_batch_run.store_merged_batch_results(
        st.session_state,
        records=records,
        new_rows=new_rows,
        new_successes=new_successes,
        pending_edit_target_names=_pending_edit_target_names,
    )


def _batch_run_hooks() -> ptc_batch_run.BatchRunHooks:
    return ptc_batch_run.BatchRunHooks(
        selected_execution_order=_selected_execution_order,
        pending_edit_target_names=_pending_edit_target_names,
        ensure_pad_configs=_ensure_pad_configs,
        build_pad_plan_map=_build_pad_plan_map,
        build_selected_override_configs=_build_selected_override_configs,
        build_selected_optimization_contexts=_build_selected_optimization_contexts,
        reference_wells_from_state=_reference_wells_from_state,
        reference_uncertainty_models_from_state=_reference_uncertainty_models_from_state,
        resolution_snapshot_well_names=_resolution_snapshot_well_names,
        format_prepared_override_scope=_format_prepared_override_scope,
        prepared_plan_kind_label=_prepared_plan_kind_label,
        build_last_anticollision_resolution=_build_last_anticollision_resolution,
        focus_all_wells_anticollision_results=_focus_all_wells_anticollision_results,
        focus_all_wells_trajectory_results=_focus_all_wells_trajectory_results,
    )


def _run_batch_if_clicked(
    requests: list[_BatchRunRequest], records: list[WelltrackRecord]
) -> None:
    ptc_batch_run.run_batch_if_clicked(
        requests=requests,
        records=records,
        hooks=_batch_run_hooks(),
        st_module=st,
        calc_params_prefix=WT_CALC_PARAMS.prefix,
        log_compact_label=WT_LOG_COMPACT,
        log_verbose_label=WT_LOG_VERBOSE,
    )


def _build_batch_survey_csv(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_survey_csv(
        successes,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
        crs_display_suffix_func=get_crs_display_suffix,
        survey_export_dataframe_func=survey_export_dataframe,
        dls_to_pi_func=dls_to_pi,
    )


def _build_batch_survey_welltrack(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_survey_welltrack(
        successes,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
    )


def _build_batch_survey_dev_7z(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_survey_dev_7z(
        successes,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
    )


def _build_batch_survey_dev_file(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_survey_dev_file(
        successes,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
    )


def _render_batch_summary(
    summary_rows: list[dict[str, object]],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> pd.DataFrame:
    return ptc_batch_summary_panel.render_batch_summary(
        summary_rows,
        state=st.session_state,
        st_module=st,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        summary_dataframe_func=WelltrackBatchPlanner.summary_dataframe,
        arrow_safe_text_dataframe_func=arrow_safe_text_dataframe,
        batch_summary_display_df_func=_batch_summary_display_df,
        build_batch_survey_csv_func=_build_batch_survey_csv,
        build_batch_survey_welltrack_func=_build_batch_survey_welltrack,
        build_batch_survey_dev_7z_func=_build_batch_survey_dev_7z,
        build_batch_survey_dev_file_func=_build_batch_survey_dev_file,
        render_small_note_func=render_small_note,
    )


def _batch_summary_display_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    return ptc_batch_results.batch_summary_display_df(summary_df)


def _find_selected_success(
    *,
    selected_name: str,
    successes: list[SuccessfulWellPlan],
) -> SuccessfulWellPlan:
    return ptc_batch_results.find_selected_success(
        selected_name=selected_name,
        successes=successes,
    )
