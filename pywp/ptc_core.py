from __future__ import annotations

import colorsys
import hashlib
from html import escape
import json
import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

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
from streamlit.errors import StreamlitAPIException

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)


def _rerun_fragment() -> None:
    try:
        st.rerun(scope="fragment")
    except (TypeError, StreamlitAPIException):
        st.rerun()


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
    AntiCollisionIncrementalStats,
    AntiCollisionPairCacheEntry,
    AntiCollisionProgress,
    AntiCollisionWell,
    REFERENCE_ANTI_COLLISION_SCOPE_DISTANCE_M,
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
    build_incremental_anti_collision_analysis_for_successes as build_incremental_anti_collision_analysis_for_successes_shared,
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
from pywp.anticollision_rerun import (
    reference_wells_in_anti_collision_scope,
)
from pywp.constants import SMALL
from pywp.coordinate_integration import (
    DEFAULT_CRS,
    csv_export_crs,
    get_crs_display_suffix,
    transform_stations_to_crs,
    transform_xy_to_crs,
)
from pywp.coordinate_systems import CoordinateSystem
from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackPoint,
    WelltrackRecord,
    parse_welltrack_text,
    welltrack_points_to_target_pairs,
)
from pywp.models import Point3D
from pywp.models import J_PROFILE_POLICY_PREFER
from pywp.pilot_wells import (
    is_pilot_name,
    is_zbs_record,
    parent_name_for_zbs,
    parent_name_for_pilot,
    pilot_parent_key_for_record,
    pilot_name_key_for_record,
    sync_pilot_surfaces_to_parents,
    visible_well_records,
    well_name_key,
    zbs_target_points_to_pairs,
)
from pywp.welltrack_targets import (
    ordinary_record_target_layout,
    record_multi_horizontal_level_count,
    record_point_labels,
)
from pywp.planner_config import optimization_display_label
from pywp.plot_axes import (
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
    reference_well_collision_name,
    reference_well_display_label,
    reference_well_duplicate_name_keys,
)
from pywp.three_viewer import render_local_three_scene
from pywp.ui_calc_params import (
    CalcParamBinding,
    calc_param_defaults,
    calc_param_state_values_from_config,
    build_config_from_values,
    clear_kop_min_vertical_function,
    kop_min_vertical_function_from_state,
    kop_min_vertical_mode,
    set_calc_param_state_values,
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
    well_name_natural_sort_key,
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
DEFAULT_DEV_TRAJECTORY_PATH = ptc_target_import.DEFAULT_DEV_TRAJECTORY_PATH
WT_SOURCE_FORMAT_WELLTRACK = ptc_target_import.WT_SOURCE_FORMAT_WELLTRACK
WT_SOURCE_FORMAT_DEV_TRAJECTORY = ptc_target_import.WT_SOURCE_FORMAT_DEV_TRAJECTORY
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
WT_T1T3_ORDER_PANEL_ANCHOR_ID = "wt-t1-t3-order-panel"
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
WT_3D_REFERENCE_CONE_FOCUS_DISTANCE_M = REFERENCE_ANTI_COLLISION_SCOPE_DISTANCE_M
WT_THREE_MAX_HOVER_POINTS_PER_TRACE = (
    ptc_three_payload.WT_THREE_MAX_HOVER_POINTS_PER_TRACE
)
WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE = (
    ptc_three_payload.WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
)
WT_THREE_MAX_LABELS = ptc_three_payload.WT_THREE_MAX_LABELS
WT_THREE_MAX_REFERENCE_LABELS = ptc_three_payload.WT_THREE_MAX_REFERENCE_LABELS
WT_PAD_FOCUS_ALL = ptc_pad_state.WT_PAD_FOCUS_ALL
WT_RAW_RECORDS_AUTO_RENDER_POINT_LIMIT = 2000
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
WT_WELL_OVERRIDE_EDITOR = CalcParamBinding(prefix="wt_well_cfg_")
WT_WELL_CALC_OVERRIDE_ENABLED_KEY = "wt_well_calc_overrides_enabled"
WT_WELL_CALC_OVERRIDE_ENABLED_PENDING_KEY = (
    "wt_well_calc_overrides_enabled_pending"
)
WT_WELL_CALC_OVERRIDE_STATE_KEY = "wt_well_calc_overrides"
WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY = "wt_well_calc_profile_assignments"
WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY = "wt_well_calc_active_profile_id"
WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY = (
    "wt_well_calc_active_profile_id_pending"
)
WT_WELL_CALC_PROFILE_IMPORT_UPLOAD_KEY = "wt_well_calc_profile_import_upload"
WT_WELL_CALC_OVERRIDE_SELECTION_KEY = "wt_well_calc_override_selected_names"
WT_WELL_CALC_OVERRIDE_SELECTION_PENDING_KEY = (
    "wt_well_calc_override_pending_selected_names"
)
WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY = (
    "wt_well_calc_override_selected_signature"
)
WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY = "wt_well_calc_override_feedback"
WT_WELL_CALC_OVERRIDE_NAME_INPUT_ACTIVE_KEY = (
    "wt_well_calc_override_name_input_active_profile_id"
)
WT_LAST_WELL_CALC_OVERRIDE_SIGNATURE_KEY = (
    "wt_last_well_calc_override_signature"
)
WT_WELL_CALC_PROFILE_JSON_KIND = "pywp.manual_well_calc_profile"
WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION = 1
_CALC_PARAM_OVERRIDE_LABELS: dict[str, str] = {
    "md_step": "Шаг MD",
    "md_control": "Контрольный шаг",
    "lateral_tol": "Допуск XY",
    "vertical_tol": "Допуск Z",
    "entry_inc_target": "INC на t1",
    "entry_inc_tol": "Допуск INC",
    "max_inc": "Макс INC",
    "max_total_md_postcheck": "Постпроверка MD",
    "dls_build_max": "ПИ BUILD",
    "dls_build2_enabled": "BUILD2 отдельно",
    "dls_build2_max": "ПИ BUILD2",
    "dls_horizontal_max": "ПИ HORIZONTAL",
    "kop_min_vertical": "KOP",
    "min_hold_inc_enabled": "Мин HOLD отдельно",
    "min_hold_inc": "Мин HOLD, deg",
    "optimization_mode": "Оптимизация",
    "turn_solver_max_restarts": "Рестарты",
    "turn_solver_mode": "Метод",
    "interpolation_method": "Интерполяция",
    "j_profile_policy": "J-профиль",
    "offer_j_profile": "Предлагать J",
}
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
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = ()
    target_points: tuple[Point3D, ...] = ()
    target_labels: tuple[str, ...] = ()


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
            visible_well_records(records, include_zbs=False),
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
    parent_name_by_family_key: dict[str, str] = {}
    for record in records:
        if is_pilot_name(record.name):
            continue
        parent_name_by_family_key.setdefault(
            pilot_parent_key_for_record(record),
            str(record.name),
        )
    for record in records:
        name = str(record.name)
        if is_pilot_name(name):
            parent_name = parent_name_by_family_key.get(
                pilot_parent_key_for_record(name)
            )
            if parent_name is not None and parent_name in color_map:
                color_map[name] = color_map[parent_name]
            continue
        pilot_name = (
            None
            if is_zbs_record(record)
            else name_by_key.get(pilot_name_key_for_record(record))
        )
        if pilot_name is not None:
            color_map[pilot_name] = color_map[name]
    return color_map


def _failed_target_only_wells(
    *,
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]],
) -> list[_TargetOnlyWell]:
    rows_by_key = {
        well_name_key(row.get("Скважина", "")): row
        for row in summary_rows
        if str(row.get("Скважина", "")).strip()
    }
    pending_edit_names = set(_pending_edit_target_names())
    target_only_wells: list[_TargetOnlyWell] = []
    for record in records:
        row = rows_by_key.get(well_name_key(record.name))
        if row is None:
            continue
        status = str(row.get("Статус", "")).strip()
        if status == "OK" or (
            status == "Не рассчитана" and str(record.name) not in pending_edit_names
        ):
            continue
        target_only = _target_only_well_from_record(
            record=record,
            status=status or "Ошибка расчета",
            problem=str(row.get("Проблема", "")).strip(),
        )
        if target_only is not None:
            target_only_wells.append(target_only)
    return target_only_wells


def _target_only_well_from_record(
    *,
    record: WelltrackRecord,
    status: str,
    problem: str,
) -> _TargetOnlyWell | None:
    target_points = tuple(
        Point3D(x=float(point.x), y=float(point.y), z=float(point.z))
        for point in tuple(record.points)
    )
    if not target_points:
        return None
    if is_zbs_record(record):
        surface = target_points[0]
        try:
            target_pairs = zbs_target_points_to_pairs(tuple(record.points))
        except ValueError:
            target_pairs = ()
        if target_pairs:
            t1, t3 = target_pairs[0][0], target_pairs[-1][1]
        elif len(target_points) >= 2:
            t1, t3 = target_points[0], target_points[-1]
        else:
            t1 = t3 = target_points[0]
    else:
        try:
            layout = ordinary_record_target_layout(record)
        except ValueError:
            surface = target_points[0]
            target_pairs = ()
        else:
            surface = layout.surface
            target_pairs = layout.target_pairs
            t1 = layout.t1
            t3 = layout.final_target
        if target_pairs:
            t1, t3 = target_pairs[0][0], target_pairs[-1][1]
        elif len(target_points) >= 2:
            start_index = 1 if len(target_points) >= 3 else 0
            t1, t3 = target_points[start_index], target_points[-1]
        else:
            t1 = t3 = target_points[0]
    return _TargetOnlyWell(
        name=str(record.name),
        surface=surface,
        t1=t1,
        t3=t3,
        target_pairs=target_pairs,
        target_points=target_points,
        target_labels=record_point_labels(record),
        status=str(status).strip(),
        problem=str(problem).strip(),
    )


def _overview_target_only_wells(
    *,
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]],
    successes: list[SuccessfulWellPlan] | list[object],
) -> list[_TargetOnlyWell]:
    rows_by_key = {
        well_name_key(row.get("Скважина", "")): row
        for row in summary_rows
        if str(row.get("Скважина", "")).strip()
    }
    success_keys = {
        well_name_key(getattr(success, "name", ""))
        for success in successes
        if str(getattr(success, "name", "")).strip()
    }
    target_only_wells = _failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    existing_keys = {
        well_name_key(getattr(target_only, "name", ""))
        for target_only in target_only_wells
        if str(getattr(target_only, "name", "")).strip()
    }
    for record in records:
        record_key = well_name_key(record.name)
        if record_key in success_keys or record_key in existing_keys:
            continue
        row = rows_by_key.get(record_key)
        status = str((row or {}).get("Статус", "")).strip()
        if status == "OK":
            continue
        target_only = _target_only_well_from_record(
            record=record,
            status=status or "Не рассчитана",
            problem=str((row or {}).get("Проблема", "")).strip(),
        )
        if target_only is not None:
            target_only_wells.append(target_only)
            existing_keys.add(record_key)
    return target_only_wells


def _record_target_point_labels(record: WelltrackRecord) -> tuple[str, ...]:
    return record_point_labels(record)


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


def _display_only_reference_wells_for_analysis(
    *,
    reference_wells: Iterable[ImportedTrajectoryWell],
    analysis_reference_wells: Iterable[ImportedTrajectoryWell],
) -> tuple[ImportedTrajectoryWell, ...]:
    analysis_identities = {
        (str(well.name).strip(), str(well.kind).strip())
        for well in analysis_reference_wells
    }
    result: list[ImportedTrajectoryWell] = []
    for reference_well in reference_wells:
        kind = str(reference_well.kind).strip()
        raw_identity = (str(reference_well.name).strip(), kind)
        display_identity = (reference_well_display_label(reference_well).strip(), kind)
        if raw_identity in analysis_identities or display_identity in analysis_identities:
            continue
        result.append(reference_well)
    return tuple(result)


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
    edit_pads: list[dict[str, object]] | None = None,
    edit_wells: list[dict[str, object]] | None = None,
    extra_bounds: dict[str, list[float]] | None = None,
    extra_lines: list[dict[str, object]] | None = None,
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
        edit_pads=edit_pads,
        edit_wells=edit_wells,
        extra_bounds=extra_bounds,
        extra_lines=extra_lines,
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
    display_name_by_well_name: Mapping[str, str] | None = None,
    pilot_study_points_by_name: Mapping[str, tuple[Point3D, ...]] | None = None,
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
        display_name_by_well_name=display_name_by_well_name,
        pilot_study_points_by_name=pilot_study_points_by_name,
        focus_well_names=focus_well_names,
        render_mode=resolved_render_mode,
        fallback_color=_well_color,
    )


def _all_wells_anticollision_three_payload(
    analysis: AntiCollisionAnalysis,
    *,
    previous_successes_by_name: Mapping[str, SuccessfulWellPlan] | None = None,
    target_only_wells: list[_TargetOnlyWell] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    name_to_color: Mapping[str, str] | None = None,
    display_name_by_well_name: Mapping[str, str] | None = None,
    pilot_study_points_by_name: Mapping[str, tuple[Point3D, ...]] | None = None,
    focus_well_names: tuple[str, ...] = (),
    render_mode: str = WT_3D_RENDER_FAST,
    show_sidetrack_relative_cones: bool = False,
) -> dict[str, object]:
    analysis_reference_wells = tuple(
        well for well in analysis.wells if bool(well.is_reference_only)
    )
    loaded_reference_wells = tuple(reference_wells)
    resolved_render_mode = _resolve_3d_render_mode(
        requested_mode=render_mode,
        calculated_well_count=len(
            [well for well in analysis.wells if not bool(well.is_reference_only)]
        ),
        reference_wells=(
            loaded_reference_wells
            if loaded_reference_wells
            else analysis_reference_wells
        ),
    )
    return ptc_three_builders.anticollision_three_payload(
        analysis,
        previous_successes_by_name=previous_successes_by_name,
        target_only_wells=target_only_wells,
        reference_wells=loaded_reference_wells,
        name_to_color=name_to_color,
        display_name_by_well_name=display_name_by_well_name,
        pilot_study_points_by_name=pilot_study_points_by_name,
        focus_well_names=focus_well_names,
        render_mode=resolved_render_mode,
        show_sidetrack_relative_cones=bool(show_sidetrack_relative_cones),
    )


_THREE_AUGMENTED_PAYLOAD_CACHE: list[dict[str, object]] = []


def _cached_augmented_three_payload(
    *,
    payload: dict[str, object],
    payload_overrides: dict[str, object] | None,
) -> dict[str, object]:
    if not payload_overrides:
        return payload
    for entry in reversed(_THREE_AUGMENTED_PAYLOAD_CACHE):
        if (
            entry.get("payload") is payload
            and entry.get("payload_overrides") is payload_overrides
            and isinstance(entry.get("augmented_payload"), dict)
        ):
            return entry["augmented_payload"]  # type: ignore[return-value]
    augmented_payload = _augment_three_payload(
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
        edit_pads=payload_overrides.get("edit_pads"),
        edit_wells=payload_overrides.get("edit_wells"),
        extra_bounds=payload_overrides.get("extra_bounds"),
        extra_lines=payload_overrides.get("extra_lines"),
        extra_labels=payload_overrides.get("extra_labels"),
        extra_meshes=payload_overrides.get("extra_meshes"),
        extra_legend_items=payload_overrides.get("extra_legend_items"),
    )
    _THREE_AUGMENTED_PAYLOAD_CACHE.append(
        {
            "payload": payload,
            "payload_overrides": payload_overrides,
            "augmented_payload": augmented_payload,
        }
    )
    if len(_THREE_AUGMENTED_PAYLOAD_CACHE) > 4:
        del _THREE_AUGMENTED_PAYLOAD_CACHE[:-4]
    return augmented_payload


def _render_three_payload(
    *,
    container: object,
    payload: dict[str, object],
    height: int,
    payload_overrides: dict[str, object] | None = None,
) -> None:
    payload = _cached_augmented_three_payload(
        payload=payload,
        payload_overrides=payload_overrides,
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
    *,
    parent_successes: Iterable[SuccessfulWellPlan] | None = None,
) -> list[dict[str, object]]:
    return ptc_three_overrides.build_edit_wells_payload(
        successes,
        name_to_color,
        reference_wells=_reference_wells_from_state(),
        parent_successes=parent_successes,
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
        reference_wells=_reference_wells_from_state(),
        imported_dev_target_wells=_imported_dev_target_wells_from_state(),
    )


def _anticollision_three_payload_overrides(
    *,
    records: list[WelltrackRecord],
    analysis: AntiCollisionAnalysis,
    successes: list[SuccessfulWellPlan] | None = None,
    target_only_wells: list[_TargetOnlyWell] | None = None,
    target_only_name_to_color: Mapping[str, str] | None = None,
) -> dict[str, object]:
    return ptc_three_overrides.anticollision_three_payload_overrides(
        st.session_state,
        records=records,
        analysis=analysis,
        successes=successes,
        target_only_wells=target_only_wells,
        target_only_name_to_color=target_only_name_to_color,
        reference_wells=_reference_wells_from_state(),
        imported_dev_target_wells=_imported_dev_target_wells_from_state(),
    )


def _imported_dev_target_three_payload_overrides(
    *,
    visible_well_names: Iterable[str],
    name_to_color: Mapping[str, str],
) -> dict[str, object]:
    return ptc_three_overrides.imported_dev_target_three_payload_overrides(
        visible_well_names=visible_well_names,
        name_to_color=name_to_color,
        imported_dev_target_wells=_imported_dev_target_wells_from_state(),
    )


def _build_anti_collision_analysis(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    progress_callback: Callable[[AntiCollisionProgress], None] | None = None,
    parallel_workers: int = 0,
) -> AntiCollisionAnalysis:
    color_map = _planned_anti_collision_color_map(successes, name_to_color)
    return build_anti_collision_analysis_for_successes_shared(
        successes,
        model=model,
        name_to_color=color_map,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        progress_callback=progress_callback,
        parallel_workers=int(parallel_workers),
    )


def _build_incremental_anti_collision_analysis(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    well_signature_by_name: Mapping[str, str] | None = None,
    previous_well_cache: Mapping[str, tuple[str, AntiCollisionWell]] | None = None,
    previous_pair_cache: (
        Mapping[tuple[str, str], AntiCollisionPairCacheEntry] | None
    ) = None,
    progress_callback: Callable[[AntiCollisionProgress], None] | None = None,
    parallel_workers: int = 0,
) -> tuple[
    AntiCollisionAnalysis,
    dict[str, tuple[str, AntiCollisionWell]],
    dict[tuple[str, str], AntiCollisionPairCacheEntry],
    AntiCollisionIncrementalStats,
]:
    color_map = _planned_anti_collision_color_map(successes, name_to_color)
    return build_incremental_anti_collision_analysis_for_successes_shared(
        successes,
        model=model,
        name_to_color=color_map,
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        well_signature_by_name=well_signature_by_name,
        previous_well_cache=previous_well_cache,
        previous_pair_cache=previous_pair_cache,
        progress_callback=progress_callback,
        parallel_workers=int(parallel_workers),
    )


def _planned_anti_collision_color_map(
    successes: list[SuccessfulWellPlan],
    name_to_color: Mapping[str, str] | None,
) -> dict[str, str]:
    return {
        str(item.name): (name_to_color or {}).get(str(item.name), _well_color(index))
        for index, item in enumerate(successes)
    }


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
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    well_signatures: Mapping[str, str] | None = None,
) -> str:
    resolved_well_signatures = (
        {
            str(well_name): str(signature)
            for well_name, signature in well_signatures.items()
        }
        if well_signatures is not None
        else _anti_collision_well_signatures(
            successes=successes,
            model=model,
            name_to_color=_planned_anti_collision_color_map(successes, name_to_color),
            reference_wells=reference_wells,
            reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        )
    )
    digest = hashlib.blake2b(digest_size=20)
    for well_name, signature in sorted(resolved_well_signatures.items()):
        digest.update(str(well_name).encode("utf-8"))
        digest.update(str(signature).encode("utf-8"))
    return digest.hexdigest()


_ANTI_COLLISION_SIGNATURE_STATION_COLUMNS = (
    "MD_m",
    "INC_deg",
    "AZI_deg",
    "X_m",
    "Y_m",
    "Z_m",
)
_ANTI_COLLISION_SIGNATURE_SUMMARY_FIELDS = (
    "kop_md_m",
    "build1_dls_selected_deg_per_30m",
    "build_dls_selected_deg_per_30m",
    "md_total_m",
    "optimization_mode",
    "anti_collision_stage",
    "anti_collision_attempted_stages",
    "trajectory_type",
    "sidetrack_parent_well_name",
    "actual_parent_well_name",
    "sidetrack_parent_kind",
    "sidetrack_window_md_m",
    "sidetrack_window_x_m",
    "sidetrack_window_y_m",
    "sidetrack_window_z_m",
    "sidetrack_window_inc_deg",
    "sidetrack_window_azi_deg",
    "pilot_well_name",
)
_ANTI_COLLISION_SIGNATURE_CONFIG_FIELDS = (
    "kop_min_vertical_m",
    "dls_build_max_deg_per_30m",
    "dls_build2_max_deg_per_30m",
    "dls_horizontal_max_deg_per_30m",
    "optimization_mode",
)


def _anti_collision_well_signatures(
    *,
    successes: list[SuccessfulWellPlan],
    model: PlanningUncertaintyModel,
    name_to_color: Mapping[str, str] | None,
    reference_wells: tuple[ImportedTrajectoryWell, ...],
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
) -> dict[str, str]:
    signatures: dict[str, str] = {}
    planned_names = tuple(str(item.name) for item in successes)
    duplicate_reference_name_keys = reference_well_duplicate_name_keys(reference_wells)
    for index, success in enumerate(successes):
        name = str(success.name)
        digest = hashlib.blake2b(digest_size=20)
        digest.update(b"planned")
        digest.update(name.encode("utf-8"))
        digest.update(
            str((name_to_color or {}).get(name, _well_color(index))).encode("utf-8")
        )
        _update_uncertainty_model_cache_digest(digest, model)
        _update_point_cache_digest(digest, success.surface)
        _update_point_cache_digest(digest, success.t1)
        _update_point_cache_digest(digest, success.t3)
        _update_target_pairs_cache_digest(
            digest,
            tuple(getattr(success, "target_pairs", ()) or ()),
        )
        digest.update(
            np.asarray(
                [float(success.azimuth_deg), float(success.md_t1_m)],
                dtype=np.float64,
            ).tobytes()
        )
        _update_station_table_cache_digest(digest, success.stations)
        _update_success_context_cache_digest(digest, success)
        signatures[name] = digest.hexdigest()
    for reference_well in reference_wells:
        collision_name = reference_well_collision_name(
            reference_well,
            planned_names=planned_names,
            duplicate_name_keys=duplicate_reference_name_keys,
        )
        reference_model = (
            (reference_uncertainty_models_by_name or {}).get(str(collision_name))
            or (reference_uncertainty_models_by_name or {}).get(
                str(reference_well.name)
            )
            or model
        )
        digest = hashlib.blake2b(digest_size=20)
        digest.update(b"reference")
        digest.update(str(collision_name).encode("utf-8"))
        digest.update(str(reference_well.name).encode("utf-8"))
        digest.update(str(reference_well.kind).encode("utf-8"))
        digest.update(
            str(
                (name_to_color or {}).get(
                    str(collision_name),
                    REFERENCE_WELL_KIND_COLORS.get(
                        str(reference_well.kind),
                        "#A0A0A0",
                    ),
                )
            ).encode("utf-8")
        )
        _update_uncertainty_model_cache_digest(digest, reference_model)
        _update_point_cache_digest(digest, reference_well.surface)
        digest.update(
            np.asarray([float(reference_well.azimuth_deg)], dtype=np.float64).tobytes()
        )
        _update_station_table_cache_digest(digest, reference_well.stations)
        signatures[str(collision_name)] = digest.hexdigest()
    return signatures


def _update_point_cache_digest(digest: Any, point: Point3D | None) -> None:
    if point is None:
        digest.update(b"none")
        return
    digest.update(
        np.asarray(
            [float(point.x), float(point.y), float(point.z)],
            dtype=np.float64,
        ).tobytes()
    )


def _update_target_pairs_cache_digest(
    digest: Any,
    target_pairs: tuple[tuple[Point3D, Point3D], ...],
) -> None:
    digest.update(str(len(target_pairs)).encode("utf-8"))
    for left, right in target_pairs:
        _update_point_cache_digest(digest, left)
        _update_point_cache_digest(digest, right)


def _update_station_table_cache_digest(
    digest: Any,
    stations: pd.DataFrame,
) -> None:
    stations_subset = stations.loc[
        :,
        [
            column
            for column in _ANTI_COLLISION_SIGNATURE_STATION_COLUMNS
            if column in stations.columns
        ],
    ]
    digest.update(str(tuple(stations_subset.columns)).encode("utf-8"))
    digest.update(str(len(stations_subset)).encode("utf-8"))
    digest.update(stations_subset.to_numpy(dtype=np.float64, copy=False).tobytes())


def _update_success_context_cache_digest(
    digest: Any,
    success: SuccessfulWellPlan,
) -> None:
    summary = dict(success.summary)
    context_values: list[tuple[str, str]] = []
    for field_name in _ANTI_COLLISION_SIGNATURE_SUMMARY_FIELDS:
        context_values.append((field_name, repr(summary.get(field_name))))
    config = success.config
    for field_name in _ANTI_COLLISION_SIGNATURE_CONFIG_FIELDS:
        context_values.append((field_name, repr(getattr(config, field_name, None))))
    digest.update(repr(tuple(context_values)).encode("utf-8"))


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
        f"Ошибка anti-collision: {exc}"
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
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    progress_callback: Callable[[int, str], None] | None = None,
    parallel_workers: int = 0,
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

    last_progress_update_s_by_stage: dict[str, float] = {}
    last_progress_percent_by_stage: dict[str, int] = {}

    def _remaining_progress_text(
        *,
        elapsed_s: float,
        completed: int,
        total: int,
        parallel_workers: int = 1,
    ) -> str:
        if completed <= 0:
            return "оценка времени..."
        if completed >= total:
            return "завершение"
        workers = max(int(parallel_workers), 1)
        remaining = max(int(total) - int(completed), 0)
        if workers > 1 and completed < workers:
            eta_s = (float(elapsed_s) / max(float(completed), 1.0)) * (
                float(remaining) / float(workers)
            )
        else:
            eta_s = (
                float(elapsed_s)
                / max(float(completed), 1.0)
                * float(remaining)
            )
        return "осталось оц. " + _format_duration_ru(eta_s)

    def _stage_progress_should_emit(
        *,
        stage: str,
        percent: int,
        completed: int,
        total: int,
    ) -> bool:
        now = perf_counter()
        last_percent = int(last_progress_percent_by_stage.get(stage, -1))
        last_update_s = float(last_progress_update_s_by_stage.get(stage, 0.0))
        if (
            completed < total
            and percent == last_percent
            and now - last_update_s < 0.75
        ):
            return False
        last_progress_percent_by_stage[stage] = int(percent)
        last_progress_update_s_by_stage[stage] = float(now)
        return True

    def _emit_anti_collision_progress(progress: AntiCollisionProgress) -> None:
        stage = str(getattr(progress, "stage", "pairs") or "pairs")
        if stage == "wells":
            total = int(getattr(progress, "well_count", 0) or 0)
            completed = int(getattr(progress, "completed_well_count", 0) or 0)
            if total <= 0:
                return
            fraction = min(max(float(completed) / max(float(total), 1.0), 0.0), 1.0)
            percent = int(round(12.0 + 18.0 * fraction))
            if not _stage_progress_should_emit(
                stage=stage,
                percent=percent,
                completed=completed,
                total=total,
            ):
                return
            rebuild_total = int(getattr(progress, "rebuild_well_count", 0) or 0)
            rebuild_completed = int(
                getattr(
                    progress,
                    "completed_rebuild_well_count",
                    getattr(progress, "rebuilt_well_count", 0),
                )
                or 0
            )
            eta_completed = completed
            eta_total = total
            if rebuild_total > 0:
                eta_completed = rebuild_completed
                eta_total = rebuild_total
            eta_text = _remaining_progress_text(
                elapsed_s=float(max(progress.elapsed_s, 0.0)),
                completed=eta_completed,
                total=eta_total,
                parallel_workers=int(getattr(progress, "parallel_workers", 0) or 0),
            )
            workers = int(getattr(progress, "parallel_workers", 0) or 0)
            worker_text = f" · {workers} процессов" if workers > 1 else ""
            built_text = (
                f"{rebuild_completed}/{rebuild_total}"
                if rebuild_total > 0
                else str(int(progress.rebuilt_well_count))
            )
            progress_text = (
                "Anti-collision: конусы неопределённости "
                f"{completed}/{total} · {eta_text}{worker_text} · "
                f"кэш скважин {int(progress.reused_well_count)} · "
                f"построено {built_text}"
            )
            if progress_callback is not None:
                progress_callback(percent, progress_text)
            return

        total = int(progress.pair_count)
        completed = int(progress.completed_pair_count)
        if total <= 0:
            return
        fraction = min(max(float(completed) / max(float(total), 1.0), 0.0), 1.0)
        percent = int(round(30.0 + 36.0 * fraction))
        if not _stage_progress_should_emit(
            stage=stage,
            percent=percent,
            completed=completed,
            total=total,
        ):
            return
        elapsed_s = float(max(progress.elapsed_s, 0.0))
        workers = int(progress.parallel_workers)
        eta_text = _remaining_progress_text(
            elapsed_s=elapsed_s,
            completed=completed,
            total=total,
            parallel_workers=workers,
        )
        worker_text = f" · {workers} процессов" if workers > 1 else ""
        progress_text = (
            f"Anti-collision: пары {completed}/{total} · {eta_text}"
            f"{worker_text} · кэш {int(progress.reused_pair_count)} · "
            f"prefilter {int(progress.prefiltered_pair_count)} · "
            f"пересчёт {int(progress.recalculated_pair_count)}"
        )
        if progress_callback is not None:
            progress_callback(percent, progress_text)

    color_map = _well_color_map(records) if records else {}
    _emit(8, "Подготовка данных anti-collision.")
    all_reference_wells = tuple(reference_wells)
    scoped_reference_wells = reference_wells_in_anti_collision_scope(
        successes,
        all_reference_wells,
    )
    skipped_reference_count = max(
        len(all_reference_wells) - len(scoped_reference_wells),
        0,
    )
    if skipped_reference_count:
        _emit(
            10,
            (
                "Состав набора: "
                f"{len(scoped_reference_wells)}/{len(all_reference_wells)} "
                "фактических/утверждённых скважин рядом с расчётными; "
                f"{skipped_reference_count} дальних скважин пропущено."
            ),
        )
    planned_color_map = _planned_anti_collision_color_map(successes, color_map)
    well_signature_by_name = _anti_collision_well_signatures(
        successes=successes,
        model=uncertainty_model,
        name_to_color=planned_color_map,
        reference_wells=scoped_reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
    )
    cache_key = _anti_collision_cache_key(
        successes=successes,
        model=uncertainty_model,
        name_to_color=planned_color_map,
        reference_wells=scoped_reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        well_signatures=well_signature_by_name,
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
                "reused_well_count": int(cache.get("last_reused_well_count") or 0),
                "rebuilt_well_count": int(cache.get("last_rebuilt_well_count") or 0),
                "reused_pair_count": int(cache.get("last_reused_pair_count") or 0),
                "recalculated_pair_count": int(
                    cache.get("last_recalculated_pair_count") or 0
                ),
                "parallel_workers": int(cache.get("last_parallel_workers") or 0),
                "scoped_reference_count": int(len(scoped_reference_wells)),
                "skipped_reference_count": int(skipped_reference_count),
            }
            return analysis, recommendations, clusters
    try:
        _emit(12, "Подготовка конусов неопределённости.")
        previous_well_cache = cache.get("well_cache")
        previous_pair_cache = cache.get("pair_cache")
        if not isinstance(previous_well_cache, Mapping):
            previous_well_cache = None
        if not isinstance(previous_pair_cache, Mapping):
            previous_pair_cache = None
        analysis, well_cache, pair_cache, incremental_stats = (
            _build_incremental_anti_collision_analysis(
                successes,
                model=uncertainty_model,
                name_to_color=planned_color_map,
                reference_wells=scoped_reference_wells,
                reference_uncertainty_models_by_name=(
                    reference_uncertainty_models_by_name
                ),
                well_signature_by_name=well_signature_by_name,
                previous_well_cache=previous_well_cache,
                previous_pair_cache=previous_pair_cache,
                progress_callback=_emit_anti_collision_progress,
                parallel_workers=int(parallel_workers),
            )
        )
        if incremental_stats.reused_well_count or incremental_stats.reused_pair_count:
            _emit(
                66,
                (
                    "Инкрементальный anti-collision: "
                    f"скважины {incremental_stats.reused_well_count}/"
                    f"{incremental_stats.reused_well_count + incremental_stats.rebuilt_well_count} "
                    "из кэша, пары "
                    f"{incremental_stats.reused_pair_count}/"
                    f"{incremental_stats.reused_pair_count + incremental_stats.recalculated_pair_count} "
                    "из кэша."
                ),
            )
        else:
            _emit(
                66,
                (
                    "Anti-collision рассчитано без переиспользованных пар: "
                    f"скважин {incremental_stats.rebuilt_well_count}, "
                    f"пар {incremental_stats.recalculated_pair_count}."
                    + (
                        f" Параллельных процессов: {int(parallel_workers)}."
                        if int(parallel_workers) > 1
                        else ""
                    )
                ),
            )
        _emit(72, "Построение рекомендаций anti-collision.")
        recommendations = build_anti_collision_recommendations(
            analysis,
            well_context_by_name=_build_anticollision_well_contexts(successes),
        )
        _emit(88, "Кластеризация рекомендаций.")
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
        "well_signature_by_name": dict(well_signature_by_name),
        "well_cache": well_cache,
        "pair_cache": pair_cache,
        "last_reused_well_count": int(incremental_stats.reused_well_count),
        "last_rebuilt_well_count": int(incremental_stats.rebuilt_well_count),
        "last_reused_pair_count": int(incremental_stats.reused_pair_count),
        "last_recalculated_pair_count": int(incremental_stats.recalculated_pair_count),
        "last_parallel_workers": int(parallel_workers),
        "last_scoped_reference_count": int(len(scoped_reference_wells)),
        "last_skipped_reference_count": int(skipped_reference_count),
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
        "reused_well_count": int(incremental_stats.reused_well_count),
        "rebuilt_well_count": int(incremental_stats.rebuilt_well_count),
        "reused_pair_count": int(incremental_stats.reused_pair_count),
        "recalculated_pair_count": int(incremental_stats.recalculated_pair_count),
        "parallel_workers": int(parallel_workers),
        "scoped_reference_count": int(len(scoped_reference_wells)),
        "skipped_reference_count": int(skipped_reference_count),
    }
    return analysis, recommendations, clusters


def _current_anti_collision_cache_snapshot(
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    records: list[WelltrackRecord],
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
) -> tuple[
    AntiCollisionAnalysis,
    tuple[AntiCollisionRecommendation, ...],
    tuple[AntiCollisionRecommendationCluster, ...],
] | None:
    cache = st.session_state.get("wt_anticollision_analysis_cache")
    if not isinstance(cache, Mapping):
        return None
    cached_key = str(cache.get("key", "")).strip()
    if not cached_key:
        return None
    analysis = cache.get("analysis")
    recommendations = cache.get("recommendations")
    clusters = cache.get("clusters")
    if not isinstance(analysis, AntiCollisionAnalysis):
        return None
    if not isinstance(recommendations, tuple) or not isinstance(clusters, tuple):
        return None

    color_map = _well_color_map(records) if records else {}
    scoped_reference_wells = reference_wells_in_anti_collision_scope(
        successes,
        tuple(reference_wells),
    )
    planned_color_map = _planned_anti_collision_color_map(successes, color_map)
    well_signature_by_name = _anti_collision_well_signatures(
        successes=successes,
        model=uncertainty_model,
        name_to_color=planned_color_map,
        reference_wells=scoped_reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
    )
    cache_key = _anti_collision_cache_key(
        successes=successes,
        model=uncertainty_model,
        name_to_color=planned_color_map,
        reference_wells=scoped_reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        well_signatures=well_signature_by_name,
    )
    if cached_key != cache_key:
        return None
    return analysis, recommendations, clusters


def _format_duration_ru(seconds: float) -> str:
    total_seconds = int(round(float(max(seconds, 0.0))))
    if total_seconds < 60:
        return f"{total_seconds} с"
    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {secs:02d} с"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours} ч {remaining_minutes:02d} мин"


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
                "Пересечения",
                f"{int(state_payload.get('overlap_count') or 0)}",
            )
            extra_cols[1].metric(
                "Кластеров",
                f"{int(state_payload.get('cluster_count') or 0)}",
            )
        if "reused_pair_count" in state_payload:
            cache_cols = st.columns(2, gap="small")
            reused_wells = int(state_payload.get("reused_well_count") or 0)
            rebuilt_wells = int(state_payload.get("rebuilt_well_count") or 0)
            reused_pairs = int(state_payload.get("reused_pair_count") or 0)
            recalculated_pairs = int(state_payload.get("recalculated_pair_count") or 0)
            cache_cols[0].metric(
                "Скважины из кэша",
                f"{reused_wells}/{reused_wells + rebuilt_wells}",
            )
            cache_cols[1].metric(
                "Пары из кэша",
                f"{reused_pairs}/{reused_pairs + recalculated_pairs}",
            )
        if int(state_payload.get("parallel_workers") or 0) > 1:
            st.caption(
                "Параллельный расчёт anti-collision: "
                f"{int(state_payload.get('parallel_workers') or 0)} процессов."
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


def _imported_dev_target_wells_from_state() -> tuple[ImportedTrajectoryWell, ...]:
    raw_items = tuple(
        st.session_state.get(ptc_target_import.IMPORTED_DEV_TARGET_WELLS_STATE_KEY, ())
        or ()
    )
    return tuple(
        item
        for item in raw_items
        if isinstance(item, ImportedTrajectoryWell)
    )


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
    st.session_state.setdefault("wt_target_import_source_kind", "")
    st.session_state.setdefault("wt_imported_dev_params", ())
    st.session_state.setdefault(ptc_target_import.IMPORTED_DEV_TARGET_WELLS_STATE_KEY, ())
    st.session_state.setdefault(ptc_target_import.TARGET_IMPORT_FAILURES_STATE_KEY, ())
    st.session_state.setdefault("wt_pad_configs", {})
    st.session_state.setdefault("wt_pad_detected_meta", {})
    st.session_state.setdefault("wt_pad_selected_id", "")
    st.session_state.setdefault("wt_pad_last_applied_at", "")
    st.session_state.setdefault("wt_pad_auto_applied_on_import", False)
    st.session_state.setdefault("wt_pad_auto_order_by_target_depth", False)

    st.session_state.setdefault("wt_summary_rows", None)
    st.session_state.setdefault("wt_successes", None)
    st.session_state.setdefault("wt_last_error", "")
    st.session_state.setdefault("wt_last_run_at", "")
    st.session_state.setdefault("wt_last_runtime_s", None)
    st.session_state.setdefault("wt_last_calc_param_signature", None)
    st.session_state.setdefault(WT_LAST_WELL_CALC_OVERRIDE_SIGNATURE_KEY, None)
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
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_ENABLED_KEY, False)
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_ENABLED_PENDING_KEY, None)
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_STATE_KEY, {})
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY, {})
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY, "")
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY, None)
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_SELECTION_KEY, [])
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_SELECTION_PENDING_KEY, None)
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY, ())
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY, "")
    st.session_state.setdefault(WT_WELL_CALC_OVERRIDE_NAME_INPUT_ACTIVE_KEY, "")
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


def _surface_xyz(record: WelltrackRecord) -> tuple[float, float, float] | None:
    points = tuple(getattr(record, "points", ()) or ())
    if not points:
        return None
    surface = points[0]
    return (float(surface.x), float(surface.y), float(surface.z))


def _changed_surface_well_names(
    before_records: Iterable[WelltrackRecord],
    after_records: Iterable[WelltrackRecord],
) -> list[str]:
    before_by_name = {
        str(record.name).strip(): xyz
        for record in before_records
        if str(record.name).strip()
        for xyz in [_surface_xyz(record)]
        if xyz is not None
    }
    changed_names: list[str] = []
    for record in after_records:
        well_name = str(record.name).strip()
        if not well_name:
            continue
        before_xyz = before_by_name.get(well_name)
        after_xyz = _surface_xyz(record)
        if before_xyz is None or after_xyz is None:
            continue
        if any(
            abs(float(before_value) - float(after_value)) > 1e-9
            for before_value, after_value in zip(before_xyz, after_xyz, strict=True)
        ):
            changed_names.append(well_name)
    return _unique_well_names(changed_names)


def _coerce_highlight_point_indices(raw_indices: object) -> list[int]:
    if (
        not isinstance(raw_indices, Iterable)
        or isinstance(raw_indices, (str, bytes, Mapping))
    ):
        return []
    return sorted(
        {
            int(raw_index)
            for raw_index in raw_indices
            if (
                isinstance(raw_index, int)
                or str(raw_index).lstrip("-").isdigit()
            )
            and int(raw_index) >= 0
        }
    )


def _queue_surface_edit_feedback(
    changed_well_names: Iterable[object],
    *,
    source: str,
) -> list[str]:
    changed_names = _unique_well_names(changed_well_names)
    if not changed_names:
        return []

    existing_highlighted_names = _unique_well_names(
        [
            *(st.session_state.get("wt_edit_targets_pending_names") or []),
            *(st.session_state.get("wt_edit_targets_highlight_names") or []),
        ]
    )
    existing_highlight_points_raw = st.session_state.get(
        "wt_edit_targets_highlight_points"
    )
    existing_highlight_points: dict[str, list[int]] = {}
    if isinstance(existing_highlight_points_raw, Mapping):
        for raw_name, raw_indices in existing_highlight_points_raw.items():
            well_name = str(raw_name).strip()
            if not well_name:
                continue
            parsed_indices = _coerce_highlight_point_indices(raw_indices)
            if parsed_indices:
                existing_highlight_points[well_name] = parsed_indices

    _clear_results()
    pending_names = _unique_well_names(
        [*existing_highlighted_names, *changed_names]
    )
    for well_name in changed_names:
        row_indices = set(existing_highlight_points.get(well_name, []))
        row_indices.add(0)
        existing_highlight_points[well_name] = sorted(row_indices)

    highlight_points = {
        well_name: list(existing_highlight_points[well_name])
        for well_name in pending_names
        if existing_highlight_points.get(well_name)
    }
    st.session_state["wt_edit_targets_pending_names"] = list(pending_names)
    st.session_state["wt_edit_targets_highlight_names"] = [
        well_name for well_name in pending_names if well_name in highlight_points
    ]
    st.session_state["wt_edit_targets_highlight_points"] = highlight_points
    st.session_state["wt_edit_targets_applied"] = list(changed_names)
    st.session_state["wt_edit_targets_applied_source"] = source
    st.session_state["wt_edit_targets_last_source"] = source
    st.session_state.pop("wt_edit_targets_applied_note", None)
    st.session_state["wt_pending_selected_names"] = list(pending_names)
    return changed_names


def _bulk_horizontal_length_changes(
    records: Iterable[WelltrackRecord],
    *,
    target_length_m: float,
) -> tuple[list[dict[str, object]], list[str]]:
    return ptc_edit_targets.bulk_horizontal_length_changes(
        records=records,
        target_length_m=target_length_m,
    )


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


def _apply_edit_pad_changes(
    changes: object,
    *,
    source: str = "3d",
) -> list[str]:
    if not isinstance(changes, list) or not changes:
        return []
    base_records = list(
        st.session_state.get("wt_records_original")
        or st.session_state.get("wt_records")
        or []
    )
    working_records = list(st.session_state.get("wt_records") or base_records)
    if not base_records:
        return []
    pads = _ensure_pad_configs(base_records=base_records)
    if not pads:
        return []

    pad_by_id = {str(pad.pad_id): pad for pad in pads}
    raw_configs = st.session_state.setdefault("wt_pad_configs", {})
    updated_any = False

    for raw_change in changes:
        if not isinstance(raw_change, Mapping):
            continue
        pad_id = str(raw_change.get("pad_id") or "").strip()
        pad = pad_by_id.get(pad_id)
        anchor = raw_change.get("anchor")
        if pad is None or not isinstance(anchor, (list, tuple)) or len(anchor) < 3:
            continue
        try:
            anchor_xyz = [float(anchor[0]), float(anchor[1]), float(anchor[2])]
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(anchor_xyz)):
            continue
        next_cfg = dict(raw_configs.get(pad_id) or {})
        next_cfg["first_surface_x"] = float(anchor_xyz[0])
        next_cfg["first_surface_y"] = float(anchor_xyz[1])
        next_cfg["first_surface_z"] = float(anchor_xyz[2])
        next_cfg[ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY] = True
        raw_configs[pad_id] = next_cfg
        updated_any = True

    if not updated_any:
        return []

    updated_records = sync_pilot_surfaces_to_parents(
        apply_pad_layout(
            records=working_records,
            pads=pads,
            plan_by_pad_id=_build_pad_plan_map(pads),
        )
    )
    st.session_state["wt_records"] = list(updated_records)
    st.session_state["wt_pad_last_applied_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.session_state["wt_pad_auto_applied_on_import"] = False
    return _queue_surface_edit_feedback(
        _changed_surface_well_names(working_records, updated_records),
        source=(
            "three_viewer_pad_layout"
            if str(source).strip() == "three_viewer"
            else str(source).strip() or "pad_layout"
        ),
    )


def _handle_three_edit_event(event: object) -> bool:
    return ptc_edit_targets.handle_three_edit_event(
        st.session_state,
        event,
        apply_changes=lambda changes, source: _apply_edit_targets_changes(
            changes,
            source=source,
        ),
        apply_pad_changes=lambda changes, source: _apply_edit_pad_changes(
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
    st.session_state["wt_last_calc_param_signature"] = None
    st.session_state[WT_LAST_WELL_CALC_OVERRIDE_SIGNATURE_KEY] = None
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
    st.session_state["wt_edit_targets_highlight_points"] = {}


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
    st.session_state["wt_pad_last_applied_at"] = ""
    st.session_state["wt_pad_auto_applied_on_import"] = False
    st.session_state[ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = False
    st.session_state.pop("wt_pad_selected_id", None)
    for key in list(st.session_state.keys()):
        if str(key).startswith("wt_pad_cfg_") or str(key).startswith(
            "wt_pad_fixed_slots_editor_"
        ):
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
        line_dash = "solid"
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
        target_points = tuple(getattr(target_only, "target_points", ()) or ())
        if not target_points:
            target_points = (target_only.surface, target_only.t1, target_only.t3)
        target_labels = tuple(getattr(target_only, "target_labels", ()) or ())
        if len(target_labels) != len(target_points):
            target_labels = _target_only_fallback_labels(
                point_count=len(target_points),
                target_pairs=tuple(getattr(target_only, "target_pairs", ()) or ()),
            )
        marker_x = np.array([point.x for point in target_points], dtype=float)
        marker_y = np.array([point.y for point in target_points], dtype=float)
        x_arrays.append(marker_x)
        y_arrays.append(marker_y)
        if not focus_set or str(target_only.name) in focus_set:
            x_focus_arrays.append(marker_x)
            y_focus_arrays.append(marker_y)
        customdata = np.array(
            [
                [label, target_only.status, target_only.problem or "—"]
                for label in target_labels
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
        label_point = target_points[min(1, len(target_points) - 1)]
        fig.add_trace(
            _t1_name_trace_2d(
                well_name=str(target_only.name),
                x_value=float(label_point.x),
                y_value=float(label_point.y),
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


def _target_only_fallback_labels(
    *,
    point_count: int,
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = (),
) -> tuple[str, ...]:
    if point_count <= 0:
        return ()
    if len(target_pairs) > 1 and point_count == 1 + 2 * len(target_pairs):
        labels = ["S"]
        for level_index in range(1, len(target_pairs) + 1):
            labels.extend([f"{level_index}_t1", f"{level_index}_t3"])
        return tuple(labels)
    if point_count == 2:
        return ("t1", "t3")
    if point_count == 3:
        return ("S", "t1", "t3")
    return ("S", *(f"P{index}" for index in range(1, point_count)))


def _all_wells_anticollision_plan_figure(
    analysis: AntiCollisionAnalysis,
    *,
    previous_successes_by_name: Mapping[str, SuccessfulWellPlan] | None = None,
    target_only_wells: list[_TargetOnlyWell] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    name_to_color: Mapping[str, str] | None = None,
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
    target_color_map = dict(name_to_color or {})

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
                    name=f"{well.name}: до пересчёта",
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

    analysis_reference_wells = _analysis_reference_wells(analysis)
    display_only_reference_wells = _display_only_reference_wells_for_analysis(
        reference_wells=reference_wells,
        analysis_reference_wells=analysis_reference_wells,
    )
    for reference_well in display_only_reference_wells:
        stations = reference_well.stations
        if stations.empty or not {"X_m", "Y_m", "MD_m"}.issubset(stations.columns):
            continue
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        md_values = stations["MD_m"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=reference_well_display_label(reference_well),
                legendgroup=str(reference_well.name),
                showlegend=False,
                line={
                    "width": 1.5,
                    "color": REFERENCE_WELL_KIND_COLORS.get(
                        str(reference_well.kind),
                        "#A0A0A0",
                    ),
                },
                customdata=np.column_stack([md_values]),
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        x_arrays.append(x_values)
        y_arrays.append(y_values)
        if not focus_set:
            x_focus_arrays.append(x_values)
            y_focus_arrays.append(y_values)

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
    visual_reference_wells = (
        *analysis_reference_wells,
        *display_only_reference_wells,
    )
    for kind in _reference_kinds_present(visual_reference_wells):
        fig.add_trace(_reference_legend_trace_2d(kind))
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label_trace = _reference_name_trace_2d(
            visual_reference_wells,
            kind=kind,
        )
        if label_trace is not None:
            fig.add_trace(label_trace)
    pad_label_trace = _reference_pad_label_trace_2d(visual_reference_wells)
    if pad_label_trace is not None:
        fig.add_trace(pad_label_trace)

    for target_only in target_only_wells or ():
        line_color = target_color_map.get(str(target_only.name), "#6B7280")
        target_points = tuple(getattr(target_only, "target_points", ()) or ())
        if not target_points:
            target_points = (target_only.surface, target_only.t1, target_only.t3)
        target_labels = tuple(getattr(target_only, "target_labels", ()) or ())
        if len(target_labels) != len(target_points):
            target_labels = _target_only_fallback_labels(
                point_count=len(target_points),
                target_pairs=tuple(getattr(target_only, "target_pairs", ()) or ()),
            )
        marker_x = np.array([point.x for point in target_points], dtype=float)
        marker_y = np.array([point.y for point in target_points], dtype=float)
        x_arrays.append(marker_x)
        y_arrays.append(marker_y)
        if not focus_set or str(target_only.name) in focus_set:
            x_focus_arrays.append(marker_x)
            y_focus_arrays.append(marker_y)
        customdata = np.array(
            [
                [label, target_only.status, target_only.problem or "—"]
                for label in target_labels
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
        label_point = target_points[min(1, len(target_points) - 1)]
        fig.add_trace(
            _t1_name_trace_2d(
                well_name=str(target_only.name),
                x_value=float(label_point.x),
                y_value=float(label_point.y),
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
        title="Anti-collision: план E-N с конусами неопределённости",
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
    disabled: bool = False,
) -> TrajectoryConfig:
    binding.render_block(title=title, on_change=on_change, disabled=disabled)
    return binding.build_config()


def _coerce_calc_param_override_value(
    *,
    suffix: str,
    value: object,
) -> float | int | str | bool | None:
    default_value = calc_param_defaults()[suffix]
    if isinstance(default_value, bool):
        return bool(value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(default_value, float):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return float(numeric_value)
    return str(value)


def _normalized_calc_param_override_values(
    raw_values: object,
) -> dict[str, float | int | str | bool]:
    if not isinstance(raw_values, Mapping):
        return {}
    normalized: dict[str, float | int | str | bool] = {}
    for suffix in calc_param_defaults():
        if suffix not in raw_values:
            continue
        coerced = _coerce_calc_param_override_value(
            suffix=suffix,
            value=raw_values[suffix],
        )
        if coerced is not None:
            normalized[suffix] = coerced
    return normalized


def _manual_well_calc_override_enabled() -> bool:
    return bool(st.session_state.get(WT_WELL_CALC_OVERRIDE_ENABLED_KEY, False))


def _preserve_manual_well_calc_override_widget_state() -> None:
    """Prevent Streamlit widget cleanup for per-well override controls.

    These controls live inside the run-section fragment, while reference imports and
    other actions can rerun different fragments. Without an explicit self-assignment,
    Streamlit may drop widget-backed state like the override enabled toggle, which then
    makes the last-run override signature look stale even though calc params did not
    actually change.
    """

    keys_to_preserve = [
        WT_WELL_CALC_OVERRIDE_ENABLED_KEY,
        WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY,
        WT_WELL_CALC_OVERRIDE_SELECTION_KEY,
    ]
    keys_to_preserve.extend(
        str(key)
        for key in tuple(st.session_state.keys())
        if str(key).startswith("wt_well_calc_override_profile_name__")
    )
    for key in keys_to_preserve:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def _queue_manual_well_calc_override_enabled(enabled: bool) -> None:
    st.session_state[WT_WELL_CALC_OVERRIDE_ENABLED_PENDING_KEY] = bool(enabled)


def _consume_manual_well_calc_override_enabled() -> None:
    pending_value = st.session_state.pop(
        WT_WELL_CALC_OVERRIDE_ENABLED_PENDING_KEY,
        None,
    )
    if pending_value is None:
        return
    st.session_state[WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = bool(pending_value)


def _queue_manual_well_calc_active_profile(profile_id: str) -> None:
    st.session_state[WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY] = str(
        profile_id
    ).strip()


def _consume_manual_well_calc_active_profile() -> None:
    pending_profile_id = st.session_state.pop(
        WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_PENDING_KEY,
        None,
    )
    if pending_profile_id is None:
        return
    _set_manual_well_calc_active_profile(str(pending_profile_id))


def _normalized_manual_well_profile_payload(
    *,
    profile_id: str,
    raw_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "name": str(raw_payload.get("name", profile_id)).strip() or str(profile_id),
        "values": _normalized_calc_param_override_values(raw_payload.get("values", {})),
        "source": str(raw_payload.get("source", "")).strip(),
        "note": str(raw_payload.get("note", "")).strip(),
    }


def _manual_well_calc_profiles() -> dict[str, dict[str, object]]:
    raw_state = st.session_state.get(WT_WELL_CALC_OVERRIDE_STATE_KEY, {})
    if not isinstance(raw_state, Mapping):
        st.session_state[WT_WELL_CALC_OVERRIDE_STATE_KEY] = {}
        return {}
    normalized: dict[str, dict[str, object]] = {}
    legacy_assignments: dict[str, str] = {}
    changed = False
    legacy_state = False
    for raw_profile_id, raw_payload in raw_state.items():
        profile_id = str(raw_profile_id).strip()
        if not profile_id or not isinstance(raw_payload, Mapping):
            changed = True
            continue
        if "name" not in raw_payload:
            legacy_state = True
            changed = True
            legacy_well_name = (
                str(raw_payload.get("well_name", profile_id)).strip() or profile_id
            )
            legacy_assignments[legacy_well_name] = profile_id
        payload = _normalized_manual_well_profile_payload(
            profile_id=profile_id,
            raw_payload=raw_payload,
        )
        normalized[profile_id] = payload
        expected_name = (
            str(raw_payload.get("name", profile_id)).strip() or str(profile_id)
        )
        expected_values = _normalized_calc_param_override_values(
            raw_payload.get("values", {})
        )
        expected_source = str(raw_payload.get("source", "")).strip()
        expected_note = str(raw_payload.get("note", "")).strip()
        if (
            payload["name"] != expected_name
            or payload["values"] != expected_values
            or payload["source"] != expected_source
            or payload["note"] != expected_note
        ):
            changed = True
    if changed:
        st.session_state[WT_WELL_CALC_OVERRIDE_STATE_KEY] = normalized
    if legacy_state:
        raw_assignments = st.session_state.get(WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY)
        if not isinstance(raw_assignments, Mapping):
            st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = dict(
                legacy_assignments
            )
        else:
            merged_assignments = {
                str(raw_well_name).strip(): str(raw_profile_id).strip()
                for raw_well_name, raw_profile_id in raw_assignments.items()
                if str(raw_well_name).strip() and str(raw_profile_id).strip()
            }
            assignments_changed = merged_assignments != dict(raw_assignments)
            for well_name, profile_id in legacy_assignments.items():
                current_profile_id = str(merged_assignments.get(well_name, "")).strip()
                if current_profile_id in normalized:
                    continue
                if current_profile_id != profile_id:
                    merged_assignments[well_name] = profile_id
                    assignments_changed = True
            if assignments_changed:
                st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = (
                    merged_assignments
                )
    return normalized


def _manual_well_calc_profile_assignments(
    *,
    available_names: Iterable[str] | None = None,
) -> dict[str, str]:
    profiles = _manual_well_calc_profiles()
    raw_assignments = st.session_state.get(WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY, {})
    if not isinstance(raw_assignments, Mapping):
        st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = {}
        return {}
    visible_by_key = (
        {well_name_key(name): str(name) for name in available_names}
        if available_names is not None
        else None
    )
    normalized: dict[str, str] = {}
    changed = False
    for raw_well_name, raw_profile_id in raw_assignments.items():
        raw_well_name_text = str(raw_well_name)
        raw_profile_id_text = str(raw_profile_id)
        well_name = raw_well_name_text.strip()
        profile_id = raw_profile_id_text.strip()
        if not well_name or profile_id not in profiles:
            changed = True
            continue
        if visible_by_key is not None:
            visible_name = visible_by_key.get(well_name_key(well_name))
            if visible_name is None:
                changed = True
                continue
            well_name = visible_name
        normalized[well_name] = profile_id
        if well_name != raw_well_name_text or profile_id != raw_profile_id_text:
            changed = True
    if changed:
        st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = normalized
    return normalized


def _manual_well_calc_profile_option_ids() -> list[str]:
    return [
        profile_id
        for profile_id, _payload in sorted(
            _manual_well_calc_profiles().items(),
            key=lambda item: (
                well_name_natural_sort_key(str(item[1].get("name", item[0]))),
                str(item[0]),
            ),
        )
    ]


def _set_manual_well_calc_active_profile(profile_id: str) -> None:
    active_profile_id = str(profile_id).strip()
    if active_profile_id not in _manual_well_calc_profiles():
        active_profile_id = ""
    st.session_state[WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = active_profile_id


def _manual_well_calc_active_profile_id(*, persist: bool = True) -> str:
    profiles = _manual_well_calc_profiles()
    active_profile_id = str(
        st.session_state.get(WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY, "")
    ).strip()
    if active_profile_id not in profiles:
        option_ids = _manual_well_calc_profile_option_ids()
        active_profile_id = option_ids[0] if option_ids else ""
        if persist:
            st.session_state[WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY] = active_profile_id
    return active_profile_id


def _manual_well_calc_profile_label(
    *,
    profile_id: str,
    assignments: Mapping[str, str],
) -> str:
    payload = dict(_manual_well_calc_profiles().get(str(profile_id), {}))
    assigned_count = sum(
        1
        for assigned_profile_id in assignments.values()
        if assigned_profile_id == profile_id
    )
    suffix = f" ({assigned_count} скв.)" if assigned_count > 0 else ""
    return f"{str(payload.get('name', profile_id)).strip() or str(profile_id)}{suffix}"


def _manual_well_calc_override_items() -> tuple[
    tuple[str, tuple[tuple[str, object], ...]],
    ...,
]:
    profiles = _manual_well_calc_profiles()
    assignments = _manual_well_calc_profile_assignments()
    items: list[tuple[str, tuple[tuple[str, object], ...]]] = []
    for well_name, profile_id in sorted(assignments.items()):
        values = _normalized_calc_param_override_values(
            profiles.get(str(profile_id), {}).get("values", {})
        )
        if not values:
            continue
        items.append(
            (
                str(well_name),
                tuple((suffix, values[suffix]) for suffix in sorted(values)),
            )
        )
    return tuple(items)


def _manual_well_calc_override_signature() -> tuple[object, ...]:
    if not _manual_well_calc_override_enabled():
        return (False, ())
    items = _manual_well_calc_override_items()
    if not items:
        return (True, ())
    return (True, items)


def _coerce_manual_well_override_selection(
    names: object,
    *,
    available_names: list[str],
) -> list[str]:
    visible_by_key = {well_name_key(name): str(name) for name in available_names}
    coerced: list[str] = []
    seen: set[str] = set()
    for raw_name in names if isinstance(names, (list, tuple, set)) else []:
        visible_name = visible_by_key.get(well_name_key(raw_name))
        if visible_name is None or visible_name in seen:
            continue
        coerced.append(visible_name)
        seen.add(visible_name)
    return coerced


def _sync_manual_well_override_selection(
    *,
    available_names: list[str],
) -> list[str]:
    pending_names = st.session_state.pop(
        WT_WELL_CALC_OVERRIDE_SELECTION_PENDING_KEY,
        None,
    )
    if pending_names is not None:
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_KEY] = (
            _coerce_manual_well_override_selection(
                pending_names,
                available_names=available_names,
            )
        )
    current = _coerce_manual_well_override_selection(
        st.session_state.get(WT_WELL_CALC_OVERRIDE_SELECTION_KEY, []),
        available_names=available_names,
    )
    if current != st.session_state.get(WT_WELL_CALC_OVERRIDE_SELECTION_KEY, []):
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_KEY] = list(current)
    return current


def _manual_well_names_for_profile_ids(
    *,
    profile_ids: Iterable[str],
    available_names: Iterable[str] | None = None,
) -> list[str]:
    normalized_profile_ids = {
        str(profile_id).strip()
        for profile_id in profile_ids
        if str(profile_id).strip()
    }
    if not normalized_profile_ids:
        return []
    normalized_available_names = (
        _unique_well_names(available_names) if available_names is not None else None
    )
    assignments = _manual_well_calc_profile_assignments(
        available_names=normalized_available_names
    )
    ordered_names = (
        normalized_available_names
        if normalized_available_names is not None
        else _unique_well_names(assignments.keys())
    )
    return [
        str(well_name)
        for well_name in ordered_names
        if assignments.get(str(well_name)) in normalized_profile_ids
    ]


def _queue_batch_selection_additions(
    *,
    well_names: Iterable[str],
    available_names: Iterable[str],
) -> list[str]:
    normalized_available_names = _unique_well_names(available_names)
    if not normalized_available_names:
        return []
    current_source = st.session_state.get("wt_pending_selected_names")
    if current_source is None:
        current_source = st.session_state.get("wt_selected_names", [])
    current_selection = _coerce_manual_well_override_selection(
        current_source,
        available_names=normalized_available_names,
    )
    additions = _coerce_manual_well_override_selection(
        well_names,
        available_names=normalized_available_names,
    )
    if not additions:
        return current_selection
    merged_selection = _coerce_manual_well_override_selection(
        [*current_selection, *additions],
        available_names=normalized_available_names,
    )
    if merged_selection != current_selection:
        st.session_state["wt_pending_selected_names"] = list(merged_selection)
    return merged_selection


def _set_manual_well_override_editor_from_config(
    config: TrajectoryConfig,
) -> None:
    set_calc_param_state_values(
        prefix=WT_WELL_OVERRIDE_EDITOR.prefix,
        values=calc_param_state_values_from_config(config),
    )


def _effective_manual_well_profile_values(
    *,
    base_config: TrajectoryConfig,
    profile_id: str,
    records_by_name: Mapping[str, WelltrackRecord] | None = None,
) -> dict[str, float | int | str | bool]:
    values = calc_param_state_values_from_config(base_config)
    normalized_profile_id = str(profile_id).strip()
    payload = _manual_well_calc_profiles().get(normalized_profile_id, {})
    manual_values = _normalized_calc_param_override_values(payload.get("values", {}))
    values.update(manual_values)
    kop_function = kop_min_vertical_function_from_state(prefix=WT_CALC_PARAMS.prefix)
    records_lookup = (
        records_by_name if isinstance(records_by_name, Mapping) else None
    )
    if (
        not normalized_profile_id
        or records_lookup is None
        or kop_function is None
        or "kop_min_vertical" in manual_values
    ):
        return values
    effective_config = build_config_from_values(values)
    assignments = _manual_well_calc_profile_assignments()
    for well_name, assigned_profile_id in sorted(
        assignments.items(),
        key=lambda item: well_name_natural_sort_key(str(item[0])),
    ):
        if str(assigned_profile_id).strip() != normalized_profile_id:
            continue
        record = records_lookup.get(str(well_name))
        if record is None:
            continue
        evaluated_kop_m = _evaluated_kop_min_vertical_for_record(
            record=record,
            base_config=effective_config,
            kop_function=kop_function,
        )
        if evaluated_kop_m is not None:
            values["kop_min_vertical"] = float(evaluated_kop_m)
            break
    return values


def _manual_well_override_selection_signature(
    *,
    active_profile_id: str,
    values: Mapping[str, float | int | str | bool],
    source: str = "profile",
) -> tuple[object, ...]:
    signature_source = "base" if str(source).strip() == "base" else "profile"
    return (
        signature_source,
        str(active_profile_id),
        WT_CALC_PARAMS.state_signature(),
        tuple((suffix, values[suffix]) for suffix in sorted(values)),
    )


def _sync_manual_well_override_editor_selection(
    *,
    base_config: TrajectoryConfig,
    active_profile_id: str,
    records_by_name: Mapping[str, WelltrackRecord] | None = None,
) -> None:
    base_values = calc_param_state_values_from_config(base_config)
    effective_values = (
        _effective_manual_well_profile_values(
            base_config=base_config,
            profile_id=active_profile_id,
            records_by_name=records_by_name,
        )
        if active_profile_id
        else base_values
    )
    selection_signature = _manual_well_override_selection_signature(
        active_profile_id=active_profile_id,
        values=effective_values,
    )
    base_loaded_signature = _manual_well_override_selection_signature(
        active_profile_id=active_profile_id,
        values=base_values,
        source="base",
    )
    stored_signature = st.session_state.get(
        WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY,
        (),
    )
    if not isinstance(stored_signature, (list, tuple)):
        stored_signature = ()
    if tuple(stored_signature) in (
        selection_signature,
        base_loaded_signature,
    ):
        return
    editor_config = (
        build_config_from_values(effective_values)
        if active_profile_id
        else base_config
    )
    _set_manual_well_override_editor_from_config(editor_config)
    st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = (
        selection_signature
    )


def _manual_well_override_diff_values(
    *,
    base_config: TrajectoryConfig,
    override_config: TrajectoryConfig,
) -> dict[str, float | int | str | bool]:
    base_values = calc_param_state_values_from_config(base_config)
    override_values = calc_param_state_values_from_config(override_config)
    return _manual_well_override_delta_values(
        reference_values=base_values,
        target_values=override_values,
    )


def _manual_well_override_changed_values(
    *,
    current_config: TrajectoryConfig,
    editor_config: TrajectoryConfig,
) -> dict[str, float | int | str | bool]:
    current_values = calc_param_state_values_from_config(current_config)
    editor_values = calc_param_state_values_from_config(editor_config)
    return _manual_well_override_delta_values(
        reference_values=current_values,
        target_values=editor_values,
    )


def _manual_well_override_delta_values(
    *,
    reference_values: Mapping[str, float | int | str | bool],
    target_values: Mapping[str, float | int | str | bool],
) -> dict[str, float | int | str | bool]:
    delta_values: dict[str, float | int | str | bool] = {}
    paired_optional_fields = (
        ("dls_build2_enabled", "dls_build2_max"),
        ("min_hold_inc_enabled", "min_hold_inc"),
    )
    optional_field_suffixes = {
        suffix
        for enabled_suffix, value_suffix in paired_optional_fields
        for suffix in (enabled_suffix, value_suffix)
    }
    for suffix, target_value in target_values.items():
        if suffix in optional_field_suffixes:
            continue
        if target_value != reference_values.get(suffix):
            delta_values[suffix] = target_value
    for enabled_suffix, value_suffix in paired_optional_fields:
        reference_enabled = bool(reference_values.get(enabled_suffix, False))
        target_enabled = bool(target_values.get(enabled_suffix, False))
        if reference_enabled != target_enabled:
            delta_values[enabled_suffix] = target_enabled
            if target_enabled:
                delta_values[value_suffix] = target_values[value_suffix]
            continue
        if target_enabled and target_values.get(value_suffix) != reference_values.get(
            value_suffix
        ):
            delta_values[value_suffix] = target_values[value_suffix]
    return delta_values


def _manual_well_override_changed_field_count_from_values(
    *,
    reference_values: Mapping[str, float | int | str | bool],
    target_values: Mapping[str, float | int | str | bool],
) -> int:
    changed_fields: set[str] = set()
    paired_optional_fields = (
        ("dls_build2_enabled", "dls_build2_max", "dls_build2"),
        ("min_hold_inc_enabled", "min_hold_inc", "min_hold_inc"),
    )
    optional_field_suffixes = {
        suffix
        for enabled_suffix, value_suffix, _field_name in paired_optional_fields
        for suffix in (enabled_suffix, value_suffix)
    }
    for suffix, target_value in target_values.items():
        if suffix in optional_field_suffixes:
            continue
        if target_value != reference_values.get(suffix):
            changed_fields.add(suffix)
    for enabled_suffix, value_suffix, field_name in paired_optional_fields:
        reference_enabled = bool(reference_values.get(enabled_suffix, False))
        target_enabled = bool(target_values.get(enabled_suffix, False))
        if reference_enabled != target_enabled:
            changed_fields.add(field_name)
            continue
        if target_enabled and target_values.get(value_suffix) != reference_values.get(
            value_suffix
        ):
            changed_fields.add(field_name)
    return int(len(changed_fields))


def _manual_well_override_diff_field_count(
    *,
    base_config: TrajectoryConfig,
    override_config: TrajectoryConfig,
) -> int:
    return _manual_well_override_changed_field_count_from_values(
        reference_values=calc_param_state_values_from_config(base_config),
        target_values=calc_param_state_values_from_config(override_config),
    )


def _manual_well_override_changed_field_count(
    *,
    current_config: TrajectoryConfig,
    editor_config: TrajectoryConfig,
) -> int:
    return _manual_well_override_changed_field_count_from_values(
        reference_values=calc_param_state_values_from_config(current_config),
        target_values=calc_param_state_values_from_config(editor_config),
    )


def _new_manual_well_calc_profile_id() -> str:
    existing_ids = set(_manual_well_calc_profiles())
    profile_index = 1
    while True:
        profile_id = f"cfg-{profile_index}"
        if profile_id not in existing_ids:
            return profile_id
        profile_index += 1


def _next_manual_well_calc_profile_name() -> str:
    existing_names = {
        well_name_key(str(payload.get("name", "")).strip())
        for payload in _manual_well_calc_profiles().values()
    }
    profile_index = 1
    while True:
        profile_name = f"Конфигурация {profile_index}"
        if well_name_key(profile_name) not in existing_names:
            return profile_name
        profile_index += 1


def _manual_well_calc_profile_name_key(active_profile_id: str) -> str:
    normalized_profile_id = str(active_profile_id).strip() or "none"
    return f"wt_well_calc_override_profile_name__{normalized_profile_id}"


def _unique_manual_well_calc_profile_name(
    *,
    profile_name: str,
    current_profile_id: str | None = None,
) -> str:
    desired_name = str(profile_name).strip() or _next_manual_well_calc_profile_name()
    profiles = _manual_well_calc_profiles()
    desired_key = well_name_key(desired_name)
    if all(
        profile_id == current_profile_id
        or well_name_key(str(payload.get("name", "")).strip()) != desired_key
        for profile_id, payload in profiles.items()
    ):
        return desired_name
    suffix = 2
    while True:
        candidate = f"{desired_name} ({suffix})"
        candidate_key = well_name_key(candidate)
        if all(
            profile_id == current_profile_id
            or well_name_key(str(payload.get("name", "")).strip()) != candidate_key
            for profile_id, payload in profiles.items()
        ):
            return candidate
        suffix += 1


def _store_manual_well_calc_profile(
    *,
    profile_id: str,
    profile_name: str,
    values: Mapping[str, object],
    source: str,
    note: str = "",
) -> str:
    normalized_profile_id = str(profile_id).strip()
    if not normalized_profile_id:
        return ""
    profiles = dict(_manual_well_calc_profiles())
    resolved_name = _unique_manual_well_calc_profile_name(
        profile_name=profile_name,
        current_profile_id=normalized_profile_id,
    )
    profiles[normalized_profile_id] = {
        "name": resolved_name,
        "values": _normalized_calc_param_override_values(values),
        "source": str(source).strip(),
        "note": str(note).strip(),
    }
    st.session_state[WT_WELL_CALC_OVERRIDE_STATE_KEY] = profiles
    st.session_state[WT_WELL_CALC_OVERRIDE_NAME_INPUT_ACTIVE_KEY] = ""
    return resolved_name


def _manual_well_calc_profile_id_by_name(profile_name: str) -> str:
    desired_key = well_name_key(str(profile_name).strip())
    if not desired_key:
        return ""
    for profile_id, payload in _manual_well_calc_profiles().items():
        current_name = str(payload.get("name", "")).strip()
        if well_name_key(current_name) == desired_key:
            return str(profile_id)
    return ""


def _manual_well_calc_profile_export_payload(profile_id: str) -> dict[str, object]:
    normalized_profile_id = str(profile_id).strip()
    if not normalized_profile_id:
        return {}
    payload = dict(_manual_well_calc_profiles().get(normalized_profile_id, {}))
    if not payload:
        return {}
    profile_name = str(payload.get("name", normalized_profile_id)).strip() or str(
        normalized_profile_id
    )
    return {
        "kind": WT_WELL_CALC_PROFILE_JSON_KIND,
        "schema_version": WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION,
        "name": profile_name,
        "values": _normalized_calc_param_override_values(payload.get("values", {})),
        "source": str(payload.get("source", "")).strip() or "Ручная настройка",
        "note": str(payload.get("note", "")).strip(),
    }


def _manual_well_calc_profile_export_json(profile_id: str) -> str:
    payload = _manual_well_calc_profile_export_payload(profile_id)
    if not payload:
        return ""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _manual_well_calc_profile_export_file_name(profile_id: str) -> str:
    payload = _manual_well_calc_profile_export_payload(profile_id)
    profile_name = str(payload.get("name", "")).strip() or "configuration"
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", profile_name).strip(" .")
    return f"{safe_name or 'configuration'}.json"


def _import_manual_well_calc_profile_json_bytes(
    raw_bytes: bytes,
) -> tuple[str, str, bool]:
    try:
        parsed = json.loads(bytes(raw_bytes).decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Не удалось прочитать JSON конфигурации.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON конфигурации должен содержать объект.")
    kind = str(parsed.get("kind", "")).strip()
    if kind and kind != WT_WELL_CALC_PROFILE_JSON_KIND:
        raise ValueError("JSON не похож на файл конфигурации расчёта.")
    schema_version = parsed.get("schema_version", WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION)
    try:
        schema_version_int = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise ValueError("У JSON конфигурации некорректная версия схемы.") from exc
    if schema_version_int != WT_WELL_CALC_PROFILE_JSON_SCHEMA_VERSION:
        raise ValueError("Версия схемы JSON конфигурации не поддерживается.")
    profile_name = str(parsed.get("name", "")).strip()
    if not profile_name:
        raise ValueError("В JSON конфигурации отсутствует имя.")
    values = _normalized_calc_param_override_values(parsed.get("values", {}))
    source = str(parsed.get("source", "")).strip() or "Импорт JSON"
    note = str(parsed.get("note", "")).strip()
    existing_profile_id = _manual_well_calc_profile_id_by_name(profile_name)
    profile_id = existing_profile_id or _new_manual_well_calc_profile_id()
    resolved_name = _store_manual_well_calc_profile(
        profile_id=profile_id,
        profile_name=profile_name,
        values=values,
        source=source,
        note=note,
    )
    _queue_manual_well_calc_active_profile(profile_id)
    return profile_id, resolved_name, bool(existing_profile_id)


def _import_manual_well_calc_profile_json_payloads(
    payloads: Iterable[tuple[str, bytes]],
) -> tuple[int, int, list[str], str, tuple[str, ...]]:
    imported_count = 0
    updated_count = 0
    error_messages: list[str] = []
    last_profile_id = ""
    touched_profile_ids: list[str] = []
    for file_name, raw_bytes in payloads:
        normalized_name = str(file_name).strip() or "configuration.json"
        try:
            profile_id, _resolved_name, updated_existing = (
                _import_manual_well_calc_profile_json_bytes(raw_bytes)
            )
        except ValueError as exc:
            error_messages.append(f"{normalized_name}: {exc}")
            continue
        imported_count += 1
        updated_count += int(updated_existing)
        last_profile_id = str(profile_id)
        touched_profile_ids.append(str(profile_id))
    if last_profile_id:
        _queue_manual_well_calc_active_profile(last_profile_id)
    return (
        imported_count,
        updated_count,
        error_messages,
        last_profile_id,
        tuple(_unique_well_names(touched_profile_ids)),
    )


def _create_manual_well_calc_profile() -> str:
    profile_id = _new_manual_well_calc_profile_id()
    profile_name = _next_manual_well_calc_profile_name()
    _store_manual_well_calc_profile(
        profile_id=profile_id,
        profile_name=profile_name,
        values={},
        source="Ручная настройка",
    )
    _queue_manual_well_calc_active_profile(profile_id)
    return profile_id


def _manual_well_calc_profile_save_source(existing_source: str) -> str:
    source = str(existing_source).strip()
    if source == "Импорт .dev":
        return "Импорт .dev + ручная настройка"
    if source.startswith("Импорт .dev +"):
        return source
    return "Ручная настройка"


def _apply_manual_well_override_editor(
    *,
    base_config: TrajectoryConfig,
    active_profile_id: str,
    profile_name: str,
) -> tuple[bool, str, int]:
    if not active_profile_id:
        return False, "", 0
    profiles = _manual_well_calc_profiles()
    existing_payload = dict(profiles.get(active_profile_id, {}))
    editor_config = WT_WELL_OVERRIDE_EDITOR.build_config()
    diff_values = _manual_well_override_diff_values(
        base_config=base_config,
        override_config=editor_config,
    )
    changed_field_count = _manual_well_override_diff_field_count(
        base_config=base_config,
        override_config=editor_config,
    )
    resolved_name = _store_manual_well_calc_profile(
        profile_id=active_profile_id,
        profile_name=profile_name,
        values=diff_values,
        source=_manual_well_calc_profile_save_source(
            str(existing_payload.get("source", "")).strip()
        ),
        note=str(existing_payload.get("note", "")).strip(),
    )
    updated_payload = dict(_manual_well_calc_profiles().get(active_profile_id, {}))
    changed = updated_payload != existing_payload
    return changed, resolved_name, int(changed_field_count)


def _apply_manual_well_override_editor_to_all_profiles(
    *,
    base_config: TrajectoryConfig,
    active_profile_id: str,
    profile_name: str,
    records_by_name: Mapping[str, WelltrackRecord] | None = None,
) -> tuple[bool, str, int, int]:
    normalized_profile_id = str(active_profile_id).strip()
    if not normalized_profile_id:
        return False, "", 0, 0
    profiles = _manual_well_calc_profiles()
    if normalized_profile_id not in profiles:
        return False, "", 0, 0
    active_effective_config = build_config_from_values(
        _effective_manual_well_profile_values(
            base_config=base_config,
            profile_id=normalized_profile_id,
            records_by_name=records_by_name,
        )
    )
    editor_config = WT_WELL_OVERRIDE_EDITOR.build_config()
    changed_values = _manual_well_override_changed_values(
        current_config=active_effective_config,
        editor_config=editor_config,
    )
    changed_field_count = _manual_well_override_changed_field_count(
        current_config=active_effective_config,
        editor_config=editor_config,
    )
    changed, resolved_name, _changed_field_count = _apply_manual_well_override_editor(
        base_config=base_config,
        active_profile_id=normalized_profile_id,
        profile_name=profile_name,
    )
    updated_profile_count = int(changed)
    if not changed_values:
        return changed, resolved_name, 0, updated_profile_count
    current_profiles = dict(_manual_well_calc_profiles())
    for profile_id, existing_payload_raw in current_profiles.items():
        normalized_target_profile_id = str(profile_id).strip()
        if not normalized_target_profile_id or normalized_target_profile_id == normalized_profile_id:
            continue
        existing_payload = dict(existing_payload_raw)
        target_effective_values = _effective_manual_well_profile_values(
            base_config=base_config,
            profile_id=normalized_target_profile_id,
            records_by_name=records_by_name,
        )
        target_effective_values.update(changed_values)
        merged_config = build_config_from_values(target_effective_values)
        diff_values = _manual_well_override_diff_values(
            base_config=base_config,
            override_config=merged_config,
        )
        _store_manual_well_calc_profile(
            profile_id=normalized_target_profile_id,
            profile_name=str(
                existing_payload.get("name", normalized_target_profile_id)
            ).strip()
            or str(normalized_target_profile_id),
            values=diff_values,
            source=_manual_well_calc_profile_save_source(
                str(existing_payload.get("source", "")).strip()
            ),
            note=str(existing_payload.get("note", "")).strip(),
        )
        updated_payload = dict(
            _manual_well_calc_profiles().get(normalized_target_profile_id, {})
        )
        if updated_payload != existing_payload:
            updated_profile_count += 1
    return bool(updated_profile_count), resolved_name, int(changed_field_count), int(
        updated_profile_count
    )


def _rename_manual_well_calc_profile(
    *,
    profile_id: str,
    profile_name: str,
) -> tuple[bool, str]:
    normalized_profile_id = str(profile_id).strip()
    if not normalized_profile_id:
        return False, ""
    existing_payload = dict(_manual_well_calc_profiles().get(normalized_profile_id, {}))
    if not existing_payload:
        return False, ""
    resolved_name = _store_manual_well_calc_profile(
        profile_id=normalized_profile_id,
        profile_name=profile_name,
        values=existing_payload.get("values", {}),
        source=str(existing_payload.get("source", "")).strip() or "Ручная настройка",
        note=str(existing_payload.get("note", "")).strip(),
    )
    updated_payload = dict(_manual_well_calc_profiles().get(normalized_profile_id, {}))
    return updated_payload != existing_payload, resolved_name


def _handle_manual_well_calc_profile_name_change(profile_id: str) -> None:
    normalized_profile_id = str(profile_id).strip()
    if not normalized_profile_id or not _manual_well_calc_override_enabled():
        return
    changed, resolved_name = _rename_manual_well_calc_profile(
        profile_id=normalized_profile_id,
        profile_name=str(
            st.session_state.get(
                _manual_well_calc_profile_name_key(normalized_profile_id),
                "",
            )
        ),
    )
    if changed:
        st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
            f'Имя конфигурации обновлено: "{resolved_name}".'
        )


def _assign_manual_well_calc_profile_to_wells(
    *,
    profile_id: str,
    well_names: Iterable[str],
) -> int:
    normalized_profile_id = str(profile_id).strip()
    if normalized_profile_id not in _manual_well_calc_profiles():
        return 0
    assignments = dict(_manual_well_calc_profile_assignments())
    assigned_count = 0
    for well_name in _unique_well_names(well_names):
        if assignments.get(str(well_name)) == normalized_profile_id:
            continue
        assignments[str(well_name)] = normalized_profile_id
        assigned_count += 1
    st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = assignments
    return assigned_count


def _clear_manual_well_profile_assignments(
    *,
    well_names: Iterable[str],
) -> int:
    assignments = dict(_manual_well_calc_profile_assignments())
    cleared_count = 0
    for well_name in _unique_well_names(well_names):
        if assignments.pop(str(well_name), None) is not None:
            cleared_count += 1
    st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = assignments
    return cleared_count


def _delete_manual_well_calc_profile(profile_id: str) -> tuple[int, int]:
    normalized_profile_id = str(profile_id).strip()
    profiles = dict(_manual_well_calc_profiles())
    if normalized_profile_id not in profiles:
        return 0, 0
    profiles.pop(normalized_profile_id, None)
    assignments = dict(_manual_well_calc_profile_assignments())
    cleared_assignments = 0
    for well_name, assigned_profile_id in list(assignments.items()):
        if assigned_profile_id != normalized_profile_id:
            continue
        assignments.pop(well_name, None)
        cleared_assignments += 1
    st.session_state[WT_WELL_CALC_OVERRIDE_STATE_KEY] = profiles
    st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = assignments
    _queue_manual_well_calc_active_profile("")
    return 1, cleared_assignments


def _dev_summary_override_values(
    summary: ptc_target_import.DevTargetImportSummary,
) -> dict[str, float | int | str | bool]:
    if bool(getattr(summary, "simple_target_only", False)):
        return {}
    entry_inc_target = min(max(float(summary.entry_inc_deg), 0.5), 89.0)
    values: dict[str, float | int | str | bool] = {
        "kop_min_vertical": float(summary.kop_md_m),
        "use_fixed_kop": True,
        "entry_inc_target": float(entry_inc_target),
    }
    build_dls_values = tuple(
        float(value)
        for value in summary.build1_dls_deg_per_30m
        if np.isfinite(float(value))
    )
    if build_dls_values:
        values["dls_build_max"] = float(dls_to_pi(max(build_dls_values)))
    build2_dls_values = tuple(
        float(value)
        for value in summary.build2_dls_deg_per_30m
        if np.isfinite(float(value))
    )
    if (
        build_dls_values
        and build2_dls_values
        and str(summary.profile_label).strip() != "J-профиль"
    ):
        build1_max = float(max(build_dls_values))
        build2_max = float(max(build2_dls_values))
        if abs(build2_max - build1_max) > 1e-6:
            values["dls_build2_enabled"] = True
            values["dls_build2_max"] = float(dls_to_pi(build2_max))
    horizontal_dls_values = tuple(
        float(value)
        for value in summary.horizontal_dls_deg_per_30m
        if np.isfinite(float(value))
    )
    if horizontal_dls_values:
        values["dls_horizontal_max"] = float(dls_to_pi(max(horizontal_dls_values)))
    if str(summary.profile_label).strip() == "J-профиль":
        values["j_profile_policy"] = J_PROFILE_POLICY_PREFER
        values["offer_j_profile"] = True
    return values


def _dev_summary_uses_separate_build2(
    summary: ptc_target_import.DevTargetImportSummary,
) -> bool:
    if bool(getattr(summary, "simple_target_only", False)):
        return False
    if str(summary.profile_label).strip() == "J-профиль":
        return False
    build1_dls_values = tuple(
        float(value)
        for value in summary.build1_dls_deg_per_30m
        if np.isfinite(float(value))
    )
    build2_dls_values = tuple(
        float(value)
        for value in summary.build2_dls_deg_per_30m
        if np.isfinite(float(value))
    )
    if not build1_dls_values or not build2_dls_values:
        return False
    return bool(abs(max(build2_dls_values) - max(build1_dls_values)) > 1e-6)


def _manual_well_calc_profile_id_from_dev_summary(well_name: str) -> str:
    return f"dev::{well_name_key(well_name)}"


def _apply_dev_params_to_manual_well_overrides(
    *,
    well_names: Iterable[str],
) -> tuple[int, list[str]]:
    summaries = tuple(st.session_state.get("wt_imported_dev_params", ()))
    summary_by_key = {
        well_name_key(summary.well_name): summary
        for summary in summaries
        if str(summary.well_name).strip()
        and not bool(getattr(summary, "simple_target_only", False))
    }
    profiles = dict(_manual_well_calc_profiles())
    assignments = dict(_manual_well_calc_profile_assignments())
    applied_count = 0
    missing_names: list[str] = []
    first_profile_id = ""
    for well_name in _unique_well_names(well_names):
        summary = summary_by_key.get(well_name_key(well_name))
        if summary is None:
            missing_names.append(str(well_name))
            continue
        profile_id = _manual_well_calc_profile_id_from_dev_summary(well_name)
        stored_values = _dev_summary_override_values(summary)
        if not _dev_summary_uses_separate_build2(summary):
            stored_values.pop("dls_build2_enabled", None)
            stored_values.pop("dls_build2_max", None)
        profiles[profile_id] = {
            "name": str(well_name),
            "values": stored_values,
            "source": "Импорт .dev",
            "note": "KOP / INC / PI из .dev",
        }
        assignments[str(well_name)] = profile_id
        if not first_profile_id:
            first_profile_id = profile_id
        applied_count += 1
    st.session_state[WT_WELL_CALC_OVERRIDE_STATE_KEY] = profiles
    st.session_state[WT_WELL_CALC_OVERRIDE_ASSIGNMENTS_KEY] = assignments
    if first_profile_id:
        _queue_manual_well_calc_active_profile(first_profile_id)
    return applied_count, missing_names


def _imported_dev_param_well_names(
    *,
    available_names: Iterable[str],
    summaries: Iterable[object] | None = None,
) -> list[str]:
    normalized_available_names = _unique_well_names(available_names)
    if not normalized_available_names:
        return []
    summary_source = (
        tuple(st.session_state.get("wt_imported_dev_params", ()))
        if summaries is None
        else tuple(summaries)
    )
    eligible_keys = {
        well_name_key(getattr(summary, "well_name", ""))
        for summary in summary_source
        if str(getattr(summary, "well_name", "")).strip()
        and not bool(getattr(summary, "simple_target_only", False))
    }
    if not eligible_keys:
        return []
    return [
        str(well_name)
        for well_name in normalized_available_names
        if well_name_key(well_name) in eligible_keys
    ]


def _auto_apply_imported_dev_param_overrides(
    *,
    records: Iterable[WelltrackRecord],
    dev_summaries: Iterable[object],
) -> int:
    available_names = _unique_well_names(record.name for record in records)
    imported_dev_target_names = _imported_dev_param_well_names(
        available_names=available_names,
        summaries=dev_summaries,
    )
    if not imported_dev_target_names:
        return 0
    applied_count, missing_names = _apply_dev_params_to_manual_well_overrides(
        well_names=imported_dev_target_names,
    )
    if applied_count <= 0:
        return 0
    st.session_state[WT_WELL_CALC_OVERRIDE_ENABLED_KEY] = True
    _queue_batch_selection_additions(
        well_names=imported_dev_target_names,
        available_names=available_names,
    )
    feedback = (
        "После импорта автоматически созданы или обновлены "
        f"конфигурации из .dev: {applied_count}."
    )
    if missing_names:
        feedback += " Без .dev параметров: " + ", ".join(missing_names) + "."
    st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = feedback
    return applied_count


def _manual_well_override_rows(
    *,
    available_names: list[str],
) -> list[dict[str, object]]:
    profiles = _manual_well_calc_profiles()
    assignments = _manual_well_calc_profile_assignments(available_names=available_names)
    wells_by_profile: dict[str, list[str]] = {profile_id: [] for profile_id in profiles}
    unassigned_wells: list[str] = []
    for well_name in available_names:
        profile_id = assignments.get(str(well_name))
        if profile_id in profiles:
            wells_by_profile.setdefault(profile_id, []).append(str(well_name))
        else:
            unassigned_wells.append(str(well_name))
    rows: list[dict[str, object]] = []
    for profile_id in _manual_well_calc_profile_option_ids():
        payload = dict(profiles.get(profile_id, {}))
        values = _normalized_calc_param_override_values(payload.get("values", {}))
        rows.append(
            {
                "Конфигурация": str(payload.get("name", profile_id)).strip()
                or str(profile_id),
                "Скважины": ", ".join(wells_by_profile.get(profile_id, ())) or "—",
                "Источник": str(payload.get("source", "")).strip() or "—",
                "Изменено полей": int(len(values)),
                "Параметры": ", ".join(
                    _CALC_PARAM_OVERRIDE_LABELS.get(suffix, suffix)
                    for suffix in values
                )
                or "—",
                "Примечание": str(payload.get("note", "")).strip() or "—",
            }
        )
    rows.append(
        {
            "Конфигурация": "Общие параметры",
            "Скважины": ", ".join(unassigned_wells) or "—",
            "Источник": "—",
            "Изменено полей": 0,
            "Параметры": "—",
            "Примечание": "—",
        }
    )
    return rows


def _render_manual_well_calc_overrides(
    *,
    records: list[WelltrackRecord],
) -> None:
    _preserve_manual_well_calc_override_widget_state()
    available_names = _unique_well_names(record.name for record in records)
    assignments = _manual_well_calc_profile_assignments(
        available_names=available_names
    )
    if not available_names:
        return
    base_config = WT_CALC_PARAMS.build_config()
    st.markdown("#### Индивидуальные параметры по скважинам")
    st.caption(
        "Создайте отдельные конфигурации расчёта и назначьте их нужным "
        "скважинам. Скважины без назначения считают общие параметры выше."
    )
    _consume_manual_well_calc_override_enabled()
    _consume_manual_well_calc_active_profile()
    imported_dev_target_names = _imported_dev_param_well_names(
        available_names=available_names,
    )
    toggle_cols = st.columns([2.2, 1.35, 2.1], gap="small")
    with toggle_cols[0]:
        st.toggle(
            "Использовать индивидуальные параметры",
            key=WT_WELL_CALC_OVERRIDE_ENABLED_KEY,
        )
    with toggle_cols[1]:
        apply_dev_clicked = st.button(
            "Подтянуть из .dev",
            icon=":material/file_download:",
            width="stretch",
            disabled=not _manual_well_calc_override_enabled()
            or not imported_dev_target_names,
        )
    overrides_enabled = _manual_well_calc_override_enabled()
    feedback_message = str(
        st.session_state.pop(WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY, "")
    ).strip()
    if feedback_message:
        st.info(feedback_message)
    selected_names = _sync_manual_well_override_selection(
        available_names=available_names
    )
    option_ids = _manual_well_calc_profile_option_ids()
    active_profile_id = _manual_well_calc_active_profile_id()
    records_by_name = {str(record.name): record for record in records}
    _sync_manual_well_override_editor_selection(
        base_config=base_config,
        active_profile_id=active_profile_id,
        records_by_name=records_by_name,
    )
    active_payload = dict(_manual_well_calc_profiles().get(active_profile_id, {}))
    profile_name_input_key = _manual_well_calc_profile_name_key(active_profile_id)
    last_name_input_profile_id = str(
        st.session_state.get(WT_WELL_CALC_OVERRIDE_NAME_INPUT_ACTIVE_KEY, "")
    ).strip()
    if active_profile_id and (
        profile_name_input_key not in st.session_state
        or last_name_input_profile_id != active_profile_id
    ):
        st.session_state[profile_name_input_key] = str(
            active_payload.get("name", active_profile_id)
        )
    st.session_state[WT_WELL_CALC_OVERRIDE_NAME_INPUT_ACTIVE_KEY] = active_profile_id
    control_col, editor_col = st.columns([1.1, 1.6], gap="large")
    with control_col:
        active_profile_id = str(
            st.selectbox(
                "Конфигурация",
                options=option_ids or [""],
                format_func=lambda profile_id: (
                    "Нет конфигураций"
                    if not str(profile_id).strip()
                    else _manual_well_calc_profile_label(
                        profile_id=str(profile_id),
                        assignments=assignments,
                    )
                ),
                key=WT_WELL_CALC_OVERRIDE_ACTIVE_PROFILE_KEY,
                disabled=not overrides_enabled or not option_ids,
            )
        ).strip()
        if active_profile_id not in _manual_well_calc_profiles():
            active_profile_id = ""
        active_payload = dict(_manual_well_calc_profiles().get(active_profile_id, {}))
        st.text_input(
            "Имя конфигурации",
            key=_manual_well_calc_profile_name_key(active_profile_id),
            on_change=_handle_manual_well_calc_profile_name_change,
            args=(active_profile_id,),
            disabled=not overrides_enabled or not active_profile_id,
        )
        profile_export_json = _manual_well_calc_profile_export_json(active_profile_id)
        profile_export_file_name = _manual_well_calc_profile_export_file_name(
            active_profile_id
        )
        profile_action_cols = st.columns(2, gap="small")
        create_profile_clicked = profile_action_cols[0].button(
            "Новая",
            icon=":material/add:",
            width="stretch",
            disabled=not overrides_enabled,
        )
        delete_profile_clicked = profile_action_cols[1].button(
            "Удалить",
            icon=":material/delete:",
            width="stretch",
            disabled=not overrides_enabled or not active_profile_id,
        )
        if imported_dev_target_names:
            st.caption(
                "Будут созданы или обновлены отдельные конфигурации для всех "
                "скважин текущего набора, загруженных через `.dev`."
            )
        uploaded_profile_files = st.file_uploader(
            "JSON конфигураций",
            type=["json"],
            accept_multiple_files=True,
            key=WT_WELL_CALC_PROFILE_IMPORT_UPLOAD_KEY,
            disabled=not overrides_enabled,
        )
        profile_io_cols = st.columns(2, gap="small")
        import_profile_clicked = profile_io_cols[0].button(
            "Импорт",
            icon=":material/upload_file:",
            width="stretch",
            disabled=not overrides_enabled or not uploaded_profile_files,
        )
        profile_io_cols[1].download_button(
            "Экспорт",
            data=profile_export_json.encode("utf-8"),
            file_name=profile_export_file_name,
            mime="application/json",
            icon=":material/download:",
            width="stretch",
            disabled=not overrides_enabled
            or not active_profile_id
            or not profile_export_json,
        )
        selected_names = _coerce_manual_well_override_selection(
            st.multiselect(
                "Скважины для назначения",
                options=available_names,
                key=WT_WELL_CALC_OVERRIDE_SELECTION_KEY,
                disabled=not overrides_enabled,
            ),
            available_names=available_names,
        )
        assignment_cols = st.columns(3, gap="small")
        select_all_assignments_clicked = assignment_cols[0].button(
            "Выбрать все",
            icon=":material/done_all:",
            width="stretch",
            disabled=not overrides_enabled or not available_names,
        )
        assign_profile_clicked = assignment_cols[1].button(
            "Назначить выбранным",
            icon=":material/link:",
            width="stretch",
            disabled=not overrides_enabled
            or not active_profile_id
            or not selected_names,
        )
        clear_assignment_clicked = assignment_cols[2].button(
            "Снять назначение",
            icon=":material/link_off:",
            width="stretch",
            disabled=not overrides_enabled or not selected_names,
        )
    with editor_col:
        if (
            active_profile_id
            and kop_min_vertical_function_from_state(prefix=WT_CALC_PARAMS.prefix)
            is not None
            and "kop_min_vertical"
            not in _normalized_calc_param_override_values(
                active_payload.get("values", {})
            )
        ):
            st.caption(
                "KOP остаётся общим по зависимости от TVD. Локальным он "
                "станет только после сохранения изменённого KOP в этой "
                "конфигурации."
            )
        _build_config_form(
            binding=WT_WELL_OVERRIDE_EDITOR,
            title="Параметры конфигурации",
            disabled=not overrides_enabled or not active_profile_id,
        )
        editor_action_cols = st.columns([1.0, 1.2, 1.9, 1.6], gap="small")
        load_global_clicked = editor_action_cols[0].button(
            "Вернуть дефолт",
            icon=":material/content_copy:",
            width="stretch",
            disabled=not overrides_enabled or not active_profile_id,
        )
        save_profile_clicked = editor_action_cols[1].button(
            "Сохранить конфигурацию",
            type="primary",
            icon=":material/save:",
            width="stretch",
            disabled=not overrides_enabled or not active_profile_id,
        )
        save_all_profiles_clicked = editor_action_cols[2].button(
            "Применить для всех и сохранить",
            icon=":material/publish:",
            width="stretch",
            disabled=not overrides_enabled or not active_profile_id,
        )
        if not active_profile_id:
            st.caption("Создайте конфигурацию или подтяните параметры из `.dev`.")
    if create_profile_clicked:
        created_profile_id = _create_manual_well_calc_profile()
        created_payload = dict(_manual_well_calc_profiles().get(created_profile_id, {}))
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = ()
        st.session_state[
            _manual_well_calc_profile_name_key(created_profile_id)
        ] = str(created_payload.get("name", created_profile_id))
        st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
            f'Создана конфигурация "{created_payload.get("name", created_profile_id)}".'
        )
        _rerun_fragment()
    if delete_profile_clicked:
        affected_well_names = _manual_well_names_for_profile_ids(
            profile_ids=[active_profile_id],
            available_names=available_names,
        )
        deleted_count, cleared_assignments = _delete_manual_well_calc_profile(
            active_profile_id
        )
        if deleted_count > 0:
            _queue_batch_selection_additions(
                well_names=affected_well_names,
                available_names=available_names,
            )
            st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = ()
            st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
                "Конфигурация удалена. "
                f"Скважин переведено на общие параметры: {cleared_assignments}."
            )
            _rerun_fragment()
    if apply_dev_clicked:
        applied_count, missing_names = _apply_dev_params_to_manual_well_overrides(
            well_names=imported_dev_target_names,
        )
        if applied_count > 0:
            _queue_manual_well_calc_override_enabled(True)
            _queue_batch_selection_additions(
                well_names=imported_dev_target_names,
                available_names=available_names,
            )
        feedback = f"Конфигурации из .dev созданы или обновлены: {applied_count}."
        if missing_names:
            feedback += " Без .dev параметров: " + ", ".join(missing_names) + "."
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = ()
        st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = feedback
        _rerun_fragment()
    if import_profile_clicked:
        if not uploaded_profile_files:
            st.warning("Загрузите один или несколько JSON файлов конфигураций.")
        else:
            (
                imported_count,
                updated_count,
                error_messages,
                last_profile_id,
                touched_profile_ids,
            ) = (
                _import_manual_well_calc_profile_json_payloads(
                    (
                        (str(item.name or "configuration.json"), item.getvalue())
                        for item in uploaded_profile_files
                    )
                )
            )
            if imported_count > 0:
                _queue_manual_well_calc_override_enabled(True)
                _queue_batch_selection_additions(
                    well_names=_manual_well_names_for_profile_ids(
                        profile_ids=touched_profile_ids,
                        available_names=available_names,
                    ),
                    available_names=available_names,
                )
                st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = ()
                created_count = int(imported_count - updated_count)
                feedback = (
                    "Импортировано конфигураций: "
                    f"{imported_count} (новых: {created_count}, обновлено: {updated_count})."
                )
                if error_messages:
                    feedback += " Ошибки: " + " | ".join(error_messages)
                st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = feedback
                _rerun_fragment()
            if error_messages:
                st.warning(
                    "Не удалось импортировать конфигурации: "
                    + " | ".join(error_messages)
                )
    if select_all_assignments_clicked:
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_PENDING_KEY] = list(
            available_names
        )
        _rerun_fragment()
    if assign_profile_clicked:
        assigned_count = _assign_manual_well_calc_profile_to_wells(
            profile_id=active_profile_id,
            well_names=selected_names,
        )
        if assigned_count > 0:
            _queue_batch_selection_additions(
                well_names=selected_names,
                available_names=available_names,
            )
        profile_label = str(active_payload.get("name", active_profile_id)).strip() or str(
            active_profile_id
        )
        st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
            f'Конфигурация "{profile_label}" назначена скважинам: {assigned_count}.'
        )
        _rerun_fragment()
    if clear_assignment_clicked:
        cleared_count = _clear_manual_well_profile_assignments(well_names=selected_names)
        if cleared_count > 0:
            _queue_batch_selection_additions(
                well_names=selected_names,
                available_names=available_names,
            )
        st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
            f"Назначения сняты для скважин: {cleared_count}."
        )
        _rerun_fragment()
    if load_global_clicked:
        base_values = calc_param_state_values_from_config(base_config)
        _set_manual_well_override_editor_from_config(base_config)
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = (
            _manual_well_override_selection_signature(
                active_profile_id=active_profile_id,
                values=base_values,
                source="base",
            )
        )
        _rerun_fragment()
    if save_profile_clicked:
        changed, resolved_name, changed_field_count = _apply_manual_well_override_editor(
            base_config=base_config,
            active_profile_id=active_profile_id,
            profile_name=str(
                st.session_state.get(
                    _manual_well_calc_profile_name_key(active_profile_id),
                    active_payload.get("name", active_profile_id),
                )
            ),
        )
        if changed:
            _queue_batch_selection_additions(
                well_names=_manual_well_names_for_profile_ids(
                    profile_ids=[active_profile_id],
                    available_names=available_names,
                ),
                available_names=available_names,
            )
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = ()
        if changed:
            st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
                f'Конфигурация "{resolved_name}" сохранена. '
                f"Изменённых полей: {changed_field_count}."
            )
        else:
            st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
                f'Конфигурация "{resolved_name}" уже совпадает с текущими значениями.'
            )
        _rerun_fragment()
    if save_all_profiles_clicked:
        changed, resolved_name, changed_field_count, updated_profile_count = (
            _apply_manual_well_override_editor_to_all_profiles(
                base_config=base_config,
                active_profile_id=active_profile_id,
                profile_name=str(
                    st.session_state.get(
                        _manual_well_calc_profile_name_key(active_profile_id),
                        active_payload.get("name", active_profile_id),
                    )
                ),
                records_by_name=records_by_name,
            )
        )
        if changed:
            affected_profile_ids = (
                _manual_well_calc_profile_option_ids()
                if changed_field_count > 0
                else [active_profile_id]
            )
            _queue_batch_selection_additions(
                well_names=_manual_well_names_for_profile_ids(
                    profile_ids=affected_profile_ids,
                    available_names=available_names,
                ),
                available_names=available_names,
            )
        st.session_state[WT_WELL_CALC_OVERRIDE_SELECTION_SIGNATURE_KEY] = ()
        if changed_field_count > 0:
            st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
                f'Конфигурация "{resolved_name}" сохранена и применена ко всем '
                f"конфигурациям. Изменённых полей: {changed_field_count}. "
                f"Обновлено конфигураций: {updated_profile_count}."
            )
        elif changed:
            st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
                f'Конфигурация "{resolved_name}" сохранена. '
                "Изменённых полей для применения ко всем конфигурациям нет."
            )
        else:
            st.session_state[WT_WELL_CALC_OVERRIDE_FEEDBACK_KEY] = (
                f'Конфигурация "{resolved_name}" уже совпадает с текущими значениями.'
            )
        _rerun_fragment()
    current_rows = _manual_well_override_rows(available_names=available_names)
    if current_rows:
        active_items = _manual_well_calc_override_items()
        if not _manual_well_calc_override_enabled() and active_items:
            st.caption(
                "Индивидуальные конфигурации сохранены, но сейчас отключены и "
                "в расчёте не участвуют."
            )
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(current_rows)),
            hide_index=True,
            width="stretch",
        )


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
        return "весь связанный кластер конфликтов"
    if kind == "recommendation":
        return "одно событие пересечения"
    return "локальный план пересчёта"


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


def _format_selected_calc_config_scope(
    *,
    selected_names: Iterable[str],
) -> list[dict[str, object]]:
    ordered_names = list(
        dict.fromkeys(
            str(name).strip() for name in selected_names if str(name).strip()
        )
    )
    if not ordered_names:
        return []
    manual_profiles = (
        _manual_well_calc_profiles() if _manual_well_calc_override_enabled() else {}
    )
    manual_assignments = (
        _manual_well_calc_profile_assignments()
        if _manual_well_calc_override_enabled()
        else {}
    )
    prepared_scope_by_well = {
        str(row.get("Скважина", "")).strip(): row
        for row in _format_prepared_override_scope(selected_names=ordered_names)
        if str(row.get("Скважина", "")).strip()
    }
    rows: list[dict[str, object]] = []
    for well_name in ordered_names:
        profile_id = str(manual_assignments.get(well_name, "")).strip()
        profile_payload = dict(manual_profiles.get(profile_id, {}))
        config_name = (
            str(profile_payload.get("name", "")).strip()
            if profile_id
            else "Общие параметры"
        ) or "Общие параметры"
        prepared_row = dict(prepared_scope_by_well.get(well_name, {}))
        rows.append(
            {
                "Скважина": well_name,
                "Конфигурация": config_name,
                "Источник": (
                    str(profile_payload.get("source", "")).strip()
                    if profile_id
                    else "Общие параметры"
                )
                or "Общие параметры",
                "Локальный режим": str(
                    prepared_row.get("Локальный режим", "Общий режим")
                ).strip()
                or "Общий режим",
                "Маневр": str(prepared_row.get("Маневр", "—")).strip() or "—",
            }
        )
    return rows


def _welltrack_record_entry_tvd_m(record: WelltrackRecord) -> float | None:
    try:
        layout = ordinary_record_target_layout(record)
    except ValueError:
        return None
    return float(layout.t1.z)


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
    manual_profiles = (
        _manual_well_calc_profiles() if _manual_well_calc_override_enabled() else {}
    )
    manual_assignments = (
        _manual_well_calc_profile_assignments()
        if _manual_well_calc_override_enabled()
        else {}
    )
    for well_name in sorted(str(name) for name in selected_names):
        profile_id = str(manual_assignments.get(well_name, "")).strip()
        manual_payload = dict(manual_profiles.get(profile_id, {}))
        manual_values = _normalized_calc_param_override_values(
            manual_payload.get("values", {})
        )
        payload = dict(prepared.get(well_name, {}))
        update_fields = dict(payload.get("update_fields", {}))
        if manual_values:
            effective_values = calc_param_state_values_from_config(base_config)
            effective_values.update(manual_values)
            config = build_config_from_values(effective_values)
        else:
            config = base_config
        if update_fields:
            config = config.validated_copy(**update_fields)
        explicit_kop_override = (
            "kop_min_vertical" in manual_values
            or "kop_min_vertical_m" in update_fields
        )
        if kop_function is not None and records_by_name is not None:
            record = records_by_name.get(well_name)
            if record is not None and not explicit_kop_override:
                evaluated_kop_m = _evaluated_kop_min_vertical_for_record(
                    record=record,
                    base_config=config,
                    kop_function=kop_function,
                )
                if evaluated_kop_m is not None:
                    config = config.validated_copy(kop_min_vertical_m=evaluated_kop_m)
        if config == base_config and not update_fields and not manual_values:
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
    st.markdown("### Результат пересчёта anti-collision")
    if resolution_kind == "cluster":
        caption = (
            "После пересчёта траекторий пересчитан SF по актуальным "
            "конфликтам затронутых скважин."
        )
    else:
        caption = (
            "Сравнение выполнено на исходном конфликтном интервале "
            "с текущим пресетом неопределённости."
        )
    used_preset = str(resolution.get("uncertainty_preset", "")).strip()
    if used_preset and used_preset != str(current_preset):
        caption += (
            f" Последний пересчёт считался на пресете "
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
                "Актуальные кластеры после пересчёта: "
                + "; ".join(str(label) for label in current_cluster_labels)
            )
        else:
            st.caption("Для исходных скважин активных кластеров не осталось.")
        items = list(resolution.get("items", ()) or ())
        if not items:
            st.success(
                "Оставшихся пересечений по затронутым скважинам не найдено."
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
                            "Пересечение сейчас, м": _format_overlap_value(
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
                        "Пересечение до, м": _format_overlap_value(
                            resolution.get("before_overlap_m")
                        ),
                        "Пересечение после, м": _format_overlap_value(
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
        "Не удалось подготовить пересчёт по выбранной рекомендации: "
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
            else "Для этого пересчёта по кластеру доступны только справочные рекомендации."
        )
        st.session_state["wt_prepared_well_overrides"] = {}
        st.session_state["wt_prepared_override_message"] = (
            "Пересчёт по кластеру недоступен: " + blocking_message
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
                " Фокус пересчёта: "
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
                " Область пересчёта по кластеру: "
                + ", ".join(str(name) for name in target_well_names)
                + "."
            )
            if expanded_scope:
                message += (
                    " Соседние скважины других кустов подключатся автоматически "
                    "к тому же кластеру конфликтов: "
                    + ", ".join(expanded_scope)
                    + "."
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
        "Не удалось подготовить пересчёт по кластеру: контекст конфликта недоступен."
    )
    st.session_state["wt_prepared_recommendation_id"] = ""
    st.session_state["wt_prepared_recommendation_snapshot"] = None
    st.session_state["wt_pending_selected_names"] = None


def _build_source_payload_from_state() -> _WelltrackSourcePayload:
    source_format = ptc_target_import.normalized_target_source_format(
        st.session_state
    )
    source_mode = str(st.session_state.get("wt_source_mode", "")).strip()
    fixed_t1_enabled = bool(
        st.session_state.get("wt_source_dev_fixed_t1_enabled", False)
    )
    fixed_t1_inc_by_well = (
        _dev_fixed_t1_inc_by_well_from_state(
            selected_names=tuple(
                str(name).strip()
                for name in st.session_state.get("wt_source_dev_fixed_t1_well_names", [])
                if str(name).strip()
            )
        )
        if fixed_t1_enabled
        else ()
    )

    if source_format == WT_SOURCE_FORMAT_TARGET_TABLE:
        return _WelltrackSourcePayload(
            mode=WT_SOURCE_MODE_TARGET_TABLE,
            source_format=source_format,
            table_rows=pd.DataFrame(
                st.session_state.get("wt_source_table_df", _empty_source_table_df())
            ),
        )

    if source_format == WT_SOURCE_FORMAT_DEV_TRAJECTORY:
        if source_mode == WT_SOURCE_MODE_FILE_PATH:
            return _WelltrackSourcePayload(
                mode=source_mode,
                source_format=source_format,
                source_path=str(st.session_state.get("wt_source_path", "")).strip(),
                dev_fixed_t1_inc_by_well=fixed_t1_inc_by_well,
            )

        if source_mode == WT_SOURCE_MODE_UPLOAD:
            raw_files = st.session_state.get("wt_source_dev_upload_files") or []
            uploaded_files = raw_files if isinstance(raw_files, list) else [raw_files]
            payloads = tuple(
                (
                    str(getattr(item, "name", f"dev_{index + 1}.dev")),
                    bytes(item.getvalue()),
                )
                for index, item in enumerate(uploaded_files)
                if item is not None
            )
            return _WelltrackSourcePayload(
                mode=source_mode,
                source_format=source_format,
                source_files=payloads,
                dev_fixed_t1_inc_by_well=fixed_t1_inc_by_well,
            )

        if source_mode == WT_SOURCE_MODE_INLINE_TEXT:
            return _WelltrackSourcePayload(
                mode=source_mode,
                source_format=source_format,
                source_text=str(st.session_state.get("wt_source_dev_inline", "")),
                dev_fixed_t1_inc_by_well=fixed_t1_inc_by_well,
            )

        return _WelltrackSourcePayload(
            mode=source_mode or WT_SOURCE_MODE_FILE_PATH,
            source_format=source_format,
            dev_fixed_t1_inc_by_well=fixed_t1_inc_by_well,
        )

    if source_mode == WT_SOURCE_MODE_FILE_PATH:
        source_path = str(st.session_state.get("wt_source_path", "")).strip()
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_format=source_format,
            source_text=_read_welltrack_file(source_path),
        )

    if source_mode == WT_SOURCE_MODE_UPLOAD:
        uploaded_file = st.session_state.get("wt_source_upload_file")
        if uploaded_file is None:
            return _WelltrackSourcePayload(
                mode=source_mode,
                source_format=source_format,
            )
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_format=source_format,
            source_text=_decode_welltrack_payload(
                uploaded_file.getvalue(),
                source_label=f"Загруженный файл `{uploaded_file.name}`",
            ),
        )

    if source_mode == WT_SOURCE_MODE_INLINE_TEXT:
        return _WelltrackSourcePayload(
            mode=source_mode,
            source_format=source_format,
            source_text=str(st.session_state.get("wt_source_inline", "")),
        )

    return _WelltrackSourcePayload(
        mode=source_mode or WT_SOURCE_MODE_FILE_PATH,
        source_format=source_format or WT_SOURCE_FORMAT_TARGET_TABLE,
    )


def _dev_fixed_t1_input_key(well_name: str) -> str:
    normalized_name = str(well_name).strip()
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", normalized_name).strip("_") or "well"
    digest = hashlib.sha1(normalized_name.encode("utf-8")).hexdigest()[:12]
    return f"wt_source_dev_fixed_t1_inc__{slug}__{digest}"


def _dev_fixed_t1_default_inc_deg() -> float:
    return 86.0


def _dev_fixed_t1_common_inc_deg() -> float:
    raw_value = st.session_state.get(
        "wt_source_dev_fixed_t1_common_inc_deg",
        _dev_fixed_t1_default_inc_deg(),
    )
    if isinstance(raw_value, (int, float)) and np.isfinite(float(raw_value)):
        return float(min(max(float(raw_value), 0.5), 89.0))
    return _dev_fixed_t1_default_inc_deg()


def _dev_fixed_t1_inc_by_well_from_state(
    *,
    selected_names: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    if bool(st.session_state.get("wt_source_dev_fixed_t1_common_enabled", False)):
        common_threshold = _dev_fixed_t1_common_inc_deg()
        return tuple(
            (str(well_name).strip(), float(common_threshold))
            for well_name in selected_names
            if str(well_name).strip()
        )
    default_inc_deg = _dev_fixed_t1_default_inc_deg()
    values: list[tuple[str, float]] = []
    for well_name in selected_names:
        normalized_name = str(well_name).strip()
        if not normalized_name:
            continue
        raw_value = st.session_state.get(
            _dev_fixed_t1_input_key(normalized_name),
            default_inc_deg,
        )
        threshold = (
            float(raw_value)
            if isinstance(raw_value, (int, float)) and np.isfinite(float(raw_value))
            else default_inc_deg
        )
        values.append(
            (
                normalized_name,
                float(min(max(threshold, 0.5), 89.0)),
            )
        )
    return tuple(values)


def _preview_dev_source_well_names_from_state() -> tuple[str, ...]:
    source_mode = str(st.session_state.get("wt_source_mode", "")).strip()
    source_path = str(st.session_state.get("wt_source_path", "")).strip()
    source_text = str(st.session_state.get("wt_source_dev_inline", ""))
    raw_files = st.session_state.get("wt_source_dev_upload_files") or []
    uploaded_files = raw_files if isinstance(raw_files, list) else [raw_files]
    payloads = tuple(
        (
            str(getattr(item, "name", f"dev_{index + 1}.dev")),
            bytes(item.getvalue()),
        )
        for index, item in enumerate(uploaded_files)
        if item is not None
    )
    return ptc_target_import.dev_source_preview_well_names(
        source_mode=source_mode,
        source_path=source_path,
        source_files=payloads,
        source_text=source_text,
    )


def _sync_dev_fixed_t1_selection_state(
    available_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_names = tuple(
        str(name).strip() for name in available_names if str(name).strip()
    )
    available_set = set(normalized_names)
    pending_selection = st.session_state.pop(
        "wt_source_dev_fixed_t1_pending_well_names",
        None,
    )
    if pending_selection is not None:
        st.session_state["wt_source_dev_fixed_t1_well_names"] = [
            str(name).strip()
            for name in pending_selection
            if str(name).strip() in available_set
        ]
    current_selection = [
        str(name).strip()
        for name in st.session_state.get("wt_source_dev_fixed_t1_well_names", [])
        if str(name).strip()
    ]
    filtered_selection = [
        name for name in current_selection if name in available_set
    ]
    if filtered_selection != current_selection:
        st.session_state["wt_source_dev_fixed_t1_well_names"] = filtered_selection
    default_inc_deg = _dev_fixed_t1_default_inc_deg()
    for well_name in filtered_selection:
        input_key = _dev_fixed_t1_input_key(well_name)
        raw_value = st.session_state.get(input_key, default_inc_deg)
        raw_value_is_number = isinstance(raw_value, (int, float)) and np.isfinite(
            float(raw_value)
        )
        normalized_value = (
            float(raw_value) if raw_value_is_number else default_inc_deg
        )
        if (input_key not in st.session_state) or (
            raw_value_is_number and abs(float(raw_value) - normalized_value) > 1e-9
        ) or (not raw_value_is_number):
            st.session_state[input_key] = float(min(max(normalized_value, 0.5), 89.0))
    return normalized_names, tuple(filtered_selection)


def _render_dev_fixed_t1_controls(*, available_names: Sequence[str]) -> None:
    normalized_names, selected_names = _sync_dev_fixed_t1_selection_state(
        available_names
    )
    enabled = st.toggle(
        "Определять `t1` по фиксированному INC (зенитному углу входа в пласт)",
        key="wt_source_dev_fixed_t1_enabled",
    )
    if not enabled:
        return
    selection_cols = st.columns([3.2, 1.0], gap="small", vertical_alignment="bottom")
    multiselect_col = selection_cols[0] if len(selection_cols) > 0 else st
    select_all_col = selection_cols[1] if len(selection_cols) > 1 else st
    getattr(multiselect_col, "multiselect", st.multiselect)(
        "Скважины для режима t1 по INC",
        options=list(normalized_names),
        key="wt_source_dev_fixed_t1_well_names",
        disabled=not normalized_names,
    )
    select_all_clicked = getattr(select_all_col, "button", st.button)(
        "Выбрать все",
        key="wt_source_dev_fixed_t1_select_all",
        icon=":material/done_all:",
        width="stretch",
        disabled=not normalized_names,
    )
    if select_all_clicked:
        st.session_state["wt_source_dev_fixed_t1_pending_well_names"] = list(
            normalized_names
        )
        _rerun_fragment()
    if not normalized_names:
        st.caption("Список скважин появится после выбора источника `.dev`.")
        return
    selected_names = tuple(
        str(name).strip()
        for name in st.session_state.get("wt_source_dev_fixed_t1_well_names", [])
        if str(name).strip()
    )
    if not selected_names:
        st.caption(
            "Выберите скважины, для которых `t1` нужно определять по фиксированному INC."
        )
        return
    common_enabled = st.toggle(
        "Общее значение для выбранных скважин",
        key="wt_source_dev_fixed_t1_common_enabled",
        disabled=not selected_names,
    )
    if common_enabled:
        st.caption("Один порог `INC` будет применён ко всем выбранным скважинам.")
        st.number_input(
            "Общий INC для t1, deg",
            min_value=0.5,
            max_value=89.0,
            step=0.5,
            key="wt_source_dev_fixed_t1_common_inc_deg",
        )
    else:
        header_cols = st.columns([3.2, 1.1], gap="small")
        threshold_col = header_cols[1] if len(header_cols) > 1 else st
        st.caption("Для каждой выбранной скважины задайте свой порог `INC`.")
        getattr(threshold_col, "caption", st.caption)("INC для t1, deg")
    default_inc_deg = _dev_fixed_t1_default_inc_deg()
    if common_enabled:
        st.caption(
            "Индивидуальные значения сохраняются, но пока используется общий порог."
        )
    else:
        for well_name in selected_names:
            input_key = _dev_fixed_t1_input_key(well_name)
            if input_key not in st.session_state:
                st.session_state[input_key] = default_inc_deg
            row_cols = st.columns([3.2, 1.1], gap="small")
            name_col = row_cols[0] if len(row_cols) > 0 else st
            input_col = row_cols[1] if len(row_cols) > 1 else st
            getattr(name_col, "markdown", st.markdown)(f"`{well_name}`")
            getattr(input_col, "number_input", st.number_input)(
                f"INC для t1: {well_name}",
                min_value=0.5,
                max_value=89.0,
                step=0.5,
                key=input_key,
                label_visibility="collapsed",
            )
    st.caption(
        "Для выбранных скважин `t1` берётся как первая точка участка `BUILD2` "
        "с `INC` не ниже заданного значения. Дальше остаток траектории "
        "читается как участок после `t1`."
    )


def _render_target_table_help() -> None:
    st.html(
        """
        <style>
        .pywp-target-help-table-wrap {
            display: inline-block;
            max-width: 100%;
        }
        .pywp-target-help-table {
            width: auto;
            border-collapse: collapse;
            table-layout: auto;
            font-size: 0.88rem;
            line-height: 1.25;
        }
        .pywp-target-help-table th,
        .pywp-target-help-table td {
            padding: 0.16rem 0.5rem;
            border-bottom: 1px solid #E3EAF2;
            white-space: nowrap;
            text-align: left;
        }
        .pywp-target-help-table th {
            color: #5F6F80;
            font-size: 0.79rem;
            font-weight: 600;
        }
        .pywp-target-help-table tr:last-child td {
            border-bottom: none;
        }
        .pywp-target-help-table .pywp-target-help-coord {
            color: #7A8794;
            font-size: 0.76rem;
        }
        </style>
        """,
        width="content",
    )

    def _render_example(
        title: str,
        rows: list[dict[str, str]],
        description: str,
    ) -> None:
        columns = ("Wellname", "Point", "X", "Y", "Z")
        header_html = "".join(
            f"<th class=\"{'pywp-target-help-coord' if column in {'X', 'Y', 'Z'} else ''}\">{escape(column)}</th>"
            for column in columns
        )
        rows_html = "".join(
            "<tr>"
            + "".join(
                f"<td class=\"{'pywp-target-help-coord' if column in {'X', 'Y', 'Z'} else ''}\">{escape(str(row.get(column, '')))}</td>"
                for column in columns
            )
            + "</tr>"
            for row in rows
        )
        st.markdown(f"#### {title}")
        st.html(
            (
                "<div class='pywp-target-help-table-wrap'>"
                "<table class='pywp-target-help-table'>"
                f"<thead><tr>{header_html}</tr></thead>"
                f"<tbody>{rows_html}</tbody>"
                "</table>"
                "</div>"
            ),
            width="content",
        )
        st.caption(description)

    st.markdown("Ниже показаны типовые примеры для различных типов скважин.")

    _render_example(
        "Обычная скважина",
        [
            {"Wellname": "well_01", "Point": "S", "X": "Xs", "Y": "Ys", "Z": "Zs"},
            {"Wellname": "well_01", "Point": "t1", "X": "X1", "Y": "Y1", "Z": "Z1"},
            {"Wellname": "well_01", "Point": "t3", "X": "X3", "Y": "Y3", "Z": "Z3"},
        ],
        "Базовый вариант: одна скважина, точка `S`, вход в пласт `t1` и конечная цель `t3`.",
    )
    _render_example(
        "Скважина с последовательностью целей",
        [
            {"Wellname": "well_01", "Point": "S", "X": "Xs", "Y": "Ys", "Z": "Zs"},
            {"Wellname": "well_01", "Point": "t1", "X": "X1", "Y": "Y1", "Z": "Z1"},
            {"Wellname": "well_01", "Point": "t2", "X": "X2", "Y": "Y2", "Z": "Z2"},
            {"Wellname": "well_01", "Point": "t3", "X": "X3", "Y": "Y3", "Z": "Z3"},
            {"Wellname": "well_01", "Point": "t4", "X": "X4", "Y": "Y4", "Z": "Z4"},
        ],
        "Если одна скважина должна последовательно пройти через несколько целей, используйте одно и то же имя и задавайте точки подряд: `t1`, `t2`, `t3`, ... без пропусков.",
    )
    _render_example(
        "Пилот",
        [
            {
                "Wellname": "well_01_PL",
                "Point": "S",
                "X": "Xs",
                "Y": "Ys",
                "Z": "Zs",
            },
            {
                "Wellname": "well_01_PL",
                "Point": "PL1",
                "X": "Xpl1",
                "Y": "Ypl1",
                "Z": "Zpl1",
            },
            {
                "Wellname": "well_01_PL",
                "Point": "PL2",
                "X": "Xpl2",
                "Y": "Ypl2",
                "Z": "Zpl2",
            },
        ],
        "Пилот задаётся отдельной записью с тем же базовым именем и суффиксом `_PL`. Например, пилот `well_01_PL` будет связан с основной скважиной `well_01`.",
    )
    _render_example(
        "ЗБС от фактической скважины",
        [
            {
                "Wellname": "fact_01_ZBS",
                "Point": "t1",
                "X": "X1",
                "Y": "Y1",
                "Z": "Z1",
            },
            {
                "Wellname": "fact_01_ZBS",
                "Point": "t3",
                "X": "X3",
                "Y": "Y3",
                "Z": "Z3",
            },
        ],
        "Можно также использовать имя `fact_01_2`. Точку `S` для такого ЗБС указывать не нужно: система берёт старт из уже загруженной фактической скважины. Имя `fact_well` должно совпадать с именем загруженной фактической скважины.",
    )
    _render_example(
        "Ствол от пилота",
        [
            {
                "Wellname": "well_01_2",
                "Point": "S",
                "X": "Xs",
                "Y": "Ys",
                "Z": "Zs",
            },
            {
                "Wellname": "well_01_2",
                "Point": "t1",
                "X": "X1",
                "Y": "Y1",
                "Z": "Z1",
            },
            {
                "Wellname": "well_01_2",
                "Point": "t3",
                "X": "X3",
                "Y": "Y3",
                "Z": "Z3",
            },
        ],
        "Если проектный ствол назван `well_01_2`, система свяжет его с пилотом `well_01_PL`. В этом случае точка `S` нужна, потому что это отдельная проектная траектория. Имя `well` должно совпадать с основной скважиной.",
    )
    _render_example(
        "Многопластовая скважина",
        [
            {"Wellname": "well_02", "Point": "S", "X": "Xs", "Y": "Ys", "Z": "Zs"},
            {
                "Wellname": "well_02",
                "Point": "1_t1",
                "X": "X1_1",
                "Y": "Y1_1",
                "Z": "Z1_1",
            },
            {
                "Wellname": "well_02",
                "Point": "1_t3",
                "X": "X1_3",
                "Y": "Y1_3",
                "Z": "Z1_3",
            },
            {
                "Wellname": "well_02",
                "Point": "2_t1",
                "X": "X2_1",
                "Y": "Y2_1",
                "Z": "Z2_1",
            },
            {
                "Wellname": "well_02",
                "Point": "2_t3",
                "X": "X2_3",
                "Y": "Y2_3",
                "Z": "Z2_3",
            },
        ],
        "Многопластовая скважина задаётся следующей последовательностью точек: `S`, `1_t1`, `1_t3`, `2_t1`, `2_t3`, ... . Каждая пара `N_t1` / `N_t3` описывает отдельный участок ГС.",
    )


def _render_source_input() -> None:
    if str(st.session_state.get("wt_source_format", "")).strip() not in set(
        WT_SOURCE_FORMAT_OPTIONS
    ):
        st.session_state["wt_source_format"] = ptc_target_import.normalized_target_source_format(
            st.session_state
        )

    format_label_col, format_radio_col = st.columns(
        [0.9, 4.7], gap="small", vertical_alignment="center"
    )
    with format_label_col:
        st.markdown("**Выберите формат импорта:**")
    with format_radio_col:
        source_format = st.radio(
            "Выберите формат импорта:",
            options=list(WT_SOURCE_FORMAT_OPTIONS),
            horizontal=True,
            key="wt_source_format",
            label_visibility="collapsed",
        )
    current_source_path = str(st.session_state.get("wt_source_path", "")).strip()
    if (
        source_format == WT_SOURCE_FORMAT_DEV_TRAJECTORY
        and current_source_path == str(DEFAULT_WELLTRACK_PATH)
    ):
        st.session_state["wt_source_path"] = str(DEFAULT_DEV_TRAJECTORY_PATH)
    elif (
        source_format == WT_SOURCE_FORMAT_WELLTRACK
        and current_source_path == str(DEFAULT_DEV_TRAJECTORY_PATH)
    ):
        st.session_state["wt_source_path"] = str(DEFAULT_WELLTRACK_PATH)

    if source_format in {WT_SOURCE_FORMAT_WELLTRACK, WT_SOURCE_FORMAT_DEV_TRAJECTORY}:
        if str(st.session_state.get("wt_source_mode", "")).strip() not in set(
            WT_SOURCE_WELLTRACK_MODES
        ):
            st.session_state["wt_source_mode"] = WT_SOURCE_MODE_FILE_PATH
        mode_title = (
            "Способ загрузки WELLTRACK"
            if source_format == WT_SOURCE_FORMAT_WELLTRACK
            else "Способ загрузки .dev"
        )
        source_mode = st.radio(
            mode_title,
            options=list(WT_SOURCE_WELLTRACK_MODES),
            horizontal=True,
            key="wt_source_mode",
        )
    else:
        source_mode = WT_SOURCE_MODE_TARGET_TABLE
        st.session_state["wt_source_mode"] = WT_SOURCE_MODE_TARGET_TABLE

    if source_mode == WT_SOURCE_MODE_FILE_PATH:
        if source_format == WT_SOURCE_FORMAT_DEV_TRAJECTORY:
            st.text_input(
                "Путь к .dev файлу или папке",
                key="wt_source_path",
                placeholder="tests/test_data/dev_target_import",
            )
            _render_dev_fixed_t1_controls(
                available_names=_preview_dev_source_well_names_from_state()
            )
            if st.button(
                "Импорт целей",
                type="primary",
                icon=":material/upload_file:",
                width="content",
            ):
                st.session_state["wt_source_parse_clicked"] = True
            return
        with st.form("wt_source_path_form", clear_on_submit=False):
            st.text_input(
                (
                    "Путь к файлу WELLTRACK"
                    if source_format == WT_SOURCE_FORMAT_WELLTRACK
                    else "Путь к .dev файлу или папке"
                ),
                key="wt_source_path",
                placeholder=(
                    "tests/test_data/WELLTRACKS3.INC"
                    if source_format == WT_SOURCE_FORMAT_WELLTRACK
                    else "tests/test_data/dev_target_import"
                ),
            )
            parse_clicked = st.form_submit_button(
                "Импорт целей",
                type="primary",
                icon=":material/upload_file:",
                width="content",
            )
        if parse_clicked:
            st.session_state["wt_source_parse_clicked"] = True
        return

    if source_mode == WT_SOURCE_MODE_UPLOAD:
        if source_format == WT_SOURCE_FORMAT_DEV_TRAJECTORY:
            st.file_uploader(
                ".dev файлы",
                type=["dev", "txt"],
                accept_multiple_files=True,
                key="wt_source_dev_upload_files",
            )
            st.caption("Можно загрузить один или несколько `.dev` файлов.")
            _render_dev_fixed_t1_controls(
                available_names=_preview_dev_source_well_names_from_state()
            )
            if st.button(
                "Импорт целей",
                type="primary",
                icon=":material/upload_file:",
                width="content",
            ):
                st.session_state["wt_source_parse_clicked"] = True
            return
        form_key = (
            "wt_source_upload_form"
            if source_format == WT_SOURCE_FORMAT_WELLTRACK
            else "wt_source_dev_upload_form"
        )
        with st.form(form_key, clear_on_submit=False):
            if source_format == WT_SOURCE_FORMAT_WELLTRACK:
                st.file_uploader(
                    "Файл ECLIPSE/INC",
                    type=["inc", "txt", "data", "ecl"],
                    key="wt_source_upload_file",
                )
            else:
                st.file_uploader(
                    ".dev файлы",
                    type=["dev", "txt"],
                    accept_multiple_files=True,
                    key="wt_source_dev_upload_files",
                )
                st.caption(
                    "Можно загрузить один или несколько `.dev` файлов."
                )
            parse_clicked = st.form_submit_button(
                "Импорт целей",
                type="primary",
                icon=":material/upload_file:",
                width="content",
            )
        if parse_clicked:
            st.session_state["wt_source_parse_clicked"] = True
        return

    if source_mode == WT_SOURCE_MODE_INLINE_TEXT:
        if source_format == WT_SOURCE_FORMAT_DEV_TRAJECTORY:
            st.text_area(
                "Текст .dev",
                key="wt_source_dev_inline",
                height=220,
                placeholder=(
                    "# SURVEY FROM PYWP\n"
                    "# WELL NAME:                WELL-1\n"
                    "MD X Y Z TVD DX DY AZIM_TN INCL DLS AZIM_GN\n"
                ),
            )
            st.caption(
                "Если в тексте есть строка `# WELL NAME: ...`, имя скважины "
                "будет взято из неё."
            )
            _render_dev_fixed_t1_controls(
                available_names=_preview_dev_source_well_names_from_state()
            )
            if st.button(
                "Импорт целей",
                type="primary",
                icon=":material/upload_file:",
                width="content",
            ):
                st.session_state["wt_source_parse_clicked"] = True
            return
        form_key = (
            "wt_source_inline_form"
            if source_format == WT_SOURCE_FORMAT_WELLTRACK
            else "wt_source_dev_inline_form"
        )
        with st.form(form_key, clear_on_submit=False):
            if source_format == WT_SOURCE_FORMAT_WELLTRACK:
                st.text_area(
                    "Текст WELLTRACK",
                    key="wt_source_inline",
                    height=220,
                    placeholder="WELLTRACK 'WELL-1'\n457091 891257 -63.2 0\n457707 890374 1852 1\n/",
                )
            else:
                st.text_area(
                    "Текст .dev",
                    key="wt_source_dev_inline",
                    height=220,
                    placeholder=(
                        "# SURVEY FROM PYWP\n"
                        "# WELL NAME:                WELL-1\n"
                        "MD X Y Z TVD DX DY AZIM_TN INCL DLS AZIM_GN\n"
                    ),
                )
                st.caption(
                    "Если в тексте есть строка `# WELL NAME: ...`, имя скважины "
                    "будет взято из неё."
                )
            parse_clicked = st.form_submit_button(
                "Импорт целей",
                type="primary",
                icon=":material/upload_file:",
                width="content",
            )
        if parse_clicked:
            st.session_state["wt_source_parse_clicked"] = True
        return

    with st.expander("Таблица точек целей", expanded=True):
        st.caption(
            "Вставьте из Excel колонки `Wellname`, `Point`, `X`, `Y`, `Z`. "
            "В колонке `Point`: `S` - устье скважины; обычная скважина - `S`, `t1`, `t3` "
            "(две точки в пласте); несколько точек - `S`, `t1`, `t2`, `t3`, ... ."
        )
        with st.expander(
            "Как задавать скважины с пилотом, ЗБС и многопластовые скважины",
            expanded=False,
        ):
            _render_target_table_help()
        source_table_df = _normalize_source_table_df_for_ui(
            st.session_state.get("wt_source_table_df", _empty_source_table_df())
        )
        with st.form("wt_source_table_form", clear_on_submit=False):
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
            action_cols = st.columns([1.15, 3.3, 1.15], gap="small")
            parse_action_col = action_cols[0] if len(action_cols) > 0 else st
            clear_action_col = (
                action_cols[2]
                if len(action_cols) > 2
                else (action_cols[1] if len(action_cols) > 1 else st)
            )
            parse_submit_button = getattr(
                parse_action_col,
                "form_submit_button",
                st.form_submit_button,
            )
            parse_table_clicked = parse_submit_button(
                "Импорт целей",
                type="primary",
                icon=":material/upload_file:",
                width="stretch",
            )
            clear_submit_button = getattr(
                clear_action_col,
                "form_submit_button",
                st.form_submit_button,
            )
            clear_table_clicked = clear_submit_button(
                "Очистить",
                icon=":material/delete:",
                width="stretch",
            )
        if clear_table_clicked and not parse_table_clicked:
            st.session_state["wt_source_table_df"] = _empty_source_table_df()
            st.session_state["wt_source_table_editor_nonce"] = (
                int(st.session_state.get("wt_source_table_editor_nonce", 0)) + 1
            )
            _rerun_fragment()
        elif parse_table_clicked:
            st.session_state["wt_source_table_df"] = _normalize_source_table_df_for_ui(
                pd.DataFrame(edited_table)
            )
            st.session_state["wt_source_parse_clicked"] = True


def _store_parsed_records(records: list[WelltrackRecord]) -> bool:
    return _store_parsed_records_with_metadata(
        records=records,
        dev_summaries=[],
        source_kind="welltrack",
    )


def _store_parsed_records_with_metadata(
    *,
    records: list[WelltrackRecord],
    dev_summaries: list[ptc_target_import.DevTargetImportSummary],
    imported_dev_wells: list[ImportedTrajectoryWell] | None = None,
    import_failures: list[ptc_target_import.TargetImportFailure] | None = None,
    source_kind: str,
) -> bool:
    result = ptc_target_import.store_imported_records(
        st.session_state,
        records=list(records),
        dev_summaries=list(dev_summaries),
        imported_dev_wells=list(imported_dev_wells or []),
        failures=list(import_failures or []),
        source_kind=str(source_kind),
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
    plan_map = {
        str(pad_id): plan
        for pad_id, plan in _build_pad_plan_map(pads).items()
        if str(pad_id) in auto_layout_pad_ids
    }
    if not plan_map:
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
    st.session_state[ptc_pad_state.WT_PAD_LAYOUT_DETAILS_OPEN_KEY] = False
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
    resolved_pads = (
        pads if pads is not None else _ensure_pad_configs(base_records=records)
    )
    config_raw = st.session_state.get("wt_pad_configs", {})
    config_map = dict(config_raw if isinstance(config_raw, Mapping) else {})
    changed = False
    for pad in resolved_pads:
        pilot_parent_names = [
            str(well.name)
            for well in pad.wells
            if pilot_parent_key_for_record(well) in pilot_parent_keys
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
    source_payload: _WelltrackSourcePayload | None,
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
    if source_payload is None:
        source_payload = _build_source_payload_from_state()

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
            parse_result = operation.parse()
            records = list(parse_result.records)
            dev_summaries = list(parse_result.dev_summaries)
            imported_dev_wells = list(parse_result.imported_dev_wells)
            import_failures = list(parse_result.failures)
            auto_layout_applied = _store_parsed_records_with_metadata(
                records=records,
                dev_summaries=dev_summaries,
                imported_dev_wells=imported_dev_wells,
                import_failures=import_failures,
                source_kind=str(operation.source_kind),
            )
            auto_applied_dev_configs = _auto_apply_imported_dev_param_overrides(
                records=records,
                dev_summaries=dev_summaries,
            )
            status.write(operation.count_message(len(records)))
            detailed_dev_param_summaries = tuple(
                summary
                for summary in dev_summaries
                if not bool(getattr(summary, "simple_target_only", False))
            )
            if detailed_dev_param_summaries:
                st.session_state["wt_records_overview_expand_once"] = True
                status.write(
                    "Прочитаны параметры траекторий .dev. Они показаны ниже в статусе загрузки целей."
                )
            if auto_applied_dev_configs > 0:
                status.write(
                    "Индивидуальные параметры из .dev применены автоматически "
                    f"для {auto_applied_dev_configs} скважин."
                )
            simple_dev_target_names = sorted(
                {
                    str(well.name).strip()
                    for well in imported_dev_wells
                    if len(pd.DataFrame(well.stations).index) == 3
                    and str(well.name).strip()
                }
            )
            if simple_dev_target_names:
                st.session_state["wt_records_overview_expand_once"] = True
                status.write(
                    "`.dev` с тремя точками импортированы как обычные цели "
                    "`S / t1 / t3`. Для них используются общие параметры расчёта."
                )
            if import_failures:
                st.session_state["wt_records_overview_expand_once"] = True
                status.write(
                    "Часть .dev скважин пропущена при импорте. Причины показаны ниже в статусе загрузки целей."
                )
            if auto_layout_applied:
                status.write(ptc_target_import.AUTO_LAYOUT_APPLIED_MESSAGE)
            elapsed = perf_counter() - started
            status.update(
                label=operation.success_label(elapsed),
                state="complete",
                expanded=False,
            )
            st.rerun()
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
    overview_expand_once = bool(st.session_state.pop("wt_records_overview_expand_once", False))
    well_count = int(
        sum(
            not is_pilot_name(record.name) and not is_zbs_record(record)
            for record in records
        )
    )
    pilot_count = int(sum(is_pilot_name(record.name) for record in records))
    zbs_records = [record for record in records if is_zbs_record(record)]
    zbs_count = int(len(zbs_records))
    multi_horizontal_count = int(
        sum(
            not is_pilot_name(record.name)
            and record_multi_horizontal_level_count(record) > 1
            for record in records
        )
    )
    x1, x2, x3, x4, x5 = st.columns(5, gap="small")
    x1.metric("Скважин", f"{well_count}")
    x2.metric("Пилотов", f"{pilot_count}")
    x3.metric("Боковых стволов", f"{zbs_count}")
    x4.metric("Многопластовых скважин", f"{multi_horizontal_count}")
    x5.metric("Ошибки импорта", f"{problem_count}")
    if zbs_records:
        parent_names = sorted(
            {
                parent_name_for_zbs(record.name)
                for record in zbs_records
                if str(record.name).strip()
            }
        )
        st.info(
            "Есть ЗБС: для расчёта загрузите "
            + (
                f'фактическую основную скважину "{parent_names[0]}"'
                if len(parent_names) == 1
                else "фактические основные скважины "
                + ", ".join(f'"{name}"' for name in parent_names)
            )
            + "."
        )
    with st.expander(
        "Статус загрузки целей",
        expanded=bool(problem_count > 0 or overview_expand_once),
    ):
        imported_dev_params = tuple(st.session_state.get("wt_imported_dev_params", ()))
        has_manual_assignments = bool(
            _manual_well_calc_override_enabled()
            and _manual_well_calc_profile_assignments(
                available_names=_unique_well_names(record.name for record in records)
            )
        )
        if imported_dev_params:
            st.markdown("#### Прочитанные параметры .dev")
            if not has_manual_assignments:
                st.caption(
                    "Параметры показаны для контроля. Расчёт ниже пока "
                    "использует настройки из блока параметров расчёта."
                )
            st.dataframe(
                ptc_target_import.dev_target_import_summary_dataframe(
                    imported_dev_params
                ),
                hide_index=True,
                width="stretch",
            )
            st.markdown("")
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
    _render_bulk_horizontal_length_preprocess(records=records)


def _render_bulk_horizontal_length_preprocess(
    *,
    records: list[WelltrackRecord],
) -> None:
    preprocess_length_key = "wt_preprocess_horizontal_length_m"
    preprocess_feedback_key = "wt_preprocess_feedback_info"
    preprocess_selected_names_key = "wt_preprocess_selected_names"
    preprocess_pending_names_key = "wt_preprocess_pending_selected_names"
    preprocess_pad_key = "wt_preprocess_select_pad_id"
    raw_preprocess_length = st.session_state.get(preprocess_length_key)
    if not isinstance(raw_preprocess_length, (int, float)) or not np.isfinite(
        float(raw_preprocess_length)
    ) or float(raw_preprocess_length) <= 0.0:
        st.session_state[preprocess_length_key] = 1500.0
    preprocess_all_names = _sync_preprocess_selection_state(
        records=records,
        selected_names_key=preprocess_selected_names_key,
        pending_names_key=preprocess_pending_names_key,
    )
    pads, _, well_names_by_pad_id = _pad_membership(records)
    pad_ids = [str(pad.pad_id) for pad in pads]
    if (
        pad_ids
        and str(st.session_state.get(preprocess_pad_key, "")).strip() not in pad_ids
    ):
        st.session_state[preprocess_pad_key] = pad_ids[0]

    with st.expander("Изменить длину ГС", expanded=False):
        st.caption(
            "Эта опция удлиняет или срезает ГС вдоль линии `t1 -> t3`."
        )
        feedback_message = str(st.session_state.pop(preprocess_feedback_key, "")).strip()
        if feedback_message:
            st.info(feedback_message)
        select_cols = st.columns(
            [7.0, 1.25],
            gap="small",
            vertical_alignment="bottom",
        )
        select_col = select_cols[0] if len(select_cols) > 0 else st
        select_all_col = select_cols[1] if len(select_cols) > 1 else st
        getattr(select_col, "multiselect", st.multiselect)(
            "Скважины для изменения длины ГС",
            options=preprocess_all_names,
            key=preprocess_selected_names_key,
            disabled=not preprocess_all_names,
        )
        select_all_clicked = getattr(select_all_col, "button", st.button)(
            "Выбрать все скважины",
            key="wt_preprocess_select_all",
            icon=":material/done_all:",
            width="stretch",
            disabled=not preprocess_all_names,
        )

        add_pad_clicked = False
        replace_with_pad_clicked = False
        if len(pad_ids) > 1:
            pad_cols = st.columns(
                [2.5, 1.45, 1.45, 2.85],
                gap="small",
                vertical_alignment="bottom",
            )
            pad_col = pad_cols[0] if len(pad_cols) > 0 else st
            pad_add_col = pad_cols[1] if len(pad_cols) > 1 else st
            pad_only_col = pad_cols[2] if len(pad_cols) > 2 else st
            getattr(pad_col, "selectbox", st.selectbox)(
                "Куст для изменения ГС",
                options=pad_ids,
                format_func=lambda value: _pad_display_label(
                    next(pad for pad in pads if str(pad.pad_id) == str(value))
                ),
                key=preprocess_pad_key,
            )
            add_pad_clicked = getattr(pad_add_col, "button", st.button)(
                "Добавить куст в выбор",
                key="wt_preprocess_add_pad",
                icon=":material/filter_alt:",
                width="stretch",
            )
            replace_with_pad_clicked = getattr(pad_only_col, "button", st.button)(
                "Только этот куст",
                key="wt_preprocess_only_pad",
                icon=":material/rule:",
                width="stretch",
            )

        if select_all_clicked:
            st.session_state[preprocess_pending_names_key] = list(preprocess_all_names)
            _rerun_fragment()
        if add_pad_clicked:
            selected_pad_id = str(st.session_state.get(preprocess_pad_key, "")).strip()
            current_selected = [
                str(name)
                for name in st.session_state.get(preprocess_selected_names_key, [])
            ]
            st.session_state[preprocess_pending_names_key] = _unique_well_names(
                [
                    *current_selected,
                    *well_names_by_pad_id.get(selected_pad_id, ()),
                ]
            )
            _rerun_fragment()
        if replace_with_pad_clicked:
            selected_pad_id = str(st.session_state.get(preprocess_pad_key, "")).strip()
            st.session_state[preprocess_pending_names_key] = list(
                well_names_by_pad_id.get(selected_pad_id, ())
            )
            _rerun_fragment()

        preprocess_cols = st.columns(
            [2.4, 1.2, 2.4],
            gap="small",
            vertical_alignment="bottom",
        )
        preprocess_length_col = preprocess_cols[0] if len(preprocess_cols) > 0 else st
        preprocess_action_col = preprocess_cols[1] if len(preprocess_cols) > 1 else st
        length_input = getattr(preprocess_length_col, "number_input", st.number_input)
        action_button = getattr(preprocess_action_col, "button", st.button)
        length_input(
            "Новая длина ГС, м",
            key=preprocess_length_key,
            min_value=1.0,
            step=100.0,
            help=(
                "Для каждой пары `t1/t3` будет изменена только координата `t3`. "
                "Направление участка в пространстве сохранится."
            ),
        )
        apply_horizontal_length_clicked = action_button(
            "Применить",
            key="wt_preprocess_horizontal_length_apply",
            icon=":material/straighten:",
            width="stretch",
            disabled=not preprocess_all_names,
        )
        if apply_horizontal_length_clicked:
            selected_preprocess_names = _coerce_preprocess_selection(
                st.session_state.get(preprocess_selected_names_key, []),
                all_names=preprocess_all_names,
            )
            if not selected_preprocess_names:
                st.warning("Выберите хотя бы одну скважину.")
                return
            selected_keys = {
                well_name_key(name) for name in selected_preprocess_names
            }
            excluded_message = _preprocess_excluded_records_message(records)
            st.session_state[preprocess_feedback_key] = excluded_message
            selected_records = [
                record
                for record in records
                if not is_pilot_name(record.name)
                and well_name_key(record.name) in selected_keys
            ]
            try:
                horizontal_length_changes, skipped_names = _bulk_horizontal_length_changes(
                    records=selected_records,
                    target_length_m=float(
                        st.session_state.get(preprocess_length_key, 0.0)
                    ),
                )
            except ValueError as exc:
                st.warning(str(exc))
                return
            if not horizontal_length_changes:
                if skipped_names:
                    skipped_text = ", ".join(skipped_names[:6])
                    skipped_suffix = "..." if len(skipped_names) > 6 else ""
                    st.warning(
                        "Не удалось скорректировать длину ГС для: "
                        f"{skipped_text}{skipped_suffix}."
                    )
                else:
                    st.info(
                        "Все доступные `t3` уже соответствуют заданной длине ГС."
                    )
                return
            updated_names = _apply_edit_targets_changes(
                horizontal_length_changes,
                source="bulk_horizontal_length_preprocess",
            )
            _clear_t1_t3_order_resolution_state()
            skipped_note = ""
            if skipped_names:
                skipped_text = ", ".join(skipped_names[:6])
                skipped_suffix = "..." if len(skipped_names) > 6 else ""
                skipped_note = f"Пропущено: {skipped_text}{skipped_suffix}."
            st.session_state["wt_edit_targets_applied_note"] = skipped_note
            st.session_state["wt_records_overview_expand_once"] = True
            if updated_names:
                st.rerun()


def _records_overview_dataframe(
    records: list[WelltrackRecord],
) -> pd.DataFrame:
    base_df = ptc_target_records.records_overview_dataframe(
        records,
        wellhead_z_tolerance_m=WT_IMPORT_WELLHEAD_Z_TOLERANCE_M,
    )
    simple_dev_note_by_key = {
        well_name_key(well.name): (
            "Импорт из .dev: 3 точки `S / t1 / t3`, используются общие параметры расчёта."
        )
        for well in _imported_dev_target_wells_from_state()
        if len(pd.DataFrame(well.stations).index) == 3
        and str(well.name).strip()
    }
    if simple_dev_note_by_key and not base_df.empty:
        base_df = base_df.copy()
        for row_index, row in base_df.iterrows():
            note = simple_dev_note_by_key.get(
                well_name_key(str(row.get("Скважина", "")).strip())
            )
            if not note:
                continue
            current_note = str(row.get("Примечание", "")).strip()
            base_df.at[row_index, "Примечание"] = (
                note
                if current_note in {"", "—"}
                else f"{current_note}; {note}"
            )
    raw_failures = st.session_state.get(ptc_target_import.TARGET_IMPORT_FAILURES_STATE_KEY, ())
    failure_rows: list[dict[str, object]] = []
    if isinstance(raw_failures, (list, tuple)):
        for item in raw_failures:
            well_name = str(getattr(item, "well_name", "")).strip()
            problem = str(getattr(item, "problem", "")).strip()
            source_label = str(getattr(item, "source_label", "")).strip()
            if not well_name and not problem:
                continue
            failure_rows.append(
                {
                    "Скважина": well_name or "—",
                    "Точек": "—",
                    "Отход t1, м": "—",
                    "Длина ГС, м": "—",
                    "Примечание": (
                        f".dev: {source_label}"
                        if source_label
                        else "Ошибка импорта .dev"
                    ),
                    "Статус": "❌",
                    "Проблема": problem or "Не удалось импортировать .dev траекторию.",
                }
            )
    if not failure_rows:
        return base_df
    return pd.concat(
        [base_df, pd.DataFrame(failure_rows, columns=list(base_df.columns))],
        ignore_index=True,
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
            "ПИ min, deg/10м": (
                None
                if item.dls_min_deg_per_30m is None
                else dls_to_pi(item.dls_min_deg_per_30m)
            ),
            "ПИ avg, deg/10м": (
                None
                if item.dls_mean_deg_per_30m is None
                else dls_to_pi(item.dls_mean_deg_per_30m)
            ),
            "ПИ max, deg/10м": (
                None
                if item.dls_max_deg_per_30m is None
                else dls_to_pi(item.dls_max_deg_per_30m)
            ),
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
    return _cached_reference_fund_analyses(
        actual_wells,
        cache_key="wt_actual_fund_analysis_cache",
    )


def _approved_fund_analyses(
    approved_wells: tuple[ImportedTrajectoryWell, ...],
) -> tuple[ActualFundWellAnalysis, ...]:
    return _cached_reference_fund_analyses(
        approved_wells,
        cache_key="wt_approved_fund_analysis_cache",
    )


def _cached_reference_fund_analyses(
    reference_wells: tuple[ImportedTrajectoryWell, ...],
    *,
    cache_key: str,
) -> tuple[ActualFundWellAnalysis, ...]:
    signature = _actual_fund_analysis_signature(reference_wells)
    cached = st.session_state.get(cache_key)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        analyses = cached.get("analyses")
        if isinstance(analyses, tuple):
            return analyses
    analyses = build_actual_fund_well_analyses(reference_wells)
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
            "Учитываются горизонтальные скважины с нормальным профилем "
            "(KOP, HOLD, ГС, ПИ в допустимых пределах)."
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
                "KOP подбирается по фактическому фонду на глубине входа в ГС (TVD t1)."
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
            "Профиль восстановлен по инклинометрии (XYZ/MD). "
            "Участки: вертикаль, набор, стабилизация, горизонталь. Отмечены KOP и начало ГС."
        )
        _render_actual_fund_well_detail(analyses)


def _render_raw_records_table(records: list[WelltrackRecord]) -> None:
    highlight_names = {
        str(name)
        for name in (st.session_state.get("wt_edit_targets_highlight_names") or [])
        if str(name).strip()
    }
    highlight_source = str(st.session_state.get("wt_edit_targets_last_source", "")).strip()
    raw_highlight_points = st.session_state.get("wt_edit_targets_highlight_points")
    highlight_points: dict[str, set[int]] = {}
    if isinstance(raw_highlight_points, Mapping):
        for raw_name, raw_indices in raw_highlight_points.items():
            indices = set(_coerce_highlight_point_indices(raw_indices))
            if indices:
                highlight_points[str(raw_name)] = indices

    point_count = sum(len(getattr(record, "points", ()) or ()) for record in records)
    large_point_table = point_count > WT_RAW_RECORDS_AUTO_RENDER_POINT_LIMIT
    needs_explicit_open = large_point_table and not highlight_names
    edit_mode_key = "wt_raw_records_edit_mode"
    editor_nonce_key = "wt_raw_records_editor_nonce"
    table_key = "wt_show_raw_records_table"
    highlight_signature_key = "wt_show_raw_records_table_highlight_signature"
    st.session_state.setdefault(edit_mode_key, False)
    st.session_state.setdefault(editor_nonce_key, 0)
    if large_point_table and highlight_names:
        highlight_signature = (
            tuple(sorted(highlight_names)),
            tuple(
                sorted(
                    (name, tuple(sorted(indices)))
                    for name, indices in highlight_points.items()
                )
            ),
            highlight_source,
        )
        if st.session_state.get(highlight_signature_key) != highlight_signature:
            st.session_state[table_key] = False
            st.session_state[highlight_signature_key] = highlight_signature
    else:
        st.session_state[highlight_signature_key] = None
    with st.expander(
        "Текущие точки скважин",
        expanded=bool(highlight_names),
    ):
        show_full_large_table = False
        if needs_explicit_open:
            st.session_state.setdefault(table_key, False)
            show_full_large_table = bool(
                st.toggle(
                    "Показать полную таблицу точек",
                    key=table_key,
                    help=(
                        "Большие WELLTRACK-файлы могут содержать десятки тысяч "
                        "точек. Таблица строится только по запросу, чтобы обычный "
                        "rerun после расчёта не блокировал результаты."
                    ),
                )
            )
            if not show_full_large_table:
                st.caption(
                    f"Таблица скрыта для ускорения страницы: {point_count} точек. "
                    "Расчёт использует полный импортированный набор данных."
                )
                return
        elif large_point_table and highlight_names:
            st.session_state.setdefault(table_key, False)
            show_full_large_table = bool(
                st.toggle(
                    "Показать полную таблицу точек",
                    key=table_key,
                    help=(
                        "После редактирования по умолчанию показываются только "
                        "изменённые скважины, чтобы не блокировать страницу на "
                        "больших наборах точек."
                    ),
                )
            )

        records_for_table = records
        if large_point_table and highlight_names and not show_full_large_table:
            records_for_table = [
                record
                for record in records
                if str(record.name).strip() in highlight_names
            ]
            filtered_point_count = sum(
                len(getattr(record, "points", ()) or ()) for record in records_for_table
            )
            st.caption(
                "Для ускорения показаны только изменённые скважины: "
                f"{len(records_for_table)} из {len(records)} "
                f"({filtered_point_count} из {point_count} точек)."
            )

        raw_records_df = ptc_target_records.raw_records_dataframe(records_for_table)
        if bool(st.session_state.get(edit_mode_key)):
            st.caption(
                "Можно менять координаты `X/Y/Z` и вставлять их из Excel. "
                "Изменения применяются только после кнопки «Сохранить изменения»."
            )
            with st.form(
                f"wt_raw_records_form_{int(st.session_state.get(editor_nonce_key, 0))}",
                clear_on_submit=False,
            ):
                edited_table = st.data_editor(
                    raw_records_df,
                    key=(
                        "wt_raw_records_editor_"
                        f"{int(st.session_state.get(editor_nonce_key, 0))}"
                    ),
                    hide_index=True,
                    num_rows="fixed",
                    width="stretch",
                    disabled=["Скважина", "Точка"],
                    column_config={
                        "Скважина": st.column_config.TextColumn("Скважина"),
                        "Точка": st.column_config.TextColumn("Точка"),
                        "X, м": st.column_config.NumberColumn("X, м"),
                        "Y, м": st.column_config.NumberColumn("Y, м"),
                        "Z, м": st.column_config.NumberColumn("Z, м"),
                    },
                )
                action_cols = st.columns([1.35, 1.0, 4.0], gap="small")
                save_submit_button = getattr(
                    action_cols[0],
                    "form_submit_button",
                    st.form_submit_button,
                )
                cancel_submit_button = getattr(
                    action_cols[1],
                    "form_submit_button",
                    st.form_submit_button,
                )
                save_clicked = save_submit_button(
                    "Сохранить изменения",
                    type="primary",
                    icon=":material/save:",
                    width="content",
                )
                cancel_clicked = cancel_submit_button(
                    "Отмена",
                    width="content",
                )
            if cancel_clicked:
                st.session_state[edit_mode_key] = False
                st.session_state[editor_nonce_key] = (
                    int(st.session_state.get(editor_nonce_key, 0)) + 1
                )
                _rerun_fragment()
            if save_clicked:
                try:
                    changes = ptc_edit_targets.raw_records_editor_changes(
                        records_for_table,
                        edited_table,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    if changes:
                        _apply_edit_targets_changes(
                            changes,
                            source="raw_records_table",
                        )
                    st.session_state[edit_mode_key] = False
                    st.session_state[editor_nonce_key] = (
                        int(st.session_state.get(editor_nonce_key, 0)) + 1
                    )
                    st.rerun()
            return

        edit_clicked = st.button(
            "Редактировать",
            icon=":material/edit:",
            width="content",
        )
        if edit_clicked:
            st.session_state[edit_mode_key] = True
            _rerun_fragment()
        if highlight_names:
            if highlight_source == "bulk_horizontal_length_preprocess":
                st.success(
                    "Скорректированные точки `t3` подсвечены. "
                    "Запустите расчёт для обновления траекторий."
                )
            elif highlight_source == "pad_layout":
                st.success(
                    "Изменённые координаты устьев подсвечены. "
                    "Запустите расчёт для обновления траекторий."
                )
            elif highlight_source == "three_viewer_pad_layout":
                st.success(
                    "Изменённые в 3D-редакторе координаты устьев подсвечены. "
                    "Запустите расчёт для обновления траекторий."
                )
            elif highlight_source == "raw_records_table":
                st.success(
                    "Изменённые в таблице точки подсвечены. "
                    "Запустите расчёт для обновления траекторий."
                )
            else:
                st.success(
                    "Изменённые в 3D-редакторе точки подсвечены. "
                    "Запустите расчёт для обновления траекторий."
                )
        raw_df = arrow_safe_text_dataframe(raw_records_df)
        if highlight_names and not raw_df.empty:
            point_indices = raw_df.groupby("Скважина", sort=False).cumcount()

            def _highlight_edit_rows(row: pd.Series) -> list[str]:
                well_name = str(row.get("Скважина", ""))
                point_index = int(point_indices.loc[row.name])
                explicit_indices = highlight_points.get(well_name)
                if (
                    well_name in highlight_names
                    and explicit_indices is not None
                    and point_index in explicit_indices
                ):
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
    panel_context = st.container(border=True) if border else nullcontext()
    with panel_context:
        st.markdown(
            f'<div id="{WT_T1T3_ORDER_PANEL_ANCHOR_ID}"></div>',
            unsafe_allow_html=True,
        )
        resolution_message = _t1_t3_order_resolution_message()
        if resolution_message is not None:
            level, message = resolution_message
            if level == "success":
                st.success(message)
            else:
                st.info(message)
        anchor_by_well_name = _t1_t3_order_anchor_by_well_name(records)
        detected_issues = _detected_t1_t3_order_issues(
            records,
            anchor_by_well_name=anchor_by_well_name,
        )
        issues = _current_t1_t3_order_issues(records, detected_issues=detected_issues)
        if not issues:
            if detected_issues:
                st.info(
                    "Активных предупреждений по порядку `t1/t3` нет. "
                    "Для отмеченных скважин текущий порядок оставлен без изменений."
                )
            else:
                st.success("Проверка порядка t1/t3 — без замечаний.")
            return

        st.warning(
            "От устья до t1 дальше, чем до t3 — возможно, t1 и t3 перепутаны местами."
        )
        header_cols = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 3.0], gap="small")
        header_cols[0].markdown(
            "<div style='text-align: center;'><strong>Исправить</strong></div>",
            unsafe_allow_html=True,
        )
        header_cols[1].markdown("**Скважина**")
        header_cols[2].markdown("**Отход →t1, м**")
        header_cols[3].markdown("**Отход →t3, м**")
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
            anchor_label = str(getattr(item, "anchor_label", "S→") or "S→")
            row_cols[2].markdown(f"{anchor_label}{float(item.t1_offset_m):.2f}")
            row_cols[3].markdown(f"{anchor_label}{float(item.t3_offset_m):.2f}")
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
                anchor_by_well_name=anchor_by_well_name,
                min_delta_m=WT_T1T3_MIN_DELTA_M,
            )
            original_records = st.session_state.get("wt_records_original")
            if original_records is None:
                original_records = list(records)
            st.session_state["wt_records_original"] = swap_t1_t3_for_wells(
                records=list(original_records),
                well_names=target_names,
                anchor_by_well_name=anchor_by_well_name,
                min_delta_m=WT_T1T3_MIN_DELTA_M,
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
            "в исходных позициях, чтобы не ломать порядок MD."
        )


def _detected_t1_t3_order_issues(
    records: list[WelltrackRecord],
    *,
    anchor_by_well_name: dict[str, Point3D] | None = None,
) -> list[object]:
    return detect_t1_t3_order_issues(
        visible_well_records(records, include_zbs=True),
        min_delta_m=WT_T1T3_MIN_DELTA_M,
        anchor_by_well_name=(
            anchor_by_well_name
            if anchor_by_well_name is not None
            else _t1_t3_order_anchor_by_well_name(records)
        ),
    )


def _current_t1_t3_order_issues(
    records: list[WelltrackRecord],
    *,
    detected_issues: list[object] | None = None,
) -> list[object]:
    issues = (
        detected_issues
        if detected_issues is not None
        else _detected_t1_t3_order_issues(records)
    )
    acknowledged_well_names = _t1_t3_order_acknowledged_well_names()
    return [
        item for item in issues if str(item.well_name) not in acknowledged_well_names
    ]


def _t1_t3_order_anchor_by_well_name(
    records: list[WelltrackRecord],
) -> dict[str, Point3D]:
    record_list = list(records)
    zbs_records = [
        record
        for record in record_list
        if is_zbs_record(record) and len(record.points) >= 2
    ]
    if not zbs_records:
        return {}
    zbs_parent_keys = {
        well_name_key(parent_name_for_zbs(record.name)) for record in zbs_records
    }
    anchor_by_well_name: dict[str, Point3D] = {}
    parent_points_by_key: dict[str, tuple[Point3D, ...]] = {}
    parent_points_priority_by_key: dict[str, int] = {}
    for record in record_list:
        parent_key = well_name_key(record.name)
        if (
            is_pilot_name(record.name)
            or is_zbs_record(record)
            or not record.points
            or parent_key not in zbs_parent_keys
        ):
            continue
        points = tuple(record.points)
        parent_points = tuple(
            Point3D(x=float(point.x), y=float(point.y), z=float(point.z))
            for point in points
        )
        if not parent_points:
            continue
        anchor_by_well_name[str(record.name)] = parent_points[0]
        parent_points_by_key[parent_key] = parent_points
        parent_points_priority_by_key[parent_key] = 2

    for reference_well in _reference_wells_from_state():
        parent_key = well_name_key(reference_well.name)
        if parent_key not in zbs_parent_keys:
            continue
        reference_points = _reference_well_points(reference_well)
        priority = 3 if str(reference_well.kind) == REFERENCE_WELL_ACTUAL else 1
        if reference_points:
            existing_priority = parent_points_priority_by_key.get(parent_key, -1)
            if priority >= existing_priority:
                parent_points_by_key[parent_key] = reference_points
                parent_points_priority_by_key[parent_key] = priority
        anchor_by_well_name.setdefault(
            str(reference_well.name),
            reference_well.surface,
        )

    for record in zbs_records:
        parent_points = parent_points_by_key.get(
            well_name_key(parent_name_for_zbs(record.name))
        )
        if not parent_points:
            continue
        anchor_by_well_name[str(record.name)] = _nearest_xy_anchor_to_targets(
            parent_points=parent_points,
            target_points=tuple(record.points),
        )
    return anchor_by_well_name


def _reference_well_points(
    reference_well: ImportedTrajectoryWell,
) -> tuple[Point3D, ...]:
    stations = reference_well.stations
    if not isinstance(stations, pd.DataFrame) or not {"X_m", "Y_m", "Z_m"}.issubset(
        stations.columns
    ):
        return (reference_well.surface,)
    if stations.empty:
        return (reference_well.surface,)
    return tuple(
        Point3D(x=float(row.X_m), y=float(row.Y_m), z=float(row.Z_m))
        for row in stations[["X_m", "Y_m", "Z_m"]].itertuples(index=False)
    )


def _nearest_xy_anchor_to_targets(
    *,
    parent_points: tuple[Point3D, ...],
    target_points: tuple[WelltrackPoint, ...],
) -> Point3D:
    best_point = parent_points[0]
    best_distance = float("inf")
    for parent_point in parent_points:
        for target in target_points:
            distance = float(
                np.hypot(
                    float(parent_point.x) - float(target.x),
                    float(parent_point.y) - float(target.y),
                )
            )
            if distance < best_distance:
                best_distance = distance
                best_point = parent_point
    return best_point


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
    return ptc_pad_state.detect_ui_pads(
        records,
        remembered_degenerate_well_names=ptc_target_import.simple_dev_target_well_names_from_state(
            st.session_state
        ),
    )


def _ensure_pad_configs(base_records: list[WelltrackRecord]) -> list[WellPad]:
    return ptc_pad_state.ensure_pad_configs(
        st.session_state,
        base_records=visible_well_records(base_records, include_zbs=False),
    )


def _build_pad_plan_map(pads: list[WellPad]) -> dict[str, PadLayoutPlan]:
    return ptc_pad_state.build_pad_plan_map(st.session_state, pads)


def _project_pads_for_ui(records: list[WelltrackRecord]) -> list[WellPad]:
    return ptc_pad_state.project_pads_for_ui(
        st.session_state, visible_well_records(records, include_zbs=False)
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
        st.session_state,
        visible_well_records(records, include_zbs=False),
    )


def _well_label_display_names(
    records: list[WelltrackRecord],
) -> dict[str, str]:
    _, _, well_names_by_pad_id = _pad_membership(records)
    display_names: dict[str, str] = {}
    for ordered_names in well_names_by_pad_id.values():
        for slot_index, well_name in enumerate(ordered_names, start=1):
            display_names[str(well_name)] = f"{well_name} ({int(slot_index)})"
    return display_names


def _normalize_focus_pad_id(
    *,
    records: list[WelltrackRecord],
    requested_pad_id: str | None,
) -> str:
    return ptc_pad_state.normalize_focus_pad_id(
        st.session_state,
        records=visible_well_records(records, include_zbs=False),
        requested_pad_id=requested_pad_id,
    )


def _focus_pad_well_names(
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str | None,
) -> tuple[str, ...]:
    return ptc_pad_state.focus_pad_well_names(
        st.session_state,
        records=visible_well_records(records, include_zbs=False),
        focus_pad_id=focus_pad_id,
    )


def _focus_pad_fixed_well_names(
    *,
    records: list[WelltrackRecord],
    focus_pad_id: str | None,
) -> tuple[str, ...]:
    return ptc_pad_state.focus_pad_fixed_well_names(
        st.session_state,
        records=visible_well_records(records, include_zbs=False),
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


def _toggle_pad_layout_details_panel() -> bool:
    return ptc_pad_state.toggle_pad_layout_details_open(st.session_state)


def _render_pad_layout_panel(records: list[WelltrackRecord]) -> None:
    base_records = st.session_state.get("wt_records_original")
    if base_records is None:
        base_records = list(records)
    pads = _ensure_pad_configs(base_records=list(base_records))
    if not pads:
        _render_t1_t3_order_panel(records=records, border=True)
        return

    with st.container(border=True):
        _render_t1_t3_order_panel(records=records, border=False)
        st.toggle(
            "Авто-порядок по глубине целевого пласта",
            key="wt_pad_auto_order_by_target_depth",
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
        selected_id = str(st.session_state.get("wt_pad_selected_id", pad_ids[0]))
        if selected_id not in pad_ids:
            selected_id = pad_ids[0]
            st.session_state["wt_pad_selected_id"] = selected_id

        summary_selected_pad_meta = pad_metadata.get(selected_id)
        details_open = ptc_pad_state.pad_layout_details_open(st.session_state)
        st.button(
            (
                "Скрыть настройки выбранного куста"
                if details_open
                else "Настроить положение куста, НДС и расстояние между устьями."
            ),
            key="wt_pad_layout_details_toggle",
            icon=(
                ":material/expand_less:"
                if details_open
                else ":material/expand_more:"
            ),
            width="content",
            on_click=_toggle_pad_layout_details_panel,
        )
        details_open = ptc_pad_state.pad_layout_details_open(st.session_state)
        if not details_open:
            summary_auto_name_notice = str(
                getattr(summary_selected_pad_meta, "auto_name_notice", "")
            ).strip()
            if summary_auto_name_notice:
                st.info(summary_auto_name_notice)
            return

        selected_id = str(
            st.selectbox("Выберите куст", options=pad_ids, key="wt_pad_selected_id")
        )
        selected_pad = next(
            (pad for pad in pads if str(pad.pad_id) == selected_id), pads[0]
        )
        selected_pad_meta = pad_metadata.get(selected_id)
        source_surfaces_defined = bool(
            getattr(selected_pad_meta, "source_surfaces_defined", False)
        )
        auto_name_notice = str(
            getattr(selected_pad_meta, "auto_name_notice", "")
        ).strip()
        config_map = st.session_state.get("wt_pad_configs", {})
        selected_cfg: dict[str, object] = dict(
            config_map.get(selected_id, _pad_config_defaults(selected_pad))
        )
        resolved_nds_azimuth_deg = ptc_pad_state.resolved_pad_nds_azimuth_deg(
            st.session_state,
            pad=selected_pad,
            nds_azimuth_deg=float(selected_cfg["nds_azimuth_deg"]),
        )
        selected_cfg["nds_azimuth_deg"] = resolved_nds_azimuth_deg
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
            "allow_source_surface_edit": (
                f"wt_pad_cfg_allow_source_surface_edit_{selected_id}"
            ),
            "apply_auto_order": f"wt_pad_cfg_apply_auto_order_{selected_id}",
        }
        for field, widget_key in widget_keys.items():
            if field in {
                "surface_anchor_center",
                "allow_source_surface_edit",
                "apply_auto_order",
            }:
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = (
                        previous_anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                        if field == "surface_anchor_center"
                        else (
                            bool(
                                selected_cfg.get(
                                    ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY,
                                    False,
                                )
                            )
                            if field == "allow_source_surface_edit"
                            else bool(
                                selected_cfg.get(
                                    ptc_pad_state.WT_PAD_APPLY_AUTO_ORDER_KEY,
                                    False,
                                )
                            )
                        )
                    )
                continue
            if widget_key not in st.session_state:
                st.session_state[widget_key] = float(selected_cfg[field])

        if auto_name_notice:
            st.info(auto_name_notice)
        allow_source_surface_edit = False
        apply_auto_order = False
        if source_surfaces_defined:
            allow_source_surface_edit = st.toggle(
                "Разрешить редактирование позиций куста",
                key=widget_keys["allow_source_surface_edit"],
            )
            if not allow_source_surface_edit:
                st.session_state[widget_keys["apply_auto_order"]] = False
            apply_auto_order = st.toggle(
                "Применить авто-порядок",
                key=widget_keys["apply_auto_order"],
                disabled=not allow_source_surface_edit,
            )
        surface_controls_disabled = bool(
            source_surfaces_defined and not allow_source_surface_edit
        )
        if surface_controls_disabled:
            selected_cfg[ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY] = False
            selected_cfg[ptc_pad_state.WT_PAD_APPLY_AUTO_ORDER_KEY] = False
            config_map[selected_id] = selected_cfg
            st.session_state["wt_pad_configs"] = config_map
            selected_cfg = dict(_pad_config_for_ui(selected_pad))
            st.session_state[widget_keys["surface_anchor_center"]] = (
                str(selected_cfg.get("surface_anchor_mode"))
                == PAD_SURFACE_ANCHOR_CENTER
            )
            for field in (
                "spacing_m",
                "nds_azimuth_deg",
                "first_surface_x",
                "first_surface_y",
                "first_surface_z",
            ):
                st.session_state[widget_keys[field]] = float(selected_cfg[field])

        anchor_center = st.toggle(
            "Координата куста = центр расстановки",
            key=widget_keys["surface_anchor_center"],
            disabled=surface_controls_disabled,
        )
        anchor_mode = (
            PAD_SURFACE_ANCHOR_CENTER
            if bool(anchor_center)
            else PAD_SURFACE_ANCHOR_FIRST
        )
        if anchor_mode != previous_anchor_mode:
            previous_auto_nds = ptc_pad_state.resolved_pad_nds_azimuth_deg(
                st.session_state,
                pad=selected_pad,
                nds_azimuth_deg=estimate_pad_nds_azimuth_deg(
                    wells=selected_pad.wells,
                    surface_x=float(selected_pad.surface.x),
                    surface_y=float(selected_pad.surface.y),
                    surface_anchor_mode=previous_anchor_mode,
                ),
            )
            current_nds = float(
                st.session_state.get(
                    widget_keys["nds_azimuth_deg"],
                    selected_cfg["nds_azimuth_deg"],
                )
            )
            if abs(current_nds - previous_auto_nds) <= 1e-6:
                st.session_state[widget_keys["nds_azimuth_deg"]] = float(
                    ptc_pad_state.resolved_pad_nds_azimuth_deg(
                        st.session_state,
                        pad=selected_pad,
                        nds_azimuth_deg=estimate_pad_nds_azimuth_deg(
                            wells=selected_pad.wells,
                            surface_x=float(selected_pad.surface.x),
                            surface_y=float(selected_pad.surface.y),
                            surface_anchor_mode=anchor_mode,
                        ),
                    )
                )
            previous_anchor_xyz = ptc_pad_state.pad_anchor_defaults(
                st.session_state,
                pad=selected_pad,
                anchor_mode=previous_anchor_mode,
            )
            next_anchor_xyz = ptc_pad_state.pad_anchor_defaults(
                st.session_state,
                pad=selected_pad,
                anchor_mode=anchor_mode,
            )
            current_anchor_xyz = (
                float(
                    st.session_state.get(
                        widget_keys["first_surface_x"],
                        selected_cfg["first_surface_x"],
                    )
                ),
                float(
                    st.session_state.get(
                        widget_keys["first_surface_y"],
                        selected_cfg["first_surface_y"],
                    )
                ),
                float(
                    st.session_state.get(
                        widget_keys["first_surface_z"],
                        selected_cfg["first_surface_z"],
                    )
                ),
            )
            if all(
                abs(float(current_value) - float(previous_value)) <= 1e-6
                for current_value, previous_value in zip(
                    current_anchor_xyz,
                    previous_anchor_xyz,
                    strict=True,
                )
            ):
                st.session_state[widget_keys["first_surface_x"]] = float(
                    next_anchor_xyz[0]
                )
                st.session_state[widget_keys["first_surface_y"]] = float(
                    next_anchor_xyz[1]
                )
                st.session_state[widget_keys["first_surface_z"]] = float(
                    next_anchor_xyz[2]
                )

        p1, p2, p3, p4, p5 = st.columns(5, gap="small")
        spacing_m = p1.number_input(
            "Расстояние между устьями, м",
            min_value=0.0,
            step=5.0,
            key=widget_keys["spacing_m"],
            disabled=surface_controls_disabled,
        )
        nds_azimuth_deg = p2.number_input(
            "НДС (азимут), deg",
            min_value=0.0,
            max_value=360.0,
            step=10.0,
            key=widget_keys["nds_azimuth_deg"],
            disabled=surface_controls_disabled,
        )
        first_surface_x = p3.number_input(
            (
                "S куста X (East), м"
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                else "S1 X (East), м"
            ),
            step=10.0,
            key=widget_keys["first_surface_x"],
            disabled=surface_controls_disabled,
        )
        first_surface_y = p4.number_input(
            (
                "S куста Y (North), м"
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                else "S1 Y (North), м"
            ),
            step=10.0,
            key=widget_keys["first_surface_y"],
            disabled=surface_controls_disabled,
        )
        first_surface_z = p5.number_input(
            (
                "S куста Z (TVD), м"
                if anchor_mode == PAD_SURFACE_ANCHOR_CENTER
                else "S1 Z (TVD), м"
            ),
            step=10.0,
            key=widget_keys["first_surface_z"],
            disabled=surface_controls_disabled,
        )

        selected_cfg["spacing_m"] = float(max(spacing_m, 0.0))
        selected_cfg["nds_azimuth_deg"] = float(nds_azimuth_deg) % 360.0
        selected_cfg["first_surface_x"] = float(first_surface_x)
        selected_cfg["first_surface_y"] = float(first_surface_y)
        selected_cfg["first_surface_z"] = float(first_surface_z)
        selected_cfg["surface_anchor_mode"] = anchor_mode
        selected_cfg[ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY] = bool(
            allow_source_surface_edit
        )
        selected_cfg[ptc_pad_state.WT_PAD_APPLY_AUTO_ORDER_KEY] = bool(
            allow_source_surface_edit and apply_auto_order
        )

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
                ),
                "Скважина": st.column_config.SelectboxColumn(
                    "Скважина",
                    options=sorted(
                        (str(well.name) for well in selected_pad.wells),
                        key=well_name_natural_sort_key,
                    ),
                    required=False,
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
            _rerun_fragment()
        selected_cfg["fixed_slots"] = fixed_slots
        config_map[selected_id] = selected_cfg
        st.session_state["wt_pad_configs"] = config_map

        preview_assignments = ptc_pad_state.pad_surface_assignments(
            st.session_state,
            pad=selected_pad,
            config=selected_cfg,
        )
        well_by_name = {
            str(well.name): well for well in selected_pad.wells
        }
        fixed_slot_by_name = {str(name): int(slot) for slot, name in fixed_slots}
        preview_rows: list[dict[str, object]] = []
        for assignment in preview_assignments:
            well = well_by_name.get(str(assignment.well_name))
            if well is None:
                continue
            slot_index = int(assignment.slot_index)
            if source_surfaces_defined and not allow_source_surface_edit:
                fixation_label = "Исх."
            elif source_surfaces_defined and not bool(apply_auto_order):
                fixation_label = "Исх."
            else:
                fixation_label = (
                    "Да"
                    if fixed_slot_by_name.get(str(assignment.well_name))
                    == int(slot_index)
                    else "Авто"
                )
            row = {
                "Порядок": int(slot_index),
                "Скважина": str(assignment.well_name),
                "Фиксация": fixation_label,
            }
            if surface_controls_disabled:
                row["Текущее S X, м"] = float(assignment.surface_x_m)
                row["Текущее S Y, м"] = float(assignment.surface_y_m)
            else:
                row["Новое S X, м"] = float(assignment.surface_x_m)
                row["Новое S Y, м"] = float(assignment.surface_y_m)
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
            "Применить координаты устьев",
            type="primary",
            icon=":material/tune:",
            width="stretch",
            disabled=surface_controls_disabled,
        )
        reset_clicked = a2.button(
            "Вернуть исходные координаты устьев",
            icon=":material/restart_alt:",
            width="stretch",
            disabled=surface_controls_disabled,
        )

        if apply_clicked:
            plan_map = _build_pad_plan_map(pads)
            updated_records = sync_pilot_surfaces_to_parents(
                apply_pad_layout(
                    records=list(records),
                    pads=pads,
                    plan_by_pad_id=plan_map,
                )
            )
            st.session_state["wt_records"] = list(updated_records)
            st.session_state["wt_pad_last_applied_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.session_state["wt_pad_auto_applied_on_import"] = False
            _queue_surface_edit_feedback(
                _changed_surface_well_names(records, updated_records),
                source="pad_layout",
            )
            st.toast("Координаты устьев обновлены по параметрам кустов.")
            st.rerun()

        if reset_clicked:
            st.session_state["wt_records"] = list(base_records)
            _clear_pad_state()
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


def _coerce_preprocess_selection(
    names: object,
    *,
    all_names: list[str],
) -> list[str]:
    visible_by_key = {well_name_key(name): str(name) for name in all_names}
    coerced: list[str] = []
    seen: set[str] = set()
    for raw_name in names if isinstance(names, (list, tuple, set)) else []:
        visible_name = visible_by_key.get(well_name_key(raw_name))
        if visible_name is None or visible_name in seen:
            continue
        coerced.append(visible_name)
        seen.add(visible_name)
    return coerced


def _sync_preprocess_selection_state(
    *,
    records: list[WelltrackRecord],
    selected_names_key: str,
    pending_names_key: str,
) -> list[str]:
    pilot_parent_keys = {
        well_name_key(parent_name_for_pilot(record.name))
        for record in records
        if is_pilot_name(record.name)
    }
    all_names = _unique_well_names(
        str(record.name)
        for record in records
        if str(record.name).strip()
        and not is_pilot_name(record.name)
        and ptc_target_records.record_horizontal_length_preprocess_skip_reason(
            record,
            has_pilot=pilot_parent_key_for_record(record) in pilot_parent_keys,
        )
        == "—"
    )
    pending_names = st.session_state.pop(pending_names_key, None)
    if pending_names is not None:
        st.session_state[selected_names_key] = _coerce_preprocess_selection(
            pending_names,
            all_names=all_names,
        )
    current = _coerce_preprocess_selection(
        st.session_state.get(selected_names_key, []),
        all_names=all_names,
    )
    if current != st.session_state.get(selected_names_key, []):
        st.session_state[selected_names_key] = list(current)
    if not current and all_names:
        st.session_state[selected_names_key] = list(all_names)
    return all_names


def _preprocess_excluded_records_message(
    records: list[WelltrackRecord],
) -> str:
    pilot_parent_keys = {
        well_name_key(parent_name_for_pilot(record.name))
        for record in records
        if is_pilot_name(record.name)
    }
    excluded_items: list[str] = []
    for record in records:
        if is_pilot_name(record.name):
            continue
        reason = ptc_target_records.record_horizontal_length_preprocess_skip_reason(
            record,
            has_pilot=pilot_parent_key_for_record(record) in pilot_parent_keys,
        )
        if reason == "—":
            continue
        excluded_items.append(f"{record.name} ({reason})")
    if not excluded_items:
        return ""
    preview = ", ".join(excluded_items[:6])
    suffix = "..." if len(excluded_items) > 6 else ""
    return f"Изменение длины ГС не применялось к: {preview}{suffix}."


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
            "по умолчанию выделяются нерассчитанные, ошибочные и скважины с предупреждениями."
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
        format_selected_calc_config_scope=_format_selected_calc_config_scope,
        build_selected_optimization_contexts=_build_selected_optimization_contexts,
        reference_wells_from_state=_reference_wells_from_state,
        reference_uncertainty_models_from_state=_reference_uncertainty_models_from_state,
        resolution_snapshot_well_names=_resolution_snapshot_well_names,
        format_prepared_override_scope=_format_prepared_override_scope,
        prepared_plan_kind_label=_prepared_plan_kind_label,
        build_last_anticollision_resolution=_build_last_anticollision_resolution,
        focus_all_wells_anticollision_results=_focus_all_wells_anticollision_results,
        focus_all_wells_trajectory_results=_focus_all_wells_trajectory_results,
        manual_override_signature=_manual_well_calc_override_signature,
        manual_override_signature_key=WT_LAST_WELL_CALC_OVERRIDE_SIGNATURE_KEY,
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


def _build_batch_survey_excel(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_survey_excel(
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


def _build_batch_target_csv(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_target_csv(
        records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs,
        transform_xy_func=transform_xy_to_crs,
        crs_display_suffix_func=get_crs_display_suffix,
        survey_export_dataframe_func=survey_export_dataframe,
    )


def _build_batch_target_excel(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_target_excel(
        records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        csv_export_crs_func=csv_export_crs,
        transform_xy_func=transform_xy_to_crs,
        crs_display_suffix_func=get_crs_display_suffix,
        survey_export_dataframe_func=survey_export_dataframe,
    )


def _build_batch_target_welltrack(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_target_welltrack(
        records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
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
        reference_wells=_reference_wells_from_state(),
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
    )


def _build_batch_survey_dev_files(
    successes: list[SuccessfulWellPlan],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> tuple[ptc_batch_results.DevExportFilePayload, ...]:
    return ptc_batch_results.build_batch_survey_dev_files(
        successes,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        reference_wells=_reference_wells_from_state(),
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
    )


def _build_batch_target_dev_7z(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_target_dev_7z(
        records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
    )


def _build_batch_target_dev_files(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> tuple[ptc_batch_results.DevExportFilePayload, ...]:
    return ptc_batch_results.build_batch_target_dev_files(
        records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
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
        reference_wells=_reference_wells_from_state(),
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
    )


def _build_batch_target_dev_file(
    records: list[WelltrackRecord],
    *,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_target_dev_file(
        records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
    )


def _build_batch_export_package_files(
    *,
    successes: list[SuccessfulWellPlan] | None = None,
    records: list[WelltrackRecord] | None = None,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> tuple[ptc_batch_results.ExportFilePayload, ...]:
    return ptc_batch_results.build_batch_export_package_files(
        successes=successes,
        records=records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        reference_wells=_reference_wells_from_state(),
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
        transform_xy_func=transform_xy_to_crs,
        crs_display_suffix_func=get_crs_display_suffix,
        survey_export_dataframe_func=survey_export_dataframe,
        dls_to_pi_func=dls_to_pi,
    )


def _build_batch_export_package_zip(
    *,
    successes: list[SuccessfulWellPlan] | None = None,
    records: list[WelltrackRecord] | None = None,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> bytes:
    return ptc_batch_results.build_batch_export_package_zip(
        successes=successes,
        records=records,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        reference_wells=_reference_wells_from_state(),
        csv_export_crs_func=csv_export_crs,
        transform_stations_func=transform_stations_to_crs,
        transform_xy_func=transform_xy_to_crs,
        crs_display_suffix_func=get_crs_display_suffix,
        survey_export_dataframe_func=survey_export_dataframe,
        dls_to_pi_func=dls_to_pi,
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
        build_batch_survey_excel_func=_build_batch_survey_excel,
        build_batch_survey_welltrack_func=_build_batch_survey_welltrack,
        build_batch_survey_dev_files_func=_build_batch_survey_dev_files,
        build_batch_survey_dev_7z_func=_build_batch_survey_dev_7z,
        build_batch_survey_dev_file_func=_build_batch_survey_dev_file,
        build_batch_target_csv_func=_build_batch_target_csv,
        build_batch_target_excel_func=_build_batch_target_excel,
        build_batch_target_welltrack_func=_build_batch_target_welltrack,
        build_batch_target_dev_files_func=_build_batch_target_dev_files,
        build_batch_target_dev_7z_func=_build_batch_target_dev_7z,
        build_batch_target_dev_file_func=_build_batch_target_dev_file,
        build_batch_export_package_files_func=_build_batch_export_package_files,
        build_batch_export_package_zip_func=_build_batch_export_package_zip,
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
