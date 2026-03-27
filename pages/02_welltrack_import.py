from __future__ import annotations

import colorsys
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pywp import TrajectoryConfig, TrajectoryPlanner
from pywp.actual_fund_analysis import (
    CALIBRATION_STATUS_READY,
    actual_fund_metrics_rows,
    actual_fund_pad_rows,
    build_actual_fund_well_metrics,
    calibrate_uncertainty_from_actual_fund,
)
from pywp.anticollision import (
    AntiCollisionAnalysis,
    anti_collision_method_caption,
    anti_collision_report_events,
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
    AntiCollisionRecommendationCluster,
    AntiCollisionRecommendation,
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
    build_anticollision_well_contexts as build_anticollision_well_contexts_shared,
    build_cluster_prepared_overrides as build_cluster_prepared_overrides_shared,
    build_prepared_optimization_context as build_prepared_optimization_context_shared,
    build_recommendation_prepared_overrides as build_recommendation_prepared_overrides_shared,
    recommendation_intervals_for_moving_well as recommendation_intervals_for_moving_well_shared,
)
from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackRecord,
    decode_welltrack_bytes,
    parse_welltrack_points_table,
    parse_welltrack_text,
    welltrack_points_to_targets,
)
from pywp.models import (
    OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
    OPTIMIZATION_MINIMIZE_KOP,
)
from pywp.reference_trajectories import (
    ImportedTrajectoryWell,
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    REFERENCE_WELL_KIND_COLORS,
    REFERENCE_WELL_KIND_LABELS,
    parse_reference_trajectory_text_with_kind,
    parse_reference_trajectory_welltrack_text,
    reference_well_display_label,
)
from pywp.plotly_config import DEFAULT_3D_CAMERA, trajectory_plotly_chart_config
from pywp.planner_config import optimization_display_label
from pywp.three_viewer import render_local_three_scene
from pywp.plot_axes import (
    equalized_axis_ranges,
    equalized_xy_ranges,
    linear_tick_values,
    nice_tick_step,
)
from pywp.solver_diagnostics import summarize_problem_ru
from pywp.solver_diagnostics_ui import render_solver_diagnostics
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND,
    UNCERTAINTY_PRESET_OPTIONS,
    PlanningUncertaintyModel,
    build_uncertainty_tube_mesh,
    normalize_uncertainty_preset,
    planning_uncertainty_model_for_preset,
    uncertainty_ribbon_polygon,
    uncertainty_preset_label,
)
from pywp.ui_calc_params import (
    CalcParamBinding,
)
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
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
from pywp.welltrack_quality import (
    detect_t1_t3_order_issues,
    swap_t1_t3_for_wells,
)
from pywp.well_pad import (
    PadLayoutPlan,
    WellPad,
    apply_pad_layout,
    detect_well_pads,
    ordered_pad_wells,
)
from pywp.welltrack_batch import (
    DynamicClusterExecutionContext,
    ensure_successful_plan_baseline,
    SuccessfulWellPlan,
    WelltrackBatchPlanner,
    merge_batch_results,
    rebuild_optimization_context,
    recommended_batch_selection,
)

DEFAULT_WELLTRACK_PATH = Path("tests/test_data/WELLTRACKS3.INC")
WT_UI_DEFAULTS_VERSION = 14
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
REFERENCE_LABEL_HORIZONTAL_INC_THRESHOLD_DEG = 80.0
REFERENCE_LABEL_HORIZONTAL_MIN_INTERVAL_M = 100.0
WT_3D_RENDER_OPTIONS: tuple[str, ...] = (
    WT_3D_RENDER_AUTO,
    WT_3D_RENDER_DETAIL,
    WT_3D_RENDER_FAST,
)
WT_3D_BACKEND_OPTIONS: tuple[str, ...] = (
    WT_3D_BACKEND_PLOTLY,
    WT_3D_BACKEND_THREE_LOCAL,
)
WT_3D_FAST_REFERENCE_WELL_THRESHOLD = 10
WT_3D_FAST_REFERENCE_POINT_THRESHOLD = 1200
WT_3D_FAST_CALC_WELL_THRESHOLD = 14
WT_3D_FAST_REFERENCE_TARGET_POINTS = 72
WT_3D_FAST_CALC_TARGET_POINTS = 180
WT_3D_FAST_REFERENCE_CONE_WELL_LIMIT = 6
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
_BATCH_SUMMARY_RENAME_COLUMNS: dict[str, str] = {
    "Рестарты решателя": "Рестарты",
    "Классификация целей": "Цели",
    "Горизонтальный отход t1, м": "Отход t1, м",
    "Длина HORIZONTAL, м": "HORIZONTAL, м",
}
_BATCH_SUMMARY_DISPLAY_ORDER: tuple[str, ...] = (
    "Скважина",
    "Точек",
    "Цели",
    "Сложность",
    "Отход t1, м",
    "KOP MD, м",
    "HORIZONTAL, м",
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


def _build_well_color_palette() -> tuple[str, ...]:
    """Return a long palette with high local contrast for adjacent wells.

    The hue order intentionally jumps around the wheel instead of walking
    sequentially, so neighboring indices remain visually distinct. Reds,
    blacks and grays are excluded on purpose to keep collision overlays and
    reference wells semantically separate.
    """

    hue_degrees: tuple[float, ...] = (
        24.0,
        48.0,
        72.0,
        96.0,
        120.0,
        144.0,
        168.0,
        192.0,
        216.0,
        240.0,
        264.0,
        288.0,
        312.0,
    )
    hue_jump_order: tuple[int, ...] = (0, 6, 12, 5, 11, 4, 10, 3, 9, 2, 8, 1, 7)
    lightness_saturation_bands: tuple[tuple[float, float], ...] = (
        (0.46, 0.82),
        (0.62, 0.74),
        (0.40, 0.88),
        (0.55, 0.90),
    )
    ordered_hues = tuple(hue_degrees[index] for index in hue_jump_order)
    colors: list[str] = []
    for index in range(len(ordered_hues) * len(lightness_saturation_bands)):
        hue_deg = ordered_hues[index % len(ordered_hues)]
        lightness, saturation = lightness_saturation_bands[index % len(lightness_saturation_bands)]
        red, green, blue = colorsys.hls_to_rgb(hue_deg / 360.0, lightness, saturation)
        colors.append(
            "#{:02X}{:02X}{:02X}".format(
                int(round(red * 255.0)),
                int(round(green * 255.0)),
                int(round(blue * 255.0)),
            )
        )
    return tuple(colors)


WELL_COLOR_PALETTE: tuple[str, ...] = _build_well_color_palette()


def _well_color(index: int) -> str:
    return WELL_COLOR_PALETTE[index % len(WELL_COLOR_PALETTE)]


def _well_color_map(records: list[WelltrackRecord]) -> dict[str, str]:
    return {
        str(record.name): _well_color(index) for index, record in enumerate(records)
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


def _reference_point_count(reference_wells: Iterable[ImportedTrajectoryWell]) -> int:
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
    if normalized == WT_3D_RENDER_DETAIL:
        return WT_3D_RENDER_DETAIL
    if normalized == WT_3D_RENDER_FAST:
        return WT_3D_RENDER_FAST
    reference_well_list = tuple(reference_wells)
    if calculated_well_count >= WT_3D_FAST_CALC_WELL_THRESHOLD:
        return WT_3D_RENDER_FAST
    if len(reference_well_list) >= WT_3D_FAST_REFERENCE_WELL_THRESHOLD:
        return WT_3D_RENDER_FAST
    if _reference_point_count(reference_well_list) >= WT_3D_FAST_REFERENCE_POINT_THRESHOLD:
        return WT_3D_RENDER_FAST
    return WT_3D_RENDER_DETAIL


def _decimated_station_frame(stations: pd.DataFrame, *, target_points: int) -> pd.DataFrame:
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
        if str(getattr(well, "kind", getattr(well, "well_kind", ""))) == str(kind)
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
        line={
            "width": 2.5,
            "color": REFERENCE_WELL_KIND_COLORS.get(str(kind), "#A0A0A0"),
        },
        hoverinfo="skip",
    )


def _anticollision_focus_reference_names(analysis: AntiCollisionAnalysis) -> set[str]:
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


def _t1_name_trace_3d(*, well_name: str, t1: Point3D, color: str) -> go.Scatter3d:
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
        points.append((float(anchor[0]), float(anchor[1]), float(anchor[2]), str(well.name)))
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
    for x_value, y_value, z_value in zip(x_array, y_array, z_array, strict=False):
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
    trace_opacity = (
        1.0 if getattr(trace, "opacity", None) is None else float(trace.opacity)
    )
    if "lines" in mode:
        line = trace.line or {}
        color, opacity = _plotly_color_and_opacity(
            getattr(line, "color", None),
            fallback_opacity=trace_opacity,
        )
        segments = _split_nan_separated_xyz_segments(
            x_values=() if trace.x is None else trace.x,
            y_values=() if trace.y is None else trace.y,
            z_values=() if trace.z is None else trace.z,
        )
        if segments:
            payload["lines"].append(
                {
                    "name": trace_name,
                    "segments": segments,
                    "color": color,
                    "opacity": opacity,
                    "dash": str(getattr(line, "dash", "solid") or "solid"),
                }
            )
            if _trace_showlegend(trace):
                payload["legend_items"].append(
                    {
                        "label": trace_name,
                        "color": color,
                        "opacity": opacity,
                    }
                )
    if "markers" in mode:
        marker = trace.marker or {}
        color, opacity = _plotly_color_and_opacity(
            getattr(marker, "color", None),
            fallback_opacity=trace_opacity,
        )
        if not (trace_name or opacity > 0.01):
            return payload
        x_array = np.asarray(list(() if trace.x is None else trace.x), dtype=float)
        y_array = np.asarray(list(() if trace.y is None else trace.y), dtype=float)
        z_array = np.asarray(list(() if trace.z is None else trace.z), dtype=float)
        mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
        points = [
            [float(x_value), float(y_value), float(z_value)]
            for x_value, y_value, z_value in zip(
                x_array[mask],
                y_array[mask],
                z_array[mask],
                strict=False,
            )
        ]
        if points:
            marker_size = getattr(marker, "size", 6)
            if isinstance(marker_size, (list, tuple, np.ndarray)):
                marker_size = marker_size[0] if len(marker_size) else 6
            payload["points"].append(
                {
                    "name": trace_name,
                    "points": points,
                    "color": color,
                    "opacity": opacity,
                    "size": float(marker_size),
                    "symbol": str(getattr(marker, "symbol", "circle") or "circle"),
                }
            )
            if _trace_showlegend(trace):
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
        x_array = np.asarray(list(() if trace.x is None else trace.x), dtype=float)
        y_array = np.asarray(list(() if trace.y is None else trace.y), dtype=float)
        z_array = np.asarray(list(() if trace.z is None else trace.z), dtype=float)
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
                }
            )
    return payload


def _mesh3d_trace_to_three_payload(trace: go.Mesh3d) -> dict[str, object] | None:
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
            for x_value, y_value, z_value in zip(x_array, y_array, z_array, strict=False)
        ],
        "faces": [
            [int(i_value), int(j_value), int(k_value)]
            for i_value, j_value, k_value in zip(i_array, j_array, k_array, strict=False)
        ],
        "color": color,
        "opacity": opacity,
    }
    return payload


def _optimize_three_payload(payload: dict[str, object]) -> dict[str, object]:
    optimized = dict(payload)
    optimized["lines"] = _merge_three_line_payloads(payload.get("lines") or [])
    optimized["points"] = _merge_three_point_payloads(payload.get("points") or [])
    optimized["meshes"] = _merge_three_mesh_payloads(payload.get("meshes") or [])
    optimized["labels"] = list(payload.get("labels") or [])
    optimized["legend"] = list(payload.get("legend") or [])
    return optimized


def _merge_three_line_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[list[list[float]]]] = {}
    ordered_keys: list[tuple[str, float, str]] = []
    for item in items:
        color = str(item.get("color") or "#0F172A")
        opacity = float(item.get("opacity") or 1.0)
        dash = str(item.get("dash") or "solid")
        key = (color, opacity, dash)
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
    for color, opacity, dash in ordered_keys:
        segments = grouped[(color, opacity, dash)]
        if not segments:
            continue
        merged.append(
            {
                "segments": segments,
                "color": color,
                "opacity": float(opacity),
                "dash": dash,
            }
        )
    return merged


def _merge_three_point_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, float, str], list[list[float]]] = {}
    ordered_keys: list[tuple[str, float, float, str]] = []
    for item in items:
        color = str(item.get("color") or "#0F172A")
        opacity = float(item.get("opacity") or 1.0)
        size = float(item.get("size") or 6.0)
        symbol = str(item.get("symbol") or "circle")
        key = (color, opacity, size, symbol)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].extend(
            [
                point
                for point in (item.get("points") or [])
                if isinstance(point, list) and len(point) == 3
            ]
        )
    merged: list[dict[str, object]] = []
    for color, opacity, size, symbol in ordered_keys:
        points = grouped[(color, opacity, size, symbol)]
        if not points:
            continue
        merged.append(
            {
                "points": points,
                "color": color,
                "opacity": float(opacity),
                "size": float(size),
                "symbol": symbol,
            }
        )
    return merged


def _merge_three_mesh_payloads(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], dict[str, object]] = {}
    ordered_keys: list[tuple[str, float]] = []
    for item in items:
        color = str(item.get("color") or "#94A3B8")
        opacity = float(item.get("opacity") or 1.0)
        key = (color, opacity)
        if key not in grouped:
            grouped[key] = {
                "vertices": [],
                "faces": [],
                "color": color,
                "opacity": float(opacity),
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
        "title": str(getattr(getattr(fig.layout, "title", None), "text", "") or ""),
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
) -> None:
    if str(backend) == WT_3D_BACKEND_THREE_LOCAL:
        with container:
            render_local_three_scene(
                _plotly_3d_figure_to_three_payload(figure),
                height=height,
            )
        return
    container.plotly_chart(
        figure,
        config=trajectory_plotly_chart_config(),
        width="stretch",
    )


def _build_anti_collision_analysis(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
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
                for column in ("MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m")
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
                for column in ("MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m")
                if column in reference_well.stations.columns
            ],
        ]
        digest.update(str(tuple(stations_subset.columns)).encode("utf-8"))
        digest.update(stations_subset.to_numpy(dtype=np.float64, copy=True).tobytes())
    return digest.hexdigest()


def _cached_anti_collision_view_model(
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    records: list[WelltrackRecord],
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
) -> tuple[
    AntiCollisionAnalysis,
    tuple[AntiCollisionRecommendation, ...],
    tuple[AntiCollisionRecommendationCluster, ...],
]:
    color_map = _well_color_map(records) if records else {}
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
            return analysis, recommendations, clusters
    analysis = _build_anti_collision_analysis(
        successes,
        model=uncertainty_model,
        name_to_color=color_map,
        reference_wells=reference_wells,
    )
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=_build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)
    st.session_state["wt_anticollision_analysis_cache"] = {
        "key": cache_key,
        "analysis": analysis,
        "recommendations": recommendations,
        "clusters": clusters,
    }
    return analysis, recommendations, clusters


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
    if end_md <= start_md + 1e-9:
        return None

    interior_mask = (md_values > start_md + 1e-9) & (md_values < end_md - 1e-9)
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
    dls_pi_values = dls_to_pi(
        stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float)
    )
    segment_values = (
        stations["segment"].astype(str).to_numpy(dtype=object)
        if "segment" in stations.columns
        else np.full(len(stations), "—", dtype=object)
    )
    customdata = np.empty((len(stations), 3), dtype=object)
    customdata[:, 0] = stations["MD_m"].to_numpy(dtype=float)
    customdata[:, 1] = dls_pi_values
    customdata[:, 2] = segment_values
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
            "size": 6,
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
            {"Wellname": "", "Point": "", "X": np.nan, "Y": np.nan, "Z": np.nan},
            {"Wellname": "", "Point": "", "X": np.nan, "Y": np.nan, "Z": np.nan},
            {"Wellname": "", "Point": "", "X": np.nan, "Y": np.nan, "Z": np.nan},
        ]
    )


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


def _clear_actual_fund_calibration_state() -> None:
    st.session_state["wt_actual_fund_calibration_result"] = None
    st.session_state["wt_actual_fund_custom_model"] = None
    if (
        str(st.session_state.get("wt_anticollision_uncertainty_preset", "")).strip()
        == UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND
    ):
        st.session_state["wt_anticollision_uncertainty_preset"] = DEFAULT_UNCERTAINTY_PRESET


def _actual_fund_custom_model_from_state() -> PlanningUncertaintyModel | None:
    model = st.session_state.get("wt_actual_fund_custom_model")
    return model if isinstance(model, PlanningUncertaintyModel) else None


def _set_reference_wells_for_kind(
    *,
    kind: str,
    wells: Iterable[ImportedTrajectoryWell],
) -> None:
    normalized_kind = str(kind)
    key = _reference_wells_state_key(normalized_kind)
    st.session_state[key] = tuple(wells)
    actual_wells = tuple(
        st.session_state.get(_reference_wells_state_key(REFERENCE_WELL_ACTUAL)) or ()
    )
    approved_wells = tuple(
        st.session_state.get(_reference_wells_state_key(REFERENCE_WELL_APPROVED)) or ()
    )
    st.session_state["wt_reference_wells"] = tuple(actual_wells) + tuple(approved_wells)
    if normalized_kind == REFERENCE_WELL_ACTUAL:
        _clear_actual_fund_calibration_state()


def _reference_wells_from_state() -> tuple[ImportedTrajectoryWell, ...]:
    actual_wells = tuple(
        st.session_state.get(_reference_wells_state_key(REFERENCE_WELL_ACTUAL)) or ()
    )
    approved_wells = tuple(
        st.session_state.get(_reference_wells_state_key(REFERENCE_WELL_APPROVED)) or ()
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
        st.session_state[_reference_wells_state_key(REFERENCE_WELL_ACTUAL)] = actual_legacy
        st.session_state[_reference_wells_state_key(REFERENCE_WELL_APPROVED)] = approved_legacy
    return legacy_combined


def _reset_anticollision_view_state(*, clear_prepared: bool) -> None:
    st.session_state["wt_anticollision_analysis_cache"] = {}
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
    st.session_state.setdefault(_reference_source_text_key(REFERENCE_WELL_ACTUAL), "")
    st.session_state.setdefault(_reference_source_text_key(REFERENCE_WELL_APPROVED), "")
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

    if int(st.session_state.get("wt_ui_defaults_version", 0)) < WT_UI_DEFAULTS_VERSION:
        _apply_profile_defaults(force=True)
        st.session_state["wt_ui_defaults_version"] = WT_UI_DEFAULTS_VERSION

    st.session_state.setdefault("wt_records", None)
    st.session_state.setdefault("wt_records_original", None)
    st.session_state.setdefault("wt_reference_wells", ())
    st.session_state.setdefault("wt_selected_names", [])
    st.session_state.setdefault("wt_pending_selected_names", None)
    st.session_state.setdefault("wt_loaded_at", "")
    st.session_state.setdefault("wt_pad_configs", {})
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
    st.session_state.setdefault("wt_results_view_mode", "Отдельная скважина")
    st.session_state.setdefault("wt_results_all_view_mode", "Траектории")
    st.session_state.setdefault("wt_3d_render_mode", WT_3D_RENDER_AUTO)
    st.session_state.setdefault("wt_3d_backend", WT_3D_BACKEND_PLOTLY)
    st.session_state.setdefault(
        "wt_anticollision_uncertainty_preset", DEFAULT_UNCERTAINTY_PRESET
    )
    st.session_state.setdefault("wt_actual_fund_analysis_view_mode", "По скважинам")
    st.session_state.setdefault("wt_actual_fund_base_preset", DEFAULT_UNCERTAINTY_PRESET)
    st.session_state.setdefault("wt_actual_fund_calibration_result", None)
    st.session_state.setdefault("wt_actual_fund_custom_model", None)
    if str(st.session_state.get("wt_results_view_mode", "")).strip() not in {
        "Отдельная скважина",
        "Все скважины",
    }:
        st.session_state["wt_results_view_mode"] = "Отдельная скважина"
    if str(st.session_state.get("wt_results_all_view_mode", "")).strip() not in {
        "Траектории",
        "Anti-collision",
    }:
        st.session_state["wt_results_all_view_mode"] = "Траектории"
    if str(st.session_state.get("wt_3d_render_mode", "")).strip() not in set(WT_3D_RENDER_OPTIONS):
        st.session_state["wt_3d_render_mode"] = WT_3D_RENDER_AUTO
    if str(st.session_state.get("wt_3d_backend", "")).strip() not in set(WT_3D_BACKEND_OPTIONS):
        st.session_state["wt_3d_backend"] = WT_3D_BACKEND_PLOTLY
    st.session_state["wt_anticollision_uncertainty_preset"] = (
        normalize_uncertainty_preset(
            st.session_state.get(
                "wt_anticollision_uncertainty_preset",
                DEFAULT_UNCERTAINTY_PRESET,
            ),
            allow_custom=_actual_fund_custom_model_from_state() is not None,
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


def _clear_results() -> None:
    st.session_state["wt_summary_rows"] = None
    st.session_state["wt_successes"] = None
    st.session_state["wt_pending_selected_names"] = None
    st.session_state["wt_last_error"] = ""
    st.session_state["wt_last_run_at"] = ""
    st.session_state["wt_last_runtime_s"] = None
    st.session_state["wt_last_run_log_lines"] = []
    st.session_state["wt_results_view_mode"] = "Отдельная скважина"
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


def _clear_pad_state() -> None:
    st.session_state["wt_pad_configs"] = {}
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
        st.session_state.setdefault("wt_cfg_vertical_tol", float(tolerance_legacy_value))
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
        return _decode_welltrack_payload(payload, source_label=f"Файл `{file_path}`")
    except OSError as exc:
        st.error(f"Не удалось прочитать файл `{file_path}`: {exc}")
        return ""


def _all_wells_3d_figure(
    successes: list[SuccessfulWellPlan],
    *,
    target_only_wells: list[_TargetOnlyWell] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    name_to_color: dict[str, str] | None = None,
    render_mode: str = WT_3D_RENDER_AUTO,
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
    color_map = name_to_color or {
        str(item.name): _well_color(index) for index, item in enumerate(successes)
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
                customdata=np.column_stack(
                    [
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
                    "Z/TVD: %{z:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m<br>"
                    "ПИ: %{customdata[1]:.2f} deg/10m"
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
        x_focus_arrays.append(np.array([surface.x, t1.x, t3.x], dtype=float))
        y_focus_arrays.append(np.array([surface.y, t1.y, t3.y], dtype=float))
        z_focus_arrays.append(np.array([surface.z, t1.z, t3.z], dtype=float))
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
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label_trace = _reference_name_trace_3d(reference_wells, kind=kind)
        if label_trace is not None:
            fig.add_trace(label_trace)

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
        np.concatenate(x_focus_arrays) if x_focus_arrays else np.concatenate(x_arrays)
    ) if x_arrays else np.array([0.0], dtype=float)
    y_values = (
        np.concatenate(y_focus_arrays) if y_focus_arrays else np.concatenate(y_arrays)
    ) if y_arrays else np.array([0.0], dtype=float)
    z_values = (
        np.concatenate(z_focus_arrays) if z_focus_arrays else np.concatenate(z_arrays)
    ) if z_arrays else np.array([0.0], dtype=float)
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
    height: int = 560,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
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
        x_focus_arrays.append(stations["X_m"].to_numpy(dtype=float))
        y_focus_arrays.append(stations["Y_m"].to_numpy(dtype=float))
        fig.add_trace(
            go.Scatter(
                x=stations["X_m"],
                y=stations["Y_m"],
                mode="lines",
                name=name,
                line={"width": 4, "color": line_color, "dash": line_dash},
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
                line={"width": 3.5, "color": line_color},
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
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label_trace = _reference_name_trace_2d(reference_wells, kind=kind)
        if label_trace is not None:
            fig.add_trace(label_trace)

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
        np.concatenate(x_focus_arrays) if x_focus_arrays else np.concatenate(x_arrays)
    ) if x_arrays else np.array([0.0], dtype=float)
    y_values = (
        np.concatenate(y_focus_arrays) if y_focus_arrays else np.concatenate(y_arrays)
    ) if y_arrays else np.array([0.0], dtype=float)
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


def _all_wells_anticollision_3d_figure(
    analysis: AntiCollisionAnalysis,
    *,
    previous_successes_by_name: Mapping[str, SuccessfulWellPlan] | None = None,
    render_mode: str = WT_3D_RENDER_AUTO,
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
            [well for well in analysis.wells if not bool(well.is_reference_only)]
        ),
        reference_wells=reference_wells,
    )
    focus_reference_names = (
        set(
            sorted(_anticollision_focus_reference_names(analysis))[
                :WT_3D_FAST_REFERENCE_CONE_WELL_LIMIT
            ]
        )
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
        if tube_mesh is not None:
            x_arrays.append(tube_mesh.vertices_xyz[:, 0])
            y_arrays.append(tube_mesh.vertices_xyz[:, 1])
            z_arrays.append(tube_mesh.vertices_xyz[:, 2])
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
            terminal_ring = np.asarray(overlay.samples[-1].ring_xyz, dtype=float)
            x_arrays.append(terminal_ring[:, 0])
            y_arrays.append(terminal_ring[:, 1])
            z_arrays.append(terminal_ring[:, 2])
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
                        "width": 1.5,
                        "color": _lighten_hex(str(well.color)),
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
        md_values = stations["MD_m"].to_numpy(dtype=float)
        x_arrays.append(x_values)
        y_arrays.append(y_values)
        z_arrays.append(z_values)
        if not is_reference_only:
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
                line={"width": 5, "color": str(well.color)},
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{z:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m<br>"
                    "ПИ: %{customdata[1]:.2f} deg/10m<br>"
                    "Сегмент: %{customdata[2]}"
                    "<extra>%{fullData.name}</extra>"
                ),
                customdata=_trajectory_hover_customdata(stations),
            )
        )
        previous_success = (previous_successes_by_name or {}).get(str(well.name))
        if previous_success is not None and not previous_success.stations.empty:
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
                        "ПИ: %{customdata[1]:.2f} deg/10m<br>"
                        "Сегмент: %{customdata[2]}"
                        f"<extra>{well_label}</extra>"
                    ),
                )
            )
        if (well.t1 is not None) and (well.t3 is not None) and not is_reference_only:
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
            x_arrays.append(np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float))
            y_arrays.append(np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float))
            z_arrays.append(np.array([well.surface.z, well.t1.z, well.t3.z], dtype=float))
            x_focus_arrays.append(np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float))
            y_focus_arrays.append(np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float))
            z_focus_arrays.append(np.array([well.surface.z, well.t1.z, well.t3.z], dtype=float))

    if aggregated_reference_wells:
        for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
            combined_trace = _combined_reference_trace_3d(
                reference_wells=[
                    well for well in aggregated_reference_wells if str(well.well_kind) == str(kind)
                ],
                kind=kind,
                target_points_per_well=WT_3D_FAST_REFERENCE_TARGET_POINTS,
            )
            if combined_trace is not None:
                fig.add_trace(combined_trace)
                x_arrays.append(np.asarray(combined_trace.x, dtype=float))
                y_arrays.append(np.asarray(combined_trace.y, dtype=float))
                z_arrays.append(np.asarray(combined_trace.z, dtype=float))
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label_trace = _reference_name_trace_3d(
            [
                ImportedTrajectoryWell(
                    name=str(well.name),
                    kind=str(well.well_kind),
                    stations=well.stations,
                    surface=well.surface,
                    azimuth_deg=0.0,
                )
                for well in analysis.wells
                if bool(well.is_reference_only)
            ],
            kind=kind,
        )
        if label_trace is not None:
            fig.add_trace(label_trace)

    overlap_legend_added = False
    for corridor in analysis.corridors:
        mesh = collision_corridor_tube_mesh(corridor)
        if mesh is not None:
            x_arrays.append(mesh.vertices_xyz[:, 0])
            y_arrays.append(mesh.vertices_xyz[:, 1])
            z_arrays.append(mesh.vertices_xyz[:, 2])
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
            sphere_x, sphere_y, sphere_z = collision_corridor_point_sphere_mesh(
                corridor
            )
            x_arrays.append(sphere_x.reshape(-1))
            y_arrays.append(sphere_y.reshape(-1))
            z_arrays.append(sphere_z.reshape(-1))
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
            dls_to_pi(
                well.stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float)
            ),
        )
        conflict_customdata = np.column_stack([md_segment, dls_segment])
        x_arrays.append(x_segment)
        y_arrays.append(y_segment)
        z_arrays.append(z_segment)
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
                    "ПИ: %{customdata[1]:.2f} deg/10m"
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
                    "ПИ: %{customdata[1]:.2f} deg/10m"
                    "<extra>Конфликтный участок ствола</extra>"
                ),
            )
        )
        segment_legend_added = True

    x_values = (
        np.concatenate(x_focus_arrays) if x_focus_arrays else np.concatenate(x_arrays)
    ) if x_arrays else np.array([0.0], dtype=float)
    y_values = (
        np.concatenate(y_focus_arrays) if y_focus_arrays else np.concatenate(y_arrays)
    ) if y_arrays else np.array([0.0], dtype=float)
    z_values = (
        np.concatenate(z_focus_arrays) if z_focus_arrays else np.concatenate(z_arrays)
    ) if z_arrays else np.array([0.0], dtype=float)
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
            "xaxis": {"range": x_range, "tickvals": x_tickvals, **xy_axis_style},
            "yaxis": {"range": y_range, "tickvals": y_tickvals, **xy_axis_style},
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
    height: int = 620,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
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
            if not bool(well.is_reference_only):
                x_focus_arrays.append(ribbon[:, 0])
                y_focus_arrays.append(ribbon[:, 1])

        stations = well.stations
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        md_values = stations["MD_m"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=well_label,
                legendgroup=str(well.name),
                line={"width": 4, "color": str(well.color)},
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
                customdata=np.column_stack([md_values]),
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
            x_focus_arrays.append(previous_x)
            y_focus_arrays.append(previous_y)
        if (well.t1 is not None) and (well.t3 is not None) and not bool(well.is_reference_only):
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
            x_arrays.append(np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float))
            y_arrays.append(np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float))
            x_focus_arrays.append(np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float))
            y_focus_arrays.append(np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float))

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
        x_focus_arrays.append(x_segment)
        y_focus_arrays.append(y_segment)
        segment_legend_added = True
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label_trace = _reference_name_trace_2d(
            [
                ImportedTrajectoryWell(
                    name=str(well.name),
                    kind=str(well.well_kind),
                    stations=well.stations,
                    surface=well.surface,
                    azimuth_deg=0.0,
                )
                for well in analysis.wells
                if bool(well.is_reference_only)
            ],
            kind=kind,
        )
        if label_trace is not None:
            fig.add_trace(label_trace)

    x_values = (
        np.concatenate(x_focus_arrays) if x_focus_arrays else np.concatenate(x_arrays)
    ) if x_arrays else np.array([0.0], dtype=float)
    y_values = (
        np.concatenate(y_focus_arrays) if y_focus_arrays else np.concatenate(y_arrays)
    ) if y_arrays else np.array([0.0], dtype=float)
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


def _render_anticollision_panel(successes: list[SuccessfulWellPlan]) -> None:
    reference_wells = _reference_wells_from_state()
    if len(successes) + len(reference_wells) < 2:
        st.info("Для anti-collision нужно минимум две успешно рассчитанные скважины.")
        return

    custom_actual_fund_model = _actual_fund_custom_model_from_state()
    preset_options = list(UNCERTAINTY_PRESET_OPTIONS.keys())
    if custom_actual_fund_model is not None:
        preset_options.append(UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND)
    normalized_preset = normalize_uncertainty_preset(
        st.session_state.get(
            "wt_anticollision_uncertainty_preset",
            DEFAULT_UNCERTAINTY_PRESET,
        ),
        allow_custom=custom_actual_fund_model is not None,
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
    uncertainty_model = planning_uncertainty_model_for_preset(
        selected_preset,
        custom_model=custom_actual_fund_model,
    )
    records = list(st.session_state.get("wt_records") or [])
    analysis, recommendations, clusters = _cached_anti_collision_view_model(
        successes=successes,
        uncertainty_model=uncertainty_model,
        records=records,
        reference_wells=reference_wells,
    )
    previous_successes_by_name = {
        str(name): value
        for name, value in (
            st.session_state.get("wt_last_anticollision_previous_successes") or {}
        ).items()
    }
    st.caption(
        f"Пресет: {uncertainty_preset_label(selected_preset)}. "
        f"{anti_collision_method_caption(uncertainty_model)}"
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
    if str(selected_3d_backend) == WT_3D_BACKEND_THREE_LOCAL:
        st.caption(
            "Активен локальный Three.js viewer: 3D-смысл сцены сохраняется, "
            "но детальные Plotly-hover подсказки доступны только в режиме Plotly."
        )

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Проверено пар", f"{int(analysis.pair_count)}")
    m2.metric("Пар с overlap", f"{int(analysis.overlapping_pair_count)}")
    m3.metric("Пересечения в t1/t3", f"{int(analysis.target_overlap_pair_count)}")
    worst_sf = analysis.worst_separation_factor
    m4.metric("Минимальный SF", "—" if worst_sf is None else f"{float(worst_sf):.2f}")
    with st.expander("Что такое SF?", expanded=False):
        st.markdown(_sf_help_markdown())

    chart_col1, chart_col2 = st.columns(2, gap="medium")
    anticollision_3d_figure = _all_wells_anticollision_3d_figure(
        analysis,
        previous_successes_by_name=previous_successes_by_name,
        render_mode=selected_render_mode,
    )
    _render_plotly_or_three_3d(
        container=chart_col1,
        figure=anticollision_3d_figure,
        backend=selected_3d_backend,
        height=660,
    )
    chart_col2.plotly_chart(
        _all_wells_anticollision_plan_figure(
            analysis,
            previous_successes_by_name=previous_successes_by_name,
        ),
        width="stretch",
    )
    _render_last_anticollision_resolution(current_preset=selected_preset)

    if not analysis.zones:
        st.success(
            "Пересечения 2σ конусов неопределенности не обнаружены для рассчитанного набора."
        )
        return

    target_zones = [zone for zone in analysis.zones if int(zone.priority_rank) < 2]
    if target_zones:
        st.warning(
            "Найдены пересечения, затрагивающие точки целей t1/t3. Они вынесены "
            "в начало отчета и должны разбираться в первую очередь."
        )
    else:
        st.warning("Найдены пересечения 2σ конусов неопределенности по траекториям.")

    report_events = anti_collision_report_events(analysis)
    report_df = arrow_safe_text_dataframe(
        pd.DataFrame(anti_collision_report_rows(analysis))
    )
    st.markdown("### Отчет по anti-collision")
    st.caption(
        "Смежные и пересекающиеся corridor-интервалы одной и той же collision природы "
        f"в отчете объединяются в одно событие. Всего событий: {len(report_events)}."
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
            "Смежных зон": st.column_config.NumberColumn("Смежных зон", format="%d"),
        },
    )

    recommendation_df = arrow_safe_text_dataframe(
        pd.DataFrame(anti_collision_recommendation_rows(recommendations))
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
            "Ожидаемый маневр": st.column_config.TextColumn("Ожидаемый маневр"),
            "Рекомендация": st.column_config.TextColumn("Рекомендация"),
            "Подготовка пересчета": st.column_config.TextColumn("Подготовка пересчета"),
        },
    )

    cluster_df = arrow_safe_text_dataframe(
        pd.DataFrame(anti_collision_cluster_rows(clusters))
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
            "Траектория": st.column_config.NumberColumn("Траектория", format="%d"),
            "SF min": st.column_config.NumberColumn("SF min", format="%.2f"),
            "Ожидаемый маневр": st.column_config.TextColumn("Ожидаемый маневр"),
            "Стартовый шаг": st.column_config.TextColumn("Стартовый шаг"),
            "Порядок": st.column_config.TextColumn("Порядок"),
            "К пересчету": st.column_config.TextColumn("К пересчету"),
            "Подготовка пересчета": st.column_config.TextColumn("Подготовка пересчета"),
        },
    )

    st.info(
        "Как использовать подготовку пересчета: "
        "`Подготовить одно событие` применяйте для локального отдельного конфликта. "
        "`Подготовить весь кластер` применяйте, когда одна и та же скважина участвует "
        "в нескольких связанных конфликтах. В каждый момент активен только один "
        "подготовленный план: новая подготовка заменяет предыдущую."
    )

    actionable_clusters = [item for item in clusters if bool(item.can_prepare_rerun)]
    if actionable_clusters:
        cluster_ids = [item.cluster_id for item in actionable_clusters]
        if (
            str(st.session_state.get("wt_anticollision_prepared_cluster_id", ""))
            not in cluster_ids
        ):
            st.session_state["wt_anticollision_prepared_cluster_id"] = cluster_ids[0]
        cluster_select_col, cluster_button_col = st.columns([6.0, 1.8], gap="small")
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
                    )
                st.toast(
                    "Подготовлен план пересчета для всего связанного кластера. "
                    "Он заменил предыдущий prepared plan."
                )
                st.rerun()

    actionable_recommendations = [
        item for item in recommendations if bool(item.can_prepare_rerun)
    ]
    if actionable_recommendations:
        actionable_ids = [item.recommendation_id for item in actionable_recommendations]
        if (
            str(st.session_state.get("wt_anticollision_prepared_recommendation_id", ""))
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
                    if str(item.recommendation_id) == str(selected_recommendation_id)
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


def _build_selected_override_configs(
    *,
    base_config: TrajectoryConfig,
    selected_names: set[str],
) -> dict[str, TrajectoryConfig]:
    prepared = st.session_state.get("wt_prepared_well_overrides", {}) or {}
    config_map: dict[str, TrajectoryConfig] = {}
    for well_name in sorted(str(name) for name in selected_names):
        payload = dict(prepared.get(well_name, {}))
        update_fields = dict(payload.get("update_fields", {}))
        if not update_fields:
            continue
        config_map[well_name] = base_config.validated_copy(**update_fields)
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
        "affected_wells": tuple(str(name) for name in recommendation.affected_wells),
        "action_label": str(recommendation.action_label),
        "source_label": recommendation_display_label(recommendation),
    }


def _cluster_snapshot(
    cluster: AntiCollisionRecommendationCluster,
) -> dict[str, object]:
    items = tuple(_recommendation_snapshot(item) for item in cluster.recommendations)
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
        "recommendation_count": int(cluster.recommendation_count),
        "before_sf": float(before_sf),
        "rerun_order_label": str(cluster.rerun_order_label),
        "first_rerun_well": (
            None if cluster.first_rerun_well is None else str(cluster.first_rerun_well)
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
                "related_recommendation_count": int(step.related_recommendation_count),
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
            min(evaluation_a.min_separation_factor, evaluation_b.min_separation_factor)
        ),
        float(max(evaluation_a.max_overlap_depth_m, evaluation_b.max_overlap_depth_m)),
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


def _resolution_snapshot_well_names(snapshot: dict[str, object]) -> tuple[str, ...]:
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
        "Пара", f"{resolution.get('well_a', '—')} ↔ {resolution.get('well_b', '—')}"
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
) -> tuple[dict[str, dict[str, object]], list[str], tuple[dict[str, object], ...]]:
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
    snapshot = _cluster_snapshot(cluster)
    st.session_state["wt_prepared_well_overrides"] = prepared
    st.session_state["wt_prepared_recommendation_snapshot"] = (
        snapshot if prepared else None
    )
    if prepared:
        message = str(cluster.summary)
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
            if str(step.well_name) in prepared
        ]
        st.session_state["wt_pending_selected_names"] = ordered_wells or list(
            prepared.keys()
        )
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
        options=["Файл по пути", "Загрузить файл", "Вставить текст", "Вставить таблицу"],
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
        note_col, clear_col = st.columns([5.0, 1.2], gap="small", vertical_alignment="bottom")
        with note_col:
            st.caption(
                "Вставьте таблицу в формате `Wellname`, `Point`, `X`, `Y`, `Z`. "
                "Поддерживается copy/paste из Excel, Google Sheets и похожих таблиц. "
                "Point принимает `wellhead`, `t1`, `t3`."
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
        edited_table = st.data_editor(
            st.session_state.get("wt_source_table_df", _empty_source_table_df()),
            key=f"wt_source_table_editor_{int(st.session_state.get('wt_source_table_editor_nonce', 0))}",
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Wellname": st.column_config.TextColumn("Wellname"),
                "Point": st.column_config.SelectboxColumn(
                    "Point",
                    options=["wellhead", "t1", "t3"],
                ),
                "X": st.column_config.NumberColumn("X"),
                "Y": st.column_config.NumberColumn("Y"),
                "Z": st.column_config.NumberColumn("Z"),
            },
        )
        st.session_state["wt_source_table_df"] = pd.DataFrame(edited_table)

    return _WelltrackSourcePayload(
        mode=source_mode,
        table_rows=pd.DataFrame(st.session_state["wt_source_table_df"]),
    )


def _store_parsed_records(records: list[WelltrackRecord]) -> bool:
    all_names = [record.name for record in records]
    st.session_state["wt_records"] = list(records)
    st.session_state["wt_records_original"] = list(records)
    st.session_state["wt_loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _clear_pad_state()
    st.session_state["wt_last_error"] = ""
    _clear_results()
    auto_layout_applied = _auto_apply_pad_layout_if_shared_surface(
        records=list(records)
    )
    st.session_state["wt_selected_names"] = list(all_names)
    return auto_layout_applied


def _auto_apply_pad_layout_if_shared_surface(records: list[WelltrackRecord]) -> bool:
    pads = detect_well_pads(records)
    well_count_with_surface = sum(1 for record in records if record.points)
    if well_count_with_surface <= 1:
        return False
    if len(pads) != 1:
        return False
    if len(pads[0].wells) != well_count_with_surface:
        return False

    pads = _ensure_pad_configs(base_records=list(records))
    plan_map = _build_pad_plan_map(pads)
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


def _render_import_controls() -> tuple[_WelltrackSourcePayload, bool, bool, bool]:
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
        st.session_state[_reference_wells_state_key(REFERENCE_WELL_ACTUAL)] = ()
        st.session_state[_reference_wells_state_key(REFERENCE_WELL_APPROVED)] = ()
        _clear_actual_fund_calibration_state()
        st.session_state["wt_selected_names"] = []
        st.session_state["wt_loaded_at"] = ""
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
        with st.status("Чтение и преобразование таблицы точек...", expanded=True) as status:
            started = perf_counter()
            try:
                status.write("Проверка строк таблицы и сборка точек wellhead / t1 / t3.")
                records = parse_welltrack_points_table(
                    pd.DataFrame(table_rows).to_dict(orient="records")
                )
                auto_layout_applied = _store_parsed_records(records=records)
                status.write(f"Собрано скважин из таблицы: {len(records)}.")
                if auto_layout_applied:
                    status.write(
                        "Обнаружен общий исходный устьевой S: устья автоматически "
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
        st.warning("Источник пустой. Загрузите файл или вставьте текст WELLTRACK.")
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
                    "Обнаружен общий исходный устьевой S: устья автоматически "
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
            _clear_pad_state()
            st.session_state["wt_last_error"] = str(exc)
            status.write(str(exc))
            status.update(
                label="Ошибка парсинга WELLTRACK", state="error", expanded=True
            )


def _render_records_overview(records: list[WelltrackRecord]) -> None:
    ready_count = sum(1 for record in records if len(record.points) == 3)
    x1, x2, x3 = st.columns(3, gap="small")
    x1.metric("Скважин в файле", f"{len(records)}")
    x2.metric("Готово к расчету", f"{ready_count}")
    x3.metric("Загружено", st.session_state.get("wt_loaded_at", "—"))

    parsed_df = pd.DataFrame(
        [
            {
                "Скважина": record.name,
                "Точек": len(record.points),
                "Готова к расчету (3 точки)": len(record.points) == 3,
            }
            for record in records
        ]
    )
    st.markdown("### Загруженные скважины")
    st.dataframe(arrow_safe_text_dataframe(parsed_df), width="stretch", hide_index=True)


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
    st.caption(_reference_kind_help(kind))

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

    action_col, clear_col = st.columns([1.5, 1.0], gap="small")
    import_clicked = action_col.button(
        f"Импортировать {title.lower()}",
        key=f"wt_reference_import_{kind}",
        type="primary",
        icon=":material/upload_file:",
        width="stretch",
    )
    clear_clicked = clear_col.button(
        f"Очистить {title.lower()}",
        key=f"wt_reference_clear_{kind}",
        icon=":material/delete:",
        width="stretch",
    )

    if import_clicked:
        with st.status(f"Импорт {title.lower()}...", expanded=True) as status:
            started = perf_counter()
            try:
                if mode == "Вставить XYZ/MD текст":
                    parsed = parse_reference_trajectory_text_with_kind(
                        str(st.session_state.get(_reference_source_text_key(kind), "")),
                        default_kind=kind,
                    )
                elif mode == "Загрузить XYZ/MD файл":
                    payload = b"" if uploaded_xyz_file is None else uploaded_xyz_file.getvalue()
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
                            str(st.session_state.get(_reference_welltrack_path_key(kind), ""))
                        ),
                        kind=kind,
                    )
                else:
                    payload = (
                        b"" if uploaded_welltrack_file is None else uploaded_welltrack_file.getvalue()
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


def _render_actual_fund_analysis_panel() -> None:
    actual_wells = _reference_kind_wells(REFERENCE_WELL_ACTUAL)
    if not actual_wells:
        return

    metrics = build_actual_fund_well_metrics(actual_wells)
    eligible_metrics = [item for item in metrics if bool(item.is_analysis_eligible)]
    excluded_horizontal_metrics = [
        item for item in metrics if bool(item.is_horizontal) and not bool(item.is_analysis_eligible)
    ]
    pad_count = len({str(item.pad_group) for item in eligible_metrics if str(item.pad_group) != "—"})
    eligible_kop_values = [
        float(item.kop_md_m)
        for item in eligible_metrics
        if item.kop_md_m is not None
    ]

    with st.expander("Анализ фактического фонда", expanded=False):
        st.caption(
            "В анализ попадают только фактические скважины. Для калибровки "
            "учитываются только горизонтальные скважины без аномалий: если у "
            "горизонтальной скважины не удаётся устойчиво выделить `KOP / HOLD / "
            "терминальный high-angle interval` или получается аномально высокий "
            "устойчивый `ПИ`, такая скважина исключается из анализа и калибровки. "
            "Пары одного семейства вида `7401` / `7401_PL` / `7401_2` из calibration-scan "
            "исключаются. Группировка по кустам строится по ведущему числовому коду "
            "скважины без двух последних цифр: `6101/6102/6103 -> 61`, `8203/8210 -> 82`."
        )
        a1, a2, a3, a4 = st.columns(4, gap="small")
        a1.metric("Фактических скважин", f"{len(actual_wells)}")
        a2.metric("В анализе", f"{len(eligible_metrics)}")
        a3.metric("Кустов", f"{pad_count}")
        a4.metric(
            "Медианный KOP, м",
            "—" if not eligible_kop_values else f"{float(np.median(eligible_kop_values)):.0f}",
        )
        if excluded_horizontal_metrics:
            st.info(
                "Из анализа исключено горизонтальных скважин: "
                f"{len(excluded_horizontal_metrics)}. Причины видны в таблице."
            )

        view_mode = st.radio(
            "Свод фактического фонда",
            options=["По скважинам", "По кустам"],
            key="wt_actual_fund_analysis_view_mode",
            horizontal=True,
        )
        if str(view_mode) == "По кустам":
            df = pd.DataFrame(actual_fund_pad_rows(metrics))
        else:
            df = pd.DataFrame(actual_fund_metrics_rows(metrics))
        st.dataframe(
            arrow_safe_text_dataframe(df),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Калибровка пользовательской функции конусов")
        st.caption(
            "Это не formal ISCWSA toolcode calibration, а planning-level field fit: "
            "единая эмпирическая шкала, которая уменьшает базовый пресет так, чтобы "
            "ложные overlap по фактическому горизонтальному фонду не доминировали."
        )
        calibration_base_preset = st.selectbox(
            "Базовый пресет для калибровки",
            options=list(UNCERTAINTY_PRESET_OPTIONS.keys()),
            format_func=uncertainty_preset_label,
            key="wt_actual_fund_base_preset",
        )
        if st.button(
            "Откалибровать пользовательскую функцию",
            icon=":material/tune:",
            width="stretch",
        ):
            result = calibrate_uncertainty_from_actual_fund(
                actual_wells=actual_wells,
                base_model=planning_uncertainty_model_for_preset(calibration_base_preset),
                base_preset=calibration_base_preset,
            )
            st.session_state["wt_actual_fund_calibration_result"] = result
            st.session_state["wt_actual_fund_custom_model"] = result.custom_model
            if result.custom_model is not None:
                st.session_state["wt_anticollision_uncertainty_preset"] = (
                    UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND
                )
                _reset_anticollision_view_state(clear_prepared=False)
                st.toast(
                    "Пользовательская функция конусов построена и выбрана в anti-collision."
                )
            st.rerun()

        calibration_result = st.session_state.get("wt_actual_fund_calibration_result")
        if calibration_result is not None:
            c1, c2, c3, c4 = st.columns(4, gap="small")
            c1.metric("Проверено пар", f"{int(calibration_result.analyzed_pair_count)}")
            c2.metric(
                "Пропущено пар одного семейства",
                f"{int(calibration_result.skipped_same_family_pair_count)}",
            )
            c3.metric(
                "Overlap до",
                f"{int(calibration_result.overlapping_pair_count_before)}",
            )
            c4.metric(
                "Overlap после",
                "—"
                if calibration_result.overlapping_pair_count_after is None
                else f"{int(calibration_result.overlapping_pair_count_after)}",
            )
            scale_label = (
                "—"
                if calibration_result.scale_factor is None
                else f"{float(calibration_result.scale_factor):.2f}"
            )
            worst_before = (
                "—"
                if calibration_result.worst_separation_factor_before is None
                else f"{float(calibration_result.worst_separation_factor_before):.2f}"
            )
            worst_after = (
                "—"
                if calibration_result.worst_separation_factor_after is None
                else f"{float(calibration_result.worst_separation_factor_after):.2f}"
            )
            st.caption(
                f"Статус: {str(calibration_result.status)}. "
                f"Scale={scale_label}, SF до={worst_before}, SF после={worst_after}. "
                f"База: {uncertainty_preset_label(calibration_result.base_preset)}."
            )
            if str(calibration_result.status) == CALIBRATION_STATUS_READY:
                st.success(str(calibration_result.note))
            else:
                st.info(str(calibration_result.note))


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
                                "MD max, м": float(item.stations["MD_m"].iloc[-1]),
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
                        "MD (из файла), м": float(point.md),
                    }
                )
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(raw_rows)),
            width="stretch",
            hide_index=True,
        )


def _render_t1_t3_order_panel(records: list[WelltrackRecord]) -> None:
    issues = detect_t1_t3_order_issues(records, min_delta_m=WT_T1T3_MIN_DELTA_M)
    if not issues:
        return

    with st.container(border=True):
        st.markdown("### Проверка порядка t1/t3")
        st.warning(
            "Найдены скважины, где `t1` дальше от устья `S` (куста) по горизонтальному "
            "отходу, чем `t3`. Вероятно, порядок точек `t1/t3` перепутан."
        )
        issue_rows = [
            {
                "Скважина": item.well_name,
                "Отход S→t1, м": float(item.t1_offset_m),
                "Отход S→t3, м": float(item.t3_offset_m),
                "Δ (t1 - t3), м": float(item.delta_m),
            }
            for item in issues
        ]
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(issue_rows)),
            width="stretch",
            hide_index=True,
        )
        if st.button(
            "Исправить порядок t1/t3 для отмеченных скважин",
            type="primary",
            icon=":material/swap_horiz:",
            width="stretch",
        ):
            target_names = {str(item.well_name) for item in issues}
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
            _clear_results()
            st.toast(f"Порядок t1/t3 исправлен для {len(target_names)} скважин.")
            st.rerun()
        st.caption(
            "Исправление меняет местами координаты `t1` и `t3`, но сохраняет MD "
            "во 2-й и 3-й позиции, чтобы не ломать порядок MD."
        )


def _pad_config_defaults(pad: WellPad) -> dict[str, float]:
    return {
        "spacing_m": float(DEFAULT_PAD_SPACING_M),
        "nds_azimuth_deg": float(pad.auto_nds_azimuth_deg),
        "first_surface_x": float(pad.surface.x),
        "first_surface_y": float(pad.surface.y),
        "first_surface_z": float(pad.surface.z),
    }


def _ensure_pad_configs(base_records: list[WelltrackRecord]) -> list[WellPad]:
    pads = detect_well_pads(base_records)
    existing = st.session_state.get("wt_pad_configs", {})
    merged: dict[str, dict[str, float]] = {}
    for pad in pads:
        defaults = _pad_config_defaults(pad)
        current = existing.get(str(pad.pad_id), {})
        merged[str(pad.pad_id)] = {
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
        }
    st.session_state["wt_pad_configs"] = merged

    pad_ids = [str(pad.pad_id) for pad in pads]
    if not pad_ids:
        st.session_state["wt_pad_selected_id"] = ""
        return pads
    if str(st.session_state.get("wt_pad_selected_id", "")) not in pad_ids:
        st.session_state["wt_pad_selected_id"] = pad_ids[0]
    return pads


def _build_pad_plan_map(pads: list[WellPad]) -> dict[str, PadLayoutPlan]:
    config_map = st.session_state.get("wt_pad_configs", {})
    plan_map: dict[str, PadLayoutPlan] = {}
    for pad in pads:
        pad_id = str(pad.pad_id)
        cfg = config_map.get(pad_id, _pad_config_defaults(pad))
        plan_map[pad_id] = PadLayoutPlan(
            pad_id=pad_id,
            first_surface_x=float(cfg["first_surface_x"]),
            first_surface_y=float(cfg["first_surface_y"]),
            first_surface_z=float(cfg["first_surface_z"]),
            spacing_m=float(max(cfg["spacing_m"], 0.0)),
            nds_azimuth_deg=float(cfg["nds_azimuth_deg"]) % 360.0,
        )
    return plan_map


def _render_pad_layout_panel(records: list[WelltrackRecord]) -> None:
    base_records = st.session_state.get("wt_records_original")
    if base_records is None:
        base_records = list(records)
    pads = _ensure_pad_configs(base_records=list(base_records))
    if not pads:
        return

    with st.container(border=True):
        st.markdown("### Кусты и расчет устьев")
        st.caption(
            "Куст определяется по совпадающим координатам устья S при импорте. "
            "Последовательность бурения строится по проекции середины (t1+t3)/2 вдоль НДС. "
            "Авто НДС — это стартовая геометрическая оценка по главной оси облака "
            "midpoint(t1, t3); для почти изотропных кустов она деградирует до "
            "стабильного fallback по направлению S→центроид и должна считаться "
            "рекомендацией, а не жестким инженерным решением."
        )
        if bool(st.session_state.get("wt_pad_auto_applied_on_import", False)):
            st.info(
                "После импорта исходные устья совпадали, поэтому текущие координаты S "
                "были автоматически скорректированы по параметрам этого блока. "
                "Если нужно вернуться к исходному WELLTRACK, нажмите "
                "'Вернуть исходные устья'."
            )
        pad_rows = [
            {
                "Куст": str(pad.pad_id),
                "Скважин": int(len(pad.wells)),
                "Авто НДС, deg": float(pad.auto_nds_azimuth_deg),
                "S X, м": float(pad.surface.x),
                "S Y, м": float(pad.surface.y),
                "S Z, м": float(pad.surface.z),
            }
            for pad in pads
        ]
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
        config_map = st.session_state.get("wt_pad_configs", {})
        selected_cfg = dict(
            config_map.get(selected_id, _pad_config_defaults(selected_pad))
        )

        widget_keys = {
            "spacing_m": f"wt_pad_cfg_spacing_m_{selected_id}",
            "nds_azimuth_deg": f"wt_pad_cfg_nds_azimuth_deg_{selected_id}",
            "first_surface_x": f"wt_pad_cfg_first_surface_x_{selected_id}",
            "first_surface_y": f"wt_pad_cfg_first_surface_y_{selected_id}",
            "first_surface_z": f"wt_pad_cfg_first_surface_z_{selected_id}",
        }
        for field, widget_key in widget_keys.items():
            if widget_key not in st.session_state:
                st.session_state[widget_key] = float(selected_cfg[field])

        p1, p2, p3, p4, p5 = st.columns(5, gap="small")
        spacing_m = p1.number_input(
            "Расстояние между устьями, м",
            min_value=0.0,
            step=1.0,
            key=widget_keys["spacing_m"],
            help="Шаг по кусту между соседними устьями скважин.",
        )
        nds_azimuth_deg = p2.number_input(
            "НДС (азимут), deg",
            min_value=0.0,
            max_value=360.0,
            step=0.5,
            key=widget_keys["nds_azimuth_deg"],
            help="Направление движения станка по кусту.",
        )
        first_surface_x = p3.number_input(
            "S1 X (East), м",
            step=10.0,
            key=widget_keys["first_surface_x"],
        )
        first_surface_y = p4.number_input(
            "S1 Y (North), м",
            step=10.0,
            key=widget_keys["first_surface_y"],
        )
        first_surface_z = p5.number_input(
            "S1 Z (TVD), м",
            step=10.0,
            key=widget_keys["first_surface_z"],
        )

        selected_cfg["spacing_m"] = float(max(spacing_m, 0.0))
        selected_cfg["nds_azimuth_deg"] = float(nds_azimuth_deg) % 360.0
        selected_cfg["first_surface_x"] = float(first_surface_x)
        selected_cfg["first_surface_y"] = float(first_surface_y)
        selected_cfg["first_surface_z"] = float(first_surface_z)
        config_map[selected_id] = selected_cfg
        st.session_state["wt_pad_configs"] = config_map

        ordered_wells = ordered_pad_wells(
            pad=selected_pad,
            nds_azimuth_deg=float(selected_cfg["nds_azimuth_deg"]),
        )
        angle_rad = np.deg2rad(float(selected_cfg["nds_azimuth_deg"]))
        ux = float(np.sin(angle_rad))
        uy = float(np.cos(angle_rad))
        preview_rows: list[dict[str, object]] = []
        for slot_index, well in enumerate(ordered_wells, start=1):
            shift_m = float(slot_index - 1) * float(selected_cfg["spacing_m"])
            preview_rows.append(
                {
                    "Порядок": int(slot_index),
                    "Скважина": str(well.name),
                    "Середина t1-t3 X, м": float(well.midpoint_x),
                    "Середина t1-t3 Y, м": float(well.midpoint_y),
                    "Новое S X, м": float(
                        selected_cfg["first_surface_x"] + shift_m * ux
                    ),
                    "Новое S Y, м": float(
                        selected_cfg["first_surface_y"] + shift_m * uy
                    ),
                    "Новое S Z, м": float(selected_cfg["first_surface_z"]),
                }
            )
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
        )
        reset_clicked = a2.button(
            "Вернуть исходные устья",
            icon=":material/restart_alt:",
            width="stretch",
        )

        if apply_clicked:
            plan_map = _build_pad_plan_map(pads)
            updated_records = apply_pad_layout(
                records=list(base_records),
                pads=pads,
                plan_by_pad_id=plan_map,
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
        current = [name for name in st.session_state.get(key, []) if name in all_names]
        if current != st.session_state.get(key, []):
            st.session_state[key] = list(current)
        if not current and recommended_names:
            st.session_state[key] = list(recommended_names)

    _sync_key("wt_selected_names")
    return all_names, recommended_names


def _render_batch_selection_status(
    records: list[WelltrackRecord], summary_rows: list[dict[str, object]] | None
) -> None:
    if summary_rows:
        st.caption(
            "Результаты по невыбранным скважинам сохраняются. Для следующего запуска "
            "по умолчанию выделяются нерассчитанные, ошибочные и warning-кейсы."
        )
        return

    all_names = [record.name for record in records]
    rows_by_name = {
        str(row.get("Скважина", "")).strip(): row for row in (summary_rows or [])
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
    with st.form("welltrack_run_form", clear_on_submit=False):
        st.markdown("#### Запуск / пересчет выбранных скважин")
        select_col, action_col = st.columns(
            [6.0, 1.4],
            gap="small",
            vertical_alignment="bottom",
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
        st.caption(
            "Используйте этот блок и для первого расчета набора, и для пересчета "
            "любой выбранной части скважин. Применяются параметры расчета ниже, "
            "а результаты остальных скважин не будут затронуты."
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
                    "Локальный режим": st.column_config.TextColumn("Локальный режим"),
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
    pad_layout_active = bool(str(st.session_state.get("wt_pad_last_applied_at", "")))
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
    log_verbosity = str(st.session_state.get("wt_log_verbosity", WT_LOG_COMPACT))
    verbose_log_enabled = log_verbosity == WT_LOG_VERBOSE
    config_by_name = _build_selected_override_configs(
        base_config=request.config,
        selected_names=selected_set,
    )
    optimization_context_by_name = _build_selected_optimization_contexts(
        selected_names=selected_set,
        current_successes=list(st.session_state.get("wt_successes") or ()),
    )
    current_success_by_name = {
        str(item.name): item for item in (st.session_state.get("wt_successes") or ())
    }
    prepared_snapshot = dict(
        st.session_state.get("wt_prepared_recommendation_snapshot") or {}
    )
    prepared_override_names = {
        str(name)
        for name in (st.session_state.get("wt_prepared_well_overrides") or {}).keys()
    }
    previous_anticollision_successes = {
        str(name): current_success_by_name[str(name)]
        for name in sorted(
            prepared_override_names.intersection(current_success_by_name)
        )
    }
    dynamic_cluster_context = None
    if str(prepared_snapshot.get("kind", "")).strip() == "cluster":
        target_well_names = _resolution_snapshot_well_names(prepared_snapshot)
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
                initial_successes=tuple(st.session_state.get("wt_successes") or ()),
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
        with st.spinner("Выполняется расчет WELLTRACK-набора...", show_time=True):
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
            set_phase(f"Старт расчета набора. Выбрано скважин: {len(selected_set)}.")
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
                overall = (float(index) - 1.0 + local_fraction) / max(float(total), 1.0)
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
                    summarize_problem_ru(raw_problem_text) if raw_problem_text else ""
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
                        append_log(f"{name}: расчет завершен успешно.{restart_suffix}")
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
            skipped_policy_count = int(len(batch_metadata.skipped_selected_names))
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
                            "пересчета оставлены: " + ", ".join(skipped_names) + "."
                        )
            elapsed_s = perf_counter() - started
            progress.progress(100, text="Batch-расчет завершен.")
            _store_merged_batch_results(
                records=records_for_run,
                new_rows=summary_rows,
                new_successes=successes,
            )
            applied_affected_wells = {
                str(name) for name in prepared_snapshot.get("affected_wells", ())
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
                    uncertainty_model=planning_uncertainty_model_for_preset(preset),
                    uncertainty_preset=preset,
                )
                st.session_state["wt_last_anticollision_resolution"] = resolution
                st.session_state["wt_last_anticollision_previous_successes"] = (
                    previous_anticollision_successes
                )
                _focus_all_wells_anticollision_results()
            else:
                st.session_state["wt_last_anticollision_resolution"] = None
                st.session_state["wt_last_anticollision_previous_successes"] = {}
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


def _render_batch_summary(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    summary_df = WelltrackBatchPlanner.summary_dataframe(summary_rows)
    if not summary_df.empty:
        summary_df = arrow_safe_text_dataframe(summary_df)

    ok_count = 0
    warning_count = 0
    err_count = 0
    not_run_count = 0
    if not summary_df.empty and {"Статус", "Проблема"}.issubset(summary_df.columns):
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
    p5.metric("Время расчета", "—" if run_time is None else f"{float(run_time):.2f} с")
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
            "Точек": st.column_config.NumberColumn("Точек", format="%d", width="small"),
            "Цели": st.column_config.TextColumn("Цели", width="small"),
            "Сложность": st.column_config.TextColumn("Сложность", width="small"),
            "Отход t1, м": st.column_config.NumberColumn(
                "Отход t1, м",
                format="%.2f",
                width="small",
            ),
            "KOP MD, м": st.column_config.NumberColumn(
                "KOP MD, м",
                format="%.2f",
                width="small",
            ),
            "HORIZONTAL, м": st.column_config.NumberColumn(
                "HORIZONTAL, м",
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
            "Проблема": st.column_config.TextColumn("Проблема", width="medium"),
            "Модель траектории": st.column_config.TextColumn(
                "Модель траектории",
                width="medium",
            ),
        },
    )
    st.download_button(
        "Скачать сводку (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="welltrack_summary.csv",
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )
    return summary_df


def _batch_summary_display_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    display_df = summary_df.rename(columns=_BATCH_SUMMARY_RENAME_COLUMNS).copy()
    ordered = [
        column
        for column in _BATCH_SUMMARY_DISPLAY_ORDER
        if column in display_df.columns
    ]
    trailing = [column for column in display_df.columns if column not in ordered]
    return display_df[ordered + trailing]


def _ensure_selected_success_baseline(
    *,
    selected_name: str,
    successes: list[SuccessfulWellPlan],
) -> SuccessfulWellPlan:
    selected = next(item for item in successes if str(item.name) == str(selected_name))
    updated = ensure_successful_plan_baseline(success=selected)
    if updated is selected:
        return selected
    st.session_state["wt_successes"] = [
        updated if str(item.name) == str(selected_name) else item for item in successes
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
            render_mode=selected_render_mode,
        )
        _render_plotly_or_three_3d(
            container=c1,
            figure=overview_3d_figure,
            backend=selected_3d_backend,
            height=620,
        )
        c2.plotly_chart(
            _all_wells_plan_figure(
                successes,
                target_only_wells=target_only_wells,
                reference_wells=reference_wells,
                name_to_color=name_to_color,
            ),
            width="stretch",
        )
        return

    _render_anticollision_panel(successes)


def run_page() -> None:
    st.set_page_config(page_title="Импорт WELLTRACK", layout="wide")
    _init_state()
    apply_page_style(max_width_px=1700)
    render_hero(title="Импорт WELLTRACK", subtitle="")
    source_payload, parse_clicked, clear_clicked, reset_params_clicked = (
        _render_import_controls()
    )
    _handle_import_actions(
        source_payload=source_payload,
        parse_clicked=parse_clicked,
        clear_clicked=clear_clicked,
        reset_params_clicked=reset_params_clicked,
    )

    records = st.session_state.get("wt_records")
    if records is None:
        st.info("Загрузите источник и нажмите «Прочитать WELLTRACK».")
        return
    if not records:
        st.warning("В источнике не найдено ни одного WELLTRACK блока.")
        return

    _render_records_overview(records=records)
    _render_reference_trajectory_panel()
    _render_raw_records_table(records=records)
    _render_t1_t3_order_panel(records=records)
    _render_pad_layout_panel(records=records)
    all_names, _ = _sync_selection_state(records=records)
    requests = _render_batch_run_forms(records=records, all_names=all_names)
    _run_batch_if_clicked(requests=requests, records=records)
    _render_batch_log()
    if st.session_state.get("wt_last_error"):
        render_solver_diagnostics(st.session_state["wt_last_error"])

    summary_rows = st.session_state.get("wt_summary_rows")
    successes = st.session_state.get("wt_successes")
    if not summary_rows:
        render_small_note("Результаты расчета появятся после запуска batch-расчета.")
        return
    _render_batch_summary(summary_rows=summary_rows)
    if not successes:
        st.warning("Все выбранные скважины завершились ошибками расчета.")
        return
    _render_success_tabs(
        successes=successes,
        records=list(records),
        summary_rows=list(summary_rows),
    )


if __name__ == "__main__":
    run_page()
