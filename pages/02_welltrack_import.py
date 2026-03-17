from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pywp import TrajectoryConfig, TrajectoryPlanner
from pywp.anticollision import (
    AntiCollisionAnalysis,
    anti_collision_method_caption,
    anti_collision_report_events,
    anti_collision_report_rows,
    analyze_anti_collision,
    build_anti_collision_well,
    collision_corridor_plan_polygon,
    collision_corridor_point_sphere_mesh,
    collision_corridor_tube_mesh,
)
from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
)
from pywp.anticollision_recommendations import (
    AntiCollisionRecommendation,
    AntiCollisionWellContext,
    anti_collision_recommendation_rows,
    build_anti_collision_recommendations,
    recommendation_display_label,
)
from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackRecord,
    decode_welltrack_bytes,
    parse_welltrack_text,
    welltrack_points_to_targets,
)
from pywp.plotly_config import DEFAULT_3D_CAMERA, trajectory_plotly_chart_config
from pywp.planner_config import optimization_display_label
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
    SuccessfulWellPlan,
    WelltrackBatchPlanner,
    merge_batch_results,
    recommended_batch_selection,
)

DEFAULT_WELLTRACK_PATH = Path("tests/test_data/WELLTRACKS3.INC")
WT_UI_DEFAULTS_VERSION = 13
WT_LOG_COMPACT = "Краткий"
WT_LOG_VERBOSE = "Подробный"
WT_LOG_LEVEL_OPTIONS: tuple[str, ...] = (WT_LOG_COMPACT, WT_LOG_VERBOSE)
WT_T1T3_MIN_DELTA_M = 0.5
WELL_COLOR_PALETTE: tuple[str, ...] = (
    "#0B6E4F",
    "#3A86FF",
    "#00798C",
    "#FFB703",
    "#6A4C93",
    "#1F7A8C",
    "#3D5A80",
    "#F4A261",
    "#2A9D8F",
    "#4D908E",
    "#577590",
    "#8E9AAF",
)
_WT_LEGACY_KEY_ALIASES: dict[str, str] = {
    "wt_cfg_md_step_m": "wt_cfg_md_step",
    "wt_cfg_md_step_control_m": "wt_cfg_md_control",
    "wt_cfg_pos_tolerance_m": "wt_cfg_pos_tol",
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
class _TargetOnlyWell:
    name: str
    surface: Point3D
    t1: Point3D
    t3: Point3D
    status: str
    problem: str


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


def _build_anti_collision_analysis(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: dict[str, str] | None = None,
) -> AntiCollisionAnalysis:
    wells = [
        build_anti_collision_well(
            name=item.name,
            color=(name_to_color or {}).get(str(item.name), _well_color(index)),
            stations=item.stations,
            surface=item.surface,
            t1=item.t1,
            t3=item.t3,
            azimuth_deg=float(item.azimuth_deg),
            md_t1_m=float(item.md_t1_m),
            model=model,
        )
        for index, item in enumerate(successes)
    ]
    return analyze_anti_collision(wells)


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
            "X_m": np.interp(segment_md, md_values, stations["X_m"].to_numpy(dtype=float)),
            "Y_m": np.interp(segment_md, md_values, stations["Y_m"].to_numpy(dtype=float)),
            "Z_m": np.interp(segment_md, md_values, stations["Z_m"].to_numpy(dtype=float)),
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


def _init_state() -> None:
    st.session_state.setdefault("wt_source_mode", "Файл по пути")
    st.session_state.setdefault("wt_source_path", str(DEFAULT_WELLTRACK_PATH))
    st.session_state.setdefault("wt_source_inline", "")
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
    st.session_state.setdefault(
        "wt_anticollision_uncertainty_preset", DEFAULT_UNCERTAINTY_PRESET
    )
    st.session_state.setdefault("wt_prepared_well_overrides", {})
    st.session_state.setdefault("wt_prepared_override_message", "")
    st.session_state.setdefault("wt_prepared_recommendation_id", "")


def _clear_results() -> None:
    st.session_state["wt_summary_rows"] = None
    st.session_state["wt_successes"] = None
    st.session_state["wt_pending_selected_names"] = None
    st.session_state["wt_last_error"] = ""
    st.session_state["wt_last_run_at"] = ""
    st.session_state["wt_last_runtime_s"] = None
    st.session_state["wt_last_run_log_lines"] = []
    st.session_state["wt_prepared_well_overrides"] = {}
    st.session_state["wt_prepared_override_message"] = ""
    st.session_state["wt_prepared_recommendation_id"] = ""


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
    name_to_color: dict[str, str] | None = None,
    height: int = 620,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
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
        z_arrays.append(stations["Z_m"].to_numpy(dtype=float))
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
                    "size": 6,
                    "color": "rgba(255,255,255,0.65)",
                    "line": {"width": 2, "color": line_color},
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

    x_values = np.concatenate(x_arrays) if x_arrays else np.array([0.0], dtype=float)
    y_values = np.concatenate(y_arrays) if y_arrays else np.array([0.0], dtype=float)
    z_values = np.concatenate(z_arrays) if z_arrays else np.array([0.0], dtype=float)
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
    name_to_color: dict[str, str] | None = None,
    height: int = 560,
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
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
                    "size": 8,
                    "color": "rgba(255,255,255,0.70)",
                    "line": {"width": 2, "color": line_color},
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
        surface = item.surface
        t1 = item.t1
        t3 = item.t3
        x_arrays.append(np.array([surface.x, t1.x, t3.x], dtype=float))
        y_arrays.append(np.array([surface.y, t1.y, t3.y], dtype=float))
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
    x_values = np.concatenate(x_arrays) if x_arrays else np.array([0.0], dtype=float)
    y_values = np.concatenate(y_arrays) if y_arrays else np.array([0.0], dtype=float)
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
    analysis: AntiCollisionAnalysis, height: int = 660
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    well_lookup = {str(well.name): well for well in analysis.wells}

    for well in analysis.wells:
        overlay = well.overlay
        tube_mesh = build_uncertainty_tube_mesh(overlay)
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
                    name=f"{well.name} cone",
                    legendgroup=str(well.name),
                    showlegend=False,
                    color=str(well.color),
                    opacity=0.12,
                    flatshading=True,
                    hoverinfo="skip",
                )
            )
        if overlay.samples:
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
                    name=f"{well.name}: граница конуса",
                    legendgroup=str(well.name),
                    showlegend=False,
                    line={
                        "width": 1.5,
                        "color": _lighten_hex(str(well.color)),
                    },
                    hoverinfo="skip",
                )
            )

        stations = well.stations
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        z_values = stations["Z_m"].to_numpy(dtype=float)
        md_values = stations["MD_m"].to_numpy(dtype=float)
        x_arrays.append(x_values)
        y_arrays.append(y_values)
        z_arrays.append(z_values)

        fig.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                name=str(well.name),
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
                    f"<extra>{well.name}</extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[well.surface.x, well.t1.x, well.t3.x],
                y=[well.surface.y, well.t1.y, well.t3.y],
                z=[well.surface.z, well.t1.z, well.t3.z],
                mode="markers",
                name=f"{well.name}: цели",
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
        x_arrays.append(np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float))
        y_arrays.append(np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float))
        z_arrays.append(np.array([well.surface.z, well.t1.z, well.t3.z], dtype=float))

    overlap_legend_added = False
    for corridor in analysis.corridors:
        mesh = collision_corridor_tube_mesh(corridor)
        if mesh is not None:
            x_arrays.append(mesh.vertices_xyz[:, 0])
            y_arrays.append(mesh.vertices_xyz[:, 1])
            z_arrays.append(mesh.vertices_xyz[:, 2])
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
            sphere_x, sphere_y, sphere_z = collision_corridor_point_sphere_mesh(corridor)
            x_arrays.append(sphere_x.reshape(-1))
            y_arrays.append(sphere_y.reshape(-1))
            z_arrays.append(sphere_z.reshape(-1))
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

    x_values = np.concatenate(x_arrays) if x_arrays else np.array([0.0], dtype=float)
    y_values = np.concatenate(y_arrays) if y_arrays else np.array([0.0], dtype=float)
    z_values = np.concatenate(z_arrays) if z_arrays else np.array([0.0], dtype=float)
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
    analysis: AntiCollisionAnalysis, height: int = 620
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    well_lookup = {str(well.name): well for well in analysis.wells}

    for well in analysis.wells:
        overlay = well.overlay
        ribbon = uncertainty_ribbon_polygon(overlay, projection="plan")
        if len(ribbon) >= 3:
            fig.add_trace(
                go.Scatter(
                    x=ribbon[:, 0],
                    y=ribbon[:, 1],
                    mode="lines",
                    name=f"{well.name} cone",
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

        stations = well.stations
        x_values = stations["X_m"].to_numpy(dtype=float)
        y_values = stations["Y_m"].to_numpy(dtype=float)
        md_values = stations["MD_m"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=str(well.name),
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
        fig.add_trace(
            go.Scatter(
                x=[well.surface.x, well.t1.x, well.t3.x],
                y=[well.surface.y, well.t1.y, well.t3.y],
                mode="markers",
                name=f"{well.name}: цели",
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
        x_arrays.append(np.array([well.surface.x, well.t1.x, well.t3.x], dtype=float))
        y_arrays.append(np.array([well.surface.y, well.t1.y, well.t3.y], dtype=float))

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
        segment_legend_added = True

    x_values = np.concatenate(x_arrays) if x_arrays else np.array([0.0], dtype=float)
    y_values = np.concatenate(y_arrays) if y_arrays else np.array([0.0], dtype=float)
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
    if len(successes) < 2:
        st.info("Для anti-collision нужно минимум две успешно рассчитанные скважины.")
        return

    st.session_state["wt_anticollision_uncertainty_preset"] = normalize_uncertainty_preset(
        st.session_state.get(
            "wt_anticollision_uncertainty_preset",
            DEFAULT_UNCERTAINTY_PRESET,
        )
    )
    selected_preset = st.selectbox(
        "Пресет неопределенности для anti-collision",
        options=list(UNCERTAINTY_PRESET_OPTIONS.keys()),
        index=list(UNCERTAINTY_PRESET_OPTIONS.keys()).index(
            st.session_state["wt_anticollision_uncertainty_preset"]
        ),
        format_func=uncertainty_preset_label,
        key="wt_anticollision_uncertainty_preset",
        help=(
            "Определяет уровень консерватизма planning-level конусов неопределенности "
            "для batch anti-collision анализа."
        ),
    )
    uncertainty_model = planning_uncertainty_model_for_preset(selected_preset)
    records = list(st.session_state.get("wt_records") or [])
    analysis = _build_anti_collision_analysis(
        successes,
        model=uncertainty_model,
        name_to_color=_well_color_map(records) if records else None,
    )
    st.caption(
        f"Пресет: {uncertainty_preset_label(selected_preset)}. "
        f"{anti_collision_method_caption(uncertainty_model)}"
    )

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Проверено пар", f"{int(analysis.pair_count)}")
    m2.metric("Пар с overlap", f"{int(analysis.overlapping_pair_count)}")
    m3.metric("Пересечения в t1/t3", f"{int(analysis.target_overlap_pair_count)}")
    worst_sf = analysis.worst_separation_factor
    m4.metric("Минимальный SF", "—" if worst_sf is None else f"{float(worst_sf):.2f}")

    chart_col1, chart_col2 = st.columns(2, gap="medium")
    chart_col1.plotly_chart(
        _all_wells_anticollision_3d_figure(analysis),
        width="stretch",
    )
    chart_col2.plotly_chart(
        _all_wells_anticollision_plan_figure(analysis),
        width="stretch",
    )

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
        st.warning(
            "Найдены пересечения 2σ конусов неопределенности по траекториям."
        )

    report_events = anti_collision_report_events(analysis)
    report_df = arrow_safe_text_dataframe(pd.DataFrame(anti_collision_report_rows(analysis)))
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

    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=_build_anticollision_well_contexts(successes),
    )
    recommendation_df = arrow_safe_text_dataframe(
        pd.DataFrame(anti_collision_recommendation_rows(recommendations))
    )
    st.markdown("### Рекомендации")
    st.caption(
        "Рекомендации делят конфликты на три класса: цели, vertical-участки и "
        "прочие траекторные пересечения. Автоподготовка пересчета сейчас включена "
        "для vertical-кейсов и для pairwise trajectory-collision rerun одной "
        "скважины относительно фиксированной reference-траектории."
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
            "Overlap max, м": st.column_config.NumberColumn("Overlap max, м", format="%.2f"),
            "Spacing t1, м": st.column_config.TextColumn("Spacing t1, м"),
            "Spacing t3, м": st.column_config.TextColumn("Spacing t3, м"),
            "Рекомендация": st.column_config.TextColumn("Рекомендация"),
            "Подготовка пересчета": st.column_config.TextColumn("Подготовка пересчета"),
        },
    )

    actionable_recommendations = [
        item for item in recommendations if bool(item.can_prepare_rerun)
    ]
    if actionable_recommendations:
        actionable_ids = [item.recommendation_id for item in actionable_recommendations]
        if str(st.session_state.get("wt_anticollision_prepared_recommendation_id", "")) not in actionable_ids:
            st.session_state["wt_anticollision_prepared_recommendation_id"] = actionable_ids[0]
        action_col, button_col = st.columns([6.0, 1.8], gap="small")
        with action_col:
            selected_recommendation_id = st.selectbox(
                "Подготовить пересчет по anti-collision рекомендации",
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
                "Подготовить пересчет",
                type="primary",
                icon=":material/build:",
                width="stretch",
            ):
                selected_recommendation = next(
                    item
                    for item in actionable_recommendations
                    if str(item.recommendation_id) == str(selected_recommendation_id)
                )
                _prepare_rerun_from_recommendation(
                    selected_recommendation,
                    successes=successes,
                    uncertainty_model=uncertainty_model,
                )
                st.toast(
                    "План пересчета подготовлен. В блоке пакетного расчета уже "
                    "предвыбраны затронутые скважины и overrides."
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
    rows: list[dict[str, object]] = []
    for well_name in sorted(str(name) for name in prepared.keys()):
        payload = dict(prepared.get(well_name, {}))
        update_fields = dict(payload.get("update_fields", {}))
        optimization_mode = str(update_fields.get("optimization_mode", "")).strip()
        optimization_label = optimization_display_label(optimization_mode)
        rows.append(
            {
                "Скважина": str(well_name),
                "Оптимизация": optimization_label,
                "Источник": str(payload.get("source", "—")).strip() or "—",
                "Причина": str(payload.get("reason", "—")).strip() or "—",
            }
        )
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
) -> dict[str, AntiCollisionOptimizationContext]:
    prepared = st.session_state.get("wt_prepared_well_overrides", {}) or {}
    context_map: dict[str, AntiCollisionOptimizationContext] = {}
    for well_name in sorted(str(name) for name in selected_names):
        payload = dict(prepared.get(well_name, {}))
        optimization_context = payload.get("optimization_context")
        if isinstance(optimization_context, AntiCollisionOptimizationContext):
            context_map[well_name] = optimization_context
    return context_map


def _build_anticollision_well_contexts(
    successes: list[SuccessfulWellPlan],
) -> dict[str, AntiCollisionWellContext]:
    contexts: dict[str, AntiCollisionWellContext] = {}
    for success in successes:
        summary = dict(success.summary)
        kop_md_raw = summary.get("kop_md_m")
        kop_md_m = None
        if kop_md_raw is not None:
            try:
                kop_md_m = float(kop_md_raw)
            except (TypeError, ValueError):
                kop_md_m = None
        contexts[str(success.name)] = AntiCollisionWellContext(
            well_name=str(success.name),
            kop_md_m=kop_md_m,
            kop_min_vertical_m=float(success.config.kop_min_vertical_m),
            md_total_m=float(summary.get("md_total_m", 0.0)),
            optimization_mode=str(success.config.optimization_mode),
        )
    return contexts


def _prepare_rerun_from_recommendation(
    recommendation: AntiCollisionRecommendation,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> None:
    prepared: dict[str, dict[str, object]] = {}
    success_by_name = {str(item.name): item for item in successes}
    for suggestion in recommendation.override_suggestions:
        reference_name = (
            str(recommendation.well_b)
            if str(suggestion.well_name) == str(recommendation.well_a)
            else str(recommendation.well_a)
        )
        optimization_context = _build_prepared_optimization_context(
            recommendation=recommendation,
            moving_success=success_by_name.get(str(suggestion.well_name)),
            reference_success=success_by_name.get(reference_name),
            uncertainty_model=uncertainty_model,
        )
        prepared[str(suggestion.well_name)] = {
            "update_fields": dict(suggestion.config_updates),
            "source": recommendation_display_label(recommendation),
            "reason": str(suggestion.reason),
            "optimization_context": optimization_context,
        }
    st.session_state["wt_prepared_well_overrides"] = prepared
    st.session_state["wt_prepared_override_message"] = str(recommendation.summary)
    st.session_state["wt_prepared_recommendation_id"] = str(recommendation.recommendation_id)
    st.session_state["wt_pending_selected_names"] = list(recommendation.affected_wells)


def _build_prepared_optimization_context(
    *,
    recommendation: AntiCollisionRecommendation,
    moving_success: SuccessfulWellPlan | None,
    reference_success: SuccessfulWellPlan | None,
    uncertainty_model: PlanningUncertaintyModel,
) -> AntiCollisionOptimizationContext | None:
    if moving_success is None or reference_success is None:
        return None
    moving_is_a = str(moving_success.name) == str(recommendation.well_a)
    candidate_md_start_m = (
        float(recommendation.md_a_start_m)
        if moving_is_a
        else float(recommendation.md_b_start_m)
    )
    candidate_md_end_m = (
        float(recommendation.md_a_end_m)
        if moving_is_a
        else float(recommendation.md_b_end_m)
    )
    reference_md_start_m = (
        float(recommendation.md_b_start_m)
        if moving_is_a
        else float(recommendation.md_a_start_m)
    )
    reference_md_end_m = (
        float(recommendation.md_b_end_m)
        if moving_is_a
        else float(recommendation.md_a_end_m)
    )
    reference_path = build_anti_collision_reference_path(
        well_name=str(reference_success.name),
        stations=reference_success.stations,
        md_start_m=reference_md_start_m,
        md_end_m=reference_md_end_m,
        sample_step_m=50.0,
        model=uncertainty_model,
    )
    return AntiCollisionOptimizationContext(
        candidate_md_start_m=float(candidate_md_start_m),
        candidate_md_end_m=float(candidate_md_end_m),
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=uncertainty_model,
        references=(reference_path,),
    )


def _render_source_input() -> str:
    st.markdown("### Источник WELLTRACK")
    source_mode = st.radio(
        "Режим загрузки",
        options=["Файл по пути", "Загрузить файл", "Вставить текст"],
        horizontal=True,
        key="wt_source_mode",
    )

    if source_mode == "Файл по пути":
        source_path = st.text_input(
            "Путь к файлу WELLTRACK",
            key="wt_source_path",
            placeholder="tests/test_data/WELLTRACKS3.INC",
        )
        return _read_welltrack_file(source_path)

    if source_mode == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Файл ECLIPSE/INC", type=["inc", "txt", "data", "ecl"]
        )
        if uploaded_file is None:
            return ""
        return _decode_welltrack_payload(
            uploaded_file.getvalue(),
            source_label=f"Загруженный файл `{uploaded_file.name}`",
        )

    return st.text_area(
        "Текст WELLTRACK",
        key="wt_source_inline",
        height=220,
        placeholder="WELLTRACK 'WELL-1' ...",
    )


def _store_parsed_records(records: list[WelltrackRecord]) -> bool:
    all_names = [record.name for record in records]
    st.session_state["wt_records"] = list(records)
    st.session_state["wt_records_original"] = list(records)
    st.session_state["wt_loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _clear_pad_state()
    st.session_state["wt_last_error"] = ""
    _clear_results()
    auto_layout_applied = _auto_apply_pad_layout_if_shared_surface(records=list(records))
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


def _render_import_controls() -> tuple[str, bool, bool, bool]:
    source_col, action_col = st.columns(
        [4.0, 1.2], gap="small", vertical_alignment="bottom"
    )
    with source_col:
        source_text = _render_source_input()
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
        source_text,
        bool(parse_clicked),
        bool(clear_clicked),
        bool(reset_params_clicked),
    )


def _handle_import_actions(
    source_text: str,
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
        st.session_state["wt_selected_names"] = []
        st.session_state["wt_loaded_at"] = ""
        _clear_pad_state()
        _clear_results()
        st.rerun()

    if not parse_clicked:
        return
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


def _sync_selection_state(records: list[WelltrackRecord]) -> tuple[list[str], list[str]]:
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
        info_col, action_col = st.columns([6.0, 1.4], gap="small")
        with info_col:
            st.info(
                str(
                    st.session_state.get("wt_prepared_override_message", "")
                    or "Подготовлен пересчет по anti-collision рекомендации."
                )
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
                st.rerun()
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(prepared_rows)),
            width="stretch",
            hide_index=True,
            column_config={
                "Скважина": st.column_config.TextColumn("Скважина"),
                "Оптимизация": st.column_config.TextColumn("Оптимизация"),
                "Источник": st.column_config.TextColumn("Источник"),
                "Причина": st.column_config.TextColumn("Причина"),
            },
        )
        st.caption(
            "Этот план пересчета будет применен только к выбранным ниже скважинам. "
            "Остальные скважины используют общие параметры без overrides."
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
    selected_set = set(request.selected_names)
    if not selected_set:
        st.warning("Выберите минимум одну скважину для расчета.")
        return

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
    )
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

    try:
        with st.spinner("Выполняется расчет WELLTRACK-набора...", show_time=True):
            started = perf_counter()
            append_log(
                f"Старт batch-расчета. Выбрано скважин: {len(selected_set)}. "
                f"Детализация лога: {log_verbosity}."
            )
            if config_by_name:
                append_log(
                    "Для части выбранных скважин активны подготовленные anti-collision "
                    "overrides."
                )
            if optimization_context_by_name:
                append_log(
                    "Для части выбранных скважин активирован anti-collision avoidance "
                    "mode на конфликтном окне."
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
                config=request.config,
                config_by_name=config_by_name,
                optimization_context_by_name=optimization_context_by_name,
                progress_callback=on_progress,
                solver_progress_callback=on_solver_progress,
                record_done_callback=on_record_done,
            )
            elapsed_s = perf_counter() - started
            progress.progress(100, text="Batch-расчет завершен.")
            _store_merged_batch_results(
                records=records_for_run,
                new_rows=summary_rows,
                new_successes=successes,
            )
            st.session_state["wt_last_error"] = ""
            st.session_state["wt_last_run_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.session_state["wt_last_runtime_s"] = float(elapsed_s)
            st.session_state["wt_prepared_well_overrides"] = {}
            st.session_state["wt_prepared_override_message"] = ""
            st.session_state["wt_prepared_recommendation_id"] = ""
            append_log(
                f"Batch-расчет завершен. Успешно: {len(successes)}, "
                f"ошибок: {len(summary_rows) - len(successes)}. "
                f"Затраченное время: {elapsed_s:.2f} с.",
            )
            if successes:
                phase_placeholder.success(
                    f"Расчет завершен за {elapsed_s:.2f} с. Успешно: {len(successes)}"
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
    ordered = [column for column in _BATCH_SUMMARY_DISPLAY_ORDER if column in display_df.columns]
    trailing = [column for column in display_df.columns if column not in ordered]
    return display_df[ordered + trailing]


def _render_success_tabs(
    *,
    successes: list[SuccessfulWellPlan],
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]],
) -> None:
    name_to_color = _well_color_map(records)
    target_only_wells = _failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    tab_single, tab_all = st.tabs(["Отдельная скважина", "Все скважины"])
    with tab_single:
        selected_name = st.selectbox(
            "Скважина", options=[item.name for item in successes]
        )
        selected = next(item for item in successes if item.name == selected_name)
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
            title_trajectory="3D траектория и ПИ",
            title_plan="План и вертикальный разрез",
            border=True,
        )
        render_result_tables(
            view=well_view,
            t1_horizontal_offset_m=t1_horizontal_offset_m,
            summary_tab_label="Сводка",
            survey_tab_label="Инклинометрия",
            survey_file_name=f"{selected_name}_survey.csv",
        )

    with tab_all:
        overview_tab, anticollision_tab = st.tabs(["Траектории", "Anti-collision"])
        with overview_tab:
            if target_only_wells:
                st.caption(
                    "Для непростроенных скважин на обзорных графиках показаны только "
                    "точки S/t1/t3, без траектории."
                )
            c1, c2 = st.columns(2, gap="medium")
            c1.plotly_chart(
                _all_wells_3d_figure(
                    successes,
                    target_only_wells=target_only_wells,
                    name_to_color=name_to_color,
                ),
                config=trajectory_plotly_chart_config(),
                width="stretch",
            )
            c2.plotly_chart(
                _all_wells_plan_figure(
                    successes,
                    target_only_wells=target_only_wells,
                    name_to_color=name_to_color,
                ),
                width="stretch",
            )
        with anticollision_tab:
            _render_anticollision_panel(successes)


def run_page() -> None:
    st.set_page_config(page_title="Импорт WELLTRACK", layout="wide")
    _init_state()
    apply_page_style(max_width_px=1700)
    render_hero(title="Импорт WELLTRACK", subtitle="")
    source_text, parse_clicked, clear_clicked, reset_params_clicked = (
        _render_import_controls()
    )
    _handle_import_actions(
        source_text=source_text,
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
