from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pywp import TrajectoryConfig, TrajectoryPlanner
from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackRecord,
    decode_welltrack_bytes,
    parse_welltrack_text,
)
from pywp.plot_axes import equalized_axis_ranges, linear_tick_values, nice_tick_step
from pywp.solver_diagnostics import summarize_problem_ru
from pywp.solver_diagnostics_ui import render_solver_diagnostics
from pywp.ui_calc_params import (
    CalcParamBinding,
)
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_utils import (
    arrow_safe_text_dataframe,
    dls_to_pi,
    format_distance,
    format_run_log_line,
)
from pywp.ui_well_panels import (
    render_plan_section_panel,
    render_run_log_panel,
    render_trajectory_dls_panel,
)
from pywp.well_pad import (
    PadLayoutPlan,
    WellPad,
    apply_pad_layout,
    detect_well_pads,
    ordered_pad_wells,
)
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner

DEFAULT_WELLTRACK_PATH = Path("tests/test_data/WELLTRACKS.INC")
WT_UI_DEFAULTS_VERSION = 12
WELL_COLOR_PALETTE: tuple[str, ...] = (
    "#0B6E4F",
    "#D1495B",
    "#00798C",
    "#FF9F1C",
    "#6A4C93",
    "#1F7A8C",
    "#3D5A80",
    "#E76F51",
    "#2A9D8F",
    "#4D908E",
    "#577590",
    "#BC4749",
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


def _well_color(index: int) -> str:
    return WELL_COLOR_PALETTE[index % len(WELL_COLOR_PALETTE)]


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
    st.session_state.setdefault("wt_loaded_at", "")
    st.session_state.setdefault("wt_pad_configs", {})
    st.session_state.setdefault("wt_pad_selected_id", "")
    st.session_state.setdefault("wt_pad_last_applied_at", "")

    st.session_state.setdefault("wt_summary_rows", None)
    st.session_state.setdefault("wt_successes", None)
    st.session_state.setdefault("wt_last_error", "")
    st.session_state.setdefault("wt_last_run_at", "")
    st.session_state.setdefault("wt_last_runtime_s", None)
    st.session_state.setdefault("wt_last_run_log_lines", [])


def _clear_results() -> None:
    st.session_state["wt_summary_rows"] = None
    st.session_state["wt_successes"] = None
    st.session_state["wt_last_error"] = ""
    st.session_state["wt_last_run_at"] = ""
    st.session_state["wt_last_runtime_s"] = None
    st.session_state["wt_last_run_log_lines"] = []


def _clear_pad_state() -> None:
    st.session_state["wt_pad_configs"] = {}
    st.session_state["wt_pad_selected_id"] = ""
    st.session_state["wt_pad_last_applied_at"] = ""
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
    successes: list[SuccessfulWellPlan], height: int = 620
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    for index, item in enumerate(successes):
        line_color = _well_color(index)
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
                line={"width": 5, "color": line_color},
                customdata=np.column_stack(
                    [
                        stations["MD_m"].to_numpy(dtype=float),
                        dls_to_pi(
                            stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float)
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

    x_values = np.concatenate(x_arrays) if x_arrays else np.array([0.0], dtype=float)
    y_values = np.concatenate(y_arrays) if y_arrays else np.array([0.0], dtype=float)
    z_values = np.concatenate(z_arrays) if z_arrays else np.array([0.0], dtype=float)
    x_range, y_range, z_range = equalized_axis_ranges(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )
    xy_span = x_range[1] - x_range[0]
    xy_dtick = nice_tick_step(xy_span, target_ticks=6)
    xy_tickvals = linear_tick_values(axis_range=x_range, step=xy_dtick)
    xy_tick0 = float(np.floor(min(x_range[0], y_range[0]) / xy_dtick) * xy_dtick)
    xy_axis_style = {
        "tickmode": "array",
        "tickvals": xy_tickvals,
        "dtick": xy_dtick,
        "tick0": xy_tick0,
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
            "xaxis": {"range": x_range, **xy_axis_style},
            "yaxis": {"range": y_range, **xy_axis_style},
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
    successes: list[SuccessfulWellPlan], height: int = 560
) -> go.Figure:
    fig = go.Figure()
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    for index, item in enumerate(successes):
        line_color = _well_color(index)
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
                line={"width": 4, "color": line_color},
                customdata=np.column_stack(
                    [
                        stations["Z_m"].to_numpy(dtype=float),
                        stations["MD_m"].to_numpy(dtype=float),
                        dls_to_pi(
                            stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float)
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
    xy_min = float(min(np.min(x_values), np.min(y_values)))
    xy_max = float(max(np.max(x_values), np.max(y_values)))
    xy_span = max(xy_max - xy_min, 1.0)
    half = 0.5 * xy_span * 1.08
    center = 0.5 * (xy_min + xy_max)
    xy_range = [center - half, center + half]
    xy_dtick = nice_tick_step(xy_range[1] - xy_range[0], target_ticks=6)
    xy_tickvals = linear_tick_values(axis_range=xy_range, step=xy_dtick)

    fig.update_layout(
        title="Все рассчитанные скважины (план E-N, X=Восток, Y=Север)",
        xaxis_title="X / Восток (м)",
        yaxis_title="Y / Север (м)",
        xaxis={
            "range": xy_range,
            "tickmode": "array",
            "tickvals": xy_tickvals,
            "tickformat": ".0f",
        },
        yaxis={
            "range": xy_range,
            "tickmode": "array",
            "tickvals": xy_tickvals,
            "tickformat": ".0f",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def _build_config_form() -> TrajectoryConfig:
    WT_CALC_PARAMS.render_block()
    return WT_CALC_PARAMS.build_config()


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
            placeholder="tests/test_data/WELLTRACKS.INC",
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


def _store_parsed_records(records: list[WelltrackRecord]) -> None:
    st.session_state["wt_records"] = list(records)
    st.session_state["wt_records_original"] = list(records)
    st.session_state["wt_loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["wt_selected_names"] = [record.name for record in records]
    _clear_pad_state()
    st.session_state["wt_last_error"] = ""
    _clear_results()


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
                "Сбрасывает только параметры расчета и солвера к рекомендованным "
                "значениям. Импортированный WELLTRACK и выбранные скважины не удаляются."
            ),
        )
    return source_text, bool(parse_clicked), bool(clear_clicked), bool(reset_params_clicked)


def _handle_import_actions(
    source_text: str, parse_clicked: bool, clear_clicked: bool, reset_params_clicked: bool
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
            _store_parsed_records(records=records)
            status.write(f"Найдено блоков WELLTRACK: {len(records)}.")
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
            "Последовательность бурения строится по проекции середины (t1+t3)/2 вдоль НДС."
        )
        pad_rows = [
            {
                "Куст": str(pad.pad_id),
                "Скважин": int(len(pad.wells)),
                "S X, м": float(pad.surface.x),
                "S Y, м": float(pad.surface.y),
                "S Z, м": float(pad.surface.z),
                "Авто НДС, deg": float(pad.auto_nds_azimuth_deg),
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
        selected_pad = next((pad for pad in pads if str(pad.pad_id) == selected_id), pads[0])
        config_map = st.session_state.get("wt_pad_configs", {})
        selected_cfg = dict(config_map.get(selected_id, _pad_config_defaults(selected_pad)))

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
                    "Новое S X, м": float(selected_cfg["first_surface_x"] + shift_m * ux),
                    "Новое S Y, м": float(selected_cfg["first_surface_y"] + shift_m * uy),
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
            _clear_results()
            st.toast("Координаты устьев обновлены по параметрам кустов.")
            st.rerun()

        if reset_clicked:
            st.session_state["wt_records"] = list(base_records)
            st.session_state["wt_pad_last_applied_at"] = ""
            _clear_results()
            st.rerun()

        if str(st.session_state.get("wt_pad_last_applied_at", "")):
            st.caption(
                f"Последнее обновление устьев: {st.session_state['wt_pad_last_applied_at']}"
            )


def _sync_selected_names(records: list[WelltrackRecord]) -> list[str]:
    all_names = [record.name for record in records]
    selected_names = [
        name
        for name in st.session_state.get("wt_selected_names", [])
        if name in all_names
    ]
    if not selected_names:
        selected_names = all_names
        st.session_state["wt_selected_names"] = selected_names
    return all_names


def _render_batch_run_form(all_names: list[str]) -> tuple[TrajectoryConfig, bool]:
    with st.form("welltrack_run_form", clear_on_submit=False):
        st.markdown("### Выбор скважин и запуск расчета")
        st.multiselect(
            "Скважины для расчета",
            options=all_names,
            key="wt_selected_names",
            help="Для каждой скважины ожидается ровно 3 точки: S, t1, t3.",
        )
        st.caption(
            "Точки `S/t1/t3` подставляются автоматически из входного WELLTRACK. "
            "Ниже задаются универсальные параметры расчета и солвера."
        )
        config = _build_config_form()
        run_clicked = st.form_submit_button(
            "Рассчитать выбранные скважины",
            type="primary",
            icon=":material/play_arrow:",
        )
    return config, bool(run_clicked)


def _run_batch_if_clicked(
    run_clicked: bool, records: list[WelltrackRecord], config: TrajectoryConfig
) -> None:
    if not run_clicked:
        return
    selected_set = set(st.session_state.get("wt_selected_names", []))
    if not selected_set:
        st.warning("Выберите минимум одну скважину для расчета.")
        return
    batch = WelltrackBatchPlanner(planner=TrajectoryPlanner())
    run_started_s = perf_counter()
    log_lines: list[str] = []
    progress = st.progress(0, text="Подготовка batch-расчета...")
    phase_placeholder = st.empty()
    try:
        with st.spinner("Выполняется расчет WELLTRACK-набора...", show_time=True):
            started = perf_counter()
            start_msg = format_run_log_line(
                run_started_s,
                f"Старт batch-расчета. Выбрано скважин: {len(selected_set)}.",
            )
            log_lines.append(start_msg)
            phase_placeholder.caption(
                f"Старт расчета набора. Выбрано скважин: {len(selected_set)}."
            )

            def on_progress(index: int, total: int, name: str) -> None:
                progress.progress(
                    int((index / max(total, 1)) * 100),
                    text=f"{index}/{total}: {name}",
                )
                phase_placeholder.caption(
                    f"Расчет скважины {index}/{total}: {name}"
                )
                line = format_run_log_line(
                    run_started_s,
                    f"Расчет скважины {index}/{total}: {name}",
                )
                log_lines.append(line)

            summary_rows, successes = batch.evaluate(
                records=records,
                selected_names=selected_set,
                config=config,
                progress_callback=on_progress,
            )
            elapsed_s = perf_counter() - started
            progress.progress(100, text="Batch-расчет завершен.")
            st.session_state["wt_summary_rows"] = summary_rows
            st.session_state["wt_successes"] = successes
            st.session_state["wt_last_error"] = ""
            st.session_state["wt_last_run_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.session_state["wt_last_runtime_s"] = float(elapsed_s)
            done_msg = format_run_log_line(
                run_started_s,
                f"Batch-расчет завершен. Успешно: {len(successes)}, "
                f"ошибок: {len(summary_rows) - len(successes)}. "
                f"Затраченное время: {elapsed_s:.2f} с.",
            )
            log_lines.append(done_msg)
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
        err_msg = format_run_log_line(
            run_started_s,
            f"Ошибка batch-расчета: {summarize_problem_ru(str(exc))}",
        )
        log_lines.append(err_msg)
        phase_placeholder.error("Batch-расчет завершился ошибкой")
    finally:
        st.session_state["wt_last_run_log_lines"] = log_lines
        progress.empty()


def _render_batch_log() -> None:
    render_run_log_panel(st.session_state.get("wt_last_run_log_lines"))


def _render_batch_summary(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    summary_df = WelltrackBatchPlanner.summary_dataframe(summary_rows)
    if not summary_df.empty:
        summary_df = arrow_safe_text_dataframe(summary_df)

    ok_count = (
        int((summary_df["Статус"] == "OK").sum())
        if not summary_df.empty and "Статус" in summary_df
        else 0
    )
    err_count = int(len(summary_df) - ok_count) if not summary_df.empty else 0
    p1, p2, p3, p4 = st.columns(4, gap="small")
    p1.metric("Строк в отчете", f"{len(summary_df)}")
    p2.metric("Успешно", f"{ok_count}")
    p3.metric("Ошибки", f"{err_count}")
    run_time = st.session_state.get("wt_last_runtime_s")
    p4.metric("Время расчета", "—" if run_time is None else f"{float(run_time):.2f} с")
    render_small_note(
        f"Последний запуск: {st.session_state.get('wt_last_run_at', '—')}"
    )

    st.markdown("### Сводка расчета")
    st.dataframe(
        summary_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Скважина": st.column_config.TextColumn("Скважина"),
            "Точек": st.column_config.NumberColumn("Точек", format="%d"),
            "Статус": st.column_config.TextColumn("Статус"),
            "Тип траектории": st.column_config.TextColumn("Тип траектории"),
            "Сложность": st.column_config.TextColumn("Сложность"),
            "Горизонтальный отход t1, м": st.column_config.NumberColumn(
                "Отход t1, м",
                format="%.2f",
            ),
            "Длина HORIZONTAL, м": st.column_config.NumberColumn(
                "HORIZONTAL, м",
                format="%.2f",
            ),
            "INC в t1, deg": st.column_config.NumberColumn("INC t1, deg", format="%.2f"),
            "ЗУ HOLD, deg": st.column_config.NumberColumn("ЗУ HOLD, deg", format="%.2f"),
            "Макс ПИ, deg/10m": st.column_config.NumberColumn(
                "Макс ПИ, deg/10m",
                format="%.2f",
            ),
            "Проблема": st.column_config.TextColumn("Проблема"),
        },
    )
    st.download_button(
        "Скачать сводку (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="welltrack_summary.csv",
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )
    return summary_df


def _render_success_tabs(successes: list[SuccessfulWellPlan]) -> None:
    tab_single, tab_all = st.tabs(["Отдельная скважина", "Все скважины"])
    with tab_single:
        selected_name = st.selectbox("Скважина", options=[item.name for item in successes])
        selected = next(item for item in successes if item.name == selected_name)
        stations = selected.stations
        surface = selected.surface
        t1 = selected.t1
        t3 = selected.t3
        azimuth_deg = float(selected.azimuth_deg)
        md_t1_m = float(selected.md_t1_m)
        cfg = selected.config

        render_trajectory_dls_panel(
            stations=stations,
            surface=surface,
            t1=t1,
            t3=t3,
            md_t1_m=md_t1_m,
            dls_limits=cfg.dls_limits_deg_per_30m,
            title=None,
            border=False,
        )
        render_plan_section_panel(
            stations=stations,
            surface=surface,
            t1=t1,
            t3=t3,
            azimuth_deg=azimuth_deg,
            title=None,
            border=False,
        )

        st.caption(
            f"Скважина `{selected_name}`: t1 отход `{format_distance(float(np.hypot(t1.x - surface.x, t1.y - surface.y)))}`, "
            f"тип `{selected.summary.get('trajectory_type', '—')}`, "
            f"сложность `{selected.summary.get('well_complexity', '—')}`."
        )

    with tab_all:
        c1, c2 = st.columns(2, gap="medium")
        c1.plotly_chart(_all_wells_3d_figure(successes), width="stretch")
        c2.plotly_chart(_all_wells_plan_figure(successes), width="stretch")


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

    if st.session_state.get("wt_last_error"):
        render_solver_diagnostics(st.session_state["wt_last_error"])

    records = st.session_state.get("wt_records")
    if records is None:
        st.info("Загрузите источник и нажмите «Прочитать WELLTRACK».")
        return
    if not records:
        st.warning("В источнике не найдено ни одного WELLTRACK блока.")
        return

    _render_records_overview(records=records)
    _render_raw_records_table(records=records)
    _render_pad_layout_panel(records=records)
    all_names = _sync_selected_names(records=records)
    config, run_clicked = _render_batch_run_form(all_names=all_names)
    _run_batch_if_clicked(run_clicked=run_clicked, records=records, config=config)
    _render_batch_log()

    summary_rows = st.session_state.get("wt_summary_rows")
    successes = st.session_state.get("wt_successes")
    if not summary_rows:
        render_small_note("Результаты расчета появятся после запуска batch-расчета.")
        return
    _render_batch_summary(summary_rows=summary_rows)
    if not successes:
        st.warning("Все выбранные скважины завершились ошибками расчета.")
        return
    _render_success_tabs(successes=successes)


if __name__ == "__main__":
    run_page()
