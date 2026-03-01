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
    parse_welltrack_text,
)
from pywp.planner_config import (
    CFG_DEFAULTS,
    OBJECTIVE_OPTIONS,
    TURN_SOLVER_OPTIONS,
    build_trajectory_config,
)
from pywp.plot_axes import equalized_axis_ranges, linear_tick_values, nice_tick_step
from pywp.solver_diagnostics_ui import render_solver_diagnostics
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_utils import arrow_safe_text_dataframe, format_distance, format_run_log_line
from pywp.visualization import (
    dls_figure,
    plan_view_figure,
    section_view_figure,
    trajectory_3d_figure,
)
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner

DEFAULT_WELLTRACK_PATH = Path("tests/test_data/WELLTRACKS.INC")
WT_UI_DEFAULTS_VERSION = 7
WT_PROFILE_DEFAULTS: dict[str, float | int | str] = {
    "wt_cfg_md_step_m": float(CFG_DEFAULTS.md_step_m),
    "wt_cfg_md_step_control_m": float(CFG_DEFAULTS.md_step_control_m),
    "wt_cfg_pos_tolerance_m": float(CFG_DEFAULTS.pos_tolerance_m),
    "wt_cfg_entry_inc_target_deg": float(CFG_DEFAULTS.entry_inc_target_deg),
    "wt_cfg_entry_inc_tolerance_deg": float(CFG_DEFAULTS.entry_inc_tolerance_deg),
    "wt_cfg_max_inc_deg": float(CFG_DEFAULTS.max_inc_deg),
    "wt_cfg_dls_build_min": float(CFG_DEFAULTS.dls_build_min_deg_per_30m),
    "wt_cfg_dls_build_max": float(CFG_DEFAULTS.dls_build_max_deg_per_30m),
    "wt_cfg_kop_min_vertical_m": float(CFG_DEFAULTS.kop_min_vertical_m),
    "wt_cfg_objective_mode": str(CFG_DEFAULTS.objective_mode),
    "wt_cfg_turn_solver_mode": str(CFG_DEFAULTS.turn_solver_mode),
    "wt_cfg_turn_solver_qmc_samples": int(CFG_DEFAULTS.turn_solver_qmc_samples),
    "wt_cfg_turn_solver_local_starts": int(CFG_DEFAULTS.turn_solver_local_starts),
}


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
    st.session_state.setdefault("wt_selected_names", [])
    st.session_state.setdefault("wt_loaded_at", "")

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


def _apply_profile_defaults(force: bool) -> None:
    for key, value in WT_PROFILE_DEFAULTS.items():
        if force or key not in st.session_state:
            st.session_state[key] = value


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
        return file_path.read_text(encoding="utf-8")
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
    for item in successes:
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
                line={"width": 5},
                customdata=np.column_stack(
                    [
                        stations["MD_m"].to_numpy(dtype=float),
                        stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float),
                    ]
                ),
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{z:.2f} m<br>"
                    "MD: %{customdata[0]:.2f} m<br>"
                    "DLS: %{customdata[1]:.2f} deg/30m"
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
                marker={"size": 4},
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
    for item in successes:
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
                line={"width": 4},
                customdata=np.column_stack(
                    [
                        stations["Z_m"].to_numpy(dtype=float),
                        stations["MD_m"].to_numpy(dtype=float),
                        stations["DLS_deg_per_30m"].fillna(0.0).to_numpy(dtype=float),
                    ]
                ),
                hovertemplate=(
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Z/TVD: %{customdata[0]:.2f} m<br>"
                    "MD: %{customdata[1]:.2f} m<br>"
                    "DLS: %{customdata[2]:.2f} deg/30m"
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
    st.markdown("### Параметры расчета")
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    md_step_m = c1.number_input(
        "Шаг MD",
        key="wt_cfg_md_step_m",
        min_value=1.0,
        step=1.0,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_md_step_m"]),
    )
    md_step_control_m = c2.number_input(
        "Контрольный шаг MD",
        key="wt_cfg_md_step_control_m",
        min_value=0.5,
        step=0.5,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_md_step_control_m"]),
    )
    pos_tolerance_m = c3.number_input(
        "Допуск по позиции, м",
        key="wt_cfg_pos_tolerance_m",
        min_value=0.1,
        step=0.1,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_pos_tolerance_m"]),
    )
    entry_inc_target_deg = c4.number_input(
        "Целевой INC входа в пласт",
        key="wt_cfg_entry_inc_target_deg",
        min_value=70.0,
        max_value=89.0,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_entry_inc_target_deg"]),
        step=0.5,
    )
    entry_inc_tolerance_deg = c5.number_input(
        "Допуск INC",
        key="wt_cfg_entry_inc_tolerance_deg",
        min_value=0.1,
        max_value=5.0,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_entry_inc_tolerance_deg"]),
        step=0.1,
    )
    max_inc_deg = c6.number_input(
        "Макс INC по стволу",
        key="wt_cfg_max_inc_deg",
        min_value=80.0,
        max_value=120.0,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_max_inc_deg"]),
        step=0.5,
        help="Глобальное ограничение по зенитному углу. При недостатке этого лимита расчет потребует overbend.",
    )

    d1, d2, d3 = st.columns(3, gap="small")
    dls_build_min = d1.number_input(
        "Мин DLS BUILD",
        key="wt_cfg_dls_build_min",
        min_value=0.1,
        step=0.1,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_dls_build_min"]),
    )
    dls_build_max = d2.number_input(
        "Макс DLS BUILD",
        key="wt_cfg_dls_build_max",
        min_value=0.1,
        step=0.1,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_dls_build_max"]),
    )
    kop_min_vertical_m = d3.number_input(
        "Мин VERTICAL до KOP, м",
        key="wt_cfg_kop_min_vertical_m",
        min_value=0.0,
        step=10.0,
        value=float(WT_PROFILE_DEFAULTS["wt_cfg_kop_min_vertical_m"]),
    )

    with st.expander("Параметры солвера", expanded=False):
        e1, e2, e3 = st.columns(3, gap="small")
        objective_mode = e1.selectbox(
            "Целевая функция",
            options=list(OBJECTIVE_OPTIONS.keys()),
            key="wt_cfg_objective_mode",
            format_func=lambda key: OBJECTIVE_OPTIONS[str(key)],
            help=(
                "Определяет приоритет оптимизации среди допустимых решений. "
                "Рекомендуется «Максимизировать длину HOLD»: как правило, дает более устойчивую "
                "траекторию и снижает потребность в ручной подстройке."
            ),
        )
        turn_solver_mode = e2.selectbox(
            "Метод TURN-решателя",
            options=list(TURN_SOLVER_OPTIONS.keys()),
            key="wt_cfg_turn_solver_mode",
            format_func=lambda key: TURN_SOLVER_OPTIONS[str(key)],
            help=(
                "Least Squares (TRF) — быстрый и стабильный метод для большинства задач "
                "(рекомендуемый дефолт). DE Hybrid — более тяжелый по времени, но полезен "
                "для самых сложных геометрий."
            ),
        )
        e3.number_input(
            "TURN QMC samples",
            key="wt_cfg_turn_solver_qmc_samples",
            min_value=0,
            value=int(WT_PROFILE_DEFAULTS["wt_cfg_turn_solver_qmc_samples"]),
            step=4,
            help=(
                "Количество дополнительных стартовых точек (Latin Hypercube). "
                "Больше — выше устойчивость в сложных кейсах, но медленнее расчет. "
                "Дефолт 24 — оптимальный баланс."
            ),
        )
        st.number_input(
            "TURN local starts",
            key="wt_cfg_turn_solver_local_starts",
            min_value=1,
            value=int(WT_PROFILE_DEFAULTS["wt_cfg_turn_solver_local_starts"]),
            step=1,
            help=(
                "Сколько лучших стартовых точек запускать локальным решателем. "
                "Больше — выше шанс попасть в хорошее решение, но выше время расчета. "
                "Дефолт 12 — рабочий стандарт."
            ),
        )

    return build_trajectory_config(
        md_step_m=float(md_step_m),
        md_step_control_m=float(md_step_control_m),
        pos_tolerance_m=float(pos_tolerance_m),
        entry_inc_target_deg=float(entry_inc_target_deg),
        entry_inc_tolerance_deg=float(entry_inc_tolerance_deg),
        max_inc_deg=float(max_inc_deg),
        dls_build_min_deg_per_30m=float(dls_build_min),
        dls_build_max_deg_per_30m=float(dls_build_max),
        kop_min_vertical_m=float(kop_min_vertical_m),
        objective_mode=str(objective_mode),
        turn_solver_mode=str(turn_solver_mode),
        turn_solver_qmc_samples=int(st.session_state["wt_cfg_turn_solver_qmc_samples"]),
        turn_solver_local_starts=int(
            st.session_state["wt_cfg_turn_solver_local_starts"]
        ),
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
            placeholder="tests/test_data/WELLTRACKS.INC",
        )
        return _read_welltrack_file(source_path)

    if source_mode == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Файл ECLIPSE/INC", type=["inc", "txt", "data", "ecl"]
        )
        if uploaded_file is None:
            return ""
        return uploaded_file.getvalue().decode("utf-8", errors="replace")

    return st.text_area(
        "Текст WELLTRACK",
        key="wt_source_inline",
        height=220,
        placeholder="WELLTRACK 'WELL-1' ...",
    )


def _store_parsed_records(records: list[WelltrackRecord]) -> None:
    st.session_state["wt_records"] = records
    st.session_state["wt_loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["wt_selected_names"] = [record.name for record in records]
    st.session_state["wt_last_error"] = ""
    _clear_results()


def run_page() -> None:
    st.set_page_config(page_title="Импорт WELLTRACK", layout="wide")
    _init_state()
    apply_page_style(max_width_px=1700)
    render_hero(title="Импорт WELLTRACK", subtitle="")

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

    if reset_params_clicked:
        _apply_profile_defaults(force=True)
        st.toast("Параметры расчета сброшены к рекомендованным.")

    if clear_clicked:
        st.session_state["wt_records"] = None
        st.session_state["wt_selected_names"] = []
        st.session_state["wt_loaded_at"] = ""
        _clear_results()
        st.rerun()

    if parse_clicked:
        if not source_text.strip():
            st.warning("Источник пустой. Загрузите файл или вставьте текст WELLTRACK.")
        else:
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
                    st.session_state["wt_last_error"] = str(exc)
                    status.write(str(exc))
                    status.update(
                        label="Ошибка парсинга WELLTRACK", state="error", expanded=True
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
    with st.expander("Исходные данные скважин (вход WELLTRACK)", expanded=False):
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

    all_names = [record.name for record in records]
    selected_names = [
        name
        for name in st.session_state.get("wt_selected_names", [])
        if name in all_names
    ]
    if not selected_names:
        selected_names = all_names
        st.session_state["wt_selected_names"] = selected_names

    with st.form("welltrack_run_form", clear_on_submit=False):
        st.markdown("### Выбор скважин и запуск расчета")
        st.multiselect(
            "Скважины для расчета",
            options=all_names,
            key="wt_selected_names",
            help="Для каждой скважины ожидается ровно 3 точки: S, t1, t3.",
        )
        config = _build_config_form()
        run_clicked = st.form_submit_button(
            "Рассчитать выбранные скважины",
            type="primary",
            icon=":material/play_arrow:",
        )

    if run_clicked:
        selected_set = set(st.session_state.get("wt_selected_names", []))
        if not selected_set:
            st.warning("Выберите минимум одну скважину для расчета.")
        else:
            batch = WelltrackBatchPlanner(planner=TrajectoryPlanner())
            run_started_s = perf_counter()
            log_lines: list[str] = []
            progress = st.progress(0, text="Подготовка batch-расчета...")
            with st.status(
                "Выполняется расчет WELLTRACK-набора...", expanded=True
            ) as status:
                try:
                    started = perf_counter()
                    start_msg = format_run_log_line(
                        run_started_s,
                        f"Старт batch-расчета. Выбрано скважин: {len(selected_set)}.",
                    )
                    status.write(start_msg)
                    log_lines.append(start_msg)

                    def on_progress(index: int, total: int, name: str) -> None:
                        progress.progress(
                            int((index / max(total, 1)) * 100),
                            text=f"{index}/{total}: {name}",
                        )
                        line = format_run_log_line(
                            run_started_s,
                            f"Расчет скважины {index}/{total}: {name}",
                        )
                        status.write(line)
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
                    status.write(done_msg)
                    log_lines.append(done_msg)
                    if successes:
                        status.update(
                            label=f"Расчет завершен за {elapsed_s:.2f} с. Успешно: {len(successes)}",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status.update(
                            label=f"Расчет завершен за {elapsed_s:.2f} с, но без успешных скважин.",
                            state="error",
                            expanded=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["wt_last_error"] = str(exc)
                    err_msg = format_run_log_line(
                        run_started_s, f"Ошибка batch-расчета: {exc}"
                    )
                    status.write(err_msg)
                    log_lines.append(err_msg)
                    status.update(
                        label="Batch-расчет завершился ошибкой",
                        state="error",
                        expanded=True,
                    )
                finally:
                    st.session_state["wt_last_run_log_lines"] = log_lines
                    progress.empty()

    summary_rows = st.session_state.get("wt_summary_rows")
    successes = st.session_state.get("wt_successes")
    run_log_lines = st.session_state.get("wt_last_run_log_lines")
    if run_log_lines:
        with st.container(border=True):
            st.markdown("### Лог расчета")
            st.code("\n".join(run_log_lines), language="text")

    if not summary_rows:
        render_small_note("Результаты расчета появятся после запуска batch-расчета.")
        return

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
    st.dataframe(summary_df, width="stretch", hide_index=True)
    st.download_button(
        "Скачать сводку (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="welltrack_summary.csv",
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )

    if not successes:
        st.warning("Все выбранные скважины завершились ошибками расчета.")
        return

    tab_single, tab_all = st.tabs(["Отдельная скважина", "Все скважины"])
    with tab_single:
        selected_name = st.selectbox(
            "Скважина", options=[item.name for item in successes]
        )
        selected = next(item for item in successes if item.name == selected_name)
        stations = selected.stations
        surface = selected.surface
        t1 = selected.t1
        t3 = selected.t3
        azimuth_deg = float(selected.azimuth_deg)
        md_t1_m = float(selected.md_t1_m)
        cfg = selected.config

        c1, c2 = st.columns(2, gap="medium")
        c1.plotly_chart(
            trajectory_3d_figure(
                stations, surface=surface, t1=t1, t3=t3, md_t1_m=md_t1_m
            ),
            width="stretch",
        )
        c2.plotly_chart(
            dls_figure(stations, dls_limits=cfg.dls_limits_deg_per_30m), width="stretch"
        )

        c3, c4 = st.columns(2, gap="medium")
        c3.plotly_chart(
            plan_view_figure(stations, surface=surface, t1=t1, t3=t3), width="stretch"
        )
        c4.plotly_chart(
            section_view_figure(
                stations, surface=surface, azimuth_deg=azimuth_deg, t1=t1, t3=t3
            ),
            width="stretch",
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


if __name__ == "__main__":
    run_page()
