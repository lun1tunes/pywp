from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pywp import TrajectoryConfig, TrajectoryPlanner
from pywp.eclipse_welltrack import WelltrackParseError, parse_welltrack_text
from pywp.models import OBJECTIVE_MAXIMIZE_HOLD, OBJECTIVE_MINIMIZE_BUILD_DLS
from pywp.ui_utils import arrow_safe_text_dataframe, format_distance
from pywp.visualization import dls_figure, plan_view_figure, section_view_figure, trajectory_3d_figure
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner

OBJECTIVE_OPTIONS = {
    OBJECTIVE_MAXIMIZE_HOLD: "Максимизировать длину HOLD",
    OBJECTIVE_MINIMIZE_BUILD_DLS: "Минимизировать DLS на BUILD",
}
DEFAULT_WELLTRACK_PATH = Path("tests/test_data/WELLTRACKS.INC")


def _hash_text(value: str) -> int:
    return hash(value)


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


def _all_wells_3d_figure(successes: list[SuccessfulWellPlan], height: int = 620) -> go.Figure:
    fig = go.Figure()
    for item in successes:
        name = item.name
        stations = item.stations
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
        fig.add_trace(
            go.Scatter3d(
                x=[surface.x, t1.x, t3.x],
                y=[surface.y, t1.y, t3.y],
                z=[surface.z, t1.z, t3.z],
                mode="markers",
                name=f"{name}: S/t1/t3",
                marker={"size": 4},
                showlegend=False,
                hovertemplate=(
                    "X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z/TVD: %{z:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        title="Все рассчитанные скважины (3D)",
        scene={
            "xaxis_title": "X / Восток (м)",
            "yaxis_title": "Y / Север (м)",
            "zaxis_title": "Z / TVD (м)",
            "aspectmode": "data",
        },
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def _all_wells_plan_figure(successes: list[SuccessfulWellPlan], height: int = 560) -> go.Figure:
    fig = go.Figure()
    for item in successes:
        name = item.name
        stations = item.stations
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
    fig.update_layout(
        title="Все рассчитанные скважины (план E-N)",
        xaxis_title="Восток (м)",
        yaxis_title="Север (м)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def _build_config() -> TrajectoryConfig:
    st.markdown("### Параметры расчета")
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    md_step_m = c1.number_input("Шаг MD", min_value=1.0, value=10.0, step=1.0)
    md_step_control_m = c2.number_input("Контрольный шаг MD", min_value=0.5, value=2.0, step=0.5)
    pos_tolerance_m = c3.number_input("Допуск по позиции, м", min_value=0.1, value=2.0, step=0.1)
    entry_inc_target_deg = c4.number_input("Целевой INC входа", min_value=70.0, max_value=89.0, value=86.0, step=0.5)
    entry_inc_tolerance_deg = c5.number_input("Допуск INC", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

    d1, d2, d3, d4 = st.columns(4, gap="small")
    dls_build_min = d1.number_input("Мин DLS BUILD", min_value=0.1, value=0.5, step=0.1)
    dls_build_max = d2.number_input("Макс DLS BUILD", min_value=0.1, value=3.0, step=0.1)
    kop_min_vertical_m = d3.number_input("Мин VERTICAL до KOP, м", min_value=0.0, value=300.0, step=10.0)
    objective_mode = d4.selectbox(
        "Целевая функция",
        options=list(OBJECTIVE_OPTIONS.keys()),
        index=0,
        format_func=lambda key: OBJECTIVE_OPTIONS[str(key)],
    )

    max_build = max(float(dls_build_min), float(dls_build_max))
    min_build = min(float(dls_build_min), float(dls_build_max))
    return TrajectoryConfig(
        md_step_m=float(md_step_m),
        md_step_control_m=float(md_step_control_m),
        pos_tolerance_m=float(pos_tolerance_m),
        entry_inc_target_deg=float(entry_inc_target_deg),
        entry_inc_tolerance_deg=float(entry_inc_tolerance_deg),
        dls_build_min_deg_per_30m=min_build,
        dls_build_max_deg_per_30m=max_build,
        kop_min_vertical_m=float(kop_min_vertical_m),
        objective_mode=str(objective_mode),
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD_REV": max_build,
            "HOLD_REV": 2.0,
            "DROP_REV": max_build,
            "BUILD1": max_build,
            "HOLD": 2.0,
            "BUILD2": max_build,
            "HORIZONTAL": 2.0,
        },
    )


def _parse_source_text() -> str:
    st.markdown("### Источник WELLTRACK")
    source_mode = st.radio(
        "Режим загрузки",
        options=["Файл по пути", "Загрузить файл", "Вставить текст"],
        horizontal=True,
    )

    if source_mode == "Файл по пути":
        source_path = st.text_input(
            "Путь к файлу WELLTRACK",
            value=str(DEFAULT_WELLTRACK_PATH),
            placeholder="tests/test_data/WELLTRACKS.INC",
        )
        return _read_welltrack_file(source_path)

    if source_mode == "Загрузить файл":
        uploaded_file = st.file_uploader("Файл ECLIPSE/INC", type=["inc", "txt", "data", "ecl"])
        if uploaded_file is None:
            return ""
        return uploaded_file.getvalue().decode("utf-8", errors="replace")

    return st.text_area("Текст WELLTRACK", height=220, placeholder="WELLTRACK 'WELL-1' ...")


def run_page() -> None:
    st.set_page_config(page_title="Импорт WELLTRACK (ECLIPSE)", layout="wide")
    st.title("Импорт WELLTRACK (ECLIPSE)")
    st.caption(
        "Поддерживается формат `WELLTRACK <имя>` с точками `X Y Z MD`, комментарии `--`, "
        "завершение скважины `/` или `;`."
    )

    source_text = _parse_source_text()
    if not source_text.strip():
        st.info("Загрузите файл или вставьте текст с WELLTRACK.")
        return

    st.caption(f"Контрольная сумма содержимого: `{_hash_text(source_text)}`")

    try:
        records = parse_welltrack_text(source_text)
    except WelltrackParseError as exc:
        st.error(f"Ошибка парсинга WELLTRACK: {exc}")
        return

    if not records:
        st.warning("В файле не найдено ни одного блока WELLTRACK.")
        return

    parsed_df = pd.DataFrame(
        [{"Скважина": record.name, "Точек": len(record.points), "Готова к расчету (3 точки)": len(record.points) == 3} for record in records]
    )
    st.markdown("### Загруженные скважины")
    st.dataframe(arrow_safe_text_dataframe(parsed_df), width="stretch", hide_index=True)

    names = [record.name for record in records]
    selected_names = st.multiselect("Скважины для расчета", options=names, default=names)
    if not selected_names:
        st.info("Выберите минимум одну скважину для расчета.")
        return

    config = _build_config()
    batch = WelltrackBatchPlanner(planner=TrajectoryPlanner())
    summary_rows, successes = batch.evaluate(records=records, selected_names=set(selected_names), config=config)
    summary_df = batch.summary_dataframe(summary_rows)
    if not summary_df.empty:
        summary_df = arrow_safe_text_dataframe(summary_df)

    st.markdown("### Результат расчета")
    if summary_df.empty:
        st.warning("Нет результатов расчета для выбранных скважин.")
        return
    st.dataframe(summary_df, width="stretch", hide_index=True)

    if not successes:
        st.warning("Все выбранные скважины завершились с ошибками расчета.")
        return

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
        config = selected.config

        c1, c2 = st.columns(2, gap="medium")
        c1.plotly_chart(
            trajectory_3d_figure(stations, surface=surface, t1=t1, t3=t3, md_t1_m=md_t1_m),
            width="stretch",
        )
        c2.plotly_chart(dls_figure(stations, dls_limits=config.dls_limits_deg_per_30m), width="stretch")

        c3, c4 = st.columns(2, gap="medium")
        c3.plotly_chart(plan_view_figure(stations, surface=surface, t1=t1, t3=t3), width="stretch")
        c4.plotly_chart(
            section_view_figure(stations, surface=surface, azimuth_deg=azimuth_deg, t1=t1, t3=t3),
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
