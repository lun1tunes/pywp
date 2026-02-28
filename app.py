from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from pywp import Point3D, TrajectoryConfig, TrajectoryPlanner
from pywp.classification import interpolate_limits, reference_table_rows
from pywp.models import OBJECTIVE_MAXIMIZE_HOLD, OBJECTIVE_MINIMIZE_BUILD_DLS
from pywp.planner import PlanningError
from pywp.visualization import dls_figure, plan_view_figure, section_view_figure, trajectory_3d_figure

SCENARIOS = {
    "Композитный: короткий вынос": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(600.0, 800.0, 2400.0),
        "t3": Point3D(1500.0, 2000.0, 2500.0),
    },
    "Композитный: средний вынос": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(800.0, 1066.6667, 2400.0),
        "t3": Point3D(2600.0, 3466.6667, 2600.0),
    },
    "Композитный: длинный вынос": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(900.0, 1200.0, 2500.0),
        "t3": Point3D(2800.0, 3733.3333, 2650.0),
    },
}
OBJECTIVE_OPTIONS = {
    OBJECTIVE_MAXIMIZE_HOLD: "Максимизировать длину HOLD",
    OBJECTIVE_MINIMIZE_BUILD_DLS: "Минимизировать DLS на BUILD",
}


def _format_distance(value_m: float) -> str:
    if value_m < 1e-6:
        return "< 1e-6 m"
    if value_m < 1e-3:
        return f"{value_m:.2e} m"
    if value_m < 1.0:
        return f"{value_m:.4f} m"
    return f"{value_m:.2f} m"


def _horizontal_offset_m(point: Point3D, reference: Point3D) -> float:
    dx = float(point.x - reference.x)
    dy = float(point.y - reference.y)
    return float((dx * dx + dy * dy) ** 0.5)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 500px at 20% -20%, #E8F1FF 0%, #F8FAFD 45%, #FAFCFF 100%);
        }
        .block-container {
            max-width: 1650px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            border: 1px solid #D6E4FF;
            background: linear-gradient(135deg, #F1F7FF 0%, #FFFFFF 60%);
            border-radius: 16px;
            padding: 1.05rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 24px rgba(15, 33, 66, 0.06);
        }
        .hero h2 {
            margin: 0 0 0.25rem 0;
            font-size: 1.55rem;
            letter-spacing: 0.01em;
            color: #12355B;
        }
        .hero p {
            margin: 0;
            color: #355070;
            font-size: 0.97rem;
        }
        .small-note {
            color: #54657D;
            font-size: 0.87rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _classification_reference_frame() -> pd.DataFrame:
    return pd.DataFrame(reference_table_rows())


def _classification_interpolated_frame(gv_m: float) -> pd.DataFrame:
    limits = interpolate_limits(gv_m=gv_m)
    return pd.DataFrame(
        [
            {"Параметр": "ГВ (интерполированная), м", "Значение": round(float(limits.gv_m), 2)},
            {
                "Параметр": "Окно обратного направления, м",
                "Значение": f"{limits.reverse_min_m:.1f} - {limits.reverse_max_m:.1f}",
            },
            {"Параметр": "Отход t1: Обычная (до), м", "Значение": round(float(limits.ordinary_offset_max_m), 2)},
            {"Параметр": "Отход t1: Сложная (до), м", "Значение": round(float(limits.complex_offset_max_m), 2)},
            {"Параметр": "ЗУ HOLD: Обычная (до), deg", "Значение": round(float(limits.hold_ordinary_max_deg), 2)},
            {"Параметр": "ЗУ HOLD: Сложная (до), deg", "Значение": round(float(limits.hold_complex_max_deg), 2)},
        ]
    )


def _render_classification_block(gv_m: float, summary: dict[str, float | str] | None = None) -> None:
    st.markdown("### Классификация сложности скважин")
    st.caption("Базовые пороги по ГВ и линейная интерполяция для фактической ГВ t1.")
    st.dataframe(_classification_reference_frame(), width="stretch", hide_index=True)
    st.dataframe(_classification_interpolated_frame(gv_m=gv_m), width="stretch", hide_index=True)
    if summary is not None:
        st.markdown("#### Результат классификации для текущего расчета")
        result_df = pd.DataFrame(
            [
                {"Параметр": "Тип траектории", "Значение": summary.get("trajectory_type", "—")},
                {"Параметр": "Сложность скважины", "Значение": summary.get("well_complexity", "—")},
                {"Параметр": "Сложность по отходу t1", "Значение": summary.get("well_complexity_by_offset", "—")},
                {"Параметр": "Сложность по ЗУ HOLD", "Значение": summary.get("well_complexity_by_hold", "—")},
            ]
        )
        st.dataframe(result_df, width="stretch", hide_index=True)


def _apply_scenario(name: str) -> None:
    values = SCENARIOS[name]
    for prefix, point_key in (("surface", "surface"), ("t1", "t1"), ("t3", "t3")):
        point = values[point_key]
        st.session_state[f"{prefix}_x"] = float(point.x)
        st.session_state[f"{prefix}_y"] = float(point.y)
        st.session_state[f"{prefix}_z"] = float(point.z)


def _init_state() -> None:
    default_scenario = next(iter(SCENARIOS.keys()))
    st.session_state.setdefault("scenario_name", default_scenario)

    if "surface_x" not in st.session_state:
        _apply_scenario(st.session_state["scenario_name"])

    st.session_state.setdefault("md_step", 10.0)
    st.session_state.setdefault("md_control", 2.0)
    st.session_state.setdefault("pos_tol", 2.0)
    st.session_state.setdefault("entry_inc_target", 86.0)
    st.session_state.setdefault("entry_inc_tol", 2.0)
    st.session_state.setdefault("dls_build_min", 0.5)
    st.session_state.setdefault("dls_build_max", 3.0)
    st.session_state.setdefault("kop_min_vertical", 300.0)
    st.session_state.setdefault("objective_mode", OBJECTIVE_MAXIMIZE_HOLD)

    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_built_at", "")
    st.session_state.setdefault("last_input_signature", None)


def _current_input_signature() -> tuple[object, ...]:
    keys = (
        "surface_x",
        "surface_y",
        "surface_z",
        "t1_x",
        "t1_y",
        "t1_z",
        "t3_x",
        "t3_y",
        "t3_z",
        "md_step",
        "md_control",
        "pos_tol",
        "entry_inc_target",
        "entry_inc_tol",
        "dls_build_min",
        "dls_build_max",
        "kop_min_vertical",
    )
    signature = [float(st.session_state[key]) for key in keys]
    signature.append(str(st.session_state["objective_mode"]))
    return tuple(signature)


def _build_points_from_state() -> tuple[Point3D, Point3D, Point3D]:
    surface = Point3D(
        x=float(st.session_state["surface_x"]),
        y=float(st.session_state["surface_y"]),
        z=float(st.session_state["surface_z"]),
    )
    t1 = Point3D(
        x=float(st.session_state["t1_x"]),
        y=float(st.session_state["t1_y"]),
        z=float(st.session_state["t1_z"]),
    )
    t3 = Point3D(
        x=float(st.session_state["t3_x"]),
        y=float(st.session_state["t3_y"]),
        z=float(st.session_state["t3_z"]),
    )
    return surface, t1, t3


def _build_config_from_state() -> TrajectoryConfig:
    dls_build_min = float(st.session_state["dls_build_min"])
    dls_build_max = float(st.session_state["dls_build_max"])

    return TrajectoryConfig(
        md_step_m=float(st.session_state["md_step"]),
        md_step_control_m=float(st.session_state["md_control"]),
        pos_tolerance_m=float(st.session_state["pos_tol"]),
        entry_inc_target_deg=float(st.session_state["entry_inc_target"]),
        entry_inc_tolerance_deg=float(st.session_state["entry_inc_tol"]),
        dls_build_min_deg_per_30m=min(dls_build_min, dls_build_max),
        dls_build_max_deg_per_30m=max(dls_build_min, dls_build_max),
        kop_min_vertical_m=float(st.session_state["kop_min_vertical"]),
        objective_mode=str(st.session_state["objective_mode"]),
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD_REV": max(dls_build_min, dls_build_max),
            "HOLD_REV": 2.0,
            "DROP_REV": max(dls_build_min, dls_build_max),
            "BUILD1": max(dls_build_min, dls_build_max),
            "HOLD": 2.0,
            "BUILD2": max(dls_build_min, dls_build_max),
            "HORIZONTAL": 2.0,
        },
    )


def _run_planner() -> None:
    surface, t1, t3 = _build_points_from_state()
    config = _build_config_from_state()

    planner = TrajectoryPlanner()
    result = planner.plan(surface=surface, t1=t1, t3=t3, config=config)

    st.session_state["last_result"] = {
        "surface": surface,
        "t1": t1,
        "t3": t3,
        "config": config,
        "stations": result.stations,
        "summary": result.summary,
        "azimuth_deg": result.azimuth_deg,
        "md_t1_m": result.md_t1_m,
    }
    st.session_state["last_error"] = ""
    st.session_state["last_built_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["last_input_signature"] = _current_input_signature()


def run_app() -> None:
    st.set_page_config(page_title="Планировщик траектории скважины", layout="wide")

    _init_state()
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
          <h2>Планировщик траектории скважины</h2>
          <p>Профиль фиксирован по инженерным шаблонам. Для близких t1 используется режим «Цели в обратном направлении» с дополнительными сегментами BUILD_REV/HOLD_REV/DROP_REV, затем BUILD1 -> HOLD -> BUILD2 -> HORIZONTAL.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_mid, top_right = st.columns([1.5, 1.1, 3.0], gap="small")
    with top_left:
        st.selectbox("Шаблон", options=list(SCENARIOS.keys()), key="scenario_name")
    with top_mid:
        st.markdown("<div class='small-note'>Действия шаблона</div>", unsafe_allow_html=True)
        if st.button("Применить шаблон", icon=":material/sync:", width="stretch"):
            _apply_scenario(st.session_state["scenario_name"])
            st.rerun()
        if st.button("Очистить результат", icon=":material/delete:", width="stretch"):
            st.session_state["last_result"] = None
            st.session_state["last_error"] = ""
            st.session_state["last_built_at"] = ""
            st.session_state["last_input_signature"] = None
            st.rerun()
    with top_right:
        st.markdown(
            "<div class='small-note'>Подсказка: измените параметры в форме и нажмите «Построить траекторию». "
            "Последний успешный расчет сохраняется на экране до следующего запуска.</div>",
            unsafe_allow_html=True,
        )

    with st.form("planner_form", clear_on_submit=False, enter_to_submit=False, border=False):
        with st.container(border=True):
            st.markdown("### Рабочая область ввода")
            c0, c1, c2, c3, c4 = st.columns(
                [1.0, 1.25, 1.25, 1.25, 2.35],
                gap="small",
                border=True,
                vertical_alignment="top",
            )

            with c0:
                st.markdown("**Расчет**")
                build_clicked = st.form_submit_button(
                    "Построить траекторию",
                    type="primary",
                    icon=":material/play_arrow:",
                    width="stretch",
                )
                if st.session_state["last_built_at"]:
                    st.caption(f"Последний расчет: {st.session_state['last_built_at']}")

            with c1:
                st.markdown("**Устье S**")
                st.number_input("S X (E), m", key="surface_x", step=10.0)
                st.number_input("S Y (N), m", key="surface_y", step=10.0)
                st.number_input("S Z (TVD), m", key="surface_z", step=10.0)

            with c2:
                st.markdown("**Точка входа t1**")
                st.number_input("t1 X (E), m", key="t1_x", step=10.0)
                st.number_input("t1 Y (N), m", key="t1_y", step=10.0)
                st.number_input("t1 Z (TVD), m", key="t1_z", step=10.0)

            with c3:
                st.markdown("**Концевая точка t3**")
                st.number_input("t3 X (E), m", key="t3_x", step=10.0)
                st.number_input("t3 Y (N), m", key="t3_y", step=10.0)
                st.number_input("t3 Z (TVD), m", key="t3_z", step=10.0)

            with c4:
                st.markdown("**Ограничения профиля**")
                r1, r2, r3 = st.columns(3, gap="small")
                r1.number_input("Шаг MD", key="md_step", min_value=1.0, step=1.0)
                r2.number_input("Контрольный шаг MD", key="md_control", min_value=0.5, step=0.5)
                r3.number_input("Допуск по позиции", key="pos_tol", min_value=0.1, step=0.1)

                rr1, rr2 = st.columns(2, gap="small")
                rr1.number_input("Целевой INC входа", key="entry_inc_target", min_value=70.0, max_value=89.0, step=0.5)
                rr2.number_input("Допуск INC входа", key="entry_inc_tol", min_value=0.1, max_value=5.0, step=0.1)

                rd1, rd2 = st.columns(2, gap="small")
                rd1.number_input("Мин DLS BUILD", key="dls_build_min", min_value=0.1, step=0.1, help="deg/30m")
                rd2.number_input(
                    "Макс DLS BUILD",
                    key="dls_build_max",
                    min_value=0.1,
                    step=0.1,
                    help="для BUILD1 и BUILD2",
                )
                st.number_input(
                    "Мин VERTICAL до KOP, м",
                    key="kop_min_vertical",
                    min_value=0.0,
                    step=10.0,
                    help="Минимальный VERTICAL участок от S до начала BUILD1.",
                )
                st.selectbox(
                    "Целевая функция",
                    options=list(OBJECTIVE_OPTIONS.keys()),
                    index=list(OBJECTIVE_OPTIONS.keys()).index(str(st.session_state["objective_mode"])),
                    key="objective_mode",
                    format_func=lambda value: OBJECTIVE_OPTIONS[str(value)],
                )

    if build_clicked:
        try:
            _run_planner()
        except PlanningError as exc:
            st.session_state["last_error"] = str(exc)

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])

    last_result = st.session_state.get("last_result")
    if last_result is None:
        st.info("Задайте параметры и нажмите «Построить траекторию».")
        surface, t1, _ = _build_points_from_state()
        gv_m = float(t1.z - surface.z)
        _render_classification_block(gv_m=gv_m, summary=None)
        return

    is_stale = st.session_state.get("last_input_signature") != _current_input_signature()
    if is_stale:
        st.warning("Параметры изменились после последнего расчета. Показан предыдущий результат. Нажмите «Построить траекторию» для обновления.")

    summary = last_result["summary"]
    stations = last_result["stations"]
    surface = last_result["surface"]
    t1 = last_result["t1"]
    t3 = last_result["t3"]
    config = last_result["config"]
    azimuth_deg = float(last_result["azimuth_deg"])
    md_t1_m = float(last_result["md_t1_m"])
    t1_horizontal_offset_m = _horizontal_offset_m(point=t1, reference=surface)

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Горизонтальный отход t1", _format_distance(t1_horizontal_offset_m))
    m2.metric("Угол входа в пласт", f"{summary['entry_inc_deg']:.2f} deg")
    m3.metric("ЗУ секции HOLD", f"{float(summary['hold_inc_deg']):.2f} deg")
    m4.metric("Сложность", str(summary["well_complexity"]))

    m5, m6, m7 = st.columns(3, gap="small")
    m5.metric("Тип траектории", str(summary["trajectory_type"]))
    m6.metric("KOP MD", _format_distance(float(summary["kop_md_m"])))
    m7.metric("Макс DLS", f"{summary['max_dls_total_deg_per_30m']:.2f} deg/30m")

    with st.container(border=True):
        st.markdown("### 3D траектория и DLS")
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row1_col1.plotly_chart(
            trajectory_3d_figure(stations, surface=surface, t1=t1, t3=t3, md_t1_m=md_t1_m),
            width="stretch",
        )
        row1_col2.plotly_chart(
            dls_figure(stations, dls_limits=config.dls_limits_deg_per_30m),
            width="stretch",
        )

    with st.container(border=True):
        st.markdown("### План и вертикальный разрез")
        row2_col1, row2_col2 = st.columns(2, gap="medium")
        row2_col1.plotly_chart(
            plan_view_figure(stations, surface=surface, t1=t1, t3=t3),
            width="stretch",
        )
        row2_col2.plotly_chart(
            section_view_figure(stations, surface=surface, azimuth_deg=azimuth_deg, t1=t1, t3=t3),
            width="stretch",
        )

    tab_summary, tab_survey, tab_classification = st.tabs(
        ["Сводка", "Инклинометрия", "Классификация сложности скважин"]
    )
    with tab_summary:
        hidden_metrics = {"distance_t1_m", "distance_t3_m"}
        summary_visible = {key: value for key, value in summary.items() if key not in hidden_metrics}
        summary_visible["t1_horizontal_offset_m"] = t1_horizontal_offset_m
        summary_df = pd.DataFrame({"metric": list(summary_visible.keys()), "value": list(summary_visible.values())})
        st.dataframe(summary_df, width="stretch", hide_index=True)

    with tab_survey:
        st.dataframe(stations, width="stretch")
        st.download_button(
            "Скачать CSV инклинометрии",
            data=stations.to_csv(index=False).encode("utf-8"),
            file_name="well_survey.csv",
            mime="text/csv",
            icon=":material/download:",
            width="content",
        )

    with tab_classification:
        gv_m = float(t1.z - surface.z)
        _render_classification_block(gv_m=gv_m, summary=summary)


if __name__ == "__main__":
    if not st.runtime.exists():
        raise SystemExit("Запустите приложение командой `streamlit run app.py`.")
    run_app()
