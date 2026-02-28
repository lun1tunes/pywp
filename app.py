from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from time import perf_counter

import pandas as pd
import streamlit as st

from pywp import Point3D, TrajectoryConfig, TrajectoryPlanner
from pywp.models import (
    OBJECTIVE_MAXIMIZE_HOLD,
    OBJECTIVE_MINIMIZE_BUILD_DLS,
    TURN_SOLVER_DE_HYBRID,
    TURN_SOLVER_LEAST_SQUARES,
)
from pywp.planner import PlanningError
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_utils import arrow_safe_text_dataframe, format_distance
from pywp.visualization import (
    dls_figure,
    plan_view_figure,
    section_view_figure,
    trajectory_3d_figure,
)

SCENARIOS = {
    "default": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(600.0, 800.0, 2400.0),
        "t3": Point3D(1500.0, 2000.0, 2500.0),
    },
    "Типовой: ГВ 3000, обратное направление, обычная": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(300.0, 400.0, 3000.0),
        "t3": Point3D(1020.0, 1360.0, 3083.9122),
    },
    "Типовой: ГВ 2000, обратное направление, очень сложная": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(210.0, 280.0, 2000.0),
        "t3": Point3D(810.0, 1080.0, 2069.9268),
    },
    "Типовой: ГВ 2000, прямое направление, обычная": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(540.0, 720.0, 2000.0),
        "t3": Point3D(1140.0, 1520.0, 2069.9268),
    },
    "Типовой: ГВ 3600, прямое направление, сложная": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(1500.0, 2000.0, 3600.0),
        "t3": Point3D(2520.0, 3360.0, 3718.8756),
    },
    "Типовой: ГВ 3600, прямое направление, очень сложная": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(2160.0, 2880.0, 3600.0),
        "t3": Point3D(3360.0, 4480.0, 3739.8536),
    },
}
OBJECTIVE_OPTIONS = {
    OBJECTIVE_MAXIMIZE_HOLD: "Максимизировать длину HOLD",
    OBJECTIVE_MINIMIZE_BUILD_DLS: "Минимизировать DLS на BUILD",
}
TURN_SOLVER_OPTIONS = {
    TURN_SOLVER_LEAST_SQUARES: "Least Squares (TRF, рекомендуется)",
    TURN_SOLVER_DE_HYBRID: "DE Hybrid (глобальный + локальный)",
}
CFG_DEFAULTS = TrajectoryConfig()
UI_DEFAULTS_VERSION = 4


def _horizontal_offset_m(point: Point3D, reference: Point3D) -> float:
    dx = float(point.x - reference.x)
    dy = float(point.y - reference.y)
    return float((dx * dx + dy * dy) ** 0.5)


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

    st.session_state.setdefault("md_step", float(CFG_DEFAULTS.md_step_m))
    st.session_state.setdefault("md_control", float(CFG_DEFAULTS.md_step_control_m))
    st.session_state.setdefault("pos_tol", float(CFG_DEFAULTS.pos_tolerance_m))
    st.session_state.setdefault(
        "entry_inc_target", float(CFG_DEFAULTS.entry_inc_target_deg)
    )
    st.session_state.setdefault(
        "entry_inc_tol", float(CFG_DEFAULTS.entry_inc_tolerance_deg)
    )
    st.session_state.setdefault(
        "dls_build_min", float(CFG_DEFAULTS.dls_build_min_deg_per_30m)
    )
    st.session_state.setdefault(
        "dls_build_max", float(CFG_DEFAULTS.dls_build_max_deg_per_30m)
    )
    st.session_state.setdefault(
        "kop_min_vertical", float(CFG_DEFAULTS.kop_min_vertical_m)
    )
    st.session_state.setdefault("objective_mode", str(CFG_DEFAULTS.objective_mode))
    st.session_state.setdefault("turn_solver_mode", str(CFG_DEFAULTS.turn_solver_mode))
    st.session_state.setdefault(
        "turn_solver_qmc_samples", int(CFG_DEFAULTS.turn_solver_qmc_samples)
    )
    st.session_state.setdefault(
        "turn_solver_local_starts", int(CFG_DEFAULTS.turn_solver_local_starts)
    )
    st.session_state.setdefault("ui_defaults_version", 0)

    if int(st.session_state.get("ui_defaults_version", 0)) < UI_DEFAULTS_VERSION:
        st.session_state["md_step"] = float(CFG_DEFAULTS.md_step_m)
        st.session_state["md_control"] = float(CFG_DEFAULTS.md_step_control_m)
        st.session_state["pos_tol"] = float(CFG_DEFAULTS.pos_tolerance_m)
        st.session_state["entry_inc_target"] = float(CFG_DEFAULTS.entry_inc_target_deg)
        st.session_state["entry_inc_tol"] = float(CFG_DEFAULTS.entry_inc_tolerance_deg)
        st.session_state["dls_build_min"] = float(
            CFG_DEFAULTS.dls_build_min_deg_per_30m
        )
        st.session_state["dls_build_max"] = float(
            CFG_DEFAULTS.dls_build_max_deg_per_30m
        )
        st.session_state["kop_min_vertical"] = float(CFG_DEFAULTS.kop_min_vertical_m)
        st.session_state["objective_mode"] = str(CFG_DEFAULTS.objective_mode)
        st.session_state["turn_solver_mode"] = str(CFG_DEFAULTS.turn_solver_mode)
        st.session_state["turn_solver_qmc_samples"] = int(
            CFG_DEFAULTS.turn_solver_qmc_samples
        )
        st.session_state["turn_solver_local_starts"] = int(
            CFG_DEFAULTS.turn_solver_local_starts
        )
        st.session_state["ui_defaults_version"] = UI_DEFAULTS_VERSION

    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_built_at", "")
    st.session_state.setdefault("last_runtime_s", None)
    st.session_state.setdefault("last_input_signature", None)
    st.session_state.setdefault("solver_profile_rows", None)
    st.session_state.setdefault("solver_profile_at", "")


def _clear_result() -> None:
    st.session_state["last_result"] = None
    st.session_state["last_error"] = ""
    st.session_state["last_built_at"] = ""
    st.session_state["last_runtime_s"] = None
    st.session_state["last_input_signature"] = None


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
        "turn_solver_qmc_samples",
        "turn_solver_local_starts",
    )
    signature = [float(st.session_state[key]) for key in keys]
    signature.append(str(st.session_state["objective_mode"]))
    signature.append(str(st.session_state["turn_solver_mode"]))
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
        turn_solver_mode=str(st.session_state["turn_solver_mode"]),
        turn_solver_qmc_samples=int(st.session_state["turn_solver_qmc_samples"]),
        turn_solver_local_starts=int(st.session_state["turn_solver_local_starts"]),
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


def _validate_input(
    surface: Point3D, t1: Point3D, t3: Point3D, config: TrajectoryConfig
) -> list[str]:
    errors: list[str] = []
    if t1.z <= surface.z:
        errors.append("t1 должен быть ниже устья S по TVD.")
    if t3.z <= surface.z:
        errors.append("t3 должен быть ниже устья S по TVD.")
    if config.kop_min_vertical_m >= max(0.0, t1.z - surface.z):
        errors.append("Мин VERTICAL до KOP должен быть меньше TVD до t1.")
    if config.md_step_m < config.md_step_control_m:
        errors.append("Шаг MD должен быть больше либо равен контрольному шагу MD.")
    return errors


def _run_planner(
    surface: Point3D, t1: Point3D, t3: Point3D, config: TrajectoryConfig
) -> None:
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


def _run_solver_profiling() -> None:
    planner = TrajectoryPlanner()
    base_config = _build_config_from_state()

    started_total = perf_counter()
    rows: list[dict[str, str]] = []
    run_items = [
        (scenario_name, scenario_points, solver_mode)
        for scenario_name, scenario_points in SCENARIOS.items()
        for solver_mode in (TURN_SOLVER_LEAST_SQUARES, TURN_SOLVER_DE_HYBRID)
    ]
    total_items = len(run_items)
    progress = st.progress(0, text="Профилирование TURN-методов...")
    with st.status("Выполняется profiling TURN-решателей...", expanded=True) as status:
        for index, (scenario_name, scenario_points, solver_mode) in enumerate(
            run_items, start=1
        ):
            progress.progress(
                int(((index - 1) / max(total_items, 1)) * 100),
                text=f"{index}/{total_items}: {scenario_name}",
            )
            status.write(
                f"[{index}/{total_items}] {scenario_name} · {TURN_SOLVER_OPTIONS[solver_mode]}"
            )
            surface = scenario_points["surface"]
            t1 = scenario_points["t1"]
            t3 = scenario_points["t3"]
            config = replace(base_config, turn_solver_mode=solver_mode)
            started = perf_counter()
            try:
                result = planner.plan(surface=surface, t1=t1, t3=t3, config=config)
                elapsed_s = perf_counter() - started
                rows.append(
                    {
                        "Шаблон": scenario_name,
                        "TURN solver": TURN_SOLVER_OPTIONS[solver_mode],
                        "Статус": "OK",
                        "Время, с": f"{elapsed_s:.3f}",
                        "Промах t1, м": f"{float(result.summary['distance_t1_m']):.4f}",
                        "Промах t3, м": f"{float(result.summary['distance_t3_m']):.4f}",
                        "TURN, deg": f"{float(result.summary.get('azimuth_turn_deg', 0.0)):.2f}",
                        "Тип траектории": str(
                            result.summary.get("trajectory_type", "—")
                        ),
                    }
                )
            except PlanningError as exc:
                elapsed_s = perf_counter() - started
                rows.append(
                    {
                        "Шаблон": scenario_name,
                        "TURN solver": TURN_SOLVER_OPTIONS[solver_mode],
                        "Статус": "Ошибка",
                        "Время, с": f"{elapsed_s:.3f}",
                        "Промах t1, м": "—",
                        "Промах t3, м": "—",
                        "TURN, deg": "—",
                        "Тип траектории": "—",
                        "Причина": str(exc),
                    }
                )

        elapsed_total_s = perf_counter() - started_total
        progress.progress(100, text="Профилирование завершено.")
        status.update(
            label=f"Профилирование завершено за {elapsed_total_s:.2f} с",
            state="complete",
            expanded=False,
        )
    progress.empty()
    st.session_state["solver_profile_rows"] = rows
    st.session_state["solver_profile_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_app() -> None:
    st.set_page_config(page_title="Планировщик траектории скважины", layout="wide")

    _init_state()
    apply_page_style(max_width_px=1680)
    render_hero(title="Планировщик траектории скважины")

    top_left, top_mid, top_right = st.columns([1.5, 1.1, 3.0], gap="small")
    with top_left:
        st.selectbox("Шаблон", options=list(SCENARIOS.keys()), key="scenario_name")
    with top_mid:
        render_small_note("Действия шаблона")
        if st.button("Применить шаблон", icon=":material/sync:", width="stretch"):
            _apply_scenario(st.session_state["scenario_name"])
            st.rerun()
        if st.button("Очистить результат", icon=":material/delete:", width="stretch"):
            _clear_result()
            st.rerun()
    with top_right:
        render_small_note(
            "Подсказка: измените параметры в форме и нажмите «Построить траекторию». "
            "Последний успешный расчет сохраняется до следующего запуска."
        )

    with st.form(
        "planner_form", clear_on_submit=False, enter_to_submit=False, border=False
    ):
        with st.container(border=True):
            st.markdown("### Параметры расчёта")
            c1, c2, c3, c4 = st.columns(
                [1.25, 1.25, 1.25, 2.35],
                gap="small",
                border=True,
                vertical_alignment="top",
            )

            st.markdown("**Расчет траектории**")
            build_clicked = st.form_submit_button(
                "Построить траекторию",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )
            if st.session_state["last_built_at"]:
                st.caption(f"Последний расчет: {st.session_state['last_built_at']}")
            if st.session_state.get("last_runtime_s") is not None:
                st.caption(
                    f"Время расчета: {float(st.session_state['last_runtime_s']):.2f} с"
                )

            with c1:
                st.markdown("**Устье S**")
                st.number_input("S X (E), m", key="surface_x", step=50.0)
                st.number_input("S Y (N), m", key="surface_y", step=50.0)
                st.number_input("S Z (TVD), m", key="surface_z", step=50.0)

            with c2:
                st.markdown("**Точка входа t1**")
                st.number_input("t1 X (E), m", key="t1_x", step=50.0)
                st.number_input("t1 Y (N), m", key="t1_y", step=50.0)
                st.number_input("t1 Z (TVD), m", key="t1_z", step=50.0)

            with c3:
                st.markdown("**Концевая точка t3**")
                st.number_input("t3 X (E), m", key="t3_x", step=50.0)
                st.number_input("t3 Y (N), m", key="t3_y", step=50.0)
                st.number_input("t3 Z (TVD), m", key="t3_z", step=50.0)

            with c4:
                st.markdown("**Ограничения профиля**")
                r1, r2, r3 = st.columns(3, gap="small")
                r1.number_input("Шаг MD", key="md_step", min_value=1.0, step=1.0)
                r2.number_input(
                    "Контрольный шаг MD", key="md_control", min_value=0.5, step=0.5
                )
                r3.number_input(
                    "Допуск по позиции", key="pos_tol", min_value=0.1, step=0.1
                )

                rr1, rr2 = st.columns(2, gap="small")
                rr1.number_input(
                    "Целевой INC входа",
                    key="entry_inc_target",
                    min_value=70.0,
                    max_value=89.0,
                    step=0.5,
                )
                rr2.number_input(
                    "Допуск INC входа",
                    key="entry_inc_tol",
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                )

                rd1, rd2 = st.columns(2, gap="small")
                rd1.number_input(
                    "Мин DLS BUILD",
                    key="dls_build_min",
                    min_value=0.1,
                    value=0.5,
                    step=0.1,
                    help="deg/30m",
                )
                rd2.number_input(
                    "Макс DLS BUILD",
                    key="dls_build_max",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="для BUILD1 и BUILD2",
                )
                st.number_input(
                    "Мин VERTICAL до KOP, м",
                    key="kop_min_vertical",
                    min_value=0.0,
                    value=300.0,
                    step=10.0,
                    help="Минимальный VERTICAL участок от S до начала BUILD1.",
                )
                with st.expander("Параметры солвера", expanded=False):
                    st.selectbox(
                        "Целевая функция",
                        options=list(OBJECTIVE_OPTIONS.keys()),
                        key="objective_mode",
                        format_func=lambda value: OBJECTIVE_OPTIONS[str(value)],
                        help=(
                            "Определяет, что оптимизирует решатель среди допустимых решений. "
                            "Рекомендуется оставить «Максимизировать длину HOLD»: обычно это дает "
                            "более стабильную конструкцию и меньше ручной подстройки."
                        ),
                    )
                    st.selectbox(
                        "Метод TURN-решателя",
                        options=list(TURN_SOLVER_OPTIONS.keys()),
                        key="turn_solver_mode",
                        format_func=lambda value: TURN_SOLVER_OPTIONS[str(value)],
                        help=(
                            "Least Squares (TRF) — быстрый и достаточно устойчивый метод для большинства кейсов "
                            "(рекомендуемый дефолт). DE Hybrid — более медленный, но может помочь в особенно "
                            "сложной геометрии."
                        ),
                    )
                    rs1, rs2 = st.columns(2, gap="small")
                    rs1.number_input(
                        "TURN QMC samples",
                        key="turn_solver_qmc_samples",
                        min_value=0,
                        value=24,
                        step=4,
                        help=(
                            "Количество дополнительных стартовых точек (Latin Hypercube). "
                            "Больше значение — выше шанс найти решение в сложных случаях, но дольше расчет. "
                            "Дефолт 24 — рабочий компромисс."
                        ),
                    )
                    rs2.number_input(
                        "TURN local starts",
                        key="turn_solver_local_starts",
                        min_value=1,
                        value=12,
                        step=1,
                        help=(
                            "Сколько лучших стартовых точек запускать локальным решателем. "
                            "Больше — стабильнее подбор, но медленнее. Дефолт 12 — обычно оптимальный."
                        ),
                    )

    surface_input, t1_input, t3_input = _build_points_from_state()
    config_input = _build_config_from_state()
    preflight_errors = _validate_input(
        surface=surface_input, t1=t1_input, t3=t3_input, config=config_input
    )
    if preflight_errors:
        st.warning(
            "Предварительная проверка параметров:\n- " + "\n- ".join(preflight_errors)
        )

    with st.expander("Профилирование TURN-методов", expanded=False):
        st.caption("Сравнение методов на типовых шаблонах текущего проекта.")
        if st.button("Запустить profiling", icon=":material/speed:", width="content"):
            _run_solver_profiling()
        profile_rows = st.session_state.get("solver_profile_rows")
        if st.session_state.get("solver_profile_at"):
            st.caption(f"Последний profiling: {st.session_state['solver_profile_at']}")
        if profile_rows:
            profile_df = pd.DataFrame(profile_rows)
            st.dataframe(
                arrow_safe_text_dataframe(profile_df), width="stretch", hide_index=True
            )

    if build_clicked:
        progress = st.progress(0, text="Подготовка расчета...")
        with st.status("Расчет траектории...", expanded=True) as status:
            try:
                status.write("Проверка входных данных.")
                progress.progress(20, text="Проверка входных данных...")
                if preflight_errors:
                    raise PlanningError("; ".join(preflight_errors))
                status.write("Запуск планировщика и решателя.")
                progress.progress(55, text="Выполняется расчет траектории...")
                started = perf_counter()
                _run_planner(
                    surface=surface_input, t1=t1_input, t3=t3_input, config=config_input
                )
                elapsed_s = perf_counter() - started
                st.session_state["last_runtime_s"] = float(elapsed_s)
                progress.progress(100, text="Расчет завершен.")
                status.update(
                    label=f"Расчет завершен за {elapsed_s:.2f} с",
                    state="complete",
                    expanded=False,
                )
            except PlanningError as exc:
                st.session_state["last_error"] = str(exc)
                st.session_state["last_runtime_s"] = None
                status.write(str(exc))
                status.update(
                    label="Расчет завершился ошибкой", state="error", expanded=True
                )
            finally:
                progress.empty()

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])

    last_result = st.session_state.get("last_result")
    if last_result is None:
        st.info("Задайте параметры и нажмите «Построить траекторию».")
        return

    is_stale = (
        st.session_state.get("last_input_signature") != _current_input_signature()
    )
    if is_stale:
        st.warning(
            "Параметры изменились после последнего расчета. Показан предыдущий результат. Нажмите «Построить траекторию» для обновления."
        )

    summary = last_result["summary"]
    stations = last_result["stations"]
    surface = last_result["surface"]
    t1 = last_result["t1"]
    t3 = last_result["t3"]
    config = last_result["config"]
    azimuth_deg = float(last_result["azimuth_deg"])
    md_t1_m = float(last_result["md_t1_m"])
    t1_horizontal_offset_m = _horizontal_offset_m(point=t1, reference=surface)

    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    m1.metric("Горизонтальный отход t1", format_distance(t1_horizontal_offset_m))
    m2.metric("Угол входа в пласт", f"{summary['entry_inc_deg']:.2f} deg")
    m3.metric("ЗУ секции HOLD", f"{float(summary['hold_inc_deg']):.2f} deg")
    m4.metric("Сложность", str(summary["well_complexity"]))
    runtime_s = st.session_state.get("last_runtime_s")
    m5.metric(
        "Время расчета", "—" if runtime_s is None else f"{float(runtime_s):.2f} с"
    )

    n1, n2, n3 = st.columns(3, gap="small")
    n1.metric("Тип траектории", str(summary["trajectory_type"]))
    n2.metric("KOP MD", format_distance(float(summary["kop_md_m"])))
    n3.metric("Макс DLS", f"{summary['max_dls_total_deg_per_30m']:.2f} deg/30m")

    with st.container(border=True):
        st.markdown("### 3D траектория и DLS")
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row1_col1.plotly_chart(
            trajectory_3d_figure(
                stations, surface=surface, t1=t1, t3=t3, md_t1_m=md_t1_m
            ),
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
            section_view_figure(
                stations, surface=surface, azimuth_deg=azimuth_deg, t1=t1, t3=t3
            ),
            width="stretch",
        )

    tab_summary, tab_survey = st.tabs(["Сводка", "Инклинометрия"])
    with tab_summary:
        hidden_metrics = {"distance_t1_m", "distance_t3_m"}
        summary_visible = {
            key: value for key, value in summary.items() if key not in hidden_metrics
        }
        summary_visible["t1_horizontal_offset_m"] = t1_horizontal_offset_m
        summary_df = pd.DataFrame(
            {
                "metric": list(summary_visible.keys()),
                "value": list(summary_visible.values()),
            }
        )
        st.dataframe(
            arrow_safe_text_dataframe(summary_df), width="stretch", hide_index=True
        )

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


if __name__ == "__main__":
    if not st.runtime.exists():
        raise SystemExit("Запустите приложение командой `streamlit run app.py`.")
    run_app()
