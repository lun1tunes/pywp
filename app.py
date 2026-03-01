from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from time import perf_counter
from typing import Callable

import pandas as pd
import streamlit as st

from pywp import Point3D, TrajectoryConfig, TrajectoryPlanner
from pywp.classification import (
    COMPLEXITY_COMPLEX,
    COMPLEXITY_ORDINARY,
    COMPLEXITY_VERY_COMPLEX,
    TRAJECTORY_REVERSE_DIRECTION,
    TRAJECTORY_SAME_DIRECTION,
    complexity_label,
    interpolate_limits,
    reference_table_rows,
    trajectory_type_label,
)
from pywp.planner import PlanningError
from pywp.planner_config import (
    CFG_DEFAULTS,
    OBJECTIVE_OPTIONS,
    TURN_SOLVER_OPTIONS,
    build_trajectory_config,
)
from pywp.solver_diagnostics_ui import render_solver_diagnostics
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_utils import (
    arrow_safe_text_dataframe,
    format_distance,
    format_run_log_line,
)
from pywp.ui_well_panels import (
    render_plan_section_panel,
    render_run_log_panel,
    render_survey_table_with_download,
    render_trajectory_dls_panel,
)


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    gv_m: float
    trajectory_type: str
    complexity: str
    surface: Point3D
    t1: Point3D
    t3: Point3D
    description: str = ""


SCENARIO_PRESETS: tuple[ScenarioPreset, ...] = (
    ScenarioPreset(
        name="Базовый пример: ГВ 2400, прямое направление",
        gv_m=2400.0,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        complexity=COMPLEXITY_ORDINARY,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(600.0, 800.0, 2400.0),
        t3=Point3D(1500.0, 2000.0, 2500.0),
        description="Универсальный стартовый шаблон для быстрых проверок.",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 3000, обратное направление, обычная",
        gv_m=3000.0,
        trajectory_type=TRAJECTORY_REVERSE_DIRECTION,
        complexity=COMPLEXITY_ORDINARY,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(300.0, 400.0, 3000.0),
        t3=Point3D(1020.0, 1360.0, 3083.9122),
        description="Обратное направление в диапазоне offset из референсной таблицы.",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 3000, прямое направление, обычная",
        gv_m=3000.0,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        complexity=COMPLEXITY_ORDINARY,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(960.0, 1280.0, 3000.0),
        t3=Point3D(1860.0, 2480.0, 3083.9122),
        description="Базовый same-direction кейс на 3000 м (класс обычная).",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 3000, прямое направление, сложная",
        gv_m=3000.0,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        complexity=COMPLEXITY_COMPLEX,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1440.0, 1920.0, 3000.0),
        t3=Point3D(2460.0, 3280.0, 3083.9122),
        description="Same-direction шаблон для сложного класса на 3000 м.",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 3000, прямое направление, очень сложная",
        gv_m=3000.0,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        complexity=COMPLEXITY_VERY_COMPLEX,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(2100.0, 2800.0, 3000.0),
        t3=Point3D(3300.0, 4400.0, 3083.9122),
        description="Same-direction шаблон для очень сложного класса на 3000 м.",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 2000, обратное направление, очень сложная",
        gv_m=2000.0,
        trajectory_type=TRAJECTORY_REVERSE_DIRECTION,
        complexity=COMPLEXITY_VERY_COMPLEX,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(210.0, 280.0, 2000.0),
        t3=Point3D(810.0, 1080.0, 2069.9268),
        description="Сценарий с повышенной геометрической сложностью для reverse-кейса.",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 2000, прямое направление, обычная",
        gv_m=2000.0,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        complexity=COMPLEXITY_ORDINARY,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(540.0, 720.0, 2000.0),
        t3=Point3D(1140.0, 1520.0, 2069.9268),
        description="Базовый same-direction кейс на 2000 м.",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 3600, прямое направление, сложная",
        gv_m=3600.0,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        complexity=COMPLEXITY_COMPLEX,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(1500.0, 2000.0, 3600.0),
        t3=Point3D(2520.0, 3360.0, 3718.8756),
        description="Сложный same-direction кейс для типовой глубины 3600 м.",
    ),
    ScenarioPreset(
        name="Типовой: ГВ 3600, прямое направление, очень сложная",
        gv_m=3600.0,
        trajectory_type=TRAJECTORY_SAME_DIRECTION,
        complexity=COMPLEXITY_VERY_COMPLEX,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(2160.0, 2880.0, 3600.0),
        t3=Point3D(3360.0, 4480.0, 3739.8536),
        description="Очень сложный same-direction кейс с большим отходом t1.",
    ),
)
SCENARIO_BY_NAME = {preset.name: preset for preset in SCENARIO_PRESETS}
SCENARIOS = {
    preset.name: {
        "surface": preset.surface,
        "t1": preset.t1,
        "t3": preset.t3,
    }
    for preset in SCENARIO_PRESETS
}
DEPTH_FILTER_ALL = "all"
DEPTH_FILTER_OPTIONS: tuple[str | float, ...] = (
    DEPTH_FILTER_ALL,
    *tuple(sorted({float(preset.gv_m) for preset in SCENARIO_PRESETS})),
)
COMPLEXITY_OPTIONS: tuple[str, ...] = (
    COMPLEXITY_ORDINARY,
    COMPLEXITY_COMPLEX,
    COMPLEXITY_VERY_COMPLEX,
)
TRAJECTORY_OPTIONS: tuple[str, ...] = (
    TRAJECTORY_SAME_DIRECTION,
    TRAJECTORY_REVERSE_DIRECTION,
)
TRAJECTORY_SELECTOR_LABELS = {
    TRAJECTORY_SAME_DIRECTION: "Прямое",
    TRAJECTORY_REVERSE_DIRECTION: "Обратное",
}
COMPLEXITY_SELECTOR_LABELS = {
    COMPLEXITY_ORDINARY: "Обычная",
    COMPLEXITY_COMPLEX: "Сложная",
    COMPLEXITY_VERY_COMPLEX: "Очень сложная",
}
SUMMARY_MAIN_METRICS: tuple[tuple[str, str], ...] = (
    ("trajectory_type", "Тип траектории"),
    ("well_complexity", "Класс сложности"),
    ("entry_inc_deg", "Угол входа в пласт, deg"),
    ("hold_inc_deg", "ЗУ секции HOLD, deg"),
    ("hold_length_m", "Длина HOLD, м"),
    ("horizontal_length_m", "Длина горизонтального ствола, м"),
    ("kop_md_m", "KOP MD, м"),
    ("max_dls_total_deg_per_30m", "Макс DLS по стволу, deg/30m"),
    ("max_inc_actual_deg", "Макс INC фактический, deg"),
    ("max_inc_deg", "Макс INC лимит, deg"),
    ("md_total_m", "Итоговая MD, м"),
)
UI_DEFAULTS_VERSION = 7


def _horizontal_offset_m(point: Point3D, reference: Point3D) -> float:
    dx = float(point.x - reference.x)
    dy = float(point.y - reference.y)
    return float((dx * dx + dy * dy) ** 0.5)


def _depth_filter_label(value: str | float) -> str:
    if value == DEPTH_FILTER_ALL:
        return "Любая типовая глубина"
    return f"{int(float(value))} м"


def _preset_reverse_window_label(gv_m: float) -> str:
    limits = interpolate_limits(gv_m=float(gv_m))
    if not limits.reverse_allowed:
        return "Не допускается"
    return f"{float(limits.reverse_min_m):.0f}-{float(limits.reverse_max_m):.0f} м"


def _apply_template_filters_for_scenario(scenario_name: str) -> None:
    preset = SCENARIO_BY_NAME.get(scenario_name)
    if preset is None:
        return
    st.session_state["template_depth_filter"] = float(preset.gv_m)
    st.session_state["template_trajectory_type"] = str(preset.trajectory_type)
    st.session_state["template_complexity"] = str(preset.complexity)


def _available_trajectory_options(depth_filter: str | float) -> list[str]:
    depth_value = None if depth_filter == DEPTH_FILTER_ALL else float(depth_filter)
    available_set = {
        str(preset.trajectory_type)
        for preset in SCENARIO_PRESETS
        if depth_value is None or abs(float(preset.gv_m) - depth_value) < 1e-6
    }
    options = [code for code in TRAJECTORY_OPTIONS if code in available_set]
    return options or list(TRAJECTORY_OPTIONS)


def _available_complexity_options(
    depth_filter: str | float, trajectory_type: str
) -> list[str]:
    depth_value = None if depth_filter == DEPTH_FILTER_ALL else float(depth_filter)
    options = sorted(
        {
            str(preset.complexity)
            for preset in SCENARIO_PRESETS
            if str(preset.trajectory_type) == str(trajectory_type)
            and (depth_value is None or abs(float(preset.gv_m) - depth_value) < 1e-6)
        },
        key=lambda code: (
            list(COMPLEXITY_OPTIONS).index(code) if code in COMPLEXITY_OPTIONS else 99
        ),
    )
    return options


def _filtered_scenario_names(
    depth_filter: str | float,
    trajectory_type: str,
    complexity: str,
) -> list[str]:
    depth_value = None if depth_filter == DEPTH_FILTER_ALL else float(depth_filter)
    exact = [
        preset.name
        for preset in SCENARIO_PRESETS
        if str(preset.trajectory_type) == str(trajectory_type)
        and str(preset.complexity) == str(complexity)
        and (depth_value is None or abs(float(preset.gv_m) - depth_value) < 1e-6)
    ]
    if exact:
        return exact
    return [preset.name for preset in SCENARIO_PRESETS]


def _template_coverage_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    depth_values = sorted({float(preset.gv_m) for preset in SCENARIO_PRESETS})
    for gv in depth_values:
        for trajectory in TRAJECTORY_OPTIONS:
            options = _available_complexity_options(
                depth_filter=gv,
                trajectory_type=trajectory,
            )
            presets_count = sum(
                1
                for preset in SCENARIO_PRESETS
                if abs(float(preset.gv_m) - float(gv)) < 1e-6
                and str(preset.trajectory_type) == str(trajectory)
            )
            rows.append(
                {
                    "ГВ t1, м": f"{int(gv)}",
                    "Тип конструкции": trajectory_type_label(trajectory),
                    "Классы с примерами": (
                        ", ".join(
                            COMPLEXITY_SELECTOR_LABELS.get(code, code)
                            for code in options
                        )
                        if options
                        else "Нет"
                    ),
                    "Шаблонов в каталоге": str(presets_count),
                }
            )
    return arrow_safe_text_dataframe(pd.DataFrame(rows))


def _apply_scenario(name: str) -> None:
    values = SCENARIOS[name]
    for prefix, point_key in (("surface", "surface"), ("t1", "t1"), ("t3", "t3")):
        point = values[point_key]
        st.session_state[f"{prefix}_x"] = float(point.x)
        st.session_state[f"{prefix}_y"] = float(point.y)
        st.session_state[f"{prefix}_z"] = float(point.z)
    st.session_state["scenario_name"] = str(name)
    _apply_template_filters_for_scenario(str(name))


def _init_state() -> None:
    default_scenario = next(iter(SCENARIOS.keys()))
    st.session_state.setdefault("scenario_name", default_scenario)
    if str(st.session_state["scenario_name"]) not in SCENARIOS:
        st.session_state["scenario_name"] = default_scenario
    current_preset = SCENARIO_BY_NAME.get(str(st.session_state["scenario_name"]))
    default_template_depth: str | float = DEPTH_FILTER_ALL
    default_template_type = str(TRAJECTORY_SAME_DIRECTION)
    default_template_complexity = str(COMPLEXITY_ORDINARY)
    if current_preset is not None:
        default_template_depth = float(current_preset.gv_m)
        default_template_type = str(current_preset.trajectory_type)
        default_template_complexity = str(current_preset.complexity)
    st.session_state.setdefault("template_depth_filter", default_template_depth)
    st.session_state.setdefault(
        "template_trajectory_type",
        default_template_type,
    )
    st.session_state.setdefault("template_complexity", default_template_complexity)

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
    st.session_state.setdefault("max_inc", float(CFG_DEFAULTS.max_inc_deg))
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
    st.session_state.setdefault(
        "adaptive_grid_enabled", bool(CFG_DEFAULTS.adaptive_grid_enabled)
    )
    st.session_state.setdefault(
        "adaptive_grid_initial_size", int(CFG_DEFAULTS.adaptive_grid_initial_size)
    )
    st.session_state.setdefault(
        "adaptive_grid_refine_levels", int(CFG_DEFAULTS.adaptive_grid_refine_levels)
    )
    st.session_state.setdefault(
        "adaptive_grid_top_k", int(CFG_DEFAULTS.adaptive_grid_top_k)
    )
    st.session_state.setdefault("parallel_jobs", int(CFG_DEFAULTS.parallel_jobs))
    st.session_state.setdefault(
        "profile_cache_enabled", bool(CFG_DEFAULTS.profile_cache_enabled)
    )
    st.session_state.setdefault("ui_defaults_version", 0)

    if int(st.session_state.get("ui_defaults_version", 0)) < UI_DEFAULTS_VERSION:
        st.session_state["md_step"] = float(CFG_DEFAULTS.md_step_m)
        st.session_state["md_control"] = float(CFG_DEFAULTS.md_step_control_m)
        st.session_state["pos_tol"] = float(CFG_DEFAULTS.pos_tolerance_m)
        st.session_state["entry_inc_target"] = float(CFG_DEFAULTS.entry_inc_target_deg)
        st.session_state["entry_inc_tol"] = float(CFG_DEFAULTS.entry_inc_tolerance_deg)
        st.session_state["max_inc"] = float(CFG_DEFAULTS.max_inc_deg)
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
        st.session_state["adaptive_grid_enabled"] = bool(
            CFG_DEFAULTS.adaptive_grid_enabled
        )
        st.session_state["adaptive_grid_initial_size"] = int(
            CFG_DEFAULTS.adaptive_grid_initial_size
        )
        st.session_state["adaptive_grid_refine_levels"] = int(
            CFG_DEFAULTS.adaptive_grid_refine_levels
        )
        st.session_state["adaptive_grid_top_k"] = int(CFG_DEFAULTS.adaptive_grid_top_k)
        st.session_state["parallel_jobs"] = int(CFG_DEFAULTS.parallel_jobs)
        st.session_state["profile_cache_enabled"] = bool(
            CFG_DEFAULTS.profile_cache_enabled
        )
        _apply_template_filters_for_scenario(str(st.session_state["scenario_name"]))
        st.session_state["ui_defaults_version"] = UI_DEFAULTS_VERSION

    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_built_at", "")
    st.session_state.setdefault("last_runtime_s", None)
    st.session_state.setdefault("last_input_signature", None)
    st.session_state.setdefault("solver_profile_rows", None)
    st.session_state.setdefault("solver_profile_at", "")
    st.session_state.setdefault("last_run_log_lines", [])


def _clear_result() -> None:
    st.session_state["last_result"] = None
    st.session_state["last_error"] = ""
    st.session_state["last_built_at"] = ""
    st.session_state["last_runtime_s"] = None
    st.session_state["last_input_signature"] = None
    st.session_state["last_run_log_lines"] = []


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
        "max_inc",
        "dls_build_min",
        "dls_build_max",
        "kop_min_vertical",
        "turn_solver_qmc_samples",
        "turn_solver_local_starts",
        "adaptive_grid_initial_size",
        "adaptive_grid_refine_levels",
        "adaptive_grid_top_k",
        "parallel_jobs",
    )
    signature = [float(st.session_state[key]) for key in keys]
    signature.append(str(st.session_state["objective_mode"]))
    signature.append(str(st.session_state["turn_solver_mode"]))
    signature.append(str(bool(st.session_state["adaptive_grid_enabled"])))
    signature.append(str(bool(st.session_state["profile_cache_enabled"])))
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
    return build_trajectory_config(
        md_step_m=float(st.session_state["md_step"]),
        md_step_control_m=float(st.session_state["md_control"]),
        pos_tolerance_m=float(st.session_state["pos_tol"]),
        entry_inc_target_deg=float(st.session_state["entry_inc_target"]),
        entry_inc_tolerance_deg=float(st.session_state["entry_inc_tol"]),
        max_inc_deg=float(st.session_state["max_inc"]),
        dls_build_min_deg_per_30m=float(st.session_state["dls_build_min"]),
        dls_build_max_deg_per_30m=float(st.session_state["dls_build_max"]),
        kop_min_vertical_m=float(st.session_state["kop_min_vertical"]),
        objective_mode=str(st.session_state["objective_mode"]),
        turn_solver_mode=str(st.session_state["turn_solver_mode"]),
        turn_solver_qmc_samples=int(st.session_state["turn_solver_qmc_samples"]),
        turn_solver_local_starts=int(st.session_state["turn_solver_local_starts"]),
        adaptive_grid_enabled=bool(st.session_state["adaptive_grid_enabled"]),
        adaptive_grid_initial_size=int(st.session_state["adaptive_grid_initial_size"]),
        adaptive_grid_refine_levels=int(
            st.session_state["adaptive_grid_refine_levels"]
        ),
        adaptive_grid_top_k=int(st.session_state["adaptive_grid_top_k"]),
        parallel_jobs=int(st.session_state["parallel_jobs"]),
        profile_cache_enabled=bool(st.session_state["profile_cache_enabled"]),
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
    if config.entry_inc_target_deg > config.max_inc_deg:
        errors.append("Целевой INC входа не должен превышать Макс INC по стволу.")
    return errors


def _run_planner(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    config: TrajectoryConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> None:
    planner = TrajectoryPlanner()
    result = planner.plan(
        surface=surface,
        t1=t1,
        t3=t3,
        config=config,
        progress_callback=progress_callback,
    )

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
        for solver_mode in tuple(TURN_SOLVER_OPTIONS.keys())
    ]
    total_items = len(run_items)
    progress = st.progress(0, text="Профилирование TURN-методов...")
    with st.status(
        "Выполняется профилирование TURN-решателей...", expanded=True
    ) as status:
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


def _render_template_controls() -> None:
    with st.container(border=True):
        st.markdown("### Каталог шаблонов конструкции")
        render_small_note(
            "Выберите тип конструкции, класс сложности и типовую глубину. "
            "Шаблоны соответствуют вашей таблице классификации и ускоряют старт настройки."
        )
        h1, h2, h3 = st.columns([3.8, 1.2, 1.2], gap="small")
        with h1:
            with st.popover("Как выбирать шаблон", icon=":material/help:"):
                st.markdown(
                    "- `Прямое/Обратное` определяет базовый тип конструкции.\n"
                    "- `Класс сложности` ориентируется на референсные границы по отходу и HOLD.\n"
                    "- `Типовая ГВ t1` ограничивает каталог по глубине.\n"
                    "- Если точного совпадения нет, будет предложен ближайший вариант."
                )
        with h2:
            if st.button(
                "Применить шаблон",
                icon=":material/check_circle:",
                type="primary",
                width="stretch",
            ):
                _apply_scenario(st.session_state["scenario_name"])
                st.rerun()
        with h3:
            if st.button(
                "Очистить результат", icon=":material/delete:", width="stretch"
            ):
                _clear_result()
                st.rerun()
        f1, f2, f3, f4 = st.columns([1.2, 1.8, 1.8, 2.8], gap="small")
        with f1:
            st.selectbox(
                "Типовая ГВ t1",
                options=list(DEPTH_FILTER_OPTIONS),
                key="template_depth_filter",
                format_func=_depth_filter_label,
            )
        selected_depth_filter = st.session_state["template_depth_filter"]
        available_trajectory = _available_trajectory_options(
            depth_filter=selected_depth_filter
        )
        if (
            str(st.session_state["template_trajectory_type"])
            not in available_trajectory
        ):
            st.session_state["template_trajectory_type"] = available_trajectory[0]
        with f2:
            st.segmented_control(
                "Тип конструкции",
                options=available_trajectory,
                key="template_trajectory_type",
                format_func=lambda code: TRAJECTORY_SELECTOR_LABELS[str(code)],
                selection_mode="single",
                help="Тип строящейся конструкции относительно направления на t1.",
            )
        selected_trajectory = str(st.session_state["template_trajectory_type"])
        available_complexity = _available_complexity_options(
            depth_filter=selected_depth_filter,
            trajectory_type=selected_trajectory,
        )
        if not available_complexity:
            available_complexity = list(COMPLEXITY_OPTIONS)
        if str(st.session_state["template_complexity"]) not in available_complexity:
            st.session_state["template_complexity"] = available_complexity[0]
        with f3:
            st.segmented_control(
                "Класс сложности",
                options=available_complexity,
                key="template_complexity",
                format_func=lambda code: COMPLEXITY_SELECTOR_LABELS[str(code)],
                selection_mode="single",
                help="Класс из типовой шкалы по глубине, отходу и HOLD.",
            )

        selected_depth_filter = st.session_state["template_depth_filter"]
        selected_trajectory = str(st.session_state["template_trajectory_type"])
        selected_complexity = str(st.session_state["template_complexity"])
        filtered_names = _filtered_scenario_names(
            depth_filter=selected_depth_filter,
            trajectory_type=selected_trajectory,
            complexity=selected_complexity,
        )
        if str(st.session_state["scenario_name"]) not in filtered_names:
            st.session_state["scenario_name"] = filtered_names[0]

        with f4:
            st.selectbox(
                "Шаблон",
                options=filtered_names,
                key="scenario_name",
                help="Показаны шаблоны в выбранном диапазоне глубины и типе конструкции.",
            )

        selected_name = str(st.session_state["scenario_name"])
        selected_preset = SCENARIO_BY_NAME[selected_name]
        if selected_preset.description:
            st.caption(selected_preset.description)

        offset_t1 = _horizontal_offset_m(
            point=selected_preset.t1,
            reference=selected_preset.surface,
        )
        reverse_window = _preset_reverse_window_label(gv_m=selected_preset.gv_m)
        m1, m2, m3, m4 = st.columns(4, gap="small")
        m1.metric("ГВ t1", f"{selected_preset.gv_m:.0f} м")
        m2.metric("Отход t1", f"{offset_t1:.0f} м")
        m3.metric("Референс reverse по ГВ", reverse_window)
        m4.metric("Класс шаблона", complexity_label(selected_preset.complexity))
        with st.expander(
            "Что такое окно reverse и как оно используется?", expanded=False
        ):
            st.markdown(
                "- `Окно reverse` — это диапазон **горизонтального отхода t1** (S→t1), "
                "в котором цель относится к типу `обратное направление`.\n"
                "- Значение в карточке — **справочный референс по ГВ t1** из таблицы классификации, "
                "а не «дефолт» решателя.\n"
                "- На расчет влияет через автоматическую классификацию типа траектории "
                "(`прямое`/`обратное`) и выбор подходящего шаблона.\n"
                "- Из UI главной страницы это окно не редактируется: пороги задаются "
                "в референсной таблице на странице `Классификация скважин` "
                "(модуль `pywp/classification.py`)."
            )

        render_small_note(
            "Подсказка: после применения шаблона можно вручную скорректировать координаты S/t1/t3. "
            "Последний успешный расчет сохраняется до следующего запуска."
        )
        with st.expander("Покрытие каталога шаблонов", expanded=False):
            st.dataframe(
                _template_coverage_frame(),
                width="stretch",
                hide_index=True,
            )
        with st.expander("Референс классов по типовым глубинам", expanded=False):
            st.dataframe(
                arrow_safe_text_dataframe(pd.DataFrame(reference_table_rows())),
                width="stretch",
                hide_index=True,
            )


def _render_input_form() -> bool:
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
                st.number_input(
                    "S X (East), м",
                    key="surface_x",
                    step=50.0,
                    help="Координата X устья S в локальной системе координат.",
                )
                st.number_input(
                    "S Y (North), м",
                    key="surface_y",
                    step=50.0,
                    help="Координата Y устья S в локальной системе координат.",
                )
                st.number_input(
                    "S Z (TVD), м",
                    key="surface_z",
                    step=50.0,
                    help="Обычно 0 м для поверхности. TVD положителен вниз.",
                )

            with c2:
                st.markdown("**Точка входа t1**")
                st.number_input(
                    "t1 X (East), м",
                    key="t1_x",
                    step=50.0,
                    help="Координата X точки входа в пласт.",
                )
                st.number_input(
                    "t1 Y (North), м",
                    key="t1_y",
                    step=50.0,
                    help="Координата Y точки входа в пласт.",
                )
                st.number_input(
                    "t1 Z (TVD), м",
                    key="t1_z",
                    step=50.0,
                    help="TVD точки входа t1.",
                )

            with c3:
                st.markdown("**Концевая точка t3**")
                st.number_input(
                    "t3 X (East), м",
                    key="t3_x",
                    step=50.0,
                    help="Координата X целевой точки в пласте.",
                )
                st.number_input(
                    "t3 Y (North), м",
                    key="t3_y",
                    step=50.0,
                    help="Координата Y целевой точки в пласте.",
                )
                st.number_input(
                    "t3 Z (TVD), м",
                    key="t3_z",
                    step=50.0,
                    help="TVD конечной точки t3.",
                )

            with c4:
                st.markdown("**Ограничения профиля**")
                r1, r2, r3 = st.columns(3, gap="small")
                r1.number_input(
                    "Шаг MD, м",
                    key="md_step",
                    min_value=1.0,
                    step=1.0,
                    help="Шаг выходной инклинометрии. Меньше шаг = подробнее профиль.",
                )
                r2.number_input(
                    "Контрольный шаг MD, м",
                    key="md_control",
                    min_value=0.5,
                    step=0.5,
                    help="Внутренний расчетный шаг для проверки ограничений и качества решения.",
                )
                r3.number_input(
                    "Допуск по позиции, м",
                    key="pos_tol",
                    min_value=0.1,
                    step=0.1,
                    help="Максимально допустимый промах по t1 и t3.",
                )

                rr1, rr2, rr3 = st.columns(3, gap="small")
                rr1.number_input(
                    "Целевой INC на t1, deg",
                    key="entry_inc_target",
                    min_value=70.0,
                    max_value=89.0,
                    step=0.5,
                    help="Плановый угол входа в пласт в точке t1.",
                )
                rr2.number_input(
                    "Допуск INC на t1, deg",
                    key="entry_inc_tol",
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                    help="Отклонение от целевого INC в точке t1.",
                )
                rr3.number_input(
                    "Макс INC по стволу",
                    key="max_inc",
                    min_value=80.0,
                    max_value=120.0,
                    step=0.5,
                    help="Глобальное ограничение по зенитному углу. При недостатке этого лимита расчет потребует overbend.",
                )

                rd1, rd2 = st.columns(2, gap="small")
                rd1.number_input(
                    "Мин DLS BUILD",
                    key="dls_build_min",
                    min_value=0.1,
                    value=0.5,
                    step=0.1,
                    help="Нижняя граница поиска DLS на BUILD-сегментах (deg/30m).",
                )
                rd2.number_input(
                    "Макс DLS BUILD",
                    key="dls_build_max",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="Верхняя граница поиска DLS на BUILD1/BUILD2 (deg/30m).",
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
                    with st.popover(
                        "Что означают параметры солвера", icon=":material/tune:"
                    ):
                        st.markdown(
                            "- `Целевая функция` задает критерий выбора из допустимых профилей.\n"
                            "- `TURN` параметры влияют только на некомпланарные случаи.\n"
                            "- `Сетка поиска` — это набор пробных значений KOP и reverse INC.\n"
                            "- `Adaptive` сначала считает по редкой сетке, затем уплотняет ее около лучших решений.\n"
                            "- `Parallel jobs` использует процессы в coplanar-поиске.\n"
                            "- `Кэш профилей` сокращает повторные вычисления внутри оптимизации."
                        )
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
                    st.markdown("**Производительность и поиск**")
                    st.caption(
                        "Сетка простыми словами: это «точки, где солвер пробует решение». "
                        "Плотнее сетка = точнее поиск, но дольше расчет. "
                        "Adaptive делает это умнее: грубо в начале, подробно рядом с лучшими вариантами."
                    )
                    p1, p2 = st.columns(2, gap="small")
                    p1.toggle(
                        "Умная сетка (coarse → fine)",
                        key="adaptive_grid_enabled",
                        help=(
                            "Сначала быстрый грубый просмотр, потом уточнение около лучших решений. "
                            "Обычно быстрее, чем сразу считать очень плотную сетку."
                        ),
                    )
                    p2.toggle(
                        "Кэш профилей",
                        key="profile_cache_enabled",
                        help=(
                            "Кэширует промежуточные расчеты профиля в рамках оптимизации. "
                            "Полезно для ускорения при повторных оценках."
                        ),
                    )
                    p3, p4, p5 = st.columns(3, gap="small")
                    p3.number_input(
                        "Стартовая сетка (точек)",
                        key="adaptive_grid_initial_size",
                        min_value=2,
                        step=1,
                        help=(
                            "Сколько точек брать в первом грубом проходе. "
                            "Больше точек = плотнее стартовая сетка = медленнее, но точнее на старте."
                        ),
                    )
                    p4.number_input(
                        "Шагов уточнения",
                        key="adaptive_grid_refine_levels",
                        min_value=0,
                        step=1,
                        help=(
                            "Сколько раз дополнительно уплотнять сетку вокруг лучших кандидатов. "
                            "0 = только стартовый проход."
                        ),
                    )
                    p5.number_input(
                        "Лучших зон для уточнения (top-k)",
                        key="adaptive_grid_top_k",
                        min_value=1,
                        step=1,
                        help=(
                            "Сколько лучших кандидатов брать как «центры внимания» на следующем шаге. "
                            "Больше значение повышает надежность поиска, но увеличивает время."
                        ),
                    )
                    st.number_input(
                        "Параллельные процессы (jobs)",
                        key="parallel_jobs",
                        min_value=1,
                        step=1,
                        help=(
                            "Сколько процессов одновременно проверяют кандидаты в coplanar-режиме. "
                            "1 = без параллелизма. Обычно ставьте не больше числа физических ядер."
                        ),
                    )
    return bool(build_clicked)


def _render_solver_profiling_panel() -> None:
    with st.expander("Профилирование TURN-методов", expanded=False):
        st.caption("Сравнение методов на типовых шаблонах текущего проекта.")
        if st.button(
            "Запустить профилирование", icon=":material/speed:", width="content"
        ):
            _run_solver_profiling()
        profile_rows = st.session_state.get("solver_profile_rows")
        if st.session_state.get("solver_profile_at"):
            st.caption(
                f"Последнее профилирование: {st.session_state['solver_profile_at']}"
            )
        if profile_rows:
            profile_df = pd.DataFrame(profile_rows)
            st.dataframe(
                arrow_safe_text_dataframe(profile_df), width="stretch", hide_index=True
            )


def _run_planner_if_clicked(
    build_clicked: bool,
    preflight_errors: list[str],
    surface_input: Point3D,
    t1_input: Point3D,
    t3_input: Point3D,
    config_input: TrajectoryConfig,
) -> None:
    if not build_clicked:
        return
    run_started_s = perf_counter()
    log_lines: list[str] = []
    progress = st.progress(0, text="Подготовка расчета...")
    phase_placeholder = st.empty()
    try:
        with st.spinner("Расчет траектории...", show_time=True):
            check_msg = format_run_log_line(
                run_started_s,
                "Проверка входных данных.",
            )
            log_lines.append(check_msg)
            phase_placeholder.caption("Проверка входных данных...")
            progress.progress(20, text="Проверка входных данных...")
            if preflight_errors:
                raise PlanningError("; ".join(preflight_errors))
            solve_msg = format_run_log_line(
                run_started_s,
                "Запуск планировщика и решателя. Инициализация фаз расчета.",
            )
            log_lines.append(solve_msg)
            phase_placeholder.caption("Запуск солвера...")
            progress.progress(35, text="Запуск солвера...")

            callback_state: dict[str, object] = {"last_text": "", "last_progress": 35}

            def planner_progress(stage_text: str, stage_fraction: float) -> None:
                clamped = float(max(0.0, min(1.0, stage_fraction)))
                mapped_progress = int(35 + round(clamped * 60))
                mapped_progress = max(
                    int(callback_state["last_progress"]),
                    min(mapped_progress, 99),
                )
                progress.progress(mapped_progress, text=stage_text)
                phase_placeholder.caption(stage_text)
                if str(callback_state["last_text"]) != stage_text:
                    log_line = format_run_log_line(run_started_s, stage_text)
                    log_lines.append(log_line)
                    callback_state["last_text"] = stage_text
                callback_state["last_progress"] = mapped_progress

            started = perf_counter()
            _run_planner(
                surface=surface_input,
                t1=t1_input,
                t3=t3_input,
                config=config_input,
                progress_callback=planner_progress,
            )
            elapsed_s = perf_counter() - started
            st.session_state["last_runtime_s"] = float(elapsed_s)
            done_msg = format_run_log_line(
                run_started_s,
                f"Расчет завершен успешно. Затраченное время солвера: {elapsed_s:.2f} с.",
            )
            log_lines.append(done_msg)
            progress.progress(100, text="Расчет завершен.")
            phase_placeholder.success(f"Расчет завершен за {elapsed_s:.2f} с")
    except PlanningError as exc:
        st.session_state["last_error"] = str(exc)
        st.session_state["last_runtime_s"] = None
        err_msg = format_run_log_line(run_started_s, f"Ошибка расчета: {exc}")
        log_lines.append(err_msg)
        phase_placeholder.error("Расчет завершился ошибкой")
    finally:
        st.session_state["last_run_log_lines"] = log_lines
        progress.empty()


def _render_calculation_feedback() -> None:
    if st.session_state["last_error"]:
        render_solver_diagnostics(st.session_state["last_error"])
    render_run_log_panel(st.session_state.get("last_run_log_lines"))


def _format_summary_value(metric_key: str, metric_value: object) -> str:
    if isinstance(metric_value, str):
        return metric_value
    if metric_value is None:
        return "—"
    try:
        value = float(metric_value)
    except (TypeError, ValueError):
        return str(metric_value)

    if metric_key.endswith("_deg") or "_deg_" in metric_key:
        return f"{value:.2f}"
    if metric_key.endswith("_m") or metric_key.startswith("md_"):
        return f"{value:.2f}"
    return f"{value:.4g}"


def _render_result_overview(last_result: dict[str, object]) -> float:
    summary = last_result["summary"]
    surface = last_result["surface"]
    t1 = last_result["t1"]
    runtime_s = st.session_state.get("last_runtime_s")
    t1_horizontal_offset_m = _horizontal_offset_m(point=t1, reference=surface)
    with st.container(border=True):
        st.markdown("### Ключевые показатели")
        metrics_rows = [
            {"Показатель": "Тип траектории", "Значение": str(summary["trajectory_type"])},
            {"Показатель": "Класс сложности", "Значение": str(summary["well_complexity"])},
            {"Показатель": "INC на t1", "Значение": f"{float(summary['entry_inc_deg']):.2f} deg"},
            {"Показатель": "ЗУ HOLD", "Значение": f"{float(summary['hold_inc_deg']):.2f} deg"},
            {"Показатель": "Отход t1", "Значение": format_distance(t1_horizontal_offset_m)},
            {"Показатель": "KOP MD", "Значение": format_distance(float(summary["kop_md_m"]))},
            {
                "Показатель": "Длина HORIZONTAL",
                "Значение": format_distance(float(summary["horizontal_length_m"])),
            },
            {
                "Показатель": "Макс INC факт/лимит",
                "Значение": f"{float(summary['max_inc_actual_deg']):.2f}/{float(summary['max_inc_deg']):.2f} deg",
            },
            {
                "Показатель": "Макс DLS",
                "Значение": f"{float(summary['max_dls_total_deg_per_30m']):.2f} deg/30m",
            },
            {"Показатель": "Итоговая MD", "Значение": format_distance(float(summary["md_total_m"]))},
            {
                "Показатель": "Время расчета",
                "Значение": "—" if runtime_s is None else f"{float(runtime_s):.2f} с",
            },
        ]
        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(metrics_rows)),
            width="stretch",
            hide_index=True,
        )

        render_small_note(
            "Проверяйте соответствие фактического INC/DLS лимитам, особенно при изменении "
            "целевого угла входа и границ DLS."
        )
    return float(t1_horizontal_offset_m)


def _render_result_plots(last_result: dict[str, object]) -> None:
    stations = last_result["stations"]
    surface = last_result["surface"]
    t1 = last_result["t1"]
    t3 = last_result["t3"]
    config = last_result["config"]
    azimuth_deg = float(last_result["azimuth_deg"])
    md_t1_m = float(last_result["md_t1_m"])
    render_trajectory_dls_panel(
        stations=stations,
        surface=surface,
        t1=t1,
        t3=t3,
        md_t1_m=md_t1_m,
        dls_limits=config.dls_limits_deg_per_30m,
        title="3D траектория и DLS",
        border=True,
    )
    render_plan_section_panel(
        stations=stations,
        surface=surface,
        t1=t1,
        t3=t3,
        azimuth_deg=azimuth_deg,
        title="План и вертикальный разрез",
        border=True,
    )


def _render_result_tables(
    last_result: dict[str, object], t1_horizontal_offset_m: float
) -> None:
    summary = last_result["summary"]
    stations = last_result["stations"]
    tab_summary, tab_survey = st.tabs(["Сводка", "Инклинометрия"])
    with tab_summary:
        hidden_metrics = {"distance_t1_m", "distance_t3_m"}
        summary_visible = {
            key: value for key, value in summary.items() if key not in hidden_metrics
        }
        summary_visible["t1_horizontal_offset_m"] = t1_horizontal_offset_m

        main_rows: list[dict[str, str]] = []
        for key, label in SUMMARY_MAIN_METRICS:
            if key not in summary_visible:
                continue
            main_rows.append(
                {
                    "Показатель": label,
                    "Значение": _format_summary_value(key, summary_visible[key]),
                }
            )
        if "t1_horizontal_offset_m" in summary_visible:
            main_rows.insert(
                4,
                {
                    "Показатель": "Горизонтальный отход t1, м",
                    "Значение": _format_summary_value(
                        "t1_horizontal_offset_m",
                        summary_visible["t1_horizontal_offset_m"],
                    ),
                },
            )

        tech_rows: list[dict[str, str]] = []
        for key, value in sorted(summary_visible.items()):
            if any(key == metric_key for metric_key, _ in SUMMARY_MAIN_METRICS):
                continue
            if key == "t1_horizontal_offset_m":
                continue
            tech_rows.append(
                {
                    "Параметр": key,
                    "Значение": _format_summary_value(key, value),
                }
            )

        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(main_rows)),
            width="stretch",
            hide_index=True,
        )
        with st.expander(
            "Технические параметры и диагностика решателя", expanded=False
        ):
            st.dataframe(
                arrow_safe_text_dataframe(pd.DataFrame(tech_rows)),
                width="stretch",
                hide_index=True,
            )

    with tab_survey:
        render_survey_table_with_download(
            stations=stations,
            button_label="Скачать CSV инклинометрии",
            file_name="well_survey.csv",
        )


def _render_last_result() -> None:
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

    t1_horizontal_offset_m = _render_result_overview(last_result=last_result)
    _render_result_plots(last_result=last_result)
    _render_result_tables(
        last_result=last_result, t1_horizontal_offset_m=t1_horizontal_offset_m
    )


def run_app() -> None:
    st.set_page_config(page_title="Планировщик траектории скважины", layout="wide")

    _init_state()
    apply_page_style(max_width_px=1680)
    render_hero(title="Планировщик траектории скважины")
    _render_template_controls()
    build_clicked = _render_input_form()

    surface_input, t1_input, t3_input = _build_points_from_state()
    config_input = _build_config_from_state()
    preflight_errors = _validate_input(
        surface=surface_input, t1=t1_input, t3=t3_input, config=config_input
    )
    if preflight_errors:
        st.warning(
            "Предварительная проверка параметров:\n- " + "\n- ".join(preflight_errors)
        )
    _render_solver_profiling_panel()
    _run_planner_if_clicked(
        build_clicked=build_clicked,
        preflight_errors=preflight_errors,
        surface_input=surface_input,
        t1_input=t1_input,
        t3_input=t3_input,
        config_input=config_input,
    )
    _render_calculation_feedback()
    _render_last_result()


if __name__ == "__main__":
    if not st.runtime.exists():
        raise SystemExit("Запустите приложение командой `streamlit run app.py`.")
    run_app()
