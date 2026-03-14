from __future__ import annotations

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
    TURN_SOLVER_OPTIONS,
)
from pywp.pydantic_base import FrozenModel
from pywp.solver_diagnostics import summarize_problem_ru
from pywp.solver_diagnostics_ui import render_solver_diagnostics
from pywp.ui_calc_params import (
    CalcParamBinding,
)
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_utils import (
    arrow_safe_text_dataframe,
    format_run_log_line,
)
from pywp.ui_well_panels import render_run_log_panel
from pywp.ui_well_result import (
    SingleWellResultView,
    horizontal_offset_m,
    render_key_metrics,
    render_result_plots,
    render_result_tables,
)


class ScenarioPreset(FrozenModel):
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
UI_DEFAULTS_VERSION = 10
APP_CALC_PARAMS = CalcParamBinding(prefix="")


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


def _apply_points_to_state(surface: Point3D, t1: Point3D, t3: Point3D) -> None:
    st.session_state["surface_x"] = float(surface.x)
    st.session_state["surface_y"] = float(surface.y)
    st.session_state["surface_z"] = float(surface.z)
    st.session_state["t1_x"] = float(t1.x)
    st.session_state["t1_y"] = float(t1.y)
    st.session_state["t1_z"] = float(t1.z)
    st.session_state["t3_x"] = float(t3.x)
    st.session_state["t3_y"] = float(t3.y)
    st.session_state["t3_z"] = float(t3.z)


def _parse_points_import_text(raw_text: str) -> tuple[Point3D, Point3D, Point3D]:
    normalized = str(raw_text).replace("\\n", "\n").replace("\\r", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) != 3:
        raise ValueError("Ожидалось 3 строки (s1, t1, t3). " f"Получено: {len(lines)}.")

    parsed_points: list[Point3D] = []
    for line_no, line in enumerate(lines, start=1):
        parts = [part.strip() for part in line.split()]
        if len(parts) != 3:
            raise ValueError(
                f"Строка {line_no}: ожидались 3 значения через пробел или табуляцию."
            )
        try:
            x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise ValueError(f"Строка {line_no}: не удалось распознать числа.") from exc
        parsed_points.append(Point3D(x=x, y=y, z=z))

    return parsed_points[0], parsed_points[1], parsed_points[2]


def _parse_trajectory_import_text(raw_text: str, *, profile_label: str) -> pd.DataFrame:
    normalized = str(raw_text).replace("\\n", "\n").replace("\\r", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(
            f"Ожидалось минимум 2 строки `x y z` для {profile_label}."
        )

    rows: list[dict[str, float]] = []
    for line_no, line in enumerate(lines, start=1):
        parts = [part.strip() for part in line.split()]
        if len(parts) != 3:
            raise ValueError(
                f"Строка {line_no}: ожидались 3 значения через пробел или табуляцию."
            )
        try:
            x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise ValueError(f"Строка {line_no}: не удалось распознать числа.") from exc
        rows.append({"X_m": x, "Y_m": y, "Z_m": z})

    return pd.DataFrame(rows, columns=["X_m", "Y_m", "Z_m"], dtype=float)


def _parse_plan_csb_import_text(raw_text: str) -> pd.DataFrame:
    return _parse_trajectory_import_text(raw_text, profile_label="плана ЦСБ")


def _parse_actual_trajectory_import_text(raw_text: str) -> pd.DataFrame:
    return _parse_trajectory_import_text(raw_text, profile_label="фактического профиля")


def _clear_profile_import_state() -> None:
    st.session_state["plan_csb_df"] = None
    st.session_state["actual_profile_df"] = None
    st.session_state["actual_trajectory_df"] = None
    st.session_state["plan_csb_import_text"] = ""
    st.session_state["actual_profile_import_text"] = ""
    st.session_state["actual_trajectory_import_text"] = ""


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

    APP_CALC_PARAMS.preserve_state()
    APP_CALC_PARAMS.apply_defaults(force=False)
    st.session_state.setdefault("ui_defaults_version", 0)

    if int(st.session_state.get("ui_defaults_version", 0)) < UI_DEFAULTS_VERSION:
        APP_CALC_PARAMS.apply_defaults(force=True)
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
    st.session_state.setdefault("points_import_text", "")
    st.session_state.setdefault("plan_csb_import_text", "")
    st.session_state.setdefault("plan_csb_df", None)
    st.session_state.setdefault("actual_profile_import_text", "")
    st.session_state.setdefault("actual_profile_df", None)
    st.session_state.setdefault("actual_trajectory_import_text", "")
    st.session_state.setdefault("actual_trajectory_df", None)
    legacy_plan_df = st.session_state.get("actual_trajectory_df")
    if (
        st.session_state.get("plan_csb_df") is None
        and isinstance(legacy_plan_df, pd.DataFrame)
        and len(legacy_plan_df) > 0
    ):
        st.session_state["plan_csb_df"] = legacy_plan_df


def _clear_result() -> None:
    st.session_state["last_result"] = None
    st.session_state["last_error"] = ""
    st.session_state["last_built_at"] = ""
    st.session_state["last_runtime_s"] = None
    st.session_state["last_input_signature"] = None
    st.session_state["last_run_log_lines"] = []


def _current_input_signature() -> tuple[object, ...]:
    point_keys = (
        "surface_x",
        "surface_y",
        "surface_z",
        "t1_x",
        "t1_y",
        "t1_z",
        "t3_x",
        "t3_y",
        "t3_z",
    )
    signature = [float(st.session_state[key]) for key in point_keys]
    signature.extend(APP_CALC_PARAMS.state_signature())
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
    return APP_CALC_PARAMS.build_config()


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
    progress = st.progress(0, text="Профилирование методов решателя...")
    with st.status(
        "Выполняется профилирование методов решателя...", expanded=True
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
            config = base_config.validated_copy(turn_solver_mode=solver_mode)
            started = perf_counter()
            try:
                result = planner.plan(surface=surface, t1=t1, t3=t3, config=config)
                elapsed_s = perf_counter() - started
                rows.append(
                    {
                        "Шаблон": scenario_name,
                        "Метод решателя": TURN_SOLVER_OPTIONS[solver_mode],
                        "Статус": "OK",
                        "Время, с": f"{elapsed_s:.3f}",
                        "Промах t1, м": f"{float(result.summary['distance_t1_m']):.4f}",
                        "Промах t3, м": f"{float(result.summary['distance_t3_m']):.4f}",
                        "Поворот по азимуту, deg": f"{float(result.summary.get('azimuth_turn_deg', 0.0)):.2f}",
                        "Модель траектории": str(
                            result.summary.get("trajectory_type", "—")
                        ),
                        "Классификация целей": str(
                            result.summary.get("trajectory_target_direction", "—")
                        ),
                    }
                )
            except PlanningError as exc:
                elapsed_s = perf_counter() - started
                rows.append(
                    {
                        "Шаблон": scenario_name,
                        "Метод решателя": TURN_SOLVER_OPTIONS[solver_mode],
                        "Статус": "Ошибка",
                        "Время, с": f"{elapsed_s:.3f}",
                        "Промах t1, м": "—",
                        "Промах t3, м": "—",
                        "Поворот по азимуту, deg": "—",
                        "Модель траектории": "—",
                        "Классификация целей": "—",
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
        with st.expander(
            "Каталог шаблонов конструкции",
            expanded=False,
            icon=":material/view_module:",
        ):
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

            offset_t1 = horizontal_offset_m(
                point=selected_preset.t1,
                reference=selected_preset.surface,
            )
            reverse_window = _preset_reverse_window_label(gv_m=selected_preset.gv_m)
            m1, m2, m3, m4 = st.columns(4, gap="small")
            m1.metric("ГВ t1", f"{selected_preset.gv_m:.0f} м")
            m2.metric("Отход t1", f"{offset_t1:.0f} м")
            m3.metric("Референс reverse по ГВ", reverse_window)
            m4.metric("Класс шаблона", complexity_label(selected_preset.complexity))


def _render_point_config_block() -> None:
    with st.container(border=True):
        st.markdown("### Конфигурация точек (S, t1, t3)")
        st.caption(
            "Это геометрия задачи (координаты точек). Параметры расчета и солвера задаются отдельным блоком ниже."
        )
        st.markdown("**Импорт точек из текста (s1, t1, t3)**")
        st.text_area(
            "Формат: 3 строки __`x y z`__ для S, t1, t3",
            key="points_import_text",
            height=110,
            placeholder=("0        0        0\n" "600   800   2400\n" "1500 2000 2500"),
            help=(
                "Разделитель между координатами: пробел или табуляция (можно вставить из экселя)"
                "Порядок строк: s1, затем t1, затем t3."
            ),
        )
        import_clicked = st.form_submit_button(
            "Импортировать точки",
            type="secondary",
            icon=":material/upload_file:",
            width="content",
        )
        if import_clicked:
            try:
                imported_surface, imported_t1, imported_t3 = _parse_points_import_text(
                    str(st.session_state.get("points_import_text", ""))
                )
            except ValueError as exc:
                st.error(f"Импорт не выполнен: {exc}")
            else:
                _apply_points_to_state(
                    surface=imported_surface,
                    t1=imported_t1,
                    t3=imported_t3,
                )
                st.success("Точки успешно импортированы в поля S/t1/t3.")
        with st.expander(
            "Плановые профили ствола ЦСБ и фактические траектории для сравнения (опционально)",
            expanded=False,
        ):
            p1, p2 = st.columns(2, gap="small")
            p1.text_area(
                "План ЦСБ: формат `x y z` (много строк)",
                key="plan_csb_import_text",
                height=150,
                placeholder=(
                    "0      0      0\n"
                    "100   140   600\n"
                    "240   320   1200\n"
                    "600   800   2400\n"
                    "1500 2000 2500"
                ),
                help=(
                    "Плановый профиль ствола ЦСБ для сравнения с расчетной траекторией. "
                    "Разделитель между координатами: пробел или табуляция."
                ),
            )
            p2.text_area(
                "Фактический профиль: формат `x y z` (много строк)",
                key="actual_profile_import_text",
                height=150,
                placeholder=(
                    "0      0      0\n"
                    "96    136   590\n"
                    "233   309   1180\n"
                    "594   791   2387\n"
                    "1490 1987   2488"
                ),
                help=(
                    "Фактическая траектория скважины для визуального сравнения. "
                    "Разделитель между координатами: пробел или табуляция."
                ),
            )
            a1, a2 = st.columns(2, gap="small")
            import_profiles_clicked = a1.form_submit_button(
                "Импортировать профили",
                type="secondary",
                icon=":material/polyline:",
                width="stretch",
            )
            clear_profiles_clicked = a2.form_submit_button(
                "Очистить профили",
                type="secondary",
                icon=":material/delete_sweep:",
                width="stretch",
                on_click=_clear_profile_import_state,
            )
            if import_profiles_clicked:
                import_errors: list[str] = []
                plan_text = str(st.session_state.get("plan_csb_import_text", "")).strip()
                fact_text = str(
                    st.session_state.get("actual_profile_import_text", "")
                ).strip()
                if plan_text:
                    try:
                        plan_df = _parse_plan_csb_import_text(plan_text)
                    except ValueError as exc:
                        import_errors.append(f"План ЦСБ: {exc}")
                    else:
                        st.session_state["plan_csb_df"] = plan_df
                if fact_text:
                    try:
                        actual_df = _parse_actual_trajectory_import_text(fact_text)
                    except ValueError as exc:
                        import_errors.append(f"Фактический профиль: {exc}")
                    else:
                        st.session_state["actual_profile_df"] = actual_df
                if not plan_text and not fact_text:
                    import_errors.append(
                        "Заполните хотя бы одно поле: План ЦСБ или Фактический профиль."
                    )
                for error_message in import_errors:
                    st.error(f"Импорт не выполнен: {error_message}")
            if clear_profiles_clicked:
                st.success("Профили очищены.")

            plan_loaded = st.session_state.get("plan_csb_df")
            fact_loaded = st.session_state.get("actual_profile_df")
            plan_count = (
                len(plan_loaded)
                if isinstance(plan_loaded, pd.DataFrame) and len(plan_loaded) > 0
                else 0
            )
            fact_count = (
                len(fact_loaded)
                if isinstance(fact_loaded, pd.DataFrame) and len(fact_loaded) > 0
                else 0
            )
            if plan_count and fact_count:
                st.success(
                    f"Загружено точек: План ЦСБ — {plan_count}, Фактический профиль — {fact_count}."
                )
            elif plan_count:
                st.success(f"Загружено точек План ЦСБ: {plan_count}.")
            elif fact_count:
                st.success(f"Загружено точек Фактического профиля: {fact_count}.")
            else:
                st.caption("Профили не загружены.")
        c1, c2, c3 = st.columns(3, gap="small", border=True, vertical_alignment="top")
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


def _render_input_form() -> bool:
    with st.form(
        "planner_form", clear_on_submit=False, enter_to_submit=False, border=False
    ):
        _render_point_config_block()
        with st.container(border=True):
            APP_CALC_PARAMS.render_block(show_solver_help=True)
        with st.container(border=True):
            st.markdown("### Расчет траектории")
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
    return bool(build_clicked)


def _render_solver_profiling_panel() -> None:
    with st.expander("Профилирование методов решателя", expanded=False):
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
        err_msg = format_run_log_line(
            run_started_s,
            f"Ошибка расчета: {summarize_problem_ru(str(exc))}",
        )
        log_lines.append(err_msg)
        phase_placeholder.error("Расчет завершился ошибкой")
    finally:
        st.session_state["last_run_log_lines"] = log_lines
        progress.empty()


def _render_calculation_feedback() -> None:
    if st.session_state["last_error"]:
        render_solver_diagnostics(st.session_state["last_error"])
    render_run_log_panel(st.session_state.get("last_run_log_lines"))


def _build_single_well_result_view(
    last_result: dict[str, object],
) -> SingleWellResultView:
    plan_csb_stations = st.session_state.get("plan_csb_df")
    plan_csb_df = (
        plan_csb_stations
        if isinstance(plan_csb_stations, pd.DataFrame) and len(plan_csb_stations) > 0
        else None
    )
    actual_stations = st.session_state.get("actual_profile_df")
    if actual_stations is None:
        actual_stations = st.session_state.get("actual_trajectory_df")
    actual_df = (
        actual_stations
        if isinstance(actual_stations, pd.DataFrame) and len(actual_stations) > 0
        else None
    )
    return SingleWellResultView(
        well_name="single_well",
        surface=last_result["surface"],
        t1=last_result["t1"],
        t3=last_result["t3"],
        stations=last_result["stations"],
        summary=last_result["summary"],
        config=last_result["config"],
        azimuth_deg=float(last_result["azimuth_deg"]),
        md_t1_m=float(last_result["md_t1_m"]),
        runtime_s=st.session_state.get("last_runtime_s"),
        plan_csb_stations=plan_csb_df,
        actual_stations=actual_df,
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

    well_view = _build_single_well_result_view(last_result=last_result)
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
        survey_file_name="well_survey.csv",
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
