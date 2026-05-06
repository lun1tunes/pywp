from __future__ import annotations

from typing import Callable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import field_validator

from pywp.models import Point3D, SummaryValue, TrajectoryConfig
from pywp.planner_config import optimization_display_label
from pywp.pydantic_base import FrozenArbitraryModel, coerce_model_like
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    UNCERTAINTY_PRESET_OPTIONS,
    build_uncertainty_overlay,
    normalize_uncertainty_preset,
    planning_uncertainty_model_for_preset,
    uncertainty_model_caption,
    uncertainty_preset_label,
)
from pywp.ui_utils import arrow_safe_text_dataframe, dls_to_pi, format_distance
from pywp.ui_well_panels import (
    render_plan_section_panel,
    render_survey_table_with_download,
    render_trajectory_dls_panel,
)

SUMMARY_MAIN_METRICS: tuple[tuple[str, str], ...] = (
    ("trajectory_type", "Модель траектории"),
    ("trajectory_target_direction", "Классификация целей"),
    ("well_complexity", "Класс сложности"),
    ("entry_inc_deg", "Угол входа в пласт, deg"),
    ("hold_inc_deg", "ЗУ секции HOLD, deg"),
    ("hold_length_m", "Длина HOLD, м"),
    ("build1_dls_selected_deg_per_30m", "Выбранный BUILD1 ПИ, deg/10m"),
    ("build2_dls_selected_deg_per_30m", "Выбранный BUILD2 ПИ, deg/10m"),
    ("horizontal_length_m", "Длина ГС, м"),
    ("kop_md_m", "KOP MD, м"),
    ("max_dls_total_deg_per_30m", "Макс ПИ по стволу, deg/10m"),
    ("max_inc_actual_deg", "Макс INC фактический, deg"),
    ("max_inc_deg", "Макс INC лимит, deg"),
    ("md_total_m", "Итоговая MD, м"),
    ("max_total_md_postcheck_m", "Лимит итоговой MD (постпроверка), м"),
)

SUMMARY_TECH_HIDDEN_METRICS = frozenset(
    {
        "distance_t1_m",
        "distance_t3_m",
        "lateral_distance_t1_m",
        "lateral_distance_t3_m",
        "vertical_distance_t1_m",
        "vertical_distance_t3_m",
        "distance_t1_control_m",
        "distance_t3_control_m",
        "control_gap_t1_m",
        "control_gap_t3_m",
        "entry_inc_control_deg",
        "t1_exact_x_m",
        "t1_exact_y_m",
        "t1_exact_z_m",
        "t3_exact_x_m",
        "t3_exact_y_m",
        "t3_exact_z_m",
        "t1_miss_dx_m",
        "t1_miss_dy_m",
        "t1_miss_dz_m",
        "t3_miss_dx_m",
        "t3_miss_dy_m",
        "t3_miss_dz_m",
    }
)


class SingleWellResultView(FrozenArbitraryModel):
    well_name: str
    surface: Point3D
    t1: Point3D
    t3: Point3D
    stations: pd.DataFrame
    summary: Mapping[str, SummaryValue]
    config: TrajectoryConfig
    azimuth_deg: float
    md_t1_m: float
    runtime_s: float | None = None
    issue_messages: tuple[str, ...] = ()
    trajectory_line_dash: str = "solid"
    plan_csb_stations: pd.DataFrame | None = None
    actual_stations: pd.DataFrame | None = None

    @field_validator("surface", "t1", "t3", mode="before")
    @classmethod
    def _coerce_point3d(cls, value: object) -> Point3D:
        return coerce_model_like(value, Point3D)

    @field_validator("config", mode="before")
    @classmethod
    def _coerce_config(cls, value: object) -> TrajectoryConfig:
        return coerce_model_like(value, TrajectoryConfig)


def uncertainty_toggle_key(*, well_name: str) -> str:
    return f"show_uncertainty_ellipses::{str(well_name)}"


def uncertainty_preset_key(*, well_name: str) -> str:
    return f"uncertainty_preset::{str(well_name)}"


def horizontal_offset_m(*, point: Point3D, reference: Point3D) -> float:
    dx = float(point.x - reference.x)
    dy = float(point.y - reference.y)
    return float((dx * dx + dy * dy) ** 0.5)


def md_postcheck_issue_message(summary: Mapping[str, float | str]) -> str:
    md_postcheck_excess_m = float(summary.get("md_postcheck_excess_m", 0.0))
    if md_postcheck_excess_m <= 1e-6:
        return ""
    return (
        "Превышен лимит итоговой MD (постпроверка): "
        f"{float(summary.get('md_total_m', 0.0)):.2f} м > "
        f"{float(summary.get('max_total_md_postcheck_m', 0.0)):.2f} м "
        f"(+{md_postcheck_excess_m:.2f} м)."
    )


def collect_issue_messages(
    *,
    summary: Mapping[str, float | str],
    extra_messages: Sequence[str] = (),
) -> tuple[str, ...]:
    messages: list[str] = []
    md_message = md_postcheck_issue_message(summary)
    if md_message:
        messages.append(md_message)
    for message in extra_messages:
        text = str(message).strip()
        if text:
            messages.append(text)
    unique: list[str] = []
    for message in messages:
        if message not in unique:
            unique.append(message)
    return tuple(unique)


def _format_summary_value(metric_key: str, metric_value: object) -> str:
    if isinstance(metric_value, str):
        return metric_value
    if metric_value is None:
        return "—"
    try:
        value = float(metric_value)
    except (TypeError, ValueError):
        return str(metric_value)

    if "dls" in str(metric_key).lower():
        return f"{dls_to_pi(value):.2f}"
    if metric_key.endswith("_deg") or "_deg_" in metric_key:
        return f"{value:.2f}"
    if metric_key.endswith("_m") or metric_key.startswith("md_"):
        return f"{value:.2f}"
    return f"{value:.4g}"


def _format_summary_key(metric_key: str) -> str:
    key = str(metric_key)
    if "dls" not in key.lower():
        return key
    key = key.replace("DLS", "PI").replace("dls", "pi")
    key = key.replace("_deg_per_30m", "_deg_per_10m")
    return key


def _format_metric_text(metric_value: object, *, missing_text: str = "-") -> str:
    if metric_value is None:
        return missing_text
    text = str(metric_value).strip()
    return missing_text if not text else text


def _format_issues_text(
    *,
    summary: Mapping[str, SummaryValue],
    extra_messages: Sequence[str] = (),
    missing_text: str = "-",
) -> str:
    messages = collect_issue_messages(summary=summary, extra_messages=extra_messages)
    if not messages:
        return missing_text
    return " | ".join(messages)


def _format_runtime_text(runtime_s: float | None, *, missing_text: str = "-") -> str:
    if runtime_s is None:
        return missing_text
    return f"{float(runtime_s):.2f} с"


def _format_solver_restart_text(
    summary: Mapping[str, SummaryValue],
    *,
    missing_text: str = "-",
) -> str:
    used_raw = summary.get("solver_turn_restarts_used")
    max_raw = summary.get("solver_turn_max_restarts")
    if used_raw is None and max_raw is None:
        return missing_text
    try:
        used = int(float(0.0 if used_raw is None else used_raw))
    except (TypeError, ValueError):
        return missing_text
    if max_raw is None:
        return str(used)
    try:
        max_restarts = int(float(max_raw))
    except (TypeError, ValueError):
        return str(used)
    return f"{used} / {max_restarts}"


def build_key_metrics_rows(view: SingleWellResultView) -> list[dict[str, str]]:
    summary = view.summary
    t1_horizontal_offset_m = horizontal_offset_m(point=view.t1, reference=view.surface)

    optimization_mode = str(summary.get("optimization_mode", view.config.optimization_mode))
    optimization_label = optimization_display_label(optimization_mode)

    rows = [
        {
            "Показатель": "Модель траектории",
            "Значение": _format_metric_text(summary.get("trajectory_type")),
        },
        {
            "Показатель": "Оптимизация",
            "Значение": optimization_label,
        },
        {
            "Показатель": "Цели / сложность",
            "Значение": (
                f"{_format_metric_text(summary.get('trajectory_target_direction'))} / "
                f"{_format_metric_text(summary.get('well_complexity'))}"
            ),
        },
        {
            "Показатель": "Поворот по азимуту",
            "Значение": f"{float(summary.get('azimuth_turn_deg', 0.0)):.2f} deg",
        },
        {
            "Показатель": "Отход t1",
            "Значение": format_distance(t1_horizontal_offset_m),
        },
        {
            "Показатель": "INC t1 / ЗУ HOLD",
            "Значение": (
                f"{float(summary['entry_inc_deg']):.2f} / "
                f"{float(summary['hold_inc_deg']):.2f} deg"
            ),
        },
        {
            "Показатель": "BUILD1 / BUILD2 / Макс ПИ",
            "Значение": (
                f"{dls_to_pi(float(summary.get('build1_dls_selected_deg_per_30m', summary.get('build_dls_selected_deg_per_30m', 0.0)))):.2f} / "
                f"{dls_to_pi(float(summary.get('build2_dls_selected_deg_per_30m', summary.get('build_dls_selected_deg_per_30m', 0.0)))):.2f} / "
                f"{dls_to_pi(float(summary['max_dls_total_deg_per_30m'])):.2f} deg/10m"
            ),
        },
        {
            "Показатель": "KOP MD",
            "Значение": format_distance(float(summary["kop_md_m"])),
        },
        {
            "Показатель": "Длина ГС",
            "Значение": format_distance(float(summary["horizontal_length_m"])),
        },
        {
            "Показатель": "Макс INC факт / лимит",
            "Значение": (
                f"{float(summary['max_inc_actual_deg']):.2f} / "
                f"{float(summary['max_inc_deg']):.2f} deg"
            ),
        },
        {
            "Показатель": "Итоговая MD",
            "Значение": format_distance(float(summary["md_total_m"])),
        },
        {
            "Показатель": "Рестарты решателя",
            "Значение": _format_solver_restart_text(summary),
        },
        {
            "Показатель": "Проблемы",
            "Значение": _format_issues_text(
                summary=summary,
                extra_messages=view.issue_messages,
            ),
        },
        {
            "Показатель": "Время расчета",
            "Значение": _format_runtime_text(view.runtime_s),
        },
    ]
    return rows


def build_target_validation_rows(
    summary: Mapping[str, SummaryValue],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add_distance_row(label: str, key: str) -> None:
        value = summary.get(key)
        if value is None:
            return
        rows.append({"Показатель": label, "Значение": format_distance(float(value))})

    def add_vector_row(label: str, dx_key: str, dy_key: str, dz_key: str) -> None:
        if not all(key in summary for key in (dx_key, dy_key, dz_key)):
            return
        rows.append(
            {
                "Показатель": label,
                "Значение": (
                    f"{float(summary[dx_key]):.2f} / "
                    f"{float(summary[dy_key]):.2f} / "
                    f"{float(summary[dz_key]):.2f} м"
                ),
            }
        )

    add_distance_row("Промах t1 (аналитический, 3D)", "distance_t1_m")
    add_distance_row("Промах t3 (аналитический, 3D)", "distance_t3_m")
    add_distance_row("Промах t1 по латерали", "lateral_distance_t1_m")
    add_distance_row("Промах t1 по вертикали", "vertical_distance_t1_m")
    add_distance_row("Промах t3 по латерали", "lateral_distance_t3_m")
    add_distance_row("Промах t3 по вертикали", "vertical_distance_t3_m")
    add_distance_row("Промах t1 (control-grid)", "distance_t1_control_m")
    add_distance_row("Промах t3 (control-grid)", "distance_t3_control_m")
    add_distance_row("Расхождение t1: analytic vs control-grid", "control_gap_t1_m")
    add_distance_row("Расхождение t3: analytic vs control-grid", "control_gap_t3_m")
    add_vector_row("Компоненты промаха t1 (dX / dY / dZ)", "t1_miss_dx_m", "t1_miss_dy_m", "t1_miss_dz_m")
    add_vector_row("Компоненты промаха t3 (dX / dY / dZ)", "t3_miss_dx_m", "t3_miss_dy_m", "t3_miss_dz_m")
    if "entry_inc_deg" in summary and "entry_inc_control_deg" in summary:
        rows.append(
            {
                "Показатель": "INC на t1: analytic / control-grid",
                "Значение": (
                    f"{float(summary['entry_inc_deg']):.2f} / "
                    f"{float(summary['entry_inc_control_deg']):.2f} deg"
                ),
            }
        )
    return rows


def render_key_metrics(
    *,
    view: SingleWellResultView,
    title: str = "Ключевые показатели",
    border: bool = True,
) -> float:
    summary = view.summary
    t1_horizontal_offset_m = horizontal_offset_m(point=view.t1, reference=view.surface)
    issue_messages = collect_issue_messages(
        summary=summary,
        extra_messages=view.issue_messages,
    )

    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        metrics_rows = build_key_metrics_rows(view)
        st.table(
            arrow_safe_text_dataframe(pd.DataFrame(metrics_rows))
        )
        for message in issue_messages:
            st.warning(message)
        st.caption(
            "Проверяйте соответствие фактического INC/ПИ лимитам, особенно при изменении "
            "целевого угла входа и границ ПИ."
        )

    if border:
        with st.container(border=True):
            _render_body()
    else:
        _render_body()
    return float(t1_horizontal_offset_m)


def render_result_plots(
    *,
    view: SingleWellResultView,
    title_trajectory: str | None,
    title_plan: str | None,
    border: bool = True,
    render_3d_override: Callable[[object, go.Figure], None] | None = None,
) -> None:
    show_uncertainty = st.checkbox(
        "Показать конус неопределенности",
        key=uncertainty_toggle_key(well_name=view.well_name),
        help=(
            "Включает planning-level 2σ конус неопределенности для 3D, плана и "
            "вертикального разреза. Он строится по последовательности эллиптических "
            "сечений вдоль ствола. Это базовая визуализация по first-order "
            "ошибкам INC/AZI, без полной ISCWSA tool model."
        ),
    )
    uncertainty_overlay = None
    if show_uncertainty:
        preset_key = uncertainty_preset_key(well_name=view.well_name)
        st.session_state[preset_key] = normalize_uncertainty_preset(
            st.session_state.get(preset_key, DEFAULT_UNCERTAINTY_PRESET)
        )
        selected_preset = st.selectbox(
            "Пресет неопределенности",
            options=list(UNCERTAINTY_PRESET_OPTIONS.keys()),
            index=list(UNCERTAINTY_PRESET_OPTIONS.keys()).index(
                st.session_state[preset_key]
            ),
            format_func=uncertainty_preset_label,
            key=preset_key,
            help=(
                "Определяет уровень консерватизма planning-level модели неопределенности. "
                "Обычный MWD подходит как базовый режим, оптимистичный уменьшает конус, "
                "консервативный увеличивает его."
            ),
        )
        required_md_m = tuple(
            float(value)
            for value in (
                view.summary.get("kop_md_m"),
                view.md_t1_m,
                view.summary.get("md_total_m"),
            )
            if value is not None
        )
        model = planning_uncertainty_model_for_preset(selected_preset)
        uncertainty_overlay = build_uncertainty_overlay(
            stations=view.stations,
            surface=view.surface,
            azimuth_deg=float(view.azimuth_deg),
            model=model,
            required_md_m=required_md_m,
        )
        # st.caption(
        #     f"Пресет: {uncertainty_preset_label(selected_preset)}. "
        #     f"{uncertainty_model_caption(uncertainty_overlay.model)}"
        # )

    render_trajectory_dls_panel(
        stations=view.stations,
        well_name=view.well_name,
        surface=view.surface,
        t1=view.t1,
        t3=view.t3,
        md_t1_m=float(view.md_t1_m),
        dls_limits=view.config.dls_limits_deg_per_30m,
        title=title_trajectory,
        border=border,
        trajectory_line_dash=view.trajectory_line_dash,
        plan_csb_stations=view.plan_csb_stations,
        actual_stations=view.actual_stations,
        uncertainty_overlay=uncertainty_overlay,
        render_3d_override=render_3d_override,
    )
    render_plan_section_panel(
        stations=view.stations,
        well_name=view.well_name,
        surface=view.surface,
        t1=view.t1,
        t3=view.t3,
        azimuth_deg=float(view.azimuth_deg),
        title=title_plan,
        border=border,
        trajectory_line_dash=view.trajectory_line_dash,
        plan_csb_stations=view.plan_csb_stations,
        actual_stations=view.actual_stations,
        uncertainty_overlay=uncertainty_overlay,
    )


def render_result_tables(
    *,
    view: SingleWellResultView,
    t1_horizontal_offset_m: float,
    summary_tab_label: str = "Сводка",
    survey_tab_label: str = "Инклинометрия",
    survey_file_name: str = "well_survey.csv",
    survey_export_stations: pd.DataFrame | None = None,
    survey_export_xy_label_suffix: str = "",
    survey_export_xy_unit: str = "м",
    show_validation_section: bool = True,
    show_solver_diagnostics_section: bool = True,
) -> None:
    summary = view.summary
    optimization_mode = str(summary.get("optimization_mode", view.config.optimization_mode))
    optimization_label = optimization_display_label(optimization_mode)
    tab_summary, tab_survey = st.tabs([summary_tab_label, survey_tab_label])
    with tab_summary:
        hidden_metrics = SUMMARY_TECH_HIDDEN_METRICS
        summary_visible = {
            key: value for key, value in summary.items() if key not in hidden_metrics
        }
        summary_visible["t1_horizontal_offset_m"] = float(t1_horizontal_offset_m)
        validation_rows = build_target_validation_rows(summary)

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
        main_rows.insert(
            1,
            {
                "Показатель": "Оптимизация",
                "Значение": optimization_label,
            },
        )
        if "t1_horizontal_offset_m" in summary_visible:
            main_rows.insert(
                5,
                {
                    "Показатель": "Отход t1, м",
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
                    "Параметр": _format_summary_key(key),
                    "Значение": _format_summary_value(key, value),
                }
            )

        st.dataframe(
            arrow_safe_text_dataframe(pd.DataFrame(main_rows)),
            width="stretch",
            hide_index=True,
        )
        if show_validation_section and validation_rows:
            with st.expander("Контроль попадания и точность расчета", expanded=False):
                st.dataframe(
                    arrow_safe_text_dataframe(pd.DataFrame(validation_rows)),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Промахи t1/t3 и INC на входе считаются аналитически по сегментам профиля. "
                    "Control-grid используется для survey-таблицы, графиков и постконтроля DLS."
                )
        if show_solver_diagnostics_section:
            with st.expander("Технические параметры и диагностика решателя", expanded=False):
                st.dataframe(
                    arrow_safe_text_dataframe(pd.DataFrame(tech_rows)),
                    width="stretch",
                    hide_index=True,
                )

    with tab_survey:
        display_crs = str(view.summary.get("display_crs", "")).strip()
        fallback_xy_label_suffix = f" ({display_crs})" if display_crs else ""
        fallback_xy_unit = str(view.summary.get("display_crs_xy_unit", "м")).strip() or "м"
        display_xy_label_suffix = (
            fallback_xy_label_suffix
            if survey_export_stations is None
            else ""
        )
        display_xy_unit = fallback_xy_unit if survey_export_stations is None else "м"
        render_survey_table_with_download(
            stations=view.stations,
            button_label="Скачать CSV инклинометрии",
            file_name=survey_file_name,
            export_stations=survey_export_stations,
            xy_label_suffix=display_xy_label_suffix,
            xy_unit=display_xy_unit,
            export_xy_label_suffix=(
                survey_export_xy_label_suffix or fallback_xy_label_suffix
            ),
            export_xy_unit=survey_export_xy_unit or fallback_xy_unit,
        )
