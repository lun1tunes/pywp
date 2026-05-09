from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

import pandas as pd
import streamlit as st

from pywp import ptc_batch_results
from pywp.coordinate_integration import DEFAULT_CRS
from pywp.coordinate_systems import CoordinateSystem
from pywp.ui_theme import render_small_note
from pywp.ui_utils import arrow_safe_text_dataframe
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner

__all__ = ["render_batch_summary"]

BuildBatchSurveyCsvFunc = Callable[..., bytes]
RenderSmallNoteFunc = Callable[[str], None]


def render_batch_summary(
    summary_rows: list[dict[str, object]],
    *,
    state: MutableMapping[str, object],
    st_module: Any = st,
    target_crs: CoordinateSystem = DEFAULT_CRS,
    auto_convert: bool = True,
    source_crs: CoordinateSystem = DEFAULT_CRS,
    summary_dataframe_func: Callable[..., pd.DataFrame] = (
        WelltrackBatchPlanner.summary_dataframe
    ),
    arrow_safe_text_dataframe_func: Callable[[pd.DataFrame], pd.DataFrame] = (
        arrow_safe_text_dataframe
    ),
    batch_summary_display_df_func: Callable[[pd.DataFrame], pd.DataFrame] = (
        ptc_batch_results.batch_summary_display_df
    ),
    build_batch_survey_csv_func: BuildBatchSurveyCsvFunc = (
        ptc_batch_results.build_batch_survey_csv
    ),
    render_small_note_func: RenderSmallNoteFunc = render_small_note,
) -> pd.DataFrame:
    summary_df = summary_dataframe_func(summary_rows)
    if not summary_df.empty:
        summary_df = arrow_safe_text_dataframe_func(summary_df)

    counts = ptc_batch_results.batch_summary_status_counts(summary_df)
    p1, p2, p3, p4, p5 = st_module.columns(5, gap="small")
    p1.metric("Строк в отчете", f"{len(summary_df)}")
    p2.metric("Без замечаний", f"{counts.ok_count}")
    p3.metric("С предупреждениями", f"{counts.warning_count}")
    p4.metric("Ошибки", f"{counts.error_count}")
    run_time = state.get("wt_last_runtime_s")
    p5.metric(
        "Время расчета",
        "—" if run_time is None else f"{float(run_time):.2f} с",
    )
    if counts.not_run_count:
        st_module.caption(
            f"Не рассчитаны: {counts.not_run_count}. Это нормально для partial batch-расчета: "
            "строки остаются в отчете до отдельного запуска по этим скважинам."
        )
    render_small_note_func(f"Последний запуск: {state.get('wt_last_run_at', '—')}")
    if ptc_batch_results.has_md_postcheck_warning(summary_df):
        st_module.caption(
            "Скважины с превышением лимита итоговой MD отображаются пунктирной "
            "траекторией на графиках."
        )

    st_module.markdown("### Сводка расчета")
    display_df = batch_summary_display_df_func(summary_df)
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
    st_module.dataframe(
        display_payload,
        width="stretch",
        hide_index=True,
        column_config={
            "Скважина": st_module.column_config.TextColumn("Скважина", width="small"),
            "Точек": st_module.column_config.NumberColumn(
                "Точек", format="%d", width="small"
            ),
            "Цели": st_module.column_config.TextColumn("Цели", width="small"),
            "Сложность": st_module.column_config.TextColumn("Сложность", width="small"),
            "Отход t1, м": st_module.column_config.NumberColumn(
                "Отход t1, м",
                format="%.2f",
                width="small",
            ),
            "ГС, м": st_module.column_config.NumberColumn(
                "ГС, м",
                format="%.2f",
                width="small",
            ),
            "INC в t1, deg": st_module.column_config.NumberColumn(
                "INC t1, deg", format="%.2f", width="small"
            ),
            "ЗУ HOLD, deg": st_module.column_config.NumberColumn(
                "ЗУ HOLD, deg", format="%.2f", width="small"
            ),
            "Макс ПИ, deg/10m": st_module.column_config.NumberColumn(
                "Макс ПИ, deg/10m",
                format="%.2f",
                width="small",
            ),
            "Макс MD, м": st_module.column_config.NumberColumn(
                "Макс MD, м",
                format="%.2f",
                width="small",
            ),
            "Рестарты": st_module.column_config.TextColumn("Рестарты", width="small"),
            "Статус": st_module.column_config.TextColumn("Статус", width="small"),
            "Проблема": st_module.column_config.TextColumn("Проблема", width="medium"),
            "Модель траектории": st_module.column_config.TextColumn(
                "Модель траектории",
                width="medium",
            ),
        },
    )
    _render_survey_downloads(
        state=state,
        st_module=st_module,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
        build_batch_survey_csv_func=build_batch_survey_csv_func,
    )
    return summary_df


def _render_survey_downloads(
    *,
    state: MutableMapping[str, object],
    st_module: Any,
    target_crs: CoordinateSystem,
    auto_convert: bool,
    source_crs: CoordinateSystem,
    build_batch_survey_csv_func: BuildBatchSurveyCsvFunc,
) -> None:
    with st_module.expander("Инклинометрия скважин"):
        successes = list(state.get("wt_successes") or [])
        success_names = [str(success.name) for success in successes]
        success_name_set = set(success_names)
        selected_key = "wt_survey_download_selected_names"
        raw_selected = _as_selection_list(state.get(selected_key, []))
        selected_current = [
            str(name) for name in raw_selected if str(name) in success_name_set
        ]
        if selected_current != raw_selected:
            state[selected_key] = selected_current
        selected_names = st_module.multiselect(
            "Скважины для выгрузки",
            options=success_names,
            key=selected_key,
            placeholder="Выберите скважины",
        )
        selected_name_set = {str(name) for name in selected_names}
        selected_successes: list[SuccessfulWellPlan] = [
            success for success in successes if str(success.name) in selected_name_set
        ]
        survey_data = build_batch_survey_csv_func(
            successes,
            target_crs=target_crs,
            auto_convert=auto_convert,
            source_crs=source_crs,
        )
        selected_survey_data = build_batch_survey_csv_func(
            selected_successes,
            target_crs=target_crs,
            auto_convert=auto_convert,
            source_crs=source_crs,
        )
        all_col, selected_col = st_module.columns(2, gap="small")
        with all_col:
            st_module.download_button(
                "Скачать рассчитанные траектории всех скважин",
                data=survey_data or b"",
                file_name="welltrack_survey_all.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
                disabled=not survey_data,
            )
        with selected_col:
            st_module.download_button(
                "Скачать рассчитанные траектории выбранных скважин",
                data=selected_survey_data or b"",
                file_name="welltrack_survey_selected.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
                disabled=not selected_survey_data,
            )


def _as_selection_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []
