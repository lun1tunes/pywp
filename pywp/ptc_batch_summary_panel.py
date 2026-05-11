from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from pywp import ptc_batch_results
from pywp.coordinate_integration import DEFAULT_CRS
from pywp.coordinate_systems import CoordinateSystem
from pywp.pilot_wells import (
    is_pilot_name,
    parent_name_for_pilot,
    well_name_key,
)
from pywp.ui_theme import render_small_note
from pywp.ui_utils import arrow_safe_text_dataframe
from pywp.welltrack_batch import SuccessfulWellPlan, WelltrackBatchPlanner

__all__ = ["render_batch_summary"]

BuildBatchSurveyCsvFunc = Callable[..., bytes]
BuildBatchSurveyPayloadFunc = Callable[..., bytes]
RenderSmallNoteFunc = Callable[[str], None]
_SURVEY_DOWNLOAD_FORMAT_DEV = ".dev (7z)"
_LEGACY_SURVEY_DOWNLOAD_FORMAT_DEV_ZIP = ".dev (ZIP)"
_SURVEY_DOWNLOAD_FORMATS = ("CSV", "WELLTRACK", _SURVEY_DOWNLOAD_FORMAT_DEV)


@dataclass(frozen=True)
class _SurveyDownloadConfig:
    builder: BuildBatchSurveyPayloadFunc
    mime: str
    all_label: str
    selected_label: str
    all_file_name: str
    selected_file_name: str
    single_selected_builder: BuildBatchSurveyPayloadFunc | None = None
    single_selected_label: str = ""
    single_selected_file_name: str = ""


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
    build_batch_survey_welltrack_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_survey_welltrack
    ),
    build_batch_survey_dev_7z_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_survey_dev_7z
    ),
    build_batch_survey_dev_file_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_survey_dev_file
    ),
    render_small_note_func: RenderSmallNoteFunc = render_small_note,
) -> pd.DataFrame:
    summary_df = summary_dataframe_func(summary_rows)
    if not summary_df.empty and "Скважина" in summary_df.columns:
        summary_df = summary_df.loc[
            ~summary_df["Скважина"].map(is_pilot_name)
        ].reset_index(drop=True)
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
    pilot_df = ptc_batch_results.pilot_sidetrack_summary_df(
        list(state.get("wt_successes") or [])
    )
    if not pilot_df.empty:
        st_module.markdown("### Скважины с пилотом")
        st_module.dataframe(
            arrow_safe_text_dataframe_func(pilot_df),
            width="stretch",
            hide_index=True,
            column_config={
                "Скважина": st_module.column_config.TextColumn(
                    "Скважина", width="small"
                ),
                "Пилот": st_module.column_config.TextColumn("Пилот", width="small"),
                "Плановых точек пилота": st_module.column_config.TextColumn(
                    "Точек пилота", width="small"
                ),
                "BUILD+HOLD до точек пилота": st_module.column_config.TextColumn(
                    "BUILD+HOLD", width="small"
                ),
                "Окно MD, м": st_module.column_config.NumberColumn(
                    "Окно MD, м", format="%.2f", width="small"
                ),
                "Окно Z, м": st_module.column_config.NumberColumn(
                    "Окно Z, м", format="%.2f", width="small"
                ),
                "Окно INC, deg": st_module.column_config.NumberColumn(
                    "Окно INC, deg", format="%.2f", width="small"
                ),
                "Окно AZI, deg": st_module.column_config.NumberColumn(
                    "Окно AZI, deg", format="%.2f", width="small"
                ),
                "MD пилота, м": st_module.column_config.NumberColumn(
                    "MD пилота, м", format="%.2f", width="small"
                ),
                "MD бокового ствола, м": st_module.column_config.NumberColumn(
                    "MD бокового, м", format="%.2f", width="small"
                ),
                "Макс ПИ пилота, deg/10m": st_module.column_config.NumberColumn(
                    "Макс ПИ пилота, deg/10m", format="%.2f", width="small"
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
        build_batch_survey_welltrack_func=build_batch_survey_welltrack_func,
        build_batch_survey_dev_7z_func=build_batch_survey_dev_7z_func,
        build_batch_survey_dev_file_func=build_batch_survey_dev_file_func,
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
    build_batch_survey_welltrack_func: BuildBatchSurveyPayloadFunc,
    build_batch_survey_dev_7z_func: BuildBatchSurveyPayloadFunc,
    build_batch_survey_dev_file_func: BuildBatchSurveyPayloadFunc,
) -> None:
    with st_module.expander("Выгрузка траекторий"):
        successes = list(state.get("wt_successes") or [])
        success_names = [
            str(success.name)
            for success in successes
            if not is_pilot_name(success.name)
        ]
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
        format_key = "wt_survey_download_format"
        if (
            str(state.get(format_key, "")).strip()
            == _LEGACY_SURVEY_DOWNLOAD_FORMAT_DEV_ZIP
        ):
            state[format_key] = _SURVEY_DOWNLOAD_FORMAT_DEV
        if str(state.get(format_key, "")) not in _SURVEY_DOWNLOAD_FORMATS:
            state[format_key] = _SURVEY_DOWNLOAD_FORMATS[0]
        export_format = str(
            st_module.radio(
                "Формат выгрузки",
                options=list(_SURVEY_DOWNLOAD_FORMATS),
                key=format_key,
                horizontal=True,
            )
        )
        selected_name_set = {str(name) for name in selected_names}
        selected_successes = _successes_for_visible_selection(
            successes=successes,
            selected_names=selected_name_set,
        )
        export_config = _survey_download_config(
            export_format=export_format,
            build_batch_survey_csv_func=build_batch_survey_csv_func,
            build_batch_survey_welltrack_func=build_batch_survey_welltrack_func,
            build_batch_survey_dev_7z_func=build_batch_survey_dev_7z_func,
            build_batch_survey_dev_file_func=build_batch_survey_dev_file_func,
        )
        survey_data = export_config.builder(
            successes,
            target_crs=target_crs,
            auto_convert=auto_convert,
            source_crs=source_crs,
        )
        selected_survey_data, selected_label, selected_file_name, selected_mime = (
            _selected_download_payload(
                export_config=export_config,
                selected_successes=selected_successes,
                target_crs=target_crs,
                auto_convert=auto_convert,
                source_crs=source_crs,
            )
        )
        all_col, selected_col = st_module.columns(2, gap="small")
        with all_col:
            st_module.download_button(
                export_config.all_label,
                data=survey_data or b"",
                file_name=export_config.all_file_name,
                mime=export_config.mime,
                icon=":material/download:",
                use_container_width=True,
                disabled=not survey_data,
            )
        with selected_col:
            st_module.download_button(
                selected_label,
                data=selected_survey_data or b"",
                file_name=selected_file_name,
                mime=selected_mime,
                icon=":material/download:",
                use_container_width=True,
                disabled=not selected_survey_data,
            )


def _successes_for_visible_selection(
    *,
    successes: list[SuccessfulWellPlan],
    selected_names: set[str],
) -> list[SuccessfulWellPlan]:
    selected_parent_keys = {well_name_key(name) for name in selected_names}
    selected: list[SuccessfulWellPlan] = []
    for success in successes:
        name = str(success.name)
        parent_key = (
            well_name_key(parent_name_for_pilot(name))
            if is_pilot_name(name)
            else well_name_key(name)
        )
        if parent_key in selected_parent_keys:
            selected.append(success)
    return selected


def _survey_download_config(
    *,
    export_format: str,
    build_batch_survey_csv_func: BuildBatchSurveyCsvFunc,
    build_batch_survey_welltrack_func: BuildBatchSurveyPayloadFunc,
    build_batch_survey_dev_7z_func: BuildBatchSurveyPayloadFunc,
    build_batch_survey_dev_file_func: BuildBatchSurveyPayloadFunc,
) -> _SurveyDownloadConfig:
    if export_format == "WELLTRACK":
        return _SurveyDownloadConfig(
            builder=build_batch_survey_welltrack_func,
            mime="text/plain",
            all_label="Скачать WELLTRACK всех скважин",
            selected_label="Скачать WELLTRACK выбранных скважин",
            all_file_name="welltrack_survey_all.inc",
            selected_file_name="welltrack_survey_selected.inc",
        )
    if export_format == _SURVEY_DOWNLOAD_FORMAT_DEV:
        return _SurveyDownloadConfig(
            builder=build_batch_survey_dev_7z_func,
            mime="application/x-7z-compressed",
            all_label="Скачать .dev архив всех скважин",
            selected_label="Скачать .dev архив выбранных скважин",
            all_file_name="welltrack_survey_all_dev.7z",
            selected_file_name="welltrack_survey_selected_dev.7z",
            single_selected_builder=build_batch_survey_dev_file_func,
            single_selected_label="Скачать .dev выбранной скважины",
            single_selected_file_name="welltrack_survey_selected.dev",
        )
    return _SurveyDownloadConfig(
        builder=build_batch_survey_csv_func,
        mime="text/csv",
        all_label="Скачать рассчитанные траектории всех скважин",
        selected_label="Скачать рассчитанные траектории выбранных скважин",
        all_file_name="welltrack_survey_all.csv",
        selected_file_name="welltrack_survey_selected.csv",
    )


def _selected_download_payload(
    *,
    export_config: _SurveyDownloadConfig,
    selected_successes: list[SuccessfulWellPlan],
    target_crs: CoordinateSystem,
    auto_convert: bool,
    source_crs: CoordinateSystem,
) -> tuple[bytes, str, str, str]:
    single_builder = export_config.single_selected_builder
    if single_builder is not None and len(selected_successes) == 1:
        data = single_builder(
            selected_successes,
            target_crs=target_crs,
            auto_convert=auto_convert,
            source_crs=source_crs,
        )
        file_name = ptc_batch_results.dev_export_file_name(
            str(selected_successes[0].name)
        )
        return (
            data,
            export_config.single_selected_label or export_config.selected_label,
            file_name if data else export_config.single_selected_file_name,
            "text/plain",
        )

    data = export_config.builder(
        selected_successes,
        target_crs=target_crs,
        auto_convert=auto_convert,
        source_crs=source_crs,
    )
    return (
        data,
        export_config.selected_label,
        export_config.selected_file_name,
        export_config.mime,
    )


def _as_selection_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []
