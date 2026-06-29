from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from pywp import ptc_batch_results
from pywp.coordinate_integration import DEFAULT_CRS
from pywp.coordinate_systems import CoordinateSystem
from pywp.path_utils import normalize_user_path_text
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
BuildBatchDevFilesFunc = Callable[..., tuple[ptc_batch_results.DevExportFilePayload, ...]]
RenderSmallNoteFunc = Callable[[str], None]
_SURVEY_DOWNLOAD_FORMAT_DEV = ".dev (7z)"
_LEGACY_SURVEY_DOWNLOAD_FORMAT_DEV_ZIP = ".dev (ZIP)"
_SURVEY_DOWNLOAD_FORMATS = ("CSV", "WELLTRACK", _SURVEY_DOWNLOAD_FORMAT_DEV)
_EXPORT_KIND_TRAJECTORIES = "Траектории"
_EXPORT_KIND_TARGETS = "Цели"
_EXPORT_KINDS = (_EXPORT_KIND_TRAJECTORIES, _EXPORT_KIND_TARGETS)
_DOWNLOAD_AUTO_BUILD_ROW_LIMIT = 5000
_WINDOWS_BLOCKED_DEV_EXPORT_DRIVES = frozenset({"c", "d"})
_DEV_EXPORT_DIRECTORY_KEY = "wt_dev_export_directory"
_DEV_EXPORT_PENDING_KEY = "wt_dev_export_pending"
_DEV_EXPORT_FEEDBACK_KEY = "wt_dev_export_feedback"


@dataclass(frozen=True)
class _PendingDevExportRequest:
    directory: str
    files: tuple[ptc_batch_results.DevExportFilePayload, ...]
    conflict_file_names: tuple[str, ...]
    title: str


@dataclass(frozen=True)
class _DevExportFeedback:
    level: str
    message: str


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
    build_batch_survey_dev_files_func: BuildBatchDevFilesFunc = (
        ptc_batch_results.build_batch_survey_dev_files
    ),
    build_batch_survey_dev_7z_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_survey_dev_7z
    ),
    build_batch_survey_dev_file_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_survey_dev_file
    ),
    build_batch_target_csv_func: BuildBatchSurveyCsvFunc = (
        ptc_batch_results.build_batch_target_csv
    ),
    build_batch_target_welltrack_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_target_welltrack
    ),
    build_batch_target_dev_files_func: BuildBatchDevFilesFunc = (
        ptc_batch_results.build_batch_target_dev_files
    ),
    build_batch_target_dev_7z_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_target_dev_7z
    ),
    build_batch_target_dev_file_func: BuildBatchSurveyPayloadFunc = (
        ptc_batch_results.build_batch_target_dev_file
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
            f"Не рассчитаны: {counts.not_run_count}. Это нормально при выборочном расчёте: "
            "строки остаются в отчёте до отдельного запуска по этим скважинам."
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
        build_batch_survey_dev_files_func=build_batch_survey_dev_files_func,
        build_batch_survey_dev_7z_func=build_batch_survey_dev_7z_func,
        build_batch_survey_dev_file_func=build_batch_survey_dev_file_func,
        build_batch_target_csv_func=build_batch_target_csv_func,
        build_batch_target_welltrack_func=build_batch_target_welltrack_func,
        build_batch_target_dev_files_func=build_batch_target_dev_files_func,
        build_batch_target_dev_7z_func=build_batch_target_dev_7z_func,
        build_batch_target_dev_file_func=build_batch_target_dev_file_func,
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
    build_batch_survey_dev_files_func: BuildBatchDevFilesFunc,
    build_batch_survey_dev_7z_func: BuildBatchSurveyPayloadFunc,
    build_batch_survey_dev_file_func: BuildBatchSurveyPayloadFunc,
    build_batch_target_csv_func: BuildBatchSurveyCsvFunc,
    build_batch_target_welltrack_func: BuildBatchSurveyPayloadFunc,
    build_batch_target_dev_files_func: BuildBatchDevFilesFunc,
    build_batch_target_dev_7z_func: BuildBatchSurveyPayloadFunc,
    build_batch_target_dev_file_func: BuildBatchSurveyPayloadFunc,
) -> None:
    st_module.markdown("### Выгрузка")
    kind_key = "wt_download_kind"
    if str(state.get(kind_key, "")) not in _EXPORT_KINDS:
        state[kind_key] = _EXPORT_KIND_TRAJECTORIES
    controls_col, format_col = st_module.columns([1, 1], gap="medium")
    with controls_col:
        export_kind = str(
            st_module.radio(
                "Что выгружать",
                options=list(_EXPORT_KINDS),
                key=kind_key,
                horizontal=True,
            )
        )
    format_key = (
        "wt_target_download_format"
        if export_kind == _EXPORT_KIND_TARGETS
        else "wt_survey_download_format"
    )
    _normalize_download_format(state=state, format_key=format_key)
    with format_col:
        export_format = str(
            st_module.radio(
                "Формат выгрузки",
                options=list(_SURVEY_DOWNLOAD_FORMATS),
                key=format_key,
                horizontal=True,
            )
        )
    if export_kind == _EXPORT_KIND_TARGETS:
        _render_target_downloads(
            state=state,
            st_module=st_module,
            target_crs=target_crs,
            auto_convert=auto_convert,
            source_crs=source_crs,
            export_format=export_format,
            build_batch_target_csv_func=build_batch_target_csv_func,
            build_batch_target_welltrack_func=build_batch_target_welltrack_func,
            build_batch_target_dev_files_func=build_batch_target_dev_files_func,
            build_batch_target_dev_7z_func=build_batch_target_dev_7z_func,
            build_batch_target_dev_file_func=build_batch_target_dev_file_func,
        )
        return

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
    station_count = _success_station_row_count(successes)
    if station_count > _DOWNLOAD_AUTO_BUILD_ROW_LIMIT:
        prepare_key = "wt_prepare_survey_download_payloads"
        state.setdefault(prepare_key, False)
        prepare_payloads = bool(
            st_module.toggle(
                "Подготовить файлы выгрузки траекторий",
                key=prepare_key,
                help=(
                    "Для больших наборов файлы готовятся по кнопке, "
                    "чтобы страница не тормозила."
                ),
            )
        )
        if not prepare_payloads:
            st_module.caption(
                f"Включите подготовку файлов перед скачиванием ({station_count} строк)."
            )
            return
    all_signature = _download_signature(
        export_kind=export_kind,
        export_format=export_format,
        target_crs=target_crs,
        source_crs=source_crs,
        auto_convert=auto_convert,
        item_signature=_successes_signature(successes),
    )
    survey_data = _download_payload_from_state_cache(
        state=state,
        cache_key="wt_survey_download_all_payload_cache",
        signature=all_signature,
        build_payload=lambda: export_config.builder(
            successes,
            target_crs=target_crs,
            auto_convert=auto_convert,
            source_crs=source_crs,
        ),
    )
    selected_survey_data: bytes = b""
    selected_label = export_config.selected_label
    selected_file_name = export_config.selected_file_name
    selected_mime = export_config.mime
    if selected_successes:
        selected_signature = _download_signature(
            export_kind=export_kind,
            export_format=export_format,
            target_crs=target_crs,
            source_crs=source_crs,
            auto_convert=auto_convert,
            selected_names=tuple(selected_names),
            item_signature=_successes_signature(selected_successes),
        )
        selected_survey_data, selected_label, selected_file_name, selected_mime = (
            _download_payload_from_state_cache(
                state=state,
                cache_key="wt_survey_download_selected_payload_cache",
                signature=selected_signature,
                build_payload=lambda: _selected_download_payload(
                    export_config=export_config,
                    selected_successes=selected_successes,
                    target_crs=target_crs,
                    auto_convert=auto_convert,
                    source_crs=source_crs,
                ),
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
    if export_format == _SURVEY_DOWNLOAD_FORMAT_DEV:
        _render_dev_directory_export_controls(
            state=state,
            st_module=st_module,
            title="Сохранение .dev в папку",
            build_all_files=lambda: build_batch_survey_dev_files_func(
                successes,
                target_crs=target_crs,
                auto_convert=auto_convert,
                source_crs=source_crs,
            ),
            build_selected_files=lambda: build_batch_survey_dev_files_func(
                selected_successes,
                target_crs=target_crs,
                auto_convert=auto_convert,
                source_crs=source_crs,
            ),
            save_all_label="Сохранить все .dev",
            save_selected_label="Сохранить выбранные .dev",
            disable_all=not successes,
            disable_selected=not selected_successes,
        )


def _render_target_downloads(
    *,
    state: MutableMapping[str, object],
    st_module: Any,
    target_crs: CoordinateSystem,
    auto_convert: bool,
    source_crs: CoordinateSystem,
    export_format: str,
    build_batch_target_csv_func: BuildBatchSurveyCsvFunc,
    build_batch_target_welltrack_func: BuildBatchSurveyPayloadFunc,
    build_batch_target_dev_files_func: BuildBatchDevFilesFunc,
    build_batch_target_dev_7z_func: BuildBatchSurveyPayloadFunc,
    build_batch_target_dev_file_func: BuildBatchSurveyPayloadFunc,
) -> None:
    records = list(state.get("wt_records") or state.get("wt_records_original") or [])
    record_names = [str(record.name) for record in records if not is_pilot_name(record.name)]
    record_name_set = set(record_names)
    selected_key = "wt_target_download_selected_names"
    raw_selected = _as_selection_list(state.get(selected_key, []))
    selected_current = [
        str(name) for name in raw_selected if str(name) in record_name_set
    ]
    if selected_current != raw_selected:
        state[selected_key] = selected_current
    selected_names = st_module.multiselect(
        "Скважины для выгрузки",
        options=record_names,
        key=selected_key,
        placeholder="Выберите скважины",
    )
    selected_name_set = {str(name) for name in selected_names}
    selected_records = _records_for_visible_selection(
        records=records,
        selected_names=selected_name_set,
    )
    export_config = _target_download_config(
        export_format=export_format,
        build_batch_target_csv_func=build_batch_target_csv_func,
        build_batch_target_welltrack_func=build_batch_target_welltrack_func,
        build_batch_target_dev_7z_func=build_batch_target_dev_7z_func,
        build_batch_target_dev_file_func=build_batch_target_dev_file_func,
    )
    target_point_count = _record_point_count(records)
    if target_point_count > _DOWNLOAD_AUTO_BUILD_ROW_LIMIT:
        prepare_key = "wt_prepare_target_download_payloads"
        state.setdefault(prepare_key, False)
        prepare_payloads = bool(
            st_module.toggle(
                "Подготовить файлы выгрузки целей",
                key=prepare_key,
                help=(
                    "Для больших наборов файлы готовятся по кнопке, "
                    "чтобы страница не тормозила."
                ),
            )
        )
        if not prepare_payloads:
            st_module.caption(
                f"Выгрузка целей не сформирована автоматически: {target_point_count} точек. "
                "Включите подготовку файлов только перед скачиванием."
            )
            return
    all_signature = _download_signature(
        export_kind=_EXPORT_KIND_TARGETS,
        export_format=export_format,
        target_crs=target_crs,
        source_crs=source_crs,
        auto_convert=auto_convert,
        item_signature=_records_signature(records),
    )
    target_data = _download_payload_from_state_cache(
        state=state,
        cache_key="wt_target_download_all_payload_cache",
        signature=all_signature,
        build_payload=lambda: export_config.builder(
            records,
            target_crs=target_crs,
            auto_convert=auto_convert,
            source_crs=source_crs,
        ),
    )
    selected_target_data: bytes = b""
    selected_label = export_config.selected_label
    selected_file_name = export_config.selected_file_name
    selected_mime = export_config.mime
    if selected_records:
        selected_signature = _download_signature(
            export_kind=_EXPORT_KIND_TARGETS,
            export_format=export_format,
            target_crs=target_crs,
            source_crs=source_crs,
            auto_convert=auto_convert,
            selected_names=tuple(selected_names),
            item_signature=_records_signature(selected_records),
        )
        selected_target_data, selected_label, selected_file_name, selected_mime = (
            _download_payload_from_state_cache(
                state=state,
                cache_key="wt_target_download_selected_payload_cache",
                signature=selected_signature,
                build_payload=lambda: _selected_download_payload(
                    export_config=export_config,
                    selected_successes=selected_records,
                    target_crs=target_crs,
                    auto_convert=auto_convert,
                    source_crs=source_crs,
                ),
            )
        )
    all_col, selected_col = st_module.columns(2, gap="small")
    with all_col:
        st_module.download_button(
            export_config.all_label,
            data=target_data or b"",
            file_name=export_config.all_file_name,
            mime=export_config.mime,
            icon=":material/download:",
            use_container_width=True,
            disabled=not target_data,
        )
    with selected_col:
        st_module.download_button(
            selected_label,
            data=selected_target_data or b"",
            file_name=selected_file_name,
            mime=selected_mime,
            icon=":material/download:",
            use_container_width=True,
            disabled=not selected_target_data,
        )
    if export_format == _SURVEY_DOWNLOAD_FORMAT_DEV:
        _render_dev_directory_export_controls(
            state=state,
            st_module=st_module,
            title="Сохранение целей .dev в папку",
            build_all_files=lambda: build_batch_target_dev_files_func(
                records,
                target_crs=target_crs,
                auto_convert=auto_convert,
                source_crs=source_crs,
            ),
            build_selected_files=lambda: build_batch_target_dev_files_func(
                selected_records,
                target_crs=target_crs,
                auto_convert=auto_convert,
                source_crs=source_crs,
            ),
            save_all_label="Сохранить все .dev",
            save_selected_label="Сохранить выбранные .dev",
            disable_all=not records,
            disable_selected=not selected_records,
        )


def _normalize_download_format(
    *,
    state: MutableMapping[str, object],
    format_key: str,
) -> str:
    current_value = str(state.get(format_key, "")).strip()
    if current_value == _LEGACY_SURVEY_DOWNLOAD_FORMAT_DEV_ZIP:
        current_value = _SURVEY_DOWNLOAD_FORMAT_DEV
    if current_value not in _SURVEY_DOWNLOAD_FORMATS:
        current_value = _SURVEY_DOWNLOAD_FORMATS[0]
    state[format_key] = current_value
    return current_value


def _download_payload_from_state_cache(
    *,
    state: MutableMapping[str, object],
    cache_key: str,
    signature: tuple[object, ...],
    build_payload: Callable[[], Any],
) -> Any:
    cached = state.get(cache_key)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return cached.get("payload")
    payload = build_payload()
    state[cache_key] = {"signature": signature, "payload": payload}
    return payload


def _download_signature(
    *,
    export_kind: str,
    export_format: str,
    target_crs: CoordinateSystem,
    source_crs: CoordinateSystem,
    auto_convert: bool,
    item_signature: tuple[object, ...],
    selected_names: tuple[object, ...] = (),
) -> tuple[object, ...]:
    return (
        str(export_kind),
        str(export_format),
        _crs_signature(target_crs),
        _crs_signature(source_crs),
        bool(auto_convert),
        tuple(str(name) for name in selected_names),
        item_signature,
    )


def _crs_signature(crs: CoordinateSystem) -> str:
    return str(getattr(crs, "value", crs))


def _successes_signature(successes: list[SuccessfulWellPlan]) -> tuple[object, ...]:
    return tuple(_success_signature(success) for success in successes)


def _success_signature(success: SuccessfulWellPlan) -> tuple[object, ...]:
    stations = getattr(success, "stations", None)
    return (
        str(getattr(success, "name", "")),
        id(success),
        _frame_signature(stations),
        str(getattr(success, "md_postcheck_message", "")),
    )


def _records_signature(records: list[object]) -> tuple[object, ...]:
    return tuple(_record_signature(record) for record in records)


def _record_signature(record: object) -> tuple[object, ...]:
    points = tuple(getattr(record, "points", ()) or ())
    return (
        str(getattr(record, "name", "")),
        id(record),
        len(points),
        _point_signature(points[0]) if points else None,
        _point_signature(points[-1]) if points else None,
    )


def _frame_signature(frame: object) -> tuple[object, ...]:
    try:
        row_count = int(len(frame))
    except TypeError:
        return (0, None, None)
    if row_count <= 0 or not isinstance(frame, pd.DataFrame):
        return (row_count, None, None)
    return (
        row_count,
        tuple(str(column) for column in frame.columns),
        _frame_row_signature(frame.iloc[0]),
        _frame_row_signature(frame.iloc[-1]),
    )


def _frame_row_signature(row: pd.Series) -> tuple[object, ...]:
    values: list[object] = []
    for value in row.tolist():
        if isinstance(value, (int, float)):
            values.append(round(float(value), 9))
        else:
            values.append(str(value))
    return tuple(values)


def _point_signature(point: object) -> tuple[float, float, float, float | None]:
    md = getattr(point, "md", None)
    return (
        round(float(getattr(point, "x", 0.0)), 9),
        round(float(getattr(point, "y", 0.0)), 9),
        round(float(getattr(point, "z", 0.0)), 9),
        None if md is None else round(float(md), 9),
    )


def _success_station_row_count(successes: list[SuccessfulWellPlan]) -> int:
    total = 0
    for success in successes:
        stations = getattr(success, "stations", None)
        try:
            total += int(len(stations)) if stations is not None else 0
        except TypeError:
            continue
    return total


def _record_point_count(records: list[object]) -> int:
    total = 0
    for record in records:
        try:
            total += int(len(getattr(record, "points", ()) or ()))
        except TypeError:
            continue
    return total


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


def _records_for_visible_selection(
    *,
    records: list[object],
    selected_names: set[str],
) -> list[object]:
    selected_parent_keys = {well_name_key(name) for name in selected_names}
    selected: list[object] = []
    for record in records:
        name = str(getattr(record, "name", ""))
        parent_key = (
            well_name_key(parent_name_for_pilot(name))
            if is_pilot_name(name)
            else well_name_key(name)
        )
        if parent_key in selected_parent_keys:
            selected.append(record)
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


def _target_download_config(
    *,
    export_format: str,
    build_batch_target_csv_func: BuildBatchSurveyCsvFunc,
    build_batch_target_welltrack_func: BuildBatchSurveyPayloadFunc,
    build_batch_target_dev_7z_func: BuildBatchSurveyPayloadFunc,
    build_batch_target_dev_file_func: BuildBatchSurveyPayloadFunc,
) -> _SurveyDownloadConfig:
    if export_format == "WELLTRACK":
        return _SurveyDownloadConfig(
            builder=build_batch_target_welltrack_func,
            mime="text/plain",
            all_label="Скачать WELLTRACK целей всех скважин",
            selected_label="Скачать WELLTRACK целей выбранных скважин",
            all_file_name="welltrack_targets_all.inc",
            selected_file_name="welltrack_targets_selected.inc",
        )
    if export_format == _SURVEY_DOWNLOAD_FORMAT_DEV:
        return _SurveyDownloadConfig(
            builder=build_batch_target_dev_7z_func,
            mime="application/x-7z-compressed",
            all_label="Скачать .dev архив целей всех скважин",
            selected_label="Скачать .dev архив целей выбранных скважин",
            all_file_name="welltrack_targets_all_dev.7z",
            selected_file_name="welltrack_targets_selected_dev.7z",
            single_selected_builder=build_batch_target_dev_file_func,
            single_selected_label="Скачать .dev целей выбранной скважины",
            single_selected_file_name="welltrack_targets_selected.dev",
        )
    return _SurveyDownloadConfig(
        builder=build_batch_target_csv_func,
        mime="text/csv",
        all_label="Скачать цели всех скважин",
        selected_label="Скачать цели выбранных скважин",
        all_file_name="welltrack_targets_all.csv",
        selected_file_name="welltrack_targets_selected.csv",
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


def _render_dev_directory_export_controls(
    *,
    state: MutableMapping[str, object],
    st_module: Any,
    title: str,
    build_all_files: Callable[[], tuple[ptc_batch_results.DevExportFilePayload, ...]],
    build_selected_files: Callable[
        [], tuple[ptc_batch_results.DevExportFilePayload, ...]
    ],
    save_all_label: str,
    save_selected_label: str,
    disable_all: bool,
    disable_selected: bool,
) -> None:
    st_module.caption("`.dev` файлы можно сохранить в папку без архива.")
    st_module.text_input(
        "Папка для .dev",
        key=_DEV_EXPORT_DIRECTORY_KEY,
        placeholder=r"E:\Exports\DEV",
    )
    action_cols = st_module.columns(2, gap="small")
    with action_cols[0]:
        save_all_clicked = st_module.button(
            save_all_label,
            icon=":material/folder_open:",
            width="stretch",
            disabled=bool(disable_all),
        )
    with action_cols[1]:
        save_selected_clicked = st_module.button(
            save_selected_label,
            icon=":material/folder_open:",
            width="stretch",
            disabled=bool(disable_selected),
        )
    if save_all_clicked:
        _start_dev_directory_export(
            state=state,
            build_files=build_all_files,
            title=title,
        )
        _rerun_summary_fragment(st_module)
    if save_selected_clicked:
        _start_dev_directory_export(
            state=state,
            build_files=build_selected_files,
            title=title,
        )
        _rerun_summary_fragment(st_module)
    _show_dev_export_feedback(state=state, st_module=st_module)
    _render_dev_export_overwrite_dialog(state=state, st_module=st_module)


def _start_dev_directory_export(
    *,
    state: MutableMapping[str, object],
    build_files: Callable[[], tuple[ptc_batch_results.DevExportFilePayload, ...]],
    title: str,
) -> None:
    raw_directory = str(state.get(_DEV_EXPORT_DIRECTORY_KEY, "")).strip()
    directory, error_message = _validated_dev_export_directory(raw_directory)
    if error_message is not None or directory is None:
        _set_dev_export_feedback(
            state=state,
            level="error",
            message=error_message or "Не удалось определить папку для выгрузки.",
        )
        state.pop(_DEV_EXPORT_PENDING_KEY, None)
        return

    files = tuple(build_files())
    if not files:
        _set_dev_export_feedback(
            state=state,
            level="warning",
            message="Нет .dev файлов для сохранения в выбранной выгрузке.",
        )
        state.pop(_DEV_EXPORT_PENDING_KEY, None)
        return

    conflict_file_names = tuple(
        file_payload.file_name
        for file_payload in files
        if (directory / str(file_payload.file_name)).exists()
    )
    if conflict_file_names:
        state[_DEV_EXPORT_PENDING_KEY] = _PendingDevExportRequest(
            directory=str(directory),
            files=files,
            conflict_file_names=conflict_file_names,
            title=title,
        )
        state.pop(_DEV_EXPORT_FEEDBACK_KEY, None)
        return

    try:
        _write_dev_export_files(directory=directory, files=files, overwrite=True)
    except OSError as exc:
        _set_dev_export_feedback(
            state=state,
            level="error",
            message=f"Не удалось сохранить .dev файлы: {exc}",
        )
        state.pop(_DEV_EXPORT_PENDING_KEY, None)
        return
    state.pop(_DEV_EXPORT_PENDING_KEY, None)
    _set_dev_export_feedback(
        state=state,
        level="success",
        message=f"Сохранено .dev файлов: {len(files)}. Папка: {directory}",
    )


def _render_dev_export_overwrite_dialog(
    *,
    state: MutableMapping[str, object],
    st_module: Any,
) -> None:
    pending = state.get(_DEV_EXPORT_PENDING_KEY)
    if not isinstance(pending, _PendingDevExportRequest):
        return

    conflict_names = tuple(str(name) for name in pending.conflict_file_names)
    conflict_items = tuple(
        file_payload
        for file_payload in pending.files
        if str(file_payload.file_name) in set(conflict_names)
    )

    def _render_content() -> None:
        st_module.write("В папке уже есть .dev файлы с такими именами:")
        for file_payload in conflict_items:
            st_module.write(
                f"- {file_payload.well_name} ({file_payload.file_name})"
            )
        action_cols = st_module.columns(2, gap="small")
        with action_cols[0]:
            confirm_clicked = st_module.button(
                "Заменить файлы",
                type="primary",
                width="stretch",
            )
        with action_cols[1]:
            cancel_clicked = st_module.button(
                "Отмена",
                width="stretch",
            )
        if confirm_clicked:
            directory = Path(str(pending.directory))
            try:
                _write_dev_export_files(
                    directory=directory,
                    files=pending.files,
                    overwrite=True,
                )
            except OSError as exc:
                state.pop(_DEV_EXPORT_PENDING_KEY, None)
                _set_dev_export_feedback(
                    state=state,
                    level="error",
                    message=f"Не удалось заменить .dev файлы: {exc}",
                )
                _rerun_summary_fragment(st_module)
                return
            state.pop(_DEV_EXPORT_PENDING_KEY, None)
            _set_dev_export_feedback(
                state=state,
                level="success",
                message=(
                    f"Сохранено .dev файлов: {len(pending.files)}. "
                    f"Заменено: {len(conflict_items)}."
                ),
            )
            _rerun_summary_fragment(st_module)
        if cancel_clicked:
            state.pop(_DEV_EXPORT_PENDING_KEY, None)
            _set_dev_export_feedback(
                state=state,
                level="info",
                message="Сохранение .dev файлов отменено.",
            )
            _rerun_summary_fragment(st_module)

    dialog_factory = getattr(st_module, "dialog", None)
    if callable(dialog_factory):
        @dialog_factory(str(pending.title))
        def _dialog() -> None:
            _render_content()

        _dialog()
        return

    st_module.warning(
        "Нужна замена существующих .dev файлов. Подтвердите сохранение ниже."
    )
    _render_content()


def _validated_dev_export_directory(raw_path: str) -> tuple[Path | None, str | None]:
    path_text = normalize_user_path_text(raw_path)
    if not path_text:
        return None, "Укажите папку для выгрузки .dev файлов."
    windows_error = _windows_dev_export_directory_error(path_text)
    if windows_error is not None:
        return None, windows_error

    path = Path(path_text).expanduser()
    if path == Path(path.anchor or "/"):
        return None, "Нельзя выгружать .dev файлы в корень диска."
    if path.exists() and not path.is_dir():
        return None, "Указанный путь уже существует и не является папкой."
    return path, None


def _windows_dev_export_directory_error(path_text: str) -> str | None:
    normalized = normalize_user_path_text(path_text).replace("/", "\\")
    if not normalized:
        return None
    match = re.match(r"(?i)^([a-z]):(?:\\(.*))?$", normalized)
    if match is None:
        return None
    drive_letter = str(match.group(1) or "").strip().lower()
    if drive_letter in _WINDOWS_BLOCKED_DEV_EXPORT_DRIVES:
        return "На диски C: и D: выгрузка .dev файлов запрещена."
    tail = str(match.group(2) or "").strip("\\")
    if not tail:
        return "Нельзя выгружать .dev файлы в корень диска Windows."
    first_component = tail.split("\\", 1)[0].strip().lower()
    if first_component in {
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
    }:
        return "В системные каталоги Windows выгрузка .dev файлов запрещена."
    return None


def _write_dev_export_files(
    *,
    directory: Path,
    files: tuple[ptc_batch_results.DevExportFilePayload, ...],
    overwrite: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for file_payload in files:
        destination = directory / str(file_payload.file_name)
        if destination.exists() and not overwrite:
            raise FileExistsError(str(destination))
        temp_path = destination.with_name(destination.name + ".pywp_tmp")
        try:
            temp_path.write_bytes(bytes(file_payload.data))
            temp_path.replace(destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _set_dev_export_feedback(
    *,
    state: MutableMapping[str, object],
    level: str,
    message: str,
) -> None:
    state[_DEV_EXPORT_FEEDBACK_KEY] = _DevExportFeedback(
        level=str(level),
        message=str(message),
    )


def _show_dev_export_feedback(
    *,
    state: MutableMapping[str, object],
    st_module: Any,
) -> None:
    feedback = state.get(_DEV_EXPORT_FEEDBACK_KEY)
    if not isinstance(feedback, _DevExportFeedback):
        return
    level = str(feedback.level).strip().lower()
    message = str(feedback.message)
    if level == "success" and hasattr(st_module, "success"):
        st_module.success(message)
        return
    if level == "warning" and hasattr(st_module, "warning"):
        st_module.warning(message)
        return
    if level == "error" and hasattr(st_module, "error"):
        st_module.error(message)
        return
    if level == "info" and hasattr(st_module, "info"):
        st_module.info(message)
        return
    st_module.caption(message)


def _rerun_summary_fragment(st_module: Any) -> None:
    rerun = getattr(st_module, "rerun", None)
    if not callable(rerun):
        return
    try:
        rerun(scope="fragment")
    except TypeError:
        rerun()


def _as_selection_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []
